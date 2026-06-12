"""Torch MAPPO model and TorchRL adapters."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torchrl.objectives.multiagent import MAPPOLoss

from ant_byte_env import WRITE_VALUE_COUNT


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPOAgent(nn.Module):
    """Parameter-shared actor with movement/write-value heads and a centralized critic."""

    def __init__(
        self,
        *,
        central_obs_dim: int,
        actor_obs_dim: int,
        hidden_size: int = 128,
        write_value_count: int = WRITE_VALUE_COUNT,
    ) -> None:
        super().__init__()
        if write_value_count <= 0:
            raise ValueError("write_value_count must be positive.")
        self.write_value_count = write_value_count
        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(actor_obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.move_head = layer_init(nn.Linear(hidden_size, 5), std=0.01)
        self.write_head = layer_init(nn.Linear(hidden_size, write_value_count), std=0.01)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(central_obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def get_action_logits(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        *batch_shape, num_agents, actor_dim = actor_obs.shape
        hidden = self.actor_body(actor_obs.reshape(-1, actor_dim))
        move_logits = self.move_head(hidden).reshape(*batch_shape, num_agents, 5)
        write_logits = self.write_head(hidden).reshape(
            *batch_shape,
            num_agents,
            self.write_value_count,
        )
        return move_logits, write_logits

    def get_agent_values(self, central_obs: torch.Tensor, *, num_agents: int) -> torch.Tensor:
        value = self.critic(central_obs).unsqueeze(-2)
        return value.expand(*value.shape[:-2], num_agents, 1)

    def get_action_and_value(
        self,
        actor_obs: torch.Tensor,
        central_obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        move_logits, write_logits = self.get_action_logits(actor_obs)
        distribution = JointMoveWriteCategorical(
            move_logits=move_logits,
            write_logits=write_logits,
        )

        if action is None:
            action = distribution.deterministic_sample if deterministic else distribution.sample()
        else:
            action = action.long()
            if action.shape[-1] != 2:
                raise ValueError(f"joint actions must have final dimension 2, got {action.shape}.")

        logprob = distribution.log_prob(action)
        entropy = distribution.entropy()
        value = self.critic(central_obs).squeeze(-1)
        return action, logprob, entropy, value


class JointMoveWriteCategorical(Distribution):
    """Independent categorical heads for one ant's movement and write value."""

    arg_constraints: dict[str, Any] = {}
    has_enumerate_support = False

    def __init__(
        self,
        *,
        move_logits: torch.Tensor,
        write_logits: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.move_distribution = Categorical(logits=move_logits)
        self.write_distribution = Categorical(logits=write_logits)
        super().__init__(
            batch_shape=self.move_distribution.batch_shape,
            event_shape=torch.Size([2]),
            validate_args=validate_args,
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.stack(
            [
                self.move_distribution.sample(sample_shape),
                self.write_distribution.sample(sample_shape),
            ],
            dim=-1,
        )

    @property
    def mode(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.argmax(self.move_distribution.logits, dim=-1),
                torch.argmax(self.write_distribution.logits, dim=-1),
            ],
            dim=-1,
        )

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        action = value.long()
        if action.shape[-1] != 2:
            raise ValueError(f"joint actions must have final dimension 2, got {action.shape}.")
        return self.move_distribution.log_prob(action[..., 0]) + self.write_distribution.log_prob(
            action[..., 1]
        )

    def entropy(self) -> torch.Tensor:
        return self.move_distribution.entropy() + self.write_distribution.entropy()


class MAPPOActorAdapter(nn.Module):
    """TensorDict-facing adapter around the shared actor heads."""

    def __init__(self, agent: MAPPOAgent) -> None:
        super().__init__()
        self.agent = agent

    def forward(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.agent.get_action_logits(actor_obs)


class MAPPOCriticAdapter(nn.Module):
    """TensorDict-facing adapter around the centralized critic."""

    def __init__(self, agent: MAPPOAgent, *, num_agents: int) -> None:
        super().__init__()
        self.agent = agent
        self.num_agents = num_agents

    def forward(self, central_obs: torch.Tensor) -> torch.Tensor:
        return self.agent.get_agent_values(central_obs, num_agents=self.num_agents)


def make_torchrl_actor(agent: MAPPOAgent) -> ProbabilisticTensorDictSequential:
    actor_module = TensorDictModule(
        MAPPOActorAdapter(agent),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "move_logits"), ("agents", "write_logits")],
    )
    action_module = ProbabilisticTensorDictModule(
        in_keys=[("agents", "move_logits"), ("agents", "write_logits")],
        out_keys=[("agents", "action")],
        distribution_class=JointMoveWriteCategorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    return ProbabilisticTensorDictSequential(actor_module, action_module)


def make_torchrl_critic(agent: MAPPOAgent, *, num_agents: int) -> TensorDictModule:
    return TensorDictModule(
        MAPPOCriticAdapter(agent, num_agents=num_agents),
        in_keys=["state"],
        out_keys=[("agents", "state_value")],
    )


def make_mappo_loss(args: argparse.Namespace, agent: MAPPOAgent) -> MAPPOLoss:
    loss_module = MAPPOLoss(
        make_torchrl_actor(agent),
        make_torchrl_critic(agent, num_agents=args.num_ants),
        clip_epsilon=args.clip_coef,
        entropy_coeff=args.ent_coef,
        critic_coeff=args.vf_coef,
        normalize_advantage=args.norm_adv,
        clip_value=args.clip_coef if args.clip_vloss else None,
        functional=False,
    )
    loss_module.set_keys(
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
    )
    loss_module.make_value_estimator(gamma=args.gamma, lmbda=args.gae_lambda)
    return loss_module
