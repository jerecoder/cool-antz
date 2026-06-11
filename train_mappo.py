#!/usr/bin/env python3
"""TorchRL-backed MAPPO trainer for the forage curriculum."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torchrl.objectives.multiagent import MAPPOLoss

from ant_byte_env import (
    DEFAULT_WRITE_BITS,
    MAX_WRITE_BITS,
    WRITE_VALUE_COUNT,
    AntByteForagingEnv,
    max_write_value,
    write_value_count,
)


TensorObs = dict[str, torch.Tensor]
NumpyObs = dict[str, np.ndarray]
DEFAULT_VISION_COLORS = (
    (61, 220, 255),
    (255, 113, 206),
    (255, 226, 89),
    (93, 255, 139),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MAPPO on the AntByte forage curriculum."
    )
    parser.add_argument("--exp-name", type=str, default="mappo_forage")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", action="store_true", default=True)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)

    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument(
        "--obs-width",
        type=int,
        default=None,
        help="Pad observations to this width so one model can span larger-map curriculum stages.",
    )
    parser.add_argument(
        "--obs-height",
        type=int,
        default=None,
        help="Pad observations to this height so one model can span larger-map curriculum stages.",
    )
    parser.add_argument(
        "--actor-vision-radius",
        type=int,
        default=2,
        help="Radius of the local grid visible to each ant policy.",
    )
    parser.add_argument("--num-ants", type=int, default=2)
    parser.add_argument("--food-count", type=int, default=4)
    parser.add_argument("--food-sources", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--write-penalty", type=float, default=0.0)
    parser.add_argument(
        "--write-bits",
        type=int,
        default=DEFAULT_WRITE_BITS,
        help="Number of bits each ant can write per tile action.",
    )

    parser.add_argument(
        "--curriculum-stage",
        choices=["forage"],
        default="forage",
        help="Current stage trains cookie pickup, hub return, and tile-value writing.",
    )
    parser.add_argument(
        "--cookie-distance",
        type=int,
        default=1,
        help="Manhattan distance from the hub to the fixed cookie source.",
    )
    parser.add_argument(
        "--random-food",
        action="store_true",
        help="Use the environment's random food placement instead of the fixed curriculum source.",
    )
    parser.add_argument(
        "--random-hub",
        action="store_true",
        help="Randomize the colony hub position on every reset.",
    )
    parser.add_argument(
        "--pickup-bonus",
        type=float,
        default=0.25,
        help="Extra curriculum reward when an ant picks up a cookie bite.",
    )
    parser.add_argument(
        "--distance-bonus",
        type=float,
        default=0.02,
        help="Reward for reducing distance to the current target.",
    )
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Resume actor/critic weights from a previous curriculum checkpoint.",
    )

    args = parser.parse_args(argv)
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.num_minibatches <= 0:
        raise ValueError("--num-minibatches must be positive.")
    if args.num_envs * args.num_steps < args.num_minibatches:
        raise ValueError("--num-minibatches cannot exceed rollout batch size.")
    if args.cookie_distance <= 0:
        raise ValueError("--cookie-distance must be positive.")
    if args.food_count > 0 and args.width * args.height <= 1:
        raise ValueError("food_count requires at least one non-hub tile.")
    if args.obs_width is not None and args.obs_width < args.width:
        raise ValueError("--obs-width must be at least --width.")
    if args.obs_height is not None and args.obs_height < args.height:
        raise ValueError("--obs-height must be at least --height.")
    if args.actor_vision_radius < 0:
        raise ValueError("--actor-vision-radius must be non-negative.")
    if args.write_bits <= 0 or args.write_bits > MAX_WRITE_BITS:
        raise ValueError(f"--write-bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    return args


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


def obs_to_tensor(obs: NumpyObs, device: torch.device) -> TensorObs:
    return {key: torch.as_tensor(value, device=device) for key, value in obs.items()}


def _position_scale(height: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [max(width - 1, 1), max(height - 1, 1)],
        dtype=torch.float32,
        device=device,
    )


def _normalize_positions(positions: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    scale = _position_scale(height, width, positions.device)
    return positions.float() / scale


def _resolve_observation_grid_shape(
    obs: TensorObs,
    *,
    obs_height: int | None,
    obs_width: int | None,
) -> tuple[int, int, int, int]:
    _, current_height, current_width = obs["food"].shape
    target_height = current_height if obs_height is None else obs_height
    target_width = current_width if obs_width is None else obs_width
    if target_height < current_height or target_width < current_width:
        raise ValueError(
            "Padded observation shape must be at least as large as the environment grid."
        )
    return current_height, current_width, target_height, target_width


def _pad_grid(grid: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if grid.shape[1:] == (height, width):
        return grid

    padded = torch.zeros(
        (grid.shape[0], height, width),
        dtype=grid.dtype,
        device=grid.device,
    )
    padded[:, : grid.shape[1], : grid.shape[2]] = grid
    return padded


def build_central_observations(
    obs: TensorObs,
    *,
    food_scale: int,
    write_bits: int = DEFAULT_WRITE_BITS,
    obs_width: int | None = None,
    obs_height: int | None = None,
) -> torch.Tensor:
    """Flatten and normalize the full state used by the centralized critic."""

    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    food = obs["food"].float()
    bytes_grid = obs["bytes"].float()
    batch_size, _, _ = food.shape
    current_height, current_width, target_height, target_width = (
        _resolve_observation_grid_shape(
            obs,
            obs_height=obs_height,
            obs_width=obs_width,
        )
    )
    ants_pos = _normalize_positions(
        obs["ants_pos"],
        height=target_height,
        width=target_width,
    )
    hub_pos = _normalize_positions(
        obs["hub_pos"],
        height=target_height,
        width=target_width,
    )
    ants_carrying = obs["ants_carrying"].float()
    food_norm = _pad_grid(
        food / max(float(food_scale), 1.0),
        height=target_height,
        width=target_width,
    )
    bytes_norm = _pad_grid(
        bytes_grid / max(float(max_write_value(write_bits)), 1.0),
        height=target_height,
        width=target_width,
    )
    grid_size = torch.tensor(
        [
            current_width / max(float(target_width), 1.0),
            current_height / max(float(target_height), 1.0),
        ],
        dtype=torch.float32,
        device=food.device,
    ).expand(batch_size, -1)

    return torch.cat(
        [
            ants_pos.reshape(batch_size, -1),
            ants_carrying.reshape(batch_size, -1),
            food_norm.reshape(batch_size, -1),
            bytes_norm.reshape(batch_size, -1),
            hub_pos.reshape(batch_size, -1),
            grid_size,
        ],
        dim=-1,
    )


def build_actor_observations(
    obs: TensorObs,
    central_obs: torch.Tensor | None = None,
    *,
    food_scale: int = 1,
    actor_vision_radius: int = 2,
    write_bits: int = DEFAULT_WRITE_BITS,
    obs_width: int | None = None,
    obs_height: int | None = None,
) -> torch.Tensor:
    """Create local per-ant actor observations for the decentralized policy."""

    del central_obs
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")

    del obs_width, obs_height
    food = obs["food"].float()
    own_carrying = obs["ants_carrying"].float().unsqueeze(-1)
    local_food = build_local_food_patches(
        food,
        obs["ants_pos"],
        radius=actor_vision_radius,
        food_scale=food_scale,
    )
    local_byte_bits = build_local_byte_bit_patches(
        obs["bytes"],
        obs["ants_pos"],
        radius=actor_vision_radius,
        write_bits=write_bits,
    )
    local_hub = build_local_hub_patches(
        obs["hub_pos"],
        obs["ants_pos"],
        grid_height=food.shape[1],
        grid_width=food.shape[2],
        radius=actor_vision_radius,
    )
    local_border = build_local_border_patches(
        obs["ants_pos"],
        grid_height=food.shape[1],
        grid_width=food.shape[2],
        radius=actor_vision_radius,
    )

    return torch.cat(
        [local_food, local_byte_bits, local_hub, local_border, own_carrying],
        dim=-1,
    )


def build_local_food_patches(
    food: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    radius: int,
    food_scale: int,
) -> torch.Tensor:
    """Return flattened local food grids centered on each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    return build_local_grid_patches(food, ants_pos, radius=radius) / max(float(food_scale), 1.0)


def build_local_byte_bit_patches(
    bytes_grid: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    radius: int,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> torch.Tensor:
    """Return flattened local bit-plane patches for writable tile values."""

    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    bit_patches = []
    bytes_long = bytes_grid.long()
    for bit_index in range(write_bits):
        bit_grid = ((bytes_long >> bit_index) & 1).float()
        bit_patches.append(build_local_grid_patches(bit_grid, ants_pos, radius=radius))
    return torch.cat(bit_patches, dim=-1)


def build_local_grid_patches(
    grid: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Return flattened local grid patches centered on each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    batch_size, grid_height, grid_width = grid.shape
    num_agents = ants_pos.shape[1]
    patch_width = 2 * radius + 1
    patches = torch.zeros(
        (batch_size, num_agents, patch_width, patch_width),
        dtype=torch.float32,
        device=grid.device,
    )

    for batch_index in range(batch_size):
        for agent_index in range(num_agents):
            ant_x = int(ants_pos[batch_index, agent_index, 0])
            ant_y = int(ants_pos[batch_index, agent_index, 1])
            for patch_y, grid_y in enumerate(range(ant_y - radius, ant_y + radius + 1)):
                if not 0 <= grid_y < grid_height:
                    continue
                for patch_x, grid_x in enumerate(range(ant_x - radius, ant_x + radius + 1)):
                    if 0 <= grid_x < grid_width:
                        patches[batch_index, agent_index, patch_y, patch_x] = grid[
                            batch_index,
                            grid_y,
                            grid_x,
                        ]

    return patches.reshape(batch_size, num_agents, -1)


def build_local_border_patches(
    ants_pos: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
) -> torch.Tensor:
    """Return flattened out-of-bounds masks centered on each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    offsets = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.long,
        device=ants_pos.device,
    )
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_pairs = torch.stack([offset_x.reshape(-1), offset_y.reshape(-1)], dim=-1)
    positions = ants_pos.long().unsqueeze(2) + offset_pairs.view(1, 1, -1, 2)
    x_pos = positions[..., 0]
    y_pos = positions[..., 1]
    valid = (0 <= x_pos) & (x_pos < grid_width) & (0 <= y_pos) & (y_pos < grid_height)
    return (~valid).float()


def build_local_hub_patches(
    hub_pos: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
) -> torch.Tensor:
    """Return flattened local hub masks centered on each ant."""

    hub_grid = torch.zeros(
        (hub_pos.shape[0], grid_height, grid_width),
        dtype=torch.float32,
        device=hub_pos.device,
    )
    for batch_index in range(hub_pos.shape[0]):
        hub_x = int(hub_pos[batch_index, 0])
        hub_y = int(hub_pos[batch_index, 1])
        hub_grid[batch_index, hub_y, hub_x] = 1.0
    return build_local_grid_patches(hub_grid, ants_pos, radius=radius)


def flatten_agent_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert joint movement/write actions to the env's interleaved action vector."""

    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError(f"joint actions must have shape (batch, ants, 2), got {actions.shape}.")
    batch_size, num_agents, _ = actions.shape
    flat_actions = torch.empty(
        (batch_size, num_agents * 2),
        dtype=torch.long,
        device=actions.device,
    )
    flat_actions[:, 0::2] = actions[..., 0].long()
    flat_actions[:, 1::2] = actions[..., 1].long()
    return flat_actions


def draw_vision_squares(
    frame: np.ndarray,
    obs: NumpyObs,
    *,
    tile_size: int,
    vision_radius: int,
    colors: tuple[tuple[int, int, int], ...] = DEFAULT_VISION_COLORS,
    border_px: int = 2,
    fill_alpha: float = 0.12,
) -> np.ndarray:
    """Return a copy of ``frame`` with each ant's local vision square overlaid."""

    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if vision_radius < 0:
        raise ValueError("vision_radius must be non-negative.")
    if border_px <= 0:
        raise ValueError("border_px must be positive.")
    if not 0.0 <= fill_alpha <= 1.0:
        raise ValueError("fill_alpha must be between 0 and 1.")
    if not colors:
        raise ValueError("at least one vision color is required.")

    output = frame.copy()
    grid_height, grid_width = obs["food"].shape
    frame_height, frame_width = output.shape[:2]

    for ant_index, position in enumerate(obs["ants_pos"]):
        x_pos, y_pos = int(position[0]), int(position[1])
        left_tile = max(0, x_pos - vision_radius)
        right_tile = min(grid_width, x_pos + vision_radius + 1)
        top_tile = max(0, y_pos - vision_radius)
        bottom_tile = min(grid_height, y_pos + vision_radius + 1)
        x0 = int(np.clip(left_tile * tile_size, 0, frame_width))
        x1 = int(np.clip(right_tile * tile_size, 0, frame_width))
        y0 = int(np.clip(top_tile * tile_size, 0, frame_height))
        y1 = int(np.clip(bottom_tile * tile_size, 0, frame_height))
        if x0 >= x1 or y0 >= y1:
            continue

        color = np.array(colors[ant_index % len(colors)], dtype=np.float32)
        region = output[y0:y1, x0:x1].astype(np.float32)
        output[y0:y1, x0:x1] = (
            region * (1.0 - fill_alpha) + color * fill_alpha
        ).astype(output.dtype)

        border = min(border_px, max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2))
        output[y0 : y0 + border, x0:x1] = color.astype(output.dtype)
        output[y1 - border : y1, x0:x1] = color.astype(output.dtype)
        output[y0:y1, x0 : x0 + border] = color.astype(output.dtype)
        output[y0:y1, x1 - border : x1] = color.astype(output.dtype)

    return output


def _sample_hub_position(
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if args.random_hub:
        return (
            int(rng.integers(0, args.width)),
            int(rng.integers(0, args.height)),
        )

    return (args.width // 2, args.height // 2)


def build_curriculum_reset_options(
    args: argparse.Namespace,
    *,
    seed: int | None = None,
) -> dict[str, Any] | None:
    rng = np.random.default_rng(seed)
    hub = _sample_hub_position(args=args, rng=rng)
    if args.random_food:
        return {"hub_pos": hub}

    distance = min(args.cookie_distance, max(args.width, args.height))
    offsets = ((distance, 0), (-distance, 0), (0, distance), (0, -distance))
    for x_offset, y_offset in offsets:
        candidate = (hub[0] + x_offset, hub[1] + y_offset)
        if 0 <= candidate[0] < args.width and 0 <= candidate[1] < args.height:
            return {"hub_pos": hub, "food_positions": [candidate]}

    for y_pos in range(args.height):
        for x_pos in range(args.width):
            if (x_pos, y_pos) != hub:
                return {"hub_pos": hub, "food_positions": [(x_pos, y_pos)]}

    return {"hub_pos": hub}


def stack_obs(obs_items: list[NumpyObs]) -> NumpyObs:
    return {
        key: np.stack([obs[key] for obs in obs_items], axis=0)
        for key in obs_items[0]
    }


def make_envs(args: argparse.Namespace) -> list[AntByteForagingEnv]:
    return [
        AntByteForagingEnv(
            width=args.width,
            height=args.height,
            num_ants=args.num_ants,
            food_count=args.food_count,
            food_source_count=args.food_sources,
            max_steps=args.max_steps,
            random_food=args.random_food,
            step_penalty=args.step_penalty,
            write_penalty=args.write_penalty,
            write_bits=args.write_bits,
        )
        for _ in range(args.num_envs)
    ]


def reset_env(
    env: AntByteForagingEnv,
    *,
    seed: int | None,
    args: argparse.Namespace,
) -> tuple[NumpyObs, dict[str, int]]:
    return env.reset(seed=seed, options=build_curriculum_reset_options(args, seed=seed))


def _nearest_food_distance(position: np.ndarray, food_grid: np.ndarray) -> float | None:
    food_positions = np.argwhere(food_grid > 0)
    if food_positions.size == 0:
        return None

    x_pos, y_pos = int(position[0]), int(position[1])
    distances = np.abs(food_positions[:, 1] - x_pos) + np.abs(food_positions[:, 0] - y_pos)
    return float(distances.min())


def _distance_to_hub(position: np.ndarray, hub_pos: np.ndarray) -> float:
    return float(abs(int(position[0]) - int(hub_pos[0])) + abs(int(position[1]) - int(hub_pos[1])))


def compute_forage_curriculum_rewards(
    *,
    previous_obs: NumpyObs,
    next_obs: NumpyObs,
    env_rewards: np.ndarray,
    pickup_bonus: float,
    distance_bonus: float,
) -> np.ndarray:
    """Add simple pickup and target-progress rewards for the first curriculum."""

    shaped_rewards = env_rewards.astype(np.float32, copy=True)
    batch_size, num_agents = previous_obs["ants_carrying"].shape

    for env_index in range(batch_size):
        for agent_index in range(num_agents):
            was_carrying = bool(previous_obs["ants_carrying"][env_index, agent_index])
            is_carrying = bool(next_obs["ants_carrying"][env_index, agent_index])
            if not was_carrying and is_carrying:
                shaped_rewards[env_index] += float(pickup_bonus)

            previous_position = previous_obs["ants_pos"][env_index, agent_index]
            next_position = next_obs["ants_pos"][env_index, agent_index]
            if was_carrying:
                target_previous_distance = _distance_to_hub(
                    previous_position,
                    previous_obs["hub_pos"][env_index],
                )
                target_next_distance = _distance_to_hub(
                    next_position,
                    previous_obs["hub_pos"][env_index],
                )
            else:
                target_previous_distance = _nearest_food_distance(
                    previous_position,
                    previous_obs["food"][env_index],
                )
                target_next_distance = _nearest_food_distance(
                    next_position,
                    previous_obs["food"][env_index],
                )
                if target_previous_distance is None or target_next_distance is None:
                    continue

            progress = target_previous_distance - target_next_distance
            shaped_rewards[env_index] += float(distance_bonus) * progress

    return shaped_rewards


def make_rollout_storage(
    *,
    args: argparse.Namespace,
    actor_obs_dim: int,
    central_obs_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "actor_obs": torch.zeros(
            (args.num_steps, args.num_envs, args.num_ants, actor_obs_dim),
            device=device,
        ),
        "central_obs": torch.zeros(
            (args.num_steps, args.num_envs, central_obs_dim),
            device=device,
        ),
        "actions": torch.zeros(
            (args.num_steps, args.num_envs, args.num_ants, 2),
            dtype=torch.long,
            device=device,
        ),
        "logprobs": torch.zeros(
            (args.num_steps, args.num_envs, args.num_ants),
            device=device,
        ),
        "rewards": torch.zeros((args.num_steps, args.num_envs, 1), device=device),
        "dones": torch.zeros(
            (args.num_steps, args.num_envs, 1),
            dtype=torch.bool,
            device=device,
        ),
        "next_central_obs": torch.zeros(
            (args.num_steps, args.num_envs, central_obs_dim),
            device=device,
        ),
    }


def collect_rollout(
    *,
    args: argparse.Namespace,
    agent: MAPPOAgent,
    envs: list[AntByteForagingEnv],
    storage: dict[str, torch.Tensor],
    next_obs: NumpyObs,
    next_done: torch.Tensor,
    global_step: int,
    device: torch.device,
) -> tuple[NumpyObs, torch.Tensor, int, dict[str, float]]:
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    for step in range(args.num_steps):
        obs_tensor = obs_to_tensor(next_obs, device)
        central_obs = build_central_observations(
            obs_tensor,
            food_scale=args.food_count,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        actor_obs = build_actor_observations(
            obs_tensor,
            central_obs,
            food_scale=args.food_count,
            actor_vision_radius=args.actor_vision_radius,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )

        storage["actor_obs"][step] = actor_obs
        storage["central_obs"][step] = central_obs

        with torch.no_grad():
            actions, logprobs, _, _ = agent.get_action_and_value(actor_obs, central_obs)

        storage["actions"][step] = actions
        storage["logprobs"][step] = logprobs

        env_actions = flatten_agent_actions(actions).cpu().numpy()
        previous_obs = next_obs
        raw_next_obs_items: list[NumpyObs] = []
        env_rewards = np.zeros(args.num_envs, dtype=np.float32)
        done_flags = np.zeros(args.num_envs, dtype=np.float32)

        for env_index, env in enumerate(envs):
            obs_item, reward, terminated, truncated, _ = env.step(env_actions[env_index])
            raw_next_obs_items.append(obs_item)
            env_rewards[env_index] = float(reward)
            done_flags[env_index] = float(terminated or truncated)

        storage["dones"][step] = torch.as_tensor(done_flags, dtype=torch.bool, device=device).unsqueeze(
            -1
        )
        raw_next_obs = stack_obs(raw_next_obs_items)
        shaped_rewards = compute_forage_curriculum_rewards(
            previous_obs=previous_obs,
            next_obs=raw_next_obs,
            env_rewards=env_rewards,
            pickup_bonus=args.pickup_bonus,
            distance_bonus=args.distance_bonus,
        )
        storage["rewards"][step] = torch.as_tensor(shaped_rewards, device=device).unsqueeze(-1)

        episode_returns += shaped_rewards
        episode_lengths += 1
        reset_obs_items: list[NumpyObs] = []
        for env_index, env in enumerate(envs):
            if done_flags[env_index]:
                completed_returns.append(float(episode_returns[env_index]))
                completed_lengths.append(int(episode_lengths[env_index]))
                episode_returns[env_index] = 0.0
                episode_lengths[env_index] = 0
                reset_obs, _ = reset_env(
                    env,
                    seed=args.seed + global_step + env_index,
                    args=args,
                )
                reset_obs_items.append(reset_obs)
            else:
                reset_obs_items.append(raw_next_obs_items[env_index])

        next_obs = stack_obs(reset_obs_items)
        next_obs_tensor = obs_to_tensor(next_obs, device)
        storage["next_central_obs"][step] = build_central_observations(
            next_obs_tensor,
            food_scale=args.food_count,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        next_done = torch.as_tensor(done_flags, device=device)
        global_step += args.num_envs

    rollout_stats = {
        "episode_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "episode_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
    }
    return next_obs, next_done, global_step, rollout_stats


def rollout_storage_to_tensordict(storage: dict[str, torch.Tensor]) -> TensorDict:
    num_steps, num_envs = storage["central_obs"].shape[:2]
    return TensorDict(
        {
            ("agents", "observation"): storage["actor_obs"].detach(),
            "state": storage["central_obs"].detach(),
            ("agents", "action"): storage["actions"].detach(),
            ("agents", "sample_log_prob"): storage["logprobs"].detach(),
            ("next", "state"): storage["next_central_obs"].detach(),
            ("next", "reward"): storage["rewards"].detach(),
            ("next", "done"): storage["dones"].detach(),
            ("next", "terminated"): storage["dones"].detach(),
        },
        batch_size=[num_steps, num_envs],
    )

def update_agent(
    *,
    args: argparse.Namespace,
    agent: MAPPOAgent,
    optimizer: optim.Optimizer,
    loss_module: MAPPOLoss,
    rollout: TensorDict,
) -> dict[str, float]:
    value_rollout = TensorDict(
        {
            "state": rollout["state"],
            ("next", "state"): rollout[("next", "state")],
            ("next", "reward"): rollout[("next", "reward")],
            ("next", "done"): rollout[("next", "done")],
            ("next", "terminated"): rollout[("next", "terminated")],
        },
        batch_size=rollout.batch_size,
    )
    loss_module.value_estimator(value_rollout, time_dim=0)
    rollout.set("advantage", value_rollout["advantage"])
    rollout.set("value_target", value_rollout["value_target"])
    rollout.set(("agents", "state_value"), value_rollout[("agents", "state_value")])
    batch = rollout.reshape(rollout.shape[0] * rollout.shape[1])
    batch_size = batch.shape[0]
    minibatch_size = batch_size // args.num_minibatches
    batch_indices = np.arange(batch_size)
    metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "explained_variance": 0.0,
    }

    for _ in range(args.update_epochs):
        np.random.shuffle(batch_indices)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_indices = batch_indices[start:end]
            loss_td = loss_module(batch[minibatch_indices])
            policy_loss = loss_td["loss_objective"]
            value_loss = loss_td["loss_critic"]
            entropy_loss = loss_td["loss_entropy"]
            loss = policy_loss + value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            metrics.update(
                {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(loss_td["entropy"].mean().item()),
                    "approx_kl": float(loss_td["kl_approx"].mean().item()),
                    "clipfrac": float(loss_td["clip_fraction"].mean().item()),
                    "explained_variance": float(loss_td["explained_variance"].mean().item()),
                }
            )

        if args.target_kl is not None and metrics["approx_kl"] > args.target_kl:
            break

    return metrics


def checkpoint_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return checkpoint-safe CLI args made of torch safe-load primitives."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def load_agent_checkpoint(
    *,
    agent: MAPPOAgent,
    checkpoint_path: Path,
    central_obs_dim: int,
    actor_obs_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_central_dim = int(checkpoint["central_obs_dim"])
    saved_actor_dim = int(checkpoint["actor_obs_dim"])
    if saved_central_dim != central_obs_dim or saved_actor_dim != actor_obs_dim:
        raise ValueError(
            "Checkpoint observation dimensions do not match this run. "
            "Use the same --obs-width, --obs-height, --num-ants, and --write-bits "
            "across curriculum stages."
        )

    agent.load_state_dict(checkpoint["agent_state_dict"])
    return checkpoint


def evaluate_agent(
    *,
    agent: MAPPOAgent,
    args: argparse.Namespace,
    device: torch.device,
    num_episodes: int,
    seed_offset: int = 1_000_000,
    deterministic: bool = True,
) -> dict[str, float]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    delivered_food: list[float] = []
    delivered_fractions: list[float] = []
    successes: list[float] = []

    env = AntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        step_penalty=args.step_penalty,
        write_penalty=args.write_penalty,
        write_bits=args.write_bits,
    )
    try:
        for episode_index in range(num_episodes):
            obs, info = reset_env(
                env,
                seed=args.seed + seed_offset + episode_index,
                args=args,
            )
            episode_return = 0.0
            terminated = False
            truncated = False

            for step_index in range(args.max_steps):
                obs_batch = {key: value[np.newaxis, ...] for key, value in obs.items()}
                obs_tensor = obs_to_tensor(obs_batch, device)
                central_obs = build_central_observations(
                    obs_tensor,
                    food_scale=args.food_count,
                    write_bits=args.write_bits,
                    obs_width=args.obs_width,
                    obs_height=args.obs_height,
                )
                actor_obs = build_actor_observations(
                    obs_tensor,
                    central_obs,
                    food_scale=args.food_count,
                    actor_vision_radius=args.actor_vision_radius,
                    write_bits=args.write_bits,
                    obs_width=args.obs_width,
                    obs_height=args.obs_height,
                )
                with torch.no_grad():
                    actions, _, _, _ = agent.get_action_and_value(
                        actor_obs,
                        central_obs,
                        deterministic=deterministic,
                    )

                env_action = flatten_agent_actions(actions).cpu().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(env_action)
                episode_return += float(reward)
                if terminated or truncated:
                    episode_lengths.append(step_index + 1)
                    break
            else:
                episode_lengths.append(args.max_steps)

            delivered = float(info["delivered_food"])
            delivered_food.append(delivered)
            delivered_fractions.append(delivered / max(float(args.food_count), 1.0))
            episode_returns.append(episode_return)
            successes.append(float(terminated))
    finally:
        env.close()

    return {
        "eval_success_rate": float(np.mean(successes)),
        "eval_mean_delivered_food": float(np.mean(delivered_food)),
        "eval_mean_delivered_fraction": float(np.mean(delivered_fractions)),
        "eval_mean_episode_return": float(np.mean(episode_returns)),
        "eval_mean_episode_length": float(np.mean(episode_lengths)),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    num_episodes: int,
    device: torch.device | None = None,
    seed_offset: int = 1_000_000,
    deterministic: bool = True,
) -> dict[str, float]:
    actual_device = device
    if actual_device is None:
        actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=actual_device, weights_only=True)
    args = argparse.Namespace(**checkpoint["args"])
    if not hasattr(args, "write_bits"):
        args.write_bits = DEFAULT_WRITE_BITS
    agent = MAPPOAgent(
        central_obs_dim=int(checkpoint["central_obs_dim"]),
        actor_obs_dim=int(checkpoint["actor_obs_dim"]),
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    ).to(actual_device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    return evaluate_agent(
        agent=agent,
        args=args,
        device=actual_device,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
        deterministic=deterministic,
    )


def mastery_reached(
    metrics: dict[str, float],
    *,
    min_success_rate: float,
    min_delivered_fraction: float,
) -> bool:
    return (
        metrics["eval_success_rate"] >= min_success_rate
        and metrics["eval_mean_delivered_fraction"] >= min_delivered_fraction
    )


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    run_name = f"{args.exp_name}__seed_{args.seed}__{int(time.time())}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    envs = make_envs(args)
    for env_index, env in enumerate(envs):
        env.action_space.seed(args.seed + env_index)

    try:
        obs_items = [
            reset_env(env, seed=args.seed + env_index, args=args)[0]
            for env_index, env in enumerate(envs)
        ]
        next_obs = stack_obs(obs_items)
        obs_tensor = obs_to_tensor(next_obs, device)
        central_obs = build_central_observations(
            obs_tensor,
            food_scale=args.food_count,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        actor_obs = build_actor_observations(
            obs_tensor,
            central_obs,
            food_scale=args.food_count,
            actor_vision_radius=args.actor_vision_radius,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        central_obs_dim = central_obs.shape[-1]
        actor_obs_dim = actor_obs.shape[-1]

        agent = MAPPOAgent(
            central_obs_dim=central_obs_dim,
            actor_obs_dim=actor_obs_dim,
            hidden_size=args.hidden_size,
            write_value_count=write_value_count(args.write_bits),
        ).to(device)
        loaded_checkpoint: dict[str, Any] | None = None
        if args.load_model is not None:
            loaded_checkpoint = load_agent_checkpoint(
                agent=agent,
                checkpoint_path=args.load_model,
                central_obs_dim=central_obs_dim,
                actor_obs_dim=actor_obs_dim,
                device=device,
            )
        loss_module = make_mappo_loss(args, agent).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        if loaded_checkpoint is not None and "optimizer_state_dict" in loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        storage = make_rollout_storage(
            args=args,
            actor_obs_dim=actor_obs_dim,
            central_obs_dim=central_obs_dim,
            device=device,
        )

        next_done = torch.zeros(args.num_envs, device=device)
        global_step = 0
        num_updates = max(1, args.total_timesteps // (args.num_envs * args.num_steps))
        final_metrics: dict[str, float] = {
            "global_step": 0.0,
            "loss": 0.0,
            "episode_return": 0.0,
            "episode_length": 0.0,
        }

        for update in range(1, num_updates + 1):
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            next_obs, next_done, global_step, rollout_stats = collect_rollout(
                args=args,
                agent=agent,
                envs=envs,
                storage=storage,
                next_obs=next_obs,
                next_done=next_done,
                global_step=global_step,
                device=device,
            )

            update_metrics = update_agent(
                args=args,
                agent=agent,
                optimizer=optimizer,
                loss_module=loss_module,
                rollout=rollout_storage_to_tensordict(storage),
            )

            final_metrics = {
                **update_metrics,
                **rollout_stats,
                "global_step": float(global_step),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if not args.quiet:
                print(
                    "update={update}/{num_updates} step={step} loss={loss:.4f} "
                    "return={episode_return:.3f} len={episode_length:.1f} "
                    "entropy={entropy:.3f}".format(
                        update=update,
                        num_updates=num_updates,
                        step=global_step,
                        **final_metrics,
                    )
                )

        if args.save_model is not None:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": checkpoint_args(args),
                    "central_obs_dim": central_obs_dim,
                    "actor_obs_dim": actor_obs_dim,
                    "run_name": run_name,
                },
                args.save_model,
            )
        return final_metrics
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
