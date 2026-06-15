"""Rollout collection and PPO update loop for Torch MAPPO."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torchrl.objectives.multiagent import MAPPOLoss

from ant_byte_env import AntByteForagingEnv
from ant_byte_env.training.torch_mappo.curriculum import (
    build_curriculum_reset_options,
    compute_forage_curriculum_rewards,
)
from ant_byte_env.training.torch_mappo.model import MAPPOAgent
from ant_byte_env.training.torch_mappo.observations import (
    NumpyObs,
    build_actor_observations,
    build_central_observations,
    flatten_agent_actions,
    obs_to_tensor,
)


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
            random_hub=args.random_hub,
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
        "terminations": torch.zeros(
            (args.num_steps, args.num_envs, 1),
            dtype=torch.bool,
            device=device,
        ),
        "truncations": torch.zeros(
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
    completed_count = 0
    terminated_count = 0
    truncated_count = 0
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
        terminated_flags = np.zeros(args.num_envs, dtype=bool)
        truncated_flags = np.zeros(args.num_envs, dtype=bool)
        done_flags = np.zeros(args.num_envs, dtype=bool)

        for env_index, env in enumerate(envs):
            obs_item, reward, terminated, truncated, _ = env.step(env_actions[env_index])
            raw_next_obs_items.append(obs_item)
            env_rewards[env_index] = float(reward)
            terminated_flags[env_index] = bool(terminated)
            truncated_flags[env_index] = bool(truncated)
            done_flags[env_index] = bool(terminated or truncated)

        storage["dones"][step] = torch.as_tensor(done_flags, dtype=torch.bool, device=device).unsqueeze(
            -1
        )
        storage["terminations"][step] = torch.as_tensor(
            terminated_flags,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(-1)
        storage["truncations"][step] = torch.as_tensor(
            truncated_flags,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(-1)
        completed_count += int(np.sum(done_flags))
        terminated_count += int(np.sum(terminated_flags))
        truncated_count += int(np.sum(truncated_flags))
        raw_next_obs = stack_obs(raw_next_obs_items)
        raw_next_obs_tensor = obs_to_tensor(raw_next_obs, device)
        storage["next_central_obs"][step] = build_central_observations(
            raw_next_obs_tensor,
            food_scale=args.food_count,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
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
        next_done = torch.as_tensor(done_flags, device=device)
        global_step += args.num_envs

    rollout_stats = {
        "episode_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "episode_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
        "completed_episodes": float(completed_count),
        "terminated_episodes": float(terminated_count),
        "truncated_episodes": float(truncated_count),
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
            ("next", "terminated"): storage["terminations"].detach(),
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
