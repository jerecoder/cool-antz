"""Evaluation helpers for Torch MAPPO checkpoints and agents."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ant_byte_env import (
    DEFAULT_ACTOR_VISION_DEPTH,
    DEFAULT_WRITE_BITS,
    AntByteForagingEnv,
    actor_vision_patch_size,
    write_value_count,
)
from ant_byte_env.training.torch_mappo.checkpointing import (
    adapt_agent_state_dict_for_actor_window,
)
from ant_byte_env.training.torch_mappo.model import MAPPOAgent
from ant_byte_env.training.torch_mappo.observations import (
    build_actor_observations,
    build_central_observations,
    flatten_agent_actions,
    obs_to_tensor,
)
from ant_byte_env.training.torch_mappo.rollout import reset_env


def evaluate_agent(
    *,
    agent: MAPPOAgent,
    args: argparse.Namespace,
    device: torch.device,
    num_episodes: int,
    seed_offset: int = 1_000_000,
    deterministic: bool = True,
    shuffle_positions: bool = True,
) -> dict[str, float]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    eval_args = _evaluation_args_with_position_shuffle(
        args,
        shuffle_positions=shuffle_positions,
    )
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    delivered_food: list[float] = []
    delivered_fractions: list[float] = []
    successes: list[float] = []

    env = AntByteForagingEnv(
        width=eval_args.width,
        height=eval_args.height,
        num_ants=eval_args.num_ants,
        food_count=eval_args.food_count,
        food_source_count=eval_args.food_sources,
        max_steps=eval_args.max_steps,
        random_food=eval_args.random_food,
        random_hub=eval_args.random_hub,
        step_penalty=eval_args.step_penalty,
        write_penalty=eval_args.write_penalty,
        write_bits=eval_args.write_bits,
    )
    try:
        for episode_index in range(num_episodes):
            obs, info = reset_env(
                env,
                seed=eval_args.seed + seed_offset + episode_index,
                args=eval_args,
            )
            episode_return = 0.0
            terminated = False
            truncated = False

            for step_index in range(eval_args.max_steps):
                obs_batch = {key: value[np.newaxis, ...] for key, value in obs.items()}
                obs_tensor = obs_to_tensor(obs_batch, device)
                central_obs = build_central_observations(
                    obs_tensor,
                    food_scale=eval_args.food_count,
                    write_bits=eval_args.write_bits,
                    obs_width=eval_args.obs_width,
                    obs_height=eval_args.obs_height,
                )
                actor_obs = build_actor_observations(
                    obs_tensor,
                    central_obs,
                    food_scale=eval_args.food_count,
                    actor_vision_radius=eval_args.actor_vision_radius,
                    write_bits=eval_args.write_bits,
                    obs_width=eval_args.obs_width,
                    obs_height=eval_args.obs_height,
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
                episode_lengths.append(eval_args.max_steps)

            delivered = float(info["delivered_food"])
            delivered_food.append(delivered)
            delivered_fractions.append(delivered / max(float(eval_args.food_count), 1.0))
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
    shuffle_positions: bool = True,
) -> dict[str, float]:
    actual_device = device
    if actual_device is None:
        actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=actual_device, weights_only=True)
    args = argparse.Namespace(**checkpoint["args"])
    if not hasattr(args, "write_bits"):
        args.write_bits = DEFAULT_WRITE_BITS
    if not hasattr(args, "actor_vision_radius"):
        args.actor_vision_radius = DEFAULT_ACTOR_VISION_DEPTH
    actor_obs_dim = _actor_obs_dim_from_args(args)
    agent = MAPPOAgent(
        central_obs_dim=int(checkpoint["central_obs_dim"]),
        actor_obs_dim=actor_obs_dim,
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    ).to(actual_device)
    agent.load_state_dict(
        adapt_agent_state_dict_for_actor_window(
            checkpoint["agent_state_dict"],
            saved_actor_dim=int(checkpoint["actor_obs_dim"]),
            actor_obs_dim=actor_obs_dim,
            write_bits=args.write_bits,
            actor_vision_radius=args.actor_vision_radius,
        )
    )
    agent.eval()

    return evaluate_agent(
        agent=agent,
        args=args,
        device=actual_device,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
        deterministic=deterministic,
        shuffle_positions=shuffle_positions,
    )


def _actor_obs_dim_from_args(args: argparse.Namespace) -> int:
    patch_size = actor_vision_patch_size(int(args.actor_vision_radius))
    return patch_size * (int(args.write_bits) + 4) + 1


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


def _evaluation_args_with_position_shuffle(
    args: argparse.Namespace,
    *,
    shuffle_positions: bool,
) -> argparse.Namespace:
    values = {**vars(args)}
    values.setdefault("random_food", False)
    values.setdefault("random_hub", False)
    if shuffle_positions:
        values["random_food"] = True
        values["random_hub"] = True
    return argparse.Namespace(**values)
