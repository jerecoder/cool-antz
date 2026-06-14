"""Evaluation helpers for JAX MAPPO checkpoints and parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from ant_byte_env.jax_env import JaxAntByteForagingEnv
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.cli import parse_args
from ant_byte_env.training.jax_mappo.core import (
    JaxMAPPOParams,
    build_actor_observations,
    build_central_observations,
    flatten_agent_actions,
    get_action_and_value,
)
from ant_byte_env.training.jax_mappo.curriculum import reset_batch


def evaluate_params(
    *,
    params: JaxMAPPOParams,
    args: argparse.Namespace,
    num_episodes: int,
    seed_offset: int = 1_000_000,
    deterministic: bool = True,
) -> dict[str, float]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    env = JaxAntByteForagingEnv(
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
    eval_args = argparse.Namespace(**{**vars(args), "num_envs": 1})
    key = jax.random.PRNGKey(args.seed + seed_offset)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    delivered_food: list[float] = []
    delivered_fractions: list[float] = []
    successes: list[float] = []

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs = reset_batch(args=eval_args, env=env, key=reset_key)
        episode_return = 0.0
        episode_length = int(args.max_steps)
        episode_terminated = False

        for step_index in range(int(args.max_steps)):
            key, action_key = jax.random.split(key)
            central_obs = build_central_observations(
                obs,
                food_scale=args.food_count,
                write_bits=args.write_bits,
                obs_width=args.obs_width,
                obs_height=args.obs_height,
            )
            actor_obs = build_actor_observations(
                obs,
                food_scale=args.food_count,
                actor_vision_radius=args.actor_vision_radius,
                write_bits=args.write_bits,
                obs_width=args.obs_width,
                obs_height=args.obs_height,
            )
            actions, _, _, _ = get_action_and_value(
                params,
                actor_obs,
                central_obs,
                action_key,
                deterministic=deterministic,
            )
            state, obs, reward, terminated, truncated, _ = jax.vmap(env.step)(
                state,
                flatten_agent_actions(actions),
            )
            episode_return += float(np.asarray(reward)[0])
            episode_terminated = bool(np.asarray(terminated)[0])
            if episode_terminated or bool(np.asarray(truncated)[0]):
                episode_length = step_index + 1
                break

        delivered = float(np.asarray(state.delivered_food)[0])
        delivered_food.append(delivered)
        delivered_fractions.append(delivered / max(float(args.food_count), 1.0))
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        successes.append(float(episode_terminated))

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
    seed_offset: int = 1_000_000,
    deterministic: bool = True,
) -> dict[str, float]:
    checkpoint = read_checkpoint(checkpoint_path)
    args = _checkpoint_args_with_defaults(checkpoint.get("args", {}))
    return evaluate_params(
        params=checkpoint["params"],
        args=args,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
        deterministic=deterministic,
    )


def _checkpoint_args_with_defaults(saved_args: dict[str, object]) -> argparse.Namespace:
    args = parse_args([])
    for key, value in saved_args.items():
        setattr(args, key, value)
    return args
