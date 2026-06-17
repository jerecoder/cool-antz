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
from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training


def evaluate_params(
    *,
    params: JaxMAPPOParams,
    args: argparse.Namespace,
    num_episodes: int,
    seed_offset: int = 1_000_000,
    deterministic: bool = True,
    shuffle_positions: bool = True,
) -> dict[str, float]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    eval_source_args = _evaluation_args_with_position_shuffle(
        args,
        shuffle_positions=shuffle_positions,
    )
    env = JaxAntByteForagingEnv(
        width=eval_source_args.width,
        height=eval_source_args.height,
        num_ants=eval_source_args.num_ants,
        food_count=eval_source_args.food_count,
        food_source_count=eval_source_args.food_sources,
        max_steps=eval_source_args.max_steps,
        random_food=eval_source_args.random_food,
        random_hub=eval_source_args.random_hub,
        step_penalty=eval_source_args.step_penalty,
        write_penalty=eval_source_args.write_penalty,
        write_bits=eval_source_args.write_bits,
        write_while_moving=bool(getattr(eval_source_args, "write_while_moving", False)),
    )
    eval_args = argparse.Namespace(**{**vars(eval_source_args), "num_envs": 1})
    key = jax.random.PRNGKey(eval_args.seed + seed_offset)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    delivered_food: list[float] = []
    delivered_fractions: list[float] = []
    successes: list[float] = []

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs = reset_batch(args=eval_args, env=env, key=reset_key)
        episode_return = 0.0
        episode_length = int(eval_args.max_steps)
        episode_terminated = False

        for step_index in range(int(eval_args.max_steps)):
            key, action_key = jax.random.split(key)
            central_obs = build_central_observations(
                obs,
                food_scale=eval_args.food_count,
                write_bits=eval_args.write_bits,
                obs_width=eval_args.obs_width,
                obs_height=eval_args.obs_height,
            )
            actor_obs = build_actor_observations(
                obs,
                food_scale=eval_args.food_count,
                actor_vision_radius=eval_args.actor_vision_radius,
                write_bits=eval_args.write_bits,
                obs_width=eval_args.obs_width,
                obs_height=eval_args.obs_height,
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
        delivered_fractions.append(delivered / max(float(eval_args.food_count), 1.0))
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
    shuffle_positions: bool = True,
) -> dict[str, float]:
    raw_checkpoint = read_checkpoint(checkpoint_path)
    args = _checkpoint_args_with_defaults(raw_checkpoint.get("args", {}))
    central_obs_dim, actor_obs_dim = _checkpoint_observation_dims(args)
    checkpoint = load_checkpoint_for_training(
        checkpoint_path,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        target_write_bits=args.write_bits,
        actor_vision_radius=args.actor_vision_radius,
    )
    return evaluate_params(
        params=checkpoint["params"],
        args=args,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
        deterministic=deterministic,
        shuffle_positions=shuffle_positions,
    )


def _checkpoint_observation_dims(args: argparse.Namespace) -> tuple[int, int]:
    env = JaxAntByteForagingEnv(
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
        write_while_moving=bool(getattr(args, "write_while_moving", False)),
    )
    shape_args = argparse.Namespace(**{**vars(args), "num_envs": 1})
    _, obs = reset_batch(args=shape_args, env=env, key=jax.random.PRNGKey(args.seed))
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
    return central_obs.shape[-1], actor_obs.shape[-1]


def _checkpoint_args_with_defaults(saved_args: dict[str, object]) -> argparse.Namespace:
    args = parse_args([])
    for key, value in saved_args.items():
        setattr(args, key, value)
    return args


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
