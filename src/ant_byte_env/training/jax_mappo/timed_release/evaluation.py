"""Evaluation helpers for timed-release cooperative MAPPO checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env.training.jax_mappo.curriculum import reset_batch
from ant_byte_env.training.jax_mappo.evaluation import (
    _evaluation_actions_for_mode,
    _evaluation_args_with_position_shuffle,
    _evaluation_action_mode_default,
    _validate_evaluation_temperature,
)
from ant_byte_env.training.jax_mappo.models import critic_forward_kwargs_from_args
from ant_byte_env.training.jax_mappo.observations import (
    build_actor_observations,
    build_central_observations,
    flatten_agent_actions,
    food_observation_scale,
)
from ant_byte_env.training.jax_mappo.policy import get_action_and_value
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams
from ant_byte_env.training.jax_mappo.timed_release.cli import parse_args
from ant_byte_env.training.jax_mappo.timed_release.env import (
    TimedReleaseJaxEnv,
    make_timed_release_env,
)


def evaluate_params(
    *,
    params: JaxMAPPOParams,
    args: argparse.Namespace,
    num_episodes: int,
    seed_offset: int = 1_000_000,
    deterministic: bool | None = None,
    action_mode: str | None = None,
    move_temperature: float = 1.0,
    write_temperature: float = 1.0,
    shuffle_positions: bool = True,
) -> dict[str, float]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    eval_source_args = _evaluation_args_with_position_shuffle(
        args,
        shuffle_positions=shuffle_positions,
    )
    env = make_timed_release_env(eval_source_args)
    eval_args = argparse.Namespace(**{**vars(eval_source_args), "num_envs": 1})
    resolved_action_mode = _evaluation_action_mode_default(
        eval_args,
        deterministic=deterministic,
        action_mode=action_mode,
    )
    move_temperature = _validate_evaluation_temperature(
        move_temperature,
        name="move_temperature",
    )
    write_temperature = _validate_evaluation_temperature(
        write_temperature,
        name="write_temperature",
    )
    key = jax.random.PRNGKey(eval_args.seed + seed_offset)
    step_fn = jax.jit(
        lambda current_state, current_obs, action_key: _evaluation_step(
            env=env,
            args=eval_args,
            params=params,
            state=current_state,
            obs=current_obs,
            key=action_key,
            action_mode=resolved_action_mode,
            move_temperature=move_temperature,
            write_temperature=write_temperature,
        )
    )

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    delivered_food: list[float] = []
    delivered_fractions: list[float] = []
    successes: list[float] = []
    active_ant_steps: list[float] = []
    rank_pickups: list[np.ndarray] = []
    rank_deliveries: list[np.ndarray] = []
    rank_writes: list[np.ndarray] = []
    rank_unique_cells: list[np.ndarray] = []
    rank_first_pickup_steps: list[np.ndarray] = []
    rank_first_delivery_steps: list[np.ndarray] = []
    rank_pickup_latencies: list[np.ndarray] = []
    previous_obs: Any | None = None
    previous_food: Any | None = None
    release_steps = np.asarray(env.release_steps, dtype=np.float32)

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs = reset_batch(
            args=eval_args,
            env=env,
            key=reset_key,
            previous_obs=previous_obs,
            previous_food=previous_food,
        )
        episode_return = 0.0
        episode_length = int(eval_args.max_steps)
        episode_terminated = False
        pickups = np.zeros(eval_args.num_ants, dtype=np.float32)
        deliveries = np.zeros(eval_args.num_ants, dtype=np.float32)
        writes = np.zeros(eval_args.num_ants, dtype=np.float32)
        first_pickup = np.full(eval_args.num_ants, -1.0, dtype=np.float32)
        first_delivery = np.full(eval_args.num_ants, -1.0, dtype=np.float32)
        active_steps = np.zeros(eval_args.num_ants, dtype=np.float32)
        unique_cells = np.zeros(
            (eval_args.num_ants, eval_args.height, eval_args.width),
            dtype=bool,
        )
        _mark_active_positions(unique_cells, obs)

        for step_index in range(int(eval_args.max_steps)):
            active_before = np.asarray(obs["active_mask"])[0].astype(bool)
            active_steps += active_before.astype(np.float32)
            key, action_key = jax.random.split(key)
            state, obs, reward, terminated, truncated, infos, _ = step_fn(
                state,
                obs,
                action_key,
            )
            episode_return += float(np.asarray(reward)[0])
            pickup_events = np.asarray(infos.pickup_events_per_ant)[0].astype(np.float32)
            delivery_events = np.asarray(infos.delivery_events_per_ant)[0].astype(np.float32)
            write_events = np.asarray(infos.write_attempts_per_ant)[0].astype(np.float32)
            pickups += pickup_events
            deliveries += delivery_events
            writes += write_events
            event_step = float(step_index + 1)
            first_pickup = np.where(
                (pickup_events > 0) & (first_pickup < 0),
                event_step,
                first_pickup,
            )
            first_delivery = np.where(
                (delivery_events > 0) & (first_delivery < 0),
                event_step,
                first_delivery,
            )
            _mark_active_positions(unique_cells, obs)
            episode_terminated = bool(np.asarray(terminated)[0])
            if episode_terminated or bool(np.asarray(truncated)[0]):
                episode_length = step_index + 1
                break

        delivered = _first_env_value(state.base.delivered_food)
        rank_pickups.append(pickups)
        rank_deliveries.append(deliveries)
        rank_writes.append(writes)
        rank_unique_cells.append(unique_cells.sum(axis=(1, 2)).astype(np.float32))
        rank_first_pickup_steps.append(first_pickup)
        rank_first_delivery_steps.append(first_delivery)
        rank_pickup_latencies.append(
            np.where(first_pickup >= 0, first_pickup - release_steps, -1.0).astype(np.float32)
        )
        active_ant_steps.append(float(np.sum(active_steps)))
        delivered_food.append(delivered)
        delivered_fractions.append(delivered / max(float(eval_args.food_count), 1.0))
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        successes.append(float(episode_terminated))
        previous_obs = obs
        previous_food = state.base.initial_food

    metrics = {
        "eval_success_rate": float(np.mean(successes)),
        "eval_mean_delivered_food": float(np.mean(delivered_food)),
        "eval_mean_delivered_fraction": float(np.mean(delivered_fractions)),
        "eval_mean_episode_return": float(np.mean(episode_returns)),
        "eval_mean_episode_length": float(np.mean(episode_lengths)),
        "eval_mean_active_ant_steps": float(np.mean(active_ant_steps)),
        "eval_mean_delivered_food_per_1000_active_ant_steps": float(
            1000.0
            * np.mean(
                np.asarray(delivered_food, dtype=np.float32)
                / np.maximum(np.asarray(active_ant_steps, dtype=np.float32), 1.0)
            )
        ),
        "eval_mean_pickups": float(np.mean(np.sum(rank_pickups, axis=1))),
    }
    rank_arrays = {
        "pickups": np.asarray(rank_pickups, dtype=np.float32),
        "deliveries": np.asarray(rank_deliveries, dtype=np.float32),
        "writes": np.asarray(rank_writes, dtype=np.float32),
        "unique_cells": np.asarray(rank_unique_cells, dtype=np.float32),
        "first_pickup_step": np.asarray(rank_first_pickup_steps, dtype=np.float32),
        "first_delivery_step": np.asarray(rank_first_delivery_steps, dtype=np.float32),
        "release_to_pickup_latency": np.asarray(rank_pickup_latencies, dtype=np.float32),
    }
    for rank in range(int(eval_args.num_ants)):
        for name, values in rank_arrays.items():
            metrics[f"eval_rank_{rank}_mean_{name}"] = float(np.mean(values[:, rank]))
    return metrics


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    num_episodes: int,
    seed_offset: int = 1_000_000,
    deterministic: bool | None = None,
    action_mode: str | None = None,
    move_temperature: float = 1.0,
    write_temperature: float = 1.0,
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
        target_num_ants=args.num_ants,
        target_critic_architecture=getattr(args, "critic_architecture", "mlp"),
    )
    return evaluate_params(
        params=checkpoint["params"],
        args=args,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
        deterministic=deterministic,
        action_mode=action_mode,
        move_temperature=move_temperature,
        write_temperature=write_temperature,
        shuffle_positions=shuffle_positions,
    )


def _evaluation_step(
    *,
    env: TimedReleaseJaxEnv,
    args: argparse.Namespace,
    params: JaxMAPPOParams,
    state: Any,
    obs: Any,
    key: Any,
    action_mode: str,
    move_temperature: float,
    write_temperature: float,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    central_obs = build_central_observations(
        obs,
        food_scale=food_scale,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actions = _evaluation_actions_for_mode(
        params,
        actor_obs,
        central_obs,
        key,
        action_mode=action_mode,
        move_temperature=move_temperature,
        write_temperature=write_temperature,
        **critic_forward_kwargs_from_args(args),
    )
    if bool(getattr(args, "write_action_ablation", False)):
        actions = actions.at[..., 1].set(0)
    state, obs, reward, terminated, truncated, infos = jax.vmap(env.step)(
        state,
        flatten_agent_actions(actions),
    )
    if str(getattr(args, "reward_mode", "forage")) == "explore":
        reward = getattr(
            infos,
            "newly_visited_cells",
            jnp.zeros_like(reward, dtype=jnp.int32),
        ).astype(jnp.float32)
    return state, obs, reward, terminated, truncated, infos, actions


def _checkpoint_args_with_defaults(saved_args: dict[str, object]) -> argparse.Namespace:
    args = parse_args([])
    for key, value in saved_args.items():
        setattr(args, key, value)
    if not hasattr(args, "release_interval"):
        args.release_interval = 150
    if not hasattr(args, "initial_active_ants"):
        args.initial_active_ants = 1
    return args


def _checkpoint_observation_dims(args: argparse.Namespace) -> tuple[int, int]:
    env = make_timed_release_env(args)
    shape_args = argparse.Namespace(**{**vars(args), "num_envs": 1})
    _, obs = reset_batch(args=shape_args, env=env, key=jax.random.PRNGKey(args.seed))
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    central_obs = build_central_observations(
        obs,
        food_scale=food_scale,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    return central_obs.shape[-1], actor_obs.shape[-1]


def _mark_active_positions(unique_cells: np.ndarray, obs: Any) -> None:
    positions = np.asarray(obs["ants_pos"])[0].astype(np.int32)
    active_mask = np.asarray(obs["active_mask"])[0].astype(bool)
    for rank, active in enumerate(active_mask):
        if active:
            x_pos, y_pos = positions[rank]
            unique_cells[rank, int(y_pos), int(x_pos)] = True


def _first_env_value(value: Any) -> float:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return float(array[0])
