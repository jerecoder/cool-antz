"""Offline probes for JAX MAPPO communication checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env import ACTION_STAY, AntByteForagingEnv, write_value_count
from ant_byte_env.rendering import _env_from_args, _jax_render_reset_options, _render_frame
from ant_byte_env.runs import write_json
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.cli import parse_args
from ant_byte_env.training.jax_mappo.core import (
    JaxMAPPOParams,
    build_actor_observations,
    build_central_observations,
    flatten_agent_actions,
    get_action_and_value,
)
from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training

COMMUNICATION_PROBE_ROOT = Path("runs/autoresearch/communication_bits")
COMMUNICATION_PROBE_FILENAME = "communication_probe.json"
MAJOR_NONZERO_WRITE_FRACTION = 0.05


def probe_communication_checkpoint(
    checkpoint_path: Path,
    *,
    output_dir: Path = COMMUNICATION_PROBE_ROOT,
    num_episodes: int = 4,
    seed_offset: int = 2_000_000,
    render_rollouts: bool = True,
    max_render_frames: int | None = None,
    tile_size: int | None = 16,
) -> dict[str, Any]:
    """Run sampled and deterministic communication probes for a checkpoint."""

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    if max_render_frames is not None and max_render_frames < 1:
        raise ValueError("max_render_frames must be at least 1.")
    output_dir = Path(output_dir)
    media_dir = output_dir / "rollouts"
    if render_rollouts:
        media_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    args = _checkpoint_args_with_defaults(read_checkpoint(checkpoint_path).get("args", {}))
    params = _load_probe_params(Path(checkpoint_path), args)
    rollout_paths = {
        "sampled": media_dir / "sampled_rollout.mp4" if render_rollouts else None,
        "deterministic": media_dir / "deterministic_rollout.mp4" if render_rollouts else None,
    }
    sampled = _probe_mode(
        params=params,
        args=args,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
        deterministic=False,
        rollout_path=rollout_paths["sampled"],
        max_render_frames=max_render_frames,
        tile_size=tile_size,
    )
    deterministic = _probe_mode(
        params=params,
        args=args,
        num_episodes=num_episodes,
        seed_offset=seed_offset + 100_000,
        deterministic=True,
        rollout_path=rollout_paths["deterministic"],
        max_render_frames=max_render_frames,
        tile_size=tile_size,
    )
    probe_path = output_dir / COMMUNICATION_PROBE_FILENAME
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "probe_path": str(probe_path),
        "output_dir": str(output_dir),
        "num_episodes": int(num_episodes),
        "write_bits": int(args.write_bits),
        "write_value_count": write_value_count(args.write_bits),
        "rollout_artifact_paths": {
            name: str(path) if path is not None else None for name, path in rollout_paths.items()
        },
        "sampled": sampled,
        "deterministic": deterministic,
    }
    write_json(probe_path, payload)
    return payload


def write_action_bit_summary(
    action_histogram: np.ndarray,
    *,
    write_bits: int,
) -> dict[str, Any]:
    """Return bit-use rates and normalized entropy from a write-action histogram."""

    counts = np.asarray(action_histogram, dtype=np.int64)
    nonzero_count = int(counts[1:].sum())
    rates: list[float] = []
    entropies: list[float] = []
    for bit_index in range(write_bits):
        bit_count = sum(
            int(counts[value])
            for value in range(1, min(len(counts), write_value_count(write_bits)))
            if value & (1 << bit_index)
        )
        rate = bit_count / nonzero_count if nonzero_count > 0 else 0.0
        rates.append(float(rate))
        entropies.append(_binary_entropy(rate) if 0.0 < rate < 1.0 else 0.0)
    distinct_nonzero_values = [
        int(value)
        for value, count in enumerate(counts)
        if value > 0 and int(count) > 0
    ]
    major_nonzero_value_fractions = {
        str(value): float(counts[value] / nonzero_count)
        for value in distinct_nonzero_values
        if nonzero_count > 0 and counts[value] / nonzero_count >= MAJOR_NONZERO_WRITE_FRACTION
    }
    return {
        "nonzero_write_count": nonzero_count,
        "per_bit_activation_rates": rates,
        "per_bit_entropy": entropies,
        "write_bit_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "distinct_nonzero_values": distinct_nonzero_values,
        "major_nonzero_values": [
            int(value) for value in major_nonzero_value_fractions.keys()
        ],
        "major_nonzero_value_fractions": major_nonzero_value_fractions,
    }


def _load_probe_params(checkpoint_path: Path, args: argparse.Namespace) -> JaxMAPPOParams:
    env = _env_from_args(args, render_mode="rgb_array", tile_size=1)
    try:
        obs, _ = env.reset(
            seed=args.seed,
            options=_jax_render_reset_options(args, seed=args.seed),
        )
    finally:
        env.close()
    obs_batch = _obs_batch(obs)
    central_obs = build_central_observations(
        obs_batch,
        food_scale=args.food_count,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs_batch,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    checkpoint = load_checkpoint_for_training(
        checkpoint_path,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        target_write_bits=args.write_bits,
        actor_vision_radius=args.actor_vision_radius,
    )
    return jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])


def _probe_mode(
    *,
    params: JaxMAPPOParams,
    args: argparse.Namespace,
    num_episodes: int,
    seed_offset: int,
    deterministic: bool,
    rollout_path: Path | None,
    max_render_frames: int | None,
    tile_size: int | None,
) -> dict[str, Any]:
    action_counts = np.zeros(write_value_count(args.write_bits), dtype=np.int64)
    final_byte_counts = np.zeros(write_value_count(args.write_bits), dtype=np.int64)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    delivered_food: list[float] = []
    successes: list[float] = []
    select_actions = jax.jit(
        lambda obs_batch, action_key: _select_actions_from_batch(
            params=params,
            args=args,
            obs_batch=obs_batch,
            key=action_key,
            deterministic=deterministic,
        )
    )

    for episode_index in range(num_episodes):
        should_render = rollout_path is not None and episode_index == 0
        env = _env_from_args(
            args,
            render_mode="rgb_array" if should_render else None,
            tile_size=tile_size,
        )
        writer = (
            imageio.get_writer(rollout_path, fps=AntByteForagingEnv.metadata["render_fps"])
            if should_render
            else None
        )
        frame_count = 0
        try:
            reset_seed = int(args.seed) + seed_offset + episode_index
            obs, info = env.reset(
                seed=reset_seed,
                options=_jax_render_reset_options(args, seed=reset_seed),
            )
            if writer is not None:
                frame_count = _append_probe_frame(
                    writer,
                    env,
                    obs,
                    args=args,
                    frame_count=frame_count,
                    max_render_frames=max_render_frames,
                )
            key = jax.random.PRNGKey(reset_seed)
            episode_return = 0.0
            episode_length = int(args.max_steps)
            episode_terminated = False
            for step_index in range(int(args.max_steps)):
                key, action_key = jax.random.split(key)
                actions = select_actions(_obs_batch(obs), action_key)
                agent_actions = np.asarray(actions)[0]
                write_values = _applied_probe_write_values(
                    agent_actions,
                    write_while_moving=bool(getattr(args, "write_while_moving", False)),
                )
                action_counts += np.bincount(write_values, minlength=action_counts.shape[0])
                obs, reward, terminated, truncated, info = env.step(
                    np.asarray(flatten_agent_actions(actions))[0]
                )
                episode_return += float(reward)
                if writer is not None:
                    frame_count = _append_probe_frame(
                        writer,
                        env,
                        obs,
                        args=args,
                        frame_count=frame_count,
                        max_render_frames=max_render_frames,
                    )
                if terminated or truncated:
                    episode_length = step_index + 1
                    episode_terminated = bool(terminated)
                    break
            final_byte_counts += np.bincount(
                np.asarray(obs["bytes"]).reshape(-1).astype(np.int64),
                minlength=final_byte_counts.shape[0],
            )
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            delivered = float(info["delivered_food"])
            delivered_food.append(delivered)
            successes.append(float(episode_terminated))
        finally:
            if writer is not None:
                writer.close()
            env.close()

    bit_summary = write_action_bit_summary(action_counts, write_bits=int(args.write_bits))
    return {
        "write_action_histogram": _histogram_payload(action_counts),
        "final_grid_byte_histogram": _histogram_payload(final_byte_counts),
        "per_bit_activation_rates": bit_summary["per_bit_activation_rates"],
        "per_bit_entropy": bit_summary["per_bit_entropy"],
        "write_bit_entropy": bit_summary["write_bit_entropy"],
        "distinct_nonzero_values": bit_summary["distinct_nonzero_values"],
        "major_nonzero_values": bit_summary["major_nonzero_values"],
        "major_nonzero_value_fractions": bit_summary["major_nonzero_value_fractions"],
        "delivery_metrics": {
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "mean_delivered_food": float(np.mean(delivered_food)) if delivered_food else 0.0,
            "mean_delivered_fraction": (
                float(np.mean(delivered_food)) / max(float(args.food_count), 1.0)
                if delivered_food
                else 0.0
            ),
            "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        },
        "rollout_artifact_path": str(rollout_path) if rollout_path is not None else None,
    }


def _applied_probe_write_values(
    agent_actions: np.ndarray,
    *,
    write_while_moving: bool,
) -> np.ndarray:
    actions = np.asarray(agent_actions)
    if write_while_moving:
        return actions[:, 1].astype(np.int64)
    return np.where(actions[:, 0] == ACTION_STAY, actions[:, 1], 0).astype(np.int64)


def _select_actions(
    *,
    params: JaxMAPPOParams,
    args: argparse.Namespace,
    obs: dict[str, np.ndarray],
    key: jax.Array,
    deterministic: bool,
) -> jax.Array:
    return _select_actions_from_batch(
        params=params,
        args=args,
        obs_batch=_obs_batch(obs),
        key=key,
        deterministic=deterministic,
    )


def _select_actions_from_batch(
    *,
    params: JaxMAPPOParams,
    args: argparse.Namespace,
    obs_batch: dict[str, jax.Array],
    key: jax.Array,
    deterministic: bool,
) -> jax.Array:
    central_obs = build_central_observations(
        obs_batch,
        food_scale=args.food_count,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs_batch,
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
        key,
        deterministic=deterministic,
    )
    return actions


def _append_probe_frame(
    writer: Any,
    env: AntByteForagingEnv,
    obs: dict[str, np.ndarray],
    *,
    args: argparse.Namespace,
    frame_count: int,
    max_render_frames: int | None,
) -> int:
    if max_render_frames is not None and frame_count >= max_render_frames:
        return frame_count
    writer.append_data(_render_frame(env, obs, args=args, show_vision=True))
    return frame_count + 1


def _checkpoint_args_with_defaults(saved_args: dict[str, object]) -> argparse.Namespace:
    args = parse_args([])
    for key, value in saved_args.items():
        setattr(args, key, value)
    return args


def _obs_batch(obs: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {
        name: jnp.expand_dims(jnp.asarray(value), axis=0)
        for name, value in obs.items()
    }


def _histogram_payload(counts: np.ndarray) -> dict[str, int]:
    return {str(index): int(count) for index, count in enumerate(np.asarray(counts))}


def _binary_entropy(rate: float) -> float:
    return float(-(rate * np.log(rate) + (1.0 - rate) * np.log(1.0 - rate)) / np.log(2.0))
