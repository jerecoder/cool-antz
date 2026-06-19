"""Evaluation helpers for JAX MAPPO checkpoints and parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env.jax_autocurriculum_env import JaxAntByteAutoCurriculumEnv
from ant_byte_env.jax_env import JaxAntByteForagingEnv
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.cli import parse_args
from ant_byte_env.training.jax_mappo.core import (
    JaxMAPPOParams,
    build_actor_observations,
    build_central_observations,
    flatten_agent_actions,
    get_action_logits,
    get_action_and_value,
)
from ant_byte_env.training.jax_mappo.curriculum import reset_batch
from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training


EVALUATION_ACTION_MODES = {
    "deterministic",
    "sampled",
    "greedy_move_greedy_write",
    "greedy_move_sampled_write",
    "sampled_move_greedy_write",
    "sampled_move_sampled_write",
    "greedy_move_zero_write",
    "sampled_move_zero_write",
}


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
    env = _make_eval_env(eval_source_args)
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
    steps_per_delivered_food: list[float] = []
    ant_steps_per_delivered_food: list[float] = []
    delivered_per_1000_ant_steps: list[float] = []
    successes: list[float] = []

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs = reset_batch(args=eval_args, env=env, key=reset_key)
        episode_return = 0.0
        episode_length = int(eval_args.max_steps)
        episode_terminated = False

        for step_index in range(int(eval_args.max_steps)):
            key, action_key = jax.random.split(key)
            state, obs, reward, terminated, truncated = step_fn(state, obs, action_key)
            episode_return += float(np.asarray(reward)[0])
            episode_terminated = bool(np.asarray(terminated)[0])
            if episode_terminated or bool(np.asarray(truncated)[0]):
                episode_length = step_index + 1
                break

        delivered = float(np.asarray(state.delivered_food)[0])
        delivered_denominator = max(delivered, 1.0)
        ant_steps = float(episode_length) * max(float(eval_args.num_ants), 1.0)
        delivered_food.append(delivered)
        delivered_fractions.append(delivered / max(float(eval_args.food_count), 1.0))
        steps_per_delivered_food.append(float(episode_length) / delivered_denominator)
        ant_steps_per_delivered_food.append(ant_steps / delivered_denominator)
        delivered_per_1000_ant_steps.append(
            1000.0 * delivered / max(ant_steps, 1.0)
        )
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        successes.append(float(episode_terminated))

    return {
        "eval_success_rate": float(np.mean(successes)),
        "eval_mean_delivered_food": float(np.mean(delivered_food)),
        "eval_mean_delivered_fraction": float(np.mean(delivered_fractions)),
        "eval_mean_episode_return": float(np.mean(episode_returns)),
        "eval_mean_episode_length": float(np.mean(episode_lengths)),
        "eval_mean_steps_per_delivered_food": float(np.mean(steps_per_delivered_food)),
        "eval_mean_ant_steps_per_delivered_food": float(
            np.mean(ant_steps_per_delivered_food)
        ),
        "eval_mean_delivered_food_per_1000_ant_steps": float(
            np.mean(delivered_per_1000_ant_steps)
        ),
    }


def _evaluation_step(
    *,
    env: JaxAntByteForagingEnv | JaxAntByteAutoCurriculumEnv,
    args: argparse.Namespace,
    params: JaxMAPPOParams,
    state: Any,
    obs: Any,
    key: Any,
    action_mode: str,
    move_temperature: float,
    write_temperature: float,
) -> tuple[Any, Any, Any, Any, Any]:
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
    actions = _evaluation_actions_for_mode(
        params,
        actor_obs,
        central_obs,
        key,
        action_mode=action_mode,
        move_temperature=move_temperature,
        write_temperature=write_temperature,
    )
    if bool(getattr(args, "write_action_ablation", False)):
        actions = actions.at[..., 1].set(0)
    state, obs, reward, terminated, truncated, _ = jax.vmap(env.step)(
        state,
        flatten_agent_actions(actions),
    )
    return state, obs, reward, terminated, truncated


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


def _evaluation_actions_for_mode(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    central_obs: jax.Array,
    key: jax.Array,
    *,
    action_mode: str,
    move_temperature: float = 1.0,
    write_temperature: float = 1.0,
) -> jax.Array:
    if action_mode in {"deterministic", "sampled"}:
        actions, _, _, _ = get_action_and_value(
            params,
            actor_obs,
            central_obs,
            key,
            deterministic=action_mode == "deterministic",
        )
        return actions

    move_mode, write_mode = _split_hybrid_action_mode(action_mode)
    move_logits, write_logits = get_action_logits(params, actor_obs)
    move_key, write_key = jax.random.split(key)
    move_actions = _head_actions_for_mode(
        move_logits,
        move_key,
        mode=move_mode,
        temperature=move_temperature,
    )
    write_actions = _head_actions_for_mode(
        write_logits,
        write_key,
        mode=write_mode,
        temperature=write_temperature,
    )
    return jnp.stack([move_actions, write_actions], axis=-1).astype(jnp.int32)


def _head_actions_for_mode(
    logits: jax.Array,
    key: jax.Array,
    *,
    mode: str,
    temperature: float,
) -> jax.Array:
    if mode == "greedy":
        return jnp.argmax(logits, axis=-1)
    if mode == "sampled":
        return jax.random.categorical(key, logits / float(temperature), axis=-1)
    if mode == "zero":
        return jnp.zeros(logits.shape[:-1], dtype=jnp.int32)
    raise ValueError(f"unknown evaluation action head mode {mode!r}.")


def _split_hybrid_action_mode(action_mode: str) -> tuple[str, str]:
    if action_mode == "greedy_move_greedy_write":
        return "greedy", "greedy"
    if action_mode == "greedy_move_sampled_write":
        return "greedy", "sampled"
    if action_mode == "sampled_move_greedy_write":
        return "sampled", "greedy"
    if action_mode == "sampled_move_sampled_write":
        return "sampled", "sampled"
    if action_mode == "greedy_move_zero_write":
        return "greedy", "zero"
    if action_mode == "sampled_move_zero_write":
        return "sampled", "zero"
    raise ValueError(f"unknown evaluation action mode {action_mode!r}.")


def validate_evaluation_action_mode(action_mode: str) -> str:
    if action_mode not in EVALUATION_ACTION_MODES:
        choices = ", ".join(sorted(EVALUATION_ACTION_MODES))
        raise ValueError(f"unknown evaluation action mode {action_mode!r}; choices: {choices}")
    return action_mode


def _validate_evaluation_temperature(value: float, *, name: str) -> float:
    temperature = float(value)
    if temperature <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return temperature


def _evaluation_action_mode_default(
    args: argparse.Namespace,
    *,
    deterministic: bool | None,
    action_mode: str | None,
) -> str:
    if action_mode is not None:
        if deterministic is not None:
            raise ValueError("Pass either deterministic or action_mode, not both.")
        return validate_evaluation_action_mode(action_mode)
    if deterministic is not None:
        return "deterministic" if bool(deterministic) else "sampled"
    return "sampled" if bool(getattr(args, "autocurriculum", False)) else "deterministic"


def _checkpoint_observation_dims(args: argparse.Namespace) -> tuple[int, int]:
    env = _make_eval_env(args)
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


def _make_eval_env(args: argparse.Namespace) -> JaxAntByteForagingEnv | JaxAntByteAutoCurriculumEnv:
    common_kwargs = {
        "width": args.width,
        "height": args.height,
        "num_ants": args.num_ants,
        "food_count": args.food_count,
        "food_source_count": args.food_sources,
        "max_steps": args.max_steps,
        "random_food": args.random_food,
        "random_hub": args.random_hub,
        "step_penalty": args.step_penalty,
        "completion_bonus": getattr(args, "completion_bonus", 0.0),
        "write_penalty": args.write_penalty,
        "write_bits": args.write_bits,
        "write_while_moving": bool(getattr(args, "write_while_moving", False)),
    }
    if bool(getattr(args, "autocurriculum", False)):
        return JaxAntByteAutoCurriculumEnv(
            **common_kwargs,
            start_size=int(getattr(args, "autocurriculum_start_size", 4)),
            success_cookies=int(getattr(args, "autocurriculum_success_cookies", 6)),
            actor_vision_radius=int(getattr(args, "actor_vision_radius", 1)),
        )
    return JaxAntByteForagingEnv(**common_kwargs)


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
