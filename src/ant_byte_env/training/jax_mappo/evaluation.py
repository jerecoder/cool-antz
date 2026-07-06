"""Evaluation helpers for JAX MAPPO checkpoints and parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.cli import parse_args
from ant_byte_env.training.jax_mappo.env_factory import JaxMappoEnv, make_jax_mappo_env
from ant_byte_env.training.jax_mappo.models import (
    critic_forward_kwargs_from_args,
    get_action_logits,
)
from ant_byte_env.training.jax_mappo.observations import (
    build_actor_observations,
    build_central_observations,
    flatten_agent_actions,
    food_observation_scale,
)
from ant_byte_env.training.jax_mappo.policy import (
    get_action_and_value,
)
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams
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
    pickups: list[float] = []
    pickup_to_delivery_rates: list[float] = []
    write_action_rates: list[float] = []
    applied_write_rates: list[float] = []
    applied_nonzero_write_rates: list[float] = []
    overwrite_rates: list[float] = []
    steps_per_delivered_food: list[float] = []
    ant_steps_per_delivered_food: list[float] = []
    delivered_per_1000_ant_steps: list[float] = []
    successes: list[float] = []
    previous_obs: Any | None = None
    previous_food: Any | None = None

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
        episode_nonzero_write_actions = 0.0
        episode_applied_writes = 0.0
        episode_overwrites = 0.0

        for step_index in range(int(eval_args.max_steps)):
            key, action_key = jax.random.split(key)
            state, obs, reward, terminated, truncated, infos, actions = step_fn(
                state,
                obs,
                action_key,
            )
            episode_return += float(np.asarray(reward)[0])
            write_values = np.asarray(actions)[0, :, 1]
            episode_nonzero_write_actions += float(np.count_nonzero(write_values))
            episode_applied_writes += float(np.asarray(infos.num_writes)[0])
            episode_overwrites += float(np.asarray(infos.num_overwrites)[0])
            episode_terminated = bool(np.asarray(terminated)[0])
            if episode_terminated or bool(np.asarray(truncated)[0]):
                episode_length = step_index + 1
                break

        delivered = _first_env_value(state.delivered_food)
        remaining = _first_env_sum(state.food)
        initial = _first_env_value(state.initial_food_total)
        picked_up = max(0.0, initial - remaining)
        delivered_denominator = max(delivered, 1.0)
        ant_steps = float(episode_length) * max(float(eval_args.num_ants), 1.0)
        delivered_food.append(delivered)
        delivered_fractions.append(delivered / max(float(eval_args.food_count), 1.0))
        pickups.append(picked_up)
        pickup_to_delivery_rates.append(delivered / max(picked_up, 1.0))
        write_action_rates.append(episode_nonzero_write_actions / max(ant_steps, 1.0))
        applied_write_rates.append(episode_applied_writes / max(ant_steps, 1.0))
        applied_nonzero_write_rates.append(
            episode_nonzero_write_actions / max(ant_steps, 1.0)
        )
        overwrite_rates.append(episode_overwrites / max(ant_steps, 1.0))
        steps_per_delivered_food.append(float(episode_length) / delivered_denominator)
        ant_steps_per_delivered_food.append(ant_steps / delivered_denominator)
        delivered_per_1000_ant_steps.append(
            1000.0 * delivered / max(ant_steps, 1.0)
        )
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        successes.append(float(episode_terminated))
        previous_obs = obs
        previous_food = getattr(state, "initial_food", None)

    return {
        "eval_success_rate": float(np.mean(successes)),
        "eval_mean_delivered_food": float(np.mean(delivered_food)),
        "eval_mean_delivered_fraction": float(np.mean(delivered_fractions)),
        "eval_mean_pickups": float(np.mean(pickups)),
        "eval_mean_pickup_to_delivery_rate": float(np.mean(pickup_to_delivery_rates)),
        "eval_mean_write_action_rate": float(np.mean(write_action_rates)),
        "eval_mean_applied_write_rate": float(np.mean(applied_write_rates)),
        "eval_mean_applied_nonzero_write_rate": float(
            np.mean(applied_nonzero_write_rates)
        ),
        "eval_mean_write_overwrite_rate": float(np.mean(overwrite_rates)),
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
        agent_identity_types=getattr(args, "agent_identity_types", None),
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


def _first_env_value(value: Any) -> float:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return float(array[0])


def _first_env_sum(value: Any) -> float:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return float(np.sum(array[0]))


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
        target_agent_identity_types=getattr(args, "agent_identity_types", None),
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


def _evaluation_actions_for_mode(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    central_obs: jax.Array,
    key: jax.Array,
    *,
    action_mode: str,
    move_temperature: float = 1.0,
    write_temperature: float = 1.0,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
) -> jax.Array:
    if action_mode in {"deterministic", "sampled"}:
        actions, _, _, _ = get_action_and_value(
            params,
            actor_obs,
            central_obs,
            key,
            deterministic=action_mode == "deterministic",
            critic_architecture=critic_architecture,
            critic_num_ants=critic_num_ants,
            critic_obs_height=critic_obs_height,
            critic_obs_width=critic_obs_width,
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
        agent_identity_types=getattr(args, "agent_identity_types", None),
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    return central_obs.shape[-1], actor_obs.shape[-1]


def _make_eval_env(args: argparse.Namespace) -> JaxMappoEnv:
    return make_jax_mappo_env(args)


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
    values.setdefault("random_ant_spawn", False)
    values.setdefault("random_ant_spawn_radius", None)
    if shuffle_positions:
        values["random_food"] = True
        values["random_hub"] = True
    return argparse.Namespace(**values)
