"""JAX rollout collection for MAPPO."""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp

from ant_byte_env import ACTION_STAY
from ant_byte_env.jax_env import JaxAntByteForagingEnv, JaxAntState, JaxObs
from ant_byte_env.training.jax_mappo.core import (
    JaxMAPPOParams,
    Rollout,
    Transition,
    build_actor_observations,
    build_central_observations,
    compute_forage_curriculum_rewards,
    compute_terminal_write_entropy_bonus,
    compute_write_bit_entropy_bonus,
    compute_write_bit_penalties,
    flatten_agent_actions,
    get_action_and_value,
    get_value,
)
from ant_byte_env.training.jax_mappo.curriculum import reset_batch


def _select_reset_values(current: Any, reset: Any, dones: jax.Array) -> Any:
    done_mask = dones.astype(jnp.bool_)

    def select_leaf(current_leaf: jax.Array, reset_leaf: jax.Array) -> jax.Array:
        broadcast_shape = done_mask.shape + (1,) * (current_leaf.ndim - done_mask.ndim)
        return jnp.where(done_mask.reshape(broadcast_shape), reset_leaf, current_leaf)

    return jax.tree_util.tree_map(select_leaf, current, reset)


def _actor_observation_source(args: argparse.Namespace, obs: JaxObs) -> JaxObs:
    if not bool(getattr(args, "actor_byte_read_ablation", False)):
        return obs
    return {**obs, "bytes": jnp.zeros_like(obs["bytes"])}


def _executed_actions(args: argparse.Namespace, actions: jax.Array) -> jax.Array:
    if not bool(getattr(args, "write_action_ablation", False)):
        return actions
    return actions.at[..., 1].set(0)


def _write_action_diagnostics(
    args: argparse.Namespace,
    actions: jax.Array,
    previous_carrying: jax.Array,
    infos: Any,
) -> dict[str, jax.Array]:
    write_values = actions[..., 1].astype(jnp.float32)
    if bool(getattr(args, "write_while_moving", False)):
        applied_write_values = write_values
    else:
        applied_write_values = jnp.where(actions[..., 0] == ACTION_STAY, write_values, 0.0)
    nonzero_applied = applied_write_values > 0.0
    carrying_mask = previous_carrying.astype(jnp.bool_)
    empty_mask = jnp.logical_not(carrying_mask)
    zero_events = jnp.zeros(actions.shape[:1], dtype=jnp.float32)
    return {
        "applied_nonzero_write_actions": jnp.sum(
            nonzero_applied.astype(jnp.float32),
            axis=-1,
        ),
        "empty_nonzero_write_actions": jnp.sum(
            jnp.logical_and(nonzero_applied, empty_mask).astype(jnp.float32),
            axis=-1,
        ),
        "carrying_nonzero_write_actions": jnp.sum(
            jnp.logical_and(nonzero_applied, carrying_mask).astype(jnp.float32),
            axis=-1,
        ),
        "empty_write_action_slots": jnp.sum(empty_mask.astype(jnp.float32), axis=-1),
        "carrying_write_action_slots": jnp.sum(carrying_mask.astype(jnp.float32), axis=-1),
        "write_attempts": getattr(infos, "num_writes", zero_events).astype(jnp.float32),
        "overwrite_events": getattr(infos, "num_overwrites", zero_events).astype(jnp.float32),
    }


def collect_rollout(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    params: JaxMAPPOParams,
    states: JaxAntState,
    obs: JaxObs,
    key: jax.Array,
) -> tuple[JaxAntState, JaxObs, Rollout]:
    def scan_step(
        carry: tuple[JaxAntState, JaxObs, jax.Array],
        _: Any,
    ) -> tuple[tuple[JaxAntState, JaxObs, jax.Array], Transition]:
        current_states, current_obs, current_key = carry
        action_key, reset_key, next_key = jax.random.split(current_key, 3)
        central_obs = build_central_observations(
            current_obs,
            food_scale=args.food_count,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        actor_obs_source = _actor_observation_source(args, current_obs)
        actor_obs = build_actor_observations(
            actor_obs_source,
            food_scale=args.food_count,
            actor_vision_radius=args.actor_vision_radius,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        actions, logprobs, _, values = get_action_and_value(
            params,
            actor_obs,
            central_obs,
            action_key,
            deterministic=False,
        )
        actions_for_env = _executed_actions(args, actions)
        next_states, next_obs, env_rewards, terminated, truncated, infos = jax.vmap(env.step)(
            current_states,
            flatten_agent_actions(actions_for_env),
        )
        dones = jnp.logical_or(terminated, truncated)
        next_central_obs = build_central_observations(
            next_obs,
            food_scale=args.food_count,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        next_values = get_value(params, next_central_obs)
        rewards = compute_forage_curriculum_rewards(
            previous_obs=current_obs,
            next_obs=next_obs,
            env_rewards=env_rewards,
            pickup_bonus=args.pickup_bonus,
            distance_bonus=args.distance_bonus,
            stage_completion_events=getattr(
                infos,
                "advanced_stage",
                jnp.zeros_like(env_rewards, dtype=jnp.float32),
            ),
            stage_completion_bonus=args.stage_completion_bonus,
            delivery_byte_trail_bonus=args.delivery_byte_trail_bonus,
            delivery_byte_trail_target_tiles=args.delivery_byte_trail_target_tiles,
        )
        rewards -= compute_write_bit_penalties(
            actions,
            write_bits=args.write_bits,
            base_penalty=args.write_bit_penalty,
            decay=args.write_bit_penalty_decay,
            write_while_moving=args.write_while_moving,
        )
        rewards += compute_terminal_write_entropy_bonus(
            next_obs,
            dones,
            write_bits=args.write_bits,
            entropy_scale=args.write_entropy_bonus,
            max_bonus=args.write_entropy_bonus_cap,
        )
        previous_carrying = current_obs["ants_carrying"].astype(jnp.bool_)
        next_carrying = next_obs["ants_carrying"].astype(jnp.bool_)
        pickup_events = jnp.sum(
            jnp.logical_and(jnp.logical_not(previous_carrying), next_carrying).astype(
                jnp.float32
            ),
            axis=-1,
        )
        delivery_events = jnp.sum(
            jnp.logical_and(previous_carrying, jnp.logical_not(next_carrying)).astype(
                jnp.float32
            ),
            axis=-1,
        )
        carrying_ants = jnp.sum(next_carrying.astype(jnp.float32), axis=-1)
        remaining_food = jnp.sum(next_obs["food"].astype(jnp.float32), axis=(-2, -1))
        active_grid_size = next_obs.get("active_grid_size")
        active_size = (
            active_grid_size[..., 0].astype(jnp.float32)
            if active_grid_size is not None
            else jnp.zeros_like(env_rewards, dtype=jnp.float32)
        )
        stage_advances = getattr(
            infos,
            "advanced_stage",
            jnp.zeros_like(env_rewards, dtype=jnp.float32),
        ).astype(jnp.float32)
        stage_delivered_food = getattr(
            next_states,
            "stage_delivered_food",
            jnp.zeros_like(env_rewards, dtype=jnp.float32),
        ).astype(jnp.float32)
        nonzero_byte_tiles = jnp.sum(
            (next_obs["bytes"] > 0).astype(jnp.float32),
            axis=(-2, -1),
        )
        nonzero_byte_fraction = nonzero_byte_tiles / float(env.height * env.width)
        write_diagnostics = _write_action_diagnostics(
            args,
            actions_for_env,
            previous_carrying,
            infos,
        )

        def reset_done_envs(_: None) -> tuple[JaxAntState, JaxObs]:
            reset_states, reset_obs = reset_batch(args=args, env=env, key=reset_key)
            return (
                _select_reset_values(next_states, reset_states, dones),
                _select_reset_values(next_obs, reset_obs, dones),
            )

        def keep_current_envs(_: None) -> tuple[JaxAntState, JaxObs]:
            return next_states, next_obs

        carry_states, carry_obs = jax.lax.cond(
            jnp.any(dones),
            reset_done_envs,
            keep_current_envs,
            operand=None,
        )
        transition = Transition(
            actor_obs=actor_obs,
            central_obs=central_obs,
            actions=actions_for_env,
            logprobs=logprobs,
            rewards=rewards,
            dones=dones,
            terminations=terminated,
            truncations=truncated,
            values=values,
            next_values=next_values,
            env_rewards=env_rewards,
            pickup_events=pickup_events,
            delivery_events=delivery_events,
            carrying_ants=carrying_ants,
            remaining_food=remaining_food,
            active_size=active_size,
            stage_advances=stage_advances,
            stage_delivered_food=stage_delivered_food,
            nonzero_byte_tiles=nonzero_byte_tiles,
            nonzero_byte_fraction=nonzero_byte_fraction,
            **write_diagnostics,
        )
        return (carry_states, carry_obs, next_key), transition

    (final_states, final_obs, _), transitions = jax.lax.scan(
        scan_step,
        (states, obs, key),
        None,
        length=args.num_steps,
    )
    rollout = Rollout(
        actor_obs=transitions.actor_obs,
        central_obs=transitions.central_obs,
        actions=transitions.actions,
        logprobs=transitions.logprobs,
        rewards=transitions.rewards
        + compute_write_bit_entropy_bonus(
            transitions.actions,
            write_bits=args.write_bits,
            entropy_scale=args.write_bit_entropy_bonus,
            write_while_moving=args.write_while_moving,
        ),
        dones=transitions.dones,
        terminations=transitions.terminations,
        truncations=transitions.truncations,
        values=transitions.values,
        next_values=transitions.next_values,
        env_rewards=transitions.env_rewards,
        pickup_events=transitions.pickup_events,
        delivery_events=transitions.delivery_events,
        carrying_ants=transitions.carrying_ants,
        remaining_food=transitions.remaining_food,
        active_size=transitions.active_size,
        stage_advances=transitions.stage_advances,
        stage_delivered_food=transitions.stage_delivered_food,
        nonzero_byte_tiles=transitions.nonzero_byte_tiles,
        nonzero_byte_fraction=transitions.nonzero_byte_fraction,
        applied_nonzero_write_actions=transitions.applied_nonzero_write_actions,
        empty_nonzero_write_actions=transitions.empty_nonzero_write_actions,
        carrying_nonzero_write_actions=transitions.carrying_nonzero_write_actions,
        empty_write_action_slots=transitions.empty_write_action_slots,
        carrying_write_action_slots=transitions.carrying_write_action_slots,
        write_attempts=transitions.write_attempts,
        overwrite_events=transitions.overwrite_events,
    )
    return final_states, final_obs, rollout
