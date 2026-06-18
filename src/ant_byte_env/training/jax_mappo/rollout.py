"""JAX rollout collection for MAPPO."""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp

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
        actor_obs = build_actor_observations(
            current_obs,
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
        next_states, next_obs, env_rewards, terminated, truncated, _ = jax.vmap(env.step)(
            current_states,
            flatten_agent_actions(actions),
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
        nonzero_byte_tiles = jnp.sum(
            (next_obs["bytes"] > 0).astype(jnp.float32),
            axis=(-2, -1),
        )
        nonzero_byte_fraction = nonzero_byte_tiles / float(env.height * env.width)

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
            actions=actions,
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
            nonzero_byte_tiles=nonzero_byte_tiles,
            nonzero_byte_fraction=nonzero_byte_fraction,
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
        nonzero_byte_tiles=transitions.nonzero_byte_tiles,
        nonzero_byte_fraction=transitions.nonzero_byte_fraction,
    )
    return final_states, final_obs, rollout
