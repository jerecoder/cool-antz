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
    flatten_agent_actions,
    get_action_and_value,
    get_value,
)


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
        action_key, next_key = jax.random.split(current_key)
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
        )
        next_states, next_obs, env_rewards, terminated, truncated, _ = jax.vmap(env.step)(
            current_states,
            flatten_agent_actions(actions),
        )
        dones = jnp.logical_or(terminated, truncated)
        rewards = compute_forage_curriculum_rewards(
            previous_obs=current_obs,
            next_obs=next_obs,
            env_rewards=env_rewards,
            pickup_bonus=args.pickup_bonus,
            distance_bonus=args.distance_bonus,
        )
        transition = Transition(
            actor_obs=actor_obs,
            central_obs=central_obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=dones,
            values=values,
            env_rewards=env_rewards,
        )
        return (next_states, next_obs, next_key), transition

    (final_states, final_obs, _), transitions = jax.lax.scan(
        scan_step,
        (states, obs, key),
        None,
        length=args.num_steps,
    )
    next_central_obs = build_central_observations(
        final_obs,
        food_scale=args.food_count,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    rollout = Rollout(
        actor_obs=transitions.actor_obs,
        central_obs=transitions.central_obs,
        actions=transitions.actions,
        logprobs=transitions.logprobs,
        rewards=transitions.rewards,
        dones=transitions.dones,
        values=transitions.values,
        env_rewards=transitions.env_rewards,
        next_value=get_value(params, next_central_obs),
    )
    return final_states, final_obs, rollout
