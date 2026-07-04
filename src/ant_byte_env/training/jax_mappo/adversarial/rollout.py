"""Rollout collection for frozen-opponent adversarial MAPPO."""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp

from ant_byte_env.training.jax_mappo.adversarial.actions import actions_from_logits
from ant_byte_env.training.jax_mappo.adversarial.env import (
    JaxAdversarialAntByteEnv,
    JaxAdversarialAntState,
    reset_batch,
)
from ant_byte_env.training.jax_mappo.adversarial.observations import (
    build_team_actor_observations,
    build_team_central_observations,
)
from ant_byte_env.training.jax_mappo.adversarial.types import (
    AdversarialRollout,
    AdversarialTransition,
)
from ant_byte_env.training.jax_mappo.observations import (
    flatten_agent_actions,
    food_observation_scale,
)
from ant_byte_env.training.jax_mappo.models import (
    critic_forward_kwargs_from_args,
    get_action_logits,
)
from ant_byte_env.training.jax_mappo.policy import get_action_and_value
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams


def _select_reset_values(current: Any, reset: Any, dones: jax.Array) -> Any:
    done_mask = dones.astype(jnp.bool_)

    def select_leaf(current_leaf: jax.Array, reset_leaf: jax.Array) -> jax.Array:
        broadcast_shape = done_mask.shape + (1,) * (current_leaf.ndim - done_mask.ndim)
        return jnp.where(done_mask.reshape(broadcast_shape), reset_leaf, current_leaf)

    return jax.tree_util.tree_map(select_leaf, current, reset)


def compose_team_actions(
    learner_actions: jax.Array,
    opponent_actions: jax.Array,
    *,
    learner_team: int,
) -> jax.Array:
    if int(learner_team) == 0:
        return jnp.concatenate([learner_actions, opponent_actions], axis=-2)
    return jnp.concatenate([opponent_actions, learner_actions], axis=-2)


def _policy_action_mode(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    key: jax.Array,
    *,
    action_mode: str,
    policy_temperature: float,
) -> jax.Array:
    move_logits, write_logits = get_action_logits(params, actor_obs)
    return actions_from_logits(
        move_logits,
        write_logits,
        key,
        action_mode=action_mode,
        move_temperature=policy_temperature,
        write_temperature=policy_temperature,
    )


def collect_rollout(
    *,
    args: argparse.Namespace,
    env: JaxAdversarialAntByteEnv,
    learner_params: JaxMAPPOParams,
    opponent_params: JaxMAPPOParams,
    states: JaxAdversarialAntState,
    obs: dict[str, jax.Array],
    key: jax.Array,
) -> tuple[JaxAdversarialAntState, dict[str, jax.Array], AdversarialRollout]:
    learner_team = int(args.learner_team)
    opponent_team = 1 - learner_team
    num_ants_per_team = int(args.num_ants_per_team)
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    critic_kwargs = critic_forward_kwargs_from_args(args)

    def scan_step(
        carry: tuple[JaxAdversarialAntState, dict[str, jax.Array], jax.Array],
        _: Any,
    ) -> tuple[tuple[JaxAdversarialAntState, dict[str, jax.Array], jax.Array], AdversarialTransition]:
        current_states, current_obs, current_key = carry
        learner_key, opponent_key, reset_key, next_key = jax.random.split(current_key, 4)
        learner_actor_obs = build_team_actor_observations(
            current_obs,
            team=learner_team,
            num_ants_per_team=num_ants_per_team,
            food_scale=food_scale,
            actor_vision_radius=args.actor_vision_radius,
            write_bits=args.write_bits,
        )
        learner_central_obs = build_team_central_observations(
            current_obs,
            team=learner_team,
            num_ants_per_team=num_ants_per_team,
            food_scale=food_scale,
            write_bits=args.write_bits,
        )
        opponent_actor_obs = build_team_actor_observations(
            current_obs,
            team=opponent_team,
            num_ants_per_team=num_ants_per_team,
            food_scale=food_scale,
            actor_vision_radius=args.actor_vision_radius,
            write_bits=args.write_bits,
        )
        learner_actions, learner_logprobs, _, values = get_action_and_value(
            learner_params,
            learner_actor_obs,
            learner_central_obs,
            learner_key,
            deterministic=False,
            policy_temperature=float(args.training_rollout_temperature),
            **critic_kwargs,
        )
        opponent_actions = _policy_action_mode(
            opponent_params,
            opponent_actor_obs,
            opponent_key,
            action_mode=args.opponent_action_mode,
            policy_temperature=float(args.training_rollout_temperature),
        )
        joint_actions = compose_team_actions(
            learner_actions,
            opponent_actions,
            learner_team=learner_team,
        )
        next_states, next_obs, env_rewards, terminated, truncated, infos = jax.vmap(env.step)(
            current_states,
            flatten_agent_actions(joint_actions),
        )
        dones = jnp.logical_or(terminated, truncated)
        next_learner_central_obs = build_team_central_observations(
            next_obs,
            team=learner_team,
            num_ants_per_team=num_ants_per_team,
            food_scale=food_scale,
            write_bits=args.write_bits,
        )
        _, _, _, next_values = get_action_and_value(
            learner_params,
            build_team_actor_observations(
                next_obs,
                team=learner_team,
                num_ants_per_team=num_ants_per_team,
                food_scale=food_scale,
                actor_vision_radius=args.actor_vision_radius,
                write_bits=args.write_bits,
            ),
            next_learner_central_obs,
            learner_key,
            deterministic=True,
            policy_temperature=float(args.training_rollout_temperature),
            **critic_kwargs,
        )
        rewards = env_rewards[:, learner_team]

        def reset_done_envs(_: None):
            reset_states, reset_obs = reset_batch(args=args, env=env, key=reset_key)
            return (
                _select_reset_values(next_states, reset_states, dones),
                _select_reset_values(next_obs, reset_obs, dones),
            )

        def keep_current_envs(_: None):
            return next_states, next_obs

        carry_states, carry_obs = jax.lax.cond(
            jnp.any(dones),
            reset_done_envs,
            keep_current_envs,
            operand=None,
        )
        transition = AdversarialTransition(
            actor_obs=learner_actor_obs,
            central_obs=learner_central_obs,
            actions=learner_actions,
            joint_actions=joint_actions,
            logprobs=learner_logprobs,
            rewards=rewards,
            dones=dones,
            terminations=terminated,
            values=values,
            next_values=next_values,
            env_rewards=env_rewards,
            pickup_events=infos.pickup_events,
            delivery_events=infos.delivery_events,
            delivered_food=infos.delivered_food,
            remaining_food=infos.remaining_food,
        )
        return (carry_states, carry_obs, next_key), transition

    (final_states, final_obs, _), transitions = jax.lax.scan(
        scan_step,
        (states, obs, key),
        None,
        length=int(args.num_steps),
    )
    rollout = AdversarialRollout(
        actor_obs=transitions.actor_obs,
        central_obs=transitions.central_obs,
        actions=transitions.actions,
        joint_actions=transitions.joint_actions,
        logprobs=transitions.logprobs,
        rewards=transitions.rewards,
        dones=transitions.dones,
        terminations=transitions.terminations,
        values=transitions.values,
        next_values=transitions.next_values,
        env_rewards=transitions.env_rewards,
        pickup_events=transitions.pickup_events,
        delivery_events=transitions.delivery_events,
        delivered_food=transitions.delivered_food,
        remaining_food=transitions.remaining_food,
    )
    return final_states, final_obs, rollout
