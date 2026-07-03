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
    critic_forward_kwargs_from_args,
    evaluate_actions,
    flatten_agent_actions,
    food_observation_scale,
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


def _food_source_positions(
    food: jax.Array,
    *,
    source_count: int,
    width: int,
) -> jax.Array:
    batch_size = food.shape[0]
    if source_count <= 0:
        return jnp.zeros((batch_size, 0, 2), dtype=jnp.int32)

    flat_food = food.reshape((batch_size, -1)).astype(jnp.float32)
    selected_values, selected_flat = jax.lax.top_k(flat_food, source_count)
    positions = jnp.stack(
        [selected_flat % int(width), selected_flat // int(width)],
        axis=-1,
    ).astype(jnp.int32)
    valid_positions = selected_values > 0.0
    return jnp.where(valid_positions[..., None], positions, -jnp.ones_like(positions))


def _write_action_diagnostics(
    args: argparse.Namespace,
    actions: jax.Array,
    previous_carrying: jax.Array,
    infos: Any,
) -> dict[str, jax.Array]:
    write_values = actions[..., 1].astype(jnp.float32)
    if bool(getattr(args, "per_ant_write_channels", False)):
        bit_indices = jnp.mod(
            jnp.arange(actions.shape[-2], dtype=jnp.uint32),
            jnp.asarray(int(getattr(args, "write_bits", 1)), dtype=jnp.uint32),
        )
        write_values = jnp.asarray(
            actions[..., 1].astype(jnp.uint32)
            & jnp.left_shift(
                jnp.asarray(1, dtype=jnp.uint32),
                bit_indices,
            ),
            dtype=jnp.float32,
        )
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
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    critic_kwargs = critic_forward_kwargs_from_args(args)

    def scan_step(
        carry: tuple[JaxAntState, JaxObs, jax.Array],
        _: Any,
    ) -> tuple[tuple[JaxAntState, JaxObs, jax.Array], Transition]:
        current_states, current_obs, current_key = carry
        action_key, both_mix_key, move_mix_key, reset_key, next_key = jax.random.split(
            current_key,
            5,
        )
        central_obs = build_central_observations(
            current_obs,
            food_scale=food_scale,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        actor_obs_source = _actor_observation_source(args, current_obs)
        actor_obs = build_actor_observations(
            actor_obs_source,
            food_scale=food_scale,
            actor_vision_radius=args.actor_vision_radius,
            write_bits=args.write_bits,
            agent_identity_types=getattr(args, "agent_identity_types", None),
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        deterministic_fraction = float(getattr(args, "deterministic_rollout_fraction", 0.0))
        if bool(getattr(args, "deterministic_rollout", False)):
            deterministic_fraction = 1.0
        deterministic_move_fraction = float(
            getattr(args, "deterministic_move_rollout_fraction", 0.0)
        )
        training_rollout_temperature = float(
            getattr(args, "training_rollout_temperature", 1.0)
        )
        if deterministic_fraction <= 0.0 and deterministic_move_fraction <= 0.0:
            actions, logprobs, _, values = get_action_and_value(
                params,
                actor_obs,
                central_obs,
                action_key,
                deterministic=False,
                policy_temperature=training_rollout_temperature,
                **critic_kwargs,
            )
        elif deterministic_fraction >= 1.0:
            actions, logprobs, _, values = get_action_and_value(
                params,
                actor_obs,
                central_obs,
                action_key,
                deterministic=True,
                policy_temperature=training_rollout_temperature,
                **critic_kwargs,
            )
        else:
            sampled_actions, sampled_logprobs, _, values = get_action_and_value(
                params,
                actor_obs,
                central_obs,
                action_key,
                deterministic=False,
                policy_temperature=training_rollout_temperature,
                **critic_kwargs,
            )
            greedy_actions, greedy_logprobs, _, _ = get_action_and_value(
                params,
                actor_obs,
                central_obs,
                action_key,
                deterministic=True,
                policy_temperature=training_rollout_temperature,
                **critic_kwargs,
            )
            actions = sampled_actions
            if deterministic_move_fraction > 0.0:
                if deterministic_move_fraction >= 1.0:
                    move_greedy_mask = jnp.ones(sampled_actions.shape[:-1], dtype=jnp.bool_)
                else:
                    move_greedy_mask = jax.random.bernoulli(
                        move_mix_key,
                        p=deterministic_move_fraction,
                        shape=sampled_actions.shape[:-1],
                    )
                actions = actions.at[..., 0].set(
                    jnp.where(
                        move_greedy_mask,
                        greedy_actions[..., 0],
                        sampled_actions[..., 0],
                    )
                )
            greedy_mask = jax.random.bernoulli(
                both_mix_key,
                p=deterministic_fraction,
                shape=sampled_actions.shape[:-1],
            )
            actions = jnp.where(greedy_mask[..., None], greedy_actions, actions)
            if deterministic_move_fraction > 0.0:
                logprobs, _, values = evaluate_actions(
                    params,
                    actor_obs,
                    central_obs,
                    actions,
                    policy_temperature=training_rollout_temperature,
                    **critic_kwargs,
                )
            else:
                logprobs = jnp.where(greedy_mask, greedy_logprobs, sampled_logprobs)
        actions_for_env = _executed_actions(args, actions)
        next_states, next_obs, env_rewards, terminated, truncated, infos = jax.vmap(env.step)(
            current_states,
            flatten_agent_actions(actions_for_env),
        )
        dones = jnp.logical_or(terminated, truncated)
        next_central_obs = build_central_observations(
            next_obs,
            food_scale=food_scale,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        next_values = get_value(params, next_central_obs, **critic_kwargs)
        newly_visited_cells = getattr(
            infos,
            "newly_visited_cells",
            jnp.zeros_like(env_rewards, dtype=jnp.int32),
        ).astype(jnp.float32)
        visited_cell_count = getattr(
            infos,
            "visited_cell_count",
            jnp.zeros_like(env_rewards, dtype=jnp.int32),
        ).astype(jnp.float32)
        newly_viewed_cells = getattr(
            infos,
            "newly_viewed_cells",
            jnp.zeros_like(env_rewards, dtype=jnp.int32),
        ).astype(jnp.float32)
        viewed_cell_count = getattr(
            infos,
            "viewed_cell_count",
            visited_cell_count,
        ).astype(jnp.float32)
        visible_border_cells = getattr(
            infos,
            "visible_border_cells",
            jnp.zeros_like(env_rewards, dtype=jnp.int32),
        ).astype(jnp.float32)
        moat_width = jnp.asarray(
            max(int(getattr(args, "border_moat_width", 0)), 0),
            dtype=jnp.float32,
        )
        positions = next_obs["ants_pos"].astype(jnp.float32)
        active_grid_size = next_obs.get("active_grid_size")
        if active_grid_size is None:
            active_size = jnp.broadcast_to(
                jnp.asarray([env.width, env.height], dtype=jnp.float32),
                (next_obs["ants_pos"].shape[0], 2),
            )
        else:
            active_size = active_grid_size.astype(jnp.float32).reshape(
                (next_obs["ants_pos"].shape[0], 2)
            )
        distance_to_border = jnp.minimum(
            jnp.minimum(positions[..., 0], positions[..., 1]),
            jnp.minimum(
                active_size[:, None, 0] - 1.0 - positions[..., 0],
                active_size[:, None, 1] - 1.0 - positions[..., 1],
            ),
        )
        border_moat_cost = jnp.sum(
            jnp.maximum(moat_width - distance_to_border, 0.0),
            axis=-1,
        )
        if "obstacles" in next_obs:
            coverage_denominator = jnp.maximum(
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.sum(
                    next_obs["obstacles"].astype(jnp.int32) == 0,
                    axis=(-2, -1),
                ).astype(jnp.float32),
            )
        else:
            coverage_denominator = float(getattr(env, "open_cell_count", env.height * env.width))
        visited_cell_fraction = visited_cell_count / coverage_denominator
        viewed_cell_fraction = viewed_cell_count / coverage_denominator
        if str(getattr(args, "reward_mode", "forage")) == "explore":
            rewards = newly_visited_cells
        else:
            rewards = compute_forage_curriculum_rewards(
                previous_obs=current_obs,
                next_obs=next_obs,
                env_rewards=env_rewards,
                actions=actions_for_env,
                pickup_bonus=args.pickup_bonus,
                distance_bonus=args.distance_bonus,
                distance_progress_normalizer=getattr(
                    args,
                    "distance_progress_normalizer",
                    "map",
                ),
                carrying_hub_distance_bonus=getattr(args, "carrying_hub_distance_bonus", 0.0),
                newly_visited_cells=newly_visited_cells,
                visited_cell_fraction=visited_cell_fraction,
                visit_reward_scale=args.visit_reward_scale,
                visit_reward_decay=args.visit_reward_decay,
                newly_viewed_cells=newly_viewed_cells,
                viewed_cell_fraction=viewed_cell_fraction,
                view_reward_scale=getattr(args, "view_reward_scale", 0.0),
                view_reward_decay=getattr(args, "view_reward_decay", 1.0),
                visible_border_cells=visible_border_cells,
                border_view_penalty=getattr(args, "border_view_penalty", 0.0),
                border_moat_cost=border_moat_cost,
                border_moat_penalty=getattr(args, "border_moat_penalty", 0.0),
                stage_completion_events=getattr(
                    infos,
                    "advanced_stage",
                    jnp.zeros_like(env_rewards, dtype=jnp.float32),
                ),
                stage_completion_bonus=args.stage_completion_bonus,
                delivery_byte_trail_bonus=args.delivery_byte_trail_bonus,
                delivery_byte_trail_target_tiles=args.delivery_byte_trail_target_tiles,
                byte_follow_bonus=args.byte_follow_bonus,
                carrying_byte_write_bonus=args.carrying_byte_write_bonus,
                write_while_moving=args.write_while_moving,
                per_ant_write_channels=bool(getattr(args, "per_ant_write_channels", False)),
                write_bits=args.write_bits,
            )
            rewards -= compute_write_bit_penalties(
                actions_for_env,
                write_bits=args.write_bits,
                base_penalty=args.write_bit_penalty,
                decay=args.write_bit_penalty_decay,
                write_while_moving=args.write_while_moving,
                per_ant_write_channels=bool(getattr(args, "per_ant_write_channels", False)),
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
        reset_source_count = int(
            getattr(env, "source_count", getattr(args, "food_sources", 0))
        )

        def reset_done_envs(_: None) -> tuple[JaxAntState, JaxObs, jax.Array, jax.Array]:
            reset_states, reset_obs = reset_batch(
                args=args,
                env=env,
                key=reset_key,
                previous_obs=next_obs,
                previous_food=next_states.initial_food,
            )
            return (
                _select_reset_values(next_states, reset_states, dones),
                _select_reset_values(next_obs, reset_obs, dones),
                reset_obs["hub_pos"],
                _food_source_positions(
                    reset_obs["food"],
                    source_count=reset_source_count,
                    width=env.width,
                ),
            )

        def keep_current_envs(_: None) -> tuple[JaxAntState, JaxObs, jax.Array, jax.Array]:
            return (
                next_states,
                next_obs,
                jnp.zeros_like(next_obs["hub_pos"]),
                jnp.zeros(
                    (next_obs["food"].shape[0], reset_source_count, 2),
                    dtype=jnp.int32,
                ),
            )

        carry_states, carry_obs, reset_hub_pos, reset_food_positions = jax.lax.cond(
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
            newly_visited_cells=newly_visited_cells,
            visited_cell_count=visited_cell_count,
            visited_cell_fraction=visited_cell_fraction,
            newly_viewed_cells=newly_viewed_cells,
            viewed_cell_count=viewed_cell_count,
            viewed_cell_fraction=viewed_cell_fraction,
            visible_border_cells=visible_border_cells,
            border_moat_cost=border_moat_cost,
            nonzero_byte_tiles=nonzero_byte_tiles,
            nonzero_byte_fraction=nonzero_byte_fraction,
            reset_hub_pos=reset_hub_pos,
            reset_food_positions=reset_food_positions,
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
        + (
            jnp.zeros_like(transitions.rewards)
            if str(getattr(args, "reward_mode", "forage")) == "explore"
            else compute_write_bit_entropy_bonus(
                transitions.actions,
                write_bits=args.write_bits,
                entropy_scale=args.write_bit_entropy_bonus,
                write_while_moving=args.write_while_moving,
                per_ant_write_channels=bool(getattr(args, "per_ant_write_channels", False)),
            )
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
        newly_visited_cells=transitions.newly_visited_cells,
        visited_cell_count=transitions.visited_cell_count,
        visited_cell_fraction=transitions.visited_cell_fraction,
        newly_viewed_cells=transitions.newly_viewed_cells,
        viewed_cell_count=transitions.viewed_cell_count,
        viewed_cell_fraction=transitions.viewed_cell_fraction,
        visible_border_cells=transitions.visible_border_cells,
        border_moat_cost=transitions.border_moat_cost,
        nonzero_byte_tiles=transitions.nonzero_byte_tiles,
        nonzero_byte_fraction=transitions.nonzero_byte_fraction,
        applied_nonzero_write_actions=transitions.applied_nonzero_write_actions,
        empty_nonzero_write_actions=transitions.empty_nonzero_write_actions,
        carrying_nonzero_write_actions=transitions.carrying_nonzero_write_actions,
        empty_write_action_slots=transitions.empty_write_action_slots,
        carrying_write_action_slots=transitions.carrying_write_action_slots,
        write_attempts=transitions.write_attempts,
        overwrite_events=transitions.overwrite_events,
        reset_hub_pos=transitions.reset_hub_pos,
        reset_food_positions=transitions.reset_food_positions,
    )
    return final_states, final_obs, rollout
