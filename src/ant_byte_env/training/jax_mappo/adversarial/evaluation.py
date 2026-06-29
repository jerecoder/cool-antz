"""Diagnostic evaluation matrix for adversarial MAPPO."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env import MOVEMENT_ACTION_COUNT, write_value_count
from ant_byte_env.training.jax_mappo.adversarial.actions import actions_from_logits
from ant_byte_env.training.jax_mappo.adversarial.env import (
    JaxAdversarialAntByteEnv,
    reset_batch,
)
from ant_byte_env.training.jax_mappo.adversarial.observations import (
    build_team_actor_observations,
)
from ant_byte_env.training.jax_mappo.adversarial.rollout import compose_team_actions
from ant_byte_env.training.jax_mappo.models import get_action_logits
from ant_byte_env.training.jax_mappo.observations import (
    flatten_agent_actions,
    food_observation_scale,
)
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams

PolicyKind = Literal["model", "random"]
EvaluationProgressCallback = Callable[[str, int, int, dict[str, float]], None]


def evaluate_matrix(
    *,
    params: JaxMAPPOParams,
    opponent_params: JaxMAPPOParams,
    args: argparse.Namespace,
    env: JaxAdversarialAntByteEnv | None = None,
    progress_callback: EvaluationProgressCallback | None = None,
    progress_step_interval: int | None = None,
    fixed_hub_positions: Sequence[Sequence[int]] | None = None,
    fixed_food_positions: Sequence[Sequence[int]] | None = None,
) -> dict[str, float]:
    if int(args.eval_episodes) <= 0:
        return {}
    eval_env = env if env is not None else _make_eval_env(args)
    frozen_frozen = evaluate_matchup(
        learner_params=opponent_params,
        opponent_params=opponent_params,
        learner_kind="model",
        opponent_kind="model",
        learner_team=0,
        args=args,
        env=eval_env,
        seed_offset=100_000,
        progress_name="frozen_vs_frozen",
        progress_callback=progress_callback,
        progress_step_interval=progress_step_interval,
        fixed_hub_positions=fixed_hub_positions,
        fixed_food_positions=fixed_food_positions,
    )
    learner_frozen = evaluate_matchup(
        learner_params=params,
        opponent_params=opponent_params,
        learner_kind="model",
        opponent_kind="model",
        learner_team=0,
        args=args,
        env=eval_env,
        seed_offset=200_000,
        progress_name="learner_vs_frozen",
        progress_callback=progress_callback,
        progress_step_interval=progress_step_interval,
        fixed_hub_positions=fixed_hub_positions,
        fixed_food_positions=fixed_food_positions,
    )
    frozen_learner = evaluate_matchup(
        learner_params=params,
        opponent_params=opponent_params,
        learner_kind="model",
        opponent_kind="model",
        learner_team=1,
        args=args,
        env=eval_env,
        seed_offset=300_000,
        progress_name="frozen_vs_learner",
        progress_callback=progress_callback,
        progress_step_interval=progress_step_interval,
        fixed_hub_positions=fixed_hub_positions,
        fixed_food_positions=fixed_food_positions,
    )
    random_frozen = evaluate_matchup(
        learner_params=None,
        opponent_params=opponent_params,
        learner_kind="random",
        opponent_kind="model",
        learner_team=0,
        args=args,
        env=eval_env,
        seed_offset=400_000,
        progress_name="random_vs_frozen",
        progress_callback=progress_callback,
        progress_step_interval=progress_step_interval,
        fixed_hub_positions=fixed_hub_positions,
        fixed_food_positions=fixed_food_positions,
    )
    learner_random = evaluate_matchup(
        learner_params=params,
        opponent_params=None,
        learner_kind="model",
        opponent_kind="random",
        learner_team=0,
        args=args,
        env=eval_env,
        seed_offset=500_000,
        progress_name="learner_vs_random",
        progress_callback=progress_callback,
        progress_step_interval=progress_step_interval,
        fixed_hub_positions=fixed_hub_positions,
        fixed_food_positions=fixed_food_positions,
    )
    metrics = {}
    for prefix, payload in (
        ("eval_frozen_vs_frozen", frozen_frozen),
        ("eval_learner_vs_frozen", learner_frozen),
        ("eval_frozen_vs_learner", frozen_learner),
        ("eval_random_vs_frozen", random_frozen),
        ("eval_learner_vs_random", learner_random),
    ):
        metrics.update({f"{prefix}_{key}": value for key, value in payload.items()})
    metrics["eval_side_swapped_score_gap"] = abs(
        learner_frozen["mean_delivery_difference"]
        - frozen_learner["mean_delivery_difference"]
    )
    metrics["eval_side_swapped_signed_score_gap"] = (
        learner_frozen["mean_delivery_difference"]
        - frozen_learner["mean_delivery_difference"]
    )
    return metrics


def evaluate_matchup(
    *,
    learner_params: JaxMAPPOParams | None,
    opponent_params: JaxMAPPOParams | None,
    learner_kind: PolicyKind,
    opponent_kind: PolicyKind,
    learner_team: int,
    args: argparse.Namespace,
    env: JaxAdversarialAntByteEnv,
    seed_offset: int,
    progress_name: str | None = None,
    progress_callback: EvaluationProgressCallback | None = None,
    progress_step_interval: int | None = None,
    fixed_hub_positions: Sequence[Sequence[int]] | None = None,
    fixed_food_positions: Sequence[Sequence[int]] | None = None,
) -> dict[str, float]:
    eval_args = argparse.Namespace(**{**vars(args), "num_envs": 1})
    key = jax.random.PRNGKey(int(eval_args.seed) + int(seed_offset))
    own_deliveries: list[float] = []
    opponent_deliveries: list[float] = []
    delivery_diffs: list[float] = []
    wins: list[float] = []
    own_pickups: list[float] = []
    opponent_pickups: list[float] = []
    episode_lengths: list[float] = []
    hub_pair_distances: list[float] = []
    food_midpoint_distances: list[float] = []
    opponent_team = 1 - int(learner_team)
    food_scale = food_observation_scale(
        food_count=eval_args.food_count,
        food_sources=getattr(eval_args, "food_sources", None),
    )
    step_interval = None if progress_step_interval is None else int(progress_step_interval)
    if step_interval is not None and step_interval <= 0:
        raise ValueError("progress_step_interval must be positive when provided.")
    fixed_hub_pos = _fixed_positions_array(fixed_hub_positions, shape=(2, 2))
    fixed_food_pos = _fixed_positions_array(fixed_food_positions, shape=(-1, 2))
    step_fn = jax.jit(
        lambda current_states, current_obs, action_key: _evaluation_step(
            env=env,
            args=eval_args,
            learner_params=learner_params,
            opponent_params=opponent_params,
            learner_kind=learner_kind,
            opponent_kind=opponent_kind,
            learner_team=int(learner_team),
            states=current_states,
            obs=current_obs,
            key=action_key,
            food_scale=food_scale,
        )
    )

    for episode_index in range(int(eval_args.eval_episodes)):
        key, reset_key = jax.random.split(key)
        states, obs = _reset_eval_batch(
            args=eval_args,
            env=env,
            key=reset_key,
            fixed_hub_positions=fixed_hub_pos,
            fixed_food_positions=fixed_food_pos,
        )
        placement_stats = _placement_stats(states)
        hub_pair_distances.append(placement_stats["hub_pair_distance"])
        food_midpoint_distances.append(placement_stats["food_midpoint_distance"])
        pickup_totals = np.zeros((2,), dtype=np.float32)
        episode_length = int(eval_args.max_steps)
        for step_index in range(int(eval_args.max_steps)):
            key, action_key = jax.random.split(key)
            states, obs, terminated, truncated, infos = step_fn(states, obs, action_key)
            pickup_totals += np.asarray(infos.pickup_events)[0].astype(np.float32)
            if (
                progress_callback is not None
                and progress_name is not None
                and step_interval is not None
                and (step_index + 1) % step_interval == 0
            ):
                progress_callback(
                    progress_name,
                    episode_index + 1,
                    int(eval_args.eval_episodes),
                    {
                        "event": "step",
                        "step": float(step_index + 1),
                        "max_steps": float(eval_args.max_steps),
                    },
                )
            if bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0]):
                episode_length = step_index + 1
                break
        delivered = np.asarray(states.delivered_food)[0].astype(np.float32)
        own = float(delivered[int(learner_team)])
        opponent = float(delivered[opponent_team])
        diff = own - opponent
        own_deliveries.append(own)
        opponent_deliveries.append(opponent)
        delivery_diffs.append(diff)
        wins.append(float(diff > 0.0))
        own_pickups.append(float(pickup_totals[int(learner_team)]))
        opponent_pickups.append(float(pickup_totals[opponent_team]))
        episode_lengths.append(float(episode_length))
        if progress_callback is not None and progress_name is not None:
            progress_callback(
                progress_name,
                episode_index + 1,
                int(eval_args.eval_episodes),
                {
                    "event": "episode",
                    "step": float(eval_args.max_steps),
                    "max_steps": float(eval_args.max_steps),
                    "own_deliveries": own,
                    "opponent_deliveries": opponent,
                    "delivery_difference": diff,
                    "episode_length": float(episode_length),
                },
            )
    return {
        "mean_own_deliveries": float(np.mean(own_deliveries)),
        "mean_opponent_deliveries": float(np.mean(opponent_deliveries)),
        "mean_delivery_difference": float(np.mean(delivery_diffs)),
        "win_rate": float(np.mean(wins)),
        "mean_own_pickups": float(np.mean(own_pickups)),
        "mean_opponent_pickups": float(np.mean(opponent_pickups)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_hub_pair_distance": float(np.mean(hub_pair_distances)),
        "mean_food_midpoint_distance": float(np.mean(food_midpoint_distances)),
    }


def _fixed_positions_array(
    positions: Sequence[Sequence[int]] | None,
    *,
    shape: tuple[int, int],
) -> jax.Array | None:
    if positions is None:
        return None
    return jnp.asarray(positions, dtype=jnp.int32).reshape(shape)


def _reset_eval_batch(
    *,
    args: argparse.Namespace,
    env: JaxAdversarialAntByteEnv,
    key: jax.Array,
    fixed_hub_positions: jax.Array | None,
    fixed_food_positions: jax.Array | None,
) -> tuple[Any, dict[str, jax.Array]]:
    if fixed_hub_positions is None and fixed_food_positions is None:
        return reset_batch(args=args, env=env, key=key)
    if int(args.num_envs) != 1:
        raise ValueError("fixed evaluation layouts require num_envs=1.")
    state, obs, _ = env.reset(
        key,
        hub_pos=fixed_hub_positions,
        food_positions=fixed_food_positions,
    )
    return (
        jax.tree_util.tree_map(lambda value: value[None, ...], state),
        {name: value[None, ...] for name, value in obs.items()},
    )


def _placement_stats(states: Any) -> dict[str, float]:
    hubs = np.asarray(states.hub_pos)[0].astype(np.float32)
    hub_pair_distance = float(np.sum(np.abs(hubs[0] - hubs[1])))
    food = np.asarray(states.initial_food)[0]
    food_yx = np.argwhere(food > 0)
    if len(food_yx) == 0:
        food_midpoint_distance = 0.0
    else:
        food_xy = food_yx[:, ::-1].astype(np.float32)
        midpoint = np.mean(hubs, axis=0)
        food_midpoint_distance = float(np.mean(np.sum(np.abs(food_xy - midpoint), axis=1)))
    return {
        "hub_pair_distance": hub_pair_distance,
        "food_midpoint_distance": food_midpoint_distance,
    }


def _evaluation_step(
    *,
    env: JaxAdversarialAntByteEnv,
    args: argparse.Namespace,
    learner_params: JaxMAPPOParams | None,
    opponent_params: JaxMAPPOParams | None,
    learner_kind: PolicyKind,
    opponent_kind: PolicyKind,
    learner_team: int,
    states: Any,
    obs: dict[str, jax.Array],
    key: jax.Array,
    food_scale: float,
) -> tuple[Any, dict[str, jax.Array], jax.Array, jax.Array, Any]:
    learner_key, opponent_key = jax.random.split(key)
    opponent_team = 1 - int(learner_team)
    learner_actions = _actions_for_policy(
        learner_params,
        obs,
        team=int(learner_team),
        kind=learner_kind,
        key=learner_key,
        args=args,
        food_scale=food_scale,
    )
    opponent_actions = _actions_for_policy(
        opponent_params,
        obs,
        team=opponent_team,
        kind=opponent_kind,
        key=opponent_key,
        args=args,
        food_scale=food_scale,
    )
    joint_actions = compose_team_actions(
        learner_actions,
        opponent_actions,
        learner_team=int(learner_team),
    )
    states, obs, _, terminated, truncated, infos = jax.vmap(env.step)(
        states,
        flatten_agent_actions(joint_actions),
    )
    return states, obs, terminated, truncated, infos


def _actions_for_policy(
    params: JaxMAPPOParams | None,
    obs: dict[str, jax.Array],
    *,
    team: int,
    kind: PolicyKind,
    key: jax.Array,
    args: argparse.Namespace,
    food_scale: float,
) -> jax.Array:
    num_ants_per_team = int(args.num_ants_per_team)
    if kind == "random":
        move_key, write_key = jax.random.split(key)
        move_actions = jax.random.randint(
            move_key,
            (obs["food"].shape[0], num_ants_per_team),
            0,
            MOVEMENT_ACTION_COUNT,
        )
        write_actions = jax.random.randint(
            write_key,
            (obs["food"].shape[0], num_ants_per_team),
            0,
            write_value_count(args.write_bits),
        )
        return jnp.stack([move_actions, write_actions], axis=-1).astype(jnp.int32)
    if params is None:
        raise ValueError("model policy requires params.")
    actor_obs = build_team_actor_observations(
        obs,
        team=team,
        num_ants_per_team=num_ants_per_team,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
    )
    move_logits, write_logits = get_action_logits(params, actor_obs)
    return actions_from_logits(
        move_logits,
        write_logits,
        key,
        action_mode=str(args.eval_action_mode),
        move_temperature=float(args.training_rollout_temperature),
        write_temperature=float(args.training_rollout_temperature),
    )


def _make_eval_env(args: argparse.Namespace) -> JaxAdversarialAntByteEnv:
    return JaxAdversarialAntByteEnv(
        width=args.width,
        height=args.height,
        num_ants_per_team=args.num_ants_per_team,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        layout_margin=args.layout_margin,
        hub_center_window_size=args.hub_center_window_size,
        hub_pair_distance=args.hub_pair_distance,
        hub_pair_distance_min=args.hub_pair_distance_min,
        hub_pair_distance_max=args.hub_pair_distance_max,
        food_midpoint_window_size=args.food_midpoint_window_size,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        write_while_moving=args.write_while_moving,
        terminate_on_food_delivery=args.food_termination,
        delivery_limit=args.delivery_limit,
    )
