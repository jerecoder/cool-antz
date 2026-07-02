from __future__ import annotations

import argparse
import importlib
import json
import types
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ant_byte_env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    MOVEMENT_ACTION_COUNT,
    actor_vision_patch_size,
)
from ant_byte_env.jax_autocurriculum_env import JaxAntByteAutoCurriculumEnv
from ant_byte_env.jax_env import JaxAntByteForagingEnv
from ant_byte_env.training.jax_mappo import (
    CRITIC_ARCHITECTURES,
    LinearParams,
    build_actor_observations,
    build_central_observations,
    build_local_grid_patches,
    collect_rollout,
    compute_forage_curriculum_rewards,
    compute_gae,
    compute_terminal_write_entropy_bonus,
    compute_write_bit_entropy_bonus,
    compute_write_bit_penalties,
    food_observation_scale,
    evaluate_checkpoint,
    evaluate_params,
    flatten_agent_actions,
    get_action_and_value,
    get_value,
    init_adam_state,
    init_agent_params,
    load_checkpoint_for_training,
    main,
    parse_args,
    probe_communication_checkpoint,
    repeated_write_action_indices,
    reset_batch,
    save_checkpoint,
    write_value_count,
    warm_start_actor_params,
)
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.layout_audit import LayoutAuditTracker
from ant_byte_env.training.jax_mappo.transfer_actor import adapt_movement_head_layer
from ant_byte_env.training.jax_mappo.transfer_shapes import (
    actor_obs_dim_for_bits,
    central_obs_dim_with_ants_count,
    legacy_central_obs_dim,
)
from ant_byte_env.training.jax_mappo.core import Rollout, TrainingBatch, _shuffle_batch
from ant_byte_env.training.jax_mappo.probe import (
    _applied_probe_write_values,
    write_action_bit_summary,
)
from ant_byte_env.training.jax_mappo.runner import _rollout_stats


def _batched_reset_obs() -> dict[str, jax.Array]:
    env = JaxAntByteForagingEnv(
        width=4,
        height=3,
        num_ants=2,
        food_count=3,
        random_food=False,
    )
    _, obs, _ = env.reset(
        jax.random.PRNGKey(123),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )
    return {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}


def _rollout_args(extra: list[str] | None = None) -> argparse.Namespace:
    return parse_args(
        [
            "--total-timesteps",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "5",
            "--height",
            "3",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--cookie-distance",
            "2",
            "--max-steps",
            "8",
            "--hidden-size",
            "8",
            "--seed",
            "7",
            "--quiet",
            *(extra or []),
        ]
    )


def _params_for_args(args: argparse.Namespace, env: JaxAntByteForagingEnv):
    states, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
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
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
        critic_architecture=getattr(args, "critic_architecture", "mlp"),
        critic_num_ants=args.num_ants,
        critic_obs_height=args.obs_height or args.height,
        critic_obs_width=args.obs_width or args.width,
    )
    return params, states, obs


def _food_source_signature(food_grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    food_positions = np.argwhere(food_grid > 0)[:, ::-1]
    return tuple(tuple(int(coord) for coord in position) for position in food_positions)


def _assert_layouts_changed(
    previous_obs: dict[str, jax.Array],
    previous_food: jax.Array,
    next_obs: dict[str, jax.Array],
) -> None:
    previous_hubs = np.asarray(previous_obs["hub_pos"])
    next_hubs = np.asarray(next_obs["hub_pos"])
    previous_food_np = np.asarray(previous_food)
    next_food_np = np.asarray(next_obs["food"])
    previous_obstacles = (
        np.asarray(previous_obs["obstacles"]) if "obstacles" in previous_obs else None
    )
    next_obstacles = np.asarray(next_obs["obstacles"]) if "obstacles" in next_obs else None
    for env_index in range(next_hubs.shape[0]):
        assert tuple(next_hubs[env_index]) != tuple(previous_hubs[env_index])
        previous_sources = set(_food_source_signature(previous_food_np[env_index]))
        next_sources = set(_food_source_signature(next_food_np[env_index]))
        assert next_sources
        assert next_sources.isdisjoint(previous_sources)
        if (
            previous_obstacles is not None
            and next_obstacles is not None
            and np.any(previous_obstacles[env_index])
        ):
            assert not np.array_equal(
                previous_obstacles[env_index],
                next_obstacles[env_index],
            )


def test_jax_parse_args_accepts_autocurriculum_defaults() -> None:
    args = parse_args(
        [
            "--autocurriculum",
            "--width",
            "50",
            "--height",
            "50",
            "--obs-width",
            "50",
            "--obs-height",
            "50",
            "--food-count",
            "12",
            "--food-sources",
            "2",
            "--max-steps",
            "10000",
        ]
    )

    assert args.autocurriculum is True
    assert args.autocurriculum_start_size == 4
    assert args.autocurriculum_success_cookies == 6
    assert args.food_count == 12
    assert args.food_sources == 2


def test_jax_parse_args_tracks_available_critic_architectures() -> None:
    assert CRITIC_ARCHITECTURES == (
        "mlp",
        "structured_mlp",
        "strided_cnn",
        "resnet_cnn",
    )

    for architecture in CRITIC_ARCHITECTURES:
        args = parse_args(["--critic-architecture", architecture])

        assert args.critic_architecture == architecture


def test_jax_parse_args_accepts_resnet_cnn_critic() -> None:
    args = parse_args(["--critic-architecture", "resnet_cnn"])

    assert args.critic_architecture == "resnet_cnn"


def test_jax_parse_args_accepts_structured_mlp_critic() -> None:
    args = parse_args(["--critic-architecture", "structured_mlp"])

    assert args.critic_architecture == "structured_mlp"


def test_jax_adam_state_uses_independent_momentum_buffers() -> None:
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=8,
        actor_obs_dim=6,
        hidden_size=4,
        write_value_count=2,
    )

    opt_state = init_adam_state(params)

    for momentum, variance in zip(
        jax.tree_util.tree_leaves(opt_state.m),
        jax.tree_util.tree_leaves(opt_state.v),
    ):
        assert momentum is not variance


def test_jax_autocurriculum_reset_batch_uses_fixed_critic_shape() -> None:
    args = _rollout_args(
        [
            "--autocurriculum",
            "--width",
            "50",
            "--height",
            "50",
            "--obs-width",
            "50",
            "--obs-height",
            "50",
            "--food-count",
            "12",
            "--food-sources",
            "2",
            "--max-steps",
            "200",
        ]
    )
    env = JaxAntByteAutoCurriculumEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        start_size=args.autocurriculum_start_size,
        success_cookies=args.autocurriculum_success_cookies,
        actor_vision_radius=args.actor_vision_radius,
        random_food=args.random_food,
        random_hub=args.random_hub,
        write_bits=args.write_bits,
    )

    states, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
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

    assert states.food.shape[-2:] == (50, 50)
    assert int(np.asarray(states.active_size)[0]) == 4
    assert central_obs.shape[-1] == 7511
    assert actor_obs.shape[-1] == 50
    np.testing.assert_allclose(
        np.asarray(central_obs[0, -2:]),
        np.array([4 / 50, 4 / 50], dtype=np.float32),
    )


@pytest.mark.parametrize("maze_obstacles", [False, True])
def test_jax_reset_batch_excludes_previous_random_layout(
    maze_obstacles: bool,
) -> None:
    args = _rollout_args(
        [
            "--num-envs",
            "3",
            "--width",
            "10",
            "--height",
            "10",
            "--food-count",
            "6",
            "--food-sources",
            "2",
            "--max-steps",
            "2",
            "--random-hub",
            "--random-food",
            *(["--maze-obstacles", "--maze-seed", "17"] if maze_obstacles else []),
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        write_bits=args.write_bits,
        maze_obstacles=bool(getattr(args, "maze_obstacles", False)),
        maze_seed=int(getattr(args, "maze_seed", 0)),
    )

    states, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(101))
    empty_terminal_obs = {**obs, "food": jnp.zeros_like(obs["food"])}
    _, next_obs = reset_batch(
        args=args,
        env=env,
        key=jax.random.PRNGKey(102),
        previous_obs=empty_terminal_obs,
        previous_food=states.initial_food,
    )

    _assert_layouts_changed(obs, states.initial_food, next_obs)


def _fits_two_chebyshev_clusters(
    positions: np.ndarray,
    *,
    width: int,
    height: int,
    radius: int,
) -> bool:
    candidate_centers = [
        np.array([x_pos, y_pos], dtype=np.int32)
        for y_pos in range(height)
        for x_pos in range(width)
    ]
    for left in candidate_centers:
        left_mask = np.max(np.abs(positions - left), axis=1) <= radius
        for right in candidate_centers:
            right_mask = np.max(np.abs(positions - right), axis=1) <= radius
            if bool(np.all(left_mask | right_mask)):
                return True
    return False


def test_jax_reset_batch_can_cluster_food_around_two_macro_sources() -> None:
    args = _rollout_args(
        [
            "--num-envs",
            "4",
            "--width",
            "20",
            "--height",
            "20",
            "--layout-margin",
            "3",
            "--hub-center-window-size",
            "4",
            "--food-count",
            "20",
            "--food-sources",
            "10",
            "--food-cluster-count",
            "2",
            "--food-cluster-radius",
            "2",
            "--max-steps",
            "2",
            "--random-hub",
            "--random-food",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        layout_margin=args.layout_margin,
        write_bits=args.write_bits,
    )

    states, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(2026))
    food_grids = np.asarray(states.initial_food)
    hub_positions = np.asarray(obs["hub_pos"])

    for env_index, food_grid in enumerate(food_grids):
        positions = np.argwhere(food_grid > 0)[:, ::-1]
        assert positions.shape == (10, 2)
        assert int(food_grid.sum()) == 20
        assert _fits_two_chebyshev_clusters(
            positions,
            width=args.width,
            height=args.height,
            radius=args.food_cluster_radius,
        )
        assert np.all(positions >= args.layout_margin)
        assert np.all(positions < args.width - args.layout_margin)
        hub_start = (args.width - args.hub_center_window_size) // 2
        hub_end = hub_start + args.hub_center_window_size
        assert np.all(hub_positions[env_index] >= hub_start)
        assert np.all(hub_positions[env_index] < hub_end)


@pytest.mark.parametrize("maze_obstacles", [False, True])
def test_jax_collect_rollout_resets_truncated_envs_to_new_layout(
    maze_obstacles: bool,
) -> None:
    args = _rollout_args(
        [
            "--num-envs",
            "2",
            "--num-steps",
            "1",
            "--width",
            "10",
            "--height",
            "10",
            "--food-count",
            "6",
            "--food-sources",
            "2",
            "--max-steps",
            "1",
            "--random-hub",
            "--random-food",
            *(["--maze-obstacles", "--maze-seed", "17"] if maze_obstacles else []),
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        write_bits=args.write_bits,
        maze_obstacles=bool(getattr(args, "maze_obstacles", False)),
        maze_seed=int(getattr(args, "maze_seed", 0)),
    )
    params, states, obs = _params_for_args(args, env)

    _, final_obs, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(202),
    )

    np.testing.assert_array_equal(
        np.asarray(rollout.truncations[0]),
        np.array([True, True]),
    )
    _assert_layouts_changed(obs, states.initial_food, final_obs)


def test_jax_observation_builders_match_mappo_shapes() -> None:
    obs = _batched_reset_obs()

    central_obs = build_central_observations(obs, food_scale=3)
    actor_obs = build_actor_observations(obs, food_scale=3)

    assert central_obs.shape == (1, 54)
    assert actor_obs.shape == (1, 2, 52)
    assert bool(jnp.all(central_obs >= 0.0))
    assert bool(jnp.all(central_obs <= 1.0))
    assert bool(jnp.all(actor_obs >= 0.0))
    assert bool(jnp.all(actor_obs <= 1.0))
    np.testing.assert_allclose(
        np.asarray(central_obs[0, 6:14]),
        np.array([0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(actor_obs[0, 0, -5:]),
        np.array([0, 0, 1, 0, 0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(actor_obs[0, :, 45:47]),
        np.array([[1, 0], [0, 1]], dtype=np.float32),
    )


def test_jax_observation_builders_preserve_food_amounts() -> None:
    obs = _batched_reset_obs()

    central_obs = build_central_observations(obs, food_scale=3)
    actor_obs = build_actor_observations(obs, food_scale=3, actor_vision_radius=1)

    ants_pos_width = 4
    ants_carrying_width = 2
    ants_facing_width = 8
    ants_count_width = 12
    food_cell_index = 1
    central_food_value = central_obs[
        0,
        (
            ants_pos_width
            + ants_carrying_width
            + ants_facing_width
            + ants_count_width
            + food_cell_index
        ),
    ]
    local_food_patch_index = 5

    assert float(central_food_value) == 1.0
    assert float(central_obs[0, ants_pos_width + ants_carrying_width + ants_facing_width]) == 1.0
    assert float(actor_obs[0, 0, local_food_patch_index]) == 1.0


def test_jax_food_observation_scale_uses_source_amount_not_total_food() -> None:
    assert food_observation_scale(food_count=2500, food_sources=250) == 10.0
    assert food_observation_scale(food_count=20, food_sources=2) == 10.0
    assert food_observation_scale(food_count=48, food_sources=12) == 4.0
    assert food_observation_scale(food_count=18, food_sources=0) == 18.0


def test_jax_actor_observation_exposes_border_mask() -> None:
    obs = _batched_reset_obs()

    actor_obs = build_actor_observations(obs, food_scale=3, actor_vision_radius=1)

    assert actor_obs.shape == (1, 2, 52)
    np.testing.assert_allclose(
        np.asarray(actor_obs[0, 0, 36:45]),
        np.array([1, 1, 1.0, 1, 0, 0.0, 1, 0, 0.0], dtype=np.float32),
    )


def test_jax_actor_observation_window_contains_current_ant_tile() -> None:
    obs = _batched_reset_obs()
    obs = {
        **obs,
        "ants_pos": obs["ants_pos"].at[0, 0].set(jnp.array([2, 1], dtype=jnp.int32)),
    }
    ants_count = jnp.zeros_like(obs["ants_count"])
    ants_count = ants_count.at[0, 1, 2].set(1)
    ants_count = ants_count.at[0, 0, 0].set(1)
    obs = {**obs, "ants_count": ants_count}

    actor_obs = build_actor_observations(
        obs,
        food_scale=3,
        actor_vision_radius=1,
    )

    np.testing.assert_allclose(
        np.asarray(actor_obs[0, 0, 9:18]),
        np.array([0, 0, 0.0, 0, 0.5, 0.0, 0, 0, 0.0], dtype=np.float32),
    )


def test_jax_actor_vision_patch_is_centered_three_by_three_grid() -> None:
    grid = jnp.asarray(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
                [40.0, 41.0, 42.0, 43.0, 44.0],
            ]
        ],
        dtype=jnp.float32,
    )
    ants_pos = jnp.asarray([[[2, 2]]], dtype=jnp.int32)
    ants_facing = jnp.asarray([[2]], dtype=jnp.int32)

    patch = build_local_grid_patches(
        grid,
        ants_pos,
        radius=1,
        ants_facing=ants_facing,
    )

    assert patch.shape == (1, 1, 9)
    np.testing.assert_allclose(
        np.asarray(patch[0, 0]),
        np.array(
            [11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0],
            dtype=np.float32,
        ),
    )


def test_jax_actor_vision_patch_rotates_with_facing() -> None:
    grid = jnp.asarray(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
                [40.0, 41.0, 42.0, 43.0, 44.0],
            ]
        ],
        dtype=jnp.float32,
    )
    ants_pos = jnp.asarray([[[2, 2], [2, 2], [2, 2]]], dtype=jnp.int32)
    ants_facing = jnp.asarray([[ACTION_DOWN, ACTION_LEFT, ACTION_UP]], dtype=jnp.int32)

    patch = build_local_grid_patches(
        grid,
        ants_pos,
        radius=1,
        ants_facing=ants_facing,
    )

    np.testing.assert_allclose(
        np.asarray(patch[0, 0]),
        np.array([13.0, 23.0, 33.0, 12.0, 22.0, 32.0, 11.0, 21.0, 31.0]),
    )
    np.testing.assert_allclose(
        np.asarray(patch[0, 1]),
        np.array([33.0, 32.0, 31.0, 23.0, 22.0, 21.0, 13.0, 12.0, 11.0]),
    )
    np.testing.assert_allclose(
        np.asarray(patch[0, 2]),
        np.array([31.0, 21.0, 11.0, 32.0, 22.0, 12.0, 33.0, 23.0, 13.0]),
    )


def test_jax_actor_observation_ignores_global_targets_outside_local_vision() -> None:
    env = JaxAntByteForagingEnv(
        width=7,
        height=7,
        num_ants=1,
        food_count=1,
        random_food=False,
    )

    _, obs_a, _ = env.reset(
        jax.random.PRNGKey(123),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[6, 6]], dtype=jnp.int32),
    )
    _, obs_b, _ = env.reset(
        jax.random.PRNGKey(124),
        hub_pos=jnp.array([6, 6], dtype=jnp.int32),
        food_positions=jnp.array([[0, 0]], dtype=jnp.int32),
    )

    def center_ant(obs: dict[str, jax.Array]) -> dict[str, jax.Array]:
        obs = {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}
        obs["ants_pos"] = jnp.asarray([[[3, 3]]], dtype=jnp.int32)
        obs["ants_count"] = jnp.zeros((1, 7, 7), dtype=jnp.int32).at[:, 3, 3].set(1)
        obs["ants_facing"] = jnp.asarray([[ACTION_RIGHT]], dtype=jnp.int32)
        obs["ants_carrying"] = jnp.zeros((1, 1), dtype=jnp.int8)
        return obs

    actor_obs_a = build_actor_observations(
        center_ant(obs_a),
        food_scale=1,
        actor_vision_radius=1,
        write_bits=1,
    )
    actor_obs_b = build_actor_observations(
        center_ant(obs_b),
        food_scale=1,
        actor_vision_radius=1,
        write_bits=1,
    )

    assert actor_obs_a.shape[-1] == actor_obs_dim_for_bits(
        write_bits=1,
        actor_vision_radius=1,
    )
    np.testing.assert_allclose(np.asarray(actor_obs_a), np.asarray(actor_obs_b))


def test_jax_actor_observation_changes_for_targets_inside_local_vision() -> None:
    env = JaxAntByteForagingEnv(
        width=7,
        height=7,
        num_ants=1,
        food_count=1,
        random_food=False,
    )
    _, obs, _ = env.reset(
        jax.random.PRNGKey(125),
        hub_pos=jnp.array([3, 3], dtype=jnp.int32),
        food_positions=jnp.array([[4, 3]], dtype=jnp.int32),
    )
    obs = {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}
    obs["ants_pos"] = jnp.asarray([[[3, 3]]], dtype=jnp.int32)
    obs["ants_count"] = jnp.zeros((1, 7, 7), dtype=jnp.int32).at[:, 3, 3].set(1)
    obs["ants_facing"] = jnp.asarray([[ACTION_RIGHT]], dtype=jnp.int32)

    actor_obs = build_actor_observations(
        obs,
        food_scale=1,
        actor_vision_radius=1,
        write_bits=1,
    )

    patch_size = actor_vision_patch_size(1)
    local_food = np.asarray(actor_obs[0, 0, :patch_size])
    local_hub = np.asarray(actor_obs[0, 0, 3 * patch_size : 4 * patch_size])
    assert np.count_nonzero(local_food) == 1
    assert np.count_nonzero(local_hub) == 1


def test_jax_forage_curriculum_rewards_add_pickup_without_target_progress() -> None:
    env = JaxAntByteForagingEnv(width=4, height=3, num_ants=1, food_count=1)
    state, previous_obs, _ = env.reset(
        jax.random.PRNGKey(5),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )
    _, next_obs, env_reward, _, _, _ = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs={key: jnp.expand_dims(value, axis=0) for key, value in previous_obs.items()},
        next_obs={key: jnp.expand_dims(value, axis=0) for key, value in next_obs.items()},
        env_rewards=jnp.asarray([env_reward], dtype=jnp.float32),
        pickup_bonus=0.25,
    )

    np.testing.assert_allclose(np.asarray(shaped_rewards), np.array([0.25], dtype=np.float32))

    state, previous_obs, _ = env.reset(
        jax.random.PRNGKey(5),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[2, 0]], dtype=jnp.int32),
    )
    _, next_obs, env_reward, _, _, _ = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs={key: jnp.expand_dims(value, axis=0) for key, value in previous_obs.items()},
        next_obs={key: jnp.expand_dims(value, axis=0) for key, value in next_obs.items()},
        env_rewards=jnp.asarray([env_reward], dtype=jnp.float32),
        pickup_bonus=0.25,
    )

    np.testing.assert_allclose(np.asarray(shaped_rewards), np.array([0.0], dtype=np.float32))


def test_jax_forage_curriculum_distance_bonus_rewards_food_progress() -> None:
    env = JaxAntByteForagingEnv(width=4, height=3, num_ants=1, food_count=1)
    state, previous_obs, _ = env.reset(
        jax.random.PRNGKey(5),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[2, 0]], dtype=jnp.int32),
    )
    _, next_obs, env_reward, _, _, _ = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs={key: jnp.expand_dims(value, axis=0) for key, value in previous_obs.items()},
        next_obs={key: jnp.expand_dims(value, axis=0) for key, value in next_obs.items()},
        env_rewards=jnp.asarray([env_reward], dtype=jnp.float32),
        pickup_bonus=0.25,
        distance_bonus=0.5,
    )

    np.testing.assert_allclose(np.asarray(shaped_rewards), np.array([0.1], dtype=np.float32))


def test_jax_forage_curriculum_distance_bonus_rewards_hub_progress() -> None:
    previous_obs = {
        "food": jnp.zeros((1, 3, 4), dtype=jnp.int32),
        "ants_pos": jnp.array([[[3, 0]]], dtype=jnp.int32),
        "ants_carrying": jnp.array([[True]]),
        "hub_pos": jnp.array([[0, 0]], dtype=jnp.int32),
    }
    next_obs = {
        **previous_obs,
        "ants_pos": jnp.array([[[2, 0]]], dtype=jnp.int32),
    }

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.asarray([0.0], dtype=jnp.float32),
        pickup_bonus=0.25,
        distance_bonus=0.5,
    )

    np.testing.assert_allclose(np.asarray(shaped_rewards), np.array([0.1], dtype=np.float32))


def test_jax_forage_curriculum_distance_bonus_ignores_pickup_target_switch() -> None:
    env = JaxAntByteForagingEnv(width=4, height=3, num_ants=1, food_count=1)
    state, previous_obs, _ = env.reset(
        jax.random.PRNGKey(5),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )
    _, next_obs, env_reward, _, _, _ = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs={key: jnp.expand_dims(value, axis=0) for key, value in previous_obs.items()},
        next_obs={key: jnp.expand_dims(value, axis=0) for key, value in next_obs.items()},
        env_rewards=jnp.asarray([env_reward], dtype=jnp.float32),
        pickup_bonus=0.25,
        distance_bonus=1.0,
    )

    np.testing.assert_allclose(np.asarray(shaped_rewards), np.array([0.25], dtype=np.float32))


def test_jax_forage_curriculum_carrying_hub_distance_bonus_only_rewards_return_leg() -> None:
    previous_obs = {
        "food": jnp.zeros((4, 3, 4), dtype=jnp.int32),
        "ants_pos": jnp.array(
            [
                [[3, 0]],
                [[3, 0]],
                [[1, 0]],
                [[2, 0]],
            ],
            dtype=jnp.int32,
        ),
        "ants_carrying": jnp.array([[True], [False], [True], [True]]),
        "hub_pos": jnp.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=jnp.int32,
        ),
    }
    next_obs = {
        **previous_obs,
        "ants_pos": jnp.array(
            [
                [[2, 0]],  # carrying and closer to hub
                [[2, 0]],  # empty and closer to hub: no carrying-home reward
                [[0, 0]],  # delivery transition: no shaping on target switch
                [[3, 0]],  # carrying and farther from hub
            ],
            dtype=jnp.int32,
        ),
        "ants_carrying": jnp.array([[True], [False], [False], [True]]),
    }

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.zeros((4,), dtype=jnp.float32),
        pickup_bonus=0.0,
        carrying_hub_distance_bonus=0.5,
    )

    np.testing.assert_allclose(
        np.asarray(shaped_rewards),
        np.array([0.1, 0.0, 0.0, -0.1], dtype=np.float32),
    )


def test_jax_forage_curriculum_visit_bonus_decays_with_coverage() -> None:
    previous_obs = {
        "food": jnp.zeros((2, 4, 4), dtype=jnp.int32),
        "ants_pos": jnp.array([[[0, 0]], [[0, 0]]], dtype=jnp.int32),
        "ants_carrying": jnp.array([[False], [False]]),
        "hub_pos": jnp.array([[0, 0], [0, 0]], dtype=jnp.int32),
    }
    next_obs = previous_obs

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        pickup_bonus=0.0,
        newly_visited_cells=jnp.asarray([2.0, 1.0], dtype=jnp.float32),
        visited_cell_fraction=jnp.asarray([0.25, 0.8], dtype=jnp.float32),
        visit_reward_scale=0.1,
        visit_reward_decay=1.0,
    )

    np.testing.assert_allclose(
        np.asarray(shaped_rewards),
        np.array([1.15, 0.02], dtype=np.float32),
        rtol=1e-6,
    )


def test_jax_forage_curriculum_border_view_penalty_subtracts_visible_edges() -> None:
    previous_obs = {
        "food": jnp.zeros((2, 4, 4), dtype=jnp.int32),
        "ants_pos": jnp.array([[[0, 0]], [[2, 2]]], dtype=jnp.int32),
        "ants_carrying": jnp.array([[False], [False]]),
        "hub_pos": jnp.array([[0, 0], [0, 0]], dtype=jnp.int32),
    }
    next_obs = previous_obs

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        pickup_bonus=0.0,
        visible_border_cells=jnp.asarray([4.0, 0.0], dtype=jnp.float32),
        border_view_penalty=0.02,
    )

    np.testing.assert_allclose(
        np.asarray(shaped_rewards),
        np.array([0.92, 1.0], dtype=np.float32),
        rtol=1e-6,
    )


def test_jax_forage_curriculum_border_moat_penalty_subtracts_near_edge_cost() -> None:
    previous_obs = {
        "food": jnp.zeros((2, 4, 4), dtype=jnp.int32),
        "ants_pos": jnp.array([[[0, 0]], [[2, 2]]], dtype=jnp.int32),
        "ants_carrying": jnp.array([[False], [False]]),
        "hub_pos": jnp.array([[0, 0], [0, 0]], dtype=jnp.int32),
    }
    next_obs = previous_obs

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        pickup_bonus=0.0,
        border_moat_cost=jnp.asarray([6.0, 0.0], dtype=jnp.float32),
        border_moat_penalty=0.01,
    )

    np.testing.assert_allclose(
        np.asarray(shaped_rewards),
        np.array([0.94, 1.0], dtype=np.float32),
        rtol=1e-6,
    )


def test_jax_forage_curriculum_stage_completion_bonus_adds_only_on_stage_advance() -> None:
    previous_obs = {
        "food": jnp.zeros((2, 3, 4), dtype=jnp.int32),
        "ants_pos": jnp.array([[[0, 0]], [[0, 0]]], dtype=jnp.int32),
        "ants_carrying": jnp.array([[False], [False]]),
        "hub_pos": jnp.array([[0, 0], [0, 0]], dtype=jnp.int32),
    }
    next_obs = previous_obs

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        pickup_bonus=0.25,
        stage_completion_events=jnp.asarray([1, 0], dtype=jnp.int32),
        stage_completion_bonus=3.0,
    )

    np.testing.assert_allclose(np.asarray(shaped_rewards), np.array([4.0, 1.0]))


def test_jax_forage_curriculum_delivery_byte_trail_bonus_requires_existing_active_bytes() -> None:
    previous_obs = {
        "food": jnp.zeros((3, 3, 3), dtype=jnp.int32),
        "bytes": jnp.array(
            [
                [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
            ],
            dtype=jnp.uint8,
        ),
        "ants_pos": jnp.array([[[0, 0]], [[0, 0]], [[0, 0]]], dtype=jnp.int32),
        "ants_carrying": jnp.array([[True], [True], [True]]),
        "hub_pos": jnp.array([[0, 0], [0, 0], [0, 0]], dtype=jnp.int32),
        "active_grid_size": jnp.array([[2, 2], [2, 2], [3, 3]], dtype=jnp.int32),
    }
    next_obs = {**previous_obs, "ants_carrying": jnp.array([[False], [False], [False]])}

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32),
        pickup_bonus=0.25,
        delivery_byte_trail_bonus=0.5,
        delivery_byte_trail_target_tiles=4,
    )

    np.testing.assert_allclose(
        np.asarray(shaped_rewards),
        np.array([1.5, 1.0, 1.5], dtype=np.float32),
    )


def test_jax_forage_curriculum_byte_follow_bonus_rewards_food_progress_on_bytes() -> None:
    previous_obs = {
        "food": jnp.array(
            [
                [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=jnp.int32,
        ),
        "bytes": jnp.array(
            [
                [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
            ],
            dtype=jnp.uint8,
        ),
        "ants_pos": jnp.array([[[0, 0]], [[0, 0]]], dtype=jnp.int32),
        "ants_carrying": jnp.array([[False], [False]]),
        "hub_pos": jnp.array([[0, 0], [0, 0]], dtype=jnp.int32),
    }
    next_obs = {
        **previous_obs,
        "ants_pos": jnp.array([[[1, 0]], [[0, 1]]], dtype=jnp.int32),
    }

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        pickup_bonus=0.0,
        byte_follow_bonus=0.4,
    )

    np.testing.assert_allclose(
        np.asarray(shaped_rewards),
        np.array([0.4, 0.0], dtype=np.float32),
    )


def test_jax_forage_curriculum_carrying_byte_write_bonus_rewards_fresh_writes() -> None:
    previous_obs = {
        "food": jnp.zeros((3, 3, 3), dtype=jnp.int32),
        "bytes": jnp.array(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=jnp.uint8,
        ),
        "ants_pos": jnp.array([[[1, 1]], [[1, 1]], [[1, 1]]], dtype=jnp.int32),
        "ants_carrying": jnp.array([[True], [True], [False]]),
        "hub_pos": jnp.array([[0, 0], [0, 0], [0, 0]], dtype=jnp.int32),
    }
    next_obs = previous_obs
    actions = jnp.array(
        [
            [[ACTION_STAY, 1]],
            [[ACTION_STAY, 1]],
            [[ACTION_STAY, 1]],
        ],
        dtype=jnp.int32,
    )

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs=previous_obs,
        next_obs=next_obs,
        env_rewards=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        actions=actions,
        pickup_bonus=0.0,
        carrying_byte_write_bonus=0.07,
    )

    np.testing.assert_allclose(
        np.asarray(shaped_rewards),
        np.array([0.07, 0.0, 0.0], dtype=np.float32),
    )


def test_jax_rollout_stats_include_forage_diagnostics() -> None:
    dummy = jnp.zeros((2, 2), dtype=jnp.float32)
    rollout = Rollout(
        actor_obs=dummy,
        central_obs=dummy,
        actions=jnp.array(
            [
                [[[0, 0]], [[1, 2]]],
                [[[2, 1]], [[3, 0]]],
            ],
            dtype=jnp.int32,
        ),
        agent_masks=jnp.ones((2, 2, 1), dtype=jnp.float32),
        logprobs=dummy,
        rewards=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        dones=jnp.array([[False, True], [False, True]]),
        terminations=jnp.array([[False, True], [False, False]]),
        truncations=jnp.array([[False, False], [False, True]]),
        values=dummy,
        next_values=dummy,
        env_rewards=jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
        pickup_events=jnp.array([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float32),
        delivery_events=jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
        carrying_ants=jnp.array([[0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32),
        remaining_food=jnp.array([[7.0, 5.0], [6.0, 4.0]], dtype=jnp.float32),
        active_size=jnp.array([[12.0, 15.0], [4.0, 4.0]], dtype=jnp.float32),
        stage_advances=jnp.array([[1.0, 2.0], [0.0, 0.0]], dtype=jnp.float32),
        stage_delivered_food=jnp.array([[2.0, 4.0], [0.0, 0.0]], dtype=jnp.float32),
        newly_visited_cells=jnp.array([[1.0, 2.0], [0.0, 3.0]], dtype=jnp.float32),
        visited_cell_count=jnp.array([[3.0, 4.0], [3.0, 7.0]], dtype=jnp.float32),
        visited_cell_fraction=jnp.array([[0.1875, 0.25], [0.1875, 0.4375]], dtype=jnp.float32),
        newly_viewed_cells=jnp.array([[2.0, 3.0], [1.0, 4.0]], dtype=jnp.float32),
        viewed_cell_count=jnp.array([[5.0, 7.0], [6.0, 10.0]], dtype=jnp.float32),
        viewed_cell_fraction=jnp.array([[0.3125, 0.4375], [0.375, 0.625]], dtype=jnp.float32),
        visible_border_cells=jnp.array([[0.0, 1.0], [2.0, 3.0]], dtype=jnp.float32),
        border_moat_cost=jnp.array([[0.0, 2.0], [4.0, 6.0]], dtype=jnp.float32),
        nonzero_byte_tiles=jnp.array([[0.0, 2.0], [4.0, 6.0]], dtype=jnp.float32),
        nonzero_byte_fraction=jnp.array([[0.0, 0.125], [0.25, 0.375]], dtype=jnp.float32),
        applied_nonzero_write_actions=jnp.array(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=jnp.float32,
        ),
        empty_nonzero_write_actions=jnp.array(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=jnp.float32,
        ),
        carrying_nonzero_write_actions=jnp.zeros((2, 2), dtype=jnp.float32),
        empty_write_action_slots=jnp.ones((2, 2), dtype=jnp.float32),
        carrying_write_action_slots=jnp.zeros((2, 2), dtype=jnp.float32),
        write_attempts=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        overwrite_events=jnp.array([[0.0, 1.0], [0.0, 1.0]], dtype=jnp.float32),
        reset_hub_pos=jnp.zeros((2, 2, 2), dtype=jnp.int32),
        reset_food_positions=jnp.zeros((2, 2, 1, 2), dtype=jnp.int32),
    )

    stats = _rollout_stats(rollout)

    assert stats["episode_return"] == 5.0
    assert stats["env_return"] == 1.0
    assert stats["completed_episodes"] == 2.0
    assert stats["terminated_episodes"] == 1.0
    assert stats["truncated_episodes"] == 1.0
    assert stats["pickup_events"] == 3.0
    assert stats["delivery_events"] == 2.0
    assert stats["mean_carrying_ants"] == 0.75
    assert stats["final_mean_remaining_food"] == 5.0
    assert stats["visited_cell_events"] == 6.0
    assert stats["mean_visited_cell_count"] == 4.25
    assert stats["final_mean_visited_cell_count"] == 5.0
    assert stats["mean_visited_cell_fraction"] == 0.265625
    assert stats["final_mean_visited_cell_fraction"] == 0.3125
    assert stats["viewed_cell_events"] == 10.0
    assert stats["mean_viewed_cell_count"] == 7.0
    assert stats["final_mean_viewed_cell_count"] == 8.0
    assert stats["mean_viewed_cell_fraction"] == 0.4375
    assert stats["final_mean_viewed_cell_fraction"] == 0.5
    assert stats["mean_visible_border_cells"] == 1.5
    assert stats["final_mean_visible_border_cells"] == 2.5
    assert stats["mean_border_moat_cost"] == 3.0
    assert stats["final_mean_border_moat_cost"] == 5.0
    assert stats["write_action_nonzero_rate"] == 0.5
    assert stats["mean_write_action_value"] == 0.75
    assert stats["applied_write_action_nonzero_rate"] == 0.5
    assert stats["empty_write_action_nonzero_rate"] == 0.5
    assert stats["carrying_write_action_nonzero_rate"] == 0.0
    assert stats["nonzero_writes_per_delivery"] == 1.0
    assert stats["mean_write_attempts_per_env_step"] == 2.5
    assert stats["mean_overwrites_per_env_step"] == 0.5
    assert stats["mean_nonzero_byte_tiles"] == 3.0
    assert stats["final_mean_nonzero_byte_tiles"] == 5.0
    assert stats["mean_nonzero_byte_fraction"] == 0.1875
    assert stats["final_mean_nonzero_byte_fraction"] == 0.3125
    assert stats["autocurriculum_max_active_size"] == 15.0
    assert stats["autocurriculum_mean_active_size"] == 8.75
    assert stats["autocurriculum_final_mean_active_size"] == 4.0
    assert stats["autocurriculum_completed_stages"] == 3.0
    assert stats["autocurriculum_mean_stage_delivered_food"] == 1.5


def test_layout_audit_tracker_records_reset_layouts_and_snapshots(tmp_path: Path) -> None:
    tracker = LayoutAuditTracker(
        audit_dir=tmp_path / "layout_audit",
        snapshot_interval=2,
        width=4,
        height=4,
        stage_name="4x4",
        run_name="unit-test",
    )
    obs = {
        "food": jnp.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 4, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ],
            dtype=jnp.int32,
        ),
        "hub_pos": jnp.array([[0, 0]], dtype=jnp.int32),
        "obstacles": jnp.zeros((1, 4, 4), dtype=jnp.int8),
    }
    rollout = types.SimpleNamespace(
        dones=jnp.array([[True, False], [True, True]], dtype=jnp.bool_),
        reset_hub_pos=jnp.array(
            [
                [[1, 1], [0, 0]],
                [[2, 2], [3, 3]],
            ],
            dtype=jnp.int32,
        ),
        reset_food_positions=jnp.array(
            [
                [
                    [[0, 1], [2, 0]],
                    [[-1, -1], [-1, -1]],
                ],
                [
                    [[1, 2], [2, 1]],
                    [[0, 3], [3, 0]],
                ],
            ],
            dtype=jnp.int32,
        ),
    )

    tracker.observe_observations(
        obs=obs,
        update=0,
        global_step=0,
        reason="initial_reset",
    )
    metrics = tracker.observe_rollout_resets(
        rollout=rollout,
        update=1,
        global_step=8,
    )

    records_path = tmp_path / "layout_audit" / "layout_records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    snapshots = sorted((tmp_path / "layout_audit" / "snapshots").glob("*.png"))

    assert len(records) == 4
    assert records[0]["reason"] == "initial_reset"
    assert [record["reason"] for record in records[1:]] == ["reset", "reset", "reset"]
    assert records[1]["rollout_step_index"] == 0
    assert records[2]["env_index"] == 0
    assert records[3]["food_positions"] == [[0, 3], [3, 0]]
    assert len({record["layout_hash"] for record in records}) == 4
    assert len(snapshots) == 2
    assert metrics["layout_audit_records"] == 4.0
    assert metrics["layout_audit_unique_layouts"] == 4.0
    assert metrics["layout_audit_snapshots"] == 2.0


def test_jax_collect_rollout_explore_reward_uses_newly_visited_cells() -> None:
    args = _rollout_args(
        [
            "--width",
            "3",
            "--height",
            "3",
            "--food-count",
            "0",
            "--num-steps",
            "2",
            "--reward-mode",
            "explore",
            "--no-food-termination",
            "--deterministic-rollout",
            "--write-action-ablation",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        terminate_on_food_delivery=False,
    )
    params, states, obs = _params_for_args(args, env)
    move_bias = jnp.full_like(params.move_head.bias, -10.0).at[ACTION_RIGHT].set(10.0)
    params = params._replace(
        move_head=params.move_head._replace(
            weight=jnp.zeros_like(params.move_head.weight),
            bias=move_bias,
        ),
        write_head=params.write_head._replace(
            weight=jnp.zeros_like(params.write_head.weight),
            bias=jnp.zeros_like(params.write_head.bias),
        ),
    )

    _, _, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(123),
    )

    np.testing.assert_allclose(
        np.asarray(rollout.rewards[:, 0]),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(np.asarray(rollout.env_rewards[:, 0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        np.asarray(rollout.newly_visited_cells[:, 0]),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert float(rollout.visited_cell_count[-1, 0]) == 2.0


def test_jax_agent_samples_joint_actions_for_configured_write_bits() -> None:
    obs = _batched_reset_obs()
    central_obs = build_central_observations(obs, food_scale=3, write_bits=3)
    actor_obs = build_actor_observations(obs, food_scale=3, write_bits=3)
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=16,
        write_value_count=write_value_count(3),
    )

    actions, logprob, entropy, value = get_action_and_value(
        params,
        actor_obs,
        central_obs,
        jax.random.PRNGKey(1),
    )
    flat_actions = flatten_agent_actions(actions)

    assert actor_obs.shape == (1, 2, 70)
    assert actions.shape == (1, 2, 2)
    assert logprob.shape == (1, 2)
    assert entropy.shape == (1, 2)
    assert value.shape == (1,)
    assert flat_actions.shape == (1, 4)
    assert bool(jnp.all((0 <= actions[..., 0]) & (actions[..., 0] < MOVEMENT_ACTION_COUNT)))
    assert bool(jnp.all((0 <= actions[..., 1]) & (actions[..., 1] <= 7)))


def test_jax_structured_mlp_critic_splits_grid_and_entity_features() -> None:
    env = JaxAntByteForagingEnv(
        width=8,
        height=8,
        num_ants=2,
        food_count=4,
        random_food=False,
        write_bits=2,
    )
    _, obs, _ = env.reset(
        jax.random.PRNGKey(123),
        hub_pos=jnp.array([3, 3], dtype=jnp.int32),
        food_positions=jnp.array([[6, 6], [1, 6]], dtype=jnp.int32),
    )
    obs = {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}
    central_obs = build_central_observations(obs, food_scale=4, write_bits=2)
    actor_obs = build_actor_observations(obs, food_scale=4, write_bits=2)
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=16,
        write_value_count=write_value_count(2),
        critic_architecture="structured_mlp",
        critic_num_ants=2,
        critic_obs_height=8,
        critic_obs_width=8,
    )

    value = get_value(
        params,
        central_obs,
        critic_architecture="structured_mlp",
        critic_num_ants=2,
        critic_obs_height=8,
        critic_obs_width=8,
    )
    actions, logprob, entropy, action_value = get_action_and_value(
        params,
        actor_obs,
        central_obs,
        jax.random.PRNGKey(1),
        critic_architecture="structured_mlp",
        critic_num_ants=2,
        critic_obs_height=8,
        critic_obs_width=8,
    )

    assert params.critic_body.grid_body[0].weight.shape == (3 * 8 * 8, 512)
    assert params.critic_body.grid_body[1].weight.shape == (512, 256)
    assert params.critic_body.entity_body[0].weight.shape == (7 * 2 + 4, 128)
    assert params.critic_body.entity_body[1].weight.shape == (128, 128)
    assert params.critic_body.fusion_body[0].weight.shape == (384, 256)
    assert params.critic_body.fusion_body[1].weight.shape == (256, 256)
    assert params.value_head.weight.shape == (256, 1)
    assert value.shape == (1,)
    assert actions.shape == (1, 2, 2)
    assert logprob.shape == (1, 2)
    assert entropy.shape == (1, 2)
    assert action_value.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(value)))


def test_jax_strided_cnn_critic_produces_values_from_50x50_grid() -> None:
    env = JaxAntByteForagingEnv(
        width=50,
        height=50,
        num_ants=2,
        food_count=4,
        random_food=False,
        write_bits=2,
    )
    _, obs, _ = env.reset(
        jax.random.PRNGKey(123),
        hub_pos=jnp.array([25, 25], dtype=jnp.int32),
        food_positions=jnp.array([[40, 40], [10, 40]], dtype=jnp.int32),
    )
    obs = {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}
    central_obs = build_central_observations(obs, food_scale=4, write_bits=2)
    actor_obs = build_actor_observations(obs, food_scale=4, write_bits=2)
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=16,
        write_value_count=write_value_count(2),
        critic_architecture="strided_cnn",
        critic_num_ants=2,
        critic_obs_height=50,
        critic_obs_width=50,
    )

    value = get_value(
        params,
        central_obs,
        critic_architecture="strided_cnn",
        critic_num_ants=2,
        critic_obs_height=50,
        critic_obs_width=50,
    )
    actions, logprob, entropy, action_value = get_action_and_value(
        params,
        actor_obs,
        central_obs,
        jax.random.PRNGKey(1),
        critic_architecture="strided_cnn",
        critic_num_ants=2,
        critic_obs_height=50,
        critic_obs_width=50,
    )

    assert params.critic_body.conv_5x5.kernel.shape == (5, 5, 4, 32)
    assert params.critic_body.conv_3x3_a.kernel.shape == (3, 3, 32, 64)
    assert params.critic_body.conv_3x3_b.kernel.shape == (3, 3, 64, 64)
    assert params.critic_body.spatial_dense.weight.shape == (7 * 7 * 64, 256)
    assert params.critic_body.entity_dense.weight.shape == (7 * 2 + 4, 128)
    assert params.critic_body.fusion_dense.weight.shape == (384, 256)
    assert params.value_head.weight.shape == (256, 1)
    assert value.shape == (1,)
    assert actions.shape == (1, 2, 2)
    assert logprob.shape == (1, 2)
    assert entropy.shape == (1, 2)
    assert action_value.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(value)))


def test_jax_resnet_cnn_critic_produces_values_from_central_grid() -> None:
    env = JaxAntByteForagingEnv(
        width=8,
        height=8,
        num_ants=2,
        food_count=4,
        random_food=False,
        write_bits=2,
    )
    _, obs, _ = env.reset(
        jax.random.PRNGKey(123),
        hub_pos=jnp.array([3, 3], dtype=jnp.int32),
        food_positions=jnp.array([[6, 6], [1, 6]], dtype=jnp.int32),
    )
    obs = {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}
    central_obs = build_central_observations(obs, food_scale=4, write_bits=2)
    actor_obs = build_actor_observations(obs, food_scale=4, write_bits=2)
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=16,
        write_value_count=write_value_count(2),
        critic_architecture="resnet_cnn",
        critic_num_ants=2,
        critic_obs_height=8,
        critic_obs_width=8,
    )

    value = get_value(
        params,
        central_obs,
        critic_architecture="resnet_cnn",
        critic_num_ants=2,
        critic_obs_height=8,
        critic_obs_width=8,
    )
    actions, logprob, entropy, action_value = get_action_and_value(
        params,
        actor_obs,
        central_obs,
        jax.random.PRNGKey(1),
        critic_architecture="resnet_cnn",
        critic_num_ants=2,
        critic_obs_height=8,
        critic_obs_width=8,
    )

    assert params.critic_body.stem.kernel.shape == (3, 3, 4, 32)
    assert params.critic_body.spatial_dense.weight.shape == (256, 256)
    assert params.value_head.weight.shape == (256, 1)
    assert value.shape == (1,)
    assert actions.shape == (1, 2, 2)
    assert logprob.shape == (1, 2)
    assert entropy.shape == (1, 2)
    assert action_value.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(value)))


@pytest.mark.parametrize("critic_architecture", ["resnet_cnn", "strided_cnn"])
def test_jax_collect_rollout_supports_grid_cnn_critics(
    critic_architecture: str,
) -> None:
    args = _rollout_args(
        [
            "--critic-architecture",
            critic_architecture,
            "--width",
            "8",
            "--height",
            "8",
            "--food-count",
            "2",
            "--food-sources",
            "1",
            "--num-ants",
            "2",
            "--write-bits",
            "2",
            "--hidden-size",
            "8",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)

    _, _, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(202),
    )

    assert rollout.values.shape == (1, 1)
    assert rollout.next_values.shape == (1, 1)
    assert bool(jnp.all(jnp.isfinite(rollout.values)))
    assert bool(jnp.all(jnp.isfinite(rollout.next_values)))


def test_jax_checkpoint_load_rejects_critic_architecture_mismatch(tmp_path) -> None:
    args = _rollout_args()
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, _, obs = _params_for_args(args, env)
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
    checkpoint_path = tmp_path / "model.pkl"
    save_checkpoint(
        checkpoint_path,
        params=params,
        opt_state=init_adam_state(params),
        args=args,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        run_name="test",
        metrics={},
    )

    with pytest.raises(ValueError, match="critic architecture"):
        load_checkpoint_for_training(
            checkpoint_path,
            central_obs_dim=int(central_obs.shape[-1]),
            actor_obs_dim=int(actor_obs.shape[-1]),
            target_write_bits=args.write_bits,
            actor_vision_radius=args.actor_vision_radius,
            target_num_ants=args.num_ants,
            target_critic_architecture="resnet_cnn",
        )


def test_jax_actor_only_warm_start_keeps_target_critic(tmp_path) -> None:
    source_args = _rollout_args(["--critic-architecture", "strided_cnn"])
    source_env = JaxAntByteForagingEnv(
        width=source_args.width,
        height=source_args.height,
        num_ants=source_args.num_ants,
        food_count=source_args.food_count,
        food_source_count=source_args.food_sources,
        max_steps=source_args.max_steps,
        random_food=source_args.random_food,
        write_bits=source_args.write_bits,
    )
    source_params, _, source_obs = _params_for_args(source_args, source_env)
    central_obs = build_central_observations(
        source_obs,
        food_scale=source_args.food_count,
        write_bits=source_args.write_bits,
        obs_width=source_args.obs_width,
        obs_height=source_args.obs_height,
    )
    actor_obs = build_actor_observations(
        source_obs,
        food_scale=source_args.food_count,
        actor_vision_radius=source_args.actor_vision_radius,
        write_bits=source_args.write_bits,
        obs_width=source_args.obs_width,
        obs_height=source_args.obs_height,
    )
    checkpoint_path = tmp_path / "strided_actor_source.pkl"
    save_checkpoint(
        checkpoint_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=source_args,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        run_name="source",
        metrics={},
    )

    target_args = _rollout_args(["--critic-architecture", "mlp"])
    target_params = init_agent_params(
        jax.random.PRNGKey(99),
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        hidden_size=target_args.hidden_size,
        write_value_count=write_value_count(target_args.write_bits),
        critic_architecture="mlp",
    )
    warmed = warm_start_actor_params(
        target_params,
        checkpoint_path,
        actor_obs_dim=int(actor_obs.shape[-1]),
        target_write_bits=target_args.write_bits,
        actor_vision_radius=target_args.actor_vision_radius,
        target_num_ants=target_args.num_ants,
    )

    for warmed_leaf, source_leaf in zip(
        jax.tree_util.tree_leaves(warmed.actor_body),
        jax.tree_util.tree_leaves(source_params.actor_body),
    ):
        np.testing.assert_allclose(np.asarray(warmed_leaf), np.asarray(source_leaf))
    np.testing.assert_allclose(
        np.asarray(warmed.move_head.weight),
        np.asarray(source_params.move_head.weight),
    )
    np.testing.assert_allclose(
        np.asarray(warmed.write_head.weight),
        np.asarray(source_params.write_head.weight),
    )
    for warmed_leaf, target_leaf in zip(
        jax.tree_util.tree_leaves(warmed.critic_body),
        jax.tree_util.tree_leaves(target_params.critic_body),
    ):
        np.testing.assert_allclose(np.asarray(warmed_leaf), np.asarray(target_leaf))
    np.testing.assert_allclose(
        np.asarray(warmed.value_head.weight),
        np.asarray(target_params.value_head.weight),
    )


def test_jax_checkpoint_transfer_expands_write_bits(tmp_path) -> None:
    source_bits = 1
    target_bits = 3
    radius = 1
    hidden_size = 8
    central_obs_dim = 12
    source_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=source_bits,
        actor_vision_radius=radius,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=radius,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(source_bits),
    )
    source_path = tmp_path / "one_bit.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=source_bits,
            actor_vision_radius=radius,
            save_model=source_path,
        ),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        run_name="one_bit",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=target_bits,
        actor_vision_radius=radius,
    )

    transferred = checkpoint["params"]
    patch_size = actor_vision_patch_size(radius)
    assert transferred.actor_body[0].weight.shape == (target_actor_obs_dim, hidden_size)
    assert transferred.write_head.weight.shape[-1] == write_value_count(target_bits)
    source_hub = slice(patch_size * (2 + source_bits), patch_size * (3 + source_bits))
    target_hub = slice(patch_size * (2 + target_bits), patch_size * (3 + target_bits))
    np.testing.assert_allclose(
        np.asarray(transferred.actor_body[0].weight[target_hub]),
        np.asarray(source_params.actor_body[0].weight[source_hub]),
    )
    np.testing.assert_array_equal(
        np.asarray(repeated_write_action_indices(source_bits, target_bits)),
        np.arange(write_value_count(target_bits), dtype=np.int64)
        % write_value_count(source_bits),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.write_head.weight),
        np.asarray(
            source_params.write_head.weight[
                :,
                repeated_write_action_indices(source_bits, target_bits),
            ]
        ),
    )
    assert checkpoint["opt_state"].count.shape == ()


def test_jax_checkpoint_transfer_expands_actor_vision_radius(tmp_path) -> None:
    write_bits = 2
    source_radius = 1
    target_radius = 2
    hidden_size = 8
    central_obs_dim = 12
    source_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=source_radius,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=target_radius,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(write_bits),
    )
    source_path = tmp_path / "radius_one.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=write_bits,
            actor_vision_radius=source_radius,
            save_model=source_path,
        ),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        run_name="radius_one",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=write_bits,
        actor_vision_radius=target_radius,
    )

    transferred = checkpoint["params"]
    source_weight = np.asarray(source_params.actor_body[0].weight)
    target_weight = np.asarray(transferred.actor_body[0].weight)
    source_patch_size = actor_vision_patch_size(source_radius)
    target_patch_size = actor_vision_patch_size(target_radius)
    source_width = 2 * source_radius + 1
    target_width = 2 * target_radius + 1
    channel_count = write_bits + 4

    assert transferred.actor_body[0].weight.shape == (target_actor_obs_dim, hidden_size)
    assert checkpoint["actor_obs_dim"] == target_actor_obs_dim
    assert checkpoint["args"]["actor_vision_radius"] == target_radius
    for channel_index in range(channel_count):
        target_border_mask = np.ones((target_patch_size,), dtype=bool)
        for offset_y in range(-source_radius, source_radius + 1):
            for offset_x in range(-source_radius, source_radius + 1):
                source_index = channel_index * source_patch_size
                source_index += (offset_y + source_radius) * source_width
                source_index += offset_x + source_radius
                target_index = channel_index * target_patch_size
                target_index += (offset_y + target_radius) * target_width
                target_index += offset_x + target_radius
                target_border_mask[target_index - channel_index * target_patch_size] = False
                np.testing.assert_allclose(
                    target_weight[target_index],
                    source_weight[source_index],
                )
        target_channel = target_weight[
            channel_index * target_patch_size : (channel_index + 1) * target_patch_size
        ]
        np.testing.assert_allclose(
            target_channel[target_border_mask],
            np.zeros((target_patch_size - source_patch_size, hidden_size), dtype=np.float32),
        )

    source_tail_start = channel_count * source_patch_size
    target_tail_start = channel_count * target_patch_size
    np.testing.assert_allclose(
        target_weight[target_tail_start : target_tail_start + MOVEMENT_ACTION_COUNT],
        source_weight[source_tail_start : source_tail_start + MOVEMENT_ACTION_COUNT],
    )
    assert checkpoint["opt_state"].count.shape == ()


def test_jax_checkpoint_transfer_can_reset_expanded_write_head(tmp_path) -> None:
    source_bits = 1
    target_bits = 3
    radius = 1
    hidden_size = 8
    central_obs_dim = 12
    source_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=source_bits,
        actor_vision_radius=radius,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=radius,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(source_bits),
    )
    source_params = source_params._replace(
        write_head=LinearParams(
            weight=jnp.ones_like(source_params.write_head.weight),
            bias=jnp.ones_like(source_params.write_head.bias),
        )
    )
    source_path = tmp_path / "one_bit.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=source_bits,
            actor_vision_radius=radius,
            save_model=source_path,
        ),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        run_name="one_bit",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=target_bits,
        actor_vision_radius=radius,
        write_head_transfer="reset",
    )

    transferred = checkpoint["params"]
    np.testing.assert_allclose(
        np.asarray(transferred.write_head.weight),
        np.zeros((hidden_size, write_value_count(target_bits)), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.write_head.bias),
        np.zeros((write_value_count(target_bits),), dtype=np.float32),
    )
    assert checkpoint["args"]["write_head_transfer"] == "reset"


def test_jax_checkpoint_transfer_can_neutral_initialize_new_write_actions(tmp_path) -> None:
    source_bits = 1
    target_bits = 3
    radius = 1
    hidden_size = 8
    central_obs_dim = 12
    source_count = write_value_count(source_bits)
    target_count = write_value_count(target_bits)
    source_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=source_bits,
        actor_vision_radius=radius,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=radius,
    )
    source_write_weight = jnp.arange(hidden_size * source_count, dtype=jnp.float32).reshape(
        hidden_size,
        source_count,
    )
    source_write_bias = jnp.array([2.0, 6.0], dtype=jnp.float32)
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=source_count,
    )._replace(
        write_head=LinearParams(weight=source_write_weight, bias=source_write_bias)
    )
    source_path = tmp_path / "one_bit.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=source_bits,
            actor_vision_radius=radius,
            save_model=source_path,
        ),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        run_name="one_bit",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=target_bits,
        actor_vision_radius=radius,
        write_head_transfer="neutral-new",
    )

    transferred = checkpoint["params"]
    expected_new_weight = np.repeat(
        np.asarray(jnp.mean(source_write_weight, axis=1, keepdims=True)),
        target_count - source_count,
        axis=1,
    )
    np.testing.assert_allclose(
        np.asarray(transferred.write_head.weight[:, :source_count]),
        np.asarray(source_write_weight),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.write_head.bias[:source_count]),
        np.asarray(source_write_bias),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.write_head.weight[:, source_count:]),
        expected_new_weight,
    )
    np.testing.assert_allclose(
        np.asarray(transferred.write_head.bias[source_count:]),
        np.full((target_count - source_count,), float(jnp.mean(source_write_bias))),
    )
    assert checkpoint["args"]["write_head_transfer"] == "neutral-new"


def test_jax_checkpoint_transfer_adds_ants_count_planes_from_legacy_checkpoint(tmp_path) -> None:
    write_bits = 1
    radius = 1
    hidden_size = 8
    num_ants = 2
    obs_height = 3
    obs_width = 4
    legacy_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        include_ants_count=False,
        include_orientation=False,
        include_current_row=False,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        num_ants=num_ants,
    )
    legacy_central_dim = legacy_central_obs_dim(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    target_central_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=legacy_central_dim,
        actor_obs_dim=legacy_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(write_bits),
    )
    source_path = tmp_path / "legacy_without_ant_counts.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=write_bits,
            actor_vision_radius=radius,
            width=obs_width,
            height=obs_height,
            obs_width=obs_width,
            obs_height=obs_height,
            num_ants=num_ants,
            save_model=source_path,
        ),
        central_obs_dim=legacy_central_dim,
        actor_obs_dim=legacy_actor_obs_dim,
        run_name="legacy",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=target_central_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=write_bits,
        actor_vision_radius=radius,
        target_num_ants=num_ants,
    )

    transferred = checkpoint["params"]
    patch_size = actor_vision_patch_size(radius)
    grid_area = obs_height * obs_width
    central_prefix = 3 * num_ants
    central_orientation_width = 4 * num_ants
    assert checkpoint["central_obs_dim"] == target_central_dim
    assert checkpoint["actor_obs_dim"] == target_actor_obs_dim
    assert transferred.critic_body[0].weight.shape == (target_central_dim, hidden_size)
    assert transferred.actor_body[0].weight.shape == (target_actor_obs_dim, hidden_size)
    np.testing.assert_allclose(
        np.asarray(
            transferred.critic_body[0].weight[
                central_prefix : central_prefix + central_orientation_width
            ]
        ),
        np.zeros((central_orientation_width, hidden_size), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(
            transferred.critic_body[0].weight[
                central_prefix
                + central_orientation_width : central_prefix
                + central_orientation_width
                + grid_area
            ]
        ),
        np.zeros((grid_area, hidden_size), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.actor_body[0].weight[patch_size : 2 * patch_size]),
        np.zeros((patch_size, hidden_size), dtype=np.float32),
    )
    identity_start = patch_size * (write_bits + 4)
    np.testing.assert_allclose(
        np.asarray(transferred.actor_body[0].weight[identity_start : identity_start + num_ants]),
        np.zeros((num_ants, hidden_size), dtype=np.float32),
    )
    source_indices = np.array([0, 1, 3, 4, 6, 7])
    target_indices = np.array([2, 5, 8])
    np.testing.assert_allclose(
        np.asarray(transferred.actor_body[0].weight[source_indices]),
        np.zeros((source_indices.shape[0], hidden_size), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.actor_body[0].weight[target_indices]),
        np.asarray(source_params.actor_body[0].weight[: target_indices.shape[0]]),
    )


def test_jax_evaluate_checkpoint_uses_transfer_adapter_for_legacy_shapes(tmp_path) -> None:
    write_bits = 1
    radius = 1
    hidden_size = 8
    num_ants = 2
    obs_height = 3
    obs_width = 4
    legacy_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        include_ants_count=False,
        include_orientation=False,
        include_current_row=False,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        num_ants=num_ants,
    )
    assert legacy_actor_obs_dim != target_actor_obs_dim
    legacy_central_dim = legacy_central_obs_dim(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=legacy_central_dim,
        actor_obs_dim=legacy_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(write_bits),
    )
    source_path = tmp_path / "legacy_eval.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=write_bits,
            actor_vision_radius=radius,
            width=obs_width,
            height=obs_height,
            obs_width=obs_width,
            obs_height=obs_height,
            num_ants=num_ants,
            food_count=1,
            food_sources=1,
            max_steps=2,
            random_food=False,
            step_penalty=0.0,
            write_penalty=0.0,
            seed=3,
            cookie_distance=1,
            save_model=source_path,
        ),
        central_obs_dim=legacy_central_dim,
        actor_obs_dim=legacy_actor_obs_dim,
        run_name="legacy_eval",
        metrics={},
    )

    metrics = evaluate_checkpoint(source_path, num_episodes=1)

    assert set(metrics) >= {"eval_success_rate", "eval_mean_delivered_fraction"}


def test_jax_checkpoint_transfer_can_increase_ant_count_for_critic(tmp_path) -> None:
    write_bits = 1
    radius = 1
    hidden_size = 8
    source_num_ants = 4
    target_num_ants = 8
    obs_height = 25
    obs_width = 25
    source_central_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    target_central_dim = central_obs_dim_with_ants_count(
        num_ants=target_num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        num_ants=source_num_ants,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        num_ants=target_num_ants,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=source_central_dim,
        actor_obs_dim=actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(write_bits),
    )
    source_path = tmp_path / "four_ant.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=write_bits,
            actor_vision_radius=radius,
            width=obs_width,
            height=obs_height,
            obs_width=obs_width,
            obs_height=obs_height,
            num_ants=source_num_ants,
            food_count=23,
            food_sources=6,
            max_steps=2500,
            save_model=source_path,
        ),
        central_obs_dim=source_central_dim,
        actor_obs_dim=actor_obs_dim,
        run_name="four_ant",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=target_central_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=write_bits,
        actor_vision_radius=radius,
        target_num_ants=target_num_ants,
    )

    transferred = checkpoint["params"]
    source_weight = np.asarray(source_params.critic_body[0].weight)
    target_weight = np.asarray(transferred.critic_body[0].weight)
    source_actor_weight = np.asarray(source_params.actor_body[0].weight)
    target_actor_weight = np.asarray(transferred.actor_body[0].weight)
    grid_area = obs_height * obs_width
    source_maps_start = 7 * source_num_ants
    target_maps_start = 7 * target_num_ants
    identity_start = actor_vision_patch_size(radius) * (write_bits + 4)
    np.testing.assert_allclose(target_weight[: 2 * source_num_ants], source_weight[:8])
    np.testing.assert_allclose(
        target_weight[2 * target_num_ants : 2 * target_num_ants + source_num_ants],
        source_weight[2 * source_num_ants : 3 * source_num_ants],
    )
    np.testing.assert_allclose(
        target_weight[target_maps_start : target_maps_start + 3 * grid_area],
        source_weight[source_maps_start : source_maps_start + 3 * grid_area],
    )
    np.testing.assert_allclose(
        target_actor_weight[identity_start : identity_start + source_num_ants],
        source_actor_weight[identity_start : identity_start + source_num_ants],
    )
    np.testing.assert_allclose(
        target_actor_weight[identity_start + source_num_ants : identity_start + target_num_ants],
        np.zeros((target_num_ants - source_num_ants, hidden_size), dtype=np.float32),
    )
    np.testing.assert_allclose(
        target_weight[2 * source_num_ants : 2 * target_num_ants],
        np.zeros((2 * (target_num_ants - source_num_ants), hidden_size), dtype=np.float32),
    )
    assert checkpoint["central_obs_dim"] == target_central_dim
    assert checkpoint["actor_obs_dim"] == target_actor_obs_dim


def test_jax_strided_cnn_checkpoint_transfer_can_increase_ant_count(tmp_path) -> None:
    write_bits = 4
    radius = 1
    hidden_size = 8
    source_num_ants = 4
    target_num_ants = 8
    obs_height = 8
    obs_width = 8
    source_central_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    target_central_dim = central_obs_dim_with_ants_count(
        num_ants=target_num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    source_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        num_ants=source_num_ants,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        num_ants=target_num_ants,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=source_central_dim,
        actor_obs_dim=source_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(write_bits),
        critic_architecture="strided_cnn",
        critic_num_ants=source_num_ants,
        critic_obs_height=obs_height,
        critic_obs_width=obs_width,
    )
    source_path = tmp_path / "four_ant_strided.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=write_bits,
            actor_vision_radius=radius,
            width=obs_width,
            height=obs_height,
            obs_width=obs_width,
            obs_height=obs_height,
            num_ants=source_num_ants,
            critic_architecture="strided_cnn",
            save_model=source_path,
        ),
        central_obs_dim=source_central_dim,
        actor_obs_dim=source_actor_obs_dim,
        run_name="four_ant_strided",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=target_central_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=write_bits,
        actor_vision_radius=radius,
        target_num_ants=target_num_ants,
        target_critic_architecture="strided_cnn",
    )

    transferred = checkpoint["params"]
    source_weight = np.asarray(source_params.critic_body.entity_dense.weight)
    target_weight = np.asarray(transferred.critic_body.entity_dense.weight)
    source_actor_weight = np.asarray(source_params.actor_body[0].weight)
    target_actor_weight = np.asarray(transferred.actor_body[0].weight)
    identity_start = actor_vision_patch_size(radius) * (write_bits + 4)

    assert target_weight.shape == (7 * target_num_ants + 4, 128)
    np.testing.assert_allclose(target_weight[:8], source_weight[:8])
    np.testing.assert_allclose(target_weight[16:20], source_weight[8:12])
    np.testing.assert_allclose(target_weight[24:40], source_weight[12:28])
    np.testing.assert_allclose(target_weight[56:], source_weight[28:])
    np.testing.assert_allclose(target_weight[8:16], np.zeros((8, 128), dtype=np.float32))
    np.testing.assert_allclose(target_weight[20:24], np.zeros((4, 128), dtype=np.float32))
    np.testing.assert_allclose(target_weight[40:56], np.zeros((16, 128), dtype=np.float32))
    np.testing.assert_allclose(
        target_actor_weight[identity_start : identity_start + source_num_ants],
        source_actor_weight[identity_start : identity_start + source_num_ants],
    )
    np.testing.assert_allclose(
        target_actor_weight[
            identity_start + source_num_ants : identity_start + target_num_ants
        ],
        np.zeros((target_num_ants - source_num_ants, hidden_size), dtype=np.float32),
    )
    assert checkpoint["central_obs_dim"] == target_central_dim
    assert checkpoint["actor_obs_dim"] == target_actor_obs_dim


def test_jax_communication_probe_writes_checkpoint_schema(tmp_path) -> None:
    args = parse_args(
        [
            "--total-timesteps",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "3",
            "--height",
            "3",
            "--obs-width",
            "3",
            "--obs-height",
            "3",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "2",
            "--write-bits",
            "2",
            "--hidden-size",
            "8",
            "--seed",
            "5",
            "--quiet",
            "--random-food",
            "--random-hub",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        write_bits=args.write_bits,
    )
    _, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
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
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )
    checkpoint_path = tmp_path / "model.pkl"
    save_checkpoint(
        checkpoint_path,
        params=params,
        opt_state=init_adam_state(params),
        args=args,
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        run_name="probe",
        metrics={},
    )

    payload = probe_communication_checkpoint(
        checkpoint_path,
        output_dir=tmp_path / "probe",
        num_episodes=1,
        render_rollouts=False,
    )

    report_path = tmp_path / "probe" / "communication_probe.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["probe_path"] == str(report_path)
    assert report["write_bits"] == 2
    assert report["rollout_artifact_paths"] == {"deterministic": None, "sampled": None}
    for mode in ("sampled", "deterministic"):
        section = report[mode]
        assert set(section) >= {
            "write_action_histogram",
            "final_grid_byte_histogram",
            "per_bit_activation_rates",
            "write_bit_entropy",
            "distinct_nonzero_values",
            "major_nonzero_values",
            "major_nonzero_value_fractions",
            "delivery_metrics",
            "rollout_artifact_path",
        }
        assert len(section["per_bit_activation_rates"]) == args.write_bits
        assert set(section["delivery_metrics"]) >= {
            "success_rate",
            "mean_delivered_food",
            "mean_delivered_fraction",
        }


def test_write_action_bit_summary_reports_major_nonzero_values() -> None:
    summary = write_action_bit_summary(
        np.array([10, 50, 40, 4, 6], dtype=np.int64),
        write_bits=3,
    )

    assert summary["nonzero_write_count"] == 100
    assert summary["distinct_nonzero_values"] == [1, 2, 3, 4]
    assert summary["major_nonzero_values"] == [1, 2, 4]
    assert summary["major_nonzero_value_fractions"] == {
        "1": 0.5,
        "2": 0.4,
        "4": 0.06,
    }


def test_jax_communication_probe_counts_moving_writes_when_enabled() -> None:
    actions = np.array(
        [
            [ACTION_RIGHT, 7],
            [ACTION_STAY, 1],
            [ACTION_LEFT, 2],
        ],
        dtype=np.int64,
    )

    stay_only_values = _applied_probe_write_values(actions, write_while_moving=False)
    moving_values = _applied_probe_write_values(actions, write_while_moving=True)

    np.testing.assert_array_equal(stay_only_values, np.array([0, 1, 0]))
    np.testing.assert_array_equal(moving_values, np.array([7, 1, 2]))


def test_jax_communication_probe_rejects_empty_render_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_render_frames"):
        probe_communication_checkpoint(
            tmp_path / "missing.pkl",
            output_dir=tmp_path / "probe",
            max_render_frames=0,
        )


def test_jax_gae_respects_done_boundaries() -> None:
    rewards = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    values = jnp.zeros((2, 1), dtype=jnp.float32)
    dones = jnp.array([[False], [True]])
    next_value = jnp.zeros((1,), dtype=jnp.float32)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(np.asarray(advantages), np.array([[3.0], [2.0]]))
    np.testing.assert_allclose(np.asarray(returns), np.array([[3.0], [2.0]]))


def test_jax_gae_bootstraps_time_limit_truncations_without_cross_episode_leak() -> None:
    rewards = jnp.array([[1.0], [10.0]], dtype=jnp.float32)
    values = jnp.array([[0.5], [7.0]], dtype=jnp.float32)
    next_values = jnp.array([[4.0], [0.0]], dtype=jnp.float32)
    dones = jnp.array([[True], [False]])
    terminations = jnp.array([[False], [False]])

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        terminations=terminations,
        next_values=next_values,
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(np.asarray(advantages), np.array([[4.5], [3.0]]))
    np.testing.assert_allclose(np.asarray(returns), np.array([[5.0], [10.0]]))


def test_jax_write_bit_penalty_makes_lower_bits_more_expensive() -> None:
    actions = jnp.array(
        [
            [[ACTION_RIGHT, 7], [ACTION_STAY, 1]],
            [[ACTION_STAY, 2], [ACTION_STAY, 7]],
        ],
        dtype=jnp.int32,
    )

    penalties = compute_write_bit_penalties(
        actions,
        write_bits=3,
        base_penalty=0.01,
        decay=0.5,
    )

    np.testing.assert_allclose(np.asarray(penalties), np.array([0.01, 0.0225]))


def test_jax_write_bit_penalty_can_count_moving_writes() -> None:
    actions = jnp.array(
        [
            [[ACTION_RIGHT, 7], [ACTION_STAY, 1]],
            [[ACTION_STAY, 2], [ACTION_STAY, 7]],
        ],
        dtype=jnp.int32,
    )

    penalties = compute_write_bit_penalties(
        actions,
        write_bits=3,
        base_penalty=0.01,
        decay=0.5,
        write_while_moving=True,
    )

    np.testing.assert_allclose(np.asarray(penalties), np.array([0.0275, 0.0225]))


def test_jax_write_bit_penalty_masks_per_ant_write_channels() -> None:
    actions = jnp.array(
        [
            [[ACTION_STAY, 1], [ACTION_STAY, 1], [ACTION_STAY, 1]],
            [[ACTION_STAY, 7], [ACTION_STAY, 7], [ACTION_STAY, 7]],
        ],
        dtype=jnp.int32,
    )

    penalties = compute_write_bit_penalties(
        actions,
        write_bits=3,
        base_penalty=0.01,
        decay=0.5,
        per_ant_write_channels=True,
    )

    np.testing.assert_allclose(np.asarray(penalties), np.array([0.01, 0.0175]))


def test_jax_write_bit_penalty_masks_repeated_per_ant_channel_types() -> None:
    actions = jnp.array(
        [
            [
                [ACTION_STAY, 3],
                [ACTION_STAY, 3],
                [ACTION_STAY, 3],
                [ACTION_STAY, 3],
            ],
        ],
        dtype=jnp.int32,
    )

    penalties = compute_write_bit_penalties(
        actions,
        write_bits=2,
        base_penalty=0.01,
        decay=0.5,
        per_ant_write_channels=True,
    )

    np.testing.assert_allclose(np.asarray(penalties), np.array([0.03]))


def test_jax_terminal_write_entropy_bonus_rewards_balanced_nonzero_values() -> None:
    next_obs = {
        "bytes": jnp.array(
            [
                [[1, 1], [2, 2]],
                [[1, 2], [3, 7]],
                [[1, 2], [3, 7]],
                [[0, 0], [0, 0]],
            ],
            dtype=jnp.uint8,
        )
    }

    bonuses = compute_terminal_write_entropy_bonus(
        next_obs,
        jnp.array([True, False, True, True]),
        write_bits=3,
        entropy_scale=0.1,
        max_bonus=0.05,
    )

    expected_two_value_entropy = 0.1 * np.log(2.0) / np.log(7.0)
    np.testing.assert_allclose(
        np.asarray(bonuses),
        np.array([expected_two_value_entropy, 0.0, 0.05, 0.0]),
        rtol=1e-6,
    )


def test_jax_write_bit_entropy_bonus_is_zero_for_empty_or_collapsed_writes() -> None:
    empty_actions = jnp.array(
        [
            [[[ACTION_RIGHT, 1]]],
            [[[ACTION_RIGHT, 2]]],
        ],
        dtype=jnp.int32,
    )
    collapsed_actions = jnp.array(
        [
            [[[ACTION_STAY, 1]]],
            [[[ACTION_STAY, 1]]],
            [[[ACTION_STAY, 1]]],
        ],
        dtype=jnp.int32,
    )

    empty_bonus = compute_write_bit_entropy_bonus(
        empty_actions,
        write_bits=3,
        entropy_scale=0.5,
    )
    collapsed_bonus = compute_write_bit_entropy_bonus(
        collapsed_actions,
        write_bits=3,
        entropy_scale=0.5,
    )

    np.testing.assert_allclose(np.asarray(empty_bonus), np.zeros((2, 1)))
    np.testing.assert_allclose(np.asarray(collapsed_bonus), np.zeros((3, 1)))


def test_jax_write_bit_entropy_bonus_rewards_balanced_bits_and_preserves_total_scale() -> None:
    actions = jnp.array(
        [
            [[[ACTION_STAY, 1]]],
            [[[ACTION_STAY, 2]]],
            [[[ACTION_STAY, 1]]],
            [[[ACTION_STAY, 2]]],
        ],
        dtype=jnp.int32,
    )

    bonus = compute_write_bit_entropy_bonus(
        actions,
        write_bits=2,
        entropy_scale=0.5,
    )

    np.testing.assert_allclose(np.asarray(bonus), np.full((4, 1), 0.125), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(bonus.sum(axis=0)), np.array([0.5]), rtol=1e-6)


def test_jax_training_batch_shuffle_depends_on_update_key() -> None:
    batch = TrainingBatch(
        actor_obs=jnp.arange(4 * 1 * 1, dtype=jnp.float32).reshape(4, 1, 1),
        central_obs=jnp.arange(4 * 2, dtype=jnp.float32).reshape(4, 2),
        actions=jnp.arange(4 * 1 * 2, dtype=jnp.int32).reshape(4, 1, 2),
        agent_masks=jnp.ones((4, 1), dtype=jnp.float32),
        old_logprobs=jnp.arange(4, dtype=jnp.float32).reshape(4, 1),
        advantages=jnp.arange(4, dtype=jnp.float32),
        returns=jnp.arange(4, dtype=jnp.float32),
    )

    first = _shuffle_batch(batch, key=jax.random.PRNGKey(1))
    second = _shuffle_batch(batch, key=jax.random.PRNGKey(2))

    assert sorted(np.asarray(first.advantages).tolist()) == [0.0, 1.0, 2.0, 3.0]
    assert sorted(np.asarray(second.advantages).tolist()) == [0.0, 1.0, 2.0, 3.0]
    assert not np.array_equal(np.asarray(first.advantages), np.asarray(second.advantages))


def test_jax_legacy_four_action_movement_head_fails_loudly() -> None:
    layer = LinearParams(
        weight=jnp.zeros((8, 4), dtype=jnp.float32),
        bias=jnp.zeros((4,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="cannot be automatically mapped"):
        adapt_movement_head_layer(layer)


def test_jax_rollout_carries_unfinished_state_between_calls() -> None:
    args = _rollout_args()
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)

    states, obs, first_rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(1),
    )
    states, obs, second_rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(2),
    )

    assert not bool(np.asarray(first_rollout.dones)[0, 0])
    assert not bool(np.asarray(second_rollout.dones)[0, 0])
    assert int(np.asarray(states.step_count)[0]) == 2


def test_jax_rollout_can_mix_sampled_and_deterministic_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _rollout_args(["--deterministic-rollout-fraction", "0.5"])
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)
    observed_determinism: list[bool] = []

    import ant_byte_env.training.jax_mappo.rollout as rollout_module

    original_get_action_and_value = rollout_module.get_action_and_value

    def recording_get_action_and_value(*call_args, deterministic: bool, **kwargs):
        observed_determinism.append(bool(deterministic))
        return original_get_action_and_value(
            *call_args,
            deterministic=deterministic,
            **kwargs,
        )

    monkeypatch.setattr(
        rollout_module,
        "get_action_and_value",
        recording_get_action_and_value,
    )

    collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(3),
    )

    assert False in observed_determinism
    assert True in observed_determinism


def test_jax_rollout_passes_training_temperature_to_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _rollout_args(["--training-rollout-temperature", "0.5"])
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)
    observed_temperatures: list[float] = []

    import ant_byte_env.training.jax_mappo.rollout as rollout_module

    original_get_action_and_value = rollout_module.get_action_and_value

    def recording_get_action_and_value(
        *call_args,
        deterministic: bool,
        policy_temperature: float = 1.0,
        **kwargs,
    ):
        observed_temperatures.append(float(policy_temperature))
        return original_get_action_and_value(
            *call_args,
            deterministic=deterministic,
            policy_temperature=policy_temperature,
            **kwargs,
        )

    monkeypatch.setattr(
        rollout_module,
        "get_action_and_value",
        recording_get_action_and_value,
    )

    collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(30),
    )

    assert observed_temperatures == [0.5]


def test_jax_rollout_can_force_only_movement_head_to_greedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _rollout_args(["--deterministic-move-rollout-fraction", "1.0"])
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)

    import ant_byte_env.training.jax_mappo.rollout as rollout_module

    def fixed_get_action_and_value(
        _params,
        actor_obs,
        central_obs,
        _key,
        *,
        deterministic: bool,
        policy_temperature: float = 1.0,
    ):
        del policy_temperature
        action_shape = actor_obs.shape[:-1]
        value_shape = central_obs.shape[:-1]
        move_value = 2 if deterministic else 1
        write_value = 0 if deterministic else 1
        actions = jnp.stack(
            [
                jnp.full(action_shape, move_value, dtype=jnp.int32),
                jnp.full(action_shape, write_value, dtype=jnp.int32),
            ],
            axis=-1,
        )
        return (
            actions,
            jnp.zeros(action_shape, dtype=jnp.float32),
            jnp.zeros(action_shape, dtype=jnp.float32),
            jnp.zeros(value_shape, dtype=jnp.float32),
        )

    monkeypatch.setattr(
        rollout_module,
        "get_action_and_value",
        fixed_get_action_and_value,
    )

    _, _, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(31),
    )

    assert bool(jnp.all(rollout.actions[..., 0] == 2))
    assert bool(jnp.all(rollout.actions[..., 1] == 1))


def test_jax_rollout_can_ablate_write_actions_for_memory_probe() -> None:
    args = _rollout_args(["--write-while-moving"])
    args.write_action_ablation = True
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
        write_while_moving=args.write_while_moving,
    )
    params, states, obs = _params_for_args(args, env)

    _, _, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(4),
    )

    assert bool(jnp.all(rollout.actions[..., 1] == 0))
    assert float(jnp.sum(rollout.applied_nonzero_write_actions)) == 0.0


def test_jax_rollout_write_ablation_suppresses_write_shaping_reward() -> None:
    args = _rollout_args(["--carrying-byte-write-bonus", "1.0"])
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
        write_while_moving=args.write_while_moving,
    )
    params, states, _ = _params_for_args(args, env)
    move_bias = jnp.full_like(params.move_head.bias, -1000.0).at[ACTION_STAY].set(1000.0)
    write_bias = jnp.full_like(params.write_head.bias, -1000.0).at[1].set(1000.0)
    params = params._replace(
        move_head=LinearParams(weight=jnp.zeros_like(params.move_head.weight), bias=move_bias),
        write_head=LinearParams(weight=jnp.zeros_like(params.write_head.weight), bias=write_bias),
    )
    ants_pos = jnp.array([[[1, 1]]], dtype=jnp.int32)
    states = states._replace(
        ants_pos=ants_pos,
        ants_count=jax.vmap(env._build_ants_count_grid)(ants_pos),
        ants_carrying=jnp.array([[True]]),
        food=jnp.zeros_like(states.food),
        initial_food=jnp.zeros_like(states.initial_food),
        bytes=jnp.zeros_like(states.bytes),
        hub_pos=jnp.array([[0, 0]], dtype=jnp.int32),
    )
    obs = env.observe(states)

    args.write_action_ablation = False
    _, _, normal_rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(4),
    )
    args.write_action_ablation = True
    _, _, ablated_rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(4),
    )

    normal_actions = np.asarray(normal_rollout.actions)
    ablated_actions = np.asarray(ablated_rollout.actions)
    assert int(normal_actions[0, 0, 0, 0]) == ACTION_STAY
    assert int(normal_actions[0, 0, 0, 1]) == 1
    assert int(ablated_actions[0, 0, 0, 1]) == 0
    np.testing.assert_allclose(np.asarray(normal_rollout.rewards), np.array([[1.0]]))
    np.testing.assert_allclose(np.asarray(ablated_rollout.rewards), np.array([[0.0]]))


def test_jax_rollout_auto_resets_completed_envs() -> None:
    args = _rollout_args(
        [
            "--food-count",
            "0",
            "--width",
            "2",
            "--height",
            "1",
            "--cookie-distance",
            "1",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)

    states, _, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(3),
    )

    assert bool(np.asarray(rollout.dones)[0, 0])
    assert int(np.asarray(states.step_count)[0]) == 0


def test_jax_rollout_auto_reset_resamples_random_colony_and_cookie_sources() -> None:
    args = _rollout_args(
        [
            "--num-envs",
            "8",
            "--width",
            "8",
            "--height",
            "8",
            "--food-count",
            "8",
            "--food-sources",
            "2",
            "--max-steps",
            "1",
            "--random-food",
            "--random-hub",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)
    initial_hubs = np.asarray(obs["hub_pos"])
    initial_food = np.asarray(obs["food"])

    states, obs, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(99),
    )

    reset_hubs = np.asarray(obs["hub_pos"])
    reset_food = np.asarray(obs["food"])
    assert bool(np.all(np.asarray(rollout.dones)))
    assert int(np.asarray(states.step_count).max()) == 0
    assert len({tuple(hub) for hub in reset_hubs}) > 1
    assert len({_food_source_signature(food_grid) for food_grid in reset_food}) > 1
    assert not np.array_equal(initial_hubs, reset_hubs)
    assert not np.array_equal(initial_food, reset_food)
    for hub, food_grid in zip(reset_hubs, reset_food, strict=True):
        assert int(food_grid[hub[1], hub[0]]) == 0


def test_jax_rollout_auto_resets_done_envs_inside_scan() -> None:
    args = parse_args(
        [
            "--total-timesteps",
            "3",
            "--num-envs",
            "1",
            "--num-steps",
            "3",
            "--num-minibatches",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--obs-width",
            "5",
            "--obs-height",
            "5",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "2",
            "--cookie-distance",
            "2",
            "--hidden-size",
            "8",
            "--quiet",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        step_penalty=args.step_penalty,
        write_penalty=args.write_penalty,
        write_bits=args.write_bits,
    )
    states, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
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
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )

    final_states, _, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(1),
    )

    np.testing.assert_array_equal(
        np.asarray(rollout.dones[:, 0]),
        np.array([False, True, False]),
    )
    assert int(final_states.step_count[0]) == 1


def test_jax_evaluate_params_reports_delivery_metrics() -> None:
    args = parse_args(
        [
            "--total-timesteps",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--num-minibatches",
            "1",
            "--width",
            "3",
            "--height",
            "1",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--cookie-distance",
            "1",
            "--max-steps",
            "4",
            "--actor-vision-radius",
            "1",
            "--hidden-size",
            "1",
            "--seed",
            "19",
            "--quiet",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, _, obs = _params_for_args(args, env)
    central_obs = build_central_observations(obs, food_scale=args.food_count)
    actor_obs = build_actor_observations(
        obs,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
    )
    params = _scripted_delivery_params(
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
    )

    metrics = evaluate_params(
        params=params,
        args=args,
        num_episodes=3,
        shuffle_positions=False,
    )

    assert metrics["eval_success_rate"] == 1.0
    assert metrics["eval_mean_delivered_food"] == 1.0
    assert metrics["eval_mean_delivered_fraction"] == 1.0
    assert metrics["eval_mean_episode_return"] == 1.0
    assert metrics["eval_mean_episode_length"] == 2.0
    assert metrics["eval_mean_steps_per_delivered_food"] == 2.0
    assert metrics["eval_mean_ant_steps_per_delivered_food"] == 2.0
    assert metrics["eval_mean_delivered_food_per_1000_ant_steps"] == 500.0


def test_jax_evaluate_params_shuffles_colony_and_cookie_sources_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = parse_args(
        [
            "--total-timesteps",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--num-minibatches",
            "1",
            "--width",
            "7",
            "--height",
            "7",
            "--num-ants",
            "1",
            "--food-count",
            "6",
            "--food-sources",
            "2",
            "--max-steps",
            "1",
            "--actor-vision-radius",
            "1",
            "--hidden-size",
            "4",
            "--seed",
            "23",
            "--quiet",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        write_bits=args.write_bits,
    )
    params, _, _ = _params_for_args(args, env)
    observed_hubs: list[tuple[int, int]] = []
    observed_food: list[tuple[tuple[int, int], ...]] = []

    import ant_byte_env.training.jax_mappo.evaluation as jax_evaluation

    original_reset_batch = jax_evaluation.reset_batch

    def recording_reset_batch(
        *,
        args: argparse.Namespace,
        env: JaxAntByteForagingEnv,
        key: jax.Array,
        previous_obs=None,
        previous_food=None,
    ):
        assert args.random_hub is True
        assert args.random_food is True
        states, obs = original_reset_batch(
            args=args,
            env=env,
            key=key,
            previous_obs=previous_obs,
            previous_food=previous_food,
        )
        observed_hubs.append(tuple(int(value) for value in np.asarray(obs["hub_pos"])[0]))
        observed_food.append(_food_source_signature(np.asarray(obs["food"])[0]))
        return states, obs

    monkeypatch.setattr(jax_evaluation, "reset_batch", recording_reset_batch)

    evaluate_params(params=params, args=args, num_episodes=8)

    assert len(set(observed_hubs)) > 1
    assert len(set(observed_food)) > 1
    for hub, food_sources in zip(observed_hubs, observed_food, strict=True):
        assert hub not in food_sources


def test_jax_autocurriculum_evaluation_samples_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = parse_args(
        [
            "--autocurriculum",
            "--total-timesteps",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--num-minibatches",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--obs-width",
            "4",
            "--obs-height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "12",
            "--food-sources",
            "2",
            "--max-steps",
            "1",
            "--actor-vision-radius",
            "1",
            "--hidden-size",
            "4",
            "--seed",
            "31",
            "--quiet",
        ]
    )
    env = JaxAntByteAutoCurriculumEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        start_size=args.autocurriculum_start_size,
        success_cookies=args.autocurriculum_success_cookies,
        actor_vision_radius=args.actor_vision_radius,
        random_food=args.random_food,
        random_hub=args.random_hub,
        write_bits=args.write_bits,
    )
    params, _, _ = _params_for_args(args, env)
    observed_determinism: list[bool] = []

    import ant_byte_env.training.jax_mappo.evaluation as jax_evaluation

    original_get_action_and_value = jax_evaluation.get_action_and_value

    def recording_get_action_and_value(*call_args, deterministic: bool, **kwargs):
        observed_determinism.append(bool(deterministic))
        return original_get_action_and_value(
            *call_args,
            deterministic=deterministic,
            **kwargs,
        )

    monkeypatch.setattr(
        jax_evaluation,
        "get_action_and_value",
        recording_get_action_and_value,
    )

    evaluate_params(params=params, args=args, num_episodes=1)

    assert observed_determinism == [False]


def test_jax_evaluation_can_mix_action_head_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    import ant_byte_env.training.jax_mappo.evaluation as jax_evaluation

    move_logits = jnp.array([[[0.0, 4.0, 1.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0, 0.0]]])
    write_logits = jnp.array([[[0.0, 5.0], [6.0, 0.0]]])

    def fake_get_action_logits(params, actor_obs):
        del params, actor_obs
        return move_logits, write_logits

    monkeypatch.setattr(jax_evaluation, "get_action_logits", fake_get_action_logits)

    actions = jax_evaluation._evaluation_actions_for_mode(
        None,
        jnp.zeros((1, 2, 3), dtype=jnp.float32),
        jnp.zeros((1, 4), dtype=jnp.float32),
        jax.random.PRNGKey(0),
        action_mode="sampled_move_greedy_write",
    )
    zero_write_actions = jax_evaluation._evaluation_actions_for_mode(
        None,
        jnp.zeros((1, 2, 3), dtype=jnp.float32),
        jnp.zeros((1, 4), dtype=jnp.float32),
        jax.random.PRNGKey(0),
        action_mode="greedy_move_zero_write",
    )

    np.testing.assert_array_equal(np.asarray(actions[..., 1]), np.array([[1, 0]]))
    np.testing.assert_array_equal(np.asarray(zero_write_actions[..., 0]), np.array([[1, 0]]))
    np.testing.assert_array_equal(np.asarray(zero_write_actions[..., 1]), np.array([[0, 0]]))


def test_jax_evaluate_params_rejects_nonpositive_action_temperature() -> None:
    args = _rollout_args()
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, _, _ = _params_for_args(args, env)

    with pytest.raises(ValueError, match="move_temperature"):
        evaluate_params(
            params=params,
            args=args,
            num_episodes=1,
            action_mode="sampled_move_greedy_write",
            move_temperature=0.0,
        )


def _scripted_delivery_params(*, central_obs_dim: int, actor_obs_dim: int):
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        hidden_size=1,
        write_value_count=write_value_count(1),
    )
    carrying_index = actor_obs_dim - 5
    actor_input_weight = jnp.zeros_like(params.actor_body[0].weight).at[carrying_index, 0].set(
        10.0
    )
    actor_hidden_weight = jnp.zeros_like(params.actor_body[1].weight).at[0, 0].set(5.0)
    move_weight = jnp.zeros_like(params.move_head.weight).at[0, ACTION_LEFT].set(5.0)
    move_bias = jnp.zeros_like(params.move_head.bias).at[ACTION_RIGHT].set(2.0)
    return params._replace(
        actor_body=(
            LinearParams(
                weight=actor_input_weight,
                bias=jnp.zeros_like(params.actor_body[0].bias),
            ),
            LinearParams(
                weight=actor_hidden_weight,
                bias=jnp.zeros_like(params.actor_body[1].bias),
            ),
        ),
        move_head=LinearParams(weight=move_weight, bias=move_bias),
    )


@pytest.mark.parametrize("write_bits", ["0", "9"])
def test_jax_parse_args_rejects_invalid_write_bits(write_bits: str) -> None:
    with pytest.raises(ValueError, match="write-bits"):
        parse_args(["--write-bits", write_bits])


def test_jax_parse_args_accepts_repeated_write_channel_types() -> None:
    args = parse_args(["--num-ants", "8", "--write-bits", "4", "--per-ant-write-channels"])

    assert args.per_ant_write_channels is True


def test_jax_parse_args_accepts_write_head_transfer_modes() -> None:
    assert parse_args(["--write-head-transfer", "reset"]).write_head_transfer == "reset"
    assert (
        parse_args(["--write-head-transfer", "neutral-new"]).write_head_transfer
        == "neutral-new"
    )

    with pytest.raises(SystemExit):
        parse_args(["--write-head-transfer", "copy-old"])


def test_jax_parse_args_accepts_wandb_flags() -> None:
    args = parse_args(
        [
            "--wandb-project",
            "cool-antz",
            "--wandb-entity",
            "team",
            "--wandb-group",
            "forage",
            "--wandb-run-name",
            "phone-run",
            "--wandb-notes",
            "distance reward test",
            "--wandb-mode",
            "offline",
            "--wandb-tags",
            "jax",
            "50x50",
        ]
    )

    assert args.wandb_project == "cool-antz"
    assert args.wandb_entity == "team"
    assert args.wandb_group == "forage"
    assert args.wandb_run_name == "phone-run"
    assert args.wandb_notes == "distance reward test"
    assert args.wandb_mode == "offline"
    assert args.wandb_tags == ["jax", "50x50"]


def test_jax_wandb_defaults_keep_tracking_disabled() -> None:
    args = parse_args([])

    assert args.wandb_project is None
    assert args.wandb_entity is None
    assert args.wandb_group is None
    assert args.wandb_run_name is None
    assert args.wandb_notes is None
    assert args.wandb_mode == "online"
    assert args.wandb_tags is None


def test_jax_parse_args_accepts_log_interval() -> None:
    args = parse_args(["--log-interval", "10"])

    assert args.log_interval == 10


def test_jax_parse_args_accepts_training_rollout_temperature() -> None:
    args = parse_args(["--training-rollout-temperature", "0.5"])

    assert args.training_rollout_temperature == 0.5


def test_jax_parse_args_accepts_layout_audit_options(tmp_path: Path) -> None:
    audit_dir = tmp_path / "layout_audit"

    args = parse_args(
        [
            "--layout-audit-dir",
            str(audit_dir),
            "--layout-audit-snapshot-interval",
            "30",
        ]
    )

    assert args.layout_audit_dir == audit_dir
    assert args.layout_audit_snapshot_interval == 30


def test_jax_parse_args_accepts_random_ant_spawn() -> None:
    args = parse_args(["--random-ant-spawn"])

    assert args.random_ant_spawn is True


def test_jax_parse_args_accepts_random_ant_spawn_radius() -> None:
    args = parse_args(["--random-ant-spawn", "--random-ant-spawn-radius", "8"])

    assert args.random_ant_spawn is True
    assert args.random_ant_spawn_radius == 8


def test_jax_parse_args_accepts_hub_center_window_size() -> None:
    args = parse_args(["--random-hub", "--hub-center-window-size", "4"])

    assert args.random_hub is True
    assert args.hub_center_window_size == 4


def test_jax_parse_args_accepts_food_cluster_options() -> None:
    args = parse_args(
        [
            "--random-food",
            "--food-count",
            "250",
            "--food-sources",
            "50",
            "--food-cluster-count",
            "2",
            "--food-cluster-radius",
            "5",
        ]
    )

    assert args.food_cluster_count == 2
    assert args.food_cluster_radius == 5


def test_jax_parse_args_accepts_best_model_options(tmp_path: Path) -> None:
    best_path = tmp_path / "best.pkl"

    args = parse_args(
        [
            "--save-best-model",
            str(best_path),
            "--best-model-metric",
            "episode_return",
            "--best-model-mode",
            "max",
        ]
    )

    assert args.save_best_model == best_path
    assert args.best_model_metric == "episode_return"
    assert args.best_model_mode == "max"


def test_jax_parse_args_accepts_best_eval_selection_options(tmp_path: Path) -> None:
    best_path = tmp_path / "best.pkl"

    args = parse_args(
        [
            "--save-best-model",
            str(best_path),
            "--best-model-selection",
            "eval",
            "--best-model-metric",
            "eval_mean_delivered_food",
            "--best-eval-episodes",
            "3",
            "--best-eval-interval",
            "5",
            "--best-eval-seed-offset",
            "12345",
            "--best-eval-action-mode",
            "sampled_move_greedy_write",
            "--best-eval-move-temperature",
            "0.95",
            "--best-eval-write-temperature",
            "1.1",
            "--no-best-eval-shuffle-positions",
        ]
    )

    assert args.best_model_selection == "eval"
    assert args.best_model_metric == "eval_mean_delivered_food"
    assert args.best_eval_episodes == 3
    assert args.best_eval_interval == 5
    assert args.best_eval_seed_offset == 12345
    assert args.best_eval_action_mode == "sampled_move_greedy_write"
    assert args.best_eval_move_temperature == 0.95
    assert args.best_eval_write_temperature == 1.1
    assert args.best_eval_shuffle_positions is False


def test_jax_parse_args_accepts_deterministic_rollout_fraction() -> None:
    args = parse_args(["--deterministic-rollout-fraction", "0.25"])

    assert args.deterministic_rollout_fraction == 0.25


def test_jax_parse_args_accepts_deterministic_move_rollout_fraction() -> None:
    args = parse_args(["--deterministic-move-rollout-fraction", "0.25"])

    assert args.deterministic_move_rollout_fraction == 0.25


def test_jax_parse_args_accepts_write_action_ablation() -> None:
    args = parse_args(["--write-action-ablation"])

    assert args.write_action_ablation is True


def test_jax_parse_args_accepts_full_coverage_termination() -> None:
    args = parse_args(["--terminate-on-full-coverage"])

    assert args.terminate_on_full_coverage is True


def test_jax_parse_args_accepts_decaying_visit_reward() -> None:
    args = parse_args(["--visit-reward-scale", "0.02", "--visit-reward-decay", "1.5"])

    assert args.visit_reward_scale == 0.02
    assert args.visit_reward_decay == 1.5


@pytest.mark.parametrize("value", ["-0.01", "1.01"])
def test_jax_parse_args_rejects_invalid_deterministic_rollout_fraction(value: str) -> None:
    with pytest.raises(ValueError, match="deterministic-rollout-fraction"):
        parse_args(["--deterministic-rollout-fraction", value])


@pytest.mark.parametrize("value", ["0.0", "-0.01"])
def test_jax_parse_args_rejects_invalid_training_rollout_temperature(value: str) -> None:
    with pytest.raises(ValueError, match="training-rollout-temperature"):
        parse_args(["--training-rollout-temperature", value])


def test_jax_parse_args_rejects_layout_audit_snapshot_without_dir() -> None:
    with pytest.raises(ValueError, match="layout-audit-snapshot-interval"):
        parse_args(["--layout-audit-snapshot-interval", "30"])


def test_jax_parse_args_rejects_negative_layout_audit_snapshot_interval() -> None:
    with pytest.raises(ValueError, match="layout-audit-snapshot-interval"):
        parse_args(["--layout-audit-snapshot-interval", "-1"])


def test_jax_parse_args_rejects_negative_border_view_penalty() -> None:
    with pytest.raises(ValueError, match="border-view-penalty"):
        parse_args(["--border-view-penalty", "-0.001"])


def test_jax_parse_args_rejects_negative_border_moat_options() -> None:
    with pytest.raises(ValueError, match="border-moat-width"):
        parse_args(["--border-moat-width", "-1"])
    with pytest.raises(ValueError, match="border-moat-penalty"):
        parse_args(["--border-moat-penalty", "-0.001"])


@pytest.mark.parametrize("value", ["-0.01", "1.01"])
def test_jax_parse_args_rejects_invalid_deterministic_move_rollout_fraction(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="deterministic-move-rollout-fraction"):
        parse_args(["--deterministic-move-rollout-fraction", value])


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--log-interval", "0", "log-interval"),
        ("--write-bit-penalty", "-0.1", "write-bit-penalty"),
        ("--write-bit-penalty-decay", "1.5", "write-bit-penalty-decay"),
        ("--write-entropy-bonus", "-0.1", "write-entropy-bonus"),
        ("--write-entropy-bonus-cap", "-0.1", "write-entropy-bonus-cap"),
        ("--write-bit-entropy-bonus", "-0.1", "write-bit-entropy-bonus"),
        ("--distance-bonus", "-0.1", "distance-bonus"),
        ("--carrying-hub-distance-bonus", "-0.1", "carrying-hub-distance-bonus"),
        ("--visit-reward-scale", "-0.1", "visit-reward-scale"),
        ("--visit-reward-decay", "-0.1", "visit-reward-decay"),
        ("--stage-completion-bonus", "-0.1", "stage-completion-bonus"),
        ("--delivery-byte-trail-bonus", "-0.1", "delivery-byte-trail-bonus"),
        ("--delivery-byte-trail-target-tiles", "0", "delivery-byte-trail-target-tiles"),
        ("--byte-follow-bonus", "-0.1", "byte-follow-bonus"),
        ("--carrying-byte-write-bonus", "-0.1", "carrying-byte-write-bonus"),
        ("--best-eval-episodes", "0", "best-eval-episodes"),
        ("--best-eval-interval", "-1", "best-eval-interval"),
        ("--best-eval-seed-offset", "-1", "best-eval-seed-offset"),
        ("--best-eval-move-temperature", "0", "best-eval-move-temperature"),
        ("--best-eval-write-temperature", "0", "best-eval-write-temperature"),
    ],
)
def test_jax_parse_args_rejects_invalid_write_bit_penalty_options(
    flag: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_args([flag, value])


def test_tiny_jax_mappo_training_run_completes() -> None:
    progress_updates = []

    metrics = main(
        [
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "8",
            "--hidden-size",
            "16",
            "--seed",
            "7",
            "--write-bit-entropy-bonus",
            "0.25",
            "--quiet",
        ],
        progress_callback=lambda update, total, row: progress_updates.append(
            (update, total, row["global_step"])
        ),
    )

    assert metrics["global_step"] == 8
    assert progress_updates == [(1, 2, 4.0), (2, 2, 8.0)]
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])


def test_tiny_jax_mappo_training_respects_log_interval() -> None:
    progress_updates = []

    metrics = main(
        [
            "--total-timesteps",
            "20",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "8",
            "--hidden-size",
            "16",
            "--seed",
            "7",
            "--log-interval",
            "3",
            "--quiet",
        ],
        progress_callback=lambda update, total, row: progress_updates.append(
            (update, total, row["global_step"])
        ),
    )

    assert metrics["global_step"] == 20
    assert progress_updates == [(1, 5, 4.0), (3, 5, 12.0), (5, 5, 20.0)]


def test_tiny_jax_mappo_training_logs_wandb_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.logs: list[tuple[dict[str, object], int | None]] = []
            self.artifacts: list[object] = []
            self.finished = False

        def log(self, payload: dict[str, object], *, step: int | None = None) -> None:
            self.logs.append((payload, step))

        def log_artifact(self, artifact: object, *, aliases=None) -> None:
            del aliases
            self.artifacts.append(artifact)

        def finish(self) -> None:
            self.finished = True

    class FakeArtifact:
        def __init__(self, name: str, *, type: str) -> None:
            self.name = name
            self.type = type
            self.files: list[str] = []

        def add_file(self, path: str) -> None:
            self.files.append(path)

    fake_run = FakeRun()
    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: fake_run,
        Video=lambda *args, **kwargs: object(),
        Artifact=FakeArtifact,
    )
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "wandb":
            return fake_wandb
        return original_import_module(name)

    monkeypatch.setattr(
        "ant_byte_env.wandb_tracking.importlib.import_module",
        fake_import_module,
    )

    metrics = main(
        [
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "8",
            "--hidden-size",
            "16",
            "--seed",
            "7",
            "--wandb-project",
            "cool-antz",
            "--wandb-mode",
            "offline",
            "--wandb-notes",
            "tiny wandb artifact smoke",
            "--save-model",
            str(tmp_path / "model.pkl"),
            "--quiet",
        ],
    )

    assert metrics["global_step"] == 8
    assert [step for _, step in fake_run.logs] == [4, 8]
    assert fake_run.logs[-1][0]["global_step"] == 8.0
    assert fake_run.artifacts
    assert fake_run.finished is True


def test_tiny_jax_mappo_training_saves_best_checkpoint(tmp_path: Path) -> None:
    final_path = tmp_path / "model.pkl"
    best_path = tmp_path / "model_best.pkl"

    metrics = main(
        [
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "8",
            "--hidden-size",
            "16",
            "--seed",
            "7",
            "--save-model",
            str(final_path),
            "--save-best-model",
            str(best_path),
            "--best-model-metric",
            "episode_return",
            "--quiet",
        ],
    )

    assert metrics["global_step"] == 8
    assert final_path.exists()
    assert best_path.exists()
    checkpoint = read_checkpoint(best_path)
    assert checkpoint["args"]["save_best_model"] == str(best_path)
    assert checkpoint["metrics"]["best_model_metric_value"] == checkpoint["metrics"][
        "episode_return"
    ]
    assert checkpoint["metrics"]["best_model_update"] in {1.0, 2.0}


def test_tiny_jax_mappo_training_can_select_best_checkpoint_by_heldout_eval(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "model.pkl"
    best_path = tmp_path / "model_best.pkl"

    metrics = main(
        [
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "8",
            "--hidden-size",
            "16",
            "--seed",
            "7",
            "--save-model",
            str(final_path),
            "--save-best-model",
            str(best_path),
            "--best-model-selection",
            "eval",
            "--best-model-metric",
            "eval_mean_delivered_food",
            "--best-eval-episodes",
            "1",
            "--best-eval-interval",
            "1",
            "--best-eval-action-mode",
            "sampled_move_greedy_write",
            "--best-eval-move-temperature",
            "0.95",
            "--quiet",
        ],
    )

    assert metrics["global_step"] == 8
    assert final_path.exists()
    assert best_path.exists()
    checkpoint = read_checkpoint(best_path)
    assert checkpoint["metrics"]["best_model_selection"] == "eval"
    assert checkpoint["metrics"]["best_model_metric_value"] == checkpoint["metrics"][
        "eval_mean_delivered_food"
    ]
    assert "eval_mean_episode_length" in checkpoint["metrics"]


def test_jax_training_carries_episode_state_across_updates() -> None:
    metrics = main(
        [
            "--total-timesteps",
            "4",
            "--num-envs",
            "1",
            "--num-steps",
            "2",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "3",
            "--cookie-distance",
            "2",
            "--hidden-size",
            "16",
            "--seed",
            "7",
            "--quiet",
        ]
    )

    assert metrics["global_step"] == 4
    assert metrics["completed_episodes"] == 1
    assert metrics["truncated_episodes"] == 1
