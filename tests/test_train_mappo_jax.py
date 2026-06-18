from __future__ import annotations

import argparse
import importlib
import json
import types

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
    evaluate_checkpoint,
    evaluate_params,
    flatten_agent_actions,
    get_action_and_value,
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
)
from ant_byte_env.training.jax_mappo.transfer import (
    adapt_movement_head_layer,
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
    )
    return params, states, obs


def _food_source_signature(food_grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    food_positions = np.argwhere(food_grid > 0)[:, ::-1]
    return tuple(tuple(int(coord) for coord in position) for position in food_positions)


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


def test_jax_observation_builders_match_mappo_shapes() -> None:
    obs = _batched_reset_obs()

    central_obs = build_central_observations(obs, food_scale=3)
    actor_obs = build_actor_observations(obs, food_scale=3)

    assert central_obs.shape == (1, 54)
    assert actor_obs.shape == (1, 2, 50)
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
    np.testing.assert_allclose(
        np.asarray(actor_obs[0, 0, 36:45]),
        np.array([1, 1, 1.0, 1, 0, 0.0, 1, 0, 0.0], dtype=np.float32),
    )


def test_jax_actor_observation_exposes_border_mask() -> None:
    obs = _batched_reset_obs()

    actor_obs = build_actor_observations(obs, food_scale=3, actor_vision_radius=1)

    assert actor_obs.shape == (1, 2, 50)
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
        nonzero_byte_tiles=jnp.array([[0.0, 2.0], [4.0, 6.0]], dtype=jnp.float32),
        nonzero_byte_fraction=jnp.array([[0.0, 0.125], [0.25, 0.375]], dtype=jnp.float32),
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
    assert stats["write_action_nonzero_rate"] == 0.5
    assert stats["mean_write_action_value"] == 0.75
    assert stats["mean_nonzero_byte_tiles"] == 3.0
    assert stats["final_mean_nonzero_byte_tiles"] == 5.0
    assert stats["mean_nonzero_byte_fraction"] == 0.1875
    assert stats["final_mean_nonzero_byte_fraction"] == 0.3125


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

    assert actor_obs.shape == (1, 2, 68)
    assert actions.shape == (1, 2, 2)
    assert logprob.shape == (1, 2)
    assert entropy.shape == (1, 2)
    assert value.shape == (1,)
    assert flat_actions.shape == (1, 4)
    assert bool(jnp.all((0 <= actions[..., 0]) & (actions[..., 0] < MOVEMENT_ACTION_COUNT)))
    assert bool(jnp.all((0 <= actions[..., 1]) & (actions[..., 1] <= 7)))


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
    ):
        assert args.random_hub is True
        assert args.random_food is True
        states, obs = original_reset_batch(args=args, env=env, key=key)
        observed_hubs.append(tuple(int(value) for value in np.asarray(obs["hub_pos"])[0]))
        observed_food.append(_food_source_signature(np.asarray(obs["food"])[0]))
        return states, obs

    monkeypatch.setattr(jax_evaluation, "reset_batch", recording_reset_batch)

    evaluate_params(params=params, args=args, num_episodes=8)

    assert len(set(observed_hubs)) > 1
    assert len(set(observed_food)) > 1
    for hub, food_sources in zip(observed_hubs, observed_food, strict=True):
        assert hub not in food_sources


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
    assert args.wandb_mode == "offline"
    assert args.wandb_tags == ["jax", "50x50"]


def test_jax_wandb_defaults_keep_tracking_disabled() -> None:
    args = parse_args([])

    assert args.wandb_project is None
    assert args.wandb_entity is None
    assert args.wandb_group is None
    assert args.wandb_run_name is None
    assert args.wandb_mode == "online"
    assert args.wandb_tags is None


def test_jax_parse_args_accepts_log_interval() -> None:
    args = parse_args(["--log-interval", "10"])

    assert args.log_interval == 10


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
) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.logs: list[tuple[dict[str, object], int | None]] = []
            self.finished = False

        def log(self, payload: dict[str, object], *, step: int | None = None) -> None:
            self.logs.append((payload, step))

        def finish(self) -> None:
            self.finished = True

    fake_run = FakeRun()
    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: fake_run,
        Video=lambda *args, **kwargs: object(),
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
            "--quiet",
        ],
    )

    assert metrics["global_step"] == 8
    assert [step for _, step in fake_run.logs] == [4, 8]
    assert fake_run.logs[-1][0]["global_step"] == 8.0
    assert fake_run.finished is True


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
