from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ant_byte_env import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    MOVEMENT_ACTION_COUNT,
    AntByteForagingEnv,
)
from ant_byte_env.jax_autocurriculum_env import JaxAntByteAutoCurriculumEnv
from ant_byte_env.jax_env import JaxAntByteForagingEnv
from ant_byte_env.training.jax_mappo.core import build_actor_observations


def _batched(obs: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}


def _deliver_first_jax_source(env: JaxAntByteAutoCurriculumEnv, state, count: int = 6):
    result = None
    for _ in range(count):
        state, *_ = env.step(state, jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32))
        state, *_ = env.step(state, jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32))
        state, *_ = env.step(state, jnp.array([ACTION_LEFT, 0], dtype=jnp.int32))
        result = env.step(state, jnp.array([ACTION_LEFT, 0], dtype=jnp.int32))
        state = result[0]
    assert result is not None
    return result


def test_jax_autocurriculum_reset_uses_fixed_shape_start_stage() -> None:
    env = JaxAntByteAutoCurriculumEnv(
        width=50,
        height=50,
        num_ants=1,
        food_count=12,
        food_source_count=2,
        max_steps=200,
    )

    state, obs, info = env.reset(
        jax.random.PRNGKey(0),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[2, 0], [0, 2]], dtype=jnp.int32),
    )

    assert int(state.active_size) == 4
    assert obs["food"].shape == (50, 50)
    assert obs["ants_count"].shape == (50, 50)
    assert obs["bytes"].shape == (50, 50)
    np.testing.assert_array_equal(np.asarray(obs["active_grid_size"]), np.array([4, 4]))
    assert int(obs["food"][0, 2]) == 6
    assert int(obs["food"][2, 0]) == 6
    assert int(jnp.count_nonzero(obs["food"])) == 2
    assert int(info.stage_delivered_food) == 0
    assert int(info.active_size) == 4


def test_jax_autocurriculum_food_starts_outside_actor_view() -> None:
    env = JaxAntByteAutoCurriculumEnv(
        width=50,
        height=50,
        num_ants=1,
        food_count=12,
        food_source_count=2,
        max_steps=200,
        actor_vision_radius=1,
    )

    _, obs, _ = env.reset(jax.random.PRNGKey(1))
    actor_obs = build_actor_observations(
        _batched(obs),
        food_scale=12,
        actor_vision_radius=1,
    )

    np.testing.assert_array_equal(np.asarray(actor_obs[0, 0, :9]), np.zeros(9))
    assert int(jnp.count_nonzero(obs["food"])) == 2
    assert set(np.asarray(obs["food"])[np.asarray(obs["food"]) > 0].tolist()) == {6}


def test_jax_autocurriculum_movement_clamps_to_active_size() -> None:
    env = JaxAntByteAutoCurriculumEnv(
        width=50,
        height=50,
        num_ants=1,
        food_count=12,
        food_source_count=2,
        max_steps=200,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(2),
        hub_pos=jnp.array([3, 3], dtype=jnp.int32),
        food_positions=jnp.array([[0, 0], [1, 0]], dtype=jnp.int32),
    )

    state, obs, *_ = env.step(state, jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32))

    np.testing.assert_array_equal(np.asarray(obs["ants_pos"][0]), np.array([3, 3]))


def test_jax_autocurriculum_stage_advance_is_not_episode_done() -> None:
    env = JaxAntByteAutoCurriculumEnv(
        width=5,
        height=5,
        num_ants=1,
        food_count=12,
        food_source_count=2,
        max_steps=200,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(3),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[2, 0], [0, 2]], dtype=jnp.int32),
    )

    state, obs, reward, terminated, truncated, info = _deliver_first_jax_source(env, state)

    assert float(reward) == 1.0
    assert not bool(terminated)
    assert not bool(truncated)
    assert int(state.active_size) == 5
    assert int(state.stage_delivered_food) == 0
    assert int(info.advanced_stage) == 1
    assert int(info.completed_stage_size) == 4
    assert int(info.completed_stage_delivered_food) == 6
    np.testing.assert_array_equal(np.asarray(obs["active_grid_size"]), np.array([5, 5]))
    assert int(jnp.sum(obs["food"])) == 12


def test_jax_autocurriculum_final_stage_terminates() -> None:
    env = JaxAntByteAutoCurriculumEnv(
        width=4,
        height=4,
        num_ants=1,
        food_count=12,
        food_source_count=2,
        max_steps=200,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(4),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[2, 0], [0, 2]], dtype=jnp.int32),
    )

    state, _, _, terminated, truncated, info = _deliver_first_jax_source(env, state)

    assert bool(terminated)
    assert not bool(truncated)
    assert int(state.active_size) == 4
    assert int(state.stage_delivered_food) == 6
    assert int(info.completed_stage_size) == 4


def test_jax_autocurriculum_global_budget_truncates_mid_stage() -> None:
    env = JaxAntByteAutoCurriculumEnv(
        width=50,
        height=50,
        num_ants=1,
        food_count=12,
        food_source_count=2,
        max_steps=1,
    )
    state, _, _ = env.reset(jax.random.PRNGKey(5))

    _, _, _, terminated, truncated, info = env.step(
        state,
        jnp.array([ACTION_STAY, 0], dtype=jnp.int32),
    )

    assert not bool(terminated)
    assert bool(truncated)
    assert int(info.step_count) == 1
    assert int(info.active_size) == 4


def test_jax_reset_matches_env_observation_contract() -> None:
    env = JaxAntByteForagingEnv(
        width=4,
        height=3,
        num_ants=2,
        food_count=3,
        random_food=False,
    )

    state, obs, info = env.reset(
        jax.random.PRNGKey(0),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )

    assert state.ants_pos.shape == (2, 2)
    assert state.ants_count.shape == (3, 4)
    assert obs["ants_pos"].shape == (2, 2)
    assert obs["ants_facing"].shape == (2,)
    np.testing.assert_array_equal(np.asarray(obs["ants_facing"]), np.array([2, 2]))
    assert obs["ants_count"].shape == (3, 4)
    assert obs["food"].shape == (3, 4)
    assert obs["bytes"].shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(obs["hub_pos"]), np.array([0, 0]))
    assert int(obs["ants_count"][0, 0]) == 2
    assert int(jnp.sum(obs["ants_count"])) == 2
    assert int(info.remaining_food) == 3
    assert int(info.num_writes) == 0


def test_jax_random_hub_reset_is_seed_reproducible() -> None:
    env = JaxAntByteForagingEnv(
        width=6,
        height=5,
        num_ants=3,
        food_count=4,
        random_hub=True,
    )

    _, obs_a, _ = env.reset(jax.random.PRNGKey(123))
    _, obs_b, _ = env.reset(jax.random.PRNGKey(123))
    _, obs_c, _ = env.reset(jax.random.PRNGKey(124))

    np.testing.assert_array_equal(np.asarray(obs_a["hub_pos"]), np.asarray(obs_b["hub_pos"]))
    assert (
        not np.array_equal(np.asarray(obs_a["hub_pos"]), np.array([3, 2], dtype=np.int32))
        or not np.array_equal(np.asarray(obs_c["hub_pos"]), np.asarray(obs_a["hub_pos"]))
    )
    np.testing.assert_array_equal(
        np.asarray(obs_a["ants_pos"]),
        np.repeat(np.asarray(obs_a["hub_pos"])[None, :], 3, axis=0),
    )
    hub_x, hub_y = np.asarray(obs_a["hub_pos"])
    assert int(obs_a["ants_count"][hub_y, hub_x]) == 3
    assert int(obs_a["food"][hub_y, hub_x]) == 0


def test_jax_step_matches_pickup_delivery_and_write_rules() -> None:
    env = JaxAntByteForagingEnv(width=3, height=1, num_ants=1, food_count=1)
    state, _, _ = env.reset(
        jax.random.PRNGKey(1),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )

    state, obs, reward, terminated, truncated, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 1], dtype=jnp.int32),
    )
    assert float(reward) == 0.0
    assert not bool(terminated)
    assert not bool(truncated)
    assert int(obs["ants_carrying"][0]) == 1
    assert int(obs["food"][0, 1]) == 0
    assert int(obs["bytes"][0, 1]) == 0
    assert int(info.num_writes) == 0

    state, obs, reward, terminated, truncated, info = env.step(
        state,
        jnp.array([ACTION_LEFT, 0], dtype=jnp.int32),
    )
    assert float(reward) == 1.0
    assert bool(terminated)
    assert not bool(truncated)
    assert int(obs["ants_carrying"][0]) == 0
    assert int(info.delivered_food) == 1


def test_jax_movement_step_does_not_write_tile() -> None:
    env = JaxAntByteForagingEnv(width=4, height=4, num_ants=1, food_count=0)
    state, _, _ = env.reset(
        jax.random.PRNGKey(8),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
    )

    state, obs, _, _, _, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 1], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(obs["ants_pos"][0]), np.array([1, 0]))
    assert int(obs["bytes"][0, 1]) == 0
    assert int(info.num_writes) == 0

    _, obs, _, _, _, info = env.step(
        state,
        jnp.array([ACTION_STAY, 1], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(obs["ants_pos"][0]), np.array([1, 0]))
    assert int(obs["bytes"][0, 1]) == 1
    assert int(info.num_writes) == 1


def test_jax_write_while_moving_writes_landing_tile() -> None:
    env = JaxAntByteForagingEnv(
        width=4,
        height=4,
        num_ants=1,
        food_count=0,
        write_while_moving=True,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(8),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
    )

    _, obs, _, _, _, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 1], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(obs["ants_pos"][0]), np.array([1, 0]))
    assert int(obs["bytes"][0, 1]) == 1
    assert int(info.num_writes) == 1


def test_jax_food_state_tracks_remaining_bite_counts() -> None:
    env = JaxAntByteForagingEnv(
        width=3,
        height=1,
        num_ants=1,
        food_count=2,
        random_food=False,
    )
    state, obs, info = env.reset(
        jax.random.PRNGKey(4),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )

    assert int(state.food[0, 1]) == 2
    assert int(obs["food"][0, 1]) == 2
    assert int(state.initial_food_total) == 2
    assert int(info.remaining_food) == 2

    state, obs, _, terminated, _, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    assert int(state.food[0, 1]) == 1
    assert int(obs["food"][0, 1]) == 1
    assert int(info.remaining_food) == 1
    assert not bool(terminated)

    state, obs, _, _, _, info = env.step(
        state,
        jnp.array([ACTION_STAY, 0], dtype=jnp.int32),
    )

    assert int(state.food[0, 1]) == 1
    assert int(obs["food"][0, 1]) == 1
    assert int(info.remaining_food) == 1


def test_jax_core_matches_gym_env_for_fixed_rollout() -> None:
    gym_env = AntByteForagingEnv(
        width=3,
        height=1,
        num_ants=1,
        food_count=1,
        random_food=False,
    )
    jax_env = JaxAntByteForagingEnv(
        width=3,
        height=1,
        num_ants=1,
        food_count=1,
        random_food=False,
    )
    gym_obs, _ = gym_env.reset(
        seed=7,
        options={"hub_pos": (0, 0), "food_positions": [(1, 0)]},
    )
    jax_state, jax_obs, _ = jax_env.reset(
        jax.random.PRNGKey(7),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )

    actions = [
        np.array([ACTION_RIGHT, 1], dtype=np.int64),
        np.array([ACTION_LEFT, 0], dtype=np.int64),
    ]
    try:
        for action in actions:
            gym_obs, gym_reward, gym_terminated, gym_truncated, gym_info = gym_env.step(action)
            jax_state, jax_obs, jax_reward, jax_terminated, jax_truncated, jax_info = jax_env.step(
                jax_state,
                jnp.asarray(action, dtype=jnp.int32),
            )

            assert float(jax_reward) == gym_reward
            assert bool(jax_terminated) == gym_terminated
            assert bool(jax_truncated) == gym_truncated
            assert int(jax_info.num_writes) == gym_info["num_writes"]
            assert int(jax_info.num_overwrites) == gym_info["num_overwrites"]
            for key, value in gym_obs.items():
                np.testing.assert_array_equal(np.asarray(jax_obs[key]), value)
    finally:
        gym_env.close()


def test_jax_write_bits_control_action_range_and_overwrites() -> None:
    env = JaxAntByteForagingEnv(width=4, height=4, num_ants=3, food_count=0, write_bits=3)
    state, _, _ = env.reset(
        jax.random.PRNGKey(2),
        hub_pos=jnp.array([0, 1], dtype=jnp.int32),
    )

    state, obs, _, _, _, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0, ACTION_RIGHT, 7, ACTION_RIGHT, 3], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(
        np.asarray(env.action_nvec),
        np.array([MOVEMENT_ACTION_COUNT, 8, MOVEMENT_ACTION_COUNT, 8, MOVEMENT_ACTION_COUNT, 8]),
    )
    assert int(obs["ants_count"][1, 1]) == 3
    assert int(jnp.sum(obs["ants_count"])) == 3
    assert int(obs["bytes"][1, 1]) == 0
    assert int(info.num_writes) == 0
    assert int(info.num_overwrites) == 0

    _, obs, _, _, _, info = env.step(
        state,
        jnp.array([ACTION_STAY, 0, ACTION_STAY, 7, ACTION_STAY, 3], dtype=jnp.int32),
    )

    assert int(obs["bytes"][1, 1]) == 3
    assert int(info.num_writes) == 3
    assert int(info.num_overwrites) == 2


def test_jax_step_can_be_jitted_and_vmapped() -> None:
    env = JaxAntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    states, _, _ = jax.vmap(env.reset)(keys)
    actions = jnp.array([[2, 1], [3, 0]], dtype=jnp.int32)

    step_fn = jax.jit(jax.vmap(env.step))
    next_states, obs, reward, terminated, truncated, info = step_fn(states, actions)

    assert next_states.ants_pos.shape == (2, 1, 2)
    assert next_states.ants_count.shape == (2, 3, 3)
    assert obs["ants_count"].shape == (2, 3, 3)
    assert obs["bytes"].shape == (2, 3, 3)
    np.testing.assert_array_equal(np.asarray(reward), np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(terminated), np.array([True, True]))
    np.testing.assert_array_equal(np.asarray(truncated), np.array([False, False]))
    np.testing.assert_array_equal(np.asarray(info.num_writes), np.array([0, 0]))
