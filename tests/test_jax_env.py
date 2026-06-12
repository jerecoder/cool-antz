from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ant_byte_env import AntByteForagingEnv
from ant_byte_env.jax_env import JaxAntByteForagingEnv


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
    assert obs["ants_pos"].shape == (2, 2)
    assert obs["food"].shape == (3, 4)
    assert obs["bytes"].shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(obs["hub_pos"]), np.array([0, 0]))
    assert int(info.remaining_food) == 3
    assert int(info.num_writes) == 0


def test_jax_step_matches_pickup_delivery_and_write_rules() -> None:
    env = JaxAntByteForagingEnv(width=3, height=1, num_ants=1, food_count=1)
    state, _, _ = env.reset(
        jax.random.PRNGKey(1),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )

    state, obs, reward, terminated, truncated, info = env.step(
        state,
        jnp.array([2, 1], dtype=jnp.int32),
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
        jnp.array([4, 0], dtype=jnp.int32),
    )
    assert float(reward) == 1.0
    assert bool(terminated)
    assert not bool(truncated)
    assert int(obs["ants_carrying"][0]) == 0
    assert int(info.delivered_food) == 1


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
        jnp.array([2, 0], dtype=jnp.int32),
    )

    assert int(state.food[0, 1]) == 1
    assert int(obs["food"][0, 1]) == 1
    assert int(info.remaining_food) == 1
    assert not bool(terminated)

    state, obs, _, _, _, info = env.step(
        state,
        jnp.array([0, 0], dtype=jnp.int32),
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

    actions = [np.array([2, 1], dtype=np.int64), np.array([4, 0], dtype=np.int64)]
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
        jnp.array([2, 0, 2, 7, 2, 3], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(env.action_nvec), np.array([5, 8, 5, 8, 5, 8]))
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
    assert obs["bytes"].shape == (2, 3, 3)
    np.testing.assert_array_equal(np.asarray(reward), np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(terminated), np.array([True, True]))
    np.testing.assert_array_equal(np.asarray(truncated), np.array([False, False]))
    np.testing.assert_array_equal(np.asarray(info.num_writes), np.array([1, 1]))
