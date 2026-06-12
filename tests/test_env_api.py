from __future__ import annotations

import numpy as np
import gymnasium as gym
import pytest

from ant_byte_env import AntByteForagingEnv


def test_reset_returns_obs_and_info() -> None:
    env = AntByteForagingEnv(width=5, height=4, num_ants=2, food_count=3, seed=123)

    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert info == {
        "delivered_food": 0,
        "remaining_food": 3,
        "step_count": 0,
        "num_writes": 0,
        "num_overwrites": 0,
    }
    assert obs["ants_pos"].shape == (2, 2)
    assert obs["ants_count"].shape == (4, 5)
    assert np.all(obs["ants_pos"] == obs["hub_pos"])
    assert obs["ants_count"][obs["hub_pos"][1], obs["hub_pos"][0]] == 2
    assert obs["ants_count"].sum() == 2
    env.close()


def test_action_space_defaults_to_one_write_bit_per_ant() -> None:
    env = AntByteForagingEnv(width=5, height=4, num_ants=2, food_count=3, seed=123)

    np.testing.assert_array_equal(env.action_space.nvec, np.array([5, 2, 5, 2]))
    assert env.observation_space["bytes"].high.max() == 1
    env.close()


def test_write_bits_controls_action_space_and_tile_value_range() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=1, food_count=0, write_bits=3)
    env.reset(seed=3, options={"hub_pos": (0, 0)})

    np.testing.assert_array_equal(env.action_space.nvec, np.array([5, 8]))
    assert env.observation_space["bytes"].high.max() == 7
    obs, _, _, _, _ = env.step(np.array([2, 7], dtype=np.int64))

    ant_x, ant_y = obs["ants_pos"][0]
    assert obs["bytes"][ant_y, ant_x] == 7
    with pytest.raises(ValueError, match="0 to 7"):
        env.step(np.array([2, 8], dtype=np.int64))
    env.close()


def test_environment_is_registered_with_gymnasium() -> None:
    env = gym.make("AntByteForaging-v0", width=4, height=4, num_ants=1, food_count=1)

    obs, info = env.reset(seed=17)

    assert env.observation_space.contains(obs)
    assert info["remaining_food"] == 1
    env.close()


def test_step_returns_modern_gymnasium_tuple() -> None:
    env = AntByteForagingEnv(width=5, height=5, num_ants=2, food_count=2, seed=7)
    env.reset(seed=7)

    result = env.step(env.action_space.sample())

    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step_count"] == 1
    env.close()


def test_action_space_sample_runs_for_twenty_steps() -> None:
    env = AntByteForagingEnv(width=6, height=6, num_ants=4, food_count=5, seed=11)
    obs, _ = env.reset(seed=11)

    for _ in range(20):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break

    env.close()


def test_byte_write_updates_grid() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=1, food_count=0, seed=3)
    env.reset(seed=3, options={"hub_pos": (0, 0)})

    obs, _, _, _, info = env.step(np.array([2, 1], dtype=np.int64))

    ant_x, ant_y = obs["ants_pos"][0]
    assert obs["bytes"][ant_y, ant_x] == 1
    assert info["num_writes"] == 1
    assert info["num_overwrites"] == 0
    env.close()


def test_multiple_ants_on_same_tile_record_overwrites() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=3, food_count=0)
    env.reset(seed=13, options={"hub_pos": (0, 1)})

    obs, _, _, _, info = env.step(np.array([2, 0, 2, 1, 2, 0], dtype=np.int64))

    assert obs["ants_count"][1, 1] == 3
    assert obs["ants_count"].sum() == 3
    assert obs["bytes"][1, 1] == 0
    assert info["num_writes"] == 3
    assert info["num_overwrites"] == 2
    env.close()


def test_hub_tile_is_unwritable() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    env.reset(seed=37, options={"hub_pos": (1, 1)})

    obs, _, _, _, info = env.step(np.array([0, 1], dtype=np.int64))

    assert obs["bytes"][1, 1] == 0
    assert info["num_writes"] == 0
    assert info["num_overwrites"] == 0
    env.close()


def test_food_tile_is_unwritable_while_bitten() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=1)
    env.reset(seed=41, options={"hub_pos": (0, 0), "food_positions": [(1, 0)]})

    obs, _, _, _, info = env.step(np.array([2, 1], dtype=np.int64))

    assert obs["ants_carrying"][0] == 1
    assert obs["food"][0, 1] == 0
    assert obs["bytes"][0, 1] == 0
    assert info["num_writes"] == 0
    env.close()


def test_depleted_food_tile_becomes_writable_afterward() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=1)
    env.reset(seed=43, options={"hub_pos": (0, 0), "food_positions": [(1, 0)]})

    env.step(np.array([2, 1], dtype=np.int64))
    env.step(np.array([4, 0], dtype=np.int64))
    obs, _, _, _, info = env.step(np.array([2, 1], dtype=np.int64))

    assert obs["food"][0, 1] == 0
    assert obs["bytes"][0, 1] == 1
    assert info["num_writes"] == 1
    env.close()


def test_movement_stays_inside_grid_bounds() -> None:
    env = AntByteForagingEnv(width=2, height=2, num_ants=1, food_count=0)
    env.reset(seed=19, options={"hub_pos": (0, 0)})

    obs, _, _, _, _ = env.step(np.array([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([0, 0], dtype=np.int32))

    obs, _, _, _, _ = env.step(np.array([4, 0], dtype=np.int64))
    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([0, 0], dtype=np.int32))
    env.close()


def test_ant_facing_tracks_last_non_stay_move() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    env.reset(seed=29, options={"hub_pos": (1, 1)})

    assert env.ants_facing.tolist() == [2]

    env.step(np.array([1, 0], dtype=np.int64))
    assert env.ants_facing.tolist() == [1]

    env.step(np.array([0, 0], dtype=np.int64))
    assert env.ants_facing.tolist() == [1]

    env.step(np.array([4, 0], dtype=np.int64))
    assert env.ants_facing.tolist() == [4]
    env.close()


def test_invalid_constructor_and_action_inputs_raise() -> None:
    for kwargs in (
        {"width": 0},
        {"height": 0},
        {"num_ants": 0},
        {"food_count": -1},
        {"max_steps": 0},
        {"tile_size": 0},
        {"write_penalty": -0.1},
        {"write_bits": 0},
        {"write_bits": 9},
        {"render_mode": "ansi"},
    ):
        with pytest.raises(ValueError):
            AntByteForagingEnv(**kwargs)

    env = AntByteForagingEnv(width=3, height=3, num_ants=1)
    env.reset(seed=23)
    for action in (
        np.array([0], dtype=np.int64),
        np.array([5, 0], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
    ):
        with pytest.raises(ValueError):
            env.step(action)
    env.close()


def test_pickup_and_delivery_flow() -> None:
    env = AntByteForagingEnv(
        width=3,
        height=3,
        num_ants=1,
        food_count=2,
        random_food=False,
        seed=5,
    )
    env.reset(seed=5, options={"hub_pos": (0, 0), "food_positions": [(1, 0)]})

    obs, pickup_reward, terminated, truncated, info = env.step(
        np.array([2, 0], dtype=np.int64)
    )
    assert obs["ants_carrying"][0] == 1
    assert pickup_reward == 0.0
    assert not terminated
    assert not truncated
    assert info["remaining_food"] == 1
    assert obs["food"][0, 1] == 1

    obs, delivery_reward, terminated, truncated, info = env.step(
        np.array([4, 0], dtype=np.int64)
    )
    assert obs["ants_carrying"][0] == 0
    assert delivery_reward == 1.0
    assert not terminated
    assert not truncated
    assert info["delivered_food"] == 1

    obs, pickup_reward, _, _, info = env.step(np.array([2, 0], dtype=np.int64))
    assert obs["ants_carrying"][0] == 1
    assert pickup_reward == 0.0
    assert info["remaining_food"] == 0

    obs, delivery_reward, terminated, truncated, info = env.step(
        np.array([4, 0], dtype=np.int64)
    )
    assert obs["ants_carrying"][0] == 0
    assert delivery_reward == 1.0
    assert terminated
    assert not truncated
    assert info["delivered_food"] == 2
    env.close()


def test_carrying_ant_does_not_consume_another_food_bite() -> None:
    env = AntByteForagingEnv(
        width=3,
        height=3,
        num_ants=1,
        food_count=2,
        random_food=False,
        seed=47,
    )
    env.reset(seed=47, options={"hub_pos": (0, 0), "food_positions": [(1, 0)]})

    obs, _, _, _, info = env.step(np.array([2, 0], dtype=np.int64))
    assert obs["ants_carrying"][0] == 1
    assert obs["food"][0, 1] == 1
    assert info["remaining_food"] == 1

    obs, reward, terminated, truncated, info = env.step(np.array([0, 0], dtype=np.int64))

    assert obs["ants_carrying"][0] == 1
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert obs["food"][0, 1] == 1
    assert info["remaining_food"] == 1
    env.close()


def test_default_food_count_is_multiple_bites_in_one_source() -> None:
    env = AntByteForagingEnv(width=5, height=5, num_ants=1, food_count=8, seed=31)

    obs, info = env.reset(seed=31)

    assert info["remaining_food"] == 8
    assert np.count_nonzero(obs["food"]) == 1
    assert obs["food"].max() == 8
    env.close()


def test_max_steps_truncates() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=1, food_count=1, max_steps=1)
    env.reset(seed=9)

    _, _, _, truncated, info = env.step(np.array([0, 0], dtype=np.int64))

    assert truncated
    assert info["step_count"] == 1
    env.close()
