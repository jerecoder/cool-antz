from __future__ import annotations

import numpy as np
import gymnasium as gym

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
    assert np.all(obs["ants_pos"] == obs["hub_pos"])
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
    env.reset(seed=3)

    obs, _, _, _, info = env.step(np.array([0, 137], dtype=np.int64))

    ant_x, ant_y = obs["ants_pos"][0]
    assert obs["bytes"][ant_y, ant_x] == 137
    assert info["num_writes"] == 1
    assert info["num_overwrites"] == 0
    env.close()


def test_pickup_and_delivery_flow() -> None:
    env = AntByteForagingEnv(
        width=3,
        height=3,
        num_ants=1,
        food_count=1,
        random_food=False,
        seed=5,
    )
    env.reset(seed=5, options={"hub_pos": (0, 0), "food_positions": [(1, 0)]})

    obs, pickup_reward, terminated, truncated, info = env.step(
        np.array([2, 0], dtype=np.int64)
    )
    assert obs["ants_carrying"][0] == 1
    assert pickup_reward > 0
    assert not terminated
    assert not truncated
    assert info["remaining_food"] == 0

    obs, delivery_reward, terminated, truncated, info = env.step(
        np.array([4, 0], dtype=np.int64)
    )
    assert obs["ants_carrying"][0] == 0
    assert delivery_reward >= 9.0
    assert terminated
    assert not truncated
    assert info["delivered_food"] == 1
    env.close()


def test_max_steps_truncates() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=1, food_count=1, max_steps=1)
    env.reset(seed=9)

    _, _, _, truncated, info = env.step(np.array([0, 0], dtype=np.int64))

    assert truncated
    assert info["step_count"] == 1
    env.close()
