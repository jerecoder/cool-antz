from __future__ import annotations

import gymnasium as gym
import numpy as np

from ant_byte_env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    AntByteAutoCurriculumEnv,
)


def _deliver_first_source_cookies(env: AntByteAutoCurriculumEnv, count: int = 6):
    result = None
    for _ in range(count):
        env.step(np.array([ACTION_RIGHT, 0], dtype=np.int64))
        env.step(np.array([ACTION_RIGHT, 0], dtype=np.int64))
        env.step(np.array([ACTION_LEFT, 0], dtype=np.int64))
        result = env.step(np.array([ACTION_LEFT, 0], dtype=np.int64))
    assert result is not None
    return result


def test_autocurriculum_reset_uses_padded_start_stage_observation() -> None:
    env = AntByteAutoCurriculumEnv(max_size=6, max_steps=100, num_ants=1)

    obs, info = env.reset(
        seed=3,
        options={"hub_pos": (0, 0), "food_positions": [(2, 0), (0, 2)]},
    )

    assert env.observation_space.contains(obs)
    assert obs["food"].shape == (6, 6)
    assert obs["ants_count"].shape == (6, 6)
    assert obs["bytes"].shape == (6, 6)
    assert int(obs["food"][0, 2]) == 6
    assert int(obs["food"][2, 0]) == 6
    assert int(np.count_nonzero(obs["food"])) == 2
    assert int(obs["food"][:4, :4].sum()) == 12
    assert int(obs["food"][4:, :].sum()) == 0
    assert int(obs["food"][:, 4:].sum()) == 0
    np.testing.assert_array_equal(obs["active_grid_size"], np.array([4, 4], dtype=np.int32))
    assert info["stage_size"] == 4
    assert info["stage_index"] == 0
    assert info["stage_delivered_food"] == 0
    assert info["cookies_per_stage"] == 6
    assert info["global_step_count"] == 0
    assert info["global_step_budget"] == 100
    env.close()


def test_autocurriculum_advances_to_next_grid_after_six_deliveries() -> None:
    env = AntByteAutoCurriculumEnv(max_size=5, max_steps=100, num_ants=1)
    env.reset(seed=7, options={"hub_pos": (0, 0), "food_positions": [(2, 0), (0, 2)]})

    obs, reward, terminated, truncated, info = _deliver_first_source_cookies(env)

    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert env.observation_space.contains(obs)
    assert info["advanced_stage"] == 1
    assert info["completed_stage_size"] == 4
    assert info["completed_stage_delivered_food"] == 6
    assert info["stage_size"] == 5
    assert info["stage_index"] == 1
    assert info["stage_delivered_food"] == 0
    assert info["global_step_count"] == 24
    assert info["stage_step_count"] == 0
    np.testing.assert_array_equal(obs["hub_pos"], np.array([2, 2], dtype=np.int32))
    assert int(obs["food"][:5, :5].sum()) == 12
    assert set(np.unique(obs["food"])) <= {0, 6}
    assert int(np.count_nonzero(obs["food"])) == 2
    env.close()


def test_autocurriculum_rgb_render_keeps_stable_max_size_shape() -> None:
    env = AntByteAutoCurriculumEnv(
        max_size=5,
        max_steps=100,
        num_ants=1,
        render_mode="rgb_array",
        tile_size=8,
    )
    env.reset(seed=19, options={"hub_pos": (0, 0), "food_positions": [(2, 0), (0, 2)]})

    start_frame = env.render()
    _deliver_first_source_cookies(env)
    next_stage_frame = env.render()

    assert start_frame is not None
    assert next_stage_frame is not None
    assert start_frame.shape == (5 * 8, 5 * 8, 3)
    assert next_stage_frame.shape == start_frame.shape
    assert start_frame.dtype == np.uint8
    assert next_stage_frame.dtype == np.uint8
    env.close()


def test_autocurriculum_keeps_growing_from_five_to_six() -> None:
    env = AntByteAutoCurriculumEnv(
        max_size=6,
        max_steps=200,
        num_ants=1,
        random_food=False,
    )
    env.reset(seed=7, options={"hub_pos": (0, 0), "food_positions": [(2, 0), (0, 2)]})
    _deliver_first_source_cookies(env)

    result = None
    corner_cookie_cycle = (
        ACTION_LEFT,
        ACTION_LEFT,
        ACTION_UP,
        ACTION_UP,
        ACTION_DOWN,
        ACTION_DOWN,
        ACTION_RIGHT,
        ACTION_RIGHT,
    )
    for _ in range(6):
        for move in corner_cookie_cycle:
            result = env.step(np.array([move, 0], dtype=np.int64))

    assert result is not None
    obs, _, terminated, truncated, info = result
    assert not terminated
    assert not truncated
    assert env.observation_space.contains(obs)
    assert info["advanced_stage"] == 1
    assert info["completed_stage_size"] == 5
    assert info["completed_stage_delivered_food"] == 6
    assert info["stage_size"] == 6
    assert info["stage_index"] == 2
    assert info["stage_delivered_food"] == 0
    assert info["global_step_count"] == 72
    np.testing.assert_array_equal(obs["hub_pos"], np.array([3, 3], dtype=np.int32))
    env.close()


def test_autocurriculum_global_budget_truncates_across_stage_resets() -> None:
    env = AntByteAutoCurriculumEnv(max_size=6, max_steps=1, num_ants=1)
    env.reset(seed=11, options={"hub_pos": (0, 0), "food_positions": [(2, 0), (0, 2)]})

    _, _, terminated, truncated, info = env.step(
        np.array([ACTION_STAY, 0], dtype=np.int64)
    )

    assert not terminated
    assert truncated
    assert info["stage_size"] == 4
    assert info["stage_step_count"] == 1
    assert info["global_step_count"] == 1
    assert info["remaining_global_steps"] == 0
    env.close()


def test_autocurriculum_terminates_when_final_stage_goal_is_reached() -> None:
    env = AntByteAutoCurriculumEnv(max_size=4, max_steps=100, num_ants=1)
    env.reset(seed=13, options={"hub_pos": (0, 0), "food_positions": [(2, 0), (0, 2)]})

    _, _, terminated, truncated, info = _deliver_first_source_cookies(env)

    assert terminated
    assert not truncated
    assert info["advanced_stage"] == 0
    assert info["completed_stage_size"] == 4
    assert info["completed_stage_delivered_food"] == 6
    assert info["stage_size"] == 4
    assert info["stage_delivered_food"] == 6
    assert info["global_step_count"] == 24
    env.close()


def test_autocurriculum_environment_is_registered_with_gymnasium() -> None:
    env = gym.make("AntByteAutoCurriculum-v0", max_size=5, max_steps=20, num_ants=1)

    obs, info = env.reset(seed=17)

    assert env.observation_space.contains(obs)
    assert info["stage_size"] == 4
    assert info["cookies_per_stage"] == 6
    env.close()
