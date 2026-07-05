from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from ant_byte_env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    AntByteForagingEnv,
)


def _open_cell_next_to_wall(obstacles: np.ndarray) -> tuple[tuple[int, int], int]:
    directions = (
        (ACTION_UP, 0, -1),
        (ACTION_RIGHT, 1, 0),
        (ACTION_DOWN, 0, 1),
        (ACTION_LEFT, -1, 0),
    )
    height, width = obstacles.shape
    for y_pos in range(height):
        for x_pos in range(width):
            if obstacles[y_pos, x_pos]:
                continue
            for action, dx, dy in directions:
                next_x = x_pos + dx
                next_y = y_pos + dy
                if 0 <= next_x < width and 0 <= next_y < height:
                    if obstacles[next_y, next_x]:
                        return (x_pos, y_pos), action
    raise AssertionError("maze should contain an open cell adjacent to a wall")


def _food_source_keys(food_grid: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(x_pos), int(y_pos))
        for y_pos, x_pos in np.argwhere(np.asarray(food_grid) > 0)
    }


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
        "visited_cell_count": 1,
        "newly_visited_cells": 0,
    }
    assert obs["ants_pos"].shape == (2, 2)
    assert obs["ants_facing"].shape == (2,)
    np.testing.assert_array_equal(obs["ants_facing"], np.array([2, 2], dtype=np.int8))
    assert obs["ants_count"].shape == (4, 5)
    assert np.all(obs["ants_pos"] == obs["hub_pos"])
    assert obs["ants_count"][obs["hub_pos"][1], obs["hub_pos"][0]] == 2
    assert obs["ants_count"].sum() == 2
    env.close()


def test_per_ant_write_channels_only_update_each_ants_bit() -> None:
    env = AntByteForagingEnv(
        width=5,
        height=5,
        num_ants=3,
        food_count=0,
        write_bits=3,
        write_while_moving=True,
        per_ant_write_channels=True,
    )

    obs, _ = env.reset(seed=1, options={"hub_pos": (2, 2)})
    obs, _, _, _, info = env.step(
        np.array(
            [
                ACTION_RIGHT,
                7,
                ACTION_RIGHT,
                0,
                ACTION_RIGHT,
                7,
            ],
            dtype=np.int64,
        )
    )

    assert int(obs["bytes"][2, 3]) == 5
    assert info["num_writes"] == 3
    assert info["num_overwrites"] == 2

    obs, _, _, _, _ = env.step(
        np.array(
            [
                ACTION_STAY,
                0,
                ACTION_STAY,
                7,
                ACTION_STAY,
                0,
            ],
            dtype=np.int64,
        )
    )

    assert int(obs["bytes"][2, 3]) == 2
    env.close()


def test_per_ant_write_channels_reuse_bit_types_when_more_ants_than_bits() -> None:
    env = AntByteForagingEnv(
        width=5,
        height=5,
        num_ants=4,
        food_count=0,
        write_bits=2,
        write_while_moving=True,
        per_ant_write_channels=True,
    )

    obs, _ = env.reset(seed=1, options={"hub_pos": (2, 2)})
    obs, _, _, _, info = env.step(
        np.array(
            [
                ACTION_RIGHT,
                3,
                ACTION_RIGHT,
                2,
                ACTION_RIGHT,
                0,
                ACTION_RIGHT,
                3,
            ],
            dtype=np.int64,
        )
    )

    assert int(obs["bytes"][2, 3]) == 2
    assert info["num_writes"] == 4
    assert info["num_overwrites"] == 3

    obs, _, _, _, _ = env.step(
        np.array(
            [
                ACTION_STAY,
                3,
                ACTION_STAY,
                0,
                ACTION_STAY,
                1,
                ACTION_STAY,
                0,
            ],
            dtype=np.int64,
        )
    )

    assert int(obs["bytes"][2, 3]) == 1
    env.close()


def test_random_hub_reset_is_seed_reproducible() -> None:
    env = AntByteForagingEnv(
        width=6,
        height=5,
        num_ants=3,
        food_count=4,
        random_hub=True,
    )

    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    obs_c, _ = env.reset(seed=124)

    np.testing.assert_array_equal(obs_a["hub_pos"], obs_b["hub_pos"])
    assert (
        not np.array_equal(obs_a["hub_pos"], np.array([3, 2], dtype=np.int32))
        or not np.array_equal(obs_c["hub_pos"], obs_a["hub_pos"])
    )
    assert np.all(obs_a["ants_pos"] == obs_a["hub_pos"])
    assert obs_a["ants_count"][obs_a["hub_pos"][1], obs_a["hub_pos"][0]] == 3
    assert obs_a["food"][obs_a["hub_pos"][1], obs_a["hub_pos"][0]] == 0
    env.close()


def test_layout_margin_restricts_random_hub_and_food_sources() -> None:
    env = AntByteForagingEnv(
        width=8,
        height=8,
        num_ants=2,
        food_count=6,
        food_source_count=3,
        random_food=True,
        random_hub=True,
        layout_margin=2,
    )

    obs, _ = env.reset(seed=17)
    hub_pos = obs["hub_pos"]
    food_positions = np.argwhere(obs["food"] > 0)

    assert np.all((2 <= hub_pos) & (hub_pos < 6))
    assert food_positions.shape[0] == 3
    assert np.all((2 <= food_positions) & (food_positions < 6))
    np.testing.assert_array_equal(obs["ants_pos"], np.tile(hub_pos, (2, 1)))
    env.close()


def test_hub_center_window_restricts_random_hub() -> None:
    env = AntByteForagingEnv(
        width=50,
        height=50,
        num_ants=2,
        food_count=0,
        random_hub=True,
        layout_margin=10,
        hub_center_window_size=4,
    )

    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        hub_x, hub_y = obs["hub_pos"]
        assert 23 <= int(hub_x) < 27
        assert 23 <= int(hub_y) < 27
        np.testing.assert_array_equal(obs["ants_pos"], np.tile(obs["hub_pos"], (2, 1)))
    env.close()


def test_random_ant_spawn_is_seed_reproducible_and_avoids_food_and_hub() -> None:
    env = AntByteForagingEnv(
        width=6,
        height=5,
        num_ants=4,
        food_count=4,
        food_source_count=2,
        random_food=True,
        random_hub=True,
        random_ant_spawn=True,
    )

    obs_a, _ = env.reset(seed=321)
    obs_b, _ = env.reset(seed=321)
    obs_c, _ = env.reset(seed=322)

    np.testing.assert_array_equal(obs_a["ants_pos"], obs_b["ants_pos"])
    assert not np.array_equal(obs_a["ants_pos"], obs_c["ants_pos"])
    assert not np.all(obs_a["ants_pos"] == obs_a["hub_pos"])
    hub = tuple(int(value) for value in obs_a["hub_pos"])
    for ant_pos in obs_a["ants_pos"]:
        x_pos, y_pos = (int(ant_pos[0]), int(ant_pos[1]))
        assert (x_pos, y_pos) != hub
        assert obs_a["food"][y_pos, x_pos] == 0
    assert int(obs_a["ants_count"].sum()) == 4
    env.close()


def test_random_ant_spawn_radius_limits_random_spawn_near_hub() -> None:
    env = AntByteForagingEnv(
        width=7,
        height=7,
        num_ants=6,
        food_count=1,
        food_source_count=1,
        random_ant_spawn=True,
        random_ant_spawn_radius=1,
    )

    obs, _ = env.reset(seed=7, options={"hub_pos": (3, 3), "food_positions": [(4, 3)]})

    hub_x, hub_y = (int(value) for value in obs["hub_pos"])
    for ant_pos in obs["ants_pos"]:
        x_pos, y_pos = (int(ant_pos[0]), int(ant_pos[1]))
        assert max(abs(x_pos - hub_x), abs(y_pos - hub_y)) <= 1
        assert (x_pos, y_pos) != (hub_x, hub_y)
        assert obs["food"][y_pos, x_pos] == 0
    assert int(obs["ants_count"].sum()) == 6
    env.close()


def test_maze_obstacles_block_movement_and_appear_in_observation() -> None:
    env = AntByteForagingEnv(
        width=10,
        height=10,
        num_ants=1,
        food_count=0,
        maze_obstacles=True,
        maze_corridor_width=3,
        maze_wall_width=1,
        maze_seed=17,
    )
    start_pos, wall_action = _open_cell_next_to_wall(env.obstacles)
    obs, info = env.reset(
        seed=5,
        options={"hub_pos": start_pos, "obstacles": env.obstacles.copy()},
    )

    assert env.open_cell_count < env.width * env.height
    assert int(obs["obstacles"].sum()) > 0
    assert info["visited_cell_count"] == 1
    np.testing.assert_array_equal(obs["ants_pos"][0], np.asarray(start_pos, dtype=np.int32))

    obs, _, _, _, info = env.step(np.array([wall_action, 0], dtype=np.int64))

    np.testing.assert_array_equal(obs["ants_pos"][0], np.asarray(start_pos, dtype=np.int32))
    assert info["visited_cell_count"] == 1
    env.close()


def test_random_wall_obstacles_block_movement_and_appear_in_observation() -> None:
    env = AntByteForagingEnv(
        width=12,
        height=12,
        num_ants=1,
        food_count=0,
        random_wall_obstacles=True,
        random_wall_count_min=2,
        random_wall_count_max=3,
        random_wall_length_min=4,
        random_wall_length_max=8,
        random_wall_width=1,
        random_wall_l_turn_probability=1.0,
        maze_seed=17,
    )
    start_pos, wall_action = _open_cell_next_to_wall(env.obstacles)
    obs, info = env.reset(
        seed=5,
        options={"hub_pos": start_pos, "obstacles": env.obstacles.copy()},
    )

    assert env.open_cell_count < env.width * env.height
    assert int(obs["obstacles"].sum()) > 0
    assert info["visited_cell_count"] == 1
    np.testing.assert_array_equal(obs["ants_pos"][0], np.asarray(start_pos, dtype=np.int32))

    obs, _, _, _, info = env.step(np.array([wall_action, 0], dtype=np.int64))

    np.testing.assert_array_equal(obs["ants_pos"][0], np.asarray(start_pos, dtype=np.int32))
    assert info["visited_cell_count"] == 1
    env.close()


def test_maze_reset_places_hub_food_and_random_spawns_on_open_cells() -> None:
    env = AntByteForagingEnv(
        width=10,
        height=10,
        num_ants=4,
        food_count=6,
        food_source_count=3,
        random_food=True,
        random_hub=True,
        random_ant_spawn=True,
        maze_obstacles=True,
        maze_corridor_width=3,
        maze_wall_width=1,
        maze_seed=23,
    )

    obs, _ = env.reset(seed=9)

    assert env.observation_space.contains(obs)
    hub_x, hub_y = (int(value) for value in obs["hub_pos"])
    assert not env.obstacles[hub_y, hub_x]
    for ant_x, ant_y in obs["ants_pos"]:
        assert not env.obstacles[int(ant_y), int(ant_x)]
    for food_y, food_x in np.argwhere(obs["food"] > 0):
        assert not env.obstacles[int(food_y), int(food_x)]
    env.close()


def test_maze_reset_changes_obstacles_after_truncation() -> None:
    env = AntByteForagingEnv(
        width=10,
        height=10,
        num_ants=1,
        food_count=0,
        max_steps=1,
        maze_obstacles=True,
        maze_corridor_width=3,
        maze_wall_width=1,
        maze_seed=31,
    )
    env.reset(seed=11)
    first_obstacles = env.obstacles.copy()

    _, _, _, truncated, _ = env.step(np.array([ACTION_STAY, 0], dtype=np.int64))
    env.reset(seed=12)

    assert truncated
    assert not np.array_equal(env.obstacles, first_obstacles)
    env.close()


def test_random_reset_changes_colony_and_cookies_after_truncation() -> None:
    env = AntByteForagingEnv(
        width=8,
        height=8,
        num_ants=1,
        food_count=4,
        food_source_count=2,
        max_steps=1,
        random_food=True,
        random_hub=True,
    )
    first_obs, _ = env.reset(seed=41)
    first_hub = tuple(int(value) for value in first_obs["hub_pos"])
    first_sources = _food_source_keys(env.initial_food)

    _, _, _, truncated, _ = env.step(np.array([ACTION_STAY, 0], dtype=np.int64))
    second_obs, _ = env.reset(seed=42)
    second_hub = tuple(int(value) for value in second_obs["hub_pos"])
    second_sources = _food_source_keys(env.initial_food)

    assert truncated
    assert second_hub != first_hub
    assert second_sources
    assert second_sources.isdisjoint(first_sources)
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
    env.step(np.array([ACTION_RIGHT, 0], dtype=np.int64))
    obs, _, _, _, _ = env.step(np.array([ACTION_STAY, 7], dtype=np.int64))

    ant_x, ant_y = obs["ants_pos"][0]
    assert obs["bytes"][ant_y, ant_x] == 7
    with pytest.raises(ValueError, match="0 to 7"):
        env.step(np.array([ACTION_RIGHT, 8], dtype=np.int64))
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

    env.step(np.array([ACTION_RIGHT, 0], dtype=np.int64))
    obs, _, _, _, info = env.step(np.array([ACTION_STAY, 1], dtype=np.int64))

    ant_x, ant_y = obs["ants_pos"][0]
    assert obs["bytes"][ant_y, ant_x] == 1
    assert info["num_writes"] == 1
    assert info["num_overwrites"] == 0
    env.close()


def test_movement_step_does_not_write_tile() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=1, food_count=0, seed=3)
    env.reset(seed=3, options={"hub_pos": (0, 0)})

    obs, _, _, _, info = env.step(np.array([ACTION_RIGHT, 1], dtype=np.int64))

    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([1, 0], dtype=np.int32))
    assert obs["bytes"][0, 1] == 0
    assert info["num_writes"] == 0

    obs, _, _, _, info = env.step(np.array([ACTION_STAY, 1], dtype=np.int64))

    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([1, 0], dtype=np.int32))
    assert obs["bytes"][0, 1] == 1
    assert info["num_writes"] == 1
    env.close()


def test_write_while_moving_writes_landing_tile() -> None:
    env = AntByteForagingEnv(
        width=4,
        height=4,
        num_ants=1,
        food_count=0,
        seed=3,
        write_while_moving=True,
    )
    env.reset(seed=3, options={"hub_pos": (0, 0)})

    obs, _, _, _, info = env.step(np.array([ACTION_RIGHT, 1], dtype=np.int64))

    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([1, 0], dtype=np.int32))
    assert obs["bytes"][0, 1] == 1
    assert info["num_writes"] == 1
    env.close()


def test_multiple_ants_on_same_tile_record_overwrites() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=3, food_count=0)
    env.reset(seed=13, options={"hub_pos": (0, 1)})

    env.step(
        np.array([ACTION_RIGHT, 0, ACTION_RIGHT, 0, ACTION_RIGHT, 0], dtype=np.int64)
    )
    obs, _, _, _, info = env.step(
        np.array([ACTION_STAY, 0, ACTION_STAY, 1, ACTION_STAY, 0], dtype=np.int64)
    )

    assert obs["ants_count"][1, 1] == 3
    assert obs["ants_count"].sum() == 3
    assert obs["bytes"][1, 1] == 0
    assert info["num_writes"] == 3
    assert info["num_overwrites"] == 2
    env.close()


def test_moving_ants_on_same_tile_do_not_record_overwrites() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=3, food_count=0)
    env.reset(seed=13, options={"hub_pos": (0, 1)})

    obs, _, _, _, info = env.step(
        np.array([ACTION_RIGHT, 0, ACTION_RIGHT, 1, ACTION_RIGHT, 0], dtype=np.int64)
    )

    assert obs["ants_count"][1, 1] == 3
    assert obs["ants_count"].sum() == 3
    assert obs["bytes"][1, 1] == 0
    assert info["num_writes"] == 0
    assert info["num_overwrites"] == 0
    env.close()


def test_hub_tile_is_unwritable() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    env.reset(seed=37, options={"hub_pos": (1, 1)})

    obs, _, _, _, info = env.step(np.array([ACTION_STAY, 1], dtype=np.int64))

    assert obs["bytes"][1, 1] == 0
    assert info["num_writes"] == 0
    assert info["num_overwrites"] == 0
    env.close()


def test_food_tile_is_unwritable_while_bitten() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=1)
    env.reset(seed=41, options={"hub_pos": (0, 0), "food_positions": [(1, 0)]})

    obs, _, _, _, info = env.step(np.array([ACTION_RIGHT, 1], dtype=np.int64))

    assert obs["ants_carrying"][0] == 1
    assert obs["food"][0, 1] == 0
    assert obs["bytes"][0, 1] == 0
    assert info["num_writes"] == 0
    env.close()


def test_depleted_food_tile_becomes_writable_afterward() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=1)
    env.reset(seed=43, options={"hub_pos": (0, 0), "food_positions": [(1, 0)]})

    env.step(np.array([ACTION_RIGHT, 1], dtype=np.int64))
    obs, _, _, _, info = env.step(np.array([ACTION_STAY, 1], dtype=np.int64))

    assert obs["food"][0, 1] == 0
    assert obs["bytes"][0, 1] == 1
    assert info["num_writes"] == 1
    env.close()


def test_movement_stays_inside_grid_bounds() -> None:
    env = AntByteForagingEnv(width=2, height=2, num_ants=1, food_count=0)
    env.reset(seed=19, options={"hub_pos": (0, 0)})

    env.step(np.array([ACTION_UP, 0], dtype=np.int64))
    obs, _, _, _, _ = env.step(np.array([ACTION_LEFT, 0], dtype=np.int64))
    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([0, 0], dtype=np.int32))

    env.step(np.array([ACTION_UP, 0], dtype=np.int64))
    obs, _, _, _, _ = env.step(np.array([ACTION_LEFT, 0], dtype=np.int64))
    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([0, 0], dtype=np.int32))
    env.close()


def test_ant_moves_cardinally_and_updates_display_facing() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    env.reset(seed=29, options={"hub_pos": (1, 1)})

    assert env.ants_facing.tolist() == [2]

    obs, _, _, _, _ = env.step(np.array([ACTION_UP, 0], dtype=np.int64))
    assert env.ants_facing.tolist() == [1]
    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([1, 0], dtype=np.int32))

    obs, _, _, _, _ = env.step(np.array([ACTION_STAY, 0], dtype=np.int64))
    assert env.ants_facing.tolist() == [1]
    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([1, 0], dtype=np.int32))

    obs, _, _, _, _ = env.step(np.array([ACTION_RIGHT, 0], dtype=np.int64))
    assert env.ants_facing.tolist() == [2]
    np.testing.assert_array_equal(obs["ants_pos"][0], np.array([2, 0], dtype=np.int32))
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
        np.array([ACTION_RIGHT, 0], dtype=np.int64)
    )
    assert obs["ants_carrying"][0] == 1
    assert pickup_reward == 0.0
    assert not terminated
    assert not truncated
    assert info["remaining_food"] == 1
    assert obs["food"][0, 1] == 1

    obs, delivery_reward, terminated, truncated, info = env.step(
        np.array([ACTION_LEFT, 0], dtype=np.int64)
    )
    assert obs["ants_carrying"][0] == 0
    assert delivery_reward == 1.0
    assert not terminated
    assert not truncated
    assert info["delivered_food"] == 1

    obs, pickup_reward, _, _, info = env.step(np.array([ACTION_RIGHT, 0], dtype=np.int64))
    assert obs["ants_carrying"][0] == 1
    assert pickup_reward == 0.0
    assert info["remaining_food"] == 0

    obs, delivery_reward, terminated, truncated, info = env.step(
        np.array([ACTION_LEFT, 0], dtype=np.int64)
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

    obs, _, _, _, info = env.step(np.array([ACTION_RIGHT, 0], dtype=np.int64))
    assert obs["ants_carrying"][0] == 1
    assert obs["food"][0, 1] == 1
    assert info["remaining_food"] == 1

    obs, reward, terminated, truncated, info = env.step(
        np.array([ACTION_STAY, 0], dtype=np.int64)
    )

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

    _, _, _, truncated, info = env.step(np.array([ACTION_STAY, 0], dtype=np.int64))

    assert truncated
    assert info["step_count"] == 1
    env.close()


def test_can_terminate_when_all_cells_are_visited() -> None:
    env = AntByteForagingEnv(
        width=2,
        height=1,
        num_ants=1,
        food_count=0,
        terminate_on_food_delivery=False,
        terminate_on_full_coverage=True,
    )
    _, info = env.reset(seed=9, options={"hub_pos": (0, 0)})

    assert info["visited_cell_count"] == 1

    _, _, terminated, truncated, info = env.step(
        np.array([ACTION_RIGHT, 0], dtype=np.int64)
    )

    assert terminated
    assert not truncated
    assert info["visited_cell_count"] == 2
    env.close()
