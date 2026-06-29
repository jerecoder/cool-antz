from __future__ import annotations

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
    AntByteForagingEnv,
)
from ant_byte_env.jax_autocurriculum_env import JaxAntByteAutoCurriculumEnv
from ant_byte_env.jax_env import JaxAntByteForagingEnv
from ant_byte_env.training.jax_mappo.core import build_actor_observations


def _batched(obs: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}


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


def test_jax_layout_margin_restricts_random_hub_and_food_sources() -> None:
    env = JaxAntByteForagingEnv(
        width=8,
        height=8,
        num_ants=2,
        food_count=6,
        food_source_count=3,
        random_food=True,
        random_hub=True,
        layout_margin=2,
    )

    _, obs, _ = env.reset(jax.random.PRNGKey(17))
    hub_pos = np.asarray(obs["hub_pos"])
    food_positions = np.argwhere(np.asarray(obs["food"]) > 0)

    assert np.all((2 <= hub_pos) & (hub_pos < 6))
    assert food_positions.shape[0] == 3
    assert np.all((2 <= food_positions) & (food_positions < 6))
    np.testing.assert_array_equal(np.asarray(obs["ants_pos"]), np.tile(hub_pos, (2, 1)))


def test_jax_hub_center_window_restricts_random_hub() -> None:
    env = JaxAntByteForagingEnv(
        width=50,
        height=50,
        num_ants=2,
        food_count=0,
        random_hub=True,
        layout_margin=10,
        hub_center_window_size=4,
    )

    for seed in range(10):
        _, obs, _ = env.reset(jax.random.PRNGKey(seed))
        hub_x, hub_y = np.asarray(obs["hub_pos"])
        assert 23 <= int(hub_x) < 27
        assert 23 <= int(hub_y) < 27
        np.testing.assert_array_equal(
            np.asarray(obs["ants_pos"]),
            np.tile(np.asarray(obs["hub_pos"]), (2, 1)),
        )


def test_jax_info_tracks_newly_viewed_cells() -> None:
    env = JaxAntByteForagingEnv(
        width=5,
        height=5,
        num_ants=1,
        food_count=0,
        actor_vision_radius=1,
        terminate_on_food_delivery=False,
    )
    state, _, info = env.reset(
        jax.random.PRNGKey(3),
        hub_pos=jnp.array([2, 2], dtype=jnp.int32),
    )

    assert int(info.viewed_cell_count) == 9
    assert int(info.newly_viewed_cells) == 0
    assert int(info.visible_border_cells) == 0

    _, _, _, _, _, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    assert int(info.newly_visited_cells) == 1
    assert int(info.newly_viewed_cells) == 3
    assert int(info.viewed_cell_count) == 12
    assert int(info.visible_border_cells) == 3


def test_jax_lethal_food_is_visible_food_but_hidden_as_a_channel() -> None:
    env = JaxAntByteForagingEnv(
        width=5,
        height=3,
        num_ants=1,
        food_count=1,
        food_source_count=1,
        lethal_food_count=1,
        lethal_food_source_count=1,
        random_food=False,
        terminate_on_food_delivery=False,
    )

    _, obs, _ = env.reset(
        jax.random.PRNGKey(0),
        hub_pos=jnp.array([0, 1], dtype=jnp.int32),
        food_positions=jnp.array([[1, 1]], dtype=jnp.int32),
        lethal_food_positions=jnp.array([[2, 1]], dtype=jnp.int32),
    )

    assert int(obs["food"][1, 1]) == 1
    assert int(obs["food"][1, 2]) == 1
    assert "lethal_food" not in obs
    assert "dead_ants_count" in obs
    assert int(obs["dead_ants_count"].sum()) == 0


def test_jax_lethal_food_kills_ant_and_dead_ant_is_noop() -> None:
    env = JaxAntByteForagingEnv(
        width=5,
        height=3,
        num_ants=1,
        food_count=1,
        food_source_count=1,
        lethal_food_count=1,
        lethal_food_source_count=1,
        death_penalty=1.0,
        random_food=False,
        terminate_on_food_delivery=False,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(1),
        hub_pos=jnp.array([0, 1], dtype=jnp.int32),
        food_positions=jnp.array([[4, 1]], dtype=jnp.int32),
        lethal_food_positions=jnp.array([[1, 1]], dtype=jnp.int32),
    )

    state, obs, reward, terminated, _, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    assert float(reward) == -1.0
    np.testing.assert_array_equal(np.asarray(state.ants_pos[0]), np.array([1, 1]))
    assert bool(state.ants_alive[0]) is False
    assert bool(state.ants_carrying[0]) is False
    assert int(state.lethal_food.sum()) == 0
    assert int(obs["dead_ants_count"][1, 1]) == 1
    assert int(info.death_events) == 1
    assert bool(terminated) is True

    bytes_before = np.asarray(state.bytes).copy()
    state, obs, *_ = env.step(
        state,
        jnp.array([ACTION_RIGHT, 1], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(state.ants_pos[0]), np.array([1, 1]))
    np.testing.assert_array_equal(np.asarray(state.bytes), bytes_before)
    assert int(obs["ants_count"].sum()) == 0
    assert int(obs["dead_ants_count"][1, 1]) == 1


def test_jax_actor_observation_has_dead_ant_plane_when_lethal_food_enabled() -> None:
    env = JaxAntByteForagingEnv(
        width=5,
        height=3,
        num_ants=1,
        food_count=1,
        food_source_count=1,
        lethal_food_count=1,
        lethal_food_source_count=1,
        random_food=False,
        actor_vision_radius=1,
        terminate_on_food_delivery=False,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(2),
        hub_pos=jnp.array([0, 1], dtype=jnp.int32),
        food_positions=jnp.array([[4, 1]], dtype=jnp.int32),
        lethal_food_positions=jnp.array([[1, 1]], dtype=jnp.int32),
    )
    _, obs, *_ = env.step(state, jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32))

    actor_obs = build_actor_observations(
        _batched(obs),
        food_scale=1,
        actor_vision_radius=1,
        write_bits=env.write_bits,
    )
    patch_size = 9
    dead_plane = np.asarray(actor_obs[0, 0, 2 * patch_size : 3 * patch_size])

    assert dead_plane[4] == 1.0
    assert dead_plane.sum() == 1.0


def test_jax_maze_obstacles_block_movement_and_are_observed() -> None:
    env = JaxAntByteForagingEnv(
        width=10,
        height=10,
        num_ants=1,
        food_count=0,
        max_steps=250,
        maze_obstacles=True,
        maze_corridor_width=3,
        maze_wall_width=1,
        maze_seed=17,
        terminate_on_food_delivery=False,
        terminate_on_full_coverage=True,
    )
    start_pos, wall_action = _open_cell_next_to_wall(np.asarray(env.obstacles))
    state, obs, info = env.reset(
        jax.random.PRNGKey(5),
        hub_pos=jnp.asarray(start_pos, dtype=jnp.int32),
        obstacles=env.obstacles,
    )

    assert env.open_cell_count < env.width * env.height
    assert int(jnp.sum(obs["obstacles"])) > 0
    assert int(info.visited_cell_count) == 1
    np.testing.assert_array_equal(np.asarray(obs["ants_pos"][0]), np.asarray(start_pos))

    _, obs, _, terminated, truncated, info = env.step(
        state,
        jnp.array([wall_action, 0], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(obs["ants_pos"][0]), np.asarray(start_pos))
    assert not bool(terminated)
    assert not bool(truncated)
    assert int(info.visited_cell_count) == 1


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


def test_jax_random_ant_spawn_is_seed_reproducible_and_avoids_food_and_hub() -> None:
    env = JaxAntByteForagingEnv(
        width=6,
        height=5,
        num_ants=4,
        food_count=4,
        food_source_count=2,
        random_food=True,
        random_hub=True,
        random_ant_spawn=True,
    )

    _, obs_a, _ = env.reset(jax.random.PRNGKey(321))
    _, obs_b, _ = env.reset(jax.random.PRNGKey(321))
    _, obs_c, _ = env.reset(jax.random.PRNGKey(322))

    np.testing.assert_array_equal(np.asarray(obs_a["ants_pos"]), np.asarray(obs_b["ants_pos"]))
    assert not np.array_equal(np.asarray(obs_a["ants_pos"]), np.asarray(obs_c["ants_pos"]))
    assert not np.all(
        np.asarray(obs_a["ants_pos"]) == np.asarray(obs_a["hub_pos"])[None, :]
    )
    food = np.asarray(obs_a["food"])
    hub = tuple(int(value) for value in np.asarray(obs_a["hub_pos"]))
    for ant_pos in np.asarray(obs_a["ants_pos"]):
        x_pos, y_pos = (int(ant_pos[0]), int(ant_pos[1]))
        assert (x_pos, y_pos) != hub
        assert food[y_pos, x_pos] == 0
    assert int(jnp.sum(obs_a["ants_count"])) == 4


def test_jax_random_ant_spawn_radius_limits_random_spawn_near_hub() -> None:
    env = JaxAntByteForagingEnv(
        width=7,
        height=7,
        num_ants=6,
        food_count=1,
        food_source_count=1,
        random_ant_spawn=True,
        random_ant_spawn_radius=1,
    )

    _, obs, _ = env.reset(
        jax.random.PRNGKey(7),
        hub_pos=jnp.array([3, 3], dtype=jnp.int32),
        food_positions=jnp.array([[4, 3]], dtype=jnp.int32),
    )

    food = np.asarray(obs["food"])
    hub_x, hub_y = (int(value) for value in np.asarray(obs["hub_pos"]))
    for ant_pos in np.asarray(obs["ants_pos"]):
        x_pos, y_pos = (int(ant_pos[0]), int(ant_pos[1]))
        assert max(abs(x_pos - hub_x), abs(y_pos - hub_y)) <= 1
        assert (x_pos, y_pos) != (hub_x, hub_y)
        assert food[y_pos, x_pos] == 0
    assert int(jnp.sum(obs["ants_count"])) == 6


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


def test_jax_completion_bonus_rewards_final_delivery_only() -> None:
    env = JaxAntByteForagingEnv(
        width=3,
        height=1,
        num_ants=1,
        food_count=1,
        completion_bonus=2.5,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(1),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )

    state, _, reward, terminated, _, _ = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )
    assert float(reward) == 0.0
    assert not bool(terminated)

    _, _, reward, terminated, _, _ = env.step(
        state,
        jnp.array([ACTION_LEFT, 0], dtype=jnp.int32),
    )
    assert float(reward) == 3.5
    assert bool(terminated)


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


def test_jax_env_tracks_distinct_visited_cells_without_observation_memory() -> None:
    env = JaxAntByteForagingEnv(
        width=3,
        height=3,
        num_ants=1,
        food_count=0,
        terminate_on_food_delivery=False,
    )
    state, obs, info = env.reset(
        jax.random.PRNGKey(8),
        hub_pos=jnp.array([1, 1], dtype=jnp.int32),
    )

    assert "visited_cells" not in obs
    assert int(info.visited_cell_count) == 1
    assert int(info.newly_visited_cells) == 0

    state, _, reward, terminated, _, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    assert float(reward) == 0.0
    assert not bool(terminated)
    assert int(info.newly_visited_cells) == 1
    assert int(info.visited_cell_count) == 2

    _, _, _, terminated, _, info = env.step(
        state,
        jnp.array([ACTION_LEFT, 0], dtype=jnp.int32),
    )

    assert not bool(terminated)
    assert int(info.newly_visited_cells) == 0
    assert int(info.visited_cell_count) == 2


def test_jax_env_can_terminate_when_all_cells_are_visited() -> None:
    env = JaxAntByteForagingEnv(
        width=2,
        height=1,
        num_ants=1,
        food_count=0,
        terminate_on_food_delivery=False,
        terminate_on_full_coverage=True,
    )
    state, _, info = env.reset(
        jax.random.PRNGKey(8),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
    )

    assert int(info.visited_cell_count) == 1

    _, _, _, terminated, truncated, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 0], dtype=jnp.int32),
    )

    assert bool(terminated)
    assert not bool(truncated)
    assert int(info.visited_cell_count) == 2


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


def test_jax_per_ant_write_channels_preserve_other_ants_bits() -> None:
    env = JaxAntByteForagingEnv(
        width=5,
        height=5,
        num_ants=3,
        food_count=0,
        write_bits=3,
        write_while_moving=True,
        per_ant_write_channels=True,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(4),
        hub_pos=jnp.array([2, 2], dtype=jnp.int32),
    )

    state, obs, _, _, _, info = env.step(
        state,
        jnp.array(
            [
                ACTION_RIGHT,
                7,
                ACTION_RIGHT,
                0,
                ACTION_RIGHT,
                7,
            ],
            dtype=jnp.int32,
        ),
    )

    assert int(obs["bytes"][2, 3]) == 5
    assert int(info.num_writes) == 3
    assert int(info.num_overwrites) == 2

    _, obs, _, _, _, _ = env.step(
        state,
        jnp.array(
            [
                ACTION_STAY,
                0,
                ACTION_STAY,
                7,
                ACTION_STAY,
                0,
            ],
            dtype=jnp.int32,
        ),
    )

    assert int(obs["bytes"][2, 3]) == 2


def test_jax_per_ant_write_channels_reuse_bit_types_when_more_ants_than_bits() -> None:
    env = JaxAntByteForagingEnv(
        width=5,
        height=5,
        num_ants=4,
        food_count=0,
        write_bits=2,
        write_while_moving=True,
        per_ant_write_channels=True,
    )
    state, _, _ = env.reset(
        jax.random.PRNGKey(4),
        hub_pos=jnp.array([2, 2], dtype=jnp.int32),
    )

    state, obs, _, _, _, info = env.step(
        state,
        jnp.array(
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
            dtype=jnp.int32,
        ),
    )

    assert int(obs["bytes"][2, 3]) == 2
    assert int(info.num_writes) == 4
    assert int(info.num_overwrites) == 3

    _, obs, _, _, _, _ = env.step(
        state,
        jnp.array(
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
            dtype=jnp.int32,
        ),
    )

    assert int(obs["bytes"][2, 3]) == 1


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
