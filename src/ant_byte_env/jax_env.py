"""Pure JAX core for ant byte foraging dynamics."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env.env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    DEFAULT_ACTOR_VISION_DEPTH,
    DEFAULT_FACING,
    DEFAULT_WRITE_BITS,
    MAX_WRITE_BITS,
    MOVEMENT_ACTION_COUNT,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    max_write_value,
    write_value_count,
)
from ant_byte_env.maze import (
    generate_wide_corridor_maze,
    nearest_open_flat_lookup,
    open_flat_indices,
)


class JaxAntState(NamedTuple):
    hub_pos: jax.Array
    ants_pos: jax.Array
    ants_count: jax.Array
    ants_facing: jax.Array
    ants_carrying: jax.Array
    food: jax.Array
    initial_food: jax.Array
    bytes: jax.Array
    delivered_food: jax.Array
    step_count: jax.Array
    initial_food_total: jax.Array
    visited_cells: jax.Array
    viewed_cells: jax.Array
    obstacles: jax.Array


class JaxAntInfo(NamedTuple):
    delivered_food: jax.Array
    remaining_food: jax.Array
    step_count: jax.Array
    num_writes: jax.Array
    num_overwrites: jax.Array
    visited_cell_count: jax.Array
    newly_visited_cells: jax.Array
    viewed_cell_count: jax.Array
    newly_viewed_cells: jax.Array
    visible_border_cells: jax.Array


JaxObs = dict[str, jax.Array]


class JaxAntByteForagingEnv:
    """JIT-friendly, functional version of ``AntByteForagingEnv`` dynamics."""

    def __init__(
        self,
        *,
        width: int = 16,
        height: int = 16,
        num_ants: int = 4,
        food_count: int = 8,
        food_source_count: int = 1,
        max_steps: int = 500,
        random_food: bool = True,
        random_hub: bool = False,
        random_ant_spawn: bool = False,
        random_ant_spawn_radius: int | None = None,
        layout_margin: int = 0,
        hub_center_window_size: int = 0,
        actor_vision_radius: int = DEFAULT_ACTOR_VISION_DEPTH,
        step_penalty: float = 0.0,
        completion_bonus: float = 0.0,
        write_penalty: float = 0.0,
        write_bits: int = DEFAULT_WRITE_BITS,
        write_while_moving: bool = False,
        per_ant_write_channels: bool = False,
        terminate_on_food_delivery: bool = True,
        terminate_on_full_coverage: bool = False,
        maze_obstacles: bool = False,
        maze_corridor_width: int = 3,
        maze_wall_width: int = 1,
        maze_seed: int = 0,
        maze_layout_count: int = 64,
    ) -> None:
        self._validate_args(
            width=width,
            height=height,
            num_ants=num_ants,
            food_count=food_count,
            food_source_count=food_source_count,
            max_steps=max_steps,
            step_penalty=step_penalty,
            completion_bonus=completion_bonus,
            write_penalty=write_penalty,
            write_bits=write_bits,
            per_ant_write_channels=per_ant_write_channels,
            random_ant_spawn_radius=random_ant_spawn_radius,
            layout_margin=layout_margin,
            hub_center_window_size=hub_center_window_size,
            actor_vision_radius=actor_vision_radius,
            maze_corridor_width=maze_corridor_width,
            maze_wall_width=maze_wall_width,
            maze_layout_count=maze_layout_count,
        )
        self.width = int(width)
        self.height = int(height)
        self.num_ants = int(num_ants)
        self.food_count = int(food_count)
        self.food_source_count = int(food_source_count)
        self.max_steps = int(max_steps)
        self.random_food = bool(random_food)
        self.random_hub = bool(random_hub)
        self.random_ant_spawn = bool(random_ant_spawn)
        self.random_ant_spawn_radius = (
            None if random_ant_spawn_radius is None else int(random_ant_spawn_radius)
        )
        self.layout_margin = int(layout_margin)
        self.hub_center_window_size = int(hub_center_window_size)
        self.actor_vision_radius = int(actor_vision_radius)
        self.step_penalty = float(step_penalty)
        self.completion_bonus = float(completion_bonus)
        self.write_penalty = float(write_penalty)
        self.write_bits = int(write_bits)
        self.write_while_moving = bool(write_while_moving)
        self.per_ant_write_channels = bool(per_ant_write_channels)
        self.terminate_on_food_delivery = bool(terminate_on_food_delivery)
        self.terminate_on_full_coverage = bool(terminate_on_full_coverage)
        self.maze_obstacles = bool(maze_obstacles)
        self.maze_corridor_width = int(maze_corridor_width)
        self.maze_wall_width = int(maze_wall_width)
        self.maze_seed = int(maze_seed)
        self.maze_layout_count = int(maze_layout_count)
        obstacle_bank_np = self._build_obstacle_bank()
        open_counts_np = np.sum(~obstacle_bank_np.reshape((obstacle_bank_np.shape[0], -1)), axis=1)
        min_open_cell_count = int(np.min(open_counts_np))
        if min_open_cell_count <= 1 and self.food_count > 0:
            raise ValueError("food_count requires at least one open non-hub tile.")
        self.obstacle_bank = jnp.asarray(obstacle_bank_np, dtype=jnp.bool_)
        self.obstacles = self.obstacle_bank[0]
        open_indices_np = open_flat_indices(obstacle_bank_np[0])
        self.open_flat_indices = jnp.asarray(open_indices_np, dtype=jnp.int32)
        self.nearest_open_flat = jnp.asarray(
            nearest_open_flat_lookup(obstacle_bank_np[0]),
            dtype=jnp.int32,
        )
        self.open_cell_count = min_open_cell_count
        self.source_count = min(self.food_source_count, max(min_open_cell_count - 1, 0))
        self.write_value_count = write_value_count(self.write_bits)
        self.max_write_value = max_write_value(self.write_bits)
        self.action_nvec = jnp.asarray(
            [MOVEMENT_ACTION_COUNT, self.write_value_count] * self.num_ants,
            dtype=jnp.int32,
        )

    @staticmethod
    def _validate_args(
        *,
        width: int,
        height: int,
        num_ants: int,
        food_count: int,
        food_source_count: int,
        max_steps: int,
        step_penalty: float,
        completion_bonus: float,
        write_penalty: float,
        write_bits: int,
        per_ant_write_channels: bool,
        random_ant_spawn_radius: int | None,
        layout_margin: int,
        hub_center_window_size: int,
        actor_vision_radius: int,
        maze_corridor_width: int,
        maze_wall_width: int,
        maze_layout_count: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if num_ants <= 0:
            raise ValueError("num_ants must be positive.")
        if food_count < 0:
            raise ValueError("food_count must be non-negative.")
        if food_count > 0 and width * height <= 1:
            raise ValueError("food_count requires at least one non-hub tile.")
        if food_source_count <= 0:
            raise ValueError("food_source_count must be positive.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if step_penalty < 0:
            raise ValueError("step_penalty must be non-negative.")
        if completion_bonus < 0:
            raise ValueError("completion_bonus must be non-negative.")
        if write_penalty < 0:
            raise ValueError("write_penalty must be non-negative.")
        if random_ant_spawn_radius is not None and int(random_ant_spawn_radius) < 0:
            raise ValueError("random_ant_spawn_radius must be non-negative.")
        if int(layout_margin) < 0:
            raise ValueError("layout_margin must be non-negative.")
        if int(layout_margin) * 2 >= min(width, height):
            raise ValueError("layout_margin must leave at least one interior cell.")
        if food_count > 0 and (width - 2 * int(layout_margin)) * (
            height - 2 * int(layout_margin)
        ) <= 1:
            raise ValueError("layout_margin must leave at least two interior cells with food.")
        if int(hub_center_window_size) < 0:
            raise ValueError("hub_center_window_size must be non-negative.")
        if int(hub_center_window_size) > min(width, height):
            raise ValueError("hub_center_window_size must fit inside the grid.")
        if int(actor_vision_radius) < 0:
            raise ValueError("actor_vision_radius must be non-negative.")
        if maze_corridor_width <= 0:
            raise ValueError("maze_corridor_width must be positive.")
        if maze_wall_width <= 0:
            raise ValueError("maze_wall_width must be positive.")
        if maze_layout_count <= 0:
            raise ValueError("maze_layout_count must be positive.")
        if (
            not isinstance(write_bits, (int, np.integer))
            or write_bits <= 0
            or write_bits > MAX_WRITE_BITS
        ):
            raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    def reset(
        self,
        key: jax.Array,
        *,
        hub_pos: jax.Array | None = None,
        food_positions: jax.Array | None = None,
        obstacles: jax.Array | None = None,
        previous_obstacles: jax.Array | None = None,
    ) -> tuple[JaxAntState, JaxObs, JaxAntInfo]:
        """Return ``(state, obs, info)`` for a new episode."""

        maze_key, hub_key, food_key, ant_key = jax.random.split(key, 4)
        actual_obstacles = self._initial_obstacles(
            maze_key,
            obstacles=obstacles,
            previous_obstacles=previous_obstacles,
        )
        actual_hub_pos = self._initial_hub_pos(hub_key, hub_pos, actual_obstacles)
        food_key = food_key if self.random_hub and hub_pos is None else key
        food = self._initial_food_grid(
            food_key,
            actual_hub_pos,
            food_positions,
            actual_obstacles,
        )
        ants_pos = self._initial_ant_positions(ant_key, actual_hub_pos, food, actual_obstacles)
        ants_count = self._build_ants_count_grid(ants_pos)
        visited_cells = self._mark_visited(
            jnp.zeros((self.height, self.width), dtype=jnp.bool_),
            ants_pos,
            actual_obstacles,
        )
        viewed_cells = self._mark_viewed(
            jnp.zeros((self.height, self.width), dtype=jnp.bool_),
            ants_pos,
            actual_obstacles,
        )
        state = JaxAntState(
            hub_pos=actual_hub_pos,
            ants_pos=ants_pos.astype(jnp.int32),
            ants_count=ants_count,
            ants_facing=jnp.full((self.num_ants,), DEFAULT_FACING, dtype=jnp.int32),
            ants_carrying=jnp.zeros((self.num_ants,), dtype=jnp.bool_),
            food=food,
            initial_food=food,
            bytes=jnp.zeros((self.height, self.width), dtype=jnp.uint8),
            delivered_food=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            initial_food_total=jnp.sum(food).astype(jnp.int32),
            visited_cells=visited_cells,
            viewed_cells=viewed_cells,
            obstacles=actual_obstacles,
        )
        return state, self.observe(state), self.info(
            state,
            num_writes=0,
            num_overwrites=0,
            newly_visited_cells=0,
            newly_viewed_cells=0,
            visible_border_cells=self._count_visible_border_cells(
                ants_pos,
                actual_obstacles,
            ),
        )

    def step(
        self,
        state: JaxAntState,
        action: jax.Array,
    ) -> tuple[JaxAntState, JaxObs, jax.Array, jax.Array, jax.Array, JaxAntInfo]:
        """Advance one step; write values can optionally apply after movement."""

        flat_action = jnp.asarray(action, dtype=jnp.int32).reshape((2 * self.num_ants,))
        written_mask = jnp.zeros((self.height, self.width), dtype=jnp.bool_)
        init_carry = (
            state.ants_pos,
            state.ants_facing,
            state.ants_carrying,
            state.food,
            state.bytes,
            state.delivered_food,
            written_mask,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def scan_ant(carry, ant_index):
            (
                ants_pos,
                ants_facing,
                ants_carrying,
                food,
                bytes_grid,
                delivered_food,
                written_tiles,
                num_writes,
                num_overwrites,
                delivery_count,
            ) = carry

            action_start = 2 * ant_index
            move = flat_action[action_start]
            write_value = flat_action[action_start + 1].astype(jnp.uint8)
            next_facing = self._action_facing(ants_facing[ant_index], move)
            next_pos = self._move_position(
                ants_pos[ant_index],
                action=move,
                obstacles=state.obstacles,
            )
            ants_facing = ants_facing.at[ant_index].set(next_facing)
            ants_pos = ants_pos.at[ant_index].set(next_pos)

            x_pos = next_pos[0]
            y_pos = next_pos[1]
            tile_had_food = food[y_pos, x_pos] > 0
            tile_is_hub = jnp.all(next_pos == state.hub_pos)
            was_carrying = ants_carrying[ant_index]

            picked_up = jnp.logical_and(jnp.logical_not(was_carrying), tile_had_food)
            food = food.at[y_pos, x_pos].add(jnp.where(picked_up, -1, 0))
            carrying_after_pickup = jnp.logical_or(was_carrying, picked_up)

            delivered = jnp.logical_and(carrying_after_pickup, tile_is_hub)
            ants_carrying = ants_carrying.at[ant_index].set(
                jnp.logical_and(carrying_after_pickup, jnp.logical_not(delivered))
            )
            delivered_food = delivered_food + delivered.astype(jnp.int32)
            delivery_count = delivery_count + delivered.astype(jnp.int32)

            wants_write = jnp.logical_or(
                move == ACTION_STAY,
                jnp.asarray(self.write_while_moving, dtype=jnp.bool_),
            )
            can_write = jnp.logical_and(
                wants_write,
                jnp.logical_not(jnp.logical_or(tile_had_food, tile_is_hub)),
            )
            already_written = written_tiles[y_pos, x_pos]
            num_overwrites = num_overwrites + jnp.logical_and(can_write, already_written).astype(
                jnp.int32
            )
            written_tiles = written_tiles.at[y_pos, x_pos].set(
                jnp.logical_or(already_written, can_write)
            )
            current_value = bytes_grid[y_pos, x_pos]
            applied_write_value = write_value
            if self.per_ant_write_channels:
                ant_bit_index = jnp.mod(
                    ant_index.astype(jnp.uint8),
                    jnp.asarray(self.write_bits, dtype=jnp.uint8),
                )
                ant_bit = jnp.left_shift(
                    jnp.asarray(1, dtype=jnp.uint8),
                    ant_bit_index,
                )
                applied_write_value = (
                    current_value & jnp.bitwise_not(ant_bit)
                ) | (write_value & ant_bit)
            bytes_grid = bytes_grid.at[y_pos, x_pos].set(
                jnp.where(can_write, applied_write_value, current_value)
            )
            num_writes = num_writes + can_write.astype(jnp.int32)

            return (
                ants_pos,
                ants_facing,
                ants_carrying,
                food,
                bytes_grid,
                delivered_food,
                written_tiles,
                num_writes,
                num_overwrites,
                delivery_count,
            ), None

        (
            ants_pos,
            ants_facing,
            ants_carrying,
            food,
            bytes_grid,
            delivered_food,
            _,
            num_writes,
            num_overwrites,
            delivery_count,
        ), _ = jax.lax.scan(scan_ant, init_carry, jnp.arange(self.num_ants))

        step_count = state.step_count + jnp.asarray(1, dtype=jnp.int32)
        ants_count = self._build_ants_count_grid(ants_pos)
        step_visited_cells = self._mark_visited(
            jnp.zeros((self.height, self.width), dtype=jnp.bool_),
            ants_pos,
            state.obstacles,
        )
        newly_visited_cells = jnp.sum(
            jnp.logical_and(step_visited_cells, jnp.logical_not(state.visited_cells))
        ).astype(jnp.int32)
        visited_cells = jnp.logical_or(state.visited_cells, step_visited_cells)
        step_viewed_cells = self._mark_viewed(
            jnp.zeros((self.height, self.width), dtype=jnp.bool_),
            ants_pos,
            state.obstacles,
        )
        newly_viewed_cells = jnp.sum(
            jnp.logical_and(step_viewed_cells, jnp.logical_not(state.viewed_cells))
        ).astype(jnp.int32)
        viewed_cells = jnp.logical_or(state.viewed_cells, step_viewed_cells)
        visible_border_cells = self._count_visible_border_cells(ants_pos, state.obstacles)
        next_state = JaxAntState(
            hub_pos=state.hub_pos,
            ants_pos=ants_pos,
            ants_count=ants_count,
            ants_facing=ants_facing,
            ants_carrying=ants_carrying,
            food=food,
            initial_food=state.initial_food,
            bytes=bytes_grid,
            delivered_food=delivered_food,
            step_count=step_count,
            initial_food_total=state.initial_food_total,
            visited_cells=visited_cells,
            viewed_cells=viewed_cells,
            obstacles=state.obstacles,
        )
        completed_food = delivered_food >= state.initial_food_total
        food_terminated = jnp.logical_and(
            jnp.asarray(self.terminate_on_food_delivery, dtype=jnp.bool_),
            completed_food,
        )
        full_coverage = jnp.sum(visited_cells) >= jnp.sum(
            jnp.logical_not(state.obstacles)
        ).astype(jnp.int32)
        coverage_terminated = jnp.logical_and(
            jnp.asarray(self.terminate_on_full_coverage, dtype=jnp.bool_),
            full_coverage,
        )
        terminated = jnp.logical_or(food_terminated, coverage_terminated)
        reward = (
            delivery_count.astype(jnp.float32)
            - jnp.asarray(self.step_penalty * self.num_ants, dtype=jnp.float32)
            - num_writes.astype(jnp.float32) * jnp.asarray(self.write_penalty, dtype=jnp.float32)
            + completed_food.astype(jnp.float32)
            * jnp.asarray(self.terminate_on_food_delivery, dtype=jnp.float32)
            * jnp.asarray(self.completion_bonus, dtype=jnp.float32)
        )
        truncated = step_count >= self.max_steps
        info = self.info(
            next_state,
            num_writes=num_writes,
            num_overwrites=num_overwrites,
            newly_visited_cells=newly_visited_cells,
            newly_viewed_cells=newly_viewed_cells,
            visible_border_cells=visible_border_cells,
        )
        return next_state, self.observe(next_state), reward, terminated, truncated, info

    def observe(self, state: JaxAntState) -> JaxObs:
        return {
            "ants_pos": state.ants_pos.astype(jnp.int32),
            "ants_count": state.ants_count.astype(jnp.int32),
            "ants_carrying": state.ants_carrying.astype(jnp.int8),
            "ants_facing": state.ants_facing.astype(jnp.int32),
            "food": state.food.astype(jnp.int32),
            "bytes": state.bytes.astype(jnp.uint8),
            "obstacles": state.obstacles.astype(jnp.int8),
            "hub_pos": state.hub_pos.astype(jnp.int32),
        }

    def info(
        self,
        state: JaxAntState,
        *,
        num_writes: int | jax.Array,
        num_overwrites: int | jax.Array,
        newly_visited_cells: int | jax.Array,
        newly_viewed_cells: int | jax.Array,
        visible_border_cells: int | jax.Array,
    ) -> JaxAntInfo:
        return JaxAntInfo(
            delivered_food=state.delivered_food.astype(jnp.int32),
            remaining_food=jnp.sum(state.food).astype(jnp.int32),
            step_count=state.step_count.astype(jnp.int32),
            num_writes=jnp.asarray(num_writes, dtype=jnp.int32),
            num_overwrites=jnp.asarray(num_overwrites, dtype=jnp.int32),
            visited_cell_count=jnp.sum(state.visited_cells).astype(jnp.int32),
            newly_visited_cells=jnp.asarray(newly_visited_cells, dtype=jnp.int32),
            viewed_cell_count=jnp.sum(state.viewed_cells).astype(jnp.int32),
            newly_viewed_cells=jnp.asarray(newly_viewed_cells, dtype=jnp.int32),
            visible_border_cells=jnp.asarray(visible_border_cells, dtype=jnp.int32),
        )

    def _mark_visited(
        self,
        visited_cells: jax.Array,
        ants_pos: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        ants_pos = ants_pos.astype(jnp.int32)
        marked = visited_cells.at[ants_pos[:, 1], ants_pos[:, 0]].set(True)
        return jnp.logical_and(marked, jnp.logical_not(obstacles))

    def _mark_viewed(
        self,
        viewed_cells: jax.Array,
        ants_pos: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        radius = int(self.actor_vision_radius)
        axis = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
        offset_y = jnp.repeat(axis, 2 * radius + 1)
        offset_x = jnp.tile(axis, 2 * radius + 1)
        x_pos = jnp.clip(
            ants_pos[:, 0, None] + offset_x[None, :],
            0,
            self.width - 1,
        )
        y_pos = jnp.clip(
            ants_pos[:, 1, None] + offset_y[None, :],
            0,
            self.height - 1,
        )
        marked = viewed_cells.at[y_pos.reshape(-1), x_pos.reshape(-1)].set(True)
        return jnp.logical_and(marked, jnp.logical_not(obstacles))

    def _count_visible_border_cells(
        self,
        ants_pos: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        visible_cells = self._mark_viewed(
            jnp.zeros((self.height, self.width), dtype=jnp.bool_),
            ants_pos,
            obstacles,
        )
        x_coords = jnp.arange(self.width, dtype=jnp.int32)[None, :]
        y_coords = jnp.arange(self.height, dtype=jnp.int32)[:, None]
        border_mask = (
            (x_coords == 0)
            | (x_coords == self.width - 1)
            | (y_coords == 0)
            | (y_coords == self.height - 1)
        )
        return jnp.sum(jnp.logical_and(visible_cells, border_mask)).astype(jnp.int32)

    def _initial_obstacles(
        self,
        key: jax.Array,
        *,
        obstacles: jax.Array | None,
        previous_obstacles: jax.Array | None,
    ) -> jax.Array:
        if obstacles is not None:
            return jnp.asarray(obstacles, dtype=jnp.bool_)
        if self.obstacle_bank.shape[0] <= 1:
            return self.obstacle_bank[0]

        candidate_mask = jnp.ones((self.obstacle_bank.shape[0],), dtype=jnp.bool_)
        if previous_obstacles is not None:
            previous = jnp.asarray(previous_obstacles, dtype=jnp.bool_)
            same_as_previous = jnp.all(
                self.obstacle_bank == previous[None, :, :],
                axis=(1, 2),
            )
            candidate_mask = jnp.where(
                jnp.any(jnp.logical_not(same_as_previous)),
                jnp.logical_not(same_as_previous),
                candidate_mask,
            )
        scores = jax.random.uniform(key, candidate_mask.shape, dtype=jnp.float32)
        selected_index = jnp.argmax(jnp.where(candidate_mask, scores, -jnp.inf))
        return self.obstacle_bank[selected_index]

    def _initial_hub_pos(
        self,
        key: jax.Array,
        hub_pos: jax.Array | None,
        obstacles: jax.Array,
    ) -> jax.Array:
        if hub_pos is not None:
            return self._nearest_open_position(
                jnp.asarray(hub_pos, dtype=jnp.int32),
                obstacles,
            )
        if self.random_hub:
            flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
            flat_x = flat_indices % self.width
            flat_y = flat_indices // self.width
            is_open = jnp.logical_not(obstacles.reshape((-1,)))
            centered_open = is_open & self._inside_hub_center_window(flat_x, flat_y)
            candidate_mask = jnp.where(jnp.any(centered_open), centered_open, is_open)
            interior_open = candidate_mask & self._inside_layout_margin(flat_x, flat_y)
            candidate_mask = jnp.where(jnp.any(interior_open), interior_open, candidate_mask)
            scores = jax.random.uniform(key, is_open.shape, dtype=jnp.float32)
            flat = jnp.argmax(jnp.where(candidate_mask, scores, -jnp.inf)).astype(jnp.int32)
            return self._flat_to_position(flat)
        return self._nearest_open_position(
            jnp.asarray([self.width // 2, self.height // 2], dtype=jnp.int32),
            obstacles,
        )

    def _initial_food_grid(
        self,
        key: jax.Array,
        hub_pos: jax.Array,
        food_positions: jax.Array | None,
        obstacles: jax.Array,
    ) -> jax.Array:
        if self.food_count == 0:
            return jnp.zeros((self.height, self.width), dtype=jnp.int32)

        if food_positions is not None:
            positions = jnp.asarray(food_positions, dtype=jnp.int32).reshape((-1, 2))
            positions = self._nearest_open_positions(positions, obstacles)
            hub_flat = hub_pos[1] * self.width + hub_pos[0]
            fallback_pos = self._first_non_hub_open_position(hub_pos, obstacles)
            is_hub = jnp.all(positions == hub_pos, axis=-1)
            positions = jnp.where(is_hub[:, None], fallback_pos[None, :], positions)
        else:
            positions = self._sample_food_positions(key, hub_pos, obstacles)
        return self._distribute_food(positions)

    def _sample_food_positions(
        self,
        key: jax.Array,
        hub_pos: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        if self.source_count == 0:
            return jnp.zeros((0, 2), dtype=jnp.int32)

        flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
        flat_x = flat_indices % self.width
        flat_y = flat_indices // self.width
        is_open = jnp.logical_not(obstacles.reshape((-1,)))
        is_hub = (flat_x == hub_pos[0]) & (flat_y == hub_pos[1])
        candidate_mask = is_open & jnp.logical_not(is_hub)
        interior_mask = candidate_mask & self._inside_layout_margin(flat_x, flat_y)
        enough_interior = jnp.sum(interior_mask.astype(jnp.int32)) >= self.source_count
        candidate_mask = jnp.where(enough_interior, interior_mask, candidate_mask)
        if self.random_food:
            scores = jax.random.uniform(
                key,
                (self.width * self.height,),
                dtype=jnp.float32,
            )
        else:
            scores = -flat_indices.astype(jnp.float32)
        _, selected_flat = jax.lax.top_k(
            jnp.where(candidate_mask, scores, -jnp.inf),
            self.source_count,
        )
        return jnp.stack(
            [selected_flat % self.width, selected_flat // self.width],
            axis=-1,
        ).astype(jnp.int32)

    def _distribute_food(self, positions: jax.Array) -> jax.Array:
        position_count = positions.shape[0]
        if position_count <= 0:
            raise ValueError("food_positions must contain at least one position.")

        base_amount, extra_units = divmod(self.food_count, position_count)
        amounts = base_amount + (jnp.arange(position_count) < extra_units).astype(jnp.int32)
        return jnp.zeros((self.height, self.width), dtype=jnp.int32).at[
            positions[:, 1],
            positions[:, 0],
        ].add(amounts)

    def _initial_ant_positions(
        self,
        key: jax.Array,
        hub_pos: jax.Array,
        food: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        if not self.random_ant_spawn:
            return jnp.repeat(hub_pos.reshape(1, 2), self.num_ants, axis=0)

        flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
        flat_x = flat_indices % self.width
        flat_y = flat_indices // self.width
        is_hub = (flat_x == hub_pos[0]) & (flat_y == hub_pos[1])
        has_food = food.reshape((-1,)) > 0
        is_open = jnp.logical_not(obstacles.reshape((-1,)))
        if self.random_ant_spawn_radius is None:
            within_spawn_radius = jnp.ones_like(is_hub, dtype=bool)
        else:
            within_spawn_radius = (
                jnp.maximum(
                    jnp.abs(flat_x - hub_pos[0]),
                    jnp.abs(flat_y - hub_pos[1]),
                )
                <= self.random_ant_spawn_radius
            )
        interior_mask = self._inside_layout_margin(flat_x, flat_y)
        preferred_mask = (
            within_spawn_radius
            & jnp.logical_not(is_hub | has_food)
            & is_open
            & interior_mask
        )
        fallback_mask = (
            within_spawn_radius
            & jnp.logical_not(is_hub)
            & is_open
            & interior_mask
        )
        no_margin_fallback_mask = within_spawn_radius & jnp.logical_not(is_hub) & is_open
        candidate_mask = jnp.where(
            jnp.any(preferred_mask),
            preferred_mask,
            fallback_mask,
        )
        candidate_mask = jnp.where(
            jnp.any(candidate_mask),
            candidate_mask,
            no_margin_fallback_mask,
        )
        candidate_mask = jnp.where(
            jnp.any(candidate_mask),
            candidate_mask,
            is_open,
        )
        scores = jax.random.uniform(
            key,
            (self.num_ants, self.width * self.height),
            dtype=jnp.float32,
        )
        selected = jnp.argmax(jnp.where(candidate_mask[None, :], scores, -jnp.inf), axis=1)
        return jnp.stack([flat_x[selected], flat_y[selected]], axis=-1).astype(jnp.int32)

    def _inside_layout_margin(self, x_pos: jax.Array, y_pos: jax.Array) -> jax.Array:
        margin = jnp.asarray(self.layout_margin, dtype=jnp.int32)
        return (
            (x_pos >= margin)
            & (x_pos < self.width - margin)
            & (y_pos >= margin)
            & (y_pos < self.height - margin)
        )

    def _inside_hub_center_window(self, x_pos: jax.Array, y_pos: jax.Array) -> jax.Array:
        window_size = jnp.asarray(self.hub_center_window_size, dtype=jnp.int32)
        x_start = jnp.asarray((self.width - self.hub_center_window_size) // 2, dtype=jnp.int32)
        y_start = jnp.asarray((self.height - self.hub_center_window_size) // 2, dtype=jnp.int32)
        in_window = (
            (x_pos >= x_start)
            & (x_pos < x_start + window_size)
            & (y_pos >= y_start)
            & (y_pos < y_start + window_size)
        )
        return jnp.where(window_size <= 0, jnp.ones_like(in_window, dtype=jnp.bool_), in_window)

    def _action_facing(self, facing: jax.Array, action: jax.Array) -> jax.Array:
        valid_facing = (
            (facing == MOVE_UP)
            | (facing == MOVE_RIGHT)
            | (facing == MOVE_DOWN)
            | (facing == MOVE_LEFT)
        )
        current_facing = jnp.where(valid_facing, facing, DEFAULT_FACING)
        is_cardinal_move = (
            (action == ACTION_UP)
            | (action == ACTION_RIGHT)
            | (action == ACTION_DOWN)
            | (action == ACTION_LEFT)
        )
        return jnp.where(is_cardinal_move, action, current_facing).astype(jnp.int32)

    def _move_position(
        self,
        position: jax.Array,
        *,
        action: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        dx = jnp.where(action == ACTION_RIGHT, 1, jnp.where(action == ACTION_LEFT, -1, 0))
        dy = jnp.where(action == ACTION_DOWN, 1, jnp.where(action == ACTION_UP, -1, 0))
        should_advance = (action != ACTION_STAY).astype(jnp.int32)
        next_pos = jnp.asarray(
            [
                jnp.clip(position[0] + should_advance * dx, 0, self.width - 1),
                jnp.clip(position[1] + should_advance * dy, 0, self.height - 1),
            ],
            dtype=jnp.int32,
        )
        blocked = obstacles[next_pos[1], next_pos[0]]
        return jnp.where(
            jnp.logical_or(action == ACTION_STAY, blocked),
            position,
            next_pos,
        )

    def _build_ants_count_grid(self, ants_pos: jax.Array) -> jax.Array:
        ants_pos = ants_pos.astype(jnp.int32)
        return jnp.zeros((self.height, self.width), dtype=jnp.int32).at[
            ants_pos[:, 1],
            ants_pos[:, 0],
        ].add(1)

    def _build_obstacle_bank(self) -> np.ndarray:
        if not self.maze_obstacles:
            return np.zeros((1, self.height, self.width), dtype=bool)
        return np.stack(
            [
                self._build_obstacle_grid(seed=self.maze_seed + layout_index)
                for layout_index in range(self.maze_layout_count)
            ],
            axis=0,
        )

    def _build_obstacle_grid(self, *, seed: int) -> np.ndarray:
        if not self.maze_obstacles:
            return np.zeros((self.height, self.width), dtype=bool)
        obstacles = generate_wide_corridor_maze(
            width=self.width,
            height=self.height,
            corridor_width=self.maze_corridor_width,
            wall_width=self.maze_wall_width,
            seed=seed,
        )
        if not np.any(~obstacles):
            raise ValueError("maze obstacle layout must contain at least one open cell.")
        return obstacles

    def _nearest_open_position(self, position: jax.Array, obstacles: jax.Array) -> jax.Array:
        flat = position[1] * self.width + position[0]
        flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
        flat_x = flat_indices % self.width
        flat_y = flat_indices // self.width
        distance = jnp.abs(flat_x - position[0]) + jnp.abs(flat_y - position[1])
        is_open = jnp.logical_not(obstacles.reshape((-1,)))
        nearest_flat = jnp.argmin(
            jnp.where(is_open, distance, jnp.iinfo(jnp.int32).max)
        ).astype(jnp.int32)
        return self._flat_to_position(nearest_flat)

    def _nearest_open_positions(self, positions: jax.Array, obstacles: jax.Array) -> jax.Array:
        return jax.vmap(lambda position: self._nearest_open_position(position, obstacles))(
            positions
        )

    def _first_non_hub_open_position(
        self,
        hub_pos: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
        hub_flat = hub_pos[1] * self.width + hub_pos[0]
        candidate_mask = jnp.logical_and(
            jnp.logical_not(obstacles.reshape((-1,))),
            flat_indices != hub_flat,
        )
        fallback_flat = jnp.argmin(
            jnp.where(candidate_mask, flat_indices, jnp.iinfo(jnp.int32).max)
        ).astype(jnp.int32)
        return self._flat_to_position(fallback_flat)

    def _flat_to_position(self, flat: jax.Array) -> jax.Array:
        return jnp.asarray([flat % self.width, flat // self.width], dtype=jnp.int32)
