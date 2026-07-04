"""Two-team experimental JAX AntByte environment."""

from __future__ import annotations

import argparse
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ant_byte_env.env import ACTION_STAY, DEFAULT_FACING
from ant_byte_env.jax_env import JaxAntByteForagingEnv, JaxObs


class JaxAdversarialAntState(NamedTuple):
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


class JaxAdversarialAntInfo(NamedTuple):
    delivered_food: jax.Array
    remaining_food: jax.Array
    step_count: jax.Array
    pickup_events: jax.Array
    delivery_events: jax.Array
    num_writes: jax.Array
    num_overwrites: jax.Array
    visited_cell_count: jax.Array
    newly_visited_cells: jax.Array
    viewed_cell_count: jax.Array
    newly_viewed_cells: jax.Array
    visible_border_cells: jax.Array


class JaxAdversarialAntByteEnv(JaxAntByteForagingEnv):
    """Two-team variant with shared food/bytes and team-specific hubs."""

    team_count = 2

    def __init__(
        self,
        *,
        num_ants_per_team: int = 1,
        delivery_limit: int | None = None,
        hub_pair_distance: int = 0,
        hub_pair_distance_min: int = 0,
        hub_pair_distance_max: int = 0,
        food_midpoint_window_size: int = 0,
        **kwargs: object,
    ) -> None:
        if int(num_ants_per_team) <= 0:
            raise ValueError("num_ants_per_team must be positive.")
        self.num_ants_per_team = int(num_ants_per_team)
        super().__init__(num_ants=2 * self.num_ants_per_team, **kwargs)
        hub_pair_distance_min, hub_pair_distance_max = self._normalize_hub_distance_range(
            hub_pair_distance=hub_pair_distance,
            hub_pair_distance_min=hub_pair_distance_min,
            hub_pair_distance_max=hub_pair_distance_max,
        )
        if int(food_midpoint_window_size) < 0:
            raise ValueError("food_midpoint_window_size must be non-negative.")
        if int(food_midpoint_window_size) > max(self.width, self.height):
            raise ValueError("food_midpoint_window_size must fit inside the map.")
        if self.open_cell_count < 2:
            raise ValueError("adversarial env requires at least two open hub cells.")
        if self.food_count > 0 and self.open_cell_count < 3:
            raise ValueError("food_count requires an open non-hub tile.")
        if delivery_limit is not None and int(delivery_limit) <= 0:
            raise ValueError("delivery_limit must be positive when provided.")
        self.hub_pair_distance = int(hub_pair_distance)
        self.hub_pair_distance_min = hub_pair_distance_min
        self.hub_pair_distance_max = hub_pair_distance_max
        self.food_midpoint_window_size = int(food_midpoint_window_size)
        self.delivery_limit = None if delivery_limit is None else int(delivery_limit)
        self.team_ids = jnp.repeat(
            jnp.arange(self.team_count, dtype=jnp.int32),
            self.num_ants_per_team,
        )

    def _normalize_hub_distance_range(
        self,
        *,
        hub_pair_distance: int,
        hub_pair_distance_min: int,
        hub_pair_distance_max: int,
    ) -> tuple[int, int]:
        fixed_distance = int(hub_pair_distance)
        min_distance = int(hub_pair_distance_min)
        max_distance = int(hub_pair_distance_max)
        diameter = self.width + self.height - 2
        if fixed_distance < 0 or min_distance < 0 or max_distance < 0:
            raise ValueError("hub pair distances must be non-negative.")
        if fixed_distance > 0 and (min_distance > 0 or max_distance > 0):
            raise ValueError("use either hub_pair_distance or a hub pair distance range.")
        if fixed_distance > 0:
            min_distance = fixed_distance
            max_distance = fixed_distance
        elif min_distance > 0 or max_distance > 0:
            if min_distance == 0:
                min_distance = 1
            if max_distance == 0:
                max_distance = min_distance
        if min_distance > max_distance:
            raise ValueError("hub_pair_distance_min cannot exceed hub_pair_distance_max.")
        if max_distance > diameter:
            raise ValueError("hub pair distance cannot exceed the map Manhattan diameter.")
        return min_distance, max_distance

    def reset(
        self,
        key: jax.Array,
        *,
        hub_pos: jax.Array | None = None,
        food_positions: jax.Array | None = None,
        obstacles: jax.Array | None = None,
        previous_obstacles: jax.Array | None = None,
    ) -> tuple[JaxAdversarialAntState, JaxObs, JaxAdversarialAntInfo]:
        maze_key, hub_key, food_key, ant_key = jax.random.split(key, 4)
        actual_obstacles = self._initial_obstacles(
            maze_key,
            obstacles=obstacles,
            previous_obstacles=previous_obstacles,
        )
        actual_hub_pos = self._initial_hub_positions(hub_key, hub_pos, actual_obstacles)
        food = self._initial_food_grid_two_hubs(
            food_key,
            actual_hub_pos,
            food_positions,
            actual_obstacles,
        )
        ants_pos = self._initial_adversarial_ant_positions(
            ant_key,
            actual_hub_pos,
            food,
            actual_obstacles,
        )
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
        state = JaxAdversarialAntState(
            hub_pos=actual_hub_pos.astype(jnp.int32),
            ants_pos=ants_pos.astype(jnp.int32),
            ants_count=ants_count,
            ants_facing=jnp.full((self.num_ants,), DEFAULT_FACING, dtype=jnp.int32),
            ants_carrying=jnp.zeros((self.num_ants,), dtype=jnp.bool_),
            food=food,
            initial_food=food,
            bytes=jnp.zeros((self.height, self.width), dtype=jnp.uint8),
            delivered_food=jnp.zeros((self.team_count,), dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            initial_food_total=jnp.sum(food).astype(jnp.int32),
            visited_cells=visited_cells,
            viewed_cells=viewed_cells,
            obstacles=actual_obstacles,
        )
        return state, self.observe(state), self.info(
            state,
            pickup_events=jnp.zeros((self.team_count,), dtype=jnp.int32),
            delivery_events=jnp.zeros((self.team_count,), dtype=jnp.int32),
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
        state: JaxAdversarialAntState,
        action: jax.Array,
    ) -> tuple[
        JaxAdversarialAntState,
        JaxObs,
        jax.Array,
        jax.Array,
        jax.Array,
        JaxAdversarialAntInfo,
    ]:
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
            jnp.zeros((self.team_count,), dtype=jnp.int32),
            jnp.zeros((self.team_count,), dtype=jnp.int32),
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
                pickup_events,
                delivery_events,
                num_writes,
                num_overwrites,
            ) = carry
            team = ant_index // self.num_ants_per_team
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
            tile_is_any_hub = jnp.any(jnp.all(next_pos == state.hub_pos, axis=-1))
            tile_is_own_hub = jnp.all(next_pos == state.hub_pos[team])
            was_carrying = ants_carrying[ant_index]

            picked_up = jnp.logical_and(jnp.logical_not(was_carrying), tile_had_food)
            food = food.at[y_pos, x_pos].add(jnp.where(picked_up, -1, 0))
            carrying_after_pickup = jnp.logical_or(was_carrying, picked_up)
            pickup_events = pickup_events.at[team].add(picked_up.astype(jnp.int32))

            delivered = jnp.logical_and(carrying_after_pickup, tile_is_own_hub)
            ants_carrying = ants_carrying.at[ant_index].set(
                jnp.logical_and(carrying_after_pickup, jnp.logical_not(delivered))
            )
            delivered_food = delivered_food.at[team].add(delivered.astype(jnp.int32))
            delivery_events = delivery_events.at[team].add(delivered.astype(jnp.int32))

            wants_write = jnp.logical_or(
                move == ACTION_STAY,
                jnp.asarray(self.write_while_moving, dtype=jnp.bool_),
            )
            can_write = jnp.logical_and(
                wants_write,
                jnp.logical_not(jnp.logical_or(tile_had_food, tile_is_any_hub)),
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
                ant_bit = jnp.left_shift(jnp.asarray(1, dtype=jnp.uint8), ant_bit_index)
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
                pickup_events,
                delivery_events,
                num_writes,
                num_overwrites,
            ), None

        (
            ants_pos,
            ants_facing,
            ants_carrying,
            food,
            bytes_grid,
            delivered_food,
            _,
            pickup_events,
            delivery_events,
            num_writes,
            num_overwrites,
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
        next_state = JaxAdversarialAntState(
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
        no_food_left = jnp.sum(food) <= 0
        no_carried_food = jnp.logical_not(jnp.any(ants_carrying))
        food_terminated = (
            jnp.asarray(self.terminate_on_food_delivery, dtype=jnp.bool_)
            & no_food_left
            & no_carried_food
        )
        if self.delivery_limit is None:
            limit_terminated = jnp.asarray(False, dtype=jnp.bool_)
        else:
            limit_terminated = jnp.any(delivered_food >= int(self.delivery_limit))
        terminated = jnp.logical_or(food_terminated, limit_terminated)
        reward = jnp.asarray(
            [
                delivery_events[0] - delivery_events[1],
                delivery_events[1] - delivery_events[0],
            ],
            dtype=jnp.float32,
        )
        truncated = step_count >= self.max_steps
        info = self.info(
            next_state,
            pickup_events=pickup_events,
            delivery_events=delivery_events,
            num_writes=num_writes,
            num_overwrites=num_overwrites,
            newly_visited_cells=newly_visited_cells,
            newly_viewed_cells=newly_viewed_cells,
            visible_border_cells=visible_border_cells,
        )
        return next_state, self.observe(next_state), reward, terminated, truncated, info

    def observe(self, state: JaxAdversarialAntState) -> JaxObs:
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
        state: JaxAdversarialAntState,
        *,
        pickup_events: jax.Array,
        delivery_events: jax.Array,
        num_writes: int | jax.Array,
        num_overwrites: int | jax.Array,
        newly_visited_cells: int | jax.Array,
        newly_viewed_cells: int | jax.Array,
        visible_border_cells: int | jax.Array,
    ) -> JaxAdversarialAntInfo:
        return JaxAdversarialAntInfo(
            delivered_food=state.delivered_food.astype(jnp.int32),
            remaining_food=jnp.sum(state.food).astype(jnp.int32),
            step_count=state.step_count.astype(jnp.int32),
            pickup_events=jnp.asarray(pickup_events, dtype=jnp.int32),
            delivery_events=jnp.asarray(delivery_events, dtype=jnp.int32),
            num_writes=jnp.asarray(num_writes, dtype=jnp.int32),
            num_overwrites=jnp.asarray(num_overwrites, dtype=jnp.int32),
            visited_cell_count=jnp.sum(state.visited_cells).astype(jnp.int32),
            newly_visited_cells=jnp.asarray(newly_visited_cells, dtype=jnp.int32),
            viewed_cell_count=jnp.sum(state.viewed_cells).astype(jnp.int32),
            newly_viewed_cells=jnp.asarray(newly_viewed_cells, dtype=jnp.int32),
            visible_border_cells=jnp.asarray(visible_border_cells, dtype=jnp.int32),
        )

    def _initial_hub_positions(
        self,
        key: jax.Array,
        hub_pos: jax.Array | None,
        obstacles: jax.Array,
    ) -> jax.Array:
        if hub_pos is not None:
            positions = self._nearest_open_positions(
                jnp.asarray(hub_pos, dtype=jnp.int32).reshape((2, 2)),
                obstacles,
            )
            return self._ensure_distinct_hubs(positions, obstacles)
        if self.hub_pair_distance_max > 0:
            return self._initial_distance_matched_hub_positions(key, obstacles)
        if self.random_hub:
            flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
            flat_x = flat_indices % self.width
            flat_y = flat_indices // self.width
            is_open = jnp.logical_not(obstacles.reshape((-1,)))
            centered_open = is_open & self._inside_hub_center_window(flat_x, flat_y)
            candidate_mask = jnp.where(jnp.sum(centered_open) >= 2, centered_open, is_open)
            interior_open = candidate_mask & self._inside_layout_margin(flat_x, flat_y)
            candidate_mask = jnp.where(jnp.sum(interior_open) >= 2, interior_open, candidate_mask)
            scores = jax.random.uniform(key, is_open.shape, dtype=jnp.float32)
            _, selected_flat = jax.lax.top_k(jnp.where(candidate_mask, scores, -jnp.inf), 2)
            return jnp.stack(
                [selected_flat % self.width, selected_flat // self.width],
                axis=-1,
            ).astype(jnp.int32)
        desired = jnp.asarray(
            [
                [self.width // 4, self.height // 2],
                [(3 * self.width) // 4, self.height // 2],
            ],
            dtype=jnp.int32,
        )
        positions = self._nearest_open_positions(desired, obstacles)
        return self._ensure_distinct_hubs(positions, obstacles)

    def _initial_distance_matched_hub_positions(
        self,
        key: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        if self.random_hub:
            first_key, target_key, second_key = jax.random.split(key, 3)
            flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
            flat_x = flat_indices % self.width
            flat_y = flat_indices // self.width
            candidate_mask = self._hub_candidate_mask(flat_x, flat_y, obstacles)
            first_scores = jax.random.uniform(first_key, flat_x.shape, dtype=jnp.float32)
            first_flat = jnp.argmax(jnp.where(candidate_mask, first_scores, -jnp.inf))
            first_pos = self._flat_to_position(first_flat.astype(jnp.int32))
            target_distance = jax.random.randint(
                target_key,
                (),
                self.hub_pair_distance_min,
                self.hub_pair_distance_max + 1,
                dtype=jnp.int32,
            )
            distances = jnp.abs(flat_x - first_pos[0]) + jnp.abs(flat_y - first_pos[1])
            second_mask = candidate_mask & (flat_indices != first_flat)
            distance_error = jnp.abs(distances - target_distance)
            second_scores = (
                -distance_error.astype(jnp.float32)
                + 1e-3 * jax.random.uniform(second_key, flat_x.shape, dtype=jnp.float32)
            )
            second_flat = jnp.argmax(jnp.where(second_mask, second_scores, -jnp.inf))
            return jnp.stack(
                [first_pos, self._flat_to_position(second_flat.astype(jnp.int32))],
                axis=0,
            ).astype(jnp.int32)

        midpoint_distance = (self.hub_pair_distance_min + self.hub_pair_distance_max) // 2
        distance = min(int(midpoint_distance), self.width - 1)
        left_x = max(0, (self.width - 1 - distance) // 2)
        right_x = min(self.width - 1, left_x + distance)
        y_pos = self.height // 2
        desired = jnp.asarray([[left_x, y_pos], [right_x, y_pos]], dtype=jnp.int32)
        positions = self._nearest_open_positions(desired, obstacles)
        return self._ensure_distinct_hubs(positions, obstacles)

    def _hub_candidate_mask(
        self,
        flat_x: jax.Array,
        flat_y: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        is_open = jnp.logical_not(obstacles.reshape((-1,)))
        centered_open = is_open & self._inside_hub_center_window(flat_x, flat_y)
        candidate_mask = jnp.where(jnp.sum(centered_open) >= 2, centered_open, is_open)
        interior_open = candidate_mask & self._inside_layout_margin(flat_x, flat_y)
        return jnp.where(jnp.sum(interior_open) >= 2, interior_open, candidate_mask)

    def _ensure_distinct_hubs(self, positions: jax.Array, obstacles: jax.Array) -> jax.Array:
        fallback = self._first_open_position_excluding(positions[:1], obstacles)
        same = jnp.all(positions[0] == positions[1])
        return positions.at[1].set(jnp.where(same, fallback, positions[1]))

    def _first_open_position_excluding(
        self,
        excluded_positions: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
        flat_x = flat_indices % self.width
        flat_y = flat_indices // self.width
        excluded = jnp.any(
            (flat_x[:, None] == excluded_positions[None, :, 0])
            & (flat_y[:, None] == excluded_positions[None, :, 1]),
            axis=1,
        )
        candidate_mask = jnp.logical_not(obstacles.reshape((-1,))) & jnp.logical_not(excluded)
        scores = jnp.where(
            candidate_mask,
            (self.width * self.height - flat_indices).astype(jnp.float32),
            -jnp.inf,
        )
        return self._flat_to_position(jnp.argmax(scores).astype(jnp.int32))

    def _initial_food_grid_two_hubs(
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
            fallback_pos = self._first_open_position_excluding(hub_pos, obstacles)
            is_hub = jnp.any(jnp.all(positions[:, None, :] == hub_pos[None, :, :], axis=-1), axis=1)
            positions = jnp.where(is_hub[:, None], fallback_pos[None, :], positions)
        else:
            positions = self._sample_food_positions_two_hubs(key, hub_pos, obstacles)
        return self._distribute_food(positions)

    def _sample_food_positions_two_hubs(
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
        is_hub = jnp.any(
            (flat_x[:, None] == hub_pos[None, :, 0])
            & (flat_y[:, None] == hub_pos[None, :, 1]),
            axis=1,
        )
        candidate_mask = is_open & jnp.logical_not(is_hub)
        interior_mask = candidate_mask & self._inside_layout_margin(flat_x, flat_y)
        enough_interior = jnp.sum(interior_mask.astype(jnp.int32)) >= self.source_count
        candidate_mask = jnp.where(enough_interior, interior_mask, candidate_mask)
        midpoint_mask = candidate_mask & self._inside_food_midpoint_window(
            flat_x,
            flat_y,
            hub_pos,
        )
        enough_midpoint = jnp.sum(midpoint_mask.astype(jnp.int32)) >= self.source_count
        candidate_mask = jnp.where(enough_midpoint, midpoint_mask, candidate_mask)
        if self.random_food:
            scores = jax.random.uniform(key, (self.width * self.height,), dtype=jnp.float32)
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

    def _inside_food_midpoint_window(
        self,
        x_pos: jax.Array,
        y_pos: jax.Array,
        hub_pos: jax.Array,
    ) -> jax.Array:
        window_size = jnp.asarray(self.food_midpoint_window_size, dtype=jnp.int32)
        midpoint_x = (hub_pos[0, 0] + hub_pos[1, 0]) // 2
        midpoint_y = (hub_pos[0, 1] + hub_pos[1, 1]) // 2
        half_before = window_size // 2
        half_after = window_size - half_before
        x_start = jnp.clip(midpoint_x - half_before, 0, self.width)
        y_start = jnp.clip(midpoint_y - half_before, 0, self.height)
        x_end = jnp.clip(midpoint_x + half_after, 0, self.width)
        y_end = jnp.clip(midpoint_y + half_after, 0, self.height)
        in_window = (
            (x_pos >= x_start)
            & (x_pos < x_end)
            & (y_pos >= y_start)
            & (y_pos < y_end)
        )
        return jnp.where(window_size > 0, in_window, jnp.ones_like(in_window, dtype=bool))

    def _initial_adversarial_ant_positions(
        self,
        key: jax.Array,
        hub_pos: jax.Array,
        food: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        del key, food, obstacles
        return jnp.concatenate(
            [
                jnp.repeat(hub_pos[0].reshape(1, 2), self.num_ants_per_team, axis=0),
                jnp.repeat(hub_pos[1].reshape(1, 2), self.num_ants_per_team, axis=0),
            ],
            axis=0,
        )


def reset_batch(
    *,
    args: argparse.Namespace,
    env: JaxAdversarialAntByteEnv,
    key: jax.Array,
) -> tuple[JaxAdversarialAntState, JaxObs]:
    reset_keys = jax.random.split(key, int(args.num_envs))
    states, obs, _ = jax.vmap(env.reset)(reset_keys)
    return states, obs
