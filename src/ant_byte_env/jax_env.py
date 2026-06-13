"""Pure JAX core for ant byte foraging dynamics."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env.env import (
    ACTION_FORWARD,
    ACTION_STAY,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
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


class JaxAntInfo(NamedTuple):
    delivered_food: jax.Array
    remaining_food: jax.Array
    step_count: jax.Array
    num_writes: jax.Array
    num_overwrites: jax.Array


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
        step_penalty: float = 0.0,
        write_penalty: float = 0.0,
        write_bits: int = DEFAULT_WRITE_BITS,
    ) -> None:
        self._validate_args(
            width=width,
            height=height,
            num_ants=num_ants,
            food_count=food_count,
            food_source_count=food_source_count,
            max_steps=max_steps,
            step_penalty=step_penalty,
            write_penalty=write_penalty,
            write_bits=write_bits,
        )
        self.width = int(width)
        self.height = int(height)
        self.num_ants = int(num_ants)
        self.food_count = int(food_count)
        self.food_source_count = int(food_source_count)
        self.source_count = min(self.food_source_count, max(self.width * self.height - 1, 0))
        self.max_steps = int(max_steps)
        self.random_food = bool(random_food)
        self.step_penalty = float(step_penalty)
        self.write_penalty = float(write_penalty)
        self.write_bits = int(write_bits)
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
        write_penalty: float,
        write_bits: int,
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
        if write_penalty < 0:
            raise ValueError("write_penalty must be non-negative.")
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
    ) -> tuple[JaxAntState, JaxObs, JaxAntInfo]:
        """Return ``(state, obs, info)`` for a new episode."""

        actual_hub_pos = self._initial_hub_pos(hub_pos)
        food = self._initial_food_grid(key, actual_hub_pos, food_positions)
        ants_pos = jnp.repeat(actual_hub_pos.reshape(1, 2), self.num_ants, axis=0)
        ants_count = self._build_ants_count_grid(ants_pos)
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
        )
        return state, self.observe(state), self.info(state, num_writes=0, num_overwrites=0)

    def step(
        self,
        state: JaxAntState,
        action: jax.Array,
    ) -> tuple[JaxAntState, JaxObs, jax.Array, jax.Array, jax.Array, JaxAntInfo]:
        """Advance one step with an interleaved ``(move, write_value)`` action vector."""

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

            move = flat_action[2 * ant_index]
            write_value = flat_action[2 * ant_index + 1].astype(jnp.uint8)
            next_facing = self._turn_facing(ants_facing[ant_index], move)
            next_pos = self._move_position(ants_pos[ant_index], action=move, facing=next_facing)
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

            can_write = jnp.logical_not(jnp.logical_or(tile_had_food, tile_is_hub))
            already_written = written_tiles[y_pos, x_pos]
            num_overwrites = num_overwrites + jnp.logical_and(can_write, already_written).astype(
                jnp.int32
            )
            written_tiles = written_tiles.at[y_pos, x_pos].set(
                jnp.logical_or(already_written, can_write)
            )
            current_value = bytes_grid[y_pos, x_pos]
            bytes_grid = bytes_grid.at[y_pos, x_pos].set(
                jnp.where(can_write, write_value, current_value)
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
        )
        reward = (
            delivery_count.astype(jnp.float32)
            - jnp.asarray(self.step_penalty * self.num_ants, dtype=jnp.float32)
            - num_writes.astype(jnp.float32) * jnp.asarray(self.write_penalty, dtype=jnp.float32)
        )
        terminated = delivered_food >= state.initial_food_total
        truncated = step_count >= self.max_steps
        info = self.info(next_state, num_writes=num_writes, num_overwrites=num_overwrites)
        return next_state, self.observe(next_state), reward, terminated, truncated, info

    def observe(self, state: JaxAntState) -> JaxObs:
        return {
            "ants_pos": state.ants_pos.astype(jnp.int32),
            "ants_count": state.ants_count.astype(jnp.int32),
            "ants_carrying": state.ants_carrying.astype(jnp.int8),
            "ants_facing": state.ants_facing.astype(jnp.int32),
            "food": state.food.astype(jnp.int32),
            "bytes": state.bytes.astype(jnp.uint8),
            "hub_pos": state.hub_pos.astype(jnp.int32),
        }

    def info(
        self,
        state: JaxAntState,
        *,
        num_writes: int | jax.Array,
        num_overwrites: int | jax.Array,
    ) -> JaxAntInfo:
        return JaxAntInfo(
            delivered_food=state.delivered_food.astype(jnp.int32),
            remaining_food=jnp.sum(state.food).astype(jnp.int32),
            step_count=state.step_count.astype(jnp.int32),
            num_writes=jnp.asarray(num_writes, dtype=jnp.int32),
            num_overwrites=jnp.asarray(num_overwrites, dtype=jnp.int32),
        )

    def _initial_hub_pos(self, hub_pos: jax.Array | None) -> jax.Array:
        if hub_pos is not None:
            return jnp.asarray(hub_pos, dtype=jnp.int32)
        return jnp.asarray([self.width // 2, self.height // 2], dtype=jnp.int32)

    def _initial_food_grid(
        self,
        key: jax.Array,
        hub_pos: jax.Array,
        food_positions: jax.Array | None,
    ) -> jax.Array:
        if self.food_count == 0:
            return jnp.zeros((self.height, self.width), dtype=jnp.int32)

        if food_positions is not None:
            positions = jnp.asarray(food_positions, dtype=jnp.int32).reshape((-1, 2))
        else:
            positions = self._sample_food_positions(key, hub_pos)
        return self._distribute_food(positions)

    def _sample_food_positions(self, key: jax.Array, hub_pos: jax.Array) -> jax.Array:
        if self.source_count == 0:
            return jnp.zeros((0, 2), dtype=jnp.int32)

        candidate_count = self.width * self.height - 1
        if self.random_food:
            raw_indices = jax.random.choice(
                key,
                candidate_count,
                shape=(self.source_count,),
                replace=False,
            )
        else:
            raw_indices = jnp.arange(self.source_count, dtype=jnp.int32)

        hub_flat = hub_pos[1] * self.width + hub_pos[0]
        flat_indices = raw_indices + (raw_indices >= hub_flat).astype(jnp.int32)
        return jnp.stack(
            [flat_indices % self.width, flat_indices // self.width],
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

    def _turn_facing(self, facing: jax.Array, action: jax.Array) -> jax.Array:
        valid_facing = (
            (facing == MOVE_UP)
            | (facing == MOVE_RIGHT)
            | (facing == MOVE_DOWN)
            | (facing == MOVE_LEFT)
        )
        current_facing = jnp.where(valid_facing, facing, DEFAULT_FACING)
        turn_left = jnp.where(
            current_facing == MOVE_UP,
            MOVE_LEFT,
            jnp.where(
                current_facing == MOVE_LEFT,
                MOVE_DOWN,
                jnp.where(current_facing == MOVE_DOWN, MOVE_RIGHT, MOVE_UP),
            ),
        )
        turn_right = jnp.where(
            current_facing == MOVE_UP,
            MOVE_RIGHT,
            jnp.where(
                current_facing == MOVE_RIGHT,
                MOVE_DOWN,
                jnp.where(current_facing == MOVE_DOWN, MOVE_LEFT, MOVE_UP),
            ),
        )
        return jnp.where(
            action == ACTION_TURN_LEFT,
            turn_left,
            jnp.where(action == ACTION_TURN_RIGHT, turn_right, current_facing),
        ).astype(jnp.int32)

    def _move_position(
        self,
        position: jax.Array,
        *,
        action: jax.Array,
        facing: jax.Array,
    ) -> jax.Array:
        dx = jnp.where(facing == MOVE_RIGHT, 1, jnp.where(facing == MOVE_LEFT, -1, 0))
        dy = jnp.where(facing == MOVE_DOWN, 1, jnp.where(facing == MOVE_UP, -1, 0))
        should_advance = (action == ACTION_FORWARD).astype(jnp.int32)
        return jnp.asarray(
            [
                jnp.clip(position[0] + should_advance * dx, 0, self.width - 1),
                jnp.clip(position[1] + should_advance * dy, 0, self.height - 1),
            ],
            dtype=jnp.int32,
        )

    def _build_ants_count_grid(self, ants_pos: jax.Array) -> jax.Array:
        ants_pos = ants_pos.astype(jnp.int32)
        return jnp.zeros((self.height, self.width), dtype=jnp.int32).at[
            ants_pos[:, 1],
            ants_pos[:, 0],
        ].add(1)
