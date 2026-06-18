"""Pure JAX autocurriculum dynamics for staged ant byte foraging."""

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


class JaxAntAutoCurriculumState(NamedTuple):
    hub_pos: jax.Array
    ants_pos: jax.Array
    ants_count: jax.Array
    ants_facing: jax.Array
    ants_carrying: jax.Array
    food: jax.Array
    initial_food: jax.Array
    bytes: jax.Array
    delivered_food: jax.Array
    stage_delivered_food: jax.Array
    step_count: jax.Array
    stage_step_count: jax.Array
    initial_food_total: jax.Array
    active_size: jax.Array
    stage_index: jax.Array
    completed_stages: jax.Array
    rng_key: jax.Array


class JaxAntAutoCurriculumInfo(NamedTuple):
    delivered_food: jax.Array
    remaining_food: jax.Array
    step_count: jax.Array
    num_writes: jax.Array
    num_overwrites: jax.Array
    active_size: jax.Array
    stage_index: jax.Array
    stage_delivered_food: jax.Array
    stage_step_count: jax.Array
    completed_stages: jax.Array
    advanced_stage: jax.Array
    completed_stage_size: jax.Array
    completed_stage_delivered_food: jax.Array


JaxAutoCurriculumObs = dict[str, jax.Array]


class JaxAntByteAutoCurriculumEnv:
    """JIT-friendly autocurriculum for square forage stages."""

    def __init__(
        self,
        *,
        width: int = 50,
        height: int = 50,
        start_size: int = 4,
        success_cookies: int = 6,
        num_ants: int = 4,
        food_count: int = 12,
        food_source_count: int = 2,
        max_steps: int = 500,
        random_food: bool = True,
        random_hub: bool = False,
        step_penalty: float = 0.0,
        write_penalty: float = 0.0,
        write_bits: int = DEFAULT_WRITE_BITS,
        write_while_moving: bool = False,
        actor_vision_radius: int = DEFAULT_ACTOR_VISION_DEPTH,
    ) -> None:
        self._validate_args(
            width=width,
            height=height,
            start_size=start_size,
            success_cookies=success_cookies,
            num_ants=num_ants,
            food_count=food_count,
            food_source_count=food_source_count,
            max_steps=max_steps,
            step_penalty=step_penalty,
            write_penalty=write_penalty,
            write_bits=write_bits,
            actor_vision_radius=actor_vision_radius,
        )
        self.width = int(width)
        self.height = int(height)
        self.start_size = int(start_size)
        self.success_cookies = int(success_cookies)
        self.num_ants = int(num_ants)
        self.food_count = int(food_count)
        self.food_source_count = int(food_source_count)
        self.max_steps = int(max_steps)
        self.random_food = bool(random_food)
        self.random_hub = bool(random_hub)
        self.step_penalty = float(step_penalty)
        self.write_penalty = float(write_penalty)
        self.write_bits = int(write_bits)
        self.write_while_moving = bool(write_while_moving)
        self.actor_vision_radius = int(actor_vision_radius)
        self.write_value_count = write_value_count(self.write_bits)
        self.max_write_value = max_write_value(self.write_bits)
        self.action_nvec = jnp.asarray(
            [MOVEMENT_ACTION_COUNT, self.write_value_count] * self.num_ants,
            dtype=jnp.int32,
        )
        self._flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
        self._flat_x = self._flat_indices % self.width
        self._flat_y = self._flat_indices // self.width
        self._food_per_source = self.food_count // self.food_source_count

    @staticmethod
    def _validate_args(
        *,
        width: int,
        height: int,
        start_size: int,
        success_cookies: int,
        num_ants: int,
        food_count: int,
        food_source_count: int,
        max_steps: int,
        step_penalty: float,
        write_penalty: float,
        write_bits: int,
        actor_vision_radius: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if width != height:
            raise ValueError("autocurriculum currently requires square max grids.")
        if start_size <= 0:
            raise ValueError("start_size must be positive.")
        if start_size > width:
            raise ValueError("start_size must be no larger than the max grid.")
        if success_cookies <= 0:
            raise ValueError("success_cookies must be positive.")
        if num_ants <= 0:
            raise ValueError("num_ants must be positive.")
        if food_source_count != 2:
            raise ValueError("autocurriculum stages use exactly two food sources.")
        if food_count != success_cookies * food_source_count:
            raise ValueError("food_count must equal success_cookies * food_source_count.")
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
        if actor_vision_radius < 0:
            raise ValueError("actor_vision_radius must be non-negative.")
        _validate_far_food_capacity(
            size=int(start_size),
            source_count=int(food_source_count),
            vision_radius=int(actor_vision_radius),
        )

    def reset(
        self,
        key: jax.Array,
        *,
        hub_pos: jax.Array | None = None,
        food_positions: jax.Array | None = None,
    ) -> tuple[JaxAntAutoCurriculumState, JaxAutoCurriculumObs, JaxAntAutoCurriculumInfo]:
        rng_key, stage_key = jax.random.split(key)
        state = self._new_stage_state(
            stage_key=stage_key,
            rng_key=rng_key,
            active_size=jnp.asarray(self.start_size, dtype=jnp.int32),
            delivered_food=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            stage_index=jnp.asarray(0, dtype=jnp.int32),
            completed_stages=jnp.asarray(0, dtype=jnp.int32),
            hub_pos=hub_pos,
            food_positions=food_positions,
        )
        return state, self.observe(state), self.info(
            state,
            num_writes=0,
            num_overwrites=0,
            advanced_stage=False,
            completed_stage_size=0,
            completed_stage_delivered_food=0,
        )

    def step(
        self,
        state: JaxAntAutoCurriculumState,
        action: jax.Array,
    ) -> tuple[
        JaxAntAutoCurriculumState,
        JaxAutoCurriculumObs,
        jax.Array,
        jax.Array,
        jax.Array,
        JaxAntAutoCurriculumInfo,
    ]:
        flat_action = jnp.asarray(action, dtype=jnp.int32).reshape((2 * self.num_ants,))
        written_mask = jnp.zeros((self.height, self.width), dtype=jnp.bool_)
        init_carry = (
            state.ants_pos,
            state.ants_facing,
            state.ants_carrying,
            state.food,
            state.bytes,
            jnp.asarray(0, dtype=jnp.int32),
            written_mask,
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
                delivery_count,
                written_tiles,
                num_writes,
                num_overwrites,
            ) = carry
            action_start = 2 * ant_index
            move = flat_action[action_start]
            write_value = flat_action[action_start + 1].astype(jnp.uint8)
            next_facing = self._action_facing(ants_facing[ant_index], move)
            next_pos = self._move_position(
                ants_pos[ant_index],
                action=move,
                active_size=state.active_size,
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
                delivery_count,
                written_tiles,
                num_writes,
                num_overwrites,
            ), None

        (
            ants_pos,
            ants_facing,
            ants_carrying,
            food,
            bytes_grid,
            delivery_count,
            _,
            num_writes,
            num_overwrites,
        ), _ = jax.lax.scan(scan_ant, init_carry, jnp.arange(self.num_ants))

        step_count = state.step_count + jnp.asarray(1, dtype=jnp.int32)
        stage_step_count = state.stage_step_count + jnp.asarray(1, dtype=jnp.int32)
        delivered_food = state.delivered_food + delivery_count
        stage_delivered_food = state.stage_delivered_food + delivery_count
        stage_complete = stage_delivered_food >= self.success_cookies
        final_stage_complete = jnp.logical_and(stage_complete, state.active_size >= self.width)
        truncated = step_count >= self.max_steps
        should_advance = jnp.logical_and(
            stage_complete,
            jnp.logical_not(jnp.logical_or(final_stage_complete, truncated)),
        )
        completed_stages = state.completed_stages + stage_complete.astype(jnp.int32)
        rng_key, stage_key = jax.random.split(state.rng_key)

        interim_state = JaxAntAutoCurriculumState(
            hub_pos=state.hub_pos,
            ants_pos=ants_pos,
            ants_count=self._build_ants_count_grid(ants_pos),
            ants_facing=ants_facing,
            ants_carrying=ants_carrying,
            food=food,
            initial_food=state.initial_food,
            bytes=bytes_grid,
            delivered_food=delivered_food,
            stage_delivered_food=stage_delivered_food,
            step_count=step_count,
            stage_step_count=stage_step_count,
            initial_food_total=state.initial_food_total,
            active_size=state.active_size,
            stage_index=state.stage_index,
            completed_stages=completed_stages,
            rng_key=rng_key,
        )

        def advance_stage(_: None) -> JaxAntAutoCurriculumState:
            return self._new_stage_state(
                stage_key=stage_key,
                rng_key=rng_key,
                active_size=state.active_size + jnp.asarray(1, dtype=jnp.int32),
                delivered_food=delivered_food,
                step_count=step_count,
                stage_index=state.stage_index + jnp.asarray(1, dtype=jnp.int32),
                completed_stages=completed_stages,
                hub_pos=None,
                food_positions=None,
            )

        next_state = jax.lax.cond(
            should_advance,
            advance_stage,
            lambda _: interim_state,
            operand=None,
        )
        reward = (
            delivery_count.astype(jnp.float32)
            - jnp.asarray(self.step_penalty * self.num_ants, dtype=jnp.float32)
            - num_writes.astype(jnp.float32) * jnp.asarray(self.write_penalty, dtype=jnp.float32)
        )
        info = self.info(
            next_state,
            num_writes=num_writes,
            num_overwrites=num_overwrites,
            advanced_stage=should_advance,
            completed_stage_size=jnp.where(stage_complete, state.active_size, 0),
            completed_stage_delivered_food=jnp.where(stage_complete, stage_delivered_food, 0),
        )
        return next_state, self.observe(next_state), reward, final_stage_complete, truncated, info

    def observe(self, state: JaxAntAutoCurriculumState) -> JaxAutoCurriculumObs:
        return {
            "ants_pos": state.ants_pos.astype(jnp.int32),
            "ants_count": state.ants_count.astype(jnp.int32),
            "ants_carrying": state.ants_carrying.astype(jnp.int8),
            "ants_facing": state.ants_facing.astype(jnp.int32),
            "food": state.food.astype(jnp.int32),
            "bytes": state.bytes.astype(jnp.uint8),
            "hub_pos": state.hub_pos.astype(jnp.int32),
            "active_grid_size": jnp.stack([state.active_size, state.active_size]).astype(jnp.int32),
        }

    def info(
        self,
        state: JaxAntAutoCurriculumState,
        *,
        num_writes: int | jax.Array,
        num_overwrites: int | jax.Array,
        advanced_stage: bool | jax.Array,
        completed_stage_size: int | jax.Array,
        completed_stage_delivered_food: int | jax.Array,
    ) -> JaxAntAutoCurriculumInfo:
        return JaxAntAutoCurriculumInfo(
            delivered_food=state.delivered_food.astype(jnp.int32),
            remaining_food=jnp.sum(state.food).astype(jnp.int32),
            step_count=state.step_count.astype(jnp.int32),
            num_writes=jnp.asarray(num_writes, dtype=jnp.int32),
            num_overwrites=jnp.asarray(num_overwrites, dtype=jnp.int32),
            active_size=state.active_size.astype(jnp.int32),
            stage_index=state.stage_index.astype(jnp.int32),
            stage_delivered_food=state.stage_delivered_food.astype(jnp.int32),
            stage_step_count=state.stage_step_count.astype(jnp.int32),
            completed_stages=state.completed_stages.astype(jnp.int32),
            advanced_stage=jnp.asarray(advanced_stage, dtype=jnp.int32),
            completed_stage_size=jnp.asarray(completed_stage_size, dtype=jnp.int32),
            completed_stage_delivered_food=jnp.asarray(
                completed_stage_delivered_food,
                dtype=jnp.int32,
            ),
        )

    def _new_stage_state(
        self,
        *,
        stage_key: jax.Array,
        rng_key: jax.Array,
        active_size: jax.Array,
        delivered_food: jax.Array,
        step_count: jax.Array,
        stage_index: jax.Array,
        completed_stages: jax.Array,
        hub_pos: jax.Array | None,
        food_positions: jax.Array | None,
    ) -> JaxAntAutoCurriculumState:
        hub_key, food_key = jax.random.split(stage_key)
        actual_hub_pos = self._initial_hub_pos(hub_key, active_size, hub_pos)
        food = self._initial_food_grid(food_key, active_size, actual_hub_pos, food_positions)
        ants_pos = jnp.repeat(actual_hub_pos.reshape(1, 2), self.num_ants, axis=0)
        return JaxAntAutoCurriculumState(
            hub_pos=actual_hub_pos,
            ants_pos=ants_pos.astype(jnp.int32),
            ants_count=self._build_ants_count_grid(ants_pos),
            ants_facing=jnp.full((self.num_ants,), DEFAULT_FACING, dtype=jnp.int32),
            ants_carrying=jnp.zeros((self.num_ants,), dtype=jnp.bool_),
            food=food,
            initial_food=food,
            bytes=jnp.zeros((self.height, self.width), dtype=jnp.uint8),
            delivered_food=delivered_food.astype(jnp.int32),
            stage_delivered_food=jnp.asarray(0, dtype=jnp.int32),
            step_count=step_count.astype(jnp.int32),
            stage_step_count=jnp.asarray(0, dtype=jnp.int32),
            initial_food_total=jnp.sum(food).astype(jnp.int32),
            active_size=active_size.astype(jnp.int32),
            stage_index=stage_index.astype(jnp.int32),
            completed_stages=completed_stages.astype(jnp.int32),
            rng_key=rng_key,
        )

    def _initial_hub_pos(
        self,
        key: jax.Array,
        active_size: jax.Array,
        hub_pos: jax.Array | None,
    ) -> jax.Array:
        if hub_pos is not None:
            return jnp.asarray(hub_pos, dtype=jnp.int32)
        if self.random_hub:
            x_key, y_key = jax.random.split(key)
            return jnp.stack(
                [
                    jax.random.randint(x_key, (), 0, active_size, dtype=jnp.int32),
                    jax.random.randint(y_key, (), 0, active_size, dtype=jnp.int32),
                ]
            )
        center = active_size // jnp.asarray(2, dtype=jnp.int32)
        return jnp.stack([center, center]).astype(jnp.int32)

    def _initial_food_grid(
        self,
        key: jax.Array,
        active_size: jax.Array,
        hub_pos: jax.Array,
        food_positions: jax.Array | None,
    ) -> jax.Array:
        if food_positions is not None:
            positions = jnp.asarray(food_positions, dtype=jnp.int32).reshape(
                (self.food_source_count, 2)
            )
        else:
            positions = self._sample_food_positions(key, active_size, hub_pos)
        amounts = jnp.full((self.food_source_count,), self._food_per_source, dtype=jnp.int32)
        return jnp.zeros((self.height, self.width), dtype=jnp.int32).at[
            positions[:, 1],
            positions[:, 0],
        ].add(amounts)

    def _sample_food_positions(
        self,
        key: jax.Array,
        active_size: jax.Array,
        hub_pos: jax.Array,
    ) -> jax.Array:
        inside_active = (self._flat_x < active_size) & (self._flat_y < active_size)
        is_hub = (self._flat_x == hub_pos[0]) & (self._flat_y == hub_pos[1])
        visible_from_start = jnp.maximum(
            jnp.abs(self._flat_x - hub_pos[0]),
            jnp.abs(self._flat_y - hub_pos[1]),
        ) <= self.actor_vision_radius
        candidate_mask = inside_active & jnp.logical_not(is_hub | visible_from_start)
        if self.random_food:
            scores = jax.random.uniform(key, (self.width * self.height,), dtype=jnp.float32)
        else:
            scores = (self.width * self.height - self._flat_indices).astype(jnp.float32)
        scores = jnp.where(candidate_mask, scores, -jnp.inf)
        _, selected = jax.lax.top_k(scores, self.food_source_count)
        return jnp.stack([self._flat_x[selected], self._flat_y[selected]], axis=-1).astype(jnp.int32)

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
        active_size: jax.Array,
    ) -> jax.Array:
        dx = jnp.where(action == ACTION_RIGHT, 1, jnp.where(action == ACTION_LEFT, -1, 0))
        dy = jnp.where(action == ACTION_DOWN, 1, jnp.where(action == ACTION_UP, -1, 0))
        should_advance = (action != ACTION_STAY).astype(jnp.int32)
        return jnp.where(
            action == ACTION_STAY,
            position,
            jnp.asarray(
                [
                    jnp.clip(position[0] + should_advance * dx, 0, active_size - 1),
                    jnp.clip(position[1] + should_advance * dy, 0, active_size - 1),
                ],
                dtype=jnp.int32,
            ),
        )

    def _build_ants_count_grid(self, ants_pos: jax.Array) -> jax.Array:
        ants_pos = ants_pos.astype(jnp.int32)
        return jnp.zeros((self.height, self.width), dtype=jnp.int32).at[
            ants_pos[:, 1],
            ants_pos[:, 0],
        ].add(1)


def _validate_far_food_capacity(
    *,
    size: int,
    source_count: int,
    vision_radius: int,
) -> None:
    for hub_y in range(size):
        for hub_x in range(size):
            candidates = 0
            for y_pos in range(size):
                for x_pos in range(size):
                    if (x_pos, y_pos) == (hub_x, hub_y):
                        continue
                    if max(abs(x_pos - hub_x), abs(y_pos - hub_y)) <= vision_radius:
                        continue
                    candidates += 1
            if candidates < source_count:
                raise ValueError(
                    "start_size is too small to place two food sources outside initial actor vision."
                )
