"""Pure JAX distance autocurriculum for staged ant byte foraging."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ant_byte_env import DEFAULT_WRITE_BITS
from ant_byte_env.jax_env import JaxAntByteForagingEnv, JaxAntState, JaxObs


class JaxAntDistanceCurriculumState(NamedTuple):
    hub_pos: jax.Array
    ants_pos: jax.Array
    ants_count: jax.Array
    ants_facing: jax.Array
    ants_carrying: jax.Array
    ants_alive: jax.Array
    food: jax.Array
    lethal_food: jax.Array
    initial_food: jax.Array
    bytes: jax.Array
    delivered_food: jax.Array
    stage_delivered_food: jax.Array
    step_count: jax.Array
    stage_step_count: jax.Array
    initial_food_total: jax.Array
    stage_distance: jax.Array
    stage_index: jax.Array
    completed_stages: jax.Array
    visited_cells: jax.Array
    viewed_cells: jax.Array
    obstacles: jax.Array
    rng_key: jax.Array


class JaxAntDistanceCurriculumInfo(NamedTuple):
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
    stage_distance: jax.Array
    stage_index: jax.Array
    stage_delivered_food: jax.Array
    stage_step_count: jax.Array
    completed_stages: jax.Array
    advanced_stage: jax.Array
    completed_stage_distance: jax.Array
    completed_stage_delivered_food: jax.Array


class JaxAntByteDistanceCurriculumEnv:
    """Full-grid foraging with success-conditioned food-distance stages."""

    def __init__(
        self,
        *,
        width: int = 250,
        height: int = 250,
        start_distance: int = 2,
        max_distance: int = 128,
        distance_multiplier: int = 2,
        success_cookies: int | None = None,
        num_ants: int = 4,
        food_count: int = 8,
        food_source_count: int = 1,
        max_steps: int = 60_000,
        random_food: bool = True,
        random_hub: bool = True,
        random_ant_spawn: bool = False,
        random_ant_spawn_radius: int | None = None,
        layout_margin: int = 0,
        hub_center_window_size: int = 0,
        actor_vision_radius: int = 1,
        step_penalty: float = 0.0,
        completion_bonus: float = 0.0,
        write_penalty: float = 0.0,
        write_bits: int = DEFAULT_WRITE_BITS,
        write_while_moving: bool = False,
        per_ant_write_channels: bool = False,
        maze_obstacles: bool = False,
        maze_corridor_width: int = 3,
        maze_wall_width: int = 1,
        maze_seed: int = 0,
        maze_layout_count: int = 64,
        random_wall_obstacles: bool = False,
        random_wall_count_min: int = 1,
        random_wall_count_max: int = 3,
        random_wall_length_min: int = 4,
        random_wall_length_max: int = 14,
        random_wall_width: int = 1,
        random_wall_l_turn_probability: float = 0.5,
        random_wall_center_window_size: int = 0,
    ) -> None:
        self._validate_args(
            width=width,
            height=height,
            start_distance=start_distance,
            max_distance=max_distance,
            distance_multiplier=distance_multiplier,
            success_cookies=success_cookies,
            food_count=food_count,
            food_source_count=food_source_count,
        )
        self.width = int(width)
        self.height = int(height)
        self.start_distance = int(start_distance)
        self.max_distance = int(max_distance)
        self.distance_multiplier = int(distance_multiplier)
        self.success_cookies = int(food_count if success_cookies is None else success_cookies)
        self.num_ants = int(num_ants)
        self.food_count = int(food_count)
        self.food_source_count = int(food_source_count)
        self.max_steps = int(max_steps)
        self.random_food = bool(random_food)
        self.random_hub = bool(random_hub)
        self.completion_bonus = float(completion_bonus)
        self.base_env = JaxAntByteForagingEnv(
            width=width,
            height=height,
            num_ants=num_ants,
            food_count=food_count,
            food_source_count=food_source_count,
            max_steps=max_steps,
            random_food=False,
            random_hub=random_hub,
            random_ant_spawn=random_ant_spawn,
            random_ant_spawn_radius=random_ant_spawn_radius,
            layout_margin=layout_margin,
            hub_center_window_size=hub_center_window_size,
            actor_vision_radius=actor_vision_radius,
            step_penalty=step_penalty,
            completion_bonus=0.0,
            write_penalty=write_penalty,
            write_bits=write_bits,
            write_while_moving=write_while_moving,
            per_ant_write_channels=per_ant_write_channels,
            terminate_on_food_delivery=False,
            terminate_on_full_coverage=False,
            maze_obstacles=maze_obstacles,
            maze_corridor_width=maze_corridor_width,
            maze_wall_width=maze_wall_width,
            maze_seed=maze_seed,
            maze_layout_count=maze_layout_count,
            random_wall_obstacles=random_wall_obstacles,
            random_wall_count_min=random_wall_count_min,
            random_wall_count_max=random_wall_count_max,
            random_wall_length_min=random_wall_length_min,
            random_wall_length_max=random_wall_length_max,
            random_wall_width=random_wall_width,
            random_wall_l_turn_probability=random_wall_l_turn_probability,
            random_wall_center_window_size=random_wall_center_window_size,
        )
        self.source_count = self.base_env.source_count
        self.open_cell_count = self.base_env.open_cell_count
        self.action_nvec = self.base_env.action_nvec
        self.write_value_count = self.base_env.write_value_count
        self.max_write_value = self.base_env.max_write_value
        self.obstacle_bank = self.base_env.obstacle_bank
        self.obstacles = self.base_env.obstacles

    @staticmethod
    def _validate_args(
        *,
        width: int,
        height: int,
        start_distance: int,
        max_distance: int,
        distance_multiplier: int,
        success_cookies: int | None,
        food_count: int,
        food_source_count: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if start_distance <= 0:
            raise ValueError("start_distance must be positive.")
        if max_distance < start_distance:
            raise ValueError("max_distance must be at least start_distance.")
        if distance_multiplier <= 1:
            raise ValueError("distance_multiplier must be greater than 1.")
        if food_count <= 0:
            raise ValueError("food_count must be positive.")
        if success_cookies is not None:
            if success_cookies <= 0:
                raise ValueError("success_cookies must be positive.")
            if success_cookies > food_count:
                raise ValueError("success_cookies cannot exceed food_count.")
        if food_source_count <= 0:
            raise ValueError("food_source_count must be positive.")

    def reset(
        self,
        key: jax.Array,
        *,
        hub_pos: jax.Array | None = None,
        food_positions: jax.Array | None = None,
    ) -> tuple[JaxAntDistanceCurriculumState, JaxObs, JaxAntDistanceCurriculumInfo]:
        rng_key, stage_key = jax.random.split(key)
        state = self._new_stage_state(
            stage_key=stage_key,
            rng_key=rng_key,
            stage_distance=jnp.asarray(self.start_distance, dtype=jnp.int32),
            delivered_food=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            stage_index=jnp.asarray(0, dtype=jnp.int32),
            completed_stages=jnp.asarray(0, dtype=jnp.int32),
            hub_pos=hub_pos,
            food_positions=food_positions,
            obstacles=None,
        )
        return state, self.observe(state), self.info(
            state,
            num_writes=0,
            num_overwrites=0,
            newly_visited_cells=0,
            newly_viewed_cells=0,
            visible_border_cells=self.base_env._count_visible_border_cells(
                state.ants_pos,
                state.obstacles,
            ),
            advanced_stage=False,
            completed_stage_distance=0,
            completed_stage_delivered_food=0,
        )

    def step(
        self,
        state: JaxAntDistanceCurriculumState,
        action: jax.Array,
    ) -> tuple[
        JaxAntDistanceCurriculumState,
        JaxObs,
        jax.Array,
        jax.Array,
        jax.Array,
        JaxAntDistanceCurriculumInfo,
    ]:
        base_state = self._base_state_from_curriculum_state(state)
        next_base, _, reward, _, _, base_info = self.base_env.step(base_state, action)
        step_count = state.step_count + jnp.asarray(1, dtype=jnp.int32)
        stage_step_count = state.stage_step_count + jnp.asarray(1, dtype=jnp.int32)
        stage_delivery_count = next_base.delivered_food - state.stage_delivered_food
        delivered_food = state.delivered_food + stage_delivery_count
        stage_complete = next_base.delivered_food >= jnp.asarray(
            self.success_cookies,
            dtype=jnp.int32,
        )
        final_stage_complete = jnp.logical_and(
            stage_complete,
            state.stage_distance >= jnp.asarray(self.max_distance, dtype=jnp.int32),
        )
        truncated = step_count >= jnp.asarray(self.max_steps, dtype=jnp.int32)
        should_advance = jnp.logical_and(
            stage_complete,
            jnp.logical_not(jnp.logical_or(final_stage_complete, truncated)),
        )
        completed_stages = state.completed_stages + stage_complete.astype(jnp.int32)
        rng_key, stage_key = jax.random.split(state.rng_key)

        interim_state = self._state_from_base_state(
            next_base,
            delivered_food=delivered_food,
            stage_delivered_food=next_base.delivered_food,
            step_count=step_count,
            stage_step_count=stage_step_count,
            stage_distance=state.stage_distance,
            stage_index=state.stage_index,
            completed_stages=completed_stages,
            rng_key=rng_key,
        )

        def advance_stage(_: None) -> JaxAntDistanceCurriculumState:
            return self._new_stage_state(
                stage_key=stage_key,
                rng_key=rng_key,
                stage_distance=self._next_stage_distance(state.stage_distance),
                delivered_food=delivered_food,
                step_count=step_count,
                stage_index=state.stage_index + jnp.asarray(1, dtype=jnp.int32),
                completed_stages=completed_stages,
                hub_pos=state.hub_pos,
                food_positions=None,
                obstacles=state.obstacles,
            )

        next_state = jax.lax.cond(
            should_advance,
            advance_stage,
            lambda _: interim_state,
            operand=None,
        )
        reward = reward + final_stage_complete.astype(jnp.float32) * jnp.asarray(
            self.completion_bonus,
            dtype=jnp.float32,
        )
        info = self.info(
            next_state,
            num_writes=base_info.num_writes,
            num_overwrites=base_info.num_overwrites,
            newly_visited_cells=base_info.newly_visited_cells,
            newly_viewed_cells=base_info.newly_viewed_cells,
            visible_border_cells=base_info.visible_border_cells,
            advanced_stage=should_advance,
            completed_stage_distance=jnp.where(stage_complete, state.stage_distance, 0),
            completed_stage_delivered_food=jnp.where(
                stage_complete,
                next_base.delivered_food,
                0,
            ),
        )
        return next_state, self.observe(next_state), reward, final_stage_complete, truncated, info

    def observe(self, state: JaxAntDistanceCurriculumState) -> JaxObs:
        obs = self.base_env.observe(self._base_state_from_curriculum_state(state))
        return {
            **obs,
            "distance_curriculum_stage_distance": state.stage_distance.astype(jnp.int32),
            "distance_curriculum_stage_index": state.stage_index.astype(jnp.int32),
        }

    def info(
        self,
        state: JaxAntDistanceCurriculumState,
        *,
        num_writes: int | jax.Array,
        num_overwrites: int | jax.Array,
        newly_visited_cells: int | jax.Array,
        newly_viewed_cells: int | jax.Array,
        visible_border_cells: int | jax.Array,
        advanced_stage: bool | jax.Array,
        completed_stage_distance: int | jax.Array,
        completed_stage_delivered_food: int | jax.Array,
    ) -> JaxAntDistanceCurriculumInfo:
        base_info = self.base_env.info(
            self._base_state_from_curriculum_state(state),
            num_writes=num_writes,
            num_overwrites=num_overwrites,
            newly_visited_cells=newly_visited_cells,
            newly_viewed_cells=newly_viewed_cells,
            visible_border_cells=visible_border_cells,
        )
        return JaxAntDistanceCurriculumInfo(
            delivered_food=state.delivered_food.astype(jnp.int32),
            remaining_food=base_info.remaining_food,
            step_count=state.step_count.astype(jnp.int32),
            num_writes=base_info.num_writes,
            num_overwrites=base_info.num_overwrites,
            visited_cell_count=base_info.visited_cell_count,
            newly_visited_cells=base_info.newly_visited_cells,
            viewed_cell_count=base_info.viewed_cell_count,
            newly_viewed_cells=base_info.newly_viewed_cells,
            visible_border_cells=base_info.visible_border_cells,
            stage_distance=state.stage_distance.astype(jnp.int32),
            stage_index=state.stage_index.astype(jnp.int32),
            stage_delivered_food=state.stage_delivered_food.astype(jnp.int32),
            stage_step_count=state.stage_step_count.astype(jnp.int32),
            completed_stages=state.completed_stages.astype(jnp.int32),
            advanced_stage=jnp.asarray(advanced_stage, dtype=jnp.int32),
            completed_stage_distance=jnp.asarray(completed_stage_distance, dtype=jnp.int32),
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
        stage_distance: jax.Array,
        delivered_food: jax.Array,
        step_count: jax.Array,
        stage_index: jax.Array,
        completed_stages: jax.Array,
        hub_pos: jax.Array | None,
        food_positions: jax.Array | None,
        obstacles: jax.Array | None,
    ) -> JaxAntDistanceCurriculumState:
        maze_key, hub_key, food_key, reset_key = jax.random.split(stage_key, 4)
        actual_obstacles = self.base_env._initial_obstacles(
            maze_key,
            obstacles=obstacles,
            previous_obstacles=None,
        )
        actual_hub_pos = self.base_env._initial_hub_pos(
            hub_key,
            hub_pos,
            actual_obstacles,
        )
        actual_food_positions = (
            self._sample_food_positions_at_distance(
                food_key,
                hub_pos=actual_hub_pos,
                obstacles=actual_obstacles,
                distance=stage_distance,
            )
            if food_positions is None
            else jnp.asarray(food_positions, dtype=jnp.int32).reshape((-1, 2))
        )
        base_state, _, _ = self.base_env.reset(
            reset_key,
            hub_pos=actual_hub_pos,
            food_positions=actual_food_positions,
            obstacles=actual_obstacles,
        )
        return self._state_from_base_state(
            base_state,
            delivered_food=delivered_food,
            stage_delivered_food=jnp.asarray(0, dtype=jnp.int32),
            step_count=step_count,
            stage_step_count=jnp.asarray(0, dtype=jnp.int32),
            stage_distance=stage_distance,
            stage_index=stage_index,
            completed_stages=completed_stages,
            rng_key=rng_key,
        )

    def _sample_food_positions_at_distance(
        self,
        key: jax.Array,
        *,
        hub_pos: jax.Array,
        obstacles: jax.Array,
        distance: jax.Array,
    ) -> jax.Array:
        flat_indices = jnp.arange(self.width * self.height, dtype=jnp.int32)
        flat_x = flat_indices % self.width
        flat_y = flat_indices // self.width
        manhattan = jnp.abs(flat_x - hub_pos[0]) + jnp.abs(flat_y - hub_pos[1])
        open_non_hub = jnp.logical_not(obstacles.reshape((-1,))) & (manhattan > 0)
        exact_mask = open_non_hub & (manhattan == distance)
        enough_exact = jnp.sum(exact_mask.astype(jnp.int32)) >= int(self.source_count)
        candidate_mask = jnp.where(enough_exact, exact_mask, open_non_hub)
        random_scores = jax.random.uniform(
            key,
            (self.width * self.height,),
            dtype=jnp.float32,
        )
        deterministic_scores = -flat_indices.astype(jnp.float32) / float(
            max(self.width * self.height, 1)
        )
        base_scores = jnp.where(
            jnp.asarray(self.random_food, dtype=jnp.bool_),
            random_scores,
            deterministic_scores,
        )
        distance_scores = -jnp.abs(manhattan - distance).astype(jnp.float32) * 1_000_000.0
        scores = jnp.where(enough_exact, base_scores, distance_scores + base_scores)
        _, selected_flat = jax.lax.top_k(
            jnp.where(candidate_mask, scores, -jnp.inf),
            int(self.source_count),
        )
        return jnp.stack(
            [selected_flat % self.width, selected_flat // self.width],
            axis=-1,
        ).astype(jnp.int32)

    def _base_state_from_curriculum_state(
        self,
        state: JaxAntDistanceCurriculumState,
    ) -> JaxAntState:
        return JaxAntState(
            hub_pos=state.hub_pos,
            ants_pos=state.ants_pos,
            ants_count=state.ants_count,
            ants_facing=state.ants_facing,
            ants_carrying=state.ants_carrying,
            ants_alive=state.ants_alive,
            food=state.food,
            lethal_food=state.lethal_food,
            initial_food=state.initial_food,
            bytes=state.bytes,
            delivered_food=state.stage_delivered_food,
            step_count=state.stage_step_count,
            initial_food_total=state.initial_food_total,
            visited_cells=state.visited_cells,
            viewed_cells=state.viewed_cells,
            obstacles=state.obstacles,
        )

    def _state_from_base_state(
        self,
        base_state: JaxAntState,
        *,
        delivered_food: jax.Array,
        stage_delivered_food: jax.Array,
        step_count: jax.Array,
        stage_step_count: jax.Array,
        stage_distance: jax.Array,
        stage_index: jax.Array,
        completed_stages: jax.Array,
        rng_key: jax.Array,
    ) -> JaxAntDistanceCurriculumState:
        return JaxAntDistanceCurriculumState(
            hub_pos=base_state.hub_pos,
            ants_pos=base_state.ants_pos,
            ants_count=base_state.ants_count,
            ants_facing=base_state.ants_facing,
            ants_carrying=base_state.ants_carrying,
            ants_alive=base_state.ants_alive,
            food=base_state.food,
            lethal_food=base_state.lethal_food,
            initial_food=base_state.initial_food,
            bytes=base_state.bytes,
            delivered_food=delivered_food.astype(jnp.int32),
            stage_delivered_food=stage_delivered_food.astype(jnp.int32),
            step_count=step_count.astype(jnp.int32),
            stage_step_count=stage_step_count.astype(jnp.int32),
            initial_food_total=base_state.initial_food_total,
            stage_distance=stage_distance.astype(jnp.int32),
            stage_index=stage_index.astype(jnp.int32),
            completed_stages=completed_stages.astype(jnp.int32),
            visited_cells=base_state.visited_cells,
            viewed_cells=base_state.viewed_cells,
            obstacles=base_state.obstacles,
            rng_key=rng_key,
        )

    def _next_stage_distance(self, distance: jax.Array) -> jax.Array:
        return jnp.minimum(
            distance * jnp.asarray(self.distance_multiplier, dtype=jnp.int32),
            jnp.asarray(self.max_distance, dtype=jnp.int32),
        )
