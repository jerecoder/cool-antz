"""Timed-release wrapper for the cooperative JAX AntByte environment."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ant_byte_env import ACTION_STAY
from ant_byte_env.env import DEFAULT_FACING
from ant_byte_env.jax_env import JaxAntByteForagingEnv, JaxAntState, JaxObs
from ant_byte_env.training.jax_mappo.observations import build_forward_vision_offsets


class TimedReleaseState(NamedTuple):
    base: JaxAntState
    active_mask: jax.Array


class TimedReleaseInfo(NamedTuple):
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
    active_mask: jax.Array
    newly_active_mask: jax.Array
    pickup_events_per_ant: jax.Array
    delivery_events_per_ant: jax.Array
    write_attempts_per_ant: jax.Array


class TimedReleaseJaxEnv:
    """Wrap ``JaxAntByteForagingEnv`` with fixed-rank timed ant release."""

    def __init__(
        self,
        base_env: JaxAntByteForagingEnv,
        *,
        release_interval: int = 150,
        initial_active_ants: int = 1,
    ) -> None:
        if release_interval <= 0:
            raise ValueError("release_interval must be positive.")
        if initial_active_ants <= 0:
            raise ValueError("initial_active_ants must be positive.")
        if initial_active_ants > base_env.num_ants:
            raise ValueError("initial_active_ants cannot exceed num_ants.")
        self.base_env = base_env
        self.release_interval = int(release_interval)
        self.initial_active_ants = int(initial_active_ants)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    @property
    def release_steps(self) -> jax.Array:
        ranks = jnp.arange(self.num_ants, dtype=jnp.int32)
        delayed = (ranks - self.initial_active_ants + 1) * self.release_interval
        return jnp.maximum(delayed, 0)

    def active_mask_for_step(self, step_count: jax.Array) -> jax.Array:
        active_count = self.initial_active_ants + step_count // self.release_interval
        active_count = jnp.minimum(active_count, self.num_ants)
        return jnp.arange(self.num_ants, dtype=jnp.int32) < active_count

    def reset(
        self,
        key: jax.Array,
        *,
        hub_pos: jax.Array | None = None,
        food_positions: jax.Array | None = None,
        obstacles: jax.Array | None = None,
        previous_obstacles: jax.Array | None = None,
    ) -> tuple[TimedReleaseState, JaxObs, TimedReleaseInfo]:
        base_state, _, _ = self.base_env.reset(
            key,
            hub_pos=hub_pos,
            food_positions=food_positions,
            obstacles=obstacles,
            previous_obstacles=previous_obstacles,
        )
        hub_positions = jnp.repeat(base_state.hub_pos.reshape(1, 2), self.num_ants, axis=0)
        active_mask = self.active_mask_for_step(base_state.step_count)
        base_state = base_state._replace(
            ants_pos=hub_positions.astype(jnp.int32),
            ants_count=self._active_ants_count_grid(hub_positions, active_mask),
            ants_facing=jnp.full((self.num_ants,), DEFAULT_FACING, dtype=jnp.int32),
            ants_carrying=jnp.zeros((self.num_ants,), dtype=jnp.bool_),
        )
        visited = self._mark_visited_active(base_state.ants_pos, active_mask, base_state.obstacles)
        viewed = self._mark_viewed_active(
            base_state.ants_pos,
            base_state.ants_facing,
            active_mask,
            base_state.obstacles,
        )
        base_state = base_state._replace(visited_cells=visited, viewed_cells=viewed)
        state = TimedReleaseState(base=base_state, active_mask=active_mask)
        info = self.info(
            state,
            num_writes=0,
            num_overwrites=0,
            newly_visited_cells=jnp.sum(visited).astype(jnp.int32),
            newly_viewed_cells=jnp.sum(viewed).astype(jnp.int32),
            visible_border_cells=self._count_visible_border_cells_active(
                base_state.ants_pos,
                base_state.ants_facing,
                active_mask,
                base_state.obstacles,
            ),
            newly_active_mask=jnp.zeros((self.num_ants,), dtype=jnp.bool_),
            pickup_events_per_ant=jnp.zeros((self.num_ants,), dtype=jnp.float32),
            delivery_events_per_ant=jnp.zeros((self.num_ants,), dtype=jnp.float32),
            write_attempts_per_ant=jnp.zeros((self.num_ants,), dtype=jnp.float32),
        )
        return state, self.observe(state), info

    def step(
        self,
        state: TimedReleaseState,
        action: jax.Array,
    ) -> tuple[TimedReleaseState, JaxObs, jax.Array, jax.Array, jax.Array, TimedReleaseInfo]:
        active_before = state.active_mask.astype(jnp.bool_)
        flat_action = jnp.asarray(action, dtype=jnp.int32).reshape((self.num_ants, 2))
        inactive_actions = jnp.zeros_like(flat_action).at[:, 0].set(ACTION_STAY)
        masked_actions = jnp.where(active_before[:, None], flat_action, inactive_actions)
        write_attempts_per_ant = self._write_attempts_per_ant(state.base, masked_actions)
        previous_carrying = state.base.ants_carrying.astype(jnp.bool_)

        next_base, _, reward, terminated, truncated, base_info = self.base_env.step(
            state.base,
            masked_actions.reshape((2 * self.num_ants,)),
        )
        active_after = self.active_mask_for_step(next_base.step_count)
        newly_active_mask = jnp.logical_and(active_after, jnp.logical_not(active_before))

        step_visited = self._mark_visited_active(
            next_base.ants_pos,
            active_after,
            next_base.obstacles,
        )
        visited = jnp.logical_or(state.base.visited_cells, step_visited)
        newly_visited_cells = jnp.sum(
            jnp.logical_and(step_visited, jnp.logical_not(state.base.visited_cells))
        ).astype(jnp.int32)
        step_viewed = self._mark_viewed_active(
            next_base.ants_pos,
            next_base.ants_facing,
            active_after,
            next_base.obstacles,
        )
        viewed = jnp.logical_or(state.base.viewed_cells, step_viewed)
        newly_viewed_cells = jnp.sum(
            jnp.logical_and(step_viewed, jnp.logical_not(state.base.viewed_cells))
        ).astype(jnp.int32)
        ants_count = self._active_ants_count_grid(next_base.ants_pos, active_after)
        next_base = next_base._replace(
            ants_count=ants_count,
            visited_cells=visited,
            viewed_cells=viewed,
        )
        next_state = TimedReleaseState(base=next_base, active_mask=active_after)
        pickup_events = jnp.logical_and(
            active_before,
            jnp.logical_and(jnp.logical_not(previous_carrying), next_base.ants_carrying),
        )
        delivery_events = jnp.logical_and(
            active_before,
            jnp.logical_and(previous_carrying, jnp.logical_not(next_base.ants_carrying)),
        )
        info = self.info(
            next_state,
            num_writes=base_info.num_writes,
            num_overwrites=base_info.num_overwrites,
            newly_visited_cells=newly_visited_cells,
            newly_viewed_cells=newly_viewed_cells,
            visible_border_cells=self._count_visible_border_cells_active(
                next_base.ants_pos,
                next_base.ants_facing,
                active_after,
                next_base.obstacles,
            ),
            newly_active_mask=newly_active_mask,
            pickup_events_per_ant=pickup_events.astype(jnp.float32),
            delivery_events_per_ant=delivery_events.astype(jnp.float32),
            write_attempts_per_ant=write_attempts_per_ant.astype(jnp.float32),
        )
        return next_state, self.observe(next_state), reward, terminated, truncated, info

    def observe(self, state: TimedReleaseState) -> JaxObs:
        obs = self.base_env.observe(state.base)
        return {
            **obs,
            "ants_count": self._active_ants_count_grid(
                state.base.ants_pos,
                state.active_mask,
            ).astype(jnp.int32),
            "active_mask": state.active_mask.astype(jnp.bool_),
            "release_steps": self.release_steps.astype(jnp.int32),
        }

    def info(
        self,
        state: TimedReleaseState,
        *,
        num_writes: int | jax.Array,
        num_overwrites: int | jax.Array,
        newly_visited_cells: int | jax.Array,
        newly_viewed_cells: int | jax.Array,
        visible_border_cells: int | jax.Array,
        newly_active_mask: jax.Array,
        pickup_events_per_ant: jax.Array,
        delivery_events_per_ant: jax.Array,
        write_attempts_per_ant: jax.Array,
    ) -> TimedReleaseInfo:
        base_info = self.base_env.info(
            state.base,
            num_writes=num_writes,
            num_overwrites=num_overwrites,
            newly_visited_cells=newly_visited_cells,
            newly_viewed_cells=newly_viewed_cells,
            visible_border_cells=visible_border_cells,
        )
        return TimedReleaseInfo(
            delivered_food=base_info.delivered_food,
            remaining_food=base_info.remaining_food,
            step_count=base_info.step_count,
            num_writes=base_info.num_writes,
            num_overwrites=base_info.num_overwrites,
            visited_cell_count=base_info.visited_cell_count,
            newly_visited_cells=base_info.newly_visited_cells,
            viewed_cell_count=base_info.viewed_cell_count,
            newly_viewed_cells=base_info.newly_viewed_cells,
            visible_border_cells=base_info.visible_border_cells,
            active_mask=state.active_mask.astype(jnp.bool_),
            newly_active_mask=newly_active_mask.astype(jnp.bool_),
            pickup_events_per_ant=pickup_events_per_ant.astype(jnp.float32),
            delivery_events_per_ant=delivery_events_per_ant.astype(jnp.float32),
            write_attempts_per_ant=write_attempts_per_ant.astype(jnp.float32),
        )

    def _active_ants_count_grid(self, ants_pos: jax.Array, active_mask: jax.Array) -> jax.Array:
        ants_pos = ants_pos.astype(jnp.int32)
        return jnp.zeros((self.height, self.width), dtype=jnp.int32).at[
            ants_pos[:, 1],
            ants_pos[:, 0],
        ].add(active_mask.astype(jnp.int32))

    def _mark_visited_active(
        self,
        ants_pos: jax.Array,
        active_mask: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        return jnp.logical_and(
            self._active_ants_count_grid(ants_pos, active_mask) > 0,
            jnp.logical_not(obstacles),
        )

    def _mark_viewed_active(
        self,
        ants_pos: jax.Array,
        ants_facing: jax.Array,
        active_mask: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        offsets = build_forward_vision_offsets(
            ants_facing.astype(jnp.int32),
            depth=int(self.actor_vision_radius),
        )
        positions = ants_pos.astype(jnp.int32)[:, None, :] + offsets
        x_pos = positions[..., 0]
        y_pos = positions[..., 1]
        valid = (
            (0 <= x_pos)
            & (x_pos < self.width)
            & (0 <= y_pos)
            & (y_pos < self.height)
            & active_mask.astype(jnp.bool_)[:, None]
        )
        clipped_x = jnp.clip(x_pos, 0, self.width - 1)
        clipped_y = jnp.clip(y_pos, 0, self.height - 1)
        viewed = jnp.zeros((self.height, self.width), dtype=jnp.int32).at[
            clipped_y.reshape(-1),
            clipped_x.reshape(-1),
        ].add(valid.reshape(-1).astype(jnp.int32))
        return jnp.logical_and(viewed > 0, jnp.logical_not(obstacles))

    def _count_visible_border_cells_active(
        self,
        ants_pos: jax.Array,
        ants_facing: jax.Array,
        active_mask: jax.Array,
        obstacles: jax.Array,
    ) -> jax.Array:
        visible_cells = self._mark_viewed_active(ants_pos, ants_facing, active_mask, obstacles)
        x_coords = jnp.arange(self.width, dtype=jnp.int32)[None, :]
        y_coords = jnp.arange(self.height, dtype=jnp.int32)[:, None]
        border_mask = (
            (x_coords == 0)
            | (x_coords == self.width - 1)
            | (y_coords == 0)
            | (y_coords == self.height - 1)
        )
        return jnp.sum(jnp.logical_and(visible_cells, border_mask)).astype(jnp.int32)

    def _write_attempts_per_ant(
        self,
        state: JaxAntState,
        actions: jax.Array,
    ) -> jax.Array:
        moves = actions[:, 0]
        write_values = actions[:, 1]
        next_pos = jax.vmap(
            lambda position, move: self.base_env._move_position(
                position,
                action=move,
                obstacles=state.obstacles,
            )
        )(state.ants_pos, moves)
        x_pos = next_pos[:, 0]
        y_pos = next_pos[:, 1]
        tile_had_food = state.food[y_pos, x_pos] > 0
        tile_is_hub = jnp.all(next_pos == state.hub_pos[None, :], axis=-1)
        wants_write = jnp.logical_or(
            moves == ACTION_STAY,
            jnp.asarray(self.write_while_moving, dtype=jnp.bool_),
        )
        can_write = wants_write & jnp.logical_not(tile_had_food | tile_is_hub)
        if bool(getattr(self, "per_ant_write_channels", False)):
            bit_indices = jnp.mod(
                jnp.arange(self.num_ants, dtype=jnp.uint32),
                jnp.asarray(int(self.write_bits), dtype=jnp.uint32),
            )
            write_values = write_values.astype(jnp.uint32) & jnp.left_shift(
                jnp.asarray(1, dtype=jnp.uint32),
                bit_indices,
            )
        return can_write.astype(jnp.float32) * (write_values >= 0).astype(jnp.float32)


def make_timed_release_env(args: Any) -> TimedReleaseJaxEnv:
    base_env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        random_ant_spawn=False,
        random_ant_spawn_radius=None,
        layout_margin=int(getattr(args, "layout_margin", 0)),
        hub_center_window_size=int(getattr(args, "hub_center_window_size", 0)),
        actor_vision_radius=int(getattr(args, "actor_vision_radius", 1)),
        step_penalty=args.step_penalty,
        completion_bonus=getattr(args, "completion_bonus", 0.0),
        write_penalty=args.write_penalty,
        write_bits=args.write_bits,
        write_while_moving=bool(getattr(args, "write_while_moving", False)),
        per_ant_write_channels=bool(getattr(args, "per_ant_write_channels", False)),
        terminate_on_food_delivery=bool(getattr(args, "food_termination", True)),
        terminate_on_full_coverage=bool(getattr(args, "terminate_on_full_coverage", False)),
        maze_obstacles=bool(getattr(args, "maze_obstacles", False)),
        maze_corridor_width=int(getattr(args, "maze_corridor_width", 3)),
        maze_wall_width=int(getattr(args, "maze_wall_width", 1)),
        maze_seed=int(getattr(args, "maze_seed", 0)),
        maze_layout_count=int(getattr(args, "maze_layout_count", 64)),
        random_wall_obstacles=bool(getattr(args, "random_wall_obstacles", False)),
        random_wall_count_min=int(getattr(args, "random_wall_count_min", 1)),
        random_wall_count_max=int(getattr(args, "random_wall_count_max", 3)),
        random_wall_length_min=int(getattr(args, "random_wall_length_min", 4)),
        random_wall_length_max=int(getattr(args, "random_wall_length_max", 14)),
        random_wall_width=int(getattr(args, "random_wall_width", 1)),
        random_wall_l_turn_probability=float(
            getattr(args, "random_wall_l_turn_probability", 0.5)
        ),
        random_wall_center_window_size=int(
            getattr(args, "random_wall_center_window_size", 0)
        ),
    )
    return TimedReleaseJaxEnv(
        base_env,
        release_interval=int(getattr(args, "release_interval", 150)),
        initial_active_ants=int(getattr(args, "initial_active_ants", 1)),
    )
