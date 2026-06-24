"""Gymnasium environment for ant byte foraging."""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from ant_byte_env.maze import (
    generate_wide_corridor_maze,
    nearest_open_flat_lookup,
    open_flat_indices,
)
from ant_byte_env.sprites import load_sprites


ObsType = dict[str, np.ndarray]

FOOD_COUNT_COLOR = (255, 250, 235)
CARRIED_FOOD_COLOR = (188, 112, 45)
CARRIED_FOOD_HIGHLIGHT = (239, 167, 82)
OBSTACLE_COLOR = (54, 61, 65)
OBSTACLE_EDGE_COLOR = (34, 39, 43)
DEFAULT_WRITE_BITS = 1
MAX_WRITE_BITS = 8
WRITE_VALUE_COUNT = 2
MAX_WRITE_VALUE = WRITE_VALUE_COUNT - 1
DEFAULT_ACTOR_VISION_WIDTH = 3
DEFAULT_ACTOR_VISION_DEPTH = 1

ACTION_STAY = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_LEFT = 4
MOVEMENT_ACTION_COUNT = 5

MOVE_STAY = ACTION_STAY
MOVE_UP = ACTION_UP
MOVE_RIGHT = ACTION_RIGHT
MOVE_DOWN = ACTION_DOWN
MOVE_LEFT = ACTION_LEFT
DEFAULT_FACING = MOVE_RIGHT

FACING_ROTATIONS = {
    MOVE_UP: 90,
    MOVE_RIGHT: 0,
    MOVE_DOWN: -90,
    MOVE_LEFT: 180,
}


def write_value_count(write_bits: int) -> int:
    """Return the number of discrete values representable by ``write_bits``."""

    return 1 << int(write_bits)


def max_write_value(write_bits: int) -> int:
    """Return the largest integer value representable by ``write_bits``."""

    return write_value_count(write_bits) - 1


def per_ant_write_channel_value(
    *,
    current_value: int,
    requested_value: int,
    ant_index: int,
    write_bits: int,
) -> int:
    """Return a byte update where an ant can only change its assigned bit."""

    ant_bit = 1 << (int(ant_index) % int(write_bits))
    preserved_value = int(current_value) & ~ant_bit
    requested_bit = int(requested_value) & ant_bit
    return preserved_value | requested_bit


def actor_vision_patch_size(depth: int) -> int:
    """Return tiles in the centered square actor window."""

    if depth < 0:
        raise ValueError("actor_vision_radius must be non-negative.")
    width = 2 * int(depth) + 1
    return width * width


def food_alpha(remaining: int, initial: int) -> int:
    """Return sprite alpha for a food source based on remaining bites."""

    if remaining <= 0 or initial <= 0:
        return 0
    ratio = min(1.0, max(0.0, remaining / initial))
    return int(72 + 183 * ratio)


def facing_rotation_degrees(facing: int) -> int:
    """Return Pygame rotation degrees for a sprite whose source art faces right."""

    if facing not in FACING_ROTATIONS:
        raise ValueError(f"Unsupported ant facing direction: {facing}.")
    return FACING_ROTATIONS[facing]


def movement_facing(facing: int, action: int) -> int:
    """Return the display facing after a cardinal movement action."""

    if action in FACING_ROTATIONS:
        return int(action)
    if facing not in FACING_ROTATIONS:
        return DEFAULT_FACING
    return facing


def facing_delta(facing: int) -> tuple[int, int]:
    """Return the forward grid delta for a facing direction."""

    if facing == MOVE_RIGHT:
        return 1, 0
    if facing == MOVE_LEFT:
        return -1, 0
    if facing == MOVE_DOWN:
        return 0, 1
    if facing == MOVE_UP:
        return 0, -1
    return facing_delta(DEFAULT_FACING)


def rotate_ant_sprite(surface: pygame.Surface, facing: int) -> pygame.Surface:
    """Rotate an ant sprite so it points in the given movement direction."""

    rotated = pygame.transform.rotate(surface, facing_rotation_degrees(facing))
    if rotated.get_size() == surface.get_size():
        return rotated
    return pygame.transform.smoothscale(rotated, surface.get_size())


class AntByteForagingEnv(gym.Env[ObsType, np.ndarray]):
    """Centralized-control gridworld where ants forage and write tile bytes."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        num_ants: int = 4,
        food_count: int = 8,
        food_source_count: int = 1,
        max_steps: int = 500,
        render_mode: str | None = None,
        tile_size: int = 32,
        random_food: bool = True,
        random_hub: bool = False,
        random_ant_spawn: bool = False,
        random_ant_spawn_radius: int | None = None,
        layout_margin: int = 0,
        hub_center_window_size: int = 0,
        step_penalty: float = 0.0,
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
        seed: int | None = None,
    ) -> None:
        self._validate_constructor_args(
            width=width,
            height=height,
            num_ants=num_ants,
            food_count=food_count,
            food_source_count=food_source_count,
            max_steps=max_steps,
            render_mode=render_mode,
            tile_size=tile_size,
            step_penalty=step_penalty,
            write_penalty=write_penalty,
            write_bits=write_bits,
            per_ant_write_channels=per_ant_write_channels,
            random_ant_spawn_radius=random_ant_spawn_radius,
            layout_margin=layout_margin,
            hub_center_window_size=hub_center_window_size,
            maze_corridor_width=maze_corridor_width,
            maze_wall_width=maze_wall_width,
            maze_layout_count=maze_layout_count,
        )

        self.width = width
        self.height = height
        self.num_ants = num_ants
        self.food_count = food_count
        self.food_source_count = food_source_count
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.tile_size = tile_size
        self.random_food = random_food
        self.random_hub = bool(random_hub)
        self.random_ant_spawn = bool(random_ant_spawn)
        self.random_ant_spawn_radius = (
            None if random_ant_spawn_radius is None else int(random_ant_spawn_radius)
        )
        self.layout_margin = int(layout_margin)
        self.hub_center_window_size = int(hub_center_window_size)
        self.step_penalty = step_penalty
        self.write_penalty = write_penalty
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
        self._obstacle_bank = self._build_obstacle_bank()
        self._set_obstacle_grid(self._obstacle_bank[0])
        self.write_value_count = write_value_count(self.write_bits)
        self.max_write_value = max_write_value(self.write_bits)
        self._constructor_seed = seed
        self._has_reset = False
        self._last_episode_done = False

        max_coord = max(width, height)
        self.action_space = spaces.MultiDiscrete(
            [MOVEMENT_ACTION_COUNT, self.write_value_count] * num_ants
        )
        self.observation_space = spaces.Dict(
            {
                "ants_pos": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(num_ants, 2),
                    dtype=np.int32,
                ),
                "ants_carrying": spaces.MultiBinary(num_ants),
                "ants_facing": spaces.Box(
                    low=MOVE_UP,
                    high=MOVE_LEFT,
                    shape=(num_ants,),
                    dtype=np.int8,
                ),
                "ants_count": spaces.Box(
                    low=0,
                    high=num_ants,
                    shape=(height, width),
                    dtype=np.int32,
                ),
                "food": spaces.Box(
                    low=0,
                    high=max(food_count, 1),
                    shape=(height, width),
                    dtype=np.int32,
                ),
                "bytes": spaces.Box(
                    low=0,
                    high=self.max_write_value,
                    shape=(height, width),
                    dtype=np.uint8,
                ),
                "obstacles": spaces.MultiBinary((height, width)),
                "hub_pos": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(2,),
                    dtype=np.int32,
                ),
            }
        )

        self.hub_pos = np.zeros(2, dtype=np.int32)
        self.ants_pos = np.zeros((num_ants, 2), dtype=np.int32)
        self.ants_count = np.zeros((height, width), dtype=np.int32)
        self.ants_facing = np.full(num_ants, DEFAULT_FACING, dtype=np.int8)
        self.ants_carrying = np.zeros(num_ants, dtype=bool)
        self.food = np.zeros((height, width), dtype=np.int32)
        self.initial_food = np.zeros((height, width), dtype=np.int32)
        self.bytes = np.zeros((height, width), dtype=np.uint8)
        self.delivered_food = 0
        self.step_count = 0
        self._initial_food_total = food_count
        self.visited_cells = np.zeros((height, width), dtype=bool)

        self._window: pygame.Surface | None = None
        self._canvas: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._sprites: dict[str, pygame.Surface] | None = None

        if render_mode is not None:
            self._init_rendering()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, int]]:
        should_avoid_previous_layout = self._has_reset and self._last_episode_done
        previous_hub_pos = self.hub_pos.copy() if should_avoid_previous_layout else None
        previous_initial_food = (
            self.initial_food.copy() if should_avoid_previous_layout else None
        )
        actual_seed = seed
        if actual_seed is None and not self._has_reset:
            actual_seed = self._constructor_seed
        super().reset(seed=actual_seed)
        self._has_reset = True

        reset_options = options or {}
        self._reset_obstacle_grid(reset_options)
        self.hub_pos = self._resolve_hub_pos(
            reset_options,
            previous_hub_pos=previous_hub_pos,
        )
        self.ants_facing = np.full(self.num_ants, DEFAULT_FACING, dtype=np.int8)
        self.ants_carrying = np.zeros(self.num_ants, dtype=bool)
        self.bytes = np.zeros((self.height, self.width), dtype=np.uint8)
        self.food = self._build_food_grid(
            reset_options,
            previous_initial_food=previous_initial_food,
        )
        self.ants_pos = self._initial_ant_positions()
        self.ants_count = self._build_ants_count_grid(self.ants_pos)
        self.visited_cells = self._mark_visited(
            np.zeros((self.height, self.width), dtype=bool),
            self.ants_pos,
        )
        self.initial_food = self.food.astype(np.int32, copy=True)
        self.delivered_food = 0
        self.step_count = 0
        self._initial_food_total = int(self.food.sum())
        self._last_episode_done = False

        return self._get_obs(), self._get_info(num_writes=0, num_overwrites=0)

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, int]]:
        flat_action = self._validate_action(action)

        reward = 0.0
        reward -= self.step_penalty * self.num_ants
        num_writes = 0
        num_overwrites = 0
        written_tiles: set[tuple[int, int]] = set()

        for ant_index in range(self.num_ants):
            action_start = 2 * ant_index
            move_action = int(flat_action[action_start])
            write_value = int(flat_action[action_start + 1])
            next_facing = movement_facing(int(self.ants_facing[ant_index]), move_action)
            next_pos = self._move_position(
                self.ants_pos[ant_index],
                action=move_action,
            )
            self.ants_facing[ant_index] = next_facing
            self.ants_pos[ant_index] = next_pos

            x_pos, y_pos = int(next_pos[0]), int(next_pos[1])
            tile_had_food = self.food[y_pos, x_pos] > 0
            tile_is_hub = self._is_hub_position(x_pos=x_pos, y_pos=y_pos)
            if not self.ants_carrying[ant_index] and tile_had_food:
                self.food[y_pos, x_pos] -= 1
                self.ants_carrying[ant_index] = True

            if self.ants_carrying[ant_index] and tile_is_hub:
                self.ants_carrying[ant_index] = False
                self.delivered_food += 1
                reward += 1.0

            wants_write = self.write_while_moving or move_action == ACTION_STAY
            if not wants_write or tile_had_food or tile_is_hub:
                continue

            tile_key = (x_pos, y_pos)
            if tile_key in written_tiles:
                num_overwrites += 1
            written_tiles.add(tile_key)
            current_value = int(self.bytes[y_pos, x_pos])
            if self.per_ant_write_channels:
                write_value = per_ant_write_channel_value(
                    current_value=current_value,
                    requested_value=write_value,
                    ant_index=ant_index,
                    write_bits=self.write_bits,
                )
            self.bytes[y_pos, x_pos] = np.uint8(write_value)
            num_writes += 1

        reward -= self.write_penalty * num_writes
        self.step_count += 1
        self.ants_count = self._build_ants_count_grid(self.ants_pos)
        step_visited_cells = self._mark_visited(
            np.zeros((self.height, self.width), dtype=bool),
            self.ants_pos,
        )
        newly_visited_cells = int(np.logical_and(step_visited_cells, ~self.visited_cells).sum())
        self.visited_cells = np.logical_or(self.visited_cells, step_visited_cells)
        completed_food = self.delivered_food >= self._initial_food_total
        completed_coverage = int(self.visited_cells.sum()) >= self.open_cell_count
        terminated = (
            (self.terminate_on_food_delivery and completed_food)
            or (self.terminate_on_full_coverage and completed_coverage)
        )
        truncated = self.step_count >= self.max_steps
        self._last_episode_done = bool(terminated or truncated)

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._get_info(
                num_writes=num_writes,
                num_overwrites=num_overwrites,
                newly_visited_cells=newly_visited_cells,
            ),
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        self._init_rendering()
        frame = self._draw_frame()

        if self.render_mode == "human":
            assert self._window is not None
            assert self._clock is not None
            pygame.event.pump()
            self._window.blit(frame, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(pygame.surfarray.array3d(frame), axes=(1, 0, 2)).copy()

    def close(self) -> None:
        self._window = None
        self._canvas = None
        self._clock = None
        self._font = None
        self._sprites = None
        if pygame.display.get_init():
            pygame.display.quit()

    @staticmethod
    def _validate_constructor_args(
        *,
        width: int,
        height: int,
        num_ants: int,
        food_count: int,
        food_source_count: int,
        max_steps: int,
        render_mode: str | None,
        tile_size: int,
        step_penalty: float,
        write_penalty: float,
        write_bits: int,
        per_ant_write_channels: bool,
        random_ant_spawn_radius: int | None,
        layout_margin: int,
        hub_center_window_size: int,
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
        if food_source_count <= 0:
            raise ValueError("food_source_count must be positive.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if step_penalty < 0:
            raise ValueError("step_penalty must be non-negative.")
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
        if render_mode is not None and render_mode not in AntByteForagingEnv.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")

    def _resolve_hub_pos(
        self,
        options: dict[str, Any],
        *,
        previous_hub_pos: np.ndarray | None,
    ) -> np.ndarray:
        if "hub_pos" in options:
            return self._nearest_open_position(self._coerce_position(options["hub_pos"]))
        if self.random_hub:
            candidates = self.open_flat_indices
            centered = np.array(
                [
                    flat
                    for flat in candidates
                    if self._inside_hub_center_window(
                        x_pos=int(flat) % self.width,
                        y_pos=int(flat) // self.width,
                    )
                ],
                dtype=np.int32,
            )
            if centered.size > 0:
                candidates = centered
            preferred = np.array(
                [
                    flat
                    for flat in candidates
                    if self._inside_layout_margin(
                        x_pos=int(flat) % self.width,
                        y_pos=int(flat) // self.width,
                    )
                ],
                dtype=np.int32,
            )
            if preferred.size > 0:
                candidates = preferred
            if previous_hub_pos is not None and candidates.size > 1:
                previous_flat = int(previous_hub_pos[1]) * self.width + int(previous_hub_pos[0])
                preferred = candidates[candidates != previous_flat]
                if preferred.size > 0:
                    candidates = preferred
            flat = int(self.np_random.choice(candidates))
            return self._flat_to_position(flat)
        return self._nearest_open_position(
            np.array([self.width // 2, self.height // 2], dtype=np.int32)
        )

    def _build_food_grid(
        self,
        options: dict[str, Any],
        *,
        previous_initial_food: np.ndarray | None,
    ) -> np.ndarray:
        food = np.zeros((self.height, self.width), dtype=np.int32)
        if self.food_count == 0:
            return food

        raw_positions = options.get("food_positions")
        if raw_positions is not None:
            positions = [
                self._food_position_from_raw(position)
                for position in (
                    self._coerce_position(raw_pos) for raw_pos in raw_positions
                )
            ]
            return self._distribute_food_units(food, positions)

        candidates = self._candidate_food_positions()
        if not candidates:
            raise ValueError("food_count requires at least one non-hub tile.")

        source_count = min(self.food_source_count, len(candidates))
        if self.random_food:
            if previous_initial_food is not None:
                preferred = self._exclude_previous_food_sources(
                    candidates,
                    previous_initial_food=previous_initial_food,
                )
                if len(preferred) >= source_count:
                    candidates = preferred
            chosen_indices = self.np_random.choice(
                len(candidates),
                size=source_count,
                replace=False,
            )
            positions = [candidates[int(index)] for index in np.atleast_1d(chosen_indices)]
            return self._distribute_food_units(food, positions)

        return self._distribute_food_units(food, candidates[:source_count])

    def _exclude_previous_food_sources(
        self,
        candidates: list[tuple[int, int]],
        *,
        previous_initial_food: np.ndarray,
    ) -> list[tuple[int, int]]:
        previous_grid = np.asarray(previous_initial_food)
        if previous_grid.shape != (self.height, self.width):
            return candidates
        previous_sources = {
            (int(x_pos), int(y_pos))
            for y_pos, x_pos in np.argwhere(previous_grid > 0)
        }
        if not previous_sources:
            return candidates
        return [candidate for candidate in candidates if candidate not in previous_sources]

    def _distribute_food_units(
        self,
        food: np.ndarray,
        positions: list[tuple[int, int]],
    ) -> np.ndarray:
        if not positions:
            raise ValueError("food_positions must contain at least one position.")

        base_amount, extra_units = divmod(self.food_count, len(positions))
        for index, (x_pos, y_pos) in enumerate(positions):
            amount = base_amount + int(index < extra_units)
            food[y_pos, x_pos] += amount
        return food

    def _candidate_food_positions(self) -> list[tuple[int, int]]:
        hub_key = (int(self.hub_pos[0]), int(self.hub_pos[1]))
        candidates = [
            (x_pos, y_pos)
            for y_pos in range(self.height)
            for x_pos in range(self.width)
            if (x_pos, y_pos) != hub_key
            and not self.obstacles[y_pos, x_pos]
        ]
        preferred = [
            (x_pos, y_pos)
            for x_pos, y_pos in candidates
            if self._inside_layout_margin(x_pos=x_pos, y_pos=y_pos)
        ]
        if len(preferred) >= min(self.food_source_count, len(candidates)):
            return preferred
        return candidates

    def _initial_ant_positions(self) -> np.ndarray:
        if not self.random_ant_spawn:
            return np.repeat(self.hub_pos.reshape(1, 2), self.num_ants, axis=0)

        hub_key = (int(self.hub_pos[0]), int(self.hub_pos[1]))
        preferred = [
            (x_pos, y_pos)
            for y_pos in range(self.height)
            for x_pos in range(self.width)
            if (x_pos, y_pos) != hub_key and int(self.food[y_pos, x_pos]) <= 0
            and not self.obstacles[y_pos, x_pos]
            and self._within_ant_spawn_radius(x_pos=x_pos, y_pos=y_pos)
            and self._inside_layout_margin(x_pos=x_pos, y_pos=y_pos)
        ]
        fallback = [
            (x_pos, y_pos)
            for y_pos in range(self.height)
            for x_pos in range(self.width)
            if (x_pos, y_pos) != hub_key
            and not self.obstacles[y_pos, x_pos]
            and self._within_ant_spawn_radius(x_pos=x_pos, y_pos=y_pos)
            and self._inside_layout_margin(x_pos=x_pos, y_pos=y_pos)
        ]
        no_margin_fallback = [
            (x_pos, y_pos)
            for y_pos in range(self.height)
            for x_pos in range(self.width)
            if (x_pos, y_pos) != hub_key
            and not self.obstacles[y_pos, x_pos]
            and self._within_ant_spawn_radius(x_pos=x_pos, y_pos=y_pos)
        ]
        candidates = preferred or fallback or no_margin_fallback or [hub_key]
        chosen_indices = self.np_random.choice(
            len(candidates),
            size=self.num_ants,
            replace=True,
        )
        return np.asarray(
            [candidates[int(index)] for index in np.atleast_1d(chosen_indices)],
            dtype=np.int32,
        )

    def _within_ant_spawn_radius(self, *, x_pos: int, y_pos: int) -> bool:
        if self.random_ant_spawn_radius is None:
            return True
        return (
            max(
                abs(int(x_pos) - int(self.hub_pos[0])),
                abs(int(y_pos) - int(self.hub_pos[1])),
            )
            <= self.random_ant_spawn_radius
        )

    def _inside_layout_margin(self, *, x_pos: int, y_pos: int) -> bool:
        margin = int(self.layout_margin)
        return (
            int(x_pos) >= margin
            and int(x_pos) < self.width - margin
            and int(y_pos) >= margin
            and int(y_pos) < self.height - margin
        )

    def _inside_hub_center_window(self, *, x_pos: int, y_pos: int) -> bool:
        window_size = int(self.hub_center_window_size)
        if window_size <= 0:
            return True
        x_start = (self.width - window_size) // 2
        y_start = (self.height - window_size) // 2
        return (
            int(x_pos) >= x_start
            and int(x_pos) < x_start + window_size
            and int(y_pos) >= y_start
            and int(y_pos) < y_start + window_size
        )

    def _coerce_position(self, raw_pos: Any) -> np.ndarray:
        position = np.asarray(raw_pos, dtype=np.int32)
        if position.shape != (2,):
            raise ValueError("positions must be two-item (x, y) pairs.")
        x_pos, y_pos = int(position[0]), int(position[1])
        if not (0 <= x_pos < self.width and 0 <= y_pos < self.height):
            raise ValueError(f"position {(x_pos, y_pos)!r} is outside the grid.")
        return np.array([x_pos, y_pos], dtype=np.int32)

    def _food_position_from_raw(self, raw_pos: np.ndarray) -> tuple[int, int]:
        position = self._nearest_open_position(raw_pos)
        if np.array_equal(position, self.hub_pos):
            candidates = self._candidate_food_positions()
            if candidates:
                return candidates[0]
        return int(position[0]), int(position[1])

    def _is_hub_position(self, *, x_pos: int, y_pos: int) -> bool:
        return x_pos == int(self.hub_pos[0]) and y_pos == int(self.hub_pos[1])

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        flat_action = np.asarray(action, dtype=np.int64).reshape(-1)
        expected_shape = (2 * self.num_ants,)
        if flat_action.shape != expected_shape:
            raise ValueError(
                f"action must have shape ({expected_shape[0]},), got {flat_action.shape}."
            )
        moves = flat_action[0::2]
        writes = flat_action[1::2]
        if np.any(moves < ACTION_STAY) or np.any(moves >= MOVEMENT_ACTION_COUNT):
            raise ValueError(
                f"movement actions must be integers from 0 to {MOVEMENT_ACTION_COUNT - 1}."
            )
        if np.any(writes < 0) or np.any(writes > self.max_write_value):
            raise ValueError(
                f"write actions must be integers from 0 to {self.max_write_value}."
            )
        return flat_action

    def _move_position(self, position: np.ndarray, *, action: int) -> np.ndarray:
        original = np.asarray(position, dtype=np.int32)
        x_pos, y_pos = int(position[0]), int(position[1])
        if action in FACING_ROTATIONS:
            dx, dy = facing_delta(action)
            x_pos += dx
            y_pos += dy
        next_pos = np.array(
            [
                int(np.clip(x_pos, 0, self.width - 1)),
                int(np.clip(y_pos, 0, self.height - 1)),
            ],
            dtype=np.int32,
        )
        if self.obstacles[int(next_pos[1]), int(next_pos[0])]:
            return original.copy()
        return next_pos

    def _build_ants_count_grid(self, ants_pos: np.ndarray) -> np.ndarray:
        counts = np.zeros((self.height, self.width), dtype=np.int32)
        for x_pos, y_pos in ants_pos:
            counts[int(y_pos), int(x_pos)] += 1
        return counts

    def _mark_visited(self, visited_cells: np.ndarray, ants_pos: np.ndarray) -> np.ndarray:
        for x_pos, y_pos in ants_pos:
            if not self.obstacles[int(y_pos), int(x_pos)]:
                visited_cells[int(y_pos), int(x_pos)] = True
        return visited_cells

    def _get_obs(self) -> ObsType:
        return {
            "ants_pos": self.ants_pos.astype(np.int32, copy=True),
            "ants_carrying": self.ants_carrying.astype(np.int8, copy=True),
            "ants_facing": self.ants_facing.astype(np.int8, copy=True),
            "ants_count": self.ants_count.astype(np.int32, copy=True),
            "food": self.food.astype(np.int32, copy=True),
            "bytes": self.bytes.astype(np.uint8, copy=True),
            "obstacles": self.obstacles.astype(np.int8, copy=True),
            "hub_pos": self.hub_pos.astype(np.int32, copy=True),
        }

    def _get_info(
        self,
        *,
        num_writes: int,
        num_overwrites: int,
        newly_visited_cells: int = 0,
    ) -> dict[str, int]:
        return {
            "delivered_food": int(self.delivered_food),
            "remaining_food": int(self.food.sum()),
            "step_count": int(self.step_count),
            "num_writes": int(num_writes),
            "num_overwrites": int(num_overwrites),
            "visited_cell_count": int(self.visited_cells.sum()),
            "newly_visited_cells": int(newly_visited_cells),
        }

    def _init_rendering(self) -> None:
        if self._canvas is not None and self._sprites is not None and self._font is not None:
            return

        pygame.init()
        pygame.font.init()
        size = (self.width * self.tile_size, self.height * self.tile_size)
        self._canvas = pygame.Surface(size)
        self._sprites = load_sprites(self.tile_size)
        self._font = pygame.font.Font(None, max(10, self.tile_size // 2))
        if self.render_mode == "human":
            self._window = pygame.display.set_mode(size)
            pygame.display.set_caption("AntByteForaging-v0")
            self._clock = pygame.time.Clock()

    def _draw_frame(self) -> pygame.Surface:
        assert self._canvas is not None
        assert self._sprites is not None
        assert self._font is not None

        self._canvas.fill((215, 207, 181))
        for y_pos in range(self.height):
            for x_pos in range(self.width):
                rect = pygame.Rect(
                    x_pos * self.tile_size,
                    y_pos * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                self._canvas.blit(self._sprites["tile"], rect)
                if self.obstacles[y_pos, x_pos]:
                    self._draw_obstacle_tile(rect)
                    continue
                self._draw_byte_overlay(rect, int(self.bytes[y_pos, x_pos]))

        self._draw_grid_items()
        return self._canvas

    def _draw_byte_overlay(self, rect: pygame.Rect, byte_value: int) -> None:
        assert self._canvas is not None
        assert self._font is not None
        if byte_value == 0:
            return

        ratio = byte_value / max(float(self.max_write_value), 1.0)
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill((int(40 + 180 * ratio), 92, int(255 - 120 * ratio), 96))
        self._canvas.blit(overlay, rect.topleft)
        label = self._font.render(str(byte_value), True, (24, 31, 36))
        self._canvas.blit(label, (rect.x + 2, rect.y + 1))

    def _draw_obstacle_tile(self, rect: pygame.Rect) -> None:
        assert self._canvas is not None
        pygame.draw.rect(self._canvas, OBSTACLE_COLOR, rect)
        pygame.draw.rect(self._canvas, OBSTACLE_EDGE_COLOR, rect, max(1, self.tile_size // 14))

    def _draw_grid_items(self) -> None:
        assert self._canvas is not None
        assert self._sprites is not None
        assert self._font is not None

        self._blit_tile_sprite("hub", self.hub_pos)
        for y_pos in range(self.height):
            for x_pos in range(self.width):
                food_amount = int(self.food[y_pos, x_pos])
                if food_amount <= 0:
                    continue
                position = np.array([x_pos, y_pos], dtype=np.int32)
                initial_amount = int(self.initial_food[y_pos, x_pos])
                food_sprite = self._sprites["food"].copy()
                food_sprite.set_alpha(food_alpha(food_amount, initial_amount))
                self._blit_tile_surface(food_sprite, position)
                if food_amount > 1:
                    label = self._font.render(str(food_amount), True, FOOD_COUNT_COLOR)
                    self._canvas.blit(
                        label,
                        (
                            x_pos * self.tile_size + self.tile_size // 2,
                            y_pos * self.tile_size + self.tile_size // 2,
                        ),
                    )

        for ant_index, position in enumerate(self.ants_pos):
            ant_sprite = rotate_ant_sprite(
                self._sprites["ant"],
                int(self.ants_facing[ant_index]),
            )
            self._blit_tile_surface(ant_sprite, position)
            if self.ants_carrying[ant_index]:
                self._draw_carried_food_marker(position)

    def _blit_tile_sprite(self, sprite_name: str, position: np.ndarray) -> None:
        assert self._canvas is not None
        assert self._sprites is not None
        self._blit_tile_surface(self._sprites[sprite_name], position)

    def _blit_tile_surface(self, surface: pygame.Surface, position: np.ndarray) -> None:
        assert self._canvas is not None
        x_pos, y_pos = int(position[0]), int(position[1])
        self._canvas.blit(
            surface,
            (x_pos * self.tile_size, y_pos * self.tile_size),
        )

    def _draw_carried_food_marker(self, position: np.ndarray) -> None:
        assert self._canvas is not None
        x_pos, y_pos = int(position[0]), int(position[1])
        center = (
            x_pos * self.tile_size + 3 * self.tile_size // 4,
            y_pos * self.tile_size + self.tile_size // 4,
        )
        pygame.draw.circle(
            self._canvas,
            CARRIED_FOOD_COLOR,
            center,
            max(3, self.tile_size // 7),
        )
        pygame.draw.circle(
            self._canvas,
            CARRIED_FOOD_HIGHLIGHT,
            center,
            max(1, self.tile_size // 12),
        )

    def _reset_obstacle_grid(self, options: dict[str, Any]) -> None:
        if "obstacles" in options:
            self._set_obstacle_grid(np.asarray(options["obstacles"], dtype=bool))
            return
        if self._obstacle_bank.shape[0] <= 1:
            self._set_obstacle_grid(self._obstacle_bank[0])
            return

        previous = self.obstacles
        same_as_previous = np.array(
            [np.array_equal(layout, previous) for layout in self._obstacle_bank],
            dtype=bool,
        )
        candidate_indices = np.flatnonzero(~same_as_previous)
        if candidate_indices.size == 0:
            candidate_indices = np.arange(self._obstacle_bank.shape[0])
        selected = int(self.np_random.choice(candidate_indices))
        self._set_obstacle_grid(self._obstacle_bank[selected])

    def _set_obstacle_grid(self, obstacles: np.ndarray) -> None:
        obstacle_grid = np.asarray(obstacles, dtype=bool)
        if obstacle_grid.shape != (self.height, self.width):
            raise ValueError(
                "obstacles must have shape "
                f"{(self.height, self.width)}, got {obstacle_grid.shape}."
            )
        if not np.any(~obstacle_grid):
            raise ValueError("maze obstacle layout must contain at least one open cell.")
        self.obstacles = obstacle_grid.copy()
        self.open_flat_indices = open_flat_indices(self.obstacles)
        self.nearest_open_flat = nearest_open_flat_lookup(self.obstacles)
        self.open_cell_count = int(self.open_flat_indices.size)

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

    def _nearest_open_position(self, position: np.ndarray) -> np.ndarray:
        flat = int(position[1]) * self.width + int(position[0])
        return self._flat_to_position(int(self.nearest_open_flat[flat]))

    def _flat_to_position(self, flat: int) -> np.ndarray:
        return np.array([int(flat) % self.width, int(flat) // self.width], dtype=np.int32)
