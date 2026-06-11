"""Gymnasium environment for ant byte foraging."""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from ant_byte_env.sprites import load_sprites


ObsType = dict[str, np.ndarray]

FOOD_COUNT_COLOR = (255, 250, 235)
CARRIED_FOOD_COLOR = (188, 112, 45)
CARRIED_FOOD_HIGHLIGHT = (239, 167, 82)

MOVE_STAY = 0
MOVE_UP = 1
MOVE_RIGHT = 2
MOVE_DOWN = 3
MOVE_LEFT = 4
DEFAULT_FACING = MOVE_RIGHT

FACING_ROTATIONS = {
    MOVE_UP: 90,
    MOVE_RIGHT: 0,
    MOVE_DOWN: -90,
    MOVE_LEFT: 180,
}


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
        step_penalty: float = 0.0,
        write_penalty: float = 0.0,
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
        self.step_penalty = step_penalty
        self.write_penalty = write_penalty
        self._constructor_seed = seed
        self._has_reset = False

        max_coord = max(width, height)
        self.action_space = spaces.MultiDiscrete([5, 256] * num_ants)
        self.observation_space = spaces.Dict(
            {
                "ants_pos": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(num_ants, 2),
                    dtype=np.int32,
                ),
                "ants_carrying": spaces.MultiBinary(num_ants),
                "food": spaces.Box(
                    low=0,
                    high=max(food_count, 1),
                    shape=(height, width),
                    dtype=np.int32,
                ),
                "bytes": spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width),
                    dtype=np.uint8,
                ),
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
        self.ants_facing = np.full(num_ants, DEFAULT_FACING, dtype=np.int8)
        self.ants_carrying = np.zeros(num_ants, dtype=bool)
        self.food = np.zeros((height, width), dtype=np.int32)
        self.initial_food = np.zeros((height, width), dtype=np.int32)
        self.bytes = np.zeros((height, width), dtype=np.uint8)
        self.delivered_food = 0
        self.step_count = 0
        self._initial_food_total = food_count

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
        actual_seed = seed
        if actual_seed is None and not self._has_reset:
            actual_seed = self._constructor_seed
        super().reset(seed=actual_seed)
        self._has_reset = True

        reset_options = options or {}
        self.hub_pos = self._resolve_hub_pos(reset_options)
        self.ants_pos = np.repeat(self.hub_pos.reshape(1, 2), self.num_ants, axis=0)
        self.ants_facing = np.full(self.num_ants, DEFAULT_FACING, dtype=np.int8)
        self.ants_carrying = np.zeros(self.num_ants, dtype=bool)
        self.bytes = np.zeros((self.height, self.width), dtype=np.uint8)
        self.food = self._build_food_grid(reset_options)
        self.initial_food = self.food.astype(np.int32, copy=True)
        self.delivered_food = 0
        self.step_count = 0
        self._initial_food_total = int(self.food.sum())

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
            move = int(flat_action[2 * ant_index])
            write_byte = int(flat_action[2 * ant_index + 1])
            next_pos = self._move_position(self.ants_pos[ant_index], move)
            if move != MOVE_STAY:
                self.ants_facing[ant_index] = move
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

            if tile_had_food or tile_is_hub:
                continue

            tile_key = (x_pos, y_pos)
            if tile_key in written_tiles:
                num_overwrites += 1
            written_tiles.add(tile_key)
            self.bytes[y_pos, x_pos] = np.uint8(write_byte)
            num_writes += 1

        reward -= self.write_penalty * num_writes
        self.step_count += 1
        terminated = self.delivered_food >= self._initial_food_total
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._get_info(num_writes=num_writes, num_overwrites=num_overwrites),
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
        if render_mode is not None and render_mode not in AntByteForagingEnv.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")

    def _resolve_hub_pos(self, options: dict[str, Any]) -> np.ndarray:
        if "hub_pos" in options:
            return self._coerce_position(options["hub_pos"])
        return np.array([self.width // 2, self.height // 2], dtype=np.int32)

    def _build_food_grid(self, options: dict[str, Any]) -> np.ndarray:
        food = np.zeros((self.height, self.width), dtype=np.int32)
        if self.food_count == 0:
            return food

        raw_positions = options.get("food_positions")
        if raw_positions is not None:
            positions = [
                (int(pos[0]), int(pos[1]))
                for pos in (self._coerce_position(raw_pos) for raw_pos in raw_positions)
            ]
            return self._distribute_food_units(food, positions)

        candidates = self._candidate_food_positions()
        if not candidates:
            raise ValueError("food_count requires at least one non-hub tile.")

        source_count = min(self.food_source_count, len(candidates))
        if self.random_food:
            chosen_indices = self.np_random.choice(
                len(candidates),
                size=source_count,
                replace=False,
            )
            positions = [candidates[int(index)] for index in np.atleast_1d(chosen_indices)]
            return self._distribute_food_units(food, positions)

        return self._distribute_food_units(food, candidates[:source_count])

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
        return [
            (x_pos, y_pos)
            for y_pos in range(self.height)
            for x_pos in range(self.width)
            if (x_pos, y_pos) != hub_key
        ]

    def _coerce_position(self, raw_pos: Any) -> np.ndarray:
        position = np.asarray(raw_pos, dtype=np.int32)
        if position.shape != (2,):
            raise ValueError("positions must be two-item (x, y) pairs.")
        x_pos, y_pos = int(position[0]), int(position[1])
        if not (0 <= x_pos < self.width and 0 <= y_pos < self.height):
            raise ValueError(f"position {(x_pos, y_pos)!r} is outside the grid.")
        return np.array([x_pos, y_pos], dtype=np.int32)

    def _is_hub_position(self, *, x_pos: int, y_pos: int) -> bool:
        return x_pos == int(self.hub_pos[0]) and y_pos == int(self.hub_pos[1])

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        flat_action = np.asarray(action, dtype=np.int64).reshape(-1)
        if flat_action.shape != (2 * self.num_ants,):
            raise ValueError(
                f"action must have shape ({2 * self.num_ants},), got {flat_action.shape}."
            )
        moves = flat_action[0::2]
        writes = flat_action[1::2]
        if np.any(moves < MOVE_STAY) or np.any(moves > MOVE_LEFT):
            raise ValueError("movement actions must be integers from 0 to 4.")
        if np.any(writes < 0) or np.any(writes > 255):
            raise ValueError("write actions must be integers from 0 to 255.")
        return flat_action

    def _move_position(self, position: np.ndarray, move: int) -> np.ndarray:
        x_pos, y_pos = int(position[0]), int(position[1])
        if move == MOVE_UP:
            y_pos -= 1
        elif move == MOVE_RIGHT:
            x_pos += 1
        elif move == MOVE_DOWN:
            y_pos += 1
        elif move == MOVE_LEFT:
            x_pos -= 1
        return np.array(
            [
                int(np.clip(x_pos, 0, self.width - 1)),
                int(np.clip(y_pos, 0, self.height - 1)),
            ],
            dtype=np.int32,
        )

    def _get_obs(self) -> ObsType:
        return {
            "ants_pos": self.ants_pos.astype(np.int32, copy=True),
            "ants_carrying": self.ants_carrying.astype(np.int8, copy=True),
            "food": self.food.astype(np.int32, copy=True),
            "bytes": self.bytes.astype(np.uint8, copy=True),
            "hub_pos": self.hub_pos.astype(np.int32, copy=True),
        }

    def _get_info(self, *, num_writes: int, num_overwrites: int) -> dict[str, int]:
        return {
            "delivered_food": int(self.delivered_food),
            "remaining_food": int(self.food.sum()),
            "step_count": int(self.step_count),
            "num_writes": int(num_writes),
            "num_overwrites": int(num_overwrites),
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
                self._draw_byte_overlay(rect, int(self.bytes[y_pos, x_pos]))

        self._draw_grid_items()
        return self._canvas

    def _draw_byte_overlay(self, rect: pygame.Rect, byte_value: int) -> None:
        assert self._canvas is not None
        assert self._font is not None
        if byte_value == 0:
            return

        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill((byte_value, 92, 255 - byte_value, min(96, 28 + byte_value // 3)))
        self._canvas.blit(overlay, rect.topleft)
        label = self._font.render(str(byte_value), True, (24, 31, 36))
        self._canvas.blit(label, (rect.x + 2, rect.y + 1))

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
