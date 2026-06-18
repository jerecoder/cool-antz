"""Gymnasium autocurriculum environment for staged ant byte foraging."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ant_byte_env.env import (
    DEFAULT_ACTOR_VISION_DEPTH,
    DEFAULT_WRITE_BITS,
    MAX_WRITE_BITS,
    MOVEMENT_ACTION_COUNT,
    AntByteForagingEnv,
    ObsType,
    max_write_value,
    write_value_count,
)


class AntByteAutoCurriculumEnv(gym.Env[ObsType, np.ndarray]):
    """Square-grid foraging env that advances stages after enough deliveries.

    The active grid starts at ``start_size`` and grows by one after
    ``cookies_per_stage`` delivered food units. Observations are padded to
    ``max_size`` so the Gymnasium observation space remains stable.
    """

    metadata = AntByteForagingEnv.metadata

    def __init__(
        self,
        *,
        start_size: int = 4,
        max_size: int = 50,
        cookies_per_stage: int = 6,
        max_steps: int = 500,
        num_ants: int = 4,
        food_count: int | None = None,
        food_source_count: int = 2,
        render_mode: str | None = None,
        tile_size: int = 32,
        random_food: bool = True,
        random_hub: bool = False,
        step_penalty: float = 0.0,
        write_penalty: float = 0.0,
        write_bits: int = DEFAULT_WRITE_BITS,
        write_while_moving: bool = False,
        actor_vision_radius: int = DEFAULT_ACTOR_VISION_DEPTH,
        seed: int | None = None,
    ) -> None:
        stage_food_count = (
            int(cookies_per_stage) * int(food_source_count) if food_count is None else food_count
        )
        self._validate_constructor_args(
            start_size=start_size,
            max_size=max_size,
            cookies_per_stage=cookies_per_stage,
            max_steps=max_steps,
            num_ants=num_ants,
            food_count=stage_food_count,
            food_source_count=food_source_count,
            render_mode=render_mode,
            tile_size=tile_size,
            step_penalty=step_penalty,
            write_penalty=write_penalty,
            write_bits=write_bits,
            actor_vision_radius=actor_vision_radius,
        )

        self.start_size = int(start_size)
        self.max_size = int(max_size)
        self.cookies_per_stage = int(cookies_per_stage)
        self.max_steps = int(max_steps)
        self.num_ants = int(num_ants)
        self.food_count = int(stage_food_count)
        self.food_source_count = int(food_source_count)
        self.render_mode = render_mode
        self.tile_size = int(tile_size)
        self.random_food = bool(random_food)
        self.random_hub = bool(random_hub)
        self.step_penalty = float(step_penalty)
        self.write_penalty = float(write_penalty)
        self.write_bits = int(write_bits)
        self.write_while_moving = bool(write_while_moving)
        self.actor_vision_radius = int(actor_vision_radius)
        self.write_value_count = write_value_count(self.write_bits)
        self.max_write_value = max_write_value(self.write_bits)
        self._constructor_seed = seed
        self._has_reset = False

        max_coord = self.max_size
        self.action_space = spaces.MultiDiscrete(
            [MOVEMENT_ACTION_COUNT, self.write_value_count] * self.num_ants
        )
        self.observation_space = spaces.Dict(
            {
                "ants_pos": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(self.num_ants, 2),
                    dtype=np.int32,
                ),
                "ants_carrying": spaces.MultiBinary(self.num_ants),
                "ants_facing": spaces.Box(
                    low=1,
                    high=4,
                    shape=(self.num_ants,),
                    dtype=np.int8,
                ),
                "ants_count": spaces.Box(
                    low=0,
                    high=self.num_ants,
                    shape=(self.max_size, self.max_size),
                    dtype=np.int32,
                ),
                "food": spaces.Box(
                    low=0,
                    high=max(self.food_count, 1),
                    shape=(self.max_size, self.max_size),
                    dtype=np.int32,
                ),
                "bytes": spaces.Box(
                    low=0,
                    high=self.max_write_value,
                    shape=(self.max_size, self.max_size),
                    dtype=np.uint8,
                ),
                "hub_pos": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(2,),
                    dtype=np.int32,
                ),
                "active_grid_size": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(2,),
                    dtype=np.int32,
                ),
            }
        )

        self.current_size = self.start_size
        self.stage_index = 0
        self.completed_stages = 0
        self.global_step_count = 0
        self._env: AntByteForagingEnv | None = None

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

        self.current_size = self.start_size
        self.stage_index = 0
        self.completed_stages = 0
        self.global_step_count = 0
        obs, base_info = self._reset_stage(options=options)
        return self._pad_obs(obs), self._build_info(
            base_info,
            advanced_stage=False,
            completed_stage_size=0,
            completed_stage_delivered_food=0,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, int]]:
        if self._env is None:
            raise RuntimeError("reset must be called before step.")

        obs, reward, _, _, base_info = self._env.step(action)
        self.global_step_count += 1

        stage_delivered_food = int(base_info["delivered_food"])
        stage_complete = stage_delivered_food >= self.cookies_per_stage
        final_stage_complete = stage_complete and self.current_size >= self.max_size
        truncated = self.global_step_count >= self.max_steps
        terminated = final_stage_complete
        advanced_stage = False
        completed_stage_size = 0
        completed_stage_delivered_food = 0

        if stage_complete:
            completed_stage_size = self.current_size
            completed_stage_delivered_food = stage_delivered_food
            self.completed_stages += 1

        if stage_complete and not final_stage_complete and not truncated:
            self.current_size += 1
            self.stage_index += 1
            advanced_stage = True
            obs, base_info = self._reset_stage(options=None)

        padded_obs = self._pad_obs(obs)
        info = self._build_info(
            base_info,
            advanced_stage=advanced_stage,
            completed_stage_size=completed_stage_size,
            completed_stage_delivered_food=completed_stage_delivered_food,
        )
        return padded_obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        if self._env is None:
            return None
        return self._env.render()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None

    @staticmethod
    def _validate_constructor_args(
        *,
        start_size: int,
        max_size: int,
        cookies_per_stage: int,
        max_steps: int,
        num_ants: int,
        food_count: int,
        food_source_count: int,
        render_mode: str | None,
        tile_size: int,
        step_penalty: float,
        write_penalty: float,
        write_bits: int,
        actor_vision_radius: int,
    ) -> None:
        if start_size <= 0:
            raise ValueError("start_size must be positive.")
        if max_size < start_size:
            raise ValueError("max_size must be at least start_size.")
        if cookies_per_stage <= 0:
            raise ValueError("cookies_per_stage must be positive.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if num_ants <= 0:
            raise ValueError("num_ants must be positive.")
        if food_source_count != 2:
            raise ValueError("autocurriculum stages use exactly two food sources.")
        if food_count != cookies_per_stage * food_source_count:
            raise ValueError("food_count must equal cookies_per_stage * food_source_count.")
        if food_count > 0 and start_size * start_size <= 1:
            raise ValueError("food_count requires at least one non-hub tile.")
        if food_source_count <= 0:
            raise ValueError("food_source_count must be positive.")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
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
        if render_mode is not None and render_mode not in AntByteForagingEnv.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")
        _validate_far_food_capacity(
            size=int(start_size),
            source_count=int(food_source_count),
            vision_radius=int(actor_vision_radius),
        )

    def _reset_stage(
        self,
        *,
        options: dict[str, Any] | None,
    ) -> tuple[ObsType, dict[str, int]]:
        if self._env is not None:
            self._env.close()
        stage_options = self._stage_reset_options(options)
        self._env = AntByteForagingEnv(
            width=self.current_size,
            height=self.current_size,
            num_ants=self.num_ants,
            food_count=self.food_count,
            food_source_count=self.food_source_count,
            max_steps=self.max_steps,
            render_mode=self.render_mode,
            tile_size=self.tile_size,
            random_food=False,
            random_hub=False,
            step_penalty=self.step_penalty,
            write_penalty=self.write_penalty,
            write_bits=self.write_bits,
            write_while_moving=self.write_while_moving,
        )
        stage_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        return self._env.reset(seed=stage_seed, options=stage_options)

    def _stage_reset_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        reset_options = dict(options or {})
        hub_pos = reset_options.get("hub_pos")
        if hub_pos is None:
            hub_pos = self._sample_hub_pos()
        else:
            hub_pos = tuple(int(coord) for coord in self._coerce_position(hub_pos))
        food_positions = reset_options.get("food_positions")
        if food_positions is None:
            food_positions = self._sample_food_positions(hub_pos)
        else:
            food_positions = [
                tuple(int(coord) for coord in self._coerce_position(raw_pos))
                for raw_pos in food_positions
            ]
            self._validate_food_positions(food_positions, hub_pos=hub_pos)
        return {"hub_pos": hub_pos, "food_positions": food_positions}

    def _sample_hub_pos(self) -> tuple[int, int]:
        if self.random_hub:
            return (
                int(self.np_random.integers(0, self.current_size)),
                int(self.np_random.integers(0, self.current_size)),
            )
        center = self.current_size // 2
        return center, center

    def _sample_food_positions(self, hub_pos: tuple[int, int]) -> list[tuple[int, int]]:
        candidates = _far_food_candidates(
            size=self.current_size,
            hub_pos=hub_pos,
            vision_radius=self.actor_vision_radius,
        )
        if len(candidates) < self.food_source_count:
            raise ValueError(
                "active stage is too small to place two food sources outside initial actor vision."
            )
        if self.random_food:
            chosen_indices = self.np_random.choice(
                len(candidates),
                size=self.food_source_count,
                replace=False,
            )
            return [candidates[int(index)] for index in np.atleast_1d(chosen_indices)]
        return candidates[: self.food_source_count]

    def _validate_food_positions(
        self,
        positions: list[tuple[int, int]],
        *,
        hub_pos: tuple[int, int],
    ) -> None:
        if len(positions) != self.food_source_count:
            raise ValueError("autocurriculum stages require exactly two food positions.")
        allowed = set(
            _far_food_candidates(
                size=self.current_size,
                hub_pos=hub_pos,
                vision_radius=self.actor_vision_radius,
            )
        )
        for position in positions:
            if position not in allowed:
                raise ValueError(
                    "food_positions must be inside the active grid and outside initial actor vision."
                )

    def _coerce_position(self, raw_pos: Any) -> tuple[int, int]:
        position = np.asarray(raw_pos, dtype=np.int32)
        if position.shape != (2,):
            raise ValueError("positions must be two-item (x, y) pairs.")
        x_pos, y_pos = int(position[0]), int(position[1])
        if not (0 <= x_pos < self.current_size and 0 <= y_pos < self.current_size):
            raise ValueError(f"position {(x_pos, y_pos)!r} is outside the active grid.")
        return x_pos, y_pos

    def _pad_obs(self, obs: ObsType) -> ObsType:
        padded_obs: ObsType = {
            "ants_pos": obs["ants_pos"].astype(np.int32, copy=True),
            "ants_carrying": obs["ants_carrying"].astype(np.int8, copy=True),
            "ants_facing": obs["ants_facing"].astype(np.int8, copy=True),
            "ants_count": np.zeros((self.max_size, self.max_size), dtype=np.int32),
            "food": np.zeros((self.max_size, self.max_size), dtype=np.int32),
            "bytes": np.zeros((self.max_size, self.max_size), dtype=np.uint8),
            "hub_pos": obs["hub_pos"].astype(np.int32, copy=True),
            "active_grid_size": np.array([self.current_size, self.current_size], dtype=np.int32),
        }
        active_slice = (slice(0, self.current_size), slice(0, self.current_size))
        padded_obs["ants_count"][active_slice] = obs["ants_count"]
        padded_obs["food"][active_slice] = obs["food"]
        padded_obs["bytes"][active_slice] = obs["bytes"]
        return padded_obs

    def _build_info(
        self,
        base_info: dict[str, int],
        *,
        advanced_stage: bool,
        completed_stage_size: int,
        completed_stage_delivered_food: int,
    ) -> dict[str, int]:
        return {
            **base_info,
            "stage_size": int(self.current_size),
            "stage_index": int(self.stage_index),
            "stage_delivered_food": int(base_info["delivered_food"]),
            "stage_step_count": int(base_info["step_count"]),
            "cookies_per_stage": int(self.cookies_per_stage),
            "global_step_count": int(self.global_step_count),
            "global_step_budget": int(self.max_steps),
            "remaining_global_steps": int(max(0, self.max_steps - self.global_step_count)),
            "completed_stages": int(self.completed_stages),
            "advanced_stage": int(advanced_stage),
            "completed_stage_size": int(completed_stage_size),
            "completed_stage_delivered_food": int(completed_stage_delivered_food),
        }


def _far_food_candidates(
    *,
    size: int,
    hub_pos: tuple[int, int],
    vision_radius: int,
) -> list[tuple[int, int]]:
    hub_x, hub_y = hub_pos
    return [
        (x_pos, y_pos)
        for y_pos in range(size)
        for x_pos in range(size)
        if (x_pos, y_pos) != hub_pos
        and max(abs(x_pos - hub_x), abs(y_pos - hub_y)) > vision_radius
    ]


def _validate_far_food_capacity(
    *,
    size: int,
    source_count: int,
    vision_radius: int,
) -> None:
    for hub_y in range(size):
        for hub_x in range(size):
            if (
                len(
                    _far_food_candidates(
                        size=size,
                        hub_pos=(hub_x, hub_y),
                        vision_radius=vision_radius,
                    )
                )
                < source_count
            ):
                raise ValueError(
                    "start_size is too small to place two food sources outside initial actor vision."
                )
