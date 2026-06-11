"""Thin ant-agent adapters for model-controlled rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ant_byte_env.env import (
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_STAY,
    MOVE_UP,
    ObsType,
)

OUT_OF_BOUNDS_CODE = -1
HUB_CODE = 256
FOOD_BASE_CODE = 512


@dataclass(frozen=True)
class AntAction:
    """One ant's environment action."""

    move: int
    write_byte: int = 0

    def to_array(self) -> np.ndarray:
        return np.array([self.move, self.write_byte], dtype=np.int64)


@dataclass(frozen=True)
class AntContext:
    """Side information supplied alongside the local observation matrix."""

    ant_id: int
    position: tuple[int, int]
    carrying_food: bool
    hub_position: tuple[int, int]
    info: dict[str, Any]


@dataclass(frozen=True)
class AntTransition:
    """Feedback sent from an ant/controller to a trainable model."""

    ant_id: int
    observation: np.ndarray
    action: AntAction
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class AntModel(Protocol):
    """Protocol for custom models used by `AntAgent`."""

    def act(self, observation: np.ndarray, context: AntContext) -> AntAction:
        """Choose an action from a local observation."""

    def observe(self, transition: AntTransition) -> None:
        """Receive a transition for learning or replay storage."""


class AntAgent:
    """Converts Gymnasium observations into local MxM model inputs."""

    def __init__(
        self,
        ant_id: int,
        model: object,
        observation_size: int = 5,
        deterministic: bool = False,
    ) -> None:
        if observation_size <= 0 or observation_size % 2 == 0:
            raise ValueError("observation_size must be a positive odd integer.")
        if ant_id < 0:
            raise ValueError("ant_id must be non-negative.")

        self.ant_id = ant_id
        self.model = model
        self.observation_size = observation_size
        self.deterministic = deterministic
        self._last_observation: np.ndarray | None = None
        self._last_action: AntAction | None = None

    def act(self, obs: ObsType, info: dict[str, Any] | None = None) -> AntAction:
        local_obs = local_observation_matrix(
            obs=obs,
            ant_id=self.ant_id,
            observation_size=self.observation_size,
        )
        context = ant_context(obs=obs, ant_id=self.ant_id, info=info or {})
        action = parse_ant_action(self._call_model(local_obs, context))

        self._last_observation = local_obs
        self._last_action = action
        return action

    def observe(
        self,
        next_obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        if self._last_observation is None or self._last_action is None:
            return
        if not hasattr(self.model, "observe"):
            return

        transition = AntTransition(
            ant_id=self.ant_id,
            observation=self._last_observation,
            action=self._last_action,
            reward=reward,
            next_observation=local_observation_matrix(
                obs=next_obs,
                ant_id=self.ant_id,
                observation_size=self.observation_size,
            ),
            terminated=terminated,
            truncated=truncated,
            info=info or {},
        )
        self.model.observe(transition)

    def _call_model(self, local_obs: np.ndarray, context: AntContext) -> object:
        if hasattr(self.model, "act"):
            return self.model.act(local_obs, context)
        if hasattr(self.model, "predict"):
            prediction = self.model.predict(local_obs, deterministic=self.deterministic)
            if isinstance(prediction, tuple):
                return prediction[0]
            return prediction
        raise TypeError("model must provide either act(...) or predict(...).")


class AntColonyController:
    """Builds centralized env actions from one model-controlled ant per slot."""

    def __init__(self, ants: list[AntAgent]) -> None:
        if not ants:
            raise ValueError("AntColonyController requires at least one ant.")
        self.ants = ants
        self._last_actions: list[AntAction] = []

    def act(self, obs: ObsType, info: dict[str, Any] | None = None) -> np.ndarray:
        self._last_actions = [ant.act(obs, info=info) for ant in self.ants]
        return np.concatenate([action.to_array() for action in self._last_actions])

    def observe(
        self,
        next_obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        for ant in self.ants:
            ant.observe(
                next_obs=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )


def ant_context(obs: ObsType, ant_id: int, info: dict[str, Any]) -> AntContext:
    position = tuple(int(value) for value in obs["ants_pos"][ant_id])
    hub_position = tuple(int(value) for value in obs["hub_pos"])
    return AntContext(
        ant_id=ant_id,
        position=position,
        carrying_food=bool(obs["ants_carrying"][ant_id]),
        hub_position=hub_position,
        info=dict(info),
    )


def local_observation_matrix(
    *,
    obs: ObsType,
    ant_id: int,
    observation_size: int,
) -> np.ndarray:
    """Encode an ant-centered MxM local view as a single integer matrix."""

    if observation_size <= 0 or observation_size % 2 == 0:
        raise ValueError("observation_size must be a positive odd integer.")

    radius = observation_size // 2
    center_x, center_y = (int(value) for value in obs["ants_pos"][ant_id])
    height, width = obs["bytes"].shape
    matrix = np.full(
        (observation_size, observation_size),
        OUT_OF_BOUNDS_CODE,
        dtype=np.int32,
    )

    for local_y in range(observation_size):
        y_pos = center_y + local_y - radius
        if not 0 <= y_pos < height:
            continue
        for local_x in range(observation_size):
            x_pos = center_x + local_x - radius
            if not 0 <= x_pos < width:
                continue
            matrix[local_y, local_x] = encode_tile(obs=obs, x_pos=x_pos, y_pos=y_pos)

    return matrix


def encode_tile(obs: ObsType, x_pos: int, y_pos: int) -> int:
    if int(obs["hub_pos"][0]) == x_pos and int(obs["hub_pos"][1]) == y_pos:
        return HUB_CODE

    food_amount = int(obs["food"][y_pos, x_pos])
    if food_amount > 0:
        return FOOD_BASE_CODE + food_amount

    return int(obs["bytes"][y_pos, x_pos])


def parse_ant_action(raw_action: object) -> AntAction:
    if isinstance(raw_action, AntAction):
        action = raw_action
    else:
        values = np.asarray(raw_action, dtype=np.int64).reshape(-1)
        if values.shape == (1,):
            action = AntAction(move=int(values[0]), write_byte=0)
        elif values.shape == (2,):
            action = AntAction(move=int(values[0]), write_byte=int(values[1]))
        else:
            raise ValueError("model action must be a move or a (move, write_byte) pair.")

    if action.move not in {MOVE_STAY, MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT}:
        raise ValueError("model move must be an integer from 0 to 4.")
    if not 0 <= action.write_byte <= 255:
        raise ValueError("model write_byte must be an integer from 0 to 255.")
    return action
