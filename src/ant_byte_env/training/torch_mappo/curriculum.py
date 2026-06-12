"""Curriculum reset and reward shaping helpers for Torch MAPPO."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from ant_byte_env.training.torch_mappo.observations import NumpyObs


def _sample_hub_position(
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if args.random_hub:
        return (
            int(rng.integers(0, args.width)),
            int(rng.integers(0, args.height)),
        )

    return (args.width // 2, args.height // 2)


def build_curriculum_reset_options(
    args: argparse.Namespace,
    *,
    seed: int | None = None,
) -> dict[str, Any] | None:
    rng = np.random.default_rng(seed)
    hub = _sample_hub_position(args=args, rng=rng)
    if args.random_food:
        return {"hub_pos": hub}

    distance = min(args.cookie_distance, max(args.width, args.height))
    offsets = ((distance, 0), (-distance, 0), (0, distance), (0, -distance))
    for x_offset, y_offset in offsets:
        candidate = (hub[0] + x_offset, hub[1] + y_offset)
        if 0 <= candidate[0] < args.width and 0 <= candidate[1] < args.height:
            return {"hub_pos": hub, "food_positions": [candidate]}

    for y_pos in range(args.height):
        for x_pos in range(args.width):
            if (x_pos, y_pos) != hub:
                return {"hub_pos": hub, "food_positions": [(x_pos, y_pos)]}

    return {"hub_pos": hub}


def _nearest_food_distance(position: np.ndarray, food_grid: np.ndarray) -> float | None:
    food_positions = np.argwhere(food_grid > 0)
    if food_positions.size == 0:
        return None

    x_pos, y_pos = int(position[0]), int(position[1])
    distances = np.abs(food_positions[:, 1] - x_pos) + np.abs(food_positions[:, 0] - y_pos)
    return float(distances.min())


def _distance_to_hub(position: np.ndarray, hub_pos: np.ndarray) -> float:
    return float(abs(int(position[0]) - int(hub_pos[0])) + abs(int(position[1]) - int(hub_pos[1])))


def compute_forage_curriculum_rewards(
    *,
    previous_obs: NumpyObs,
    next_obs: NumpyObs,
    env_rewards: np.ndarray,
    pickup_bonus: float,
    distance_bonus: float,
) -> np.ndarray:
    """Add simple pickup and target-progress rewards for the first curriculum."""

    shaped_rewards = env_rewards.astype(np.float32, copy=True)
    batch_size, num_agents = previous_obs["ants_carrying"].shape

    for env_index in range(batch_size):
        for agent_index in range(num_agents):
            was_carrying = bool(previous_obs["ants_carrying"][env_index, agent_index])
            is_carrying = bool(next_obs["ants_carrying"][env_index, agent_index])
            if not was_carrying and is_carrying:
                shaped_rewards[env_index] += float(pickup_bonus)

            previous_position = previous_obs["ants_pos"][env_index, agent_index]
            next_position = next_obs["ants_pos"][env_index, agent_index]
            if was_carrying:
                target_previous_distance = _distance_to_hub(
                    previous_position,
                    previous_obs["hub_pos"][env_index],
                )
                target_next_distance = _distance_to_hub(
                    next_position,
                    previous_obs["hub_pos"][env_index],
                )
            else:
                target_previous_distance = _nearest_food_distance(
                    previous_position,
                    previous_obs["food"][env_index],
                )
                target_next_distance = _nearest_food_distance(
                    next_position,
                    previous_obs["food"][env_index],
                )
                if target_previous_distance is None or target_next_distance is None:
                    continue

            progress = target_previous_distance - target_next_distance
            shaped_rewards[env_index] += float(distance_bonus) * progress

    return shaped_rewards
