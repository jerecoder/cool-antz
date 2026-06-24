"""Rollout preview settings shared by notebook workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

NOTEBOOK_ROLLOUT_TILE_SIZE = 16
NOTEBOOK_ROLLOUT_SEED_OFFSET = 100_000
NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE = 1.0


def notebook_rollout_policy_temperature(
    metadata: Mapping[str, Any],
    *,
    key: str = "rollout_policy_temperature",
    default: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
) -> float:
    return validate_rollout_policy_temperature(
        metadata.get(key, default),
        name=key,
    )


def validate_rollout_policy_temperature(value: object, *, name: str) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative float.") from exc
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"{name} must be a non-negative float.")
    return temperature


__all__ = [
    "NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE",
    "NOTEBOOK_ROLLOUT_SEED_OFFSET",
    "NOTEBOOK_ROLLOUT_TILE_SIZE",
    "notebook_rollout_policy_temperature",
    "validate_rollout_policy_temperature",
]
