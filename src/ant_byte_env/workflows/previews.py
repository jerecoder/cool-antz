"""W&B rollout preview option helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from ant_byte_env.workflows.rollouts import NOTEBOOK_ROLLOUT_SEED_OFFSET


def wandb_preview_enabled(max_frames: int | None) -> bool:
    return max_frames is None or int(max_frames) > 0


def wandb_preview_stage_enabled(
    stage_name: object,
    enabled_stage_names: Sequence[str] | None,
) -> bool:
    if enabled_stage_names is None:
        return True
    return str(stage_name) in {str(name) for name in enabled_stage_names}


def validate_wandb_preview_stage_names(
    stages: Sequence[Mapping[str, Any]],
    enabled_stage_names: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if enabled_stage_names is None:
        return None

    requested_stage_names = tuple(str(name) for name in enabled_stage_names)
    generated_stage_names = tuple(str(stage["name"]) for stage in stages)
    generated_stage_name_set = set(generated_stage_names)
    unknown_stage_names = tuple(
        name for name in requested_stage_names if name not in generated_stage_name_set
    )
    if unknown_stage_names:
        unknown_text = ", ".join(unknown_stage_names)
        available_text = ", ".join(generated_stage_names)
        raise ValueError(
            "wandb video stage names must match generated curriculum stage names; "
            f"unknown: {unknown_text}. Available stages: {available_text}"
        )
    return requested_stage_names


def validate_wandb_video_rollout_count(value: object) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("wandb video rollout count must be a positive integer.") from exc
    if count < 1:
        raise ValueError("wandb video rollout count must be a positive integer.")
    return count


def wandb_video_seed_offset_base(value: int | None) -> int:
    if value is None:
        return NOTEBOOK_ROLLOUT_SEED_OFFSET + int(time.time_ns() % 1_000_000_000)
    try:
        seed_offset_base = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("wandb video seed offset base must be non-negative.") from exc
    if seed_offset_base < 0:
        raise ValueError("wandb video seed offset base must be non-negative.")
    return seed_offset_base


def wandb_preview_video_key(
    *,
    prefix: str,
    stage_name: object,
    preview_index: int,
    preview_count: int,
) -> str:
    stage_key = f"{prefix}/{stage_name}"
    if preview_count == 1:
        return stage_key
    return f"{stage_key}/rollout_{preview_index + 1:02d}"


__all__ = [
    "validate_wandb_preview_stage_names",
    "validate_wandb_video_rollout_count",
    "wandb_preview_enabled",
    "wandb_preview_stage_enabled",
    "wandb_preview_video_key",
    "wandb_video_seed_offset_base",
]
