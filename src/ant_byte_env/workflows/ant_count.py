"""Ant-count curriculum helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def ant_count_training_args(
    base_args: Mapping[str, Any],
    *,
    communication_bits: int,
) -> dict[str, Any]:
    return {
        **base_args,
        "width": 25,
        "height": 25,
        "obs_width": 50,
        "obs_height": 50,
        "food_count": 23,
        "food_sources": 6,
        "cookie_distance": 11,
        "max_steps": 2500,
        "write_bits": int(communication_bits),
        "write_while_moving": True,
    }


def validate_ant_count_stages(*, ant_stages: Sequence[int], source_num_ants: int) -> None:
    if any(num_ants <= source_num_ants for num_ants in ant_stages):
        raise ValueError("ant stages must increase beyond the source checkpoint's ant count.")
    if not strictly_increasing(ant_stages):
        raise ValueError("ant stages must be increasing.")


def strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def ant_count_train_args(
    *,
    common_args: Sequence[str],
    experiment_name: str,
    target_num_ants: int,
    communication_bits: int,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    load_model: Path,
    run_dir: Path,
) -> list[str]:
    return [
        *common_args,
        "--exp-name",
        f"{experiment_name}_{target_num_ants}_ants",
        "--write-bits",
        str(communication_bits),
        "--num-ants",
        str(target_num_ants),
        "--total-timesteps",
        str(update_timesteps_per_stage * global_update_cap),
        "--load-model",
        str(load_model),
        "--run-dir",
        str(run_dir),
    ]


__all__ = [
    "ant_count_train_args",
    "ant_count_training_args",
    "strictly_increasing",
    "validate_ant_count_stages",
]
