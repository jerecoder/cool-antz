"""Checkpoint path helpers for workflow artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def forage_checkpoint_paths(
    checkpoint_dir: Path,
    stages: Sequence[Mapping[str, Any]],
) -> list[Path]:
    return [
        checkpoint_dir / f"jax_mappo_forage_stage1_{stage['name']}.pkl"
        for stage in stages
    ]


def exploration_checkpoint_paths(
    checkpoint_dir: Path,
    stages: Sequence[Mapping[str, Any]],
) -> list[Path]:
    return [
        checkpoint_dir / f"jax_mappo_explore_{stage['name']}.pkl"
        for stage in stages
    ]


def maze_exploration_checkpoint_paths(
    checkpoint_dir: Path,
    stages: Sequence[Mapping[str, Any]],
) -> list[Path]:
    return [
        checkpoint_dir / f"jax_mappo_maze_explore_{stage['name']}.pkl"
        for stage in stages
    ]


def communication_checkpoint_paths(run_dir: Path, bit_stages: Sequence[int]) -> list[Path]:
    return [
        run_dir / f"{bits}_bits" / "checkpoints" / "model.pkl"
        for bits in bit_stages
    ]


def ant_count_checkpoint_paths(run_dir: Path, ant_stages: Sequence[int]) -> list[Path]:
    return [
        run_dir / f"{num_ants}_ants" / "checkpoints" / "model.pkl"
        for num_ants in ant_stages
    ]


__all__ = [
    "ant_count_checkpoint_paths",
    "communication_checkpoint_paths",
    "exploration_checkpoint_paths",
    "forage_checkpoint_paths",
    "maze_exploration_checkpoint_paths",
]
