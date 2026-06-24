"""Artifact path helpers for archived research-loop runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_run_dir(path: Path, *, matrix_root: Path, override: Path | None) -> Path:
    if override is None:
        return path
    try:
        relative = path.relative_to(matrix_root)
    except ValueError:
        relative = Path(path.name)
    return override / relative


def forage_stage_checkpoint_path(
    checkpoint_dir: Path,
    stage: Mapping[str, Any],
    *,
    selected: bool,
) -> Path:
    suffix = (
        "_best"
        if selected and bool(stage.get("select_best_checkpoint", False))
        else ""
    )
    return checkpoint_dir / f"jax_mappo_forage_stage1_{stage['name']}{suffix}.pkl"


def planned_stage_checkpoints(plan: Mapping[str, Any]) -> list[str]:
    checkpoint_dir = Path(str(plan["checkpoint_dir"]))
    return [
        str(forage_stage_checkpoint_path(checkpoint_dir, stage, selected=True))
        for stage in plan["stages"]
    ]


__all__ = [
    "forage_stage_checkpoint_path",
    "planned_stage_checkpoints",
    "resolve_run_dir",
]
