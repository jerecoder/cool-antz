"""Shared helpers for active autoresearch loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ant_byte_env.runtime.resources import (
    assert_notebook_resources_available,
    notebook_resource_snapshot,
)

AUTORESEARCH_MIN_DISK_FREE_GB = 5.0
AUTORESEARCH_MIN_MEM_AVAILABLE_GB = 4.0
AUTORESEARCH_MIN_SWAP_FREE_GB = 0.25
AUTORESEARCH_MAX_GPU_COMPUTE_MEMORY_MB = 1024


class AutoresearchResourceError(RuntimeError):
    """Raised when a long autoresearch run should not start on this machine."""


def _stage_metrics_from_run_dir(run_dir: Path) -> dict[str, float]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            return {
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            payload = json.loads(lines[-1])
            if isinstance(payload, dict):
                return {
                    str(key): float(value)
                    for key, value in payload.items()
                    if isinstance(value, (int, float))
                }

    return {}


def assert_autoresearch_resources_available(
    snapshot: dict[str, Any] | None = None,
    *,
    min_disk_free_gb: float = AUTORESEARCH_MIN_DISK_FREE_GB,
    min_mem_available_gb: float = AUTORESEARCH_MIN_MEM_AVAILABLE_GB,
    min_swap_free_gb: float = AUTORESEARCH_MIN_SWAP_FREE_GB,
    max_gpu_compute_memory_mb: int = AUTORESEARCH_MAX_GPU_COMPUTE_MEMORY_MB,
    context: str = "autoresearch",
) -> None:
    """Fail before a long autoresearch run when local resources look unsafe."""

    actual_snapshot = notebook_resource_snapshot() if snapshot is None else snapshot
    try:
        assert_notebook_resources_available(
            actual_snapshot,
            min_disk_free_gb=float(min_disk_free_gb),
            min_mem_available_gb=float(min_mem_available_gb),
            min_swap_free_gb=float(min_swap_free_gb),
            max_gpu_compute_memory_mb=int(max_gpu_compute_memory_mb),
        )
    except RuntimeError as exc:
        raise AutoresearchResourceError(
            f"Autoresearch resources look unsafe for {context}.\n{exc}"
        ) from exc
