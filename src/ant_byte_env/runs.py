"""Run directory helpers for generated experiment artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_run_id(now: datetime | None = None) -> str:
    actual = datetime.now(UTC) if now is None else now.astimezone(UTC)
    return actual.strftime("%Y%m%dT%H%M%SZ")


def prepare_run_dir(root: Path, experiment_name: str, *, run_id: str | None = None) -> Path:
    base = Path(root) / experiment_name
    stem = run_id or utc_run_id()
    candidate = base / stem
    suffix = 2
    while candidate.exists():
        candidate = base / f"{stem}-{suffix:02d}"
        suffix += 1

    (candidate / "checkpoints").mkdir(parents=True)
    (candidate / "media").mkdir()
    return candidate


def ensure_run_structure(run_dir: Path) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "checkpoints").mkdir(exist_ok=True)
    (Path(run_dir) / "media").mkdir(exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_metrics(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
