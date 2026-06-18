"""Optional Weights & Biases tracking helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence


class WandbTracker:
    """Small no-op wrapper around W&B so tracking stays optional."""

    def __init__(
        self,
        *,
        project: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
        tags: Sequence[str] | None = None,
        mode: str = "online",
        run_dir: Path | None = None,
        config: Mapping[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        self._wandb: Any | None = None
        self._run: Any | None = None
        if project is None or mode == "disabled":
            return

        self._wandb = _import_wandb()
        self._run = self._wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=name,
            tags=list(tags) if tags is not None else None,
            mode=mode,
            dir=str(run_dir) if run_dir is not None else None,
            config=dict(config or {}),
            notes=notes,
        )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | float | None = None,
    ) -> None:
        if self._run is None:
            return
        self._run.log(dict(metrics), step=None if step is None else int(step))

    def log_video(
        self,
        key: str,
        path: Path,
        *,
        step: int | float | None = None,
        fps: int = 8,
    ) -> None:
        if self._run is None or self._wandb is None:
            return
        self._run.log(
            {key: self._wandb.Video(str(path), fps=int(fps), format="mp4")},
            step=None if step is None else int(step),
        )

    def log_artifact(
        self,
        name: str,
        path: Path,
        *,
        artifact_type: str,
        aliases: Sequence[str] | None = None,
    ) -> None:
        if self._run is None or self._wandb is None:
            return
        artifact = self._wandb.Artifact(name, type=artifact_type)
        artifact.add_file(str(path))
        self._run.log_artifact(
            artifact,
            aliases=list(aliases) if aliases is not None else None,
        )

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def _import_wandb() -> Any:
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        if exc.name != "wandb":
            raise
        raise RuntimeError(
            "W&B tracking was requested, but the optional 'wandb' package is not "
            "installed. Install it with `pip install -e '.[wandb]'` or `pip install wandb`."
        ) from exc
