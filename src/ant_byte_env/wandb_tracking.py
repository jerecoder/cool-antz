"""Optional Weights & Biases tracking helpers."""

from __future__ import annotations

import importlib
import hashlib
import re
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence


_MAX_WANDB_ARTIFACT_NAME_LENGTH = 128
_WANDB_ARTIFACT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


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
        init_kwargs = {
            "project": project,
            "entity": entity,
            "group": group,
            "name": name,
            "tags": list(tags) if tags is not None else None,
            "mode": mode,
            "dir": str(run_dir) if run_dir is not None else None,
            "config": dict(config or {}),
            "notes": notes,
            "reinit": "create_new",
        }
        self._run = self._init_run(init_kwargs)

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def _init_run(self, init_kwargs: Mapping[str, Any]) -> Any | None:
        if self._wandb is None:
            return None
        try:
            return self._wandb.init(**dict(init_kwargs))
        except Exception as exc:
            warnings.warn(
                "W&B init failed; resetting the local W&B service and retrying once: "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _teardown_wandb(self._wandb)
        try:
            return self._wandb.init(**{**dict(init_kwargs), "reinit": "create_new"})
        except Exception as exc:
            warnings.warn(
                "W&B init failed after retry; continuing with W&B disabled for this run: "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._wandb = None
            return None

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | float | None = None,
    ) -> None:
        if self._run is None:
            return
        try:
            self._run.log(dict(metrics), step=None if step is None else int(step))
        except Exception as exc:
            self._disable_after_log_failure("metric logging", exc)

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
        try:
            self._run.log(
                {key: self._wandb.Video(str(path), fps=int(fps), format="mp4")},
                step=None if step is None else int(step),
            )
        except Exception as exc:
            self._disable_after_log_failure("video logging", exc)

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
        try:
            artifact = self._wandb.Artifact(_wandb_artifact_name(name), type=artifact_type)
            artifact.add_file(str(path))
            self._run.log_artifact(
                artifact,
                aliases=list(aliases) if aliases is not None else None,
            )
        except Exception as exc:
            warnings.warn(
                "W&B artifact logging failed; continuing without disabling metric "
                f"or video logging for this run: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def finish(self) -> None:
        if self._run is None:
            return
        run = self._run
        self._run = None
        try:
            run.finish()
        except Exception as exc:
            warnings.warn(
                "W&B run finish failed during cleanup; continuing shutdown without "
                f"blocking the original training result: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _disable_after_log_failure(self, operation: str, exc: Exception) -> None:
        self._run = None
        warnings.warn(
            "W&B "
            f"{operation} failed; continuing with W&B disabled for this run: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


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


def _wandb_artifact_name(name: str) -> str:
    cleaned = _WANDB_ARTIFACT_NAME_PATTERN.sub("-", str(name)).strip(".-")
    if not cleaned:
        cleaned = "artifact"
    if len(cleaned) <= _MAX_WANDB_ARTIFACT_NAME_LENGTH:
        return cleaned

    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    suffix = f"-{digest}"
    head_length = _MAX_WANDB_ARTIFACT_NAME_LENGTH - len(suffix)
    head = cleaned[:head_length].rstrip(".-")
    if not head:
        head = "artifact"[:head_length]
    return f"{head}{suffix}"


def _teardown_wandb(wandb_module: Any) -> None:
    teardown = getattr(wandb_module, "teardown", None)
    if teardown is None:
        return
    try:
        teardown(exit_code=1)
    except TypeError:
        teardown()
    except Exception as exc:
        warnings.warn(
            "W&B teardown failed while recovering from init failure; retrying anyway: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
