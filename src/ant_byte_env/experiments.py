"""Experiment config loading for reproducible AntByte runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VARARG_CONFIG_ARGS = frozenset({"wandb_tags"})


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    backend: str
    args: dict[str, Any]
    description: str = ""
    metadata: dict[str, Any] | None = None


def load_experiment_config(path: Path) -> ExperimentSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object.")

    name = str(payload.get("name", "")).strip()
    backend = str(payload.get("backend", "")).strip()
    args = payload.get("args", {})
    metadata = payload.get("metadata", {})
    if not name:
        raise ValueError("experiment config requires a non-empty name.")
    if backend not in {"torch", "jax"}:
        raise ValueError("experiment backend must be either 'torch' or 'jax'.")
    if not isinstance(args, dict):
        raise ValueError("experiment args must be a JSON object.")
    if not isinstance(metadata, dict):
        raise ValueError("experiment metadata must be a JSON object.")

    return ExperimentSpec(
        name=name,
        backend=backend,
        args=dict(args),
        description=str(payload.get("description", "")),
        metadata=dict(metadata),
    )


def config_args_to_argv(args: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in args.items():
        option = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(option)
            continue
        if isinstance(value, list):
            if key in VARARG_CONFIG_ARGS:
                argv.extend([option, *[str(item) for item in value]])
                continue
            for item in value:
                argv.extend([option, str(item)])
            continue
        argv.extend([option, str(value)])
    return argv


def normalize_overrides(overrides: list[str] | None) -> list[str]:
    if not overrides:
        return []
    if overrides[0] == "--":
        return overrides[1:]
    return overrides


def resolve_training_argv(config_path: Path, overrides: list[str] | None = None) -> list[str]:
    spec = load_experiment_config(config_path)
    return [*config_args_to_argv(spec.args), *normalize_overrides(overrides)]


def namespace_to_jsonable(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload
