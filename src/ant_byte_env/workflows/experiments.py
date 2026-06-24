"""Experiment setup helpers for notebook workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ant_byte_env.experiments import ExperimentSpec, load_experiment_config


def load_jax_experiment(config_path: Path) -> ExperimentSpec:
    experiment = load_experiment_config(config_path)
    if experiment.backend != "jax":
        raise ValueError(f"Expected a JAX experiment config, got {experiment.backend!r}.")
    return experiment


def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root / resolved


def run_jax_smoke(train_main: Callable[..., dict[str, float]]) -> dict[str, float]:
    return train_main(
        [
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--max-steps",
            "8",
            "--write-bits",
            "1",
            "--hidden-size",
            "16",
            "--seed",
            "11",
            "--quiet",
        ]
    )


__all__ = [
    "load_jax_experiment",
    "resolve_project_path",
    "run_jax_smoke",
]
