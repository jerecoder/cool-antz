"""CLI parsing for timed-release cooperative JAX MAPPO experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ant_byte_env.training.jax_mappo.cli import parse_args as parse_base_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--release-interval", type=int, default=150)
    parser.add_argument("--initial-active-ants", type=int, default=1)
    parser.add_argument("--actor-only-warm-start", action="store_true")
    timed_args, remaining = parser.parse_known_args(argv)
    args = parse_base_args(remaining)
    args.release_interval = int(timed_args.release_interval)
    args.initial_active_ants = int(timed_args.initial_active_ants)
    args.actor_only_warm_start = bool(timed_args.actor_only_warm_start)
    return _validate_args(args)


def _validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.release_interval <= 0:
        raise ValueError("--release-interval must be positive.")
    if args.initial_active_ants <= 0:
        raise ValueError("--initial-active-ants must be positive.")
    if args.initial_active_ants > args.num_ants:
        raise ValueError("--initial-active-ants cannot exceed --num-ants.")
    if bool(getattr(args, "autocurriculum", False)):
        raise ValueError("timed-release roles do not support --autocurriculum in V1.")
    if bool(getattr(args, "distance_autocurriculum", False)):
        raise ValueError("timed-release roles do not support --distance-autocurriculum in V1.")
    if str(getattr(args, "jax_parallelism", "single")) != "single":
        raise ValueError("timed-release roles currently require --jax-parallelism single.")
    if bool(getattr(args, "reset_env_each_update", False)):
        raise ValueError("timed-release roles do not support --reset-env-each-update in V1.")
    if int(getattr(args, "lethal_food_count", 0)) > 0:
        raise ValueError("timed-release roles do not support lethal food in V1.")
    return args


def dry_run_payload(
    *,
    config_path: Path,
    experiment: str,
    argv: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "backend": "jax",
        "workflow": "timed_release_roles",
        "config": str(config_path),
        "experiment": experiment,
        "argv": list(argv),
        "resolved_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "release_schedule": {
            "release_interval": int(args.release_interval),
            "initial_active_ants": int(args.initial_active_ants),
            "release_steps": [
                max(0, (rank - int(args.initial_active_ants) + 1) * int(args.release_interval))
                for rank in range(int(args.num_ants))
            ],
        },
    }
