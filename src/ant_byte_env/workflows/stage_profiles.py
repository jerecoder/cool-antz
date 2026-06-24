"""Stage profile helpers for curriculum workflow metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ant_byte_env.workflows.args import update_timesteps
from ant_byte_env.workflows.cli import argv_int


def forage_stage_training_profiles(
    stages: Sequence[Mapping[str, Any]],
    *,
    common_args: Sequence[str],
    fallback_update_timesteps: int,
    fallback_update_cap: int,
) -> list[dict[str, Any]]:
    return [
        {
            "stage": str(stage["name"]),
            "global_update_cap": int(stage.get("global_update_cap", fallback_update_cap)),
            "num_steps": int(stage["num_steps"]) if "num_steps" in stage else None,
            "gamma": float(stage["gamma"]) if "gamma" in stage else None,
            "update_timesteps": forage_stage_update_timesteps(
                stage,
                common_args=common_args,
                fallback_update_timesteps=fallback_update_timesteps,
            ),
        }
        for stage in stages
    ]


def forage_stage_update_timesteps(
    stage: Mapping[str, Any],
    *,
    common_args: Sequence[str],
    fallback_update_timesteps: int,
) -> int:
    if "update_timesteps" in stage:
        return int(stage["update_timesteps"])
    if "num_steps" not in stage:
        return int(fallback_update_timesteps)
    num_envs = argv_int(common_args, "--num-envs")
    if num_envs is None:
        return int(fallback_update_timesteps)
    return update_timesteps(num_envs=num_envs, num_steps=int(stage["num_steps"]))


__all__ = [
    "forage_stage_training_profiles",
    "forage_stage_update_timesteps",
]
