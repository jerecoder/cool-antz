"""Planning config helpers for archived research experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def research_stages(
    *,
    matrix: Mapping[str, Any],
    entry: Mapping[str, Any],
    global_update_cap: int,
    update_cap_overridden: bool,
) -> list[dict[str, Any]]:
    custom_stages = entry.get("custom_stages")
    if custom_stages is not None:
        stages = [dict(stage) for stage in custom_stages]
        if not stages:
            raise ValueError("custom_stages must contain at least one stage.")
        required_keys = {
            "name",
            "width",
            "height",
            "food_count",
            "food_sources",
            "cookie_distance",
            "max_steps",
        }
        for stage in stages:
            missing = sorted(required_keys.difference(stage))
            if missing:
                raise ValueError(f"custom stage is missing required keys: {missing}")
            stage.setdefault("global_update_cap", int(global_update_cap))
        return stages

    from ant_byte_env.notebook_workflows import build_forage_curriculum_stages

    sizes = [int(size) for size in entry.get("stage_sizes", matrix.get("default_stage_sizes", []))]
    if not sizes:
        raise ValueError("research loop forage experiments require stage_sizes.")
    if sizes != sorted(set(sizes)):
        raise ValueError("stage_sizes must be strictly increasing.")

    stages = [dict(stage) for stage in build_forage_curriculum_stages(tuple(sizes))]
    food_source_divisor = entry.get("food_source_divisor", matrix.get("food_source_divisor"))
    if food_source_divisor is not None:
        divisor = max(1, int(food_source_divisor))
        for stage in stages:
            food_count = int(stage["food_count"])
            stage["food_sources"] = max(1, min(food_count, math.ceil(food_count / divisor)))

    cookie_distance_scale = entry.get(
        "cookie_distance_scale",
        matrix.get("cookie_distance_scale"),
    )
    if cookie_distance_scale is not None:
        scale = float(cookie_distance_scale)
        if scale <= 0.0:
            raise ValueError("cookie_distance_scale must be positive.")
        for stage in stages:
            size = int(stage["width"])
            scaled = max(1, int(round(int(stage["cookie_distance"]) * scale)))
            stage["cookie_distance"] = min(scaled, max(1, size // 2))

    profile = list(entry.get("stage_training_profile", matrix.get("stage_training_profile", [])))
    stage_overrides = dict(entry.get("stage_overrides", {}))
    for stage in stages:
        stage["global_update_cap"] = int(global_update_cap)
        matching = _stage_profile_for_size(int(stage["width"]), profile)
        if matching:
            for key in ("num_steps", "gamma"):
                if key in matching:
                    stage[key] = matching[key]
            if not update_cap_overridden and "global_update_cap" in matching:
                stage["global_update_cap"] = int(matching["global_update_cap"])
        stage.update(stage_overrides)
    return stages


def research_wandb_config(
    *,
    matrix: Mapping[str, Any],
    entry: Mapping[str, Any],
    run_id: str,
    project_override: str | None,
    mode_override: str | None,
) -> dict[str, Any]:
    base = dict(matrix.get("wandb", {}))
    local = dict(entry.get("wandb", {}))
    payload = {**base, **local}
    if project_override is not None:
        payload["project"] = project_override
    if mode_override is not None:
        payload["mode"] = mode_override
    payload.setdefault("project", None)
    payload.setdefault("entity", None)
    payload.setdefault("group", "forage_improvement_loop")
    payload.setdefault("mode", "online")
    payload.setdefault("name", f"research-loop-{run_id}")
    tags = [
        *[str(tag) for tag in base.get("tags", [])],
        *[str(tag) for tag in local.get("tags", [])],
        str(entry.get("family", "")),
        run_id,
    ]
    video = dict(matrix.get("wandb_video", {}))
    video.update(dict(entry.get("wandb_video", {})))
    return {
        "project": payload.get("project"),
        "entity": payload.get("entity"),
        "group": payload.get("group"),
        "name": payload.get("name"),
        "mode": payload.get("mode"),
        "tags": [tag for tag in dict.fromkeys(tags) if tag],
        "video_stage_names": video.get("stage_names"),
        "video_max_frames": video.get("max_frames"),
    }


def research_evaluation_config(
    *,
    matrix: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "deterministic_episodes": 8,
        "sampled_episodes": 4,
        "seed_offset": 1_000_000,
        "shuffle_positions": True,
    }
    payload.update(dict(matrix.get("evaluation", {})))
    payload.update(dict(entry.get("evaluation", {})))
    action_modes: list[dict[str, Any]] = []
    for raw_mode in payload.get("action_modes", []):
        if isinstance(raw_mode, str):
            action_modes.append(
                {
                    "name": raw_mode,
                    "action_mode": raw_mode,
                    "episodes": int(payload.get("sampled_episodes", 0)),
                }
            )
            continue
        if not isinstance(raw_mode, Mapping):
            raise ValueError("evaluation action_modes entries must be strings or objects.")
        action_mode = str(raw_mode.get("action_mode", ""))
        if not action_mode:
            raise ValueError("evaluation action_modes entries require action_mode.")
        mode_payload = {
            "name": str(raw_mode.get("name", action_mode)),
            "action_mode": action_mode,
            "episodes": int(raw_mode.get("episodes", raw_mode.get("num_episodes", 0))),
            **(
                {"seed_offset": int(raw_mode["seed_offset"])}
                if raw_mode.get("seed_offset") is not None
                else {}
            ),
        }
        for key in ("move_temperature", "write_temperature"):
            if raw_mode.get(key) is not None:
                mode_payload[key] = float(raw_mode[key])
        action_modes.append(mode_payload)
    return {
        "deterministic_episodes": int(payload.get("deterministic_episodes", 0)),
        "sampled_episodes": int(payload.get("sampled_episodes", 0)),
        "seed_offset": int(payload.get("seed_offset", 1_000_000)),
        "shuffle_positions": bool(payload.get("shuffle_positions", True)),
        "action_modes": action_modes,
    }


def wandb_argv(wandb: Mapping[str, Any]) -> list[str]:
    argv: list[str] = []
    if wandb.get("project") is not None:
        argv.extend(["--wandb-project", str(wandb["project"])])
    if wandb.get("entity") is not None:
        argv.extend(["--wandb-entity", str(wandb["entity"])])
    if wandb.get("group") is not None:
        argv.extend(["--wandb-group", str(wandb["group"])])
    if wandb.get("name") is not None:
        argv.extend(["--wandb-run-name", str(wandb["name"])])
    if wandb.get("mode") is not None:
        argv.extend(["--wandb-mode", str(wandb["mode"])])
    tags = list(wandb.get("tags") or [])
    if tags:
        argv.append("--wandb-tags")
        argv.extend(str(tag) for tag in tags)
    return argv


def validate_jax_training_args(args: Sequence[str]) -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    parse_args(list(args))


def _stage_profile_for_size(size: int, profile: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in profile:
        if size <= int(row["max_size"]):
            return row
    return profile[-1] if profile else {}


__all__ = [
    "research_evaluation_config",
    "research_stages",
    "research_wandb_config",
    "validate_jax_training_args",
    "wandb_argv",
]
