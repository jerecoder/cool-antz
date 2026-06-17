"""Autoresearch helpers for the single-ant 50x50 forage curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from ant_byte_env.autoresearch import (
    _resolve_matrix_path,
    _stage_metrics_from_run_dir,
    assert_autoresearch_resources_available,
)
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.runs import write_json

DEFAULT_FORAGE_50X50_SWEEP_MATRIX = Path("autoresearch/forage_50x50_sweep.json")
FORAGE_CURRICULUM_ARG_EXCLUDES = {
    "total_timesteps",
    "width",
    "height",
    "food_count",
    "food_sources",
    "cookie_distance",
    "max_steps",
    "save_model",
    "load_model",
    "run_dir",
    "wandb_project",
    "wandb_entity",
    "wandb_group",
    "wandb_run_name",
    "wandb_mode",
    "wandb_tags",
}
FORAGE_NO_CHEAT_ACTOR_VISION_RADIUS = 1
FORAGE_NO_CHEAT_NUM_ANTS = 1
FORAGE_DISALLOWED_ACTOR_HINT_KEYS = {
    "actor_food_direction",
    "actor_food_distance",
    "actor_food_position",
    "actor_hub_direction",
    "actor_hub_distance",
    "actor_hub_position",
    "food_vector",
    "hub_vector",
    "oracle_food",
    "oracle_hub",
}


def build_forage_50x50_sweep_plan(
    *,
    matrix_path: Path = DEFAULT_FORAGE_50X50_SWEEP_MATRIX,
    phase: str,
    run_id: str,
    run_root: Path | None = None,
    stage_sizes: Sequence[int] | None = None,
    global_update_cap: int | None = None,
    num_envs: int | None = None,
    num_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = None,
    wandb_video_stage_names: Sequence[str] | None = None,
    wandb_video_max_frames: int | None = None,
) -> dict[str, Any]:
    """Return one no-cheat 50x50 forage curriculum plan."""

    matrix = load_forage_50x50_sweep_matrix(matrix_path)
    entry = forage_50x50_sweep_entry(matrix, phase=phase, run_id=run_id)
    base_config = Path(str(matrix["base_config"]))
    base_spec = load_experiment_config(base_config)
    if base_spec.backend != "jax":
        raise ValueError("forage 50x50 autoresearch requires a JAX base config.")

    entry_args = dict(entry.get("args", {}))
    merged_args = {**base_spec.args, **dict(matrix.get("default_args", {})), **entry_args}
    if num_envs is not None:
        merged_args["num_envs"] = int(num_envs)
    if num_steps is not None:
        merged_args["num_steps"] = int(num_steps)
    _assert_forage_no_cheat_args(merged_args)

    selected_stage_sizes = _forage_stage_sizes(
        matrix=matrix,
        entry=entry,
        phase=phase,
        override_stage_sizes=stage_sizes,
    )
    stages = _forage_stages_for_sizes(selected_stage_sizes)
    max_width = max(int(stage["width"]) for stage in stages)
    max_height = max(int(stage["height"]) for stage in stages)
    merged_args.setdefault("obs_width", max_width)
    merged_args.setdefault("obs_height", max_height)
    merged_args["obs_width"] = max(int(merged_args["obs_width"]), max_width)
    merged_args["obs_height"] = max(int(merged_args["obs_height"]), max_height)

    update_cap = (
        global_update_cap
        if global_update_cap is not None
        else entry.get("global_update_cap", matrix.get("global_update_cap"))
    )
    if update_cap is None:
        raise ValueError("global_update_cap is required for forage 50x50 entries.")
    update_cap = int(update_cap)
    if update_cap <= 0:
        raise ValueError("global_update_cap must be positive.")

    run_dir = _resolve_matrix_path(
        Path(str(entry["run_dir"])),
        matrix_root=Path(str(matrix["run_root"])),
        override_root=run_root,
    )
    checkpoint_dir = run_dir / "checkpoints"
    common_args = config_args_to_argv(
        {
            key: value
            for key, value in merged_args.items()
            if key not in FORAGE_CURRICULUM_ARG_EXCLUDES
        }
    )
    _validate_forage_training_args(common_args)
    update_timesteps_per_stage = int(merged_args["num_envs"]) * int(merged_args["num_steps"])
    video_stage_names = _forage_wandb_video_stage_names(
        matrix=matrix,
        entry=entry,
        override=wandb_video_stage_names,
    )
    video_max_frames = _forage_wandb_video_max_frames(
        matrix=matrix,
        entry=entry,
        override=wandb_video_max_frames,
    )
    wandb = _forage_wandb_config(
        matrix=matrix,
        entry=entry,
        phase=phase,
        run_id=run_id,
        project_override=wandb_project,
        mode_override=wandb_mode,
    )
    final_checkpoint = checkpoint_dir / f"jax_mappo_forage_stage1_{stages[-1]['name']}.pkl"
    return {
        "matrix_path": str(matrix_path),
        "phase": phase,
        "id": run_id,
        "description": entry.get("description", ""),
        "depends_on": entry.get("depends_on"),
        "base_config": str(base_config),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "final_checkpoint": str(final_checkpoint),
        "stage_sizes": [int(size) for size in selected_stage_sizes],
        "stages": stages,
        "global_update_cap": update_cap,
        "update_timesteps_per_stage": update_timesteps_per_stage,
        "total_train_env_steps": update_timesteps_per_stage * update_cap * len(stages),
        "common_args": common_args,
        "resolved_args": {
            key: value
            for key, value in merged_args.items()
            if key not in {"save_model", "load_model", "run_dir"}
        },
        "wandb": {
            **wandb,
            "video_max_frames": video_max_frames,
            "video_stage_names": video_stage_names,
        },
        "no_cheat_invariants": {
            "num_ants": FORAGE_NO_CHEAT_NUM_ANTS,
            "actor_vision_radius": FORAGE_NO_CHEAT_ACTOR_VISION_RADIUS,
            "actor_observation": "local grid only; no food/hub coordinates or direction vectors",
            "central_critic": "may use padded global state for value learning",
        },
        "curriculum_command": {
            "argv": [
                "ant-byte",
                "autoresearch",
                "forage-run",
                "--matrix",
                str(matrix_path),
                "--phase",
                phase,
                "--id",
                run_id,
            ],
        },
    }


def execute_forage_50x50_sweep_plan(
    plan: dict[str, Any],
    *,
    train_main: Callable[..., dict[str, float]] | None = None,
    run_curriculum: Callable[..., dict[str, Any]] | None = None,
    check_resources: bool = True,
    resume_completed: bool = True,
) -> dict[str, Any]:
    """Execute one forage 50x50 curriculum plan and persist a compact summary."""

    if check_resources:
        assert_autoresearch_resources_available()
    if train_main is None:
        from ant_byte_env.training.jax_mappo.runner import main as train_main
    if run_curriculum is None:
        from ant_byte_env.notebook_workflows import run_forage_curriculum as run_curriculum

    run_dir = Path(str(plan["run_dir"]))
    plan_path = run_dir / "sweep_plan.json"
    summary_path = run_dir / "sweep_summary.json"
    write_json(plan_path, plan)

    final_checkpoint = Path(str(plan["final_checkpoint"]))
    resumed = bool(resume_completed and final_checkpoint.exists())
    if resumed:
        curriculum_result = {
            "stage_metrics": [],
            "stage_checkpoint_paths": _planned_stage_checkpoints(plan),
            "final_checkpoint_path": str(final_checkpoint),
            "final_train_metrics": _stage_metrics_from_run_dir(run_dir),
        }
    else:
        wandb = dict(plan["wandb"])
        curriculum_result = run_curriculum(
            stages=list(plan["stages"]),
            checkpoint_dir=Path(str(plan["checkpoint_dir"])),
            common_args=list(plan["common_args"]),
            update_timesteps_per_stage=int(plan["update_timesteps_per_stage"]),
            global_update_cap=int(plan["global_update_cap"]),
            train_main=train_main,
            wandb_project=wandb.get("project"),
            wandb_entity=wandb.get("entity"),
            wandb_group=wandb.get("group"),
            wandb_run_name=wandb.get("name"),
            wandb_mode=str(wandb.get("mode", "online")),
            wandb_tags=wandb.get("tags"),
            wandb_video_max_frames=wandb.get("video_max_frames"),
            wandb_video_stage_names=wandb.get("video_stage_names"),
        )

    summary = {
        "plan_path": str(plan_path),
        "summary_path": str(summary_path),
        "phase": plan["phase"],
        "id": plan["id"],
        "run_dir": str(run_dir),
        "resumed": resumed,
        "final_checkpoint": str(final_checkpoint),
        "wandb": plan["wandb"],
        "curriculum": _jsonable_curriculum_result(curriculum_result),
    }
    write_json(summary_path, summary)
    return summary


def load_forage_50x50_sweep_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("forage 50x50 sweep matrix must be a JSON object.")
    if "phases" not in payload or not isinstance(payload["phases"], dict):
        raise ValueError("forage 50x50 sweep matrix requires a phases object.")
    return payload


def forage_50x50_sweep_entry(
    matrix: dict[str, Any],
    *,
    phase: str,
    run_id: str,
) -> dict[str, Any]:
    phases = matrix.get("phases", {})
    if phase not in phases:
        choices = ", ".join(sorted(phases))
        raise ValueError(f"unknown forage 50x50 sweep phase {phase!r}; choices: {choices}")
    for entry in phases[phase]:
        if str(entry.get("id")) == run_id:
            return dict(entry)
    choices = ", ".join(str(entry.get("id")) for entry in phases[phase])
    raise ValueError(f"unknown forage 50x50 sweep id {run_id!r}; choices: {choices}")


def _forage_stage_sizes(
    *,
    matrix: dict[str, Any],
    entry: dict[str, Any],
    phase: str,
    override_stage_sizes: Sequence[int] | None,
) -> list[int]:
    if override_stage_sizes is not None:
        sizes = [int(size) for size in override_stage_sizes]
    elif "stage_sizes" in entry:
        sizes = [int(size) for size in entry["stage_sizes"]]
    else:
        key = "final_stage_sizes" if phase == "final" else "screening_stage_sizes"
        sizes = [int(size) for size in matrix.get(key, [])]
    if not sizes:
        raise ValueError("stage_sizes must not be empty.")
    if any(size < 4 or size > 50 for size in sizes):
        raise ValueError("stage_sizes must stay within the 4x4 through 50x50 curriculum.")
    if sizes != sorted(set(sizes)):
        raise ValueError("stage_sizes must be strictly increasing with no duplicates.")
    return sizes


def _forage_stages_for_sizes(sizes: Sequence[int]) -> list[dict[str, Any]]:
    from ant_byte_env.notebook_workflows import build_forage_curriculum_stages

    return [
        dict(stage)
        for stage in build_forage_curriculum_stages(tuple(int(size) for size in sizes))
    ]


def _validate_forage_training_args(common_args: Sequence[str]) -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    parse_args(list(common_args))


def _assert_forage_no_cheat_args(args: dict[str, Any]) -> None:
    if int(args.get("num_ants", FORAGE_NO_CHEAT_NUM_ANTS)) != FORAGE_NO_CHEAT_NUM_ANTS:
        raise ValueError("forage 50x50 autoresearch is scoped to exactly one ant.")
    if (
        int(args.get("actor_vision_radius", FORAGE_NO_CHEAT_ACTOR_VISION_RADIUS))
        != FORAGE_NO_CHEAT_ACTOR_VISION_RADIUS
    ):
        raise ValueError("forage 50x50 autoresearch must keep actor_vision_radius at 1.")
    disallowed = sorted(FORAGE_DISALLOWED_ACTOR_HINT_KEYS & set(args))
    if disallowed:
        names = ", ".join(disallowed)
        raise ValueError(f"forage 50x50 autoresearch forbids actor oracle hints: {names}")


def _forage_wandb_video_stage_names(
    *,
    matrix: dict[str, Any],
    entry: dict[str, Any],
    override: Sequence[str] | None,
) -> list[str] | None:
    if override is not None:
        return [str(name) for name in override]
    value = entry.get("wandb_video_stage_names", matrix.get("wandb_video_stage_names"))
    if value is None:
        return None
    return [str(name) for name in value]


def _forage_wandb_video_max_frames(
    *,
    matrix: dict[str, Any],
    entry: dict[str, Any],
    override: int | None,
) -> int:
    value = override
    if value is None:
        value = int(entry.get("wandb_video_max_frames", matrix.get("wandb_video_max_frames", 600)))
    if int(value) < 0:
        raise ValueError("wandb_video_max_frames must be non-negative.")
    return int(value)


def _forage_wandb_config(
    *,
    matrix: dict[str, Any],
    entry: dict[str, Any],
    phase: str,
    run_id: str,
    project_override: str | None,
    mode_override: str | None,
) -> dict[str, Any]:
    base = dict(matrix.get("wandb", {}))
    entry_wandb = dict(entry.get("wandb", {}))
    payload = {**base, **entry_wandb}
    if project_override is not None:
        payload["project"] = project_override
    if mode_override is not None:
        payload["mode"] = mode_override
    payload.setdefault("project", None)
    payload.setdefault("entity", None)
    payload.setdefault("group", "forage_curriculum_50x50")
    payload.setdefault("mode", "online")
    payload.setdefault("name", f"forage-50x50-{phase}-{run_id}")
    tags = [
        *[str(tag) for tag in base.get("tags", [])],
        *[str(tag) for tag in entry_wandb.get("tags", [])],
        phase,
        run_id,
    ]
    payload["tags"] = list(dict.fromkeys(tags))
    return {
        "project": payload.get("project"),
        "entity": payload.get("entity"),
        "group": payload.get("group"),
        "name": payload.get("name"),
        "mode": payload.get("mode"),
        "tags": payload.get("tags"),
    }


def _planned_stage_checkpoints(plan: dict[str, Any]) -> list[str]:
    checkpoint_dir = Path(str(plan["checkpoint_dir"]))
    return [
        str(checkpoint_dir / f"jax_mappo_forage_stage1_{stage['name']}.pkl")
        for stage in plan["stages"]
    ]


def _jsonable_curriculum_result(result: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in result.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value
