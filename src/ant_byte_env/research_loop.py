"""Flexible autoresearch loop for improving forage performance."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ant_byte_env.autoresearch import (
    _stage_metrics_from_run_dir,
    assert_autoresearch_resources_available,
)
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.runs import write_json

DEFAULT_RESEARCH_LOOP_MATRIX = Path("autoresearch/loop.json")
RESEARCH_LOOP_ARG_EXCLUDES = {
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
    "wandb_notes",
}
AUTOCURRICULUM_ARG_EXCLUDES = {
    "total_timesteps",
    "save_model",
    "load_model",
    "run_dir",
    "wandb_project",
    "wandb_entity",
    "wandb_group",
    "wandb_run_name",
    "wandb_mode",
    "wandb_tags",
    "wandb_notes",
}


def load_research_loop_matrix(path: Path = DEFAULT_RESEARCH_LOOP_MATRIX) -> dict[str, Any]:
    """Load the active autoresearch loop definition."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("research loop matrix must be a JSON object.")
    if not isinstance(payload.get("experiments"), list) or not payload["experiments"]:
        raise ValueError("research loop matrix requires a non-empty experiments list.")
    ids = [str(entry.get("id", "")) for entry in payload["experiments"]]
    if any(not run_id for run_id in ids):
        raise ValueError("each research loop experiment needs an id.")
    duplicates = sorted({run_id for run_id in ids if ids.count(run_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate research loop experiment id: {duplicates[0]}")
    return payload


def research_loop_entry(matrix: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    for entry in matrix.get("experiments", []):
        if str(entry.get("id")) == run_id:
            return dict(entry)
    choices = ", ".join(str(entry.get("id")) for entry in matrix.get("experiments", []))
    raise ValueError(f"unknown research loop id {run_id!r}; choices: {choices}")


def build_research_experiment_plan(
    *,
    matrix_path: Path = DEFAULT_RESEARCH_LOOP_MATRIX,
    run_id: str,
    run_root: Path | None = None,
    global_update_cap: int | None = None,
    num_envs: int | None = None,
    num_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = None,
) -> dict[str, Any]:
    """Build one self-contained experiment plan from the active loop."""

    matrix = load_research_loop_matrix(matrix_path)
    entry = research_loop_entry(matrix, run_id=run_id)
    mode = str(entry.get("mode", "forage_curriculum"))
    if mode not in {"forage_curriculum", "autocurriculum"}:
        raise ValueError("research loop mode must be forage_curriculum or autocurriculum.")

    base_config = Path(str(entry.get("base_config", matrix["base_config"])))
    base_spec = load_experiment_config(base_config)
    if base_spec.backend != "jax":
        raise ValueError("research loop experiments require a JAX base config.")

    merged_args = {
        **base_spec.args,
        **dict(matrix.get("default_args", {})),
        **dict(entry.get("args", {})),
    }
    if num_envs is not None:
        merged_args["num_envs"] = int(num_envs)
    if num_steps is not None:
        merged_args["num_steps"] = int(num_steps)

    root = Path(str(entry.get("run_dir", f"{matrix.get('run_root', 'runs/autoresearch')}/{run_id}")))
    run_dir = _resolve_run_dir(root, matrix_root=Path(str(matrix.get("run_root", "."))), override=run_root)
    update_cap_overridden = global_update_cap is not None
    update_cap = int(
        global_update_cap
        if update_cap_overridden
        else entry.get("global_update_cap", matrix.get("default_global_update_cap", 1))
    )
    if update_cap <= 0:
        raise ValueError("global_update_cap must be positive.")

    if mode == "forage_curriculum":
        plan = _build_forage_research_plan(
            matrix=matrix,
            entry=entry,
            matrix_path=matrix_path,
            base_config=base_config,
            run_dir=run_dir,
            merged_args=merged_args,
            global_update_cap=update_cap,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            update_cap_overridden=update_cap_overridden,
        )
    else:
        plan = _build_autocurriculum_research_plan(
            matrix=matrix,
            entry=entry,
            matrix_path=matrix_path,
            base_config=base_config,
            run_dir=run_dir,
            merged_args=merged_args,
            global_update_cap=update_cap,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
        )
    plan["notes_markdown"] = research_experiment_markdown(plan)
    return plan


def execute_research_experiment_plan(
    plan: dict[str, Any],
    *,
    train_main: Callable[..., dict[str, float]] | None = None,
    run_curriculum: Callable[..., dict[str, Any]] | None = None,
    check_resources: bool = True,
    resume_completed: bool = True,
) -> dict[str, Any]:
    """Run one research-loop experiment and persist its plan, note, and summary."""

    if check_resources:
        assert_autoresearch_resources_available()
    if train_main is None:
        from ant_byte_env.training.jax_mappo.runner import main as train_main

    run_dir = Path(str(plan["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    note_path = run_dir / "experiment.md"
    summary_path = run_dir / "summary.json"
    note_path.write_text(str(plan["notes_markdown"]), encoding="utf-8")
    write_json(plan_path, {key: value for key, value in plan.items() if key != "notes_markdown"})

    if plan["mode"] == "forage_curriculum":
        summary = _execute_forage_plan(
            plan,
            train_main=train_main,
            run_curriculum=run_curriculum,
            resume_completed=resume_completed,
            plan_path=plan_path,
            note_path=note_path,
        )
    else:
        summary = _execute_autocurriculum_plan(
            plan,
            train_main=train_main,
            resume_completed=resume_completed,
            plan_path=plan_path,
            note_path=note_path,
        )

    payload = {
        **summary,
        "plan_path": str(plan_path),
        "note_path": str(note_path),
        "summary_path": str(summary_path),
    }
    write_json(summary_path, payload)
    return payload


def run_research_experiment(
    *,
    matrix_path: Path = DEFAULT_RESEARCH_LOOP_MATRIX,
    run_id: str,
    run_root: Path | None = None,
    global_update_cap: int | None = None,
    num_envs: int | None = None,
    num_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = None,
    check_resources: bool = True,
    resume_completed: bool = True,
) -> dict[str, Any]:
    """Build and execute one active-loop experiment."""

    plan = build_research_experiment_plan(
        matrix_path=matrix_path,
        run_id=run_id,
        run_root=run_root,
        global_update_cap=global_update_cap,
        num_envs=num_envs,
        num_steps=num_steps,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
    )
    return execute_research_experiment_plan(
        plan,
        check_resources=check_resources,
        resume_completed=resume_completed,
    )


def rank_research_loop_runs(
    *,
    matrix_path: Path = DEFAULT_RESEARCH_LOOP_MATRIX,
    run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Rank completed forage-loop runs by target-stage delivery quality."""

    matrix = load_research_loop_matrix(matrix_path)
    target_stage = str(matrix.get("target", {}).get("stage_name", "25x25"))
    selected_ids = list(run_ids or [str(entry["id"]) for entry in matrix["experiments"]])
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_id in selected_ids:
        entry = research_loop_entry(matrix, run_id=run_id)
        run_dir = Path(str(entry.get("run_dir", Path(str(matrix["run_root"])) / run_id)))
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            missing.append(run_id)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        target_metrics = _target_stage_metrics(summary, target_stage=target_stage)
        if not target_metrics:
            missing.append(run_id)
            continue
        score = _promotion_score(target_metrics)
        rows.append(
            {
                "id": run_id,
                "title": entry.get("title", ""),
                "family": entry.get("family", ""),
                "target_stage": target_stage,
                "score": score,
                "episode_return": float(target_metrics.get("episode_return", 0.0)),
                "delivery_events": float(target_metrics.get("delivery_events", 0.0)),
                "pickup_events": float(target_metrics.get("pickup_events", 0.0)),
                "remaining_food": float(target_metrics.get("final_mean_remaining_food", 0.0)),
                "summary_path": str(summary_path),
            }
        )
    return {
        "target_stage": target_stage,
        "ranked": sorted(rows, key=lambda row: float(row["score"]), reverse=True),
        "missing": missing,
    }


def research_experiment_markdown(plan: Mapping[str, Any]) -> str:
    target = dict(plan.get("target", {}))
    lines = [
        f"# {plan['id']}: {plan.get('title', '')}",
        "",
        f"Family: {plan.get('family', '')}",
        f"Mode: {plan.get('mode', '')}",
        f"Run directory: `{plan.get('run_dir', '')}`",
        "",
        "## Hypothesis",
        str(plan.get("hypothesis", "")),
        "",
        "## Intervention",
        str(plan.get("intervention", "")),
        "",
        "## Baseline To Beat",
        str(target.get("baseline", "")),
        "",
        "## Success Signal",
        str(plan.get("success_signal", "")),
        "",
        "## Report Notes",
        str(plan.get("report_notes", "")),
        "",
        "## Key Settings",
        "```json",
        json.dumps(plan.get("resolved_args", {}), indent=2, sort_keys=True),
        "```",
    ]
    if plan.get("mode") == "forage_curriculum":
        lines.extend(
            [
                "",
                "## Stage Schedule",
                "```json",
                json.dumps(plan.get("stages", []), indent=2, sort_keys=True),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_forage_research_plan(
    *,
    matrix: Mapping[str, Any],
    entry: Mapping[str, Any],
    matrix_path: Path,
    base_config: Path,
    run_dir: Path,
    merged_args: dict[str, Any],
    global_update_cap: int,
    wandb_project: str | None,
    wandb_mode: str | None,
    update_cap_overridden: bool,
) -> dict[str, Any]:
    stages = _research_stages(
        matrix=matrix,
        entry=entry,
        global_update_cap=global_update_cap,
        update_cap_overridden=update_cap_overridden,
    )
    max_width = max(int(stage["width"]) for stage in stages)
    max_height = max(int(stage["height"]) for stage in stages)
    merged_args["obs_width"] = max(int(merged_args.get("obs_width") or 0), max_width)
    merged_args["obs_height"] = max(int(merged_args.get("obs_height") or 0), max_height)
    common_args = config_args_to_argv(
        {
            key: value
            for key, value in merged_args.items()
            if key not in RESEARCH_LOOP_ARG_EXCLUDES
        }
    )
    _validate_jax_training_args(common_args)
    update_timesteps = int(merged_args["num_envs"]) * int(merged_args["num_steps"])
    checkpoint_dir = run_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / f"jax_mappo_forage_stage1_{stages[-1]['name']}.pkl"
    total_env_steps = sum(
        int(stage.get("global_update_cap", global_update_cap))
        * int(stage.get("num_steps", merged_args["num_steps"]))
        * int(merged_args["num_envs"])
        for stage in stages
    )
    return {
        "matrix_path": str(matrix_path),
        "id": str(entry["id"]),
        "title": str(entry.get("title", "")),
        "family": str(entry.get("family", "")),
        "mode": "forage_curriculum",
        "priority": int(entry.get("priority", 100)),
        "hypothesis": str(entry.get("hypothesis", "")),
        "intervention": str(entry.get("intervention", "")),
        "success_signal": str(entry.get("success_signal", "")),
        "report_notes": str(entry.get("report_notes", "")),
        "target": dict(matrix.get("target", {})),
        "base_config": str(base_config),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "final_checkpoint": str(final_checkpoint),
        "stage_sizes": [int(stage["width"]) for stage in stages],
        "stages": stages,
        "global_update_cap": int(global_update_cap),
        "update_timesteps_per_stage": update_timesteps,
        "total_train_env_steps": total_env_steps,
        "common_args": common_args,
        "resolved_args": {
            key: value
            for key, value in merged_args.items()
            if key not in {"save_model", "load_model", "run_dir"}
        },
        "wandb": _research_wandb_config(
            matrix=matrix,
            entry=entry,
            run_id=str(entry["id"]),
            project_override=wandb_project,
            mode_override=wandb_mode,
        ),
    }


def _build_autocurriculum_research_plan(
    *,
    matrix: Mapping[str, Any],
    entry: Mapping[str, Any],
    matrix_path: Path,
    base_config: Path,
    run_dir: Path,
    merged_args: dict[str, Any],
    global_update_cap: int,
    wandb_project: str | None,
    wandb_mode: str | None,
) -> dict[str, Any]:
    merged_args["autocurriculum"] = True
    update_timesteps = int(merged_args["num_envs"]) * int(merged_args["num_steps"])
    total_timesteps = update_timesteps * int(global_update_cap)
    merged_args["total_timesteps"] = total_timesteps
    checkpoint = run_dir / "checkpoints" / "model.pkl"
    wandb = _research_wandb_config(
        matrix=matrix,
        entry=entry,
        run_id=str(entry["id"]),
        project_override=wandb_project,
        mode_override=wandb_mode,
    )
    common_args = config_args_to_argv(
        {
            key: value
            for key, value in merged_args.items()
            if key not in AUTOCURRICULUM_ARG_EXCLUDES
        }
    )
    _validate_jax_training_args(common_args)
    training_argv = [
        *common_args,
        "--total-timesteps",
        str(total_timesteps),
        "--run-dir",
        str(run_dir),
        "--save-model",
        str(checkpoint),
        *_wandb_argv(wandb),
    ]
    return {
        "matrix_path": str(matrix_path),
        "id": str(entry["id"]),
        "title": str(entry.get("title", "")),
        "family": str(entry.get("family", "")),
        "mode": "autocurriculum",
        "priority": int(entry.get("priority", 100)),
        "hypothesis": str(entry.get("hypothesis", "")),
        "intervention": str(entry.get("intervention", "")),
        "success_signal": str(entry.get("success_signal", "")),
        "report_notes": str(entry.get("report_notes", "")),
        "target": dict(matrix.get("target", {})),
        "base_config": str(base_config),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "global_update_cap": int(global_update_cap),
        "update_timesteps": update_timesteps,
        "total_train_env_steps": total_timesteps,
        "common_args": common_args,
        "training_argv": training_argv,
        "resolved_args": {
            key: value
            for key, value in merged_args.items()
            if key not in {"save_model", "load_model", "run_dir"}
        },
        "wandb": wandb,
    }


def _execute_forage_plan(
    plan: Mapping[str, Any],
    *,
    train_main: Callable[..., dict[str, float]],
    run_curriculum: Callable[..., dict[str, Any]] | None,
    resume_completed: bool,
    plan_path: Path,
    note_path: Path,
) -> dict[str, Any]:
    if run_curriculum is None:
        from ant_byte_env.notebook_workflows import run_forage_curriculum as run_curriculum

    final_checkpoint = Path(str(plan["final_checkpoint"]))
    resumed = bool(resume_completed and final_checkpoint.exists())
    if resumed:
        curriculum = {
            "stage_metrics": [],
            "stage_checkpoint_paths": _planned_stage_checkpoints(plan),
            "final_checkpoint_path": str(final_checkpoint),
            "final_train_metrics": _stage_metrics_from_run_dir(Path(str(plan["run_dir"]))),
        }
    else:
        wandb = dict(plan["wandb"])
        curriculum = run_curriculum(
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
            wandb_notes=str(plan["notes_markdown"]),
            wandb_artifact_paths=[note_path, plan_path],
            wandb_artifact_prefix=f"research-{plan['id']}",
            wandb_video_stage_names=wandb.get("video_stage_names"),
            wandb_video_max_frames=wandb.get("video_max_frames"),
        )
    return {
        "id": plan["id"],
        "mode": plan["mode"],
        "run_dir": plan["run_dir"],
        "resumed": resumed,
        "final_checkpoint": str(final_checkpoint),
        "wandb": plan["wandb"],
        "curriculum": _jsonable(curriculum),
    }


def _execute_autocurriculum_plan(
    plan: Mapping[str, Any],
    *,
    train_main: Callable[..., dict[str, float]],
    resume_completed: bool,
    plan_path: Path,
    note_path: Path,
) -> dict[str, Any]:
    checkpoint = Path(str(plan["checkpoint"]))
    resumed = bool(resume_completed and checkpoint.exists())
    if resumed:
        train_metrics = _stage_metrics_from_run_dir(Path(str(plan["run_dir"])))
    else:
        argv = list(plan["training_argv"])
        notes = str(plan["notes_markdown"])
        if "--wandb-notes" not in argv:
            argv.extend(["--wandb-notes", notes])
        train_metrics = train_main(argv)
    artifact_log = _log_sidecar_research_artifacts(
        plan=plan,
        plan_path=plan_path,
        note_path=note_path,
    )
    return {
        "id": plan["id"],
        "mode": plan["mode"],
        "run_dir": plan["run_dir"],
        "resumed": resumed,
        "checkpoint": str(checkpoint),
        "wandb": plan["wandb"],
        "research_artifacts": artifact_log,
        "train_metrics": _jsonable(train_metrics),
    }


def _research_stages(
    *,
    matrix: Mapping[str, Any],
    entry: Mapping[str, Any],
    global_update_cap: int,
    update_cap_overridden: bool,
) -> list[dict[str, Any]]:
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

    profile = list(entry.get("stage_training_profile", matrix.get("stage_training_profile", [])))
    for stage in stages:
        stage["global_update_cap"] = int(global_update_cap)
        matching = _stage_profile_for_size(int(stage["width"]), profile)
        if matching:
            for key in ("num_steps", "gamma"):
                if key in matching:
                    stage[key] = matching[key]
            if not update_cap_overridden and "global_update_cap" in matching:
                stage["global_update_cap"] = int(matching["global_update_cap"])
    return stages


def _stage_profile_for_size(size: int, profile: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in profile:
        if size <= int(row["max_size"]):
            return row
    return profile[-1] if profile else {}


def _research_wandb_config(
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


def _wandb_argv(wandb: Mapping[str, Any]) -> list[str]:
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


def _resolve_run_dir(path: Path, *, matrix_root: Path, override: Path | None) -> Path:
    if override is None:
        return path
    try:
        relative = path.relative_to(matrix_root)
    except ValueError:
        relative = Path(path.name)
    return override / relative


def _validate_jax_training_args(args: Sequence[str]) -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    parse_args(list(args))


def _planned_stage_checkpoints(plan: Mapping[str, Any]) -> list[str]:
    checkpoint_dir = Path(str(plan["checkpoint_dir"]))
    return [
        str(checkpoint_dir / f"jax_mappo_forage_stage1_{stage['name']}.pkl")
        for stage in plan["stages"]
    ]


def _target_stage_metrics(summary: Mapping[str, Any], *, target_stage: str) -> dict[str, Any]:
    rows = list(summary.get("curriculum", {}).get("stage_metrics", []))
    matches = [dict(row) for row in rows if str(row.get("stage_name")) == target_stage]
    return matches[-1] if matches else {}


def _promotion_score(metrics: Mapping[str, Any]) -> float:
    episode_return = float(metrics.get("episode_return", 0.0))
    deliveries = float(metrics.get("delivery_events", 0.0))
    remaining = float(metrics.get("final_mean_remaining_food", 0.0))
    return episode_return + 0.02 * deliveries - 0.01 * remaining


def _log_sidecar_research_artifacts(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    note_path: Path,
) -> dict[str, Any]:
    from ant_byte_env.wandb_tracking import WandbTracker

    wandb = dict(plan["wandb"])
    tracker = WandbTracker(
        project=wandb.get("project"),
        entity=wandb.get("entity"),
        group=wandb.get("group"),
        name=f"{wandb.get('name')}-research-files" if wandb.get("name") else None,
        tags=[*list(wandb.get("tags") or []), "research-files"],
        mode=str(wandb.get("mode", "online")),
        run_dir=Path(str(plan["run_dir"])),
        config={
            "id": plan["id"],
            "mode": plan["mode"],
            "title": plan.get("title", ""),
            "family": plan.get("family", ""),
        },
        notes=str(plan.get("notes_markdown", "")),
    )
    for path in (note_path, plan_path):
        tracker.log_artifact(
            f"research-{plan['id']}-{path.stem}",
            path,
            artifact_type="research-plan",
            aliases=[str(plan["id"]), "latest"],
        )
    tracker.finish()
    return {
        "enabled": tracker.enabled,
        "logged_files": [str(note_path), str(plan_path)] if tracker.enabled else [],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
