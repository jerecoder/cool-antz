"""Flexible autoresearch loop for improving forage performance."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
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
    if mode not in {"forage_curriculum", "autocurriculum", "checkpoint_evaluation"}:
        raise ValueError(
            "research loop mode must be forage_curriculum, autocurriculum, "
            "or checkpoint_evaluation."
        )

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
    elif mode == "autocurriculum":
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
    else:
        plan = _build_checkpoint_evaluation_research_plan(
            matrix=matrix,
            entry=entry,
            matrix_path=matrix_path,
            base_config=base_config,
            run_dir=run_dir,
            merged_args=merged_args,
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
    evaluate_checkpoint: Callable[..., dict[str, float]] | None = None,
    check_resources: bool = True,
    resume_completed: bool = True,
    min_disk_free_gb: float = 5.0,
) -> dict[str, Any]:
    """Run one research-loop experiment and persist its plan, note, and summary."""

    if check_resources:
        assert_autoresearch_resources_available(
            min_disk_free_gb=min_disk_free_gb,
            context=f"research loop {plan['id']}",
        )
    if train_main is None and plan["mode"] != "checkpoint_evaluation":
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
    elif plan["mode"] == "autocurriculum":
        summary = _execute_autocurriculum_plan(
            plan,
            train_main=train_main,
            resume_completed=resume_completed,
            plan_path=plan_path,
            note_path=note_path,
        )
    else:
        summary = _execute_checkpoint_evaluation_plan(plan)

    payload: dict[str, Any] = {
        **summary,
        "plan_path": str(plan_path),
        "note_path": str(note_path),
        "summary_path": str(summary_path),
    }
    evaluation_path = run_dir / "evaluation.json"
    evaluation = _evaluate_research_plan_checkpoint(
        plan,
        summary=payload,
        evaluate_checkpoint=evaluate_checkpoint,
    )
    if evaluation:
        write_json(evaluation_path, evaluation)
        payload["evaluation"] = evaluation
        payload["evaluation_path"] = str(evaluation_path)
    write_json(summary_path, payload)
    payload["research_artifacts"] = _log_sidecar_research_artifacts(
        plan=plan,
        paths=[note_path, plan_path, summary_path, *([evaluation_path] if evaluation else [])],
        evaluation=evaluation,
    )
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
    min_disk_free_gb: float = 5.0,
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
        min_disk_free_gb=min_disk_free_gb,
    )


def run_research_loop(
    *,
    matrix_path: Path = DEFAULT_RESEARCH_LOOP_MATRIX,
    run_root: Path | None = None,
    run_ids: Sequence[str] | None = None,
    max_runs: int | None = None,
    global_update_cap: int | None = None,
    num_envs: int | None = None,
    num_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = None,
    min_disk_free_gb: float = 5.0,
    check_resources: bool = True,
    resume_completed: bool = True,
) -> dict[str, Any]:
    """Run pending active-loop experiments in priority order and write a ledger."""

    matrix = load_research_loop_matrix(matrix_path)
    entries = sorted(
        [research_loop_entry(matrix, run_id=str(entry["id"])) for entry in matrix["experiments"]],
        key=lambda entry: (int(entry.get("priority", 100)), str(entry["id"])),
    )
    requested = {str(run_id) for run_id in run_ids or []}
    selected = [entry for entry in entries if not requested or str(entry["id"]) in requested]
    if max_runs is not None:
        if int(max_runs) <= 0:
            raise ValueError("max_runs must be positive when provided.")
        selected = selected[: int(max_runs)]

    ledger_root = Path(str(matrix.get("run_root", "runs/autoresearch/forage_loop")))
    if run_root is not None:
        ledger_root = Path(run_root)
    ledger_root.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_root / "ledger.json"
    ledger_jsonl_path = ledger_root / "ledger.jsonl"

    results: list[dict[str, Any]] = []
    for entry in selected:
        run_id = str(entry["id"])
        summary = run_research_experiment(
            matrix_path=matrix_path,
            run_id=run_id,
            run_root=run_root,
            global_update_cap=global_update_cap,
            num_envs=num_envs,
            num_steps=num_steps,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            min_disk_free_gb=min_disk_free_gb,
            check_resources=check_resources,
            resume_completed=resume_completed,
        )
        ledger_row = {
            "finished_at": _utc_now(),
            "id": run_id,
            "mode": summary.get("mode"),
            "run_dir": summary.get("run_dir"),
            "summary_path": summary.get("summary_path"),
            "resumed": bool(summary.get("resumed", False)),
            "score": _summary_score(summary),
            "evaluation": summary.get("evaluation", {}),
        }
        results.append(ledger_row)
        with ledger_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ledger_row, sort_keys=True) + "\n")
        write_json(
            ledger_path,
            {
                "matrix_path": str(matrix_path),
                "updated_at": _utc_now(),
                "run_count": len(results),
                "results": results,
            },
        )

    return {
        "matrix_path": str(matrix_path),
        "ledger_path": str(ledger_path),
        "ledger_jsonl_path": str(ledger_jsonl_path),
        "results": results,
        "ranking": rank_research_loop_runs(matrix_path=matrix_path, run_ids=[row["id"] for row in results]),
    }


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
        evaluation = summary.get("evaluation", {})
        eval_score = _evaluation_score(evaluation) if isinstance(evaluation, Mapping) else None
        if not target_metrics and eval_score is None:
            missing.append(run_id)
            continue
        score = eval_score if eval_score is not None else _promotion_score(target_metrics)
        deterministic = evaluation.get("deterministic", {}) if isinstance(evaluation, Mapping) else {}
        sampled = evaluation.get("sampled", {}) if isinstance(evaluation, Mapping) else {}
        extra_evaluation = _extra_evaluation_summary(evaluation) if isinstance(evaluation, Mapping) else {}
        rows.append(
            {
                "id": run_id,
                "title": entry.get("title", ""),
                "family": entry.get("family", ""),
                "target_stage": target_stage,
                "score": score,
                "eval_deterministic_return": float(
                    deterministic.get("eval_mean_episode_return", 0.0)
                    if isinstance(deterministic, Mapping)
                    else 0.0
                ),
                "eval_sampled_return": float(
                    sampled.get("eval_mean_episode_return", 0.0)
                    if isinstance(sampled, Mapping)
                    else 0.0
                ),
                "eval_deterministic_delivered": float(
                    deterministic.get("eval_mean_delivered_food", 0.0)
                    if isinstance(deterministic, Mapping)
                    else 0.0
                ),
                "eval_sampled_delivered": float(
                    sampled.get("eval_mean_delivered_food", 0.0)
                    if isinstance(sampled, Mapping)
                    else 0.0
                ),
                "extra_evaluation": extra_evaluation,
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
        *(
            [f"Source checkpoint: `{plan['source_checkpoint']}`"]
            if plan.get("source_checkpoint")
            else []
        ),
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
        "## Evaluation Gate",
        "```json",
        json.dumps(plan.get("evaluation", {}), indent=2, sort_keys=True),
        "```",
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
    source_checkpoint = entry.get("source_checkpoint")
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
        **(
            {"source_checkpoint": str(source_checkpoint)}
            if source_checkpoint is not None
            else {}
        ),
        "stage_sizes": [int(stage["width"]) for stage in stages],
        "stages": stages,
        "global_update_cap": int(global_update_cap),
        "update_timesteps_per_stage": update_timesteps,
        "total_train_env_steps": total_env_steps,
        "common_args": common_args,
        "resolved_args": {
            key: value
            for key, value in merged_args.items()
            if key not in RESEARCH_LOOP_ARG_EXCLUDES
        },
        "evaluation": _research_evaluation_config(matrix=matrix, entry=entry),
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
        "evaluation": _research_evaluation_config(matrix=matrix, entry=entry),
        "wandb": wandb,
    }


def _build_checkpoint_evaluation_research_plan(
    *,
    matrix: Mapping[str, Any],
    entry: Mapping[str, Any],
    matrix_path: Path,
    base_config: Path,
    run_dir: Path,
    merged_args: dict[str, Any],
    wandb_project: str | None,
    wandb_mode: str | None,
) -> dict[str, Any]:
    source_checkpoint = entry.get("source_checkpoint")
    if source_checkpoint is None:
        raise ValueError("checkpoint_evaluation experiments require source_checkpoint.")
    checkpoint = Path(str(source_checkpoint))
    return {
        "matrix_path": str(matrix_path),
        "id": str(entry["id"]),
        "title": str(entry.get("title", "")),
        "family": str(entry.get("family", "")),
        "mode": "checkpoint_evaluation",
        "priority": int(entry.get("priority", 100)),
        "hypothesis": str(entry.get("hypothesis", "")),
        "intervention": str(entry.get("intervention", "")),
        "success_signal": str(entry.get("success_signal", "")),
        "report_notes": str(entry.get("report_notes", "")),
        "target": dict(matrix.get("target", {})),
        "base_config": str(base_config),
        "run_dir": str(run_dir),
        "source_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint),
        "final_checkpoint": str(checkpoint),
        "total_train_env_steps": 0,
        "common_args": [],
        "resolved_args": {
            key: value
            for key, value in merged_args.items()
            if key not in RESEARCH_LOOP_ARG_EXCLUDES
        },
        "evaluation": _research_evaluation_config(matrix=matrix, entry=entry),
        "wandb": _research_wandb_config(
            matrix=matrix,
            entry=entry,
            run_id=str(entry["id"]),
            project_override=wandb_project,
            mode_override=wandb_mode,
        ),
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
            initial_checkpoint=(
                Path(str(plan["source_checkpoint"]))
                if plan.get("source_checkpoint") is not None
                else None
            ),
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
    del plan_path, note_path
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
    return {
        "id": plan["id"],
        "mode": plan["mode"],
        "run_dir": plan["run_dir"],
        "resumed": resumed,
        "checkpoint": str(checkpoint),
        "wandb": plan["wandb"],
        "train_metrics": _jsonable(train_metrics),
    }


def _execute_checkpoint_evaluation_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = Path(str(plan["checkpoint"]))
    return {
        "id": plan["id"],
        "mode": plan["mode"],
        "run_dir": plan["run_dir"],
        "resumed": False,
        "checkpoint": str(checkpoint),
        "final_checkpoint": str(checkpoint),
        "wandb": plan["wandb"],
        "train_metrics": {},
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


def _research_evaluation_config(
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
        action_modes.append(
            {
                "name": str(raw_mode.get("name", action_mode)),
                "action_mode": action_mode,
                "episodes": int(raw_mode.get("episodes", raw_mode.get("num_episodes", 0))),
                **(
                    {"seed_offset": int(raw_mode["seed_offset"])}
                    if raw_mode.get("seed_offset") is not None
                    else {}
                ),
            }
        )
    return {
        "deterministic_episodes": int(payload.get("deterministic_episodes", 0)),
        "sampled_episodes": int(payload.get("sampled_episodes", 0)),
        "seed_offset": int(payload.get("seed_offset", 1_000_000)),
        "shuffle_positions": bool(payload.get("shuffle_positions", True)),
        "action_modes": action_modes,
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


def _evaluate_research_plan_checkpoint(
    plan: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    evaluate_checkpoint: Callable[..., dict[str, float]] | None,
) -> dict[str, Any]:
    evaluation = dict(plan.get("evaluation", {}))
    deterministic_episodes = int(evaluation.get("deterministic_episodes", 0))
    sampled_episodes = int(evaluation.get("sampled_episodes", 0))
    action_modes = [
        dict(row)
        for row in evaluation.get("action_modes", [])
        if isinstance(row, Mapping) and int(row.get("episodes", 0)) > 0
    ]
    if deterministic_episodes <= 0 and sampled_episodes <= 0 and not action_modes:
        return {}

    checkpoint_value = summary.get("final_checkpoint") or summary.get("checkpoint")
    if checkpoint_value is None:
        return {
            "error": "missing_checkpoint",
            "deterministic_episodes": deterministic_episodes,
            "sampled_episodes": sampled_episodes,
            "action_modes": action_modes,
        }
    checkpoint_path = Path(str(checkpoint_value))
    if not checkpoint_path.exists():
        return {
            "error": "missing_checkpoint_file",
            "checkpoint": str(checkpoint_path),
            "deterministic_episodes": deterministic_episodes,
            "sampled_episodes": sampled_episodes,
            "action_modes": action_modes,
        }

    if evaluate_checkpoint is None:
        from ant_byte_env.training.jax_mappo.evaluation import evaluate_checkpoint

    seed_offset = int(evaluation.get("seed_offset", 1_000_000))
    shuffle_positions = bool(evaluation.get("shuffle_positions", True))
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "shuffle_positions": shuffle_positions,
        "seed_offset": seed_offset,
    }
    if deterministic_episodes > 0:
        payload["deterministic"] = evaluate_checkpoint(
            checkpoint_path,
            num_episodes=deterministic_episodes,
            seed_offset=seed_offset,
            deterministic=True,
            shuffle_positions=shuffle_positions,
        )
    if sampled_episodes > 0:
        payload["sampled"] = evaluate_checkpoint(
            checkpoint_path,
            num_episodes=sampled_episodes,
            seed_offset=seed_offset + 100_000,
            deterministic=False,
            shuffle_positions=shuffle_positions,
        )
    for index, mode in enumerate(action_modes):
        mode_name = str(mode.get("name", mode["action_mode"]))
        payload[mode_name] = evaluate_checkpoint(
            checkpoint_path,
            num_episodes=int(mode["episodes"]),
            seed_offset=int(mode.get("seed_offset", seed_offset + 200_000 + index * 100_000)),
            action_mode=str(mode["action_mode"]),
            shuffle_positions=shuffle_positions,
        )
    return payload


def _log_sidecar_research_artifacts(
    *,
    plan: Mapping[str, Any],
    paths: Sequence[Path],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    from ant_byte_env.wandb_tracking import WandbTracker

    wandb = dict(plan["wandb"])
    tracker = WandbTracker(
        project=wandb.get("project"),
        entity=wandb.get("entity"),
        group=wandb.get("group"),
        name=f"{wandb.get('name')}-research-ledger" if wandb.get("name") else None,
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
    if evaluation:
        tracker.log_metrics(_flatten_evaluation_metrics(evaluation))
    was_enabled = tracker.enabled
    for path in paths:
        if path.exists():
            tracker.log_artifact(
                f"research-{plan['id']}-{path.stem}",
                path,
                artifact_type="research-plan",
                aliases=[str(plan["id"]), "latest"],
            )
    tracker.finish()
    return {
        "enabled": was_enabled,
        "logged_files": [str(path) for path in paths if was_enabled and path.exists()],
    }


def _flatten_evaluation_metrics(evaluation: Mapping[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for mode, metrics in evaluation.items():
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_key = str(key)
                if clean_key.startswith("eval_"):
                    clean_key = clean_key[len("eval_") :]
                flat[f"evaluation/{mode}/{clean_key}"] = float(value)
    if "score" in evaluation and isinstance(evaluation["score"], (int, float)):
        flat["evaluation/score"] = float(evaluation["score"])
    return flat


def _summary_score(summary: Mapping[str, Any]) -> float:
    evaluation = summary.get("evaluation", {})
    if isinstance(evaluation, Mapping):
        score = _evaluation_score(evaluation)
        if score is not None:
            return score

    target_stage = str(summary.get("target_stage", "25x25"))
    if "curriculum" in summary:
        metrics = _target_stage_metrics(summary, target_stage=target_stage)
    else:
        metrics = {}
    if metrics:
        return _promotion_score(metrics)
    return 0.0


def _evaluation_score(evaluation: Mapping[str, Any]) -> float | None:
    rows = [
        metrics
        for metrics in evaluation.values()
        if isinstance(metrics, Mapping) and "eval_mean_episode_return" in metrics
    ]
    if not rows:
        return None
    mean_return = sum(float(row.get("eval_mean_episode_return", 0.0)) for row in rows) / len(rows)
    mean_delivery = sum(float(row.get("eval_mean_delivered_food", 0.0)) for row in rows) / len(rows)
    mean_fraction = sum(float(row.get("eval_mean_delivered_fraction", 0.0)) for row in rows) / len(rows)
    return mean_return + 0.02 * mean_delivery + 0.5 * mean_fraction


def _extra_evaluation_summary(evaluation: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    extras: dict[str, dict[str, float]] = {}
    for mode, metrics in evaluation.items():
        if mode in {"deterministic", "sampled"} or not isinstance(metrics, Mapping):
            continue
        if "eval_mean_episode_return" not in metrics:
            continue
        extras[str(mode)] = {
            "return": float(metrics.get("eval_mean_episode_return", 0.0)),
            "delivered": float(metrics.get("eval_mean_delivered_food", 0.0)),
            "fraction": float(metrics.get("eval_mean_delivered_fraction", 0.0)),
            "success_rate": float(metrics.get("eval_success_rate", 0.0)),
            "episode_length": float(metrics.get("eval_mean_episode_length", 0.0)),
        }
    return extras


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
