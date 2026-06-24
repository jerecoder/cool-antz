"""Checkpoint evaluation and artifact logging for archived research runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ant_byte_env.research.scoring import flatten_evaluation_metrics


def evaluate_research_plan_checkpoint(
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
        eval_kwargs: dict[str, Any] = {
            "num_episodes": int(mode["episodes"]),
            "seed_offset": int(mode.get("seed_offset", seed_offset + 200_000 + index * 100_000)),
            "action_mode": str(mode["action_mode"]),
            "shuffle_positions": shuffle_positions,
        }
        for key in ("move_temperature", "write_temperature"):
            if mode.get(key) is not None:
                eval_kwargs[key] = float(mode[key])
        payload[mode_name] = evaluate_checkpoint(
            checkpoint_path,
            **eval_kwargs,
        )
    return payload


def log_sidecar_research_artifacts(
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
        tracker.log_metrics(flatten_evaluation_metrics(evaluation))
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


__all__ = [
    "evaluate_research_plan_checkpoint",
    "log_sidecar_research_artifacts",
]
