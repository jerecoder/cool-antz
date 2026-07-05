#!/usr/bin/env python3
"""Run a clean W&B size-curriculum probe with retrospective evals."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_REPO_ROOT = Path("/home/jerefigo/Documents/fun/cool-antz")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "exploration_to_forage_50x50.json"
DEFAULT_SOURCE_RELATIVE = Path(
    "runs/notebooks/exploration_to_forage_50x50/checkpoints/model_update_014000.pkl"
)
DEFAULT_WANDB_ENTITY = "jerefigueiredo-universidad-de-san-andr-s"
DEFAULT_WANDB_PROJECT = "cool-antz"


def _csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exploration-to-forage size curriculum in one clean W&B run "
            "and log retrospective evals on previously seen grid sizes."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default="jax_mappo_size_forgetting_probe_mlp")
    parser.add_argument("--stage-sizes", type=str, default=None)
    parser.add_argument(
        "--eval-stage-sizes",
        type=str,
        default=None,
        help="Comma-separated eval sizes. Defaults to the training stage sizes.",
    )
    parser.add_argument("--stage-update-cap", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--max-num-steps",
        type=int,
        default=80,
        help="Cap PPO rollout length per stage for a cheap probe. Use 0 for no cap.",
    )
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-seed-offset", type=int, default=1_700_000)
    parser.add_argument(
        "--eval-action-mode",
        type=str,
        default="greedy_move_greedy_write",
    )
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-group", type=str, default="size_forgetting_probe_mlp")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _resolve_source_checkpoint(requested: Path | None) -> Path:
    candidates: list[Path]
    if requested is not None:
        candidates = [requested]
    else:
        candidates = [
            PROJECT_ROOT / DEFAULT_SOURCE_RELATIVE,
            ORIGINAL_REPO_ROOT / DEFAULT_SOURCE_RELATIVE,
        ]
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if resolved.exists():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"source checkpoint not found; searched: {searched}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _stage_by_size(stages: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(stage["width"]): stage for stage in stages}


def _training_args(
    *,
    experiment_args: Mapping[str, Any],
    num_envs: int,
    seed: int,
    exp_name: str,
) -> dict[str, Any]:
    args = dict(experiment_args)
    args.update(
        {
            "critic_architecture": "mlp",
            "exp_name": exp_name,
            "log_interval": 1,
            "num_envs": int(num_envs),
            "seed": int(seed),
            "quiet": True,
        }
    )
    args.pop("save_best_model", None)
    return args


def _cap_stage_profiles(
    stages: Sequence[Mapping[str, Any]],
    *,
    stage_update_cap: int,
    max_num_steps: int,
) -> list[dict[str, Any]]:
    capped: list[dict[str, Any]] = []
    for stage in stages:
        row = dict(stage)
        row["global_update_cap"] = int(stage_update_cap)
        if max_num_steps > 0 and "num_steps" in row:
            row["num_steps"] = min(int(row["num_steps"]), int(max_num_steps))
        capped.append(row)
    return capped


def _stage_train_args(
    *,
    common_args: Sequence[str],
    stage: Mapping[str, Any],
    stage_update_cap: int,
    checkpoint_path: Path,
    previous_checkpoint: Path,
) -> list[str]:
    update_timesteps = int(stage["num_steps"]) * _argv_int(common_args, "--num-envs")
    train_args = [
        *common_args,
        "--total-timesteps",
        str(update_timesteps * int(stage_update_cap)),
        "--width",
        str(stage["width"]),
        "--height",
        str(stage["height"]),
        "--food-count",
        str(stage["food_count"]),
        "--food-sources",
        str(stage["food_sources"]),
        "--cookie-distance",
        str(stage["cookie_distance"]),
        "--max-steps",
        str(stage["max_steps"]),
        "--num-steps",
        str(stage["num_steps"]),
        "--gamma",
        str(stage["gamma"]),
        "--visit-reward-scale",
        str(stage["visit_reward_scale"]),
        "--save-model",
        str(checkpoint_path),
        "--load-model",
        str(previous_checkpoint),
    ]
    return train_args


def _argv_int(argv: Sequence[str], option: str) -> int:
    index = list(argv).index(option)
    return int(argv[index + 1])


def _evaluate_checkpoint_on_stage(
    checkpoint_path: Path,
    *,
    stage: Mapping[str, Any],
    num_episodes: int,
    seed_offset: int,
    action_mode: str,
) -> dict[str, float]:
    import argparse as _argparse

    from ant_byte_env.training.jax_mappo.evaluation import (
        _checkpoint_args_with_defaults,
        _checkpoint_observation_dims,
        evaluate_params,
    )
    from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training
    from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint

    raw_checkpoint = read_checkpoint(checkpoint_path)
    base_args = _checkpoint_args_with_defaults(raw_checkpoint.get("args", {}))
    values = {
        **vars(base_args),
        "width": int(stage["width"]),
        "height": int(stage["height"]),
        "food_count": int(stage["food_count"]),
        "food_sources": int(stage["food_sources"]),
        "cookie_distance": int(stage["cookie_distance"]),
        "max_steps": int(stage["max_steps"]),
        "visit_reward_scale": float(stage["visit_reward_scale"]),
        "critic_architecture": "mlp",
        "num_envs": 1,
    }
    eval_args = _argparse.Namespace(**values)
    central_obs_dim, actor_obs_dim = _checkpoint_observation_dims(eval_args)
    checkpoint = load_checkpoint_for_training(
        checkpoint_path,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        target_write_bits=eval_args.write_bits,
        actor_vision_radius=eval_args.actor_vision_radius,
        target_num_ants=eval_args.num_ants,
        target_agent_identity_types=getattr(eval_args, "agent_identity_types", None),
        target_critic_architecture="mlp",
    )
    return evaluate_params(
        params=checkpoint["params"],
        args=eval_args,
        num_episodes=int(num_episodes),
        seed_offset=int(seed_offset),
        action_mode=action_mode,
        shuffle_positions=True,
    )


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"size_forgetting_probe_mlp_{_timestamp()}"
    run_dir = PROJECT_ROOT / "runs" / "notebooks" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "stage_metrics.jsonl"
    eval_path = run_dir / "eval_matrix.jsonl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from ant_byte_env import notebook_workflows as workflows
    from ant_byte_env.training.jax_mappo import runner as jax_runner

    experiment = workflows.load_jax_experiment(args.config)
    experiment_args = _training_args(
        experiment_args=experiment.args,
        num_envs=args.num_envs,
        seed=args.seed,
        exp_name=args.exp_name,
    )
    source_checkpoint = _resolve_source_checkpoint(args.source_checkpoint)
    experiment_args["load_model"] = str(source_checkpoint)

    stage_sizes = _csv_ints(args.stage_sizes) or tuple(
        int(size) for size in experiment.metadata["stage_sizes"]
    )
    stages = workflows.build_exploration_to_forage_curriculum_stages(
        experiment_args,
        stage_sizes=stage_sizes,
        visit_reward_schedule=experiment.metadata.get("visit_reward_schedule"),
        stage_update_multiplier=1.0,
    )
    stages = _cap_stage_profiles(
        stages,
        stage_update_cap=args.stage_update_cap,
        max_num_steps=args.max_num_steps,
    )
    stage_lookup = _stage_by_size(stages)
    eval_stage_sizes = _csv_ints(args.eval_stage_sizes) or stage_sizes
    eval_stages = [stage_lookup[int(size)] for size in eval_stage_sizes]
    common_args = workflows.config_common_args(
        experiment_args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )
    update_timesteps = workflows.update_timesteps(
        num_envs=experiment_args["num_envs"],
        num_steps=max(int(stage["num_steps"]) for stage in stages),
    )
    run_plan = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config": str(args.config),
        "source_checkpoint": str(source_checkpoint),
        "critic_architecture": "mlp",
        "actor_vision_radius": int(experiment_args["actor_vision_radius"]),
        "write_bits": int(experiment_args["write_bits"]),
        "stage_update_cap": int(args.stage_update_cap),
        "num_envs": int(args.num_envs),
        "max_num_steps": int(args.max_num_steps),
        "eval_episodes": int(args.eval_episodes),
        "eval_action_mode": str(args.eval_action_mode),
        "stage_sizes": [int(stage["width"]) for stage in stages],
        "eval_stage_sizes": [int(stage["width"]) for stage in eval_stages],
        "stage_training_profiles": [
            {
                "name": str(stage["name"]),
                "width": int(stage["width"]),
                "global_update_cap": int(stage["global_update_cap"]),
                "num_steps": int(stage["num_steps"]),
                "gamma": float(stage["gamma"]),
                "max_steps": int(stage["max_steps"]),
                "food_count": int(stage["food_count"]),
                "food_sources": int(stage["food_sources"]),
                "visit_reward_scale": float(stage["visit_reward_scale"]),
            }
            for stage in stages
        ],
        "common_args": list(common_args),
        "update_timesteps_per_largest_stage": int(update_timesteps),
    }
    _write_json(run_dir / "probe_plan.json", run_plan)

    wandb = None
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb as _wandb

        wandb = _wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            mode=args.wandb_mode,
            dir=str(run_dir),
            tags=[
                "exploration-to-forage",
                "size-curriculum",
                "forgetting-probe",
                "mlp-critic",
                "vision-radius-1",
                "write-bits-1",
            ],
            notes=(
                "Clean probe for the report section on growing map size: one W&B "
                "run owns the whole 8x8-to-50x50 curriculum, using the historical "
                "3x3 actor observation radius, one write bit, and MLP critic. "
                "After each stage, checkpoints are re-evaluated on previously seen "
                "grid sizes to test for forgetting-like degradation."
            ),
            config=run_plan,
            reinit="create_new",
        )
        (run_dir / "wandb_url.txt").write_text(str(wandb_run.url) + "\n", encoding="utf-8")
        (run_dir / "wandb_id.txt").write_text(str(wandb_run.id) + "\n", encoding="utf-8")
        artifact = wandb.Artifact(f"{run_name}-plan", type="research-plan")
        artifact.add_file(str(args.config))
        artifact.add_file(str(run_dir / "probe_plan.json"))
        wandb_run.log_artifact(artifact, aliases=["latest"])

    best_eval_by_size: dict[int, float] = {}
    eval_rows: list[dict[str, Any]] = []
    previous_checkpoint = source_checkpoint
    curriculum_step_base = 0

    try:
        for stage_index, stage in enumerate(stages, start=1):
            stage_name = str(stage["name"])
            stage_update_cap = int(stage["global_update_cap"])
            stage_update_timesteps = int(stage["num_steps"]) * int(experiment_args["num_envs"])
            checkpoint_path = checkpoint_dir / f"{args.exp_name}_{stage_name}.pkl"
            print(f"Training stage {stage_index}/{len(stages)}: {stage_name}")

            def record_progress(
                update_index: int,
                total_updates: int,
                metrics: dict[str, float],
            ) -> None:
                curriculum_step = curriculum_step_base + int(metrics.get("global_step", 0.0))
                row = {
                    **stage,
                    **metrics,
                    "stage_index": stage_index,
                    "stage_name": stage_name,
                    "stage_update": int(update_index),
                    "stage_total_updates": int(total_updates),
                    "global_update_cap": int(stage_update_cap),
                    "stage_update_timesteps": int(stage_update_timesteps),
                    "curriculum_global_step": int(curriculum_step),
                    "checkpoint": str(checkpoint_path),
                }
                _append_jsonl(metrics_path, row)
                if wandb_run is not None:
                    wandb_run.log(row, step=curriculum_step)

            train_args = _stage_train_args(
                common_args=common_args,
                stage=stage,
                stage_update_cap=stage_update_cap,
                checkpoint_path=checkpoint_path,
                previous_checkpoint=previous_checkpoint,
            )
            final_metrics = jax_runner.main(
                train_args,
                progress_callback=record_progress,
            )
            previous_checkpoint = checkpoint_path
            curriculum_step_base += stage_update_timesteps * stage_update_cap

            eval_payload: dict[str, Any] = {
                "stage_index": stage_index,
                "stage_name": stage_name,
                "trained_stage_size": int(stage["width"]),
                "checkpoint": str(checkpoint_path),
                "curriculum_global_step": int(curriculum_step_base),
                "final_train_episode_return": float(final_metrics["episode_return"]),
            }
            for eval_stage in eval_stages:
                eval_size = int(eval_stage["width"])
                if eval_size > int(stage["width"]):
                    continue
                metrics = _evaluate_checkpoint_on_stage(
                    checkpoint_path,
                    stage=eval_stage,
                    num_episodes=args.eval_episodes,
                    seed_offset=args.eval_seed_offset + stage_index * 10_000 + eval_size,
                    action_mode=args.eval_action_mode,
                )
                score = float(metrics["eval_mean_delivered_food_per_1000_ant_steps"])
                previous_best = best_eval_by_size.get(eval_size)
                best_eval_by_size[eval_size] = (
                    score if previous_best is None else max(previous_best, score)
                )
                retention = (
                    None
                    if not best_eval_by_size[eval_size]
                    else score / best_eval_by_size[eval_size]
                )
                eval_row = {
                    **metrics,
                    "stage_index": stage_index,
                    "stage_name": stage_name,
                    "trained_stage_size": int(stage["width"]),
                    "eval_stage_name": str(eval_stage["name"]),
                    "eval_stage_size": eval_size,
                    "eval_episodes": int(args.eval_episodes),
                    "eval_action_mode": str(args.eval_action_mode),
                    "retention_vs_best_so_far": retention,
                    "checkpoint": str(checkpoint_path),
                    "curriculum_global_step": int(curriculum_step_base),
                }
                eval_rows.append(eval_row)
                _append_jsonl(eval_path, eval_row)
                key_prefix = f"eval_on_{eval_stage['name']}"
                eval_payload.update(
                    {
                        f"{key_prefix}/mean_delivered_food": metrics[
                            "eval_mean_delivered_food"
                        ],
                        f"{key_prefix}/mean_delivered_fraction": metrics[
                            "eval_mean_delivered_fraction"
                        ],
                        f"{key_prefix}/delivered_food_per_1000_ant_steps": score,
                        f"{key_prefix}/retention_vs_best_so_far": retention,
                    }
                )
            if wandb_run is not None:
                wandb_run.log(eval_payload, step=curriculum_step_base)

        summary = _summarize_eval_rows(eval_rows)
        _write_json(run_dir / "eval_summary.json", summary)
        if wandb_run is not None and wandb is not None:
            if eval_rows:
                table = wandb.Table(columns=list(eval_rows[0].keys()))
                for row in eval_rows:
                    table.add_data(*(row.get(column) for column in eval_rows[0].keys()))
                wandb_run.log({"eval_matrix": table}, step=curriculum_step_base)
            wandb_run.summary.update(summary)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"Run directory: {run_dir}")
    if (run_dir / "wandb_url.txt").exists():
        print((run_dir / "wandb_url.txt").read_text(encoding="utf-8").strip())


def _summarize_eval_rows(eval_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_eval_size: dict[int, list[Mapping[str, Any]]] = {}
    for row in eval_rows:
        by_eval_size.setdefault(int(row["eval_stage_size"]), []).append(row)
    summary: dict[str, Any] = {}
    for eval_size, rows in sorted(by_eval_size.items()):
        best = max(
            float(row["eval_mean_delivered_food_per_1000_ant_steps"])
            for row in rows
        )
        final_row = max(rows, key=lambda row: int(row["stage_index"]))
        final = float(final_row["eval_mean_delivered_food_per_1000_ant_steps"])
        key = f"eval_{eval_size:02d}x{eval_size:02d}"
        summary[f"{key}_best_delivered_per_1000_ant_steps"] = best
        summary[f"{key}_final_delivered_per_1000_ant_steps"] = final
        summary[f"{key}_final_retention_vs_best"] = None if best <= 0.0 else final / best
    return summary


if __name__ == "__main__":
    main()
