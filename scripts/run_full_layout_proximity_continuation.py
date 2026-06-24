#!/usr/bin/env python3
"""Run a full-layout continuation from the best proximity-source policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = PROJECT_ROOT / "experiments" / "exploration_to_forage_proximity_sources_50x50.json"
SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "notebooks"
    / "exploration_to_forage_proximity_sources_positive_only_50x50_outer_30x30_inner"
    / "checkpoints"
    / "best_proximity_sources_positive_only.pkl"
)
DEFAULT_RUN_NAME = "exploration_to_forage_proximity_sources_full_layout_50x50_from_best"
DEFAULT_EXP_NAME = "jax_mappo_full_layout_proximity_from_best"
DEFAULT_BEST_CHECKPOINT_NAME = "best_full_layout_proximity_from_best.pkl"


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def run_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_dir = PROJECT_ROOT / "runs" / "notebooks" / str(args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "best_checkpoint": checkpoint_dir / str(args.best_checkpoint_name),
    }


def source_checkpoint_path(args: argparse.Namespace) -> Path:
    source = Path(args.source_checkpoint)
    return source if source.is_absolute() else PROJECT_ROOT / source


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    from ant_byte_env import notebook_workflows as workflows

    experiment = workflows.load_jax_experiment(CONFIG_PATH)
    paths = run_paths(args)
    training_args = dict(experiment.args)
    training_args.update(
        {
            "exp_name": args.exp_name,
            "layout_margin": 0,
            "hub_center_window_size": 0,
            "seed": args.seed,
            "save_best_model": str(paths["best_checkpoint"]),
        }
    )
    for key in (
        "num_ants",
        "food_count",
        "max_steps",
        "learning_rate",
        "best_eval_interval",
        "best_eval_episodes",
        "training_rollout_temperature",
        "best_model_metric",
        "best_model_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            training_args[key] = value

    source_counts = _csv_ints(args.source_counts)
    cluster_radii = _csv_ints(args.cluster_radii)
    stages = workflows.build_food_cluster_curriculum_stages(
        training_args,
        source_counts=source_counts,
        cluster_radii=cluster_radii,
        stage_update_multiplier=args.stage_update_multiplier,
    )
    common_args = workflows.config_common_args(
        training_args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )
    update_timesteps = workflows.update_timesteps(
        num_envs=training_args["num_envs"],
        num_steps=training_args["num_steps"],
    )
    return {
        "experiment": experiment,
        "training_args": training_args,
        "stages": stages,
        "common_args": common_args,
        "update_timesteps": update_timesteps,
    }


def verify_first_stage(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from ant_byte_env.notebook_workflows import training_dimensions
    from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
    from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training

    stage = plan["stages"][0]
    train_args = [
        *plan["common_args"],
        "--total-timesteps",
        str(plan["update_timesteps"]),
        "--width",
        str(stage["width"]),
        "--height",
        str(stage["height"]),
        "--food-count",
        str(stage["food_count"]),
        "--food-sources",
        str(stage["food_sources"]),
        "--food-cluster-count",
        str(stage["food_cluster_count"]),
        "--food-cluster-radius",
        str(stage["food_cluster_radius"]),
        "--cookie-distance",
        str(stage["cookie_distance"]),
        "--max-steps",
        str(stage["max_steps"]),
        "--load-model",
        str(source_checkpoint_path(args)),
    ]
    parsed_args, central_obs_dim, actor_obs_dim = training_dimensions(train_args)
    source_checkpoint = source_checkpoint_path(args)
    checkpoint = read_checkpoint(source_checkpoint)
    load_checkpoint_for_training(
        source_checkpoint,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        target_write_bits=parsed_args.write_bits,
        actor_vision_radius=parsed_args.actor_vision_radius,
        target_num_ants=parsed_args.num_ants,
        write_head_transfer=parsed_args.write_head_transfer,
        target_critic_architecture=parsed_args.critic_architecture,
    )
    return {
        "source_checkpoint": str(source_checkpoint),
        "source_central_obs_dim": int(checkpoint["central_obs_dim"]),
        "target_central_obs_dim": central_obs_dim,
        "source_actor_obs_dim": int(checkpoint["actor_obs_dim"]),
        "target_actor_obs_dim": actor_obs_dim,
        "critic_architecture": parsed_args.critic_architecture,
        "write_bits": parsed_args.write_bits,
        "num_ants": parsed_args.num_ants,
        "actor_vision_radius": parsed_args.actor_vision_radius,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--source-checkpoint", default=str(SOURCE_CHECKPOINT))
    parser.add_argument("--best-checkpoint-name", default=DEFAULT_BEST_CHECKPOINT_NAME)
    parser.add_argument("--num-ants", type=int, default=None)
    parser.add_argument("--food-count", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--best-eval-interval", type=int, default=None)
    parser.add_argument("--best-eval-episodes", type=int, default=None)
    parser.add_argument("--best-model-metric", default=None)
    parser.add_argument("--best-model-mode", default=None)
    parser.add_argument("--training-rollout-temperature", type=float, default=None)
    parser.add_argument("--source-counts", default="8,6,4,3,2")
    parser.add_argument("--cluster-radii", default="2,2,1,1,0")
    parser.add_argument("--stage-update-multiplier", type=float, default=2.0)
    parser.add_argument("--checkpoint-video-interval-updates", type=int, default=None)
    parser.add_argument("--checkpoint-video-max-frames", type=int, default=600)
    parser.add_argument("--checkpoint-video-policy-temperature", type=float, default=0.75)
    parser.add_argument("--checkpoint-video-rollout-count", type=int, default=2)
    parser.add_argument("--max-gpu-compute-memory-mb", type=int, default=1024)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="cool-antz")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="exploration-to-forage-proximity-sources")
    args = parser.parse_args()

    from ant_byte_env import notebook_workflows as workflows
    from ant_byte_env.training.jax_mappo import runner as jax_runner

    runtime = workflows.configure_jax_notebook_runtime()
    workflows.assert_notebook_resources_available(
        runtime,
        min_disk_free_gb=3.0,
        max_gpu_compute_memory_mb=args.max_gpu_compute_memory_mb,
    )
    source_checkpoint = source_checkpoint_path(args)
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_checkpoint}")

    plan = build_plan(args)
    paths = run_paths(args)
    verification = verify_first_stage(plan, args)
    launch_manifest = {
        "run_name": args.run_name,
        "run_dir": str(paths["run_dir"]),
        "config": str(CONFIG_PATH),
        "source_checkpoint": str(source_checkpoint),
        "best_checkpoint": str(paths["best_checkpoint"]),
        "runtime": runtime,
        "verification": verification,
        "stage_update_multiplier": args.stage_update_multiplier,
        "checkpoint_video_interval_updates": args.checkpoint_video_interval_updates,
        "checkpoint_video_max_frames": args.checkpoint_video_max_frames,
        "checkpoint_video_policy_temperature": args.checkpoint_video_policy_temperature,
        "checkpoint_video_rollout_count": args.checkpoint_video_rollout_count,
        "stage_count": len(plan["stages"]),
        "stages": [
            {
                "name": stage["name"],
                "food_sources": stage["food_sources"],
                "food_cluster_radius": stage["food_cluster_radius"],
                "global_update_cap": stage["global_update_cap"],
                "num_steps": stage["num_steps"],
            }
            for stage in plan["stages"]
        ],
    }
    print(json.dumps(launch_manifest, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    (paths["run_dir"] / "launch_manifest.json").write_text(
        json.dumps(launch_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    training_result = workflows.run_forage_curriculum(
        stages=plan["stages"],
        checkpoint_dir=paths["checkpoint_dir"],
        common_args=plan["common_args"],
        update_timesteps_per_stage=plan["update_timesteps"],
        global_update_cap=plan["stages"][0]["global_update_cap"],
        train_main=jax_runner.main,
        initial_checkpoint=source_checkpoint,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.run_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=[
            "exploration-to-forage",
            "proximity-source-curriculum",
            "full-layout-randomization",
            "warm-start",
            "50x50",
            f"{plan['training_args']['num_ants']}-ants",
        ],
        wandb_notes=(
            "Warm-start from the best 30x30-inner proximity-source policy, then "
            "continue on unrestricted 50x50 random hub and food layouts. This "
            "launcher can run either a short clustered source curriculum or a "
            "direct final-stage continuation, depending on source-counts."
        ),
        wandb_artifact_paths=[CONFIG_PATH],
        wandb_artifact_prefix="exploration-to-forage-full-layout-proximity",
        checkpoint_name_prefix=args.exp_name,
        wandb_video_key_prefix="videos/exploration_to_forage/full_layout_proximity",
        wandb_video_max_frames=600,
        wandb_video_stage_names=[stage["name"] for stage in plan["stages"]],
        wandb_video_policy_temperature=0.75,
        wandb_video_rollout_count=2,
        checkpoint_video_interval_updates=args.checkpoint_video_interval_updates,
        checkpoint_video_max_frames=args.checkpoint_video_max_frames,
        checkpoint_video_policy_temperature=args.checkpoint_video_policy_temperature,
        checkpoint_video_rollout_count=args.checkpoint_video_rollout_count,
        checkpoint_video_wandb_key_prefix=(
            "videos/exploration_to_forage/full_layout_proximity/checkpoints"
        ),
    )
    (paths["run_dir"] / "training_result.json").write_text(
        json.dumps(training_result, indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
