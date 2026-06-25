#!/usr/bin/env python3
"""Run the 8-ant half-food full-layout continuation headlessly."""

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
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exploration_to_forage_full_layout_8ants_half_food_50x50.json"
)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _run_paths(experiment_name: str) -> dict[str, Path]:
    run_dir = PROJECT_ROOT / "runs" / "notebooks" / experiment_name
    return {
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "media_dir": run_dir / "media",
    }


def build_plan() -> dict[str, Any]:
    from ant_byte_env import notebook_workflows as workflows

    experiment = workflows.load_jax_experiment(CONFIG_PATH)
    experiment_args = dict(experiment.args)
    source_checkpoint = _resolve_project_path(experiment_args["load_model"])
    best_checkpoint = _resolve_project_path(experiment_args["save_best_model"])
    experiment_args["load_model"] = str(source_checkpoint)
    experiment_args["save_best_model"] = str(best_checkpoint)

    stages = workflows.build_food_cluster_curriculum_stages(
        experiment_args,
        source_counts=tuple(
            int(count) for count in experiment.metadata["food_source_counts"]
        ),
        cluster_radii=tuple(
            int(radius) for radius in experiment.metadata["food_cluster_radii"]
        ),
        visit_reward_schedule=experiment.metadata.get("visit_reward_schedule"),
        view_reward_schedule=experiment.metadata.get("view_reward_schedule"),
        stage_update_multiplier=float(
            experiment.metadata.get("stage_update_multiplier", 1.0)
        ),
    )
    common_args = workflows.config_common_args(
        experiment_args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )
    return {
        "experiment": experiment,
        "experiment_args": experiment_args,
        "source_checkpoint": source_checkpoint,
        "best_checkpoint": best_checkpoint,
        "paths": _run_paths(experiment.name),
        "stages": stages,
        "global_update_cap": max(int(stage["global_update_cap"]) for stage in stages),
        "update_timesteps": workflows.update_timesteps(
            num_envs=int(experiment_args["num_envs"]),
            num_steps=int(experiment_args["num_steps"]),
        ),
        "common_args": common_args,
    }


def verify_first_stage(plan: dict[str, Any]) -> dict[str, Any]:
    from ant_byte_env.notebook_workflows import training_dimensions
    from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
    from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training

    source_checkpoint = Path(plan["source_checkpoint"])
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_checkpoint}")

    stage = plan["stages"][0]
    train_args = [
        *plan["common_args"],
        "--total-timesteps",
        str(int(plan["update_timesteps"])),
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
        "--load-model",
        str(source_checkpoint),
    ]
    for key, option in (
        ("num_steps", "--num-steps"),
        ("gamma", "--gamma"),
        ("food_cluster_count", "--food-cluster-count"),
        ("food_cluster_radius", "--food-cluster-radius"),
        ("random_ant_spawn_radius", "--random-ant-spawn-radius"),
    ):
        if key in stage:
            train_args.extend([option, str(stage[key])])

    parsed_args, central_obs_dim, actor_obs_dim = training_dimensions(train_args)
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
        "target_central_obs_dim": int(central_obs_dim),
        "source_actor_obs_dim": int(checkpoint["actor_obs_dim"]),
        "target_actor_obs_dim": int(actor_obs_dim),
        "critic_architecture": parsed_args.critic_architecture,
        "write_bits": int(parsed_args.write_bits),
        "num_ants": int(parsed_args.num_ants),
        "actor_vision_radius": int(parsed_args.actor_vision_radius),
    }


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-project", default="cool-antz")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--memory-fraction", default="0.35")
    parser.add_argument("--max-gpu-compute-memory-mb", type=int, default=1024)
    args = parser.parse_args()

    from ant_byte_env import notebook_workflows as workflows

    runtime = workflows.configure_jax_notebook_runtime(memory_fraction=args.memory_fraction)
    workflows.assert_notebook_resources_available(
        runtime,
        min_disk_free_gb=3.0,
        max_gpu_compute_memory_mb=args.max_gpu_compute_memory_mb,
    )

    import jax
    from ant_byte_env.training.jax_mappo import runner as jax_runner

    plan = build_plan()
    verification = verify_first_stage(plan)
    experiment = plan["experiment"]
    experiment_args = plan["experiment_args"]
    paths = plan["paths"]
    metadata = experiment.metadata
    critic_tag = (
        f"{experiment_args.get('critic_architecture', 'mlp').replace('_', '-')}-critic"
    )
    manifest = {
        "config": str(CONFIG_PATH),
        "experiment": experiment.name,
        "run_dir": str(paths["run_dir"]),
        "source_checkpoint": str(plan["source_checkpoint"]),
        "best_checkpoint": str(plan["best_checkpoint"]),
        "runtime": runtime,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "verification": verification,
        "global_update_cap": int(plan["global_update_cap"]),
        "update_timesteps": int(plan["update_timesteps"]),
        "total_stage_timesteps": int(plan["global_update_cap"])
        * int(plan["update_timesteps"]),
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "stages": [
            {
                "name": stage["name"],
                "food_sources": stage["food_sources"],
                "food_cluster_radius": stage["food_cluster_radius"],
                "food_count": stage["food_count"],
                "global_update_cap": stage["global_update_cap"],
                "num_steps": stage["num_steps"],
                "gamma": stage["gamma"],
            }
            for stage in plan["stages"]
        ],
    }
    print(json.dumps(manifest, indent=2, default=_json_default, sort_keys=True))
    if args.dry_run:
        return 0

    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    paths["media_dir"].mkdir(parents=True, exist_ok=True)
    (paths["run_dir"] / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default, sort_keys=True),
        encoding="utf-8",
    )
    training_result = workflows.run_forage_curriculum(
        stages=plan["stages"],
        checkpoint_dir=paths["checkpoint_dir"],
        common_args=plan["common_args"],
        update_timesteps_per_stage=plan["update_timesteps"],
        global_update_cap=plan["global_update_cap"],
        train_main=jax_runner.main,
        initial_checkpoint=plan["source_checkpoint"],
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=experiment.name,
        wandb_run_name=experiment.name,
        wandb_mode=args.wandb_mode,
        wandb_tags=[
            "exploration-to-forage",
            "full-layout-randomization",
            "warm-start",
            "8-ants",
            "half-food",
            "two-sources",
            "moving-writes",
            critic_tag,
            "50x50",
        ],
        wandb_notes=metadata["notes"],
        wandb_artifact_paths=[CONFIG_PATH],
        wandb_artifact_prefix="exploration-to-forage-full-layout-8ants-half-food",
        checkpoint_name_prefix=experiment_args["exp_name"],
        wandb_video_key_prefix="videos/exploration_to_forage/full_layout_8ants_half_food",
        wandb_video_max_frames=int(metadata["wandb_video_max_frames"]),
        wandb_video_stage_names=tuple(metadata["wandb_preview_stage_names"]),
        wandb_video_policy_temperature=workflows.notebook_rollout_policy_temperature(
            metadata
        ),
        wandb_video_rollout_count=int(metadata.get("wandb_preview_rollout_count", 1)),
        checkpoint_video_interval_updates=int(metadata["checkpoint_video_interval_updates"]),
        checkpoint_video_max_frames=int(metadata["checkpoint_video_max_frames"]),
        checkpoint_video_policy_temperature=workflows.notebook_rollout_policy_temperature(
            metadata,
            key="checkpoint_video_policy_temperature",
        ),
        checkpoint_video_rollout_count=int(
            metadata.get("checkpoint_video_rollout_count", 1)
        ),
        checkpoint_video_wandb_key_prefix=(
            "videos/exploration_to_forage/full_layout_8ants_half_food/checkpoints"
        ),
    )
    (paths["run_dir"] / "training_result.json").write_text(
        json.dumps(training_result, indent=2, default=_json_default, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(training_result, indent=2, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
