#!/usr/bin/env python3
"""Train reward-shaping families and render comparable rollout videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_SOURCE_CHECKPOINT = PROJECT_ROOT / (
    "runs/notebooks/size_forgetting_probe_mlp_no_shaping_matched_20260705_clean/"
    "checkpoints/jax_mappo_size_forgetting_probe_mlp_no_shaping_matched_50x50.pkl"
)
DEFAULT_CONFIG = PROJECT_ROOT / "experiments/exploration_to_forage_50x50_no_shaping_probe.json"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs/notebooks/reward_shaping_family_videos_mlp_20260705"
DEFAULT_WANDB_ENTITY = "jerefigueiredo-universidad-de-san-andr-s"
DEFAULT_WANDB_PROJECT = "cool-antz"


FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "name": "baseline",
        "label": "sin shaping",
        "purpose": "continuacion con recompensa pura de entrega",
        "overrides": {},
    },
    {
        "name": "exploration",
        "label": "exploracion",
        "purpose": "premiar cobertura de celdas nuevas para romper orbitas locales",
        "overrides": {
            "visit_reward_scale": 0.01,
            "visit_reward_decay": 1.0,
        },
    },
    {
        "name": "forage_return",
        "label": "forraje/retorno",
        "purpose": "separar encontrar comida de volver cargando al hub",
        "overrides": {
            "pickup_bonus": 0.05,
            "carrying_hub_distance_bonus": 0.08,
        },
    },
    {
        "name": "byte_trails",
        "label": "escritura/trazas",
        "purpose": "premiar marcas y seguimiento de rastros sin cambiar observaciones locales",
        "overrides": {
            "delivery_byte_trail_bonus": 0.20,
            "delivery_byte_trail_target_tiles": 12.0,
            "byte_follow_bonus": 0.005,
            "carrying_byte_write_bonus": 0.004,
            "write_bit_entropy_bonus": 0.0008,
        },
    },
)


REWARD_CLI_FLAGS: tuple[tuple[str, str], ...] = (
    ("pickup_bonus", "--pickup-bonus"),
    ("distance_bonus", "--distance-bonus"),
    ("carrying_hub_distance_bonus", "--carrying-hub-distance-bonus"),
    ("visit_reward_scale", "--visit-reward-scale"),
    ("visit_reward_decay", "--visit-reward-decay"),
    ("view_reward_scale", "--view-reward-scale"),
    ("view_reward_decay", "--view-reward-decay"),
    ("delivery_byte_trail_bonus", "--delivery-byte-trail-bonus"),
    ("delivery_byte_trail_target_tiles", "--delivery-byte-trail-target-tiles"),
    ("byte_follow_bonus", "--byte-follow-bonus"),
    ("carrying_byte_write_bonus", "--carrying-byte-write-bonus"),
    ("write_bit_entropy_bonus", "--write-bit-entropy-bonus"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the final 50x50 MLP checkpoint under logical reward-shaping "
            "families, then render comparable videos."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--update-cap", type=int, default=80)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=160)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-seed-offset", type=int, default=2_800_000)
    parser.add_argument("--eval-action-mode", type=str, default="sampled_move_greedy_write")
    parser.add_argument("--eval-move-temperature", type=float, default=0.525)
    parser.add_argument("--render-seed-offset", type=int, default=3_100_000)
    parser.add_argument("--render-max-frames", type=int, default=900)
    parser.add_argument("--render-tile-size", type=int, default=8)
    parser.add_argument("--render-style", type=str, default=None)
    parser.add_argument("--write-bits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-group", type=str, default="reward_shaping_families")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Optional subset of family names to run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing checkpoints/videos when present.",
    )
    return parser.parse_args()


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


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_final_stage(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from ant_byte_env import notebook_workflows as workflows

    experiment = workflows.load_jax_experiment(config_path)
    args = dict(experiment.args)
    stage_sizes = tuple(int(size) for size in experiment.metadata["stage_sizes"])
    stages = workflows.build_exploration_to_forage_curriculum_stages(
        args,
        stage_sizes=stage_sizes,
        visit_reward_schedule=experiment.metadata.get("visit_reward_schedule"),
        stage_update_multiplier=1.0,
    )
    return args, dict(stages[-1])


def _selected_families(names: list[str] | None) -> tuple[dict[str, Any], ...]:
    if names is None:
        return FAMILIES
    available = {family["name"]: family for family in FAMILIES}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"unknown families: {', '.join(missing)}")
    return tuple(available[name] for name in names)


def _base_training_args(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment_args, final_stage = _load_final_stage(args.config)
    experiment_args.update(
        {
            "critic_architecture": "mlp",
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "seed": int(args.seed),
            "quiet": False,
            "log_interval": int(args.log_interval),
            "write_bits": int(args.write_bits),
            "write_head_transfer": "neutral-new",
            "wandb_project": None,
        }
    )
    for key in (
        "pickup_bonus",
        "distance_bonus",
        "carrying_hub_distance_bonus",
        "visit_reward_scale",
        "view_reward_scale",
        "delivery_byte_trail_bonus",
        "byte_follow_bonus",
        "carrying_byte_write_bonus",
        "write_bit_entropy_bonus",
    ):
        experiment_args[key] = 0.0
    experiment_args["delivery_byte_trail_target_tiles"] = 12.0
    experiment_args.pop("save_best_model", None)
    experiment_args.pop("load_model", None)
    final_stage["num_steps"] = int(args.num_steps)
    final_stage["global_update_cap"] = int(args.update_cap)
    return experiment_args, final_stage


def _train_family(
    *,
    family: Mapping[str, Any],
    base_args: Mapping[str, Any],
    stage: Mapping[str, Any],
    args: argparse.Namespace,
    family_dir: Path,
) -> tuple[dict[str, float], Path]:
    from ant_byte_env import notebook_workflows as workflows
    from ant_byte_env.training.jax_mappo import runner as jax_runner

    family_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = family_dir / f"{family['name']}.pkl"
    if args.skip_existing and checkpoint_path.exists():
        return {}, checkpoint_path

    combo_args = dict(base_args)
    combo_args.update(family.get("overrides", {}))
    combo_args.update(
        {
            "exp_name": f"jax_mappo_reward_shaping_family_{family['name']}",
            "wandb_project": args.wandb_project if args.wandb_mode != "disabled" else None,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "wandb_mode": args.wandb_mode,
            "wandb_run_name": f"{args.run_dir.name}_{family['name']}",
            "wandb_tags": [
                "reward-shaping",
                "family-comparison",
                "mlp-critic",
                "50x50",
                str(family["name"]),
            ],
            "wandb_notes": (
                f"Reward-shaping family comparison: {family['label']}. "
                f"Purpose: {family['purpose']}."
            ),
        }
    )
    common_args = workflows.config_common_args(
        combo_args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )
    reward_args = [
        item
        for key, flag in REWARD_CLI_FLAGS
        for item in (flag, str(combo_args.get(key, 0.0)))
    ]
    total_timesteps = int(args.update_cap) * int(args.num_envs) * int(args.num_steps)
    train_args = [
        *common_args,
        *reward_args,
        "--total-timesteps",
        str(total_timesteps),
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
        "--save-model",
        str(checkpoint_path),
        "--run-dir",
        str(family_dir),
        "--load-model",
        str(args.source_checkpoint),
        "--reset-optimizer-on-load",
    ]
    final_metrics = jax_runner.main(train_args)
    return final_metrics, checkpoint_path


def _evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    stage: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    import argparse as _argparse

    from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
    from ant_byte_env.training.jax_mappo.evaluation import (
        _checkpoint_args_with_defaults,
        _checkpoint_observation_dims,
        evaluate_params,
    )
    from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training

    raw_checkpoint = read_checkpoint(checkpoint_path)
    checkpoint_args = _checkpoint_args_with_defaults(raw_checkpoint.get("args", {}))
    values = {
        **vars(checkpoint_args),
        "width": int(stage["width"]),
        "height": int(stage["height"]),
        "food_count": int(stage["food_count"]),
        "food_sources": int(stage["food_sources"]),
        "cookie_distance": int(stage["cookie_distance"]),
        "max_steps": int(stage["max_steps"]),
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
        num_episodes=int(args.eval_episodes),
        seed_offset=int(args.eval_seed_offset),
        action_mode=args.eval_action_mode,
        move_temperature=float(args.eval_move_temperature),
        shuffle_positions=True,
    )


def _render_video(
    checkpoint_path: Path,
    output_path: Path,
    *,
    args: argparse.Namespace,
) -> Path:
    if args.skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    from ant_byte_env.rendering import render_checkpoint

    return render_checkpoint(
        checkpoint_path,
        output_path,
        backend="jax",
        seed_offset=int(args.render_seed_offset),
        show_vision=True,
        reuse_existing=False,
        max_frames=int(args.render_max_frames),
        tile_size=int(args.render_tile_size),
        action_mode=args.eval_action_mode,
        move_temperature=float(args.eval_move_temperature),
        write_temperature=1.0,
        render_style=args.render_style,
    )


def _log_summary_to_wandb(
    *,
    args: argparse.Namespace,
    rows: list[Mapping[str, Any]],
    video_paths: Mapping[str, Path],
) -> str | None:
    if args.wandb_mode == "disabled":
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=f"{args.run_dir.name}_summary",
        mode=args.wandb_mode,
        dir=str(args.run_dir),
        tags=["reward-shaping", "family-comparison", "videos", "summary"],
        config={
            "families": [row["family"] for row in rows],
            "update_cap": int(args.update_cap),
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "eval_episodes": int(args.eval_episodes),
            "eval_action_mode": args.eval_action_mode,
            "eval_move_temperature": float(args.eval_move_temperature),
            "render_max_frames": int(args.render_max_frames),
        },
        reinit="create_new",
    )
    try:
        table = wandb.Table(columns=list(rows[0].keys()))
        for row in rows:
            table.add_data(*(row.get(column) for column in rows[0].keys()))
        payload: dict[str, Any] = {"family_results": table}
        for name, path in video_paths.items():
            payload[f"policy_videos/{name}"] = wandb.Video(str(path), fps=8, format="mp4")
        run.log(payload)
        run.summary.update(
            {
                "best_family": max(
                    rows,
                    key=lambda row: float(row["eval_mean_delivered_food"]),
                )["family"],
                "family_count": len(rows),
            }
        )
        return str(run.url)
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    if not args.source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint not found: {args.source_checkpoint}")

    selected = _selected_families(args.families)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    media_dir = args.run_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    base_args, stage = _base_training_args(args)
    plan = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_checkpoint": str(args.source_checkpoint),
        "config": str(args.config),
        "stage": stage,
        "families": selected,
        "update_cap": int(args.update_cap),
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "write_bits": int(args.write_bits),
        "eval_episodes": int(args.eval_episodes),
        "eval_action_mode": args.eval_action_mode,
        "eval_move_temperature": float(args.eval_move_temperature),
        "render_max_frames": int(args.render_max_frames),
        "render_tile_size": int(args.render_tile_size),
    }
    _write_json(args.run_dir / "family_plan.json", plan)

    rows: list[dict[str, Any]] = []
    videos: dict[str, Path] = {}
    for family in selected:
        family_dir = args.run_dir / str(family["name"])
        print(f"=== family: {family['name']} ({family['label']}) ===", flush=True)
        train_metrics, checkpoint_path = _train_family(
            family=family,
            base_args=base_args,
            stage=stage,
            args=args,
            family_dir=family_dir,
        )
        print(f"evaluating {family['name']}", flush=True)
        eval_metrics = _evaluate_checkpoint(checkpoint_path, stage=stage, args=args)
        video_path = media_dir / f"reward-shaping-{family['name']}.mp4"
        print(f"rendering {video_path}", flush=True)
        _render_video(checkpoint_path, video_path, args=args)
        videos[str(family["name"])] = video_path
        row = {
            "family": family["name"],
            "label": family["label"],
            "purpose": family["purpose"],
            "checkpoint": str(checkpoint_path),
            "video": str(video_path),
            "update_cap": int(args.update_cap),
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "write_bits": int(args.write_bits),
            "train_episode_return": float(train_metrics.get("episode_return", 0.0)),
            "train_delivery_events": float(train_metrics.get("delivery_events", 0.0)),
            "train_pickup_events": float(train_metrics.get("pickup_events", 0.0)),
            "train_write_action_nonzero_rate": float(
                train_metrics.get("write_action_nonzero_rate", 0.0)
            ),
            **eval_metrics,
        }
        rows.append(row)
        _write_json(args.run_dir / "family_results.json", {"rows": rows})
        _write_csv(args.run_dir / "family_results.csv", rows)

    summary_url = _log_summary_to_wandb(args=args, rows=rows, video_paths=videos)
    if summary_url is not None:
        (args.run_dir / "wandb_summary_url.txt").write_text(summary_url + "\n", encoding="utf-8")

    print(f"results: {args.run_dir / 'family_results.csv'}")
    if summary_url:
        print(summary_url)


if __name__ == "__main__":
    main()
