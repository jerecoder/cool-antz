#!/usr/bin/env python3
"""Overnight efficiency sweep for the 50x50 / 60-ant MAPPO policy.

The parent process intentionally avoids importing JAX. Each train/eval/render
job runs in a fresh child process so GPU memory is returned between candidates.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = sys.executable
BASE_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_stabilize_from_60best.json"
)
BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "notebooks"
    / "fl50_60ants_half_food_sw_wc_8bits_stabilize_from_60best"
    / "checkpoints"
    / "best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits_stabilized.pkl"
)
WANDB_ENTITY = "jerefigueiredo-universidad-de-san-andr-s"
WANDB_PROJECT = "cool-antz"
PRIMARY_METRIC = "eval_mean_delivered_food_per_1000_ant_steps"
PROMOTION_MARGIN = 1.05
MIN_DELIVERED_FRACTION = 0.97
MIN_SUCCESS_RATE = 0.65
FIFTY_BY_FIFTY_BASE_UPDATES = 8000


VARIANTS: list[dict[str, Any]] = [
    {
        "stem": "lr5e6_clip03_temp075_u20",
        "learning_rate": 0.000005,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.75,
        "updates": 20,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr1e5_clip03_temp075_u20",
        "learning_rate": 0.00001,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.75,
        "updates": 20,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr5e6_clip05_temp075_u40",
        "learning_rate": 0.000005,
        "clip_coef": 0.05,
        "training_rollout_temperature": 0.75,
        "updates": 40,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr1e5_clip05_temp075_u40",
        "learning_rate": 0.00001,
        "clip_coef": 0.05,
        "training_rollout_temperature": 0.75,
        "updates": 40,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr2e5_clip03_temp075_u20",
        "learning_rate": 0.00002,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.75,
        "updates": 20,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr5e6_clip03_temp065_u40",
        "learning_rate": 0.000005,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.65,
        "updates": 40,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr1e5_clip03_temp065_u40",
        "learning_rate": 0.00001,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.65,
        "updates": 40,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr5e6_clip05_temp09_u40",
        "learning_rate": 0.000005,
        "clip_coef": 0.05,
        "training_rollout_temperature": 0.90,
        "updates": 40,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr1e5_clip08_temp075_u40",
        "learning_rate": 0.00001,
        "clip_coef": 0.08,
        "training_rollout_temperature": 0.75,
        "updates": 40,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr5e6_clip03_temp075_noanneal_u40",
        "learning_rate": 0.000005,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.75,
        "updates": 40,
        "anneal_lr": False,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr1e5_clip03_temp075_noanneal_u40",
        "learning_rate": 0.00001,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.75,
        "updates": 40,
        "anneal_lr": False,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr5e6_clip03_temp075_u60_eval16",
        "learning_rate": 0.000005,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.75,
        "updates": 60,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 16,
    },
    {
        "stem": "lr1e5_clip03_temp075_u60_eval16",
        "learning_rate": 0.00001,
        "clip_coef": 0.03,
        "training_rollout_temperature": 0.75,
        "updates": 60,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 1,
        "best_eval_episodes": 16,
    },
    {
        "stem": "lr5e6_clip02_temp075_u40",
        "learning_rate": 0.000005,
        "clip_coef": 0.02,
        "training_rollout_temperature": 0.75,
        "updates": 40,
        "anneal_lr": True,
        "max_grad_norm": 0.15,
        "update_epochs": 1,
        "best_eval_episodes": 8,
    },
    {
        "stem": "lr1e5_clip05_temp075_epoch2_u20",
        "learning_rate": 0.00001,
        "clip_coef": 0.05,
        "training_rollout_temperature": 0.75,
        "updates": 20,
        "anneal_lr": True,
        "max_grad_norm": 0.25,
        "update_epochs": 2,
        "best_eval_episodes": 8,
    },
]
SEEDS = (13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)
EVAL_KEYS = (
    PRIMARY_METRIC,
    "eval_mean_ant_steps_per_delivered_food",
    "eval_mean_steps_per_delivered_food",
    "eval_mean_episode_length",
    "eval_mean_delivered_food",
    "eval_mean_delivered_fraction",
    "eval_success_rate",
    "eval_mean_episode_return",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("train", "eval", "render"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed-offset", type=int, default=2_000_000)
    parser.add_argument("--eval-action-mode", default="sampled_move_greedy_write")
    parser.add_argument("--eval-move-temperature", type=float, default=0.75)
    parser.add_argument("--eval-write-temperature", type=float, default=1.0)
    parser.add_argument("--eval-cleanup-move-temperature", type=float)
    parser.add_argument("--eval-cleanup-fraction-threshold", type=float, default=0.95)
    parser.add_argument("--eval-stall-cleanup-move-temperature", type=float)
    parser.add_argument("--eval-stall-cleanup-fraction-threshold", type=float, default=0.90)
    parser.add_argument("--eval-stall-cleanup-patience-steps", type=int, default=100)
    parser.add_argument("--render-policy-temperature", type=float, default=0.75)
    parser.add_argument("--render-action-mode", default="sampled_move_greedy_write")
    parser.add_argument("--render-move-temperature", type=float, default=0.75)
    parser.add_argument("--render-write-temperature", type=float, default=1.0)
    parser.add_argument("--render-seed-offset", type=int, default=3_000_000)
    parser.add_argument("--render-max-frames", type=int, default=1200)
    parser.add_argument("--render-tile-size", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--keep-top-k", type=int, default=3)
    parser.add_argument("--min-disk-free-gb", type=float, default=2.0)
    parser.add_argument("--train-timeout-seconds", type=int, default=2400)
    parser.add_argument("--eval-timeout-seconds", type=int, default=900)
    parser.add_argument("--mem-fraction", default="0.32")
    args = parser.parse_args()

    if args.worker == "train":
        if args.config is None or args.result_json is None:
            parser.error("--worker train requires --config and --result-json")
        worker_train(args.config, args.result_json)
    elif args.worker == "eval":
        if args.checkpoint is None or args.result_json is None:
            parser.error("--worker eval requires --checkpoint and --result-json")
        worker_eval(
            args.checkpoint,
            args.result_json,
            episodes=args.eval_episodes,
            seed_offset=args.eval_seed_offset,
            action_mode=args.eval_action_mode,
            move_temperature=args.eval_move_temperature,
            write_temperature=args.eval_write_temperature,
            cleanup_move_temperature=args.eval_cleanup_move_temperature,
            cleanup_fraction_threshold=args.eval_cleanup_fraction_threshold,
            stall_cleanup_move_temperature=args.eval_stall_cleanup_move_temperature,
            stall_cleanup_fraction_threshold=args.eval_stall_cleanup_fraction_threshold,
            stall_cleanup_patience_steps=args.eval_stall_cleanup_patience_steps,
        )
    elif args.worker == "render":
        if args.checkpoint is None or args.output is None or args.result_json is None:
            parser.error("--worker render requires --checkpoint, --output, --result-json")
        worker_render(
            args.checkpoint,
            args.output,
            args.result_json,
            policy_temperature=args.render_policy_temperature,
            action_mode=args.render_action_mode,
            move_temperature=args.render_move_temperature,
            write_temperature=args.render_write_temperature,
            seed_offset=args.render_seed_offset,
            max_frames=args.render_max_frames,
            tile_size=args.render_tile_size,
        )
    else:
        parent_loop(args)


def parent_loop(args: argparse.Namespace) -> None:
    python = os.environ.get("ANTZ_PYTHON", DEFAULT_PYTHON)
    sweep_dir = args.sweep_dir or default_sweep_dir()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "configs").mkdir(exist_ok=True)
    (sweep_dir / "logs").mkdir(exist_ok=True)
    (sweep_dir / "evals").mkdir(exist_ok=True)
    (sweep_dir / "renders").mkdir(exist_ok=True)

    stop_file = sweep_dir / "STOP"
    summary_path = sweep_dir / "summary.jsonl"
    leaderboard_path = sweep_dir / "leaderboard.md"
    state_path = sweep_dir / "state.json"

    print(f"[sweep] dir={sweep_dir}", flush=True)
    print(f"[sweep] stop_file={stop_file}", flush=True)
    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(BASELINE_CHECKPOINT)
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)

    baseline = ensure_eval(
        label="baseline_stabilized_best",
        checkpoint=BASELINE_CHECKPOINT,
        eval_dir=sweep_dir / "evals",
        log_dir=sweep_dir / "logs",
        python=python,
        timeout_seconds=args.eval_timeout_seconds,
        mem_fraction=args.mem_fraction,
    )
    baseline_primary = float(baseline[PRIMARY_METRIC])
    write_json(
        state_path,
        {
            "sweep_dir": str(sweep_dir),
            "baseline_checkpoint": str(BASELINE_CHECKPOINT),
            "baseline_eval": baseline,
            "primary_metric": PRIMARY_METRIC,
            "promotion_threshold": baseline_primary * PROMOTION_MARGIN,
            "min_delivered_fraction": MIN_DELIVERED_FRACTION,
            "min_success_rate": MIN_SUCCESS_RATE,
            "started_at": now_iso(),
        },
    )
    print(
        "[baseline] "
        f"{PRIMARY_METRIC}={baseline_primary:.6g} "
        f"threshold={baseline_primary * PROMOTION_MARGIN:.6g}",
        flush=True,
    )

    start_index = next_candidate_index(summary_path)
    if start_index:
        print(f"[resume] starting at candidate index {start_index}", flush=True)

    candidate_index = start_index
    while args.max_candidates <= 0 or candidate_index < start_index + args.max_candidates:
        if stop_file.exists():
            print("[stop] STOP file found; exiting loop cleanly.", flush=True)
            break
        prepare_disk_space(PROJECT_ROOT, args.min_disk_free_gb)

        candidate = candidate_for_index(candidate_index, sweep_dir)
        config_path = write_candidate_config(candidate, sweep_dir)
        train_result_path = candidate["run_dir"] / "train_result.json"
        eval_result_path = sweep_dir / "evals" / f"{candidate['name']}.json"
        train_log = sweep_dir / "logs" / f"{candidate['name']}.train.log"
        eval_log = sweep_dir / "logs" / f"{candidate['name']}.eval.log"

        print(
            "[candidate] "
            f"idx={candidate_index} name={candidate['name']} "
            f"lr={candidate['learning_rate']} clip={candidate['clip_coef']} "
            f"temp={candidate['training_rollout_temperature']} "
            f"updates={candidate['updates']} seed={candidate['seed']}",
            flush=True,
        )

        record: dict[str, Any] = {
            "index": candidate_index,
            "name": candidate["name"],
            "variant": candidate["variant_stem"],
            "config": str(config_path),
            "run_dir": str(candidate["run_dir"]),
            "started_at": now_iso(),
            "candidate": {
                key: candidate[key]
                for key in (
                    "learning_rate",
                    "clip_coef",
                    "training_rollout_temperature",
                    "updates",
                    "anneal_lr",
                    "max_grad_norm",
                    "update_epochs",
                    "best_eval_episodes",
                    "seed",
                )
            },
        }
        train_code = run_child(
            python=python,
            child_args=[
                str(Path(__file__).resolve()),
                "--worker",
                "train",
                "--config",
                str(config_path),
                "--result-json",
                str(train_result_path),
            ],
            log_path=train_log,
            timeout_seconds=args.train_timeout_seconds,
            mem_fraction=args.mem_fraction,
            stop_file=stop_file,
        )
        record["train_returncode"] = train_code
        record["train_log"] = str(train_log)
        record["finished_train_at"] = now_iso()

        if train_code != 0 or not train_result_path.exists():
            record["status"] = "train_failed"
            append_jsonl(summary_path, record)
            write_leaderboard(summary_path, leaderboard_path, baseline)
            candidate_index += 1
            continue

        train_result = load_json(train_result_path)
        record["train_result"] = train_result
        best_checkpoint = Path(train_result["final_checkpoint_path"])
        if not best_checkpoint.exists():
            record["status"] = "missing_best_checkpoint"
            append_jsonl(summary_path, record)
            write_leaderboard(summary_path, leaderboard_path, baseline)
            candidate_index += 1
            continue

        eval_code = run_child(
            python=python,
            child_args=[
                str(Path(__file__).resolve()),
                "--worker",
                "eval",
                "--checkpoint",
                str(best_checkpoint),
                "--result-json",
                str(eval_result_path),
            ],
            log_path=eval_log,
            timeout_seconds=args.eval_timeout_seconds,
            mem_fraction=args.mem_fraction,
            stop_file=stop_file,
        )
        record["eval_returncode"] = eval_code
        record["eval_log"] = str(eval_log)
        record["finished_eval_at"] = now_iso()

        if eval_code != 0 or not eval_result_path.exists():
            record["status"] = "eval_failed"
            append_jsonl(summary_path, record)
            write_leaderboard(summary_path, leaderboard_path, baseline)
            prune_checkpoints(summary_path, sweep_dir, keep_top_k=args.keep_top_k)
            candidate_index += 1
            continue

        eval_metrics = load_json(eval_result_path)
        primary = float(eval_metrics[PRIMARY_METRIC])
        ratio = primary / baseline_primary
        promoted = (
            primary >= baseline_primary * PROMOTION_MARGIN
            and float(eval_metrics["eval_mean_delivered_fraction"])
            >= MIN_DELIVERED_FRACTION
            and float(eval_metrics["eval_success_rate"]) >= MIN_SUCCESS_RATE
        )
        record.update(
            {
                "status": "promoted" if promoted else "evaluated",
                "eval_metrics": eval_metrics,
                "primary_metric": PRIMARY_METRIC,
                "primary_value": primary,
                "baseline_primary_value": baseline_primary,
                "primary_ratio_vs_baseline": ratio,
                "promotion_threshold": baseline_primary * PROMOTION_MARGIN,
                "promoted": promoted,
                "checkpoint_paths": train_result.get("checkpoint_paths", []),
                "wandb_url": train_result.get("wandb_url"),
            }
        )
        append_jsonl(summary_path, record)
        write_leaderboard(summary_path, leaderboard_path, baseline)
        prune_checkpoints(summary_path, sweep_dir, keep_top_k=args.keep_top_k)

        print(
            "[result] "
            f"name={candidate['name']} {PRIMARY_METRIC}={primary:.6g} "
            f"ratio={ratio:.3f} frac={eval_metrics['eval_mean_delivered_fraction']:.3f} "
            f"success={eval_metrics['eval_success_rate']:.3f} "
            f"status={record['status']}",
            flush=True,
        )

        if promoted:
            render_result_path = sweep_dir / "renders" / f"{candidate['name']}.json"
            render_output = sweep_dir / "renders" / f"{candidate['name']}.mp4"
            render_log = sweep_dir / "logs" / f"{candidate['name']}.render.log"
            render_code = run_child(
                python=python,
                child_args=[
                    str(Path(__file__).resolve()),
                    "--worker",
                    "render",
                    "--checkpoint",
                    str(best_checkpoint),
                    "--output",
                    str(render_output),
                    "--result-json",
                    str(render_result_path),
                ],
                log_path=render_log,
                timeout_seconds=args.eval_timeout_seconds,
                mem_fraction=args.mem_fraction,
                stop_file=stop_file,
            )
            print(
                f"[render] name={candidate['name']} code={render_code} output={render_output}",
                flush=True,
            )

        candidate_index += 1

    print("[sweep] parent loop ended.", flush=True)


def worker_train(config_path: Path, result_path: Path) -> None:
    from ant_byte_env import notebook_workflows as workflows
    from ant_byte_env.training.jax_mappo import runner as jax_runner

    experiment = workflows.load_jax_experiment(config_path)
    experiment_args = dict(experiment.args)
    run_dir = PROJECT_ROOT / experiment.metadata["run_dir"]
    checkpoint_dir = run_dir / "checkpoints"
    source_checkpoint = workflows.resolve_project_path(
        PROJECT_ROOT, experiment_args["load_model"]
    )
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    best_checkpoint_path = workflows.resolve_project_path(
        PROJECT_ROOT, experiment_args["save_best_model"]
    )
    experiment_args["load_model"] = str(source_checkpoint)
    experiment_args["save_best_model"] = str(best_checkpoint_path)

    stages = workflows.build_food_cluster_curriculum_stages(
        experiment_args,
        source_counts=tuple(int(count) for count in experiment.metadata["food_source_counts"]),
        cluster_radii=tuple(int(radius) for radius in experiment.metadata["food_cluster_radii"]),
        visit_reward_schedule=experiment.metadata.get("visit_reward_schedule"),
        view_reward_schedule=experiment.metadata.get("view_reward_schedule"),
        stage_update_multiplier=float(experiment.metadata["stage_update_multiplier"]),
    )
    update_timesteps = workflows.update_timesteps(
        num_envs=int(experiment_args["num_envs"]),
        num_steps=int(experiment_args["num_steps"]),
    )
    update_cap = max(int(stage["global_update_cap"]) for stage in stages)
    common_args = workflows.config_common_args(
        experiment_args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )

    result = workflows.run_forage_curriculum(
        stages=stages,
        checkpoint_dir=checkpoint_dir,
        common_args=common_args,
        update_timesteps_per_stage=update_timesteps,
        global_update_cap=update_cap,
        train_main=jax_runner.main,
        initial_checkpoint=source_checkpoint,
        wandb_project=WANDB_PROJECT,
        wandb_entity=None,
        wandb_group=str(experiment.metadata["sweep_id"]),
        wandb_run_name=experiment.name,
        wandb_mode="online",
        wandb_tags=[
            "overnight-efficiency-sweep",
            "50x50",
            "60-ants",
            "no-shaping-continuation",
            "throughput-selected",
            "memory-safe-child",
        ],
        wandb_notes=experiment.metadata["notes"],
        wandb_artifact_paths=[config_path],
        wandb_artifact_prefix="overnight-efficiency-sweep",
        checkpoint_name_prefix=experiment_args["exp_name"],
        wandb_video_max_frames=None,
        wandb_video_stage_names=(),
        checkpoint_video_interval_updates=None,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint = result.get("final_checkpoint_path")
    checkpoint_paths = [
        str(path)
        for path in (
            list(result.get("best_stage_checkpoint_paths", []))
            + list(result.get("terminal_stage_checkpoint_paths", []))
        )
    ]
    write_json(
        result_path,
        {
            "final_checkpoint_path": None
            if final_checkpoint is None
            else str(final_checkpoint),
            "checkpoint_paths": checkpoint_paths,
            "final_train_metrics": result.get("final_train_metrics", {}),
            "last_reported_metrics": (result.get("stage_metrics") or [{}])[-1],
            "wandb_url": latest_wandb_url(run_dir),
        },
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def worker_eval(
    checkpoint_path: Path,
    result_path: Path,
    *,
    episodes: int,
    seed_offset: int,
    action_mode: str,
    move_temperature: float,
    write_temperature: float,
    cleanup_move_temperature: float | None = None,
    cleanup_fraction_threshold: float = 0.95,
    stall_cleanup_move_temperature: float | None = None,
    stall_cleanup_fraction_threshold: float = 0.90,
    stall_cleanup_patience_steps: int = 100,
) -> None:
    from ant_byte_env.training.jax_mappo.evaluation import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        checkpoint_path,
        num_episodes=int(episodes),
        seed_offset=int(seed_offset),
        action_mode=str(action_mode),
        move_temperature=float(move_temperature),
        write_temperature=float(write_temperature),
        cleanup_move_temperature=cleanup_move_temperature,
        cleanup_fraction_threshold=float(cleanup_fraction_threshold),
        stall_cleanup_move_temperature=stall_cleanup_move_temperature,
        stall_cleanup_fraction_threshold=float(stall_cleanup_fraction_threshold),
        stall_cleanup_patience_steps=int(stall_cleanup_patience_steps),
        shuffle_positions=True,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        result_path,
        {
            "checkpoint": str(checkpoint_path),
            "episodes": int(episodes),
            "seed_offset": int(seed_offset),
            "action_mode": str(action_mode),
            "move_temperature": float(move_temperature),
            "write_temperature": float(write_temperature),
            "cleanup_move_temperature": (
                None
                if cleanup_move_temperature is None
                else float(cleanup_move_temperature)
            ),
            "cleanup_fraction_threshold": float(cleanup_fraction_threshold),
            "stall_cleanup_move_temperature": (
                None
                if stall_cleanup_move_temperature is None
                else float(stall_cleanup_move_temperature)
            ),
            "stall_cleanup_fraction_threshold": float(
                stall_cleanup_fraction_threshold
            ),
            "stall_cleanup_patience_steps": int(stall_cleanup_patience_steps),
            **{key: float(metrics[key]) for key in EVAL_KEYS},
        },
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def worker_render(
    checkpoint_path: Path,
    output_path: Path,
    result_path: Path,
    *,
    policy_temperature: float,
    action_mode: str,
    move_temperature: float,
    write_temperature: float,
    seed_offset: int,
    max_frames: int,
    tile_size: int,
) -> None:
    from ant_byte_env.rendering import render_checkpoint

    rendered = render_checkpoint(
        checkpoint_path,
        output_path,
        backend="jax",
        reuse_existing=False,
        seed_offset=int(seed_offset),
        max_frames=int(max_frames),
        tile_size=int(tile_size),
        policy_temperature=float(policy_temperature),
        action_mode=str(action_mode),
        move_temperature=float(move_temperature),
        write_temperature=float(write_temperature),
        render_style="sprite",
        show_vision=False,
    )
    write_json(
        result_path,
        {
            "rendered": str(rendered),
            "policy_temperature": float(policy_temperature),
            "action_mode": str(action_mode),
            "move_temperature": float(move_temperature),
            "write_temperature": float(write_temperature),
            "seed_offset": int(seed_offset),
            "max_frames": int(max_frames),
            "tile_size": int(tile_size),
        },
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def candidate_for_index(index: int, sweep_dir: Path) -> dict[str, Any]:
    variant = VARIANTS[index % len(VARIANTS)]
    cycle = index // len(VARIANTS)
    seed = int(SEEDS[(index + cycle * 3) % len(SEEDS)] + 1000 * cycle)
    name = f"eff_{index:04d}_{variant['stem']}_seed{seed}"
    run_dir = sweep_dir.relative_to(PROJECT_ROOT) / name
    return {
        **variant,
        "index": index,
        "name": name,
        "variant_stem": variant["stem"],
        "seed": seed,
        "run_dir": PROJECT_ROOT / run_dir,
        "relative_run_dir": run_dir,
    }


def write_candidate_config(candidate: dict[str, Any], sweep_dir: Path) -> Path:
    base = load_json(BASE_CONFIG)
    config = copy.deepcopy(base)
    name = str(candidate["name"])
    update_cap = int(candidate["updates"])
    update_timesteps = 16 * 256
    save_best_model = (
        candidate["relative_run_dir"]
        / "checkpoints"
        / f"{name}_best.pkl"
    )
    args = config["args"]
    args.update(
        {
            "exp_name": f"jax_mappo_{name}",
            "total_timesteps": update_cap * update_timesteps,
            "learning_rate": float(candidate["learning_rate"]),
            "anneal_lr": bool(candidate["anneal_lr"]),
            "clip_coef": float(candidate["clip_coef"]),
            "ent_coef": 0.0,
            "max_grad_norm": float(candidate["max_grad_norm"]),
            "update_epochs": int(candidate["update_epochs"]),
            "training_rollout_temperature": float(
                candidate["training_rollout_temperature"]
            ),
            "step_penalty": 0.0,
            "distance_bonus": 0.0,
            "carrying_hub_distance_bonus": 0.0,
            "delivery_byte_trail_bonus": 0.0,
            "byte_follow_bonus": 0.0,
            "load_model": str(BASELINE_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "reset_optimizer_on_load": True,
            "save_best_model": str(save_best_model),
            "best_model_selection": "eval",
            "best_model_metric": PRIMARY_METRIC,
            "best_model_mode": "max",
            "best_eval_episodes": int(candidate["best_eval_episodes"]),
            "best_eval_interval": 5,
            "best_eval_action_mode": "sampled_move_greedy_write",
            "best_eval_move_temperature": 0.75,
            "best_eval_write_temperature": 1.0,
            "seed": int(candidate["seed"]),
            "quiet": True,
            "random_food": True,
            "random_hub": True,
        }
    )
    config.update(
        {
            "name": name,
            "description": (
                "Overnight no-shaping continuation candidate from the stabilized "
                "60-ant 50x50 checkpoint, selected by held-out delivery throughput."
            ),
        }
    )
    config["metadata"].update(
        {
            "source": "overnight_efficiency_sweep",
            "source_checkpoint": str(BASELINE_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "sweep_id": sweep_dir.name,
            "run_dir": str(candidate["relative_run_dir"]),
            "stage_update_multiplier": update_cap / FIFTY_BY_FIFTY_BASE_UPDATES,
            "checkpoint_video_interval_updates": None,
            "wandb_preview_rollout_count": 0,
            "wandb_video_max_frames": None,
            "notes": (
                "No-shaping overnight efficiency continuation. Reward shaping is kept "
                "off because recent speed/distance/time shaping reduced fair throughput. "
                "Promote only after a 32-episode re-eval beats the stabilized baseline."
            ),
            "overnight_candidate": {
                key: candidate[key]
                for key in (
                    "index",
                    "variant_stem",
                    "learning_rate",
                    "clip_coef",
                    "training_rollout_temperature",
                    "updates",
                    "anneal_lr",
                    "max_grad_norm",
                    "update_epochs",
                    "best_eval_episodes",
                    "seed",
                )
            },
        }
    )
    config["metadata"]["reward_scales"].update(
        {
            "step_penalty": 0.0,
            "distance_bonus": 0.0,
            "carrying_hub_distance_bonus": 0.0,
            "delivery_byte_trail_bonus": 0.0,
            "byte_follow_bonus": 0.0,
        }
    )
    config_path = sweep_dir / "configs" / f"{name}.json"
    write_json(config_path, config)
    return config_path


def ensure_eval(
    *,
    label: str,
    checkpoint: Path,
    eval_dir: Path,
    log_dir: Path,
    python: str,
    timeout_seconds: int,
    mem_fraction: str,
) -> dict[str, float]:
    result_path = eval_dir / f"{label}.json"
    log_path = log_dir / f"{label}.eval.log"
    if result_path.exists():
        return load_json(result_path)
    code = run_child(
        python=python,
        child_args=[
            str(Path(__file__).resolve()),
            "--worker",
            "eval",
            "--checkpoint",
            str(checkpoint),
            "--result-json",
            str(result_path),
        ],
        log_path=log_path,
        timeout_seconds=timeout_seconds,
        mem_fraction=mem_fraction,
        stop_file=None,
    )
    if code != 0 or not result_path.exists():
        raise RuntimeError(f"baseline eval failed; see {log_path}")
    return load_json(result_path)


def run_child(
    *,
    python: str,
    child_args: list[str],
    log_path: Path,
    timeout_seconds: int,
    mem_fraction: str,
    stop_file: Path | None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "PYTHONUNBUFFERED": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "WANDB_SILENT": "true",
            "MPLBACKEND": "Agg",
        }
    )
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [python, *child_args],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        while True:
            code = process.poll()
            if code is not None:
                wait_for_gpu_to_clear(seconds=20)
                return int(code)
            if stop_file is not None and stop_file.exists():
                terminate_process_group(process)
                wait_for_gpu_to_clear(seconds=20)
                return 143
            if time.monotonic() - started > timeout_seconds:
                log_file.write(f"\nTIMEOUT after {timeout_seconds} seconds\n")
                log_file.flush()
                terminate_process_group(process)
                wait_for_gpu_to_clear(seconds=20)
                return 124
            time.sleep(5)


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def wait_for_gpu_to_clear(*, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not gpu_compute_rows():
            return
        time.sleep(2)


def gpu_compute_rows() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def prepare_disk_space(project_root: Path, min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(project_root).free / 1024**3
    if free_gb >= min_free_gb:
        return
    cleanup_safe_caches(project_root)
    free_gb = shutil.disk_usage(project_root).free / 1024**3
    while free_gb < min_free_gb:
        print(
            f"[disk] free={free_gb:.2f}GB below {min_free_gb:.2f}GB; sleeping before retry",
            flush=True,
        )
        time.sleep(300)
        cleanup_safe_caches(project_root)
        free_gb = shutil.disk_usage(project_root).free / 1024**3


def cleanup_safe_caches(root: Path) -> None:
    safe_names = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache"}
    for path in root.rglob("*"):
        if path.is_dir() and path.name in safe_names:
            shutil.rmtree(path, ignore_errors=True)


def prune_checkpoints(summary_path: Path, sweep_dir: Path, *, keep_top_k: int) -> None:
    records = evaluated_records(summary_path)
    ranked = sorted(
        records,
        key=lambda record: float(record.get("primary_value", float("-inf"))),
        reverse=True,
    )
    keep_names = {record["name"] for record in ranked[: max(keep_top_k, 0)]}
    keep_names.update(record["name"] for record in records if record.get("promoted"))
    for record in records:
        paths = [Path(path) for path in record.get("checkpoint_paths", [])]
        for path in paths:
            if not is_path_under(path, sweep_dir):
                continue
            if record["name"] in keep_names and path.name.endswith("_best.pkl"):
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def is_path_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def write_leaderboard(
    summary_path: Path,
    leaderboard_path: Path,
    baseline: dict[str, float],
) -> None:
    records = evaluated_records(summary_path)
    ranked = sorted(
        records,
        key=lambda record: float(record.get("primary_value", float("-inf"))),
        reverse=True,
    )
    lines = [
        "# Overnight Efficiency Sweep",
        "",
        f"Updated: {now_iso()}",
        "",
        "## Baseline",
        "",
        f"- checkpoint: `{BASELINE_CHECKPOINT}`",
        f"- {PRIMARY_METRIC}: {float(baseline[PRIMARY_METRIC]):.6g}",
        f"- promotion threshold: {float(baseline[PRIMARY_METRIC]) * PROMOTION_MARGIN:.6g}",
        f"- delivered fraction: {float(baseline['eval_mean_delivered_fraction']):.6g}",
        f"- success rate: {float(baseline['eval_success_rate']):.6g}",
        "",
        "## Ranked Candidates",
        "",
        "| rank | status | candidate | primary | ratio | frac | success | len | W&B | checkpoint |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, record in enumerate(ranked, start=1):
        metrics = record["eval_metrics"]
        checkpoint = record.get("train_result", {}).get("final_checkpoint_path", "")
        lines.append(
            "| {rank} | {status} | `{name}` | {primary:.6g} | {ratio:.3f} | "
            "{frac:.3f} | {success:.3f} | {length:.1f} | {wandb} | `{checkpoint}` |".format(
                rank=rank,
                status=record.get("status", ""),
                name=record.get("name", ""),
                primary=float(record.get("primary_value", 0.0)),
                ratio=float(record.get("primary_ratio_vs_baseline", 0.0)),
                frac=float(metrics.get("eval_mean_delivered_fraction", 0.0)),
                success=float(metrics.get("eval_success_rate", 0.0)),
                length=float(metrics.get("eval_mean_episode_length", 0.0)),
                wandb=record.get("wandb_url") or "",
                checkpoint=checkpoint,
            )
        )
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluated_records(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "eval_metrics" in record:
            records.append(record)
    return records


def next_candidate_index(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    max_index = -1
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        max_index = max(max_index, int(record.get("index", -1)))
    return max_index + 1


def latest_wandb_url(run_dir: Path) -> str | None:
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    runs = sorted(wandb_dir.glob("run-*"), key=lambda path: path.stat().st_mtime)
    if not runs:
        return None
    run_id = runs[-1].name.rsplit("-", 1)[-1]
    return f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}"


def default_sweep_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / "overnight_efficiency_sweep" / f"sweep_{stamp}"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


if __name__ == "__main__":
    main()
