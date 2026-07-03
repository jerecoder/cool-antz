#!/usr/bin/env python3
"""Small continuation sweep from the strongest 100x100 bridge checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_100x100_bridge_sweep import (  # noqa: E402
    DEFAULT_PYTHON,
    DEFAULT_NUM_ANTS,
    PRIMARY_METRIC,
    WANDB_PROJECT,
    append_jsonl,
    load_json,
    next_candidate_index,
    prepare_disk_space,
    read_jsonl,
    resolve_project_path,
    run_child,
    worker_eval,
    write_json,
)


SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "bridge_100x100_sweep"
    / "sweep_20260702_222315"
    / "bridge_0005_mid_mt050_lr1e5_u10"
    / "checkpoints"
    / "bridge_0005_mid_mt050_lr1e5_u10_best.pkl"
)
BASE_SWEEP = PROJECT_ROOT / "runs" / "bridge_100x100_sweep" / "sweep_20260702_222315"

TASKS: dict[str, dict[str, Any]] = {
    "mid250_4src": {
        "food_count": 250,
        "food_sources": 4,
        "food_cluster_count": 4,
        "max_steps": 5000,
        "gate_fraction": 0.90,
        "gate_success": 0.55,
    },
    "hard375_6src": {
        "food_count": 375,
        "food_sources": 6,
        "food_cluster_count": 6,
        "max_steps": 6000,
        "gate_fraction": 0.65,
        "gate_success": 0.20,
    },
}

CANDIDATES: list[dict[str, Any]] = [
    {
        "stem": "mid250_from_best_mt050_lr5e6_u24",
        "task": "mid250_4src",
        "learning_rate": 5.0e-6,
        "training_rollout_temperature": 0.50,
        "move_temperature": 0.50,
        "updates": 24,
        "seed": 211,
        "source_checkpoint": SOURCE_CHECKPOINT,
    },
    {
        "stem": "mid250_from_best_mt0525_lr5e6_u24",
        "task": "mid250_4src",
        "learning_rate": 5.0e-6,
        "training_rollout_temperature": 0.525,
        "move_temperature": 0.525,
        "updates": 24,
        "seed": 223,
        "source_checkpoint": SOURCE_CHECKPOINT,
    },
    {
        "stem": "mid250_from_best_mt060_lr5e6_u16",
        "task": "mid250_4src",
        "learning_rate": 5.0e-6,
        "training_rollout_temperature": 0.60,
        "move_temperature": 0.60,
        "updates": 16,
        "seed": 227,
        "source_checkpoint": SOURCE_CHECKPOINT,
    },
    {
        "stem": "hard375_from_best_mt050_lr5e6_u16",
        "task": "hard375_6src",
        "learning_rate": 5.0e-6,
        "training_rollout_temperature": 0.50,
        "move_temperature": 0.50,
        "updates": 16,
        "seed": 229,
        "source_checkpoint": SOURCE_CHECKPOINT,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("train", "eval"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--eval-seed-offset", type=int, default=8_000_000)
    parser.add_argument("--train-timeout-seconds", type=int, default=5400)
    parser.add_argument("--eval-timeout-seconds", type=int, default=1500)
    parser.add_argument("--min-disk-free-gb", type=float, default=2.0)
    parser.add_argument("--mem-fraction", default="0.34")
    parser.add_argument("--num-ants", type=int, default=DEFAULT_NUM_ANTS)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
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
        )
    else:
        parent_loop(args)


def parent_loop(args: argparse.Namespace) -> None:
    python = os.environ.get("ANTZ_PYTHON", DEFAULT_PYTHON)
    if int(args.num_ants) <= 0:
        raise ValueError("--num-ants must be positive")
    source_checkpoint = resolve_project_path(args.source_checkpoint)
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_checkpoint}")
    sweep_dir = (args.sweep_dir or default_sweep_dir()).resolve()
    for child in ("configs", "logs", "evals"):
        (sweep_dir / child).mkdir(parents=True, exist_ok=True)

    stop_file = sweep_dir / "STOP"
    summary_path = sweep_dir / "summary.jsonl"
    leaderboard_path = sweep_dir / "leaderboard.md"
    write_json(
        sweep_dir / "state.json",
        {
            "base_sweep": str(BASE_SWEEP),
            "source_checkpoint": str(source_checkpoint),
            "num_ants": int(args.num_ants),
            "primary_metric": PRIMARY_METRIC,
            "started_at": now_iso(),
            "candidates": public_candidates(
                num_ants=int(args.num_ants),
                source_checkpoint=source_checkpoint,
            ),
            "tasks": TASKS,
        },
    )
    print(f"[sweep] dir={sweep_dir}", flush=True)
    print(f"[sweep] stop_file={stop_file}", flush=True)

    start_index = next_candidate_index(summary_path)
    max_count = len(CANDIDATES) if args.max_candidates is None else max(0, args.max_candidates)
    for index in range(start_index, min(len(CANDIDATES), start_index + max_count)):
        if stop_file.exists():
            print("[stop] STOP file found; exiting loop cleanly.", flush=True)
            break
        prepare_disk_space(PROJECT_ROOT, args.min_disk_free_gb)
        candidate = candidate_for_index(
            index,
            sweep_dir,
            num_ants=int(args.num_ants),
            source_checkpoint=source_checkpoint,
        )
        config_path = write_candidate_config(candidate, sweep_dir)
        train_result_path = candidate["run_dir"] / "train_result.json"
        eval_result_path = sweep_dir / "evals" / f"{candidate['name']}.json"
        train_log = sweep_dir / "logs" / f"{candidate['name']}.train.log"
        eval_log = sweep_dir / "logs" / f"{candidate['name']}.eval.log"

        print(
            "[candidate] "
            f"idx={index} name={candidate['name']} task={candidate['task']} "
            f"lr={candidate['learning_rate']} temp={candidate['training_rollout_temperature']} "
            f"updates={candidate['updates']}",
            flush=True,
        )
        record: dict[str, Any] = {
            "index": index,
            "name": candidate["name"],
            "task": candidate["task"],
            "config": str(config_path),
            "run_dir": str(candidate["run_dir"]),
            "started_at": now_iso(),
            "candidate": public_candidate(candidate),
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
            write_leaderboard(summary_path, leaderboard_path)
            continue

        train_result = load_json(train_result_path)
        checkpoint = Path(train_result["checkpoint"])
        record["train_result"] = train_result
        if not checkpoint.exists():
            record["status"] = "missing_checkpoint"
            append_jsonl(summary_path, record)
            write_leaderboard(summary_path, leaderboard_path)
            continue

        eval_code = run_child(
            python=python,
            child_args=[
                str(Path(__file__).resolve()),
                "--worker",
                "eval",
                "--checkpoint",
                str(checkpoint),
                "--result-json",
                str(eval_result_path),
                "--eval-episodes",
                str(args.eval_episodes),
                "--eval-seed-offset",
                str(args.eval_seed_offset),
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
            write_leaderboard(summary_path, leaderboard_path)
            continue

        metrics = load_json(eval_result_path)
        task = TASKS[candidate["task"]]
        alive = (
            float(metrics["eval_mean_delivered_fraction"]) >= float(task["gate_fraction"])
            and float(metrics["eval_success_rate"]) >= float(task["gate_success"])
        )
        record.update(
            {
                "status": "alive" if alive else "evaluated",
                "eval_metrics": metrics,
                "primary_metric": PRIMARY_METRIC,
                "primary_value": float(metrics[PRIMARY_METRIC]),
                "alive": alive,
                "gate_fraction": float(task["gate_fraction"]),
                "gate_success": float(task["gate_success"]),
                "checkpoint": str(checkpoint),
                "wandb_url": train_result.get("wandb_url"),
            }
        )
        append_jsonl(summary_path, record)
        write_leaderboard(summary_path, leaderboard_path)
        print(
            "[result] "
            f"name={candidate['name']} {PRIMARY_METRIC}={metrics[PRIMARY_METRIC]:.6g} "
            f"frac={metrics['eval_mean_delivered_fraction']:.3f} "
            f"success={metrics['eval_success_rate']:.3f} status={record['status']}",
            flush=True,
        )
    print("[sweep] parent loop ended.", flush=True)


def worker_train(config_path: Path, result_path: Path) -> None:
    from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
    from ant_byte_env.training.jax_mappo import runner as jax_runner

    spec = load_experiment_config(config_path)
    final_metrics = jax_runner.main(config_args_to_argv(spec.args))
    args = spec.args
    checkpoint = args.get("save_best_model") or args.get("save_model")
    if checkpoint is None:
        checkpoint = Path(args["run_dir"]) / "checkpoints" / "model.pkl"
    result = {
        "checkpoint": str(resolve_project_path(Path(checkpoint))),
        "final_train_metrics": final_metrics,
        "source_checkpoint": str(resolve_project_path(Path(args["load_model"]))),
        "wandb_url": latest_wandb_url(resolve_project_path(Path(args["run_dir"]))),
    }
    write_json(result_path, result)


def candidate_for_index(
    index: int,
    sweep_dir: Path,
    *,
    num_ants: int,
    source_checkpoint: Path,
) -> dict[str, Any]:
    base = CANDIDATES[index]
    name = f"continue{int(num_ants)}_{index:04d}_{base['stem']}"
    return {
        **base,
        "index": index,
        "name": name,
        "num_ants": int(num_ants),
        "source_checkpoint": source_checkpoint,
        "run_dir": sweep_dir / name,
    }


def write_candidate_config(candidate: dict[str, Any], sweep_dir: Path) -> Path:
    task = TASKS[candidate["task"]]
    checkpoint_dir = candidate["run_dir"] / "checkpoints"
    best_checkpoint = checkpoint_dir / f"{candidate['name']}_best.pkl"
    latest_checkpoint = checkpoint_dir / f"{candidate['name']}_latest.pkl"
    args: dict[str, Any] = {
        "exp_name": f"jax_mappo_100x100_{candidate['name']}",
        "total_timesteps": int(candidate["updates"]) * 4 * 256,
        "num_envs": 4,
        "num_steps": 256,
        "num_minibatches": 4,
        "update_epochs": 1,
        "log_interval": 1,
        "learning_rate": float(candidate["learning_rate"]),
        "anneal_lr": False,
        "clip_coef": 0.03,
        "ent_coef": 0.0,
        "vf_coef": 0.25,
        "max_grad_norm": 0.25,
        "gamma": 0.997,
        "training_rollout_temperature": float(candidate["training_rollout_temperature"]),
        "width": 100,
        "height": 100,
        "obs_width": 100,
        "obs_height": 100,
        "layout_margin": 0,
        "hub_center_window_size": 48,
        "actor_vision_radius": 2,
        "num_ants": int(candidate["num_ants"]),
        "agent_identity_types": 8,
        "food_count": int(task["food_count"]),
        "food_sources": int(task["food_sources"]),
        "food_cluster_count": int(task["food_cluster_count"]),
        "food_cluster_radius": int(task.get("food_cluster_radius", 0)),
        "cookie_distance": 18,
        "max_steps": int(task["max_steps"]),
        "reward_mode": "forage",
        "pickup_bonus": 0.05,
        "visit_reward_scale": 0.0,
        "view_reward_scale": 0.0,
        "distance_bonus": 0.0,
        "carrying_hub_distance_bonus": 0.0,
        "hidden_size": 128,
        "critic_architecture": "strided_cnn",
        "write_bits": 8,
        "write_while_moving": True,
        "write_bit_penalty": 0.0002,
        "write_bit_penalty_decay": 0.5,
        "delivery_byte_trail_bonus": 0.0,
        "byte_follow_bonus": 0.0,
        "load_model": str(candidate["source_checkpoint"]),
        "reset_optimizer_on_load": True,
        "save_model": str(latest_checkpoint),
        "save_best_model": str(best_checkpoint),
        "best_model_selection": "eval",
        "best_model_metric": PRIMARY_METRIC,
        "best_model_mode": "max",
        "best_eval_interval": 4,
        "best_eval_episodes": 4,
        "best_eval_action_mode": "sampled_move_greedy_write",
        "best_eval_move_temperature": float(candidate["move_temperature"]),
        "best_eval_write_temperature": 1.0,
        "best_eval_seed_offset": 6_500_000,
        "random_food": True,
        "random_hub": True,
        "quiet": True,
        "seed": int(candidate["seed"]),
        "run_dir": str(candidate["run_dir"]),
        "wandb_project": WANDB_PROJECT,
        "wandb_group": sweep_dir.name,
        "wandb_run_name": candidate["name"],
        "wandb_tags": [
            "100x100",
            "continuation-sweep",
            f"{int(candidate['num_ants'])}-ants",
            candidate["task"],
        ],
        "wandb_notes": (
            "100x100 continuation from the best bridge checkpoint; lower LR, "
            f"{int(candidate['num_ants'])}-ant target, fresh optimizer, and narrow "
            "temperature/harder-task probes."
        ),
    }
    payload = {
        "name": candidate["name"],
        "backend": "jax",
        "description": (
            "Continuation probe from the strongest 100x100 bridge checkpoint "
            f"with {int(candidate['num_ants'])} ants."
        ),
        "args": args,
        "metadata": {
            "base_sweep": str(BASE_SWEEP),
            "source_checkpoint": str(candidate["source_checkpoint"]),
            "task": candidate["task"],
            "task_settings": task,
            "candidate": public_candidate(candidate),
        },
    }
    config_path = sweep_dir / "configs" / f"{candidate['name']}.json"
    write_json(config_path, payload)
    return config_path


def write_leaderboard(summary_path: Path, output_path: Path) -> None:
    records = [
        record
        for record in read_jsonl(summary_path)
        if record.get("status") in {"alive", "evaluated"}
        and record.get("eval_metrics") is not None
    ]
    records.sort(key=lambda record: float(record["eval_metrics"][PRIMARY_METRIC]), reverse=True)
    lines = [
        "# 100x100 Continuation Sweep",
        "",
        f"Updated: {now_iso()}",
        "",
        "| rank | status | candidate | task | primary | delivered | frac | success | W&B |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, record in enumerate(records, start=1):
        metrics = record["eval_metrics"]
        lines.append(
            "| {rank} | {status} | `{name}` | `{task}` | {primary:.6g} | "
            "{delivered:.1f} | {frac:.3f} | {success:.3f} | {wandb} |".format(
                rank=rank,
                status=record["status"],
                name=record["name"],
                task=record["task"],
                primary=float(metrics[PRIMARY_METRIC]),
                delivered=float(metrics["eval_mean_delivered_food"]),
                frac=float(metrics["eval_mean_delivered_fraction"]),
                success=float(metrics["eval_success_rate"]),
                wandb=record.get("wandb_url") or "",
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(candidate[key]) if key == "source_checkpoint" else candidate[key]
        for key in (
            "index",
            "stem",
            "task",
            "num_ants",
            "learning_rate",
            "training_rollout_temperature",
            "move_temperature",
            "updates",
            "seed",
            "source_checkpoint",
        )
        if key in candidate
    }


def public_candidates(*, num_ants: int, source_checkpoint: Path = SOURCE_CHECKPOINT) -> list[dict[str, Any]]:
    return [
        public_candidate(
            {
                **candidate,
                "index": index,
                "num_ants": int(num_ants),
                "source_checkpoint": source_checkpoint,
            }
        )
        for index, candidate in enumerate(CANDIDATES)
    ]


def latest_wandb_url(run_dir: Path) -> str | None:
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    runs = sorted(wandb_dir.glob("run-*"))
    if not runs:
        return None
    run_id = runs[-1].name.rsplit("-", 1)[-1]
    return f"https://wandb.ai/jerefigueiredo-universidad-de-san-andr-s/{WANDB_PROJECT}/runs/{run_id}"


def default_sweep_dir() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("sweep_%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / "bridge_100x100_sweep" / f"continue_{stamp}"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


if __name__ == "__main__":
    main()
