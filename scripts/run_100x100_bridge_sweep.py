#!/usr/bin/env python3
"""Actor-warm-start bridge sweep from the 50x50 policy to 100x100 maps."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = sys.executable
SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "notebooks"
    / "fl50_60ants_half_food_sw_wc_8bits_stabilize_from_60best"
    / "checkpoints"
    / "best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits_stabilized.pkl"
)
WANDB_PROJECT = "cool-antz"
DEFAULT_NUM_ANTS = 120
PRIMARY_METRIC = "eval_mean_delivered_food_per_1000_ant_steps"
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
TASKS: dict[str, dict[str, Any]] = {
    "easy125_2src": {
        "food_count": 125,
        "food_sources": 2,
        "food_cluster_count": 2,
        "max_steps": 4000,
        "gate_fraction": 0.50,
        "gate_success": 0.10,
    },
    "mid250_4src": {
        "food_count": 250,
        "food_sources": 4,
        "food_cluster_count": 4,
        "max_steps": 5000,
        "gate_fraction": 0.35,
        "gate_success": 0.05,
    },
}

CANDIDATES: list[dict[str, Any]] = [
    {
        "stem": "easy_mt0525_lr1e5_u8",
        "task": "easy125_2src",
        "learning_rate": 1.0e-5,
        "training_rollout_temperature": 0.525,
        "move_temperature": 0.525,
        "updates": 8,
        "seed": 101,
    },
    {
        "stem": "easy_mt050_lr1e5_u8",
        "task": "easy125_2src",
        "learning_rate": 1.0e-5,
        "training_rollout_temperature": 0.50,
        "move_temperature": 0.50,
        "updates": 8,
        "seed": 107,
    },
    {
        "stem": "easy_mt060_lr1e5_u8",
        "task": "easy125_2src",
        "learning_rate": 1.0e-5,
        "training_rollout_temperature": 0.60,
        "move_temperature": 0.60,
        "updates": 8,
        "seed": 109,
    },
    {
        "stem": "easy_mt0525_lr5e6_u16",
        "task": "easy125_2src",
        "learning_rate": 5.0e-6,
        "training_rollout_temperature": 0.525,
        "move_temperature": 0.525,
        "updates": 16,
        "seed": 113,
    },
    {
        "stem": "mid_mt0525_lr1e5_u10",
        "task": "mid250_4src",
        "learning_rate": 1.0e-5,
        "training_rollout_temperature": 0.525,
        "move_temperature": 0.525,
        "updates": 10,
        "seed": 127,
    },
    {
        "stem": "mid_mt050_lr1e5_u10",
        "task": "mid250_4src",
        "learning_rate": 1.0e-5,
        "training_rollout_temperature": 0.50,
        "move_temperature": 0.50,
        "updates": 10,
        "seed": 131,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("train", "eval", "prepare"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-seed-offset", type=int, default=7_000_000)
    parser.add_argument("--train-timeout-seconds", type=int, default=3600)
    parser.add_argument("--eval-timeout-seconds", type=int, default=1200)
    parser.add_argument("--min-disk-free-gb", type=float, default=2.0)
    parser.add_argument("--mem-fraction", default="0.34")
    parser.add_argument("--num-ants", type=int, default=DEFAULT_NUM_ANTS)
    args = parser.parse_args()

    if args.worker == "prepare":
        if args.config is None or args.result_json is None:
            parser.error("--worker prepare requires --config and --result-json")
        result = prepare_actor_warm_start(args.config)
        write_json(args.result_json, result)
    elif args.worker == "train":
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
    sweep_dir = args.sweep_dir or default_sweep_dir()
    sweep_dir = sweep_dir.resolve()
    for child in ("configs", "logs", "evals", "warmstarts"):
        (sweep_dir / child).mkdir(parents=True, exist_ok=True)

    stop_file = sweep_dir / "STOP"
    summary_path = sweep_dir / "summary.jsonl"
    leaderboard_path = sweep_dir / "leaderboard.md"
    state_path = sweep_dir / "state.json"
    write_json(
        state_path,
        {
            "sweep_dir": str(sweep_dir),
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "num_ants": int(args.num_ants),
            "primary_metric": PRIMARY_METRIC,
            "started_at": now_iso(),
            "candidates": public_candidates(num_ants=int(args.num_ants)),
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
        candidate = candidate_for_index(index, sweep_dir, num_ants=int(args.num_ants))
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

    warm_start = prepare_actor_warm_start(config_path)
    spec = load_experiment_config(config_path)
    final_metrics = jax_runner.main(config_args_to_argv(spec.args))
    args = spec.args
    checkpoint = args.get("save_best_model") or args.get("save_model")
    if checkpoint is None:
        checkpoint = Path(args["run_dir"]) / "checkpoints" / "model.pkl"
    checkpoint_path = resolve_project_path(Path(checkpoint))
    run_dir = resolve_project_path(Path(args["run_dir"]))
    result = {
        "checkpoint": str(checkpoint_path),
        "warm_start": warm_start,
        "final_train_metrics": final_metrics,
        "wandb_url": latest_wandb_url(run_dir),
    }
    write_json(result_path, result)


def worker_eval(
    checkpoint_path: Path,
    result_path: Path,
    *,
    episodes: int,
    seed_offset: int,
) -> None:
    from ant_byte_env.training.jax_mappo.evaluation import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        checkpoint_path,
        num_episodes=int(episodes),
        seed_offset=int(seed_offset),
        action_mode="sampled_move_greedy_write",
        move_temperature=0.525,
        write_temperature=1.0,
        shuffle_positions=True,
    )
    write_json(
        result_path,
        {
            "checkpoint": str(checkpoint_path),
            "episodes": int(episodes),
            "seed_offset": int(seed_offset),
            "action_mode": "sampled_move_greedy_write",
            "move_temperature": 0.525,
            "write_temperature": 1.0,
            **{key: float(metrics[key]) for key in EVAL_KEYS},
        },
    )


def prepare_actor_warm_start(config_path: Path) -> dict[str, Any]:
    import argparse as argparse_module
    import jax

    from ant_byte_env import write_value_count
    from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
    from ant_byte_env.jax_env import JaxAntByteForagingEnv
    from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint, save_checkpoint
    from ant_byte_env.training.jax_mappo.cli import parse_args
    from ant_byte_env.training.jax_mappo.core import (
        build_actor_observations,
        build_central_observations,
        food_observation_scale,
        init_adam_state,
        init_agent_params,
    )
    from ant_byte_env.training.jax_mappo.curriculum import reset_batch

    spec = load_experiment_config(config_path)
    args = parse_args(config_args_to_argv(spec.args))
    source_checkpoint = resolve_project_path(
        Path(spec.metadata.get("source_checkpoint", SOURCE_CHECKPOINT))
    )
    warm_start_path = resolve_project_path(Path(spec.metadata["warm_start_checkpoint"]))
    if warm_start_path.exists():
        return {
            "path": str(warm_start_path),
            "source_checkpoint": str(source_checkpoint),
            "reused": True,
        }

    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        random_ant_spawn=getattr(args, "random_ant_spawn", False),
        random_ant_spawn_radius=getattr(args, "random_ant_spawn_radius", None),
        actor_vision_radius=int(getattr(args, "actor_vision_radius", 1)),
        step_penalty=args.step_penalty,
        completion_bonus=getattr(args, "completion_bonus", 0.0),
        write_penalty=args.write_penalty,
        write_bits=args.write_bits,
        write_while_moving=args.write_while_moving,
        per_ant_write_channels=bool(getattr(args, "per_ant_write_channels", False)),
        hub_center_window_size=int(getattr(args, "hub_center_window_size", 0)),
        terminate_on_food_delivery=bool(getattr(args, "food_termination", True)),
        terminate_on_full_coverage=bool(getattr(args, "terminate_on_full_coverage", False)),
        maze_obstacles=bool(getattr(args, "maze_obstacles", False)),
        maze_corridor_width=int(getattr(args, "maze_corridor_width", 3)),
        maze_wall_width=int(getattr(args, "maze_wall_width", 1)),
        maze_seed=int(getattr(args, "maze_seed", 0)),
        layout_margin=int(getattr(args, "layout_margin", 0)),
    )
    shape_args = argparse_module.Namespace(**{**vars(args), "num_envs": 1})
    _, obs = reset_batch(args=shape_args, env=env, key=jax.random.PRNGKey(args.seed))
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    central_obs = build_central_observations(
        obs,
        food_scale=food_scale,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        agent_identity_types=getattr(args, "agent_identity_types", None),
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    target_params = init_agent_params(
        jax.random.PRNGKey(int(args.seed) + 100_000),
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
        critic_architecture=getattr(args, "critic_architecture", "mlp"),
        critic_num_ants=args.num_ants,
        critic_obs_height=args.obs_height or args.height,
        critic_obs_width=args.obs_width or args.width,
    )
    source = read_checkpoint(source_checkpoint)
    source_params = source["params"]
    source_args = source.get("args", {})
    if tuple(source_params.actor_body[0].weight.shape) != tuple(
        target_params.actor_body[0].weight.shape
    ):
        raise ValueError(
            "Actor input shape changed; cannot actor-warm-start "
            f"{source_params.actor_body[0].weight.shape} -> "
            f"{target_params.actor_body[0].weight.shape}."
        )
    if tuple(source_params.write_head.weight.shape) != tuple(
        target_params.write_head.weight.shape
    ):
        raise ValueError("Write-head shape changed; cannot actor-warm-start.")
    target_params = target_params._replace(
        actor_body=source_params.actor_body,
        move_head=source_params.move_head,
        write_head=source_params.write_head,
    )
    warm_args = copy.copy(args)
    warm_args.load_model = None
    warm_args.save_model = warm_start_path
    warm_args.save_best_model = None
    warm_args.reset_optimizer_on_load = False
    save_checkpoint(
        warm_start_path,
        params=target_params,
        opt_state=init_adam_state(target_params),
        args=warm_args,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        run_name=f"{spec.name}_actor_warm_start",
        metrics={
            "actor_warm_start": 1.0,
            "source_central_obs_dim": float(source["central_obs_dim"]),
            "target_central_obs_dim": float(central_obs.shape[-1]),
            "source_actor_obs_dim": float(source["actor_obs_dim"]),
            "target_actor_obs_dim": float(actor_obs.shape[-1]),
        },
    )
    return {
        "path": str(warm_start_path),
        "source_checkpoint": str(source_checkpoint),
        "source_num_ants": int(source_args.get("num_ants", args.num_ants)),
        "target_num_ants": int(args.num_ants),
        "reused": False,
        "source_central_obs_dim": int(source["central_obs_dim"]),
        "target_central_obs_dim": int(central_obs.shape[-1]),
        "source_actor_obs_dim": int(source["actor_obs_dim"]),
        "target_actor_obs_dim": int(actor_obs.shape[-1]),
        "critic_policy": "fresh_target_critic",
    }


def candidate_for_index(index: int, sweep_dir: Path, *, num_ants: int) -> dict[str, Any]:
    base = CANDIDATES[index]
    name = f"bridge{int(num_ants)}_{index:04d}_{base['stem']}"
    run_dir = sweep_dir / name
    return {
        **base,
        "index": index,
        "name": name,
        "num_ants": int(num_ants),
        "run_dir": run_dir,
        "warm_start_checkpoint": sweep_dir / "warmstarts" / f"{name}_actor_warm_start.pkl",
    }


def write_candidate_config(candidate: dict[str, Any], sweep_dir: Path) -> Path:
    task = TASKS[candidate["task"]]
    checkpoint_dir = candidate["run_dir"] / "checkpoints"
    warm_start = candidate["warm_start_checkpoint"]
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
        "load_model": str(warm_start),
        "reset_optimizer_on_load": True,
        "save_model": str(latest_checkpoint),
        "save_best_model": str(best_checkpoint),
        "best_model_selection": "eval",
        "best_model_metric": PRIMARY_METRIC,
        "best_model_mode": "max",
        "best_eval_interval": max(1, min(4, int(candidate["updates"]))),
        "best_eval_episodes": 4,
        "best_eval_action_mode": "sampled_move_greedy_write",
        "best_eval_move_temperature": float(candidate["move_temperature"]),
        "best_eval_write_temperature": 1.0,
        "best_eval_seed_offset": 6_000_000,
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
            "bridge-sweep",
            "actor-warm-start",
            f"{int(candidate['num_ants'])}-ants",
            candidate["task"],
        ],
        "wandb_notes": (
            "100x100 bridge candidate: copy the stabilized 50x50 actor heads into "
            f"a {int(candidate['num_ants'])}-ant target-shaped checkpoint with a fresh "
            "100x100 strided-CNN critic."
        ),
    }
    payload = {
        "name": candidate["name"],
        "backend": "jax",
        "description": (
            "Actor-only warm-start bridge from the stabilized 50x50 60-ant policy "
            f"to a 100x100 random full-layout task with {int(candidate['num_ants'])} ants."
        ),
        "args": args,
        "metadata": {
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "warm_start_checkpoint": str(warm_start),
            "transfer_policy": "copy_actor_reset_critic",
            "task": candidate["task"],
            "task_settings": task,
            "candidate": public_candidate(candidate),
        },
    }
    config_path = sweep_dir / "configs" / f"{candidate['name']}.json"
    write_json(config_path, payload)
    return config_path


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "index",
        "stem",
        "task",
        "num_ants",
        "learning_rate",
        "training_rollout_temperature",
        "move_temperature",
        "updates",
        "seed",
    )
    return {key: candidate[key] for key in keys if key in candidate}


def public_candidates(*, num_ants: int) -> list[dict[str, Any]]:
    return [
        public_candidate({**candidate, "index": index, "num_ants": int(num_ants)})
        for index, candidate in enumerate(CANDIDATES)
    ]


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
            returncode = process.poll()
            if returncode is not None:
                return int(returncode)
            if stop_file is not None and stop_file.exists():
                terminate_process(process)
                return 143
            if time.monotonic() - started > timeout_seconds:
                print("[timeout] terminating child", file=log_file, flush=True)
                terminate_process(process)
                return 124
            time.sleep(2)


def terminate_process(process: subprocess.Popen[Any]) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def write_leaderboard(summary_path: Path, output_path: Path) -> None:
    records = read_jsonl(summary_path)
    evaluated = [
        record
        for record in records
        if record.get("status") in {"alive", "evaluated"}
        and record.get("eval_metrics") is not None
    ]
    evaluated.sort(
        key=lambda record: float(record["eval_metrics"][PRIMARY_METRIC]),
        reverse=True,
    )
    lines = [
        "# 100x100 Bridge Sweep",
        "",
        f"Updated: {now_iso()}",
        "",
        "| rank | status | candidate | task | primary | frac | success | len | W&B | checkpoint |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for rank, record in enumerate(evaluated, start=1):
        metrics = record["eval_metrics"]
        lines.append(
            "| {rank} | {status} | `{name}` | `{task}` | {primary:.6g} | "
            "{frac:.3f} | {success:.3f} | {length:.1f} | {wandb} | `{checkpoint}` |".format(
                rank=rank,
                status=record["status"],
                name=record["name"],
                task=record["task"],
                primary=float(metrics[PRIMARY_METRIC]),
                frac=float(metrics["eval_mean_delivered_fraction"]),
                success=float(metrics["eval_success_rate"]),
                length=float(metrics["eval_mean_episode_length"]),
                wandb=record.get("wandb_url") or "",
                checkpoint=record.get("checkpoint") or "",
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_disk_space(project_root: Path, min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(project_root).free / 1_000_000_000
    if free_gb < float(min_free_gb):
        raise RuntimeError(
            f"free disk is below {min_free_gb:.1f} GB ({free_gb:.2f} GB available)"
        )


def default_sweep_dir() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("sweep_%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / "bridge_100x100_sweep" / stamp


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def next_candidate_index(summary_path: Path) -> int:
    records = read_jsonl(summary_path)
    if not records:
        return 0
    return max(int(record.get("index", -1)) for record in records) + 1


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def latest_wandb_url(run_dir: Path) -> str | None:
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    runs = sorted(wandb_dir.glob("run-*"))
    if not runs:
        return None
    run_id = runs[-1].name.rsplit("-", 1)[-1]
    return f"https://wandb.ai/jerefigueiredo-universidad-de-san-andr-s/{WANDB_PROJECT}/runs/{run_id}"


if __name__ == "__main__":
    main()
