#!/usr/bin/env python3
"""Sweep deployment action settings for existing 50x50 / 60-ant checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/jerefigo/miniconda3/envs/tp1-rl/bin/python"
EVAL_WORKER = PROJECT_ROOT / "scripts" / "run_overnight_efficiency_sweep.py"
BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "notebooks"
    / "fl50_60ants_half_food_sw_wc_8bits_stabilize_from_60best"
    / "checkpoints"
    / "best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits_stabilized.pkl"
)
PRIMARY_METRIC = "eval_mean_delivered_food_per_1000_ant_steps"
PROMOTION_MARGIN = 1.05
MIN_DELIVERED_FRACTION = 0.97
MIN_SUCCESS_RATE = 0.65

ACTION_SETTINGS = [
    ("sampled_move_greedy_write", 0.25, 1.0),
    ("sampled_move_greedy_write", 0.35, 1.0),
    ("sampled_move_greedy_write", 0.50, 1.0),
    ("sampled_move_greedy_write", 0.65, 1.0),
    ("sampled_move_greedy_write", 0.75, 1.0),
    ("sampled_move_greedy_write", 0.90, 1.0),
    ("sampled_move_greedy_write", 1.10, 1.0),
    ("sampled_move_greedy_write", 1.30, 1.0),
    ("greedy_move_greedy_write", 1.0, 1.0),
    ("sampled_move_sampled_write", 0.50, 0.75),
    ("sampled_move_sampled_write", 0.75, 0.75),
    ("sampled_move_zero_write", 0.75, 1.0),
    ("greedy_move_zero_write", 1.0, 1.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-offset", type=int, default=2_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--mem-fraction", default="0.32")
    args = parser.parse_args()

    python = os.environ.get("ANTZ_PYTHON", DEFAULT_PYTHON)
    sweep_dir = args.sweep_dir
    out_dir = sweep_dir / "deployment_action_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "summary.jsonl"
    leaderboard_path = out_dir / "leaderboard.md"
    stop_file = out_dir / "STOP"

    checkpoints = [("baseline_stabilized_best", BASELINE_CHECKPOINT)]
    checkpoints.extend(top_candidate_checkpoints(sweep_dir, args.top_candidates))
    checkpoints = dedupe_checkpoints(checkpoints)
    print(f"[action-sweep] checkpoints={len(checkpoints)} out_dir={out_dir}", flush=True)

    baseline_key = setting_key(
        "baseline_stabilized_best",
        "sampled_move_greedy_write",
        0.75,
        1.0,
    )
    baseline_metrics: dict[str, Any] | None = None

    for label, checkpoint in checkpoints:
        if not checkpoint.exists():
            print(f"[skip] missing checkpoint label={label} path={checkpoint}", flush=True)
            continue
        for action_mode, move_temperature, write_temperature in ACTION_SETTINGS:
            if stop_file.exists():
                print("[action-sweep] STOP file found; exiting.", flush=True)
                write_leaderboard(summary_path, leaderboard_path, baseline_metrics)
                return
            key = setting_key(label, action_mode, move_temperature, write_temperature)
            result_path = out_dir / f"{key}.json"
            log_path = log_dir / f"{key}.log"
            if result_path.exists():
                metrics = load_json(result_path)
            else:
                print(
                    "[eval] "
                    f"label={label} mode={action_mode} "
                    f"move_temp={move_temperature:g} write_temp={write_temperature:g}",
                    flush=True,
                )
                code = run_eval(
                    python=python,
                    checkpoint=checkpoint,
                    result_path=result_path,
                    log_path=log_path,
                    action_mode=action_mode,
                    move_temperature=move_temperature,
                    write_temperature=write_temperature,
                    episodes=args.episodes,
                    seed_offset=args.seed_offset,
                    timeout_seconds=args.timeout_seconds,
                    mem_fraction=args.mem_fraction,
                    stop_file=stop_file,
                )
                if code != 0 or not result_path.exists():
                    append_jsonl(
                        summary_path,
                        {
                            "key": key,
                            "label": label,
                            "checkpoint": str(checkpoint),
                            "action_mode": action_mode,
                            "move_temperature": move_temperature,
                            "write_temperature": write_temperature,
                            "status": "failed",
                            "returncode": code,
                            "log": str(log_path),
                        },
                    )
                    continue
                metrics = load_json(result_path)
            record = {
                "key": key,
                "label": label,
                "checkpoint": str(checkpoint),
                "action_mode": action_mode,
                "move_temperature": move_temperature,
                "write_temperature": write_temperature,
                "status": "evaluated",
                "metrics": metrics,
                "primary_value": float(metrics[PRIMARY_METRIC]),
            }
            if key == baseline_key:
                baseline_metrics = metrics
            if baseline_metrics is not None:
                threshold = float(baseline_metrics[PRIMARY_METRIC]) * PROMOTION_MARGIN
                record["baseline_primary_value"] = float(baseline_metrics[PRIMARY_METRIC])
                record["promotion_threshold"] = threshold
                record["primary_ratio_vs_baseline"] = (
                    float(metrics[PRIMARY_METRIC])
                    / float(baseline_metrics[PRIMARY_METRIC])
                )
                record["promoted"] = (
                    float(metrics[PRIMARY_METRIC]) >= threshold
                    and float(metrics["eval_mean_delivered_fraction"])
                    >= MIN_DELIVERED_FRACTION
                    and float(metrics["eval_success_rate"]) >= MIN_SUCCESS_RATE
                )
            append_jsonl(summary_path, record)
            write_leaderboard(summary_path, leaderboard_path, baseline_metrics)
            print(
                "[result] "
                f"label={label} mode={action_mode} mt={move_temperature:g} "
                f"{PRIMARY_METRIC}={float(metrics[PRIMARY_METRIC]):.6g} "
                f"frac={float(metrics['eval_mean_delivered_fraction']):.3f} "
                f"success={float(metrics['eval_success_rate']):.3f}",
                flush=True,
            )

    write_leaderboard(summary_path, leaderboard_path, baseline_metrics)
    print("[action-sweep] complete", flush=True)


def top_candidate_checkpoints(sweep_dir: Path, limit: int) -> list[tuple[str, Path]]:
    summary_path = sweep_dir / "summary.jsonl"
    if not summary_path.exists() or limit <= 0:
        return []
    records: list[dict[str, Any]] = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "eval_metrics" in record:
                records.append(record)
    records.sort(
        key=lambda record: float(record.get("primary_value", float("-inf"))),
        reverse=True,
    )
    checkpoints: list[tuple[str, Path]] = []
    for record in records[:limit]:
        checkpoint = record.get("train_result", {}).get("final_checkpoint_path")
        if checkpoint:
            checkpoints.append((str(record["name"]), Path(checkpoint)))
    return checkpoints


def setting_key(
    label: str,
    action_mode: str,
    move_temperature: float,
    write_temperature: float,
) -> str:
    return (
        f"{label}__{action_mode}__mt{move_temperature:g}"
        f"__wt{write_temperature:g}"
    )


def dedupe_checkpoints(items: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    seen: set[Path] = set()
    deduped: list[tuple[str, Path]] = []
    for label, path in items:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append((label, path))
    return deduped


def run_eval(
    *,
    python: str,
    checkpoint: Path,
    result_path: Path,
    log_path: Path,
    action_mode: str,
    move_temperature: float,
    write_temperature: float,
    cleanup_move_temperature: float | None = None,
    cleanup_fraction_threshold: float = 0.95,
    stall_cleanup_move_temperature: float | None = None,
    stall_cleanup_fraction_threshold: float = 0.90,
    stall_cleanup_patience_steps: int = 100,
    episodes: int,
    seed_offset: int,
    timeout_seconds: int,
    mem_fraction: str,
    stop_file: Path,
) -> int:
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
    result_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        child_args = [
            python,
            str(EVAL_WORKER),
            "--worker",
            "eval",
            "--checkpoint",
            str(checkpoint),
            "--result-json",
            str(result_path),
            "--eval-episodes",
            str(int(episodes)),
            "--eval-seed-offset",
            str(int(seed_offset)),
            "--eval-action-mode",
            str(action_mode),
            "--eval-move-temperature",
            str(float(move_temperature)),
            "--eval-write-temperature",
            str(float(write_temperature)),
        ]
        if cleanup_move_temperature is not None:
            child_args.extend(
                [
                    "--eval-cleanup-move-temperature",
                    str(float(cleanup_move_temperature)),
                    "--eval-cleanup-fraction-threshold",
                    str(float(cleanup_fraction_threshold)),
                ]
            )
        if stall_cleanup_move_temperature is not None:
            child_args.extend(
                [
                    "--eval-stall-cleanup-move-temperature",
                    str(float(stall_cleanup_move_temperature)),
                    "--eval-stall-cleanup-fraction-threshold",
                    str(float(stall_cleanup_fraction_threshold)),
                    "--eval-stall-cleanup-patience-steps",
                    str(int(stall_cleanup_patience_steps)),
                ]
            )
        process = subprocess.Popen(
            child_args,
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
            if stop_file.exists():
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


def write_leaderboard(
    summary_path: Path,
    leaderboard_path: Path,
    baseline_metrics: dict[str, Any] | None,
) -> None:
    records = [
        json.loads(line)
        for line in summary_path.read_text(encoding="utf-8").splitlines()
        if summary_path.exists() and line.strip()
    ] if summary_path.exists() else []
    evaluated = [record for record in records if record.get("status") == "evaluated"]
    evaluated.sort(
        key=lambda record: float(record.get("primary_value", float("-inf"))),
        reverse=True,
    )
    baseline_value = (
        None if baseline_metrics is None else float(baseline_metrics[PRIMARY_METRIC])
    )
    lines = [
        "# Deployment Action Sweep",
        "",
        f"Baseline primary: {baseline_value if baseline_value is not None else 'pending'}",
        "",
        "| rank | promoted | label | action | move temp | write temp | primary | ratio | frac | success | len | checkpoint |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, record in enumerate(evaluated, start=1):
        metrics = record["metrics"]
        ratio = record.get("primary_ratio_vs_baseline", "")
        ratio_text = "" if ratio == "" else f"{float(ratio):.3f}"
        lines.append(
            "| {rank} | {promoted} | `{label}` | `{action}` | {move:.2f} | "
            "{write:.2f} | {primary:.6g} | {ratio} | {frac:.3f} | "
            "{success:.3f} | {length:.1f} | `{checkpoint}` |".format(
                rank=rank,
                promoted=str(bool(record.get("promoted", False))).lower(),
                label=record["label"],
                action=record["action_mode"],
                move=float(record["move_temperature"]),
                write=float(record["write_temperature"]),
                primary=float(record["primary_value"]),
                ratio=ratio_text,
                frac=float(metrics["eval_mean_delivered_fraction"]),
                success=float(metrics["eval_success_rate"]),
                length=float(metrics["eval_mean_episode_length"]),
                checkpoint=record["checkpoint"],
            )
        )
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
