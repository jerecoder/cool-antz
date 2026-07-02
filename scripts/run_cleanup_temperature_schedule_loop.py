#!/usr/bin/env python3
"""Continuously probe eval-only cleanup temperature schedules.

The parent process intentionally avoids importing JAX. Each setting is evaluated
in a fresh child process through ``run_overnight_efficiency_sweep.py`` so GPU
memory is returned between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from run_deployment_action_sweep import (
    BASELINE_CHECKPOINT,
    DEFAULT_PYTHON,
    MIN_DELIVERED_FRACTION,
    MIN_SUCCESS_RATE,
    PRIMARY_METRIC,
    PROJECT_ROOT,
    gpu_compute_rows,
    run_eval,
)


RENDER_WORKER = PROJECT_ROOT / "scripts" / "run_overnight_efficiency_sweep.py"
WANDB_ENTITY = "jerefigueiredo-universidad-de-san-andr-s"
WANDB_PROJECT = "cool-antz"
SCHEDULES: tuple[dict[str, float | None | str], ...] = (
    {
        "name": "fixed_0525",
        "base_move_temperature": 0.525,
        "cleanup_move_temperature": None,
        "cleanup_fraction_threshold": 0.95,
    },
    {
        "name": "cleanup0475_at095",
        "base_move_temperature": 0.525,
        "cleanup_move_temperature": 0.475,
        "cleanup_fraction_threshold": 0.95,
    },
    {
        "name": "cleanup050_at095",
        "base_move_temperature": 0.525,
        "cleanup_move_temperature": 0.50,
        "cleanup_fraction_threshold": 0.95,
    },
    {
        "name": "stall_cleanup050_frac090_pat100",
        "base_move_temperature": 0.525,
        "cleanup_move_temperature": None,
        "cleanup_fraction_threshold": 0.95,
        "stall_cleanup_move_temperature": 0.50,
        "stall_cleanup_fraction_threshold": 0.90,
        "stall_cleanup_patience_steps": 100,
    },
    {
        "name": "stall_cleanup0475_frac090_pat100",
        "base_move_temperature": 0.525,
        "cleanup_move_temperature": None,
        "cleanup_fraction_threshold": 0.95,
        "stall_cleanup_move_temperature": 0.475,
        "stall_cleanup_fraction_threshold": 0.90,
        "stall_cleanup_patience_steps": 100,
    },
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "overnight_efficiency_sweep"
    / "sweep_20260702_003029"
    / "cleanup_schedule_loop"
)


def init_wandb_run(args: argparse.Namespace, out_dir: Path) -> Any | None:
    if args.wandb_mode == "disabled":
        print("[wandb] disabled", flush=True)
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required for logged overnight runs.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    run_id_path = out_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = stable_wandb_id(f"cleanup-schedule-{out_dir.parent.name}")
        run_id_path.write_text(run_id + "\n", encoding="utf-8")

    os.environ.setdefault("WANDB_SILENT", "true")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name,
        id=run_id,
        resume="allow",
        mode=args.wandb_mode,
        tags=[
            "overnight-efficiency-sweep",
            "50x50",
            "60-ants",
            "deployment-eval",
            "cleanup-temperature-schedule",
            "memory-safe-child",
        ],
        config={
            "checkpoint": str(BASELINE_CHECKPOINT),
            "episodes": int(args.episodes),
            "start_seed_offset": int(args.start_seed_offset),
            "seed_step": int(args.seed_step),
            "schedules": list(SCHEDULES),
            "primary_metric": PRIMARY_METRIC,
            "min_delivered_fraction": MIN_DELIVERED_FRACTION,
            "min_success_rate": MIN_SUCCESS_RATE,
        },
    )
    print(f"[wandb] run={run.url}", flush=True)
    return run


def stable_wandb_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return normalized.strip("-")[:128] or "cleanup-schedule-loop"


def sync_existing_results_to_wandb(
    *,
    results_dir: Path,
    leaderboard_path: Path,
    sync_state_path: Path,
    wandb_run: Any | None,
) -> None:
    if wandb_run is None:
        return
    synced = load_synced_keys(sync_state_path)
    records = load_result_records(results_dir)
    changed = False
    for record in sorted(
        records,
        key=lambda item: (int(item["seed_offset"]), str(item["schedule_name"])),
    ):
        key = Path(str(record["path"])).stem
        if key in synced:
            continue
        log_record_to_wandb(wandb_run, record, step=len(synced))
        synced.add(key)
        changed = True
    if leaderboard_path.exists():
        wandb_run.save(str(leaderboard_path), base_path=str(leaderboard_path.parent))
        wandb_run.summary["leaderboard_path"] = str(leaderboard_path)
    if changed:
        write_json(sync_state_path, sorted(synced))


def load_synced_keys(sync_state_path: Path) -> set[str]:
    if not sync_state_path.exists():
        return set()
    return set(json.loads(sync_state_path.read_text(encoding="utf-8")))


def log_record_to_wandb(wandb_run: Any, record: dict[str, Any], *, step: int) -> None:
    schedule = str(record["schedule_name"])
    safe_schedule = stable_wandb_id(schedule)
    payload = {
        "eval_index": int(step),
        "seed_offset": int(record["seed_offset"]),
        "episodes": int(record["episodes"]),
        "schedule_name": schedule,
        "latest/primary": float(record[PRIMARY_METRIC]),
        "latest/delivered_fraction": float(record["eval_mean_delivered_fraction"]),
        "latest/success_rate": float(record["eval_success_rate"]),
        "latest/episode_length": float(record["eval_mean_episode_length"]),
        "latest/ant_steps_per_delivered_food": float(
            record["eval_mean_ant_steps_per_delivered_food"]
        ),
        f"schedule/{safe_schedule}/primary": float(record[PRIMARY_METRIC]),
        f"schedule/{safe_schedule}/delivered_fraction": float(
            record["eval_mean_delivered_fraction"]
        ),
        f"schedule/{safe_schedule}/success_rate": float(record["eval_success_rate"]),
        f"schedule/{safe_schedule}/episode_length": float(
            record["eval_mean_episode_length"]
        ),
        f"schedule/{safe_schedule}/ant_steps_per_delivered_food": float(
            record["eval_mean_ant_steps_per_delivered_food"]
        ),
    }
    wandb_run.log(payload, step=step)
    wandb_run.summary[f"{safe_schedule}/latest_primary"] = float(record[PRIMARY_METRIC])
    wandb_run.summary[f"{safe_schedule}/latest_fraction"] = float(
        record["eval_mean_delivered_fraction"]
    )
    wandb_run.summary[f"{safe_schedule}/latest_success"] = float(
        record["eval_success_rate"]
    )


def maybe_render_regular_video(
    *,
    args: argparse.Namespace,
    python: str,
    seed_offset: int,
    renders_dir: Path,
    logs_dir: Path,
    wandb_run: Any | None,
) -> None:
    video_every = int(args.video_every_seeds)
    if video_every <= 0:
        return
    seed_index = (int(seed_offset) - int(args.start_seed_offset)) // int(args.seed_step)
    if seed_index < 0 or seed_index % video_every != 0:
        return

    output_path = renders_dir / (
        f"regular_fixed0525_seed{int(seed_offset)}_"
        f"tile{int(args.video_tile_size)}.mp4"
    )
    result_path = output_path.with_suffix(".json")
    log_path = logs_dir / f"{output_path.stem}.render.log"
    if not output_path.exists() or not result_path.exists():
        print(
            "[video] "
            f"seed_offset={int(seed_offset)} output={output_path}",
            flush=True,
        )
        code = run_render(
            python=python,
            output_path=output_path,
            result_path=result_path,
            log_path=log_path,
            seed_offset=int(seed_offset),
            max_frames=int(args.video_max_frames),
            tile_size=int(args.video_tile_size),
            timeout_seconds=int(args.video_timeout_seconds),
        )
        if code != 0:
            print(f"[video] failed returncode={code} log={log_path}", flush=True)
            return
    upload_regular_video_to_wandb(
        wandb_run=wandb_run,
        video_path=output_path,
        metadata_path=result_path,
        seed_offset=int(seed_offset),
    )


def run_render(
    *,
    python: str,
    output_path: Path,
    result_path: Path,
    log_path: Path,
    seed_offset: int,
    max_frames: int,
    tile_size: int,
    timeout_seconds: int,
) -> int:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "PYTHONUNBUFFERED": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.32",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "WANDB_SILENT": "true",
            "MPLBACKEND": "Agg",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                python,
                str(RENDER_WORKER),
                "--worker",
                "render",
                "--checkpoint",
                str(BASELINE_CHECKPOINT),
                "--output",
                str(output_path),
                "--result-json",
                str(result_path),
                "--render-policy-temperature",
                "0.525",
                "--render-action-mode",
                "sampled_move_greedy_write",
                "--render-move-temperature",
                "0.525",
                "--render-write-temperature",
                "1.0",
                "--render-seed-offset",
                str(int(seed_offset)),
                "--render-max-frames",
                str(int(max_frames)),
                "--render-tile-size",
                str(int(tile_size)),
            ],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        while True:
            code = process.poll()
            if code is not None:
                wait_for_render_gpu_to_clear(seconds=20)
                return int(code)
            if time.monotonic() - started > timeout_seconds:
                log_file.write(f"\nTIMEOUT after {timeout_seconds} seconds\n")
                log_file.flush()
                terminate_render_process_group(process)
                wait_for_render_gpu_to_clear(seconds=20)
                return 124
            time.sleep(5)


def upload_regular_video_to_wandb(
    *,
    wandb_run: Any | None,
    video_path: Path,
    metadata_path: Path,
    seed_offset: int,
) -> None:
    if wandb_run is None or not video_path.exists():
        return
    import wandb

    key = f"videos/regular_fixed0525_seed{int(seed_offset)}"
    wandb_run.log({key: wandb.Video(str(video_path), fps=8, format="mp4")})
    wandb_run.save(str(video_path), base_path=str(video_path.parent))
    if metadata_path.exists():
        wandb_run.save(str(metadata_path), base_path=str(metadata_path.parent))
    wandb_run.summary["latest_regular_video"] = str(video_path)
    wandb_run.summary["latest_regular_video_seed_offset"] = int(seed_offset)


def terminate_render_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, 15)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(process.pid, 9)
    except ProcessLookupError:
        pass


def wait_for_render_gpu_to_clear(*, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not gpu_compute_rows():
            return
        time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--start-seed-offset", type=int, default=207_000_000)
    parser.add_argument("--seed-step", type=int, default=1_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--mem-fraction", default="0.32")
    parser.add_argument("--min-disk-free-gb", type=float, default=2.0)
    parser.add_argument("--max-evals", type=int, default=0)
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY)
    parser.add_argument("--wandb-group", default="overnight-efficiency-sweep")
    parser.add_argument("--wandb-name", default="cleanup-temperature-schedule-loop")
    parser.add_argument("--video-every-seeds", type=int, default=2)
    parser.add_argument("--video-tile-size", type=int, default=16)
    parser.add_argument("--video-max-frames", type=int, default=1200)
    parser.add_argument("--video-timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    python = os.environ.get("ANTZ_PYTHON", DEFAULT_PYTHON)
    out_dir = args.out_dir
    results_dir = out_dir / "results"
    logs_dir = out_dir / "logs"
    renders_dir = out_dir / "renders"
    stop_file = out_dir / "STOP"
    summary_path = out_dir / "summary.jsonl"
    leaderboard_path = out_dir / "leaderboard.md"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(BASELINE_CHECKPOINT)

    wandb_run = init_wandb_run(args, out_dir)
    print(f"[cleanup-loop] out_dir={out_dir}", flush=True)
    print(f"[cleanup-loop] stop_file={stop_file}", flush=True)
    print(f"[cleanup-loop] checkpoint={BASELINE_CHECKPOINT}", flush=True)
    print(
        "[cleanup-loop] schedules="
        + ",".join(str(schedule["name"]) for schedule in SCHEDULES),
        flush=True,
    )
    sync_existing_results_to_wandb(
        results_dir=results_dir,
        leaderboard_path=leaderboard_path,
        sync_state_path=out_dir / "wandb_synced_keys.json",
        wandb_run=wandb_run,
    )

    eval_count = 0
    seed_offset = next_seed_offset(
        args.start_seed_offset,
        args.seed_step,
        results_dir,
    )
    while args.max_evals <= 0 or eval_count < args.max_evals:
        for schedule in SCHEDULES:
            if args.max_evals > 0 and eval_count >= args.max_evals:
                break
            if stop_file.exists():
                print("[cleanup-loop] STOP file found; exiting cleanly.", flush=True)
                write_leaderboard(results_dir, leaderboard_path)
                return
            wait_for_resources(args.min_disk_free_gb)

            key = schedule_key(seed_offset, schedule)
            result_path = results_dir / f"{key}.json"
            log_path = logs_dir / f"{key}.log"

            if result_path.exists():
                metrics = load_json(result_path)
                status = "cached"
            else:
                print(
                    "[eval] "
                    f"seed_offset={seed_offset} schedule={schedule['name']} "
                    f"episodes={args.episodes}",
                    flush=True,
                )
                code = run_eval(
                    python=python,
                    checkpoint=BASELINE_CHECKPOINT,
                    result_path=result_path,
                    log_path=log_path,
                    action_mode="sampled_move_greedy_write",
                    move_temperature=float(schedule["base_move_temperature"]),
                    write_temperature=1.0,
                    cleanup_move_temperature=cleanup_temperature(schedule),
                    cleanup_fraction_threshold=float(
                        schedule["cleanup_fraction_threshold"]
                    ),
                    stall_cleanup_move_temperature=stall_cleanup_temperature(schedule),
                    stall_cleanup_fraction_threshold=float(
                        schedule.get("stall_cleanup_fraction_threshold", 0.90)
                    ),
                    stall_cleanup_patience_steps=int(
                        schedule.get("stall_cleanup_patience_steps", 100)
                    ),
                    episodes=int(args.episodes),
                    seed_offset=int(seed_offset),
                    timeout_seconds=int(args.timeout_seconds),
                    mem_fraction=str(args.mem_fraction),
                    stop_file=stop_file,
                )
                if code != 0 or not result_path.exists():
                    append_jsonl(
                        summary_path,
                        {
                            "key": key,
                            "status": "failed",
                            "returncode": code,
                            "seed_offset": seed_offset,
                            "schedule": schedule,
                            "log": str(log_path),
                        },
                    )
                    write_leaderboard(results_dir, leaderboard_path)
                    sync_existing_results_to_wandb(
                        results_dir=results_dir,
                        leaderboard_path=leaderboard_path,
                        sync_state_path=out_dir / "wandb_synced_keys.json",
                        wandb_run=wandb_run,
                    )
                    eval_count += 1
                    continue
                metrics = load_json(result_path)
                metrics = {
                    **metrics,
                    "schedule_name": str(schedule["name"]),
                    "base_move_temperature": float(
                        schedule["base_move_temperature"]
                    ),
                    "cleanup_move_temperature": cleanup_temperature(schedule),
                    "cleanup_fraction_threshold": float(
                        schedule["cleanup_fraction_threshold"]
                    ),
                    "stall_cleanup_move_temperature": stall_cleanup_temperature(schedule),
                    "stall_cleanup_fraction_threshold": float(
                        schedule.get("stall_cleanup_fraction_threshold", 0.90)
                    ),
                    "stall_cleanup_patience_steps": int(
                        schedule.get("stall_cleanup_patience_steps", 100)
                    ),
                }
                write_json(result_path, metrics)
                status = "evaluated"

            record = {
                "key": key,
                "status": status,
                "checkpoint": str(BASELINE_CHECKPOINT),
                "schedule": schedule,
                "seed_offset": int(seed_offset),
                "episodes": int(args.episodes),
                "metrics": metrics,
                "primary_value": float(metrics[PRIMARY_METRIC]),
                "meets_fraction_gate": (
                    float(metrics["eval_mean_delivered_fraction"])
                    >= MIN_DELIVERED_FRACTION
                ),
                "meets_success_gate": (
                    float(metrics["eval_success_rate"]) >= MIN_SUCCESS_RATE
                ),
            }
            append_jsonl(summary_path, record)
            write_leaderboard(results_dir, leaderboard_path)
            sync_existing_results_to_wandb(
                results_dir=results_dir,
                leaderboard_path=leaderboard_path,
                sync_state_path=out_dir / "wandb_synced_keys.json",
                wandb_run=wandb_run,
            )
            print(
                "[result] "
                f"seed={seed_offset} schedule={schedule['name']} "
                f"{PRIMARY_METRIC}={float(metrics[PRIMARY_METRIC]):.6g} "
                f"frac={float(metrics['eval_mean_delivered_fraction']):.3f} "
                f"success={float(metrics['eval_success_rate']):.3f} "
                f"len={float(metrics['eval_mean_episode_length']):.1f}",
                flush=True,
            )
            eval_count += 1
        if stop_file.exists():
            print("[cleanup-loop] STOP file found; exiting cleanly.", flush=True)
            write_leaderboard(results_dir, leaderboard_path)
            return
        maybe_render_regular_video(
            args=args,
            python=python,
            seed_offset=seed_offset,
            renders_dir=renders_dir,
            logs_dir=logs_dir,
            wandb_run=wandb_run,
        )
        seed_offset += int(args.seed_step)

    write_leaderboard(results_dir, leaderboard_path)
    sync_existing_results_to_wandb(
        results_dir=results_dir,
        leaderboard_path=leaderboard_path,
        sync_state_path=out_dir / "wandb_synced_keys.json",
        wandb_run=wandb_run,
    )
    print("[cleanup-loop] max evals reached.", flush=True)


def cleanup_temperature(schedule: dict[str, Any]) -> float | None:
    value = schedule["cleanup_move_temperature"]
    return None if value is None else float(value)


def stall_cleanup_temperature(schedule: dict[str, Any]) -> float | None:
    value = schedule.get("stall_cleanup_move_temperature")
    return None if value is None else float(value)


def schedule_key(seed_offset: int, schedule: dict[str, Any]) -> str:
    base = float(schedule["base_move_temperature"])
    cleanup = cleanup_temperature(schedule)
    stall_cleanup = stall_cleanup_temperature(schedule)
    threshold = float(schedule["cleanup_fraction_threshold"])
    if stall_cleanup is not None:
        stall_threshold = float(schedule.get("stall_cleanup_fraction_threshold", 0.90))
        patience = int(schedule.get("stall_cleanup_patience_steps", 100))
        return (
            f"seed{seed_offset}_base{base:g}_stall{stall_cleanup:g}"
            f"_frac{stall_threshold:g}_pat{patience}"
        )
    if cleanup is None:
        return f"seed{seed_offset}_fixed_mt{base:g}"
    return f"seed{seed_offset}_base{base:g}_cleanup{cleanup:g}_frac{threshold:g}"


def next_seed_offset(start: int, step: int, results_dir: Path) -> int:
    seed = int(start)
    while all(
        (results_dir / f"{schedule_key(seed, schedule)}.json").exists()
        for schedule in SCHEDULES
    ):
        seed += int(step)
    return seed


def wait_for_resources(min_disk_free_gb: float) -> None:
    while True:
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
        gpu_rows = gpu_compute_rows()
        if free_gb >= min_disk_free_gb and not gpu_rows:
            return
        print(
            "[wait] "
            f"free_gb={free_gb:.2f} gpu_compute_apps={len(gpu_rows)}",
            flush=True,
        )
        time.sleep(30)


def write_leaderboard(results_dir: Path, leaderboard_path: Path) -> None:
    records = load_result_records(results_dir)
    by_schedule: dict[str, list[dict[str, Any]]] = {}
    by_seed_schedule: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        name = str(record["schedule_name"])
        seed_offset = int(record["seed_offset"])
        by_schedule.setdefault(name, []).append(record)
        by_seed_schedule[(seed_offset, name)] = record

    fixed_by_seed = {
        seed: record
        for (seed, name), record in by_seed_schedule.items()
        if name == "fixed_0525"
    }

    lines = [
        "# Cleanup Temperature Schedule Loop",
        "",
        f"Checkpoint: `{BASELINE_CHECKPOINT}`",
        "",
        "| rank | schedule | n | mean primary | median primary | ratio vs fixed common | min frac | mean frac | mean success | mean len | gates |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows: list[dict[str, Any]] = []
    for name, schedule_records in by_schedule.items():
        primaries = [float(record[PRIMARY_METRIC]) for record in schedule_records]
        fractions = [
            float(record["eval_mean_delivered_fraction"])
            for record in schedule_records
        ]
        successes = [float(record["eval_success_rate"]) for record in schedule_records]
        lengths = [
            float(record["eval_mean_episode_length"])
            for record in schedule_records
        ]
        common_ratios: list[float] = []
        for record in schedule_records:
            fixed_record = fixed_by_seed.get(int(record["seed_offset"]))
            if fixed_record is None or fixed_record is record:
                continue
            fixed_primary = float(fixed_record[PRIMARY_METRIC])
            if fixed_primary > 0:
                common_ratios.append(float(record[PRIMARY_METRIC]) / fixed_primary)
        rows.append(
            {
                "name": name,
                "n": len(schedule_records),
                "mean_primary": statistics.fmean(primaries),
                "median_primary": statistics.median(primaries),
                "ratio_vs_fixed": (
                    statistics.fmean(common_ratios) if common_ratios else None
                ),
                "min_fraction": min(fractions),
                "mean_fraction": statistics.fmean(fractions),
                "mean_success": statistics.fmean(successes),
                "mean_length": statistics.fmean(lengths),
            }
        )

    rows.sort(
        key=lambda row: (
            row["min_fraction"] >= MIN_DELIVERED_FRACTION,
            row["mean_success"] >= MIN_SUCCESS_RATE,
            row["mean_primary"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        gates = []
        gates.append("frac" if row["min_fraction"] >= MIN_DELIVERED_FRACTION else "low_frac")
        gates.append("success" if row["mean_success"] >= MIN_SUCCESS_RATE else "low_success")
        ratio = row["ratio_vs_fixed"]
        ratio_text = "" if ratio is None else f"{ratio:.3f}"
        lines.append(
            "| {rank} | `{name}` | {n} | {mean_primary:.6g} | "
            "{median_primary:.6g} | {ratio} | {min_fraction:.3f} | "
            "{mean_fraction:.3f} | {mean_success:.3f} | {mean_length:.1f} | "
            "`{gates}` |".format(
                rank=rank,
                name=row["name"],
                n=int(row["n"]),
                mean_primary=float(row["mean_primary"]),
                median_primary=float(row["median_primary"]),
                ratio=ratio_text,
                min_fraction=float(row["min_fraction"]),
                mean_fraction=float(row["mean_fraction"]),
                mean_success=float(row["mean_success"]),
                mean_length=float(row["mean_length"]),
                gates=",".join(gates),
            )
        )
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_result_records(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        metrics = load_json(path)
        if "schedule_name" not in metrics:
            continue
        records.append({"path": str(path), **metrics})
    return records


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
