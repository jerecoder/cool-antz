#!/usr/bin/env python3
"""Continuously validate deployment move temperature across held-out seeds.

The parent process intentionally does not import JAX. It reuses the eval worker
from ``run_overnight_efficiency_sweep.py`` through a fresh child process for
each seed/temperature pair so GPU memory is released between jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from run_deployment_action_sweep import (
    BASELINE_CHECKPOINT,
    PRIMARY_METRIC,
    PROJECT_ROOT,
    DEFAULT_PYTHON,
    MIN_DELIVERED_FRACTION,
    MIN_SUCCESS_RATE,
    gpu_compute_rows,
    run_eval,
)


DEFAULT_TEMPERATURES = (0.50, 0.75, 0.55, 0.525, 0.475, 0.60)
DEFAULT_SWEEP_DIR = (
    PROJECT_ROOT
    / "runs"
    / "overnight_efficiency_sweep"
    / "sweep_20260702_003029"
    / "temperature_robustness_loop"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--start-seed-offset", type=int, default=7_000_000)
    parser.add_argument("--seed-step", type=int, default=1_000_000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--mem-fraction", default="0.32")
    parser.add_argument("--min-disk-free-gb", type=float, default=2.0)
    parser.add_argument("--max-evals", type=int, default=0)
    args = parser.parse_args()

    python = os.environ.get("ANTZ_PYTHON", DEFAULT_PYTHON)
    out_dir = args.out_dir
    results_dir = out_dir / "results"
    logs_dir = out_dir / "logs"
    stop_file = out_dir / "STOP"
    summary_path = out_dir / "summary.jsonl"
    leaderboard_path = out_dir / "leaderboard.md"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(BASELINE_CHECKPOINT)

    print(f"[temp-loop] out_dir={out_dir}", flush=True)
    print(f"[temp-loop] stop_file={stop_file}", flush=True)
    print(f"[temp-loop] checkpoint={BASELINE_CHECKPOINT}", flush=True)
    print(
        "[temp-loop] temperatures="
        + ",".join(f"{temperature:g}" for temperature in args.temperatures),
        flush=True,
    )

    eval_count = 0
    seed_offset = next_seed_offset(
        args.start_seed_offset,
        args.seed_step,
        results_dir,
        args.temperatures,
    )
    while args.max_evals <= 0 or eval_count < args.max_evals:
        for temperature in args.temperatures:
            if args.max_evals > 0 and eval_count >= args.max_evals:
                break
            if stop_file.exists():
                print("[temp-loop] STOP file found; exiting cleanly.", flush=True)
                write_leaderboard(results_dir, leaderboard_path)
                return
            wait_for_resources(args.min_disk_free_gb)

            key = f"seed{seed_offset}_mt{temperature:g}"
            result_path = results_dir / f"{key}.json"
            log_path = logs_dir / f"{key}.log"

            if result_path.exists():
                metrics = load_json(result_path)
                status = "cached"
            else:
                print(
                    "[eval] "
                    f"seed_offset={seed_offset} move_temp={temperature:g} "
                    f"episodes={args.episodes}",
                    flush=True,
                )
                code = run_eval(
                    python=python,
                    checkpoint=BASELINE_CHECKPOINT,
                    result_path=result_path,
                    log_path=log_path,
                    action_mode="sampled_move_greedy_write",
                    move_temperature=float(temperature),
                    write_temperature=1.0,
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
                            "move_temperature": temperature,
                            "log": str(log_path),
                        },
                    )
                    write_leaderboard(results_dir, leaderboard_path)
                    eval_count += 1
                    continue
                metrics = load_json(result_path)
                status = "evaluated"

            record = {
                "key": key,
                "status": status,
                "checkpoint": str(BASELINE_CHECKPOINT),
                "action_mode": "sampled_move_greedy_write",
                "move_temperature": float(temperature),
                "write_temperature": 1.0,
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
            print(
                "[result] "
                f"seed={seed_offset} mt={temperature:g} "
                f"{PRIMARY_METRIC}={float(metrics[PRIMARY_METRIC]):.6g} "
                f"frac={float(metrics['eval_mean_delivered_fraction']):.3f} "
                f"success={float(metrics['eval_success_rate']):.3f} "
                f"len={float(metrics['eval_mean_episode_length']):.1f}",
                flush=True,
            )
            eval_count += 1
        seed_offset += int(args.seed_step)

    write_leaderboard(results_dir, leaderboard_path)
    print("[temp-loop] max evals reached.", flush=True)


def next_seed_offset(
    start: int,
    step: int,
    results_dir: Path,
    temperatures: list[float] | tuple[float, ...],
) -> int:
    seed = int(start)
    while all(
        (results_dir / f"seed{seed}_mt{temperature:g}.json").exists()
        for temperature in temperatures
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
    by_temperature: dict[float, list[dict[str, Any]]] = {}
    by_seed_temperature: dict[tuple[int, float], dict[str, Any]] = {}
    for record in records:
        temperature = float(record["move_temperature"])
        seed_offset = int(record["seed_offset"])
        by_temperature.setdefault(temperature, []).append(record)
        by_seed_temperature[(seed_offset, temperature)] = record

    default_temperature = 0.75
    default_by_seed = {
        seed: record
        for (seed, temperature), record in by_seed_temperature.items()
        if temperature == default_temperature
    }

    lines = [
        "# Temperature Robustness Loop",
        "",
        f"Checkpoint: `{BASELINE_CHECKPOINT}`",
        "",
        "| rank | move temp | n | mean primary | median primary | ratio vs 0.75 common | min frac | mean frac | mean success | mean len | gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    rows: list[dict[str, Any]] = []
    for temperature, temp_records in by_temperature.items():
        primaries = [float(record[PRIMARY_METRIC]) for record in temp_records]
        fractions = [
            float(record["eval_mean_delivered_fraction"]) for record in temp_records
        ]
        successes = [float(record["eval_success_rate"]) for record in temp_records]
        lengths = [float(record["eval_mean_episode_length"]) for record in temp_records]
        common_ratios: list[float] = []
        for record in temp_records:
            seed_offset = int(record["seed_offset"])
            default_record = default_by_seed.get(seed_offset)
            if default_record is None:
                continue
            default_primary = float(default_record[PRIMARY_METRIC])
            if default_primary > 0:
                common_ratios.append(float(record[PRIMARY_METRIC]) / default_primary)
        rows.append(
            {
                "temperature": temperature,
                "n": len(temp_records),
                "mean_primary": statistics.fmean(primaries),
                "median_primary": statistics.median(primaries),
                "ratio_vs_default": (
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
        ratio = row["ratio_vs_default"]
        ratio_text = "" if ratio is None else f"{ratio:.3f}"
        lines.append(
            "| {rank} | {temperature:.3g} | {n} | {mean_primary:.6g} | "
            "{median_primary:.6g} | {ratio} | {min_fraction:.3f} | "
            "{mean_fraction:.3f} | {mean_success:.3f} | {mean_length:.1f} | "
            "`{gates}` |".format(
                rank=rank,
                temperature=float(row["temperature"]),
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
    for path in sorted(results_dir.glob("seed*_mt*.json")):
        metrics = load_json(path)
        stem = path.stem
        seed_text, temp_text = stem.split("_mt", maxsplit=1)
        records.append(
            {
                "path": str(path),
                "seed_offset": int(seed_text.removeprefix("seed")),
                "move_temperature": float(temp_text),
                **metrics,
            }
        )
    return records


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
