#!/usr/bin/env python3
"""Evaluate 100x100 checkpoints across movement-temperature settings."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = {
    "bridge_mid250_best": (
        PROJECT_ROOT
        / "runs"
        / "bridge_100x100_sweep"
        / "sweep_20260702_222315"
        / "bridge_0005_mid_mt050_lr1e5_u10"
        / "checkpoints"
        / "bridge_0005_mid_mt050_lr1e5_u10_best.pkl"
    ),
    "continue_hard375_best": (
        PROJECT_ROOT
        / "runs"
        / "bridge_100x100_sweep"
        / "continue_20260702_225027"
        / "continue_0003_hard375_from_best_mt050_lr5e6_u16"
        / "checkpoints"
        / "continue_0003_hard375_from_best_mt050_lr5e6_u16_best.pkl"
    ),
}
PRIMARY_METRIC = "eval_mean_delivered_food_per_1000_ant_steps"
METRIC_KEYS = (
    PRIMARY_METRIC,
    "eval_mean_delivered_food",
    "eval_mean_delivered_fraction",
    "eval_success_rate",
    "eval_mean_episode_length",
    "eval_mean_steps_per_delivered_food",
    "eval_mean_ant_steps_per_delivered_food",
    "eval_mean_episode_return",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=9_000_000)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.45, 0.475, 0.50, 0.525, 0.55, 0.60],
    )
    parser.add_argument(
        "--target",
        nargs=2,
        action="append",
        metavar=("NAME", "CHECKPOINT"),
        help="Evaluate one named checkpoint. Defaults to the current 100x100 bests.",
    )
    parser.add_argument("--write-temperature", type=float, default=1.0)
    args = parser.parse_args()

    targets = parse_targets(args.target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.jsonl"
    leaderboard_path = args.out_dir / "leaderboard.md"
    write_json(
        args.out_dir / "state.json",
        {
            "started_at": now_iso(),
            "episodes": int(args.episodes),
            "seed_offset": int(args.seed_offset),
            "temperatures": [float(temp) for temp in args.temperatures],
            "write_temperature": float(args.write_temperature),
            "targets": {name: str(path) for name, path in targets.items()},
        },
    )
    print(f"[eval-grid] dir={args.out_dir}", flush=True)

    existing = completed_keys(summary_path)
    for target_name, checkpoint in targets.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for temperature in args.temperatures:
            key = (target_name, float(temperature))
            if key in existing:
                print(f"[skip] target={target_name} temp={temperature}", flush=True)
                continue
            record = evaluate_one(
                target_name=target_name,
                checkpoint=checkpoint,
                move_temperature=float(temperature),
                write_temperature=float(args.write_temperature),
                episodes=int(args.episodes),
                seed_offset=int(args.seed_offset),
            )
            append_jsonl(summary_path, record)
            write_leaderboard(summary_path, leaderboard_path)
            print(
                "[result] "
                f"target={target_name} temp={temperature:g} "
                f"delivered={record['eval_mean_delivered_food']:.1f} "
                f"frac={record['eval_mean_delivered_fraction']:.3f} "
                f"success={record['eval_success_rate']:.3f} "
                f"primary={record[PRIMARY_METRIC]:.6g}",
                flush=True,
            )
    print("[eval-grid] complete.", flush=True)


def evaluate_one(
    *,
    target_name: str,
    checkpoint: Path,
    move_temperature: float,
    write_temperature: float,
    episodes: int,
    seed_offset: int,
) -> dict[str, Any]:
    from ant_byte_env.training.jax_mappo.evaluation import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        checkpoint,
        num_episodes=episodes,
        seed_offset=seed_offset,
        action_mode="sampled_move_greedy_write",
        move_temperature=move_temperature,
        write_temperature=write_temperature,
        shuffle_positions=True,
    )
    return {
        "target": target_name,
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "seed_offset": seed_offset,
        "action_mode": "sampled_move_greedy_write",
        "move_temperature": move_temperature,
        "write_temperature": write_temperature,
        "finished_at": now_iso(),
        **{key: float(metrics[key]) for key in METRIC_KEYS},
    }


def write_leaderboard(summary_path: Path, output_path: Path) -> None:
    records = read_jsonl(summary_path)
    records.sort(
        key=lambda record: (
            float(record["eval_success_rate"]),
            float(record["eval_mean_delivered_fraction"]),
            float(record[PRIMARY_METRIC]),
        ),
        reverse=True,
    )
    lines = [
        "# 100x100 Temperature Grid",
        "",
        f"Updated: {now_iso()}",
        "",
        "| rank | target | temp | delivered | frac | success | primary |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(records, start=1):
        lines.append(
            "| {rank} | `{target}` | {temp:.3g} | {delivered:.1f} | "
            "{frac:.3f} | {success:.3f} | {primary:.6g} |".format(
                rank=rank,
                target=record["target"],
                temp=float(record["move_temperature"]),
                delivered=float(record["eval_mean_delivered_food"]),
                frac=float(record["eval_mean_delivered_fraction"]),
                success=float(record["eval_success_rate"]),
                primary=float(record[PRIMARY_METRIC]),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_targets(values: list[list[str]] | None) -> dict[str, Path]:
    if not values:
        return {name: path for name, path in DEFAULT_TARGETS.items()}
    return {name: resolve_path(Path(path)) for name, path in values}


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def completed_keys(path: Path) -> set[tuple[str, float]]:
    return {
        (str(record["target"]), float(record["move_temperature"]))
        for record in read_jsonl(path)
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_out_dir() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("temp_grid_%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / "bridge_100x100_sweep" / stamp


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


if __name__ == "__main__":
    main()
