#!/usr/bin/env python3
"""Render and upload fixed-run 100x100 progress videos to W&B."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = sys.executable
WANDB_PROJECT = "cool-antz"
WANDB_RUN_ID = "coolantz100x100progressvideossingle"
WANDB_RUN_NAME = "100x100 progress videos"
WANDB_STEP_OFFSET = 100_000
DEFAULT_TARGETS: dict[str, dict[str, Any]] = {
    "bridge_mid250_best": {
        "checkpoint": PROJECT_ROOT
        / "runs"
        / "bridge_100x100_sweep"
        / "sweep_20260702_222315"
        / "bridge_0005_mid_mt050_lr1e5_u10"
        / "checkpoints"
        / "bridge_0005_mid_mt050_lr1e5_u10_best.pkl",
        "move_temperature": 0.475,
        "label": "Best bridge 100x100 mid250 checkpoint",
    },
    "continue_hard375_best": {
        "checkpoint": PROJECT_ROOT
        / "runs"
        / "bridge_100x100_sweep"
        / "continue_20260702_225027"
        / "continue_0003_hard375_from_best_mt050_lr5e6_u16"
        / "checkpoints"
        / "continue_0003_hard375_from_best_mt050_lr5e6_u16_best.pkl",
        "move_temperature": 0.5,
        "label": "Best continuation 100x100 hard375 checkpoint",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("render",))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--target-name")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--seed-offset-start", type=int, default=10_000_000)
    parser.add_argument("--seed-step", type=int, default=1_000)
    parser.add_argument("--render-timeout-seconds", type=int, default=240)
    parser.add_argument("--mem-fraction", default="0.28")
    parser.add_argument("--keep-local-videos", type=int, default=8)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-run-id", default=WANDB_RUN_ID)
    parser.add_argument("--wandb-run-name", default=WANDB_RUN_NAME)
    parser.add_argument(
        "--wandb-finish-each-cycle",
        action="store_true",
        help="Close and reopen the same W&B run after each upload so media/table pointers flush.",
    )
    parser.add_argument("--render-style", default="sprite")
    parser.add_argument("--show-vision", action="store_true")
    parser.add_argument("--target", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "MOVE_TEMP"))
    args = parser.parse_args()

    if args.worker == "render":
        if args.checkpoint is None or args.output is None or args.metadata_json is None:
            parser.error("--worker render requires --checkpoint, --output, and --metadata-json")
        worker_render(args)
        return

    parent_loop(args)


def parent_loop(args: argparse.Namespace) -> None:
    targets = parse_targets(args.target)
    python = os.environ.get("ANTZ_PYTHON", DEFAULT_PYTHON)
    out_dir = args.out_dir.resolve()
    renders_dir = out_dir / "renders"
    logs_dir = out_dir / "logs"
    for child in (renders_dir, logs_dir):
        child.mkdir(parents=True, exist_ok=True)
    stop_file = out_dir / "STOP"
    summary_path = out_dir / "summary.jsonl"
    resume_from_cycle = latest_recorded_cycle(summary_path)
    write_json(
        out_dir / "state.json",
        {
            "started_at": now_iso(),
            "interval_seconds": int(args.interval_seconds),
            "max_frames": int(args.max_frames),
            "tile_size": int(args.tile_size),
            "resume_from_cycle": int(resume_from_cycle),
            "wandb_project": args.wandb_project,
            "wandb_run_id": args.wandb_run_id,
            "wandb_run_name": args.wandb_run_name,
            "targets": {
                name: {
                    "checkpoint": str(target["checkpoint"]),
                    "move_temperature": float(target["move_temperature"]),
                    "label": target.get("label"),
                }
                for name, target in targets.items()
            },
        },
    )
    print(f"[videos] dir={out_dir}", flush=True)
    print(f"[videos] stop_file={stop_file}", flush=True)

    wandb_run = None if args.wandb_finish_each_cycle else init_wandb(args, out_dir=out_dir, targets=targets)
    try:
        cycle = resume_from_cycle
        cycles_this_run = 0
        while args.max_cycles is None or cycles_this_run < int(args.max_cycles):
            if stop_file.exists():
                print("[videos] STOP file found; exiting.", flush=True)
                break
            cycle += 1
            cycles_this_run += 1
            cycle_started = time.monotonic()
            rendered_records: list[dict[str, Any]] = []
            for target_index, (target_name, target) in enumerate(targets.items()):
                if stop_file.exists():
                    break
                seed_offset = (
                    int(args.seed_offset_start)
                    + (cycle - 1) * int(args.seed_step)
                    + target_index
                )
                record = render_target(
                    args=args,
                    python=python,
                    cycle=cycle,
                    seed_offset=seed_offset,
                    target_name=target_name,
                    target=target,
                    renders_dir=renders_dir,
                    logs_dir=logs_dir,
                    stop_file=stop_file,
                )
                append_jsonl(summary_path, record)
                if record["status"] == "rendered":
                    rendered_records.append(record)
                    prune_old_videos(renders_dir, keep=int(args.keep_local_videos))
            if rendered_records:
                if wandb_run is None:
                    wandb_run = init_wandb(args, out_dir=out_dir, targets=targets)
                upload_cycle_videos(
                    wandb_run=wandb_run,
                    cycle=cycle,
                    records=rendered_records,
                    summary_path=summary_path,
                    archive_limit=int(args.keep_local_videos),
                )
                if args.wandb_finish_each_cycle and wandb_run is not None:
                    print(f"[videos] finishing wandb upload cycle={cycle}", flush=True)
                    wandb_run.finish()
                    wandb_run = None
            elapsed = time.monotonic() - cycle_started
            sleep_for = max(0.0, float(args.interval_seconds) - elapsed)
            if args.max_cycles is not None and cycles_this_run >= int(args.max_cycles):
                break
            print(
                f"[videos] cycle={cycle} elapsed={elapsed:.1f}s sleep={sleep_for:.1f}s",
                flush=True,
            )
            sleep_until_or_stop(stop_file, sleep_for)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def render_target(
    *,
    args: argparse.Namespace,
    python: str,
    cycle: int,
    seed_offset: int,
    target_name: str,
    target: dict[str, Any],
    renders_dir: Path,
    logs_dir: Path,
    stop_file: Path,
) -> dict[str, Any]:
    checkpoint = Path(target["checkpoint"])
    if not checkpoint.exists():
        return {
            "status": "missing_checkpoint",
            "cycle": cycle,
            "target": target_name,
            "checkpoint": str(checkpoint),
            "finished_at": now_iso(),
        }
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = renders_dir / f"cycle_{cycle:06d}_{target_name}_{stamp}.mp4"
    metadata = output.with_suffix(".json")
    log_path = logs_dir / f"{output.stem}.log"
    print(
        "[videos] render "
        f"cycle={cycle} target={target_name} temp={float(target['move_temperature']):g} "
        f"seed_offset={seed_offset}",
        flush=True,
    )
    started = time.monotonic()
    code = run_render_child(
        python=python,
        args=args,
        checkpoint=checkpoint,
        output=output,
        metadata=metadata,
        log_path=log_path,
        target_name=target_name,
        seed_offset=seed_offset,
        move_temperature=float(target["move_temperature"]),
        stop_file=stop_file,
    )
    record = {
        "status": "rendered" if code == 0 and output.exists() else "render_failed",
        "cycle": int(cycle),
        "target": target_name,
        "checkpoint": str(checkpoint),
        "move_temperature": float(target["move_temperature"]),
        "seed_offset": int(seed_offset),
        "video": str(output),
        "metadata": str(metadata),
        "log": str(log_path),
        "returncode": int(code),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "finished_at": now_iso(),
    }
    if metadata.exists():
        record["render_metadata"] = load_json(metadata)
    print(
        "[videos] result "
        f"cycle={cycle} target={target_name} status={record['status']} "
        f"code={code} video={output}",
        flush=True,
    )
    return record


def run_render_child(
    *,
    python: str,
    args: argparse.Namespace,
    checkpoint: Path,
    output: Path,
    metadata: Path,
    log_path: Path,
    target_name: str,
    seed_offset: int,
    move_temperature: float,
    stop_file: Path,
) -> int:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "PYTHONUNBUFFERED": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(args.mem_fraction),
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "MPLBACKEND": "Agg",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "SDL_VIDEODRIVER": "dummy",
        }
    )
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [
                python,
                str(Path(__file__).resolve()),
                "--worker",
                "render",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--metadata-json",
                str(metadata),
                "--target-name",
                target_name,
                "--max-frames",
                str(int(args.max_frames)),
                "--tile-size",
                str(int(args.tile_size)),
                "--seed-offset-start",
                str(int(seed_offset)),
                "--render-style",
                str(args.render_style),
                "--mem-fraction",
                str(args.mem_fraction),
                "--target",
                target_name,
                str(checkpoint),
                str(float(move_temperature)),
                *(["--show-vision"] if bool(args.show_vision) else []),
            ],
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        started = time.monotonic()
        while True:
            code = process.poll()
            if code is not None:
                return int(code)
            if output.exists() and metadata.exists():
                terminate_process_group(process)
                return 0
            if stop_file.exists():
                terminate_process_group(process)
                return 143
            if time.monotonic() - started > int(args.render_timeout_seconds):
                handle.write(f"\nTIMEOUT after {int(args.render_timeout_seconds)} seconds\n")
                handle.flush()
                terminate_process_group(process)
                return 124
            time.sleep(2)


def worker_render(args: argparse.Namespace) -> None:
    from ant_byte_env.rendering import render_checkpoint

    targets = parse_targets(args.target)
    target = targets.get(str(args.target_name))
    if target is None:
        raise KeyError(f"target {args.target_name!r} was not supplied to render worker")
    started = time.monotonic()
    rendered = render_checkpoint(
        args.checkpoint.resolve(),
        args.output.resolve(),
        backend="jax",
        seed_offset=int(args.seed_offset_start),
        show_vision=bool(args.show_vision),
        max_frames=int(args.max_frames),
        tile_size=int(args.tile_size),
        policy_temperature=1.0,
        action_mode="sampled_move_greedy_write",
        move_temperature=float(target["move_temperature"]),
        write_temperature=1.0,
        render_style=str(args.render_style),
    )
    write_json(
        args.metadata_json,
        {
            "target": str(args.target_name),
            "checkpoint": str(args.checkpoint.resolve()),
            "output": str(rendered),
            "seed_offset": int(args.seed_offset_start),
            "max_frames": int(args.max_frames),
            "tile_size": int(args.tile_size),
            "move_temperature": float(target["move_temperature"]),
            "write_temperature": 1.0,
            "render_style": str(args.render_style),
            "show_vision": bool(args.show_vision),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "finished_at": now_iso(),
            "size_bytes": int(rendered.stat().st_size),
        },
    )


def init_wandb(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    targets: dict[str, dict[str, Any]],
) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        name=args.wandb_run_name,
        group="100x100-progress-videos",
        tags=["100x100", "progress-videos", "fixed-upload-run"],
        notes=(
            "Fixed W&B run for 100x100 progress videos. Training/eval runs stay "
            "separate; this run only collects periodic rendered rollouts."
        ),
        dir=str(out_dir),
        mode=args.wandb_mode,
        resume="allow",
        config={
            "interval_seconds": int(args.interval_seconds),
            "max_frames": int(args.max_frames),
            "tile_size": int(args.tile_size),
            "targets": {
                name: {
                    "checkpoint": str(target["checkpoint"]),
                    "move_temperature": float(target["move_temperature"]),
                    "label": target.get("label"),
                }
                for name, target in targets.items()
            },
        },
    )
    print(f"[videos] wandb_url={run.url}", flush=True)
    return run


def upload_cycle_videos(
    *,
    wandb_run: Any | None,
    cycle: int,
    records: list[dict[str, Any]],
    summary_path: Path,
    archive_limit: int,
) -> None:
    if wandb_run is None:
        return
    import wandb

    video_table = build_video_archive_table(
        wandb,
        summary_path=summary_path,
        limit=archive_limit,
    )
    if video_table is None:
        return
    payload: dict[str, Any] = {
        "progress_videos": video_table,
        "progress_video_cycle": int(cycle),
        "progress_video_targets": len(records),
    }
    wandb_run.log(payload, step=wandb_step_for_cycle(cycle))
    wandb_run.summary["latest_video_cycle"] = int(cycle)
    wandb_run.summary["latest_video_wandb_step"] = wandb_step_for_cycle(cycle)
    wandb_run.summary["latest_video_targets"] = [
        str(record["target"]) for record in records
    ]
    wandb_run.summary["latest_video_paths"] = [
        str(record["video"]) for record in records
    ]


def build_video_archive_table(
    wandb_module: Any,
    *,
    summary_path: Path,
    limit: int,
) -> Any | None:
    records = [
        record
        for record in read_jsonl(summary_path)
        if record.get("status") == "rendered" and Path(str(record.get("video", ""))).exists()
    ]
    if not records:
        return None
    table = wandb_module.Table(
        columns=[
            "cycle",
            "target",
            "move_temperature",
            "seed_offset",
            "finished_at",
            "video",
        ]
    )
    for record in records[-max(1, int(limit)) :]:
        table.add_data(
            int(record["cycle"]),
            str(record["target"]),
            float(record["move_temperature"]),
            int(record["seed_offset"]),
            str(record["finished_at"]),
            wandb_module.Video(str(Path(record["video"])), fps=8, format="mp4"),
        )
    return table


def parse_targets(values: list[list[str]] | None) -> dict[str, dict[str, Any]]:
    if not values:
        return {
            name: {**target, "checkpoint": Path(target["checkpoint"]).resolve()}
            for name, target in DEFAULT_TARGETS.items()
        }
    targets: dict[str, dict[str, Any]] = {}
    for name, checkpoint, move_temp in values:
        targets[str(name)] = {
            "checkpoint": resolve_path(Path(checkpoint)),
            "move_temperature": float(move_temp),
            "label": str(name),
        }
    return targets


def prune_old_videos(renders_dir: Path, *, keep: int) -> None:
    if keep <= 0:
        return
    videos = sorted(renders_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    for video in videos[:-keep]:
        metadata = video.with_suffix(".json")
        for path in (video, metadata):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def sleep_until_or_stop(stop_file: Path, seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if stop_file.exists():
            return
        time.sleep(min(5.0, deadline - time.monotonic()))


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
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


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_out_dir() -> Path:
    return PROJECT_ROOT / "runs" / "bridge_100x100_sweep" / "progress_videos_single"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_recorded_cycle(path: Path) -> int:
    if not path.exists():
        return 0
    latest = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            latest = max(latest, int(record.get("cycle", 0)))
        except (TypeError, ValueError):
            continue
    return latest


def wandb_step_for_cycle(cycle: int) -> int:
    return WANDB_STEP_OFFSET + int(cycle)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


if __name__ == "__main__":
    main()
