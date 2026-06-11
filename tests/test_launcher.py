from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from launch_random_rollout import parse_args


def test_launch_random_rollout_defaults_use_larger_random_map(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["launch_random_rollout.py"])

    args = parse_args()

    assert args.width == 16
    assert args.height == 16
    assert args.food_count == 24
    assert args.food_sources == 4
    assert args.tile_size == 36


def test_launch_random_rollout_exports_video_without_window(tmp_path: Path) -> None:
    video_path = tmp_path / "rollout.mp4"
    child_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE_") and key != "COVERAGE_PROCESS_START"
    }

    command = [
        sys.executable,
        "launch_random_rollout.py",
        "--no-window",
        "--video",
        str(video_path),
        "--video-fps",
        "4",
        "--seed",
        "123",
        "--max-steps",
        "3",
        "--tile-size",
        "16",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        stdout = result.stdout
    except subprocess.TimeoutExpired as exc:
        # After JAX tests initialize native runtimes, some Linux runners can
        # linger in child-process teardown even though the launcher finished.
        stdout = (exc.output or b"").decode()
        if not video_path.exists() or f"video saved to {video_path}" not in stdout:
            raise

    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert f"video saved to {video_path}" in stdout
