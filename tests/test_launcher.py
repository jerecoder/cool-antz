from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_launch_random_rollout_exports_video_without_window(tmp_path: Path) -> None:
    video_path = tmp_path / "rollout.mp4"

    result = subprocess.run(
        [
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
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert f"video saved to {video_path}" in result.stdout
