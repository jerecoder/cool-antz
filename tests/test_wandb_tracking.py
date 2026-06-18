from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

from ant_byte_env.wandb_tracking import WandbTracker


def test_wandb_tracker_disabled_does_not_import_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    WandbTracker(project=None, mode="online").log_metrics({"loss": 1.0})
    WandbTracker(project="cool-antz", mode="disabled").finish()


def test_wandb_tracker_logs_metrics_and_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.logs: list[tuple[dict[str, object], int | None]] = []
            self.artifacts: list[tuple[object, list[str] | None]] = []
            self.finished = False

        def log(self, payload: dict[str, object], *, step: int | None = None) -> None:
            self.logs.append((payload, step))

        def log_artifact(self, artifact: object, *, aliases: list[str] | None = None) -> None:
            self.artifacts.append((artifact, aliases))

        def finish(self) -> None:
            self.finished = True

    class FakeVideo:
        def __init__(self, path: str, *, fps: int, format: str) -> None:
            self.path = path
            self.fps = fps
            self.format = format

    class FakeArtifact:
        def __init__(self, name: str, *, type: str) -> None:
            self.name = name
            self.type = type
            self.files: list[str] = []

        def add_file(self, path: str) -> None:
            self.files.append(path)

    fake_run = FakeRun()
    init_kwargs: dict[str, object] = {}

    def fake_init(**kwargs: object) -> FakeRun:
        init_kwargs.update(kwargs)
        return fake_run

    fake_wandb = types.SimpleNamespace(init=fake_init, Video=FakeVideo, Artifact=FakeArtifact)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_wandb)
    video_path = tmp_path / "preview.mp4"
    video_path.write_bytes(b"mp4")

    tracker = WandbTracker(
        project="cool-antz",
        entity="team",
        group="forage",
        name="run-name",
        tags=["jax", "50x50"],
        mode="offline",
        run_dir=tmp_path,
        config={"gamma": 0.99},
        notes="Testing short autocurriculum reward shaping.",
    )
    tracker.log_metrics({"loss": 0.5}, step=128)
    tracker.log_video("videos/4x4", video_path, step=128, fps=8)
    tracker.log_artifact("checkpoint", video_path, artifact_type="model", aliases=["latest"])
    tracker.finish()

    assert init_kwargs["project"] == "cool-antz"
    assert init_kwargs["entity"] == "team"
    assert init_kwargs["group"] == "forage"
    assert init_kwargs["name"] == "run-name"
    assert init_kwargs["tags"] == ["jax", "50x50"]
    assert init_kwargs["mode"] == "offline"
    assert init_kwargs["dir"] == str(tmp_path)
    assert init_kwargs["config"] == {"gamma": 0.99}
    assert init_kwargs["notes"] == "Testing short autocurriculum reward shaping."
    assert fake_run.logs[0] == ({"loss": 0.5}, 128)
    video_payload, video_step = fake_run.logs[1]
    assert video_step == 128
    assert isinstance(video_payload["videos/4x4"], FakeVideo)
    assert video_payload["videos/4x4"].path == str(video_path)
    assert video_payload["videos/4x4"].fps == 8
    assert video_payload["videos/4x4"].format == "mp4"
    artifact, aliases = fake_run.artifacts[0]
    assert isinstance(artifact, FakeArtifact)
    assert artifact.name == "checkpoint"
    assert artifact.type == "model"
    assert artifact.files == [str(video_path)]
    assert aliases == ["latest"]
    assert fake_run.finished is True


def test_wandb_tracker_missing_package_has_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="pip install wandb"):
        WandbTracker(project="cool-antz")
