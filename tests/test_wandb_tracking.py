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
    assert init_kwargs["reinit"] == "create_new"
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


def test_wandb_tracker_finish_does_not_raise_connection_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRun:
        finish_calls = 0

        def finish(self) -> None:
            self.finish_calls += 1
            raise ConnectionResetError("Connection lost")

    fake_run = FakeRun()
    fake_wandb = types.SimpleNamespace(init=lambda **kwargs: fake_run)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_wandb)

    tracker = WandbTracker(project="cool-antz")

    with pytest.warns(RuntimeWarning, match="ConnectionResetError: Connection lost"):
        tracker.finish()

    tracker.finish()
    assert fake_run.finish_calls == 1


def test_wandb_tracker_finished_run_log_becomes_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeUsageError(Exception):
        pass

    class FakeRun:
        log_calls = 0

        def log(self, payload: dict[str, object], *, step: int | None = None) -> None:
            del payload, step
            self.log_calls += 1
            raise FakeUsageError("Run is finished. The call to `log` will be ignored.")

    fake_run = FakeRun()
    fake_wandb = types.SimpleNamespace(init=lambda **kwargs: fake_run)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_wandb)

    tracker = WandbTracker(project="cool-antz")

    with pytest.warns(RuntimeWarning, match="metric logging failed"):
        tracker.log_metrics({"loss": 1.0}, step=1)

    tracker.log_metrics({"loss": 2.0}, step=2)
    assert fake_run.log_calls == 1
    assert not tracker.enabled


def test_wandb_tracker_retries_init_after_service_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRun:
        pass

    init_calls: list[dict[str, object]] = []
    teardown_calls: list[int | None] = []
    fake_run = FakeRun()

    def fake_init(**kwargs: object) -> FakeRun:
        init_calls.append(kwargs)
        if len(init_calls) == 1:
            raise RuntimeError("MailboxClosedError")
        return fake_run

    fake_wandb = types.SimpleNamespace(
        init=fake_init,
        teardown=lambda exit_code=None: teardown_calls.append(exit_code),
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_wandb)

    with pytest.warns(RuntimeWarning, match="retrying once"):
        tracker = WandbTracker(project="cool-antz")

    assert tracker.enabled
    assert init_calls[0]["project"] == "cool-antz"
    assert init_calls[1]["project"] == "cool-antz"
    assert init_calls[1]["reinit"] == "create_new"
    assert teardown_calls == [1]


def test_wandb_tracker_init_failure_after_retry_becomes_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_init(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("MailboxClosedError")

    fake_wandb = types.SimpleNamespace(init=fake_init, teardown=lambda exit_code=None: None)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_wandb)

    with pytest.warns(RuntimeWarning) as warnings:
        tracker = WandbTracker(project="cool-antz")

    assert not tracker.enabled
    assert any("retrying once" in str(warning.message) for warning in warnings)
    assert any("continuing with W&B disabled" in str(warning.message) for warning in warnings)
    tracker.log_metrics({"loss": 1.0})
    tracker.finish()


def test_wandb_tracker_missing_package_has_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="pip install wandb"):
        WandbTracker(project="cool-antz")
