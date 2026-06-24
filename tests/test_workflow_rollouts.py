from __future__ import annotations

from pathlib import Path

import pytest

from ant_byte_env.workflows import rollouts as rollout_helpers


def test_rollout_suite_renders_each_checkpoint_with_stable_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_paths = [
        tmp_path / "checkpoints" / "stage_a.pkl",
        tmp_path / "checkpoints" / "stage_b.pkl",
    ]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
    captured_kwargs: list[dict[str, object]] = []
    captured_metadata: dict[str, object] = {}

    def fake_render_checkpoint(
        checkpoint: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        del checkpoint
        captured_kwargs.append(kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    def fake_create_vault_entry(**kwargs: object) -> Path:
        captured_metadata.update(dict(kwargs["metadata"]))
        return tmp_path / "vault" / "entry"

    monkeypatch.setattr(rollout_helpers, "render_checkpoint", fake_render_checkpoint)
    monkeypatch.setattr(rollout_helpers, "create_vault_entry", fake_create_vault_entry)

    result = rollout_helpers.render_rollout_suite(
        checkpoint_paths=checkpoint_paths,
        media_dir=tmp_path / "media",
        rollout_path_for_checkpoint=lambda checkpoint, media: media / f"{checkpoint.stem}.mp4",
        progress_desc="rendering",
        vault_dir=tmp_path / "vault",
        title="Preview",
        description="Notebook rollout",
        metadata={},
    )

    assert [path.name for path in result["rollout_paths"]] == ["stage_a.mp4", "stage_b.mp4"]
    assert [kwargs["seed_offset"] for kwargs in captured_kwargs] == [
        rollout_helpers.NOTEBOOK_ROLLOUT_SEED_OFFSET,
        rollout_helpers.NOTEBOOK_ROLLOUT_SEED_OFFSET + 1,
    ]
    assert captured_kwargs[0] == {
        "backend": "jax",
        "seed_offset": rollout_helpers.NOTEBOOK_ROLLOUT_SEED_OFFSET,
        "reuse_existing": False,
        "max_frames": None,
        "tile_size": rollout_helpers.NOTEBOOK_ROLLOUT_TILE_SIZE,
        "policy_temperature": rollout_helpers.NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    }
    assert captured_metadata["rollout_seed_offsets"] == [
        rollout_helpers.NOTEBOOK_ROLLOUT_SEED_OFFSET,
        rollout_helpers.NOTEBOOK_ROLLOUT_SEED_OFFSET + 1,
    ]


def test_render_jax_checkpoint_rollout_logs_single_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "model.pkl"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
    captured_render_kwargs: list[dict[str, object]] = []
    captured_vault_metadata: dict[str, object] = {}
    tracker_instances: list[object] = []

    def fake_render_checkpoint(
        checkpoint: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        assert checkpoint == checkpoint_path
        captured_render_kwargs.append(kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    def fake_create_vault_entry(**kwargs: object) -> Path:
        captured_vault_metadata.update(dict(kwargs["metadata"]))
        return tmp_path / "run" / "vault" / "entry.md"

    class FakeTracker:
        enabled = True

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.videos: list[tuple[str, Path, int | None]] = []
            self.finished = False
            tracker_instances.append(self)

        def log_video(self, key: str, path: Path, *, step=None, fps: int = 8) -> None:
            del fps
            self.videos.append((key, path, None if step is None else int(step)))

        def finish(self) -> None:
            self.finished = True

    monkeypatch.setattr(rollout_helpers, "render_checkpoint", fake_render_checkpoint)
    monkeypatch.setattr(rollout_helpers, "create_vault_entry", fake_create_vault_entry)
    monkeypatch.setattr(rollout_helpers, "WandbTracker", FakeTracker)

    result = rollout_helpers.render_jax_checkpoint_rollout(
        run_dir=tmp_path / "run",
        checkpoint_path=checkpoint_path,
        media_dir=tmp_path / "run" / "media",
        rollout_filename="policy.mp4",
        title="Policy",
        description="Single checkpoint rollout.",
        metadata={"stage": "25x25"},
        policy_temperature=0.5,
        wandb_project="cool-antz",
        wandb_group="course",
        wandb_run_name="single-rollout",
        wandb_mode="offline",
        wandb_video_key="videos/course/policy",
        wandb_step=123,
    )

    rollout_path = tmp_path / "run" / "media" / "policy.mp4"
    assert captured_render_kwargs == [
        {
            "backend": "jax",
            "reuse_existing": True,
            "max_frames": None,
            "tile_size": rollout_helpers.NOTEBOOK_ROLLOUT_TILE_SIZE,
            "policy_temperature": 0.5,
        }
    ]
    assert captured_vault_metadata["checkpoint_path"] == str(checkpoint_path)
    assert captured_vault_metadata["rollout_path"] == str(rollout_path)
    assert captured_vault_metadata["stage"] == "25x25"
    assert result["rollout_path"] == rollout_path
    assert result["vault_entry_path"] == tmp_path / "run" / "vault" / "entry.md"
    assert result["wandb_video_key"] == "videos/course/policy"
    tracker = tracker_instances[0]
    assert tracker.kwargs["project"] == "cool-antz"
    assert tracker.kwargs["group"] == "course"
    assert tracker.kwargs["name"] == "single-rollout"
    assert tracker.videos == [("videos/course/policy", rollout_path, 123)]
    assert tracker.finished is True
