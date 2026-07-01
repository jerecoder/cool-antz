from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

from ant_byte_env.workflows import rollouts as rollout_helpers
from ant_byte_env.workflows import training as training_helpers


class FakeProgress:
    def update(self, value: int) -> None:
        del value

    def set_postfix(self, **kwargs: str) -> None:
        del kwargs

    def close(self) -> None:
        pass


def test_jax_checkpoint_training_reports_progress_and_checkpoint_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_train_main(argv: list[str], *, progress_callback):
        calls.append(argv)
        progress_callback(
            2,
            4,
            {
                "loss": 0.25,
                "episode_return": 3.0,
                "global_step": 160.0,
            },
        )
        return {"loss": 0.25, "episode_return": 3.0, "global_step": 160.0}

    monkeypatch.setattr(
        training_helpers,
        "stage_update_progress",
        lambda label, total_updates: FakeProgress(),
    )

    result = training_helpers.run_jax_checkpoint_training(
        run_dir=tmp_path / "continuation",
        common_args=["--load-model", "source.pkl", "--reward-mode", "forage"],
        update_timesteps=80,
        global_update_cap=4,
        train_main=fake_train_main,
        progress_label="continue",
    )

    train_args = calls[0]
    assert "--load-model" in train_args
    assert train_args[train_args.index("--total-timesteps") + 1] == "320"
    assert train_args[train_args.index("--run-dir") + 1] == str(tmp_path / "continuation")
    assert train_args[train_args.index("--save-model") + 1] == str(result["checkpoint_path"])
    assert result["stage_metrics"] == [
        {
            "loss": 0.25,
            "episode_return": 3.0,
            "global_step": 160.0,
            "stage_update": 2,
            "stage_total_updates": 4,
            "global_update_cap": 4,
            "checkpoint": str(result["checkpoint_path"]),
        }
    ]


def test_checkpoint_training_videos_use_update_specific_seed_offsets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_offsets: list[int] = []
    captured_render_kwargs: list[dict[str, object]] = []

    def fake_render_checkpoint(
        checkpoint: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        assert checkpoint.exists()
        captured_offsets.append(int(kwargs["seed_offset"]))
        captured_render_kwargs.append(dict(kwargs))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    def fake_train_main(args: list[str], *, progress_callback, checkpoint_callback):
        del args
        tracker = argparse.Namespace(enabled=False)
        for update in (1, 2, 4):
            progress_callback(
                update,
                4,
                {
                    "global_step": float(update * 128),
                    "loss": 0.1,
                    "episode_return": 1.0,
                },
            )
            checkpoint_callback(
                update=update,
                metrics={"loss": 0.1},
                params={"weight": 1.0},
                opt_state={"count": 0},
                args=argparse.Namespace(save_model=tmp_path / "model.pkl"),
                central_obs_dim=3,
                actor_obs_dim=4,
                run_name="checkpoint-video-test",
                tracker=tracker,
                global_step=update * 128,
            )
        return {"global_step": 512.0, "loss": 0.1, "episode_return": 1.0}

    monkeypatch.setattr(
        training_helpers,
        "stage_update_progress",
        lambda label, total_updates: FakeProgress(),
    )
    monkeypatch.setattr(training_helpers, "render_checkpoint", fake_render_checkpoint)
    def fake_save_checkpoint(path: Path, **kwargs: object) -> None:
        del kwargs
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"checkpoint")

    fake_checkpointing = types.ModuleType("ant_byte_env.training.jax_mappo.checkpointing")
    fake_checkpointing.save_checkpoint = fake_save_checkpoint
    monkeypatch.setitem(
        sys.modules,
        "ant_byte_env.training.jax_mappo.checkpointing",
        fake_checkpointing,
    )

    result = training_helpers.run_jax_checkpoint_training(
        run_dir=tmp_path / "run",
        common_args=[],
        update_timesteps=128,
        global_update_cap=4,
        train_main=fake_train_main,
        checkpoint_video_interval_updates=2,
        checkpoint_video_render_style="big_scale_old_three_color",
        checkpoint_video_show_vision=False,
    )

    assert captured_offsets == [
        rollout_helpers.NOTEBOOK_ROLLOUT_SEED_OFFSET + 2,
        rollout_helpers.NOTEBOOK_ROLLOUT_SEED_OFFSET + 4,
    ]
    assert [path.name for path in result["checkpoint_video_paths"]] == [
        "model_update_000002_rollout.mp4",
        "model_update_000004_rollout.mp4",
    ]
    assert {kwargs["render_style"] for kwargs in captured_render_kwargs} == {
        "big_scale_old_three_color"
    }
    assert {kwargs["show_vision"] for kwargs in captured_render_kwargs} == {False}
