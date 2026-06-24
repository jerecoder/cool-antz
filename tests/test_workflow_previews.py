import pytest

from ant_byte_env.workflows import previews
from ant_byte_env.workflows.rollouts import NOTEBOOK_ROLLOUT_SEED_OFFSET


def test_wandb_preview_options_validate_requested_stage_names() -> None:
    stages = [{"name": "4x4"}, {"name": "8x8"}]

    assert previews.validate_wandb_preview_stage_names(stages, ["8x8"]) == ("8x8",)
    assert previews.wandb_preview_stage_enabled("8x8", ("8x8",))
    assert not previews.wandb_preview_stage_enabled("4x4", ("8x8",))

    with pytest.raises(ValueError, match="unknown: 5x5"):
        previews.validate_wandb_preview_stage_names(stages, ["5x5"])


def test_wandb_video_rollout_count_and_seed_validation() -> None:
    assert previews.validate_wandb_video_rollout_count("2") == 2
    assert previews.wandb_video_seed_offset_base(123) == 123

    with pytest.raises(ValueError, match="positive integer"):
        previews.validate_wandb_video_rollout_count(0)
    with pytest.raises(ValueError, match="non-negative"):
        previews.wandb_video_seed_offset_base(-1)


def test_wandb_preview_key_names_single_and_multiple_rollouts() -> None:
    assert (
        previews.wandb_preview_video_key(
            prefix="videos/forage",
            stage_name="4x4",
            preview_index=0,
            preview_count=1,
        )
        == "videos/forage/4x4"
    )
    assert (
        previews.wandb_preview_video_key(
            prefix="videos/forage",
            stage_name="4x4",
            preview_index=1,
            preview_count=2,
        )
        == "videos/forage/4x4/rollout_02"
    )


def test_wandb_seed_base_uses_notebook_compatible_time_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(previews.time, "time_ns", lambda: 123_456_789)

    assert previews.wandb_video_seed_offset_base(None) == (
        NOTEBOOK_ROLLOUT_SEED_OFFSET + 123_456_789
    )


def test_render_forage_wandb_previews_uses_stage_specific_paths_and_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "jax_mappo_forage_stage1_8x8.pkl"
    captured_calls: list[tuple[object, object, dict[str, object]]] = []

    def fake_render_checkpoint(checkpoint, output_path, **kwargs):
        captured_calls.append((checkpoint, output_path, kwargs))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    monkeypatch.setattr(previews, "render_checkpoint", fake_render_checkpoint)

    paths = previews.render_forage_wandb_previews(
        checkpoint_path=checkpoint_path,
        checkpoint_dir=tmp_path / "checkpoints",
        stage_index=3,
        max_frames=64,
        policy_temperature=0.5,
        rollout_count=2,
        seed_offset_base=10_000,
    )

    assert [path.name for path in paths] == [
        "jax_mappo_forage_stage1_8x8_preview_01.mp4",
        "jax_mappo_forage_stage1_8x8_preview_02.mp4",
    ]
    assert [call[2]["seed_offset"] for call in captured_calls] == [10_004, 10_005]
    assert {call[2]["backend"] for call in captured_calls} == {"jax"}
