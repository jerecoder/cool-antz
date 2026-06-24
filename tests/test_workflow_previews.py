import pytest

from ant_byte_env import notebook_workflows as workflows
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
    monkeypatch.setattr(workflows.time, "time_ns", lambda: 123_456_789)

    assert previews.wandb_video_seed_offset_base(None) == (
        NOTEBOOK_ROLLOUT_SEED_OFFSET + 123_456_789
    )


def test_notebook_workflows_reexports_preview_helpers() -> None:
    assert workflows._wandb_preview_enabled is previews.wandb_preview_enabled
    assert workflows._wandb_preview_stage_enabled is previews.wandb_preview_stage_enabled
    assert workflows._validate_wandb_preview_stage_names is (
        previews.validate_wandb_preview_stage_names
    )
    assert workflows._validate_wandb_video_rollout_count is (
        previews.validate_wandb_video_rollout_count
    )
    assert workflows._wandb_video_seed_offset_base is (
        previews.wandb_video_seed_offset_base
    )
    assert workflows._wandb_preview_video_key is previews.wandb_preview_video_key
