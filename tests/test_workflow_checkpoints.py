from pathlib import Path

from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.curricula import stages
from ant_byte_env.workflows import checkpoints


def test_checkpoint_path_helpers_match_notebook_artifact_layout(tmp_path: Path) -> None:
    forage_stages = stages.build_forage_curriculum_stages((4,))

    assert checkpoints.forage_checkpoint_paths(tmp_path / "checkpoints", forage_stages) == [
        tmp_path / "checkpoints" / "jax_mappo_forage_stage1_4x4.pkl"
    ]
    assert checkpoints.communication_checkpoint_paths(tmp_path, (2, 3)) == [
        tmp_path / "2_bits" / "checkpoints" / "model.pkl",
        tmp_path / "3_bits" / "checkpoints" / "model.pkl",
    ]
    assert checkpoints.ant_count_checkpoint_paths(tmp_path, (2, 4)) == [
        tmp_path / "2_ants" / "checkpoints" / "model.pkl",
        tmp_path / "4_ants" / "checkpoints" / "model.pkl",
    ]


def test_notebook_workflows_reexports_checkpoint_path_helpers() -> None:
    assert workflows.forage_checkpoint_paths is checkpoints.forage_checkpoint_paths
    assert workflows.communication_checkpoint_paths is (
        checkpoints.communication_checkpoint_paths
    )
    assert workflows.ant_count_checkpoint_paths is checkpoints.ant_count_checkpoint_paths
