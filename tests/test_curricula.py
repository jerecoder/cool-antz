from __future__ import annotations

from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.curricula import stages


def test_notebook_workflows_reexports_curriculum_stage_builders() -> None:
    assert workflows.build_forage_curriculum_stages is stages.build_forage_curriculum_stages
    assert workflows.build_exploration_curriculum_stages is (
        stages.build_exploration_curriculum_stages
    )
    assert workflows.build_maze_exploration_curriculum_stages is (
        stages.build_maze_exploration_curriculum_stages
    )


def test_curriculum_stage_builders_remain_available_from_curricula_package() -> None:
    stage = stages.build_forage_curriculum_stages((4,))[0]

    assert stage["name"] == "4x4"
    assert stage["food_count"] == workflows.curriculum_food_count(4)
