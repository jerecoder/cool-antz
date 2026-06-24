from __future__ import annotations

import pytest

from ant_byte_env.research.config import (
    research_evaluation_config,
    research_stages,
    research_wandb_config,
    wandb_argv,
)


def test_research_wandb_config_merges_overrides_and_dedupes_tags() -> None:
    config = research_wandb_config(
        matrix={
            "wandb": {"tags": ["course", "shared"], "entity": "team"},
            "wandb_video": {"stage_names": ["25x25"], "max_frames": 128},
        },
        entry={
            "family": "combined_reward_capacity",
            "wandb": {"tags": ["shared", "final"], "group": "archive"},
            "wandb_video": {"max_frames": 64},
        },
        run_id="DISTANCE_CAP4",
        project_override="mit-course",
        mode_override="disabled",
    )

    assert config["project"] == "mit-course"
    assert config["entity"] == "team"
    assert config["group"] == "archive"
    assert config["mode"] == "disabled"
    assert config["name"] == "research-loop-DISTANCE_CAP4"
    assert config["tags"] == [
        "course",
        "shared",
        "final",
        "combined_reward_capacity",
        "DISTANCE_CAP4",
    ]
    assert config["video_stage_names"] == ["25x25"]
    assert config["video_max_frames"] == 64


def test_research_evaluation_config_normalizes_action_modes() -> None:
    config = research_evaluation_config(
        matrix={
            "evaluation": {
                "sampled_episodes": 5,
                "action_modes": [
                    "greedy",
                    {
                        "name": "warm_sampled",
                        "action_mode": "sampled",
                        "num_episodes": 3,
                        "move_temperature": 0.75,
                    },
                ],
            }
        },
        entry={"evaluation": {"deterministic_episodes": 2}},
    )

    assert config["deterministic_episodes"] == 2
    assert config["sampled_episodes"] == 5
    assert config["seed_offset"] == 1_000_000
    assert config["shuffle_positions"] is True
    assert config["action_modes"] == [
        {"name": "greedy", "action_mode": "greedy", "episodes": 5},
        {
            "name": "warm_sampled",
            "action_mode": "sampled",
            "episodes": 3,
            "move_temperature": 0.75,
        },
    ]


def test_research_stages_applies_profile_and_stage_overrides() -> None:
    stages = research_stages(
        matrix={
            "default_stage_sizes": [5, 10],
            "stage_training_profile": [
                {"max_size": 5, "global_update_cap": 2, "num_steps": 16},
                {"max_size": 10, "global_update_cap": 4, "gamma": 0.99},
            ],
        },
        entry={"stage_overrides": {"random_food": True}},
        global_update_cap=1,
        update_cap_overridden=False,
    )

    assert [stage["width"] for stage in stages] == [5, 10]
    assert stages[0]["global_update_cap"] == 2
    assert stages[0]["num_steps"] == 16
    assert stages[0]["random_food"] is True
    assert stages[1]["global_update_cap"] == 4
    assert stages[1]["gamma"] == 0.99


def test_research_stages_rejects_invalid_size_schedule() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        research_stages(
            matrix={"default_stage_sizes": [10, 5]},
            entry={},
            global_update_cap=1,
            update_cap_overridden=False,
        )


def test_wandb_argv_renders_optional_tags_after_core_fields() -> None:
    assert wandb_argv(
        {
            "project": "mit-course",
            "entity": "team",
            "group": "archive",
            "name": "research-loop-DISTANCE_CAP4",
            "mode": "disabled",
            "tags": ["course", "DISTANCE_CAP4"],
        }
    ) == [
        "--wandb-project",
        "mit-course",
        "--wandb-entity",
        "team",
        "--wandb-group",
        "archive",
        "--wandb-run-name",
        "research-loop-DISTANCE_CAP4",
        "--wandb-mode",
        "disabled",
        "--wandb-tags",
        "course",
        "DISTANCE_CAP4",
    ]
