from pathlib import Path

import pytest

from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.workflows import ant_count


def test_ant_count_training_args_keep_25x25_task_with_50_padded_observations() -> None:
    args = ant_count.ant_count_training_args(
        {"num_envs": 16, "num_steps": 80, "write_bits": 1},
        communication_bits=3,
    )

    assert args["width"] == 25
    assert args["height"] == 25
    assert args["obs_width"] == 50
    assert args["obs_height"] == 50
    assert args["food_count"] == 23
    assert args["food_sources"] == 6
    assert args["cookie_distance"] == 11
    assert args["max_steps"] == 2500
    assert args["write_bits"] == 3
    assert args["write_while_moving"] is True


def test_validate_ant_count_stages_rejects_non_increasing_curricula() -> None:
    with pytest.raises(ValueError, match="increasing"):
        ant_count.validate_ant_count_stages(ant_stages=[2, 2], source_num_ants=1)

    with pytest.raises(ValueError, match="beyond"):
        ant_count.validate_ant_count_stages(ant_stages=[1, 2], source_num_ants=1)


def test_ant_count_train_args_builds_stage_argv(tmp_path: Path) -> None:
    argv = ant_count.ant_count_train_args(
        common_args=["--num-envs", "16"],
        experiment_name="jax_mappo_communication",
        target_num_ants=4,
        communication_bits=3,
        update_timesteps_per_stage=1280,
        global_update_cap=10,
        load_model=tmp_path / "source.pkl",
        run_dir=tmp_path / "run",
    )

    assert argv == [
        "--num-envs",
        "16",
        "--exp-name",
        "jax_mappo_communication_4_ants",
        "--write-bits",
        "3",
        "--num-ants",
        "4",
        "--total-timesteps",
        "12800",
        "--load-model",
        str(tmp_path / "source.pkl"),
        "--run-dir",
        str(tmp_path / "run"),
    ]


def test_notebook_workflows_reexports_ant_count_helpers() -> None:
    assert workflows.ant_count_training_args is ant_count.ant_count_training_args
    assert workflows.validate_ant_count_stages is ant_count.validate_ant_count_stages
    assert workflows.ant_count_train_args is ant_count.ant_count_train_args
