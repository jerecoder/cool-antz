from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant_byte_env.autoresearch import (
    AutoresearchResourceError,
    assert_autoresearch_resources_available,
)
from ant_byte_env.cli import main as cli_main
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.research_loop import (
    build_research_experiment_plan,
    execute_research_experiment_plan,
    load_research_loop_matrix,
    rank_research_loop_runs,
)
from ant_byte_env.results import index_result_metadata
from ant_byte_env.runs import append_metrics, prepare_run_dir, write_json


def test_experiment_config_loads_and_converts_args() -> None:
    spec = load_experiment_config(Path("experiments/smoke.json"))

    argv = config_args_to_argv(spec.args)

    assert spec.name == "smoke"
    assert spec.backend == "torch"
    assert "--total-timesteps" in argv
    assert "8" in argv
    assert "--quiet" in argv
    assert "--no-cuda" in argv


def test_cli_dry_run_validates_config_and_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(
        [
            "train",
            "torch",
            "--config",
            "experiments/smoke.json",
            "--dry-run",
            "--",
            "--seed",
            "11",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["backend"] == "torch"
    assert payload["experiment"] == "smoke"
    assert payload["resolved_args"]["seed"] == 11
    assert payload["resolved_args"]["total_timesteps"] == 8


def test_jax_cli_dry_run_validates_without_jax_backend(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(
        [
            "train",
            "jax",
            "--config",
            "experiments/forage_curriculum.json",
            "--dry-run",
            "--",
            "--total-timesteps",
            "8",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["backend"] == "jax"
    assert payload["experiment"] == "forage_curriculum"
    assert payload["resolved_args"]["obs_width"] == 50
    assert payload["resolved_args"]["obs_height"] == 50
    assert payload["resolved_args"]["total_timesteps"] == 8


def test_research_loop_matrix_is_self_contained_and_substantive() -> None:
    matrix = load_research_loop_matrix()
    ids = [entry["id"] for entry in matrix["experiments"]]
    families = {entry["family"] for entry in matrix["experiments"]}

    assert ids == [
        "DISTANCE_SHAPE",
        "CAPACITY4",
        "DISTANCE_CAP4",
        "DISTANCE_CAP4_SHARP",
        "DISTANCE_CAP4_BALANCED",
        "DISTANCE_CAP4_SHARP_FINE",
        "DISTANCE_CAP4_DISTILL",
        "DISTANCE_CAP4_GREEDY_TUNE",
        "DISTANCE_CAP4_MIXED_TUNE",
        "DISTANCE_CAP4_NO_WRITE",
        "DISTANCE_CAP4_SHARP_NEAR",
        "DISTANCE_CAP4_LONG_CREDIT_TUNE",
        "DISTANCE_CAP4_LONG_CREDIT_GENTLE_GREEDY",
        "DISTANCE_CAP4_LONG_CREDIT_ENTROPY_PENALTY",
        "DISTANCE_CAP4_SHARP_STABLE",
        "DISTANCE_VISION2_CAP4_SHARP",
        "DISTANCE_CAP4_SHARP_MOVE_ALIGN",
        "DISTANCE_CAP4_SHARP_SEED2",
        "DISTANCE_CAP4_SHARP_HYBRID_POLICY",
        "DISTANCE_CAP4_SHARP_TEMP_POLICY",
        "DISTANCE_CAP4_SHARP_TEMP_FINE_POLICY",
        "DISTANCE_CAP4_SHARP_T125_CONFIRM_POLICY",
        "DISTANCE_CAP4_SHARP_TEMP_CONFIRM_GRID",
        "DISTANCE_CAP4_SHARP_TEMP_HIGH_GRID",
        "DISTANCE_CAP4_SHARP_TEMP_TOP3_CONFIRM",
        "DISTANCE_CAP4_SPEED_4A_800",
        "DISTANCE_CAP4_SPEED_4A_550",
        "DISTANCE_CAP8_SPEED_8A_550",
        "DISTANCE_CAP12_SPEED_12A_550",
        "DISTANCE_CAP16_SPEED_16A_550",
        "DISTANCE_CAP16_SPEED_TEMP_GRID",
        "DISTANCE_CAP24_SPEED_RAMP_24A_430",
        "DISTANCE_CAP24_SPEED_SOURCES12_430",
        "DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430",
        "DISTANCE_CAP24_SPEED_POLISH_FINE_TEMP",
        "DISTANCE_CAP24_SPEED_SOURCES12_FINAL_430",
        "DISTANCE_CAP24_SPEED_COMPLETION_BONUS_430",
        "DISTANCE_CAP24_SPEED_COMPLETION_SHORT_430",
        "DISTANCE_CAP24_SPEED_DENSE2_430",
        "DISTANCE_CAP24_SPEED_DENSE2_TRAIL_430",
        "DISTANCE_CAP24_SPEED_SOURCES16_430",
        "DISTANCE_CAP24_SPEED_BEST_SELECT_430",
        "DISTANCE_CAP32_SPEED_SOURCES12_430",
        "DISTANCE_CAP8_SPEED_RAMP_8A",
        "DISTANCE_VISION2_CAP4",
        "VISION2",
        "NEAR_COOKIE",
        "DENSE8",
        "GAMMA999",
        "LADDER_FINE",
        "VISION2_CAP4",
        "BYTE_TRAIL",
        "AUTO_STAGE",
    ]
    assert {
        "reward_shaping",
        "combined_reward_capacity",
        "policy_sharpening",
        "exploration_capacity",
        "observation",
        "cookie_distribution",
        "food_distribution",
        "credit_assignment",
        "stage_schedule",
        "policy_distillation",
        "deterministic_policy",
        "action_ablation",
        "mode_alignment",
        "entropy_penalty",
        "stable_sharpening",
        "vision_sharpening",
        "movement_mode_alignment",
        "seed_robustness",
        "deployment_policy",
        "efficiency_finetune",
        "checkpoint_selection",
        "combined_capacity",
        "memory_shaping",
        "autocurriculum",
    } <= families
    assert matrix["target"]["stage_name"] == "25x25"
    assert matrix["target"]["minimum_promotion_episode_return"] > 1.0
    assert matrix["evaluation"]["deterministic_episodes"] >= 4
    for entry in matrix["experiments"]:
        assert entry["hypothesis"]
        assert entry["intervention"]
        assert entry["success_signal"]
        assert entry["report_notes"]


def test_research_loop_plan_builds_forage_curriculum_with_notes() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    plan = build_research_experiment_plan(
        run_id="CAPACITY4",
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )
    parsed = parse_args(plan["common_args"])

    assert plan["mode"] == "forage_curriculum"
    assert plan["id"] == "CAPACITY4"
    assert plan["stage_sizes"][-1] == 25
    assert plan["target"]["stage_name"] == "25x25"
    assert "## Hypothesis" in plan["notes_markdown"]
    assert "## Evaluation Gate" in plan["notes_markdown"]
    assert "Four ants" in plan["notes_markdown"]
    assert plan["wandb"]["mode"] == "disabled"
    assert plan["evaluation"]["sampled_episodes"] == 8
    assert parsed.num_envs == 1
    assert parsed.num_steps == 4
    assert parsed.num_ants == 4
    assert parsed.hidden_size == 192
    assert parsed.random_food
    assert parsed.random_hub


def test_research_loop_plan_can_sharpen_best_sampled_policy() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    parsed = parse_args(plan["common_args"])

    assert plan["family"] == "policy_sharpening"
    assert "argmax" in plan["hypothesis"]
    assert plan["stages"][-1]["global_update_cap"] == 1500
    assert parsed.num_ants == 4
    assert parsed.distance_bonus == 0.02
    assert parsed.ent_coef == 0.001
    assert parsed.clip_coef == 0.15

    balanced = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_BALANCED",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    balanced_parsed = parse_args(balanced["common_args"])

    assert "moderate entropy" in balanced["title"].lower()
    assert balanced["stages"][-1]["global_update_cap"] == 1500
    assert balanced_parsed.ent_coef == 0.003
    assert balanced_parsed.clip_coef == 0.15

    fine = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_FINE",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    fine_parsed = parse_args(fine["common_args"])

    assert fine["family"] == "stage_schedule"
    assert fine["stage_sizes"][-6:] == [20, 21, 22, 23, 24, 25]
    assert fine["stages"][-1]["global_update_cap"] == 1300
    assert fine_parsed.num_ants == 4
    assert fine_parsed.distance_bonus == 0.02
    assert fine_parsed.ent_coef == 0.001
    assert fine_parsed.clip_coef == 0.15

    distill = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_DISTILL",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    distill_parsed = parse_args(distill["common_args"])

    assert distill["family"] == "policy_distillation"
    assert distill["stage_sizes"] == [25]
    assert distill["stages"][0]["global_update_cap"] == 1600
    assert distill["source_checkpoint"].endswith(
        "DISTANCE_CAP4/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert distill_parsed.num_ants == 4
    assert distill_parsed.distance_bonus == 0.02
    assert distill_parsed.ent_coef == 0.0005
    assert distill_parsed.clip_coef == 0.1
    assert distill_parsed.learning_rate == 0.00008
    assert distill_parsed.update_epochs == 6

    greedy_tune = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_GREEDY_TUNE",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    greedy_parsed = parse_args(greedy_tune["common_args"])

    assert greedy_tune["family"] == "deterministic_policy"
    assert greedy_tune["stage_sizes"] == [25]
    assert greedy_tune["stages"][0]["global_update_cap"] == 1200
    assert greedy_tune["source_checkpoint"].endswith(
        "DISTANCE_CAP4_SHARP/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert greedy_parsed.deterministic_rollout is True
    assert greedy_parsed.num_ants == 4
    assert greedy_parsed.distance_bonus == 0.02
    assert greedy_parsed.ent_coef == 0.0
    assert greedy_parsed.clip_coef == 0.1
    assert greedy_parsed.learning_rate == 0.00005
    assert greedy_parsed.update_epochs == 6

    mixed_tune = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_MIXED_TUNE",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    mixed_parsed = parse_args(mixed_tune["common_args"])

    assert mixed_tune["family"] == "deterministic_policy"
    assert mixed_tune["stage_sizes"] == [25]
    assert mixed_tune["stages"][0]["global_update_cap"] == 1200
    assert mixed_tune["source_checkpoint"].endswith(
        "DISTANCE_CAP4_SHARP/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert mixed_parsed.deterministic_rollout_fraction == 0.25
    assert mixed_parsed.num_ants == 4
    assert mixed_parsed.distance_bonus == 0.02
    assert mixed_parsed.ent_coef == 0.0005
    assert mixed_parsed.clip_coef == 0.1
    assert mixed_parsed.learning_rate == 0.00005
    assert mixed_parsed.update_epochs == 6

    no_write = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_NO_WRITE",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    no_write_parsed = parse_args(no_write["common_args"])

    assert no_write["family"] == "action_ablation"
    assert no_write["stage_sizes"] == [25]
    assert no_write["stages"][0]["global_update_cap"] == 1200
    assert no_write["source_checkpoint"].endswith(
        "DISTANCE_CAP4_SHARP/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert no_write_parsed.write_action_ablation is True
    assert no_write_parsed.num_ants == 4
    assert no_write_parsed.distance_bonus == 0.02
    assert no_write_parsed.ent_coef == 0.001
    assert no_write_parsed.clip_coef == 0.1
    assert no_write_parsed.learning_rate == 0.00005
    assert no_write_parsed.update_epochs == 6

    near = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_NEAR",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    near_parsed = parse_args(near["common_args"])

    assert near["family"] == "cookie_distribution"
    assert near["stage_sizes"] == [25]
    assert near["stages"][0]["cookie_distance"] < 12
    assert near["source_checkpoint"].endswith(
        "DISTANCE_CAP4_SHARP/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert near_parsed.num_ants == 4
    assert near_parsed.distance_bonus == 0.02
    assert near_parsed.ent_coef == 0.001
    assert near_parsed.clip_coef == 0.1
    assert near_parsed.learning_rate == 0.00005
    assert near_parsed.update_epochs == 6

    long_credit = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_LONG_CREDIT_TUNE",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    long_credit_parsed = parse_args(long_credit["common_args"])

    assert long_credit["family"] == "credit_assignment"
    assert long_credit["stage_sizes"] == [25]
    assert long_credit["stages"][0]["global_update_cap"] == 900
    assert long_credit["stages"][0]["num_steps"] == 384
    assert long_credit["stages"][0]["gamma"] == 0.999
    assert long_credit["source_checkpoint"].endswith(
        "DISTANCE_CAP4_SHARP/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert long_credit_parsed.num_ants == 4
    assert long_credit_parsed.distance_bonus == 0.02
    assert long_credit_parsed.gamma == 0.999
    assert long_credit_parsed.gae_lambda == 0.98
    assert long_credit_parsed.ent_coef == 0.001
    assert long_credit_parsed.clip_coef == 0.15
    assert long_credit_parsed.learning_rate == 0.00008


def test_research_loop_plan_can_change_density_and_autocurriculum() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    dense = build_research_experiment_plan(
        run_id="DENSE8",
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )
    auto = build_research_experiment_plan(
        run_id="AUTO_STAGE",
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )

    stage_25 = dense["stages"][-1]
    assert stage_25["name"] == "25x25"
    assert stage_25["food_count"] == 23
    assert stage_25["food_sources"] == 3
    auto_parsed = parse_args(auto["common_args"])
    assert auto["mode"] == "autocurriculum"
    assert auto["total_train_env_steps"] == 8
    assert auto_parsed.autocurriculum is True
    assert auto_parsed.actor_vision_radius == 2
    assert auto_parsed.stage_completion_bonus == 3.0


def test_research_loop_plan_can_change_cookie_distance_and_stage_sizes() -> None:
    near = build_research_experiment_plan(
        run_id="NEAR_COOKIE",
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )
    fine = build_research_experiment_plan(
        run_id="LADDER_FINE",
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )

    assert near["stages"][-1]["cookie_distance"] < 12
    assert fine["stage_sizes"][-4:] == [22, 23, 24, 25]


def test_research_loop_plan_can_target_short_horizon_speed() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    four_ant = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SPEED_4A_550",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    eight_ant = build_research_experiment_plan(
        run_id="DISTANCE_CAP8_SPEED_8A_550",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    ramp = build_research_experiment_plan(
        run_id="DISTANCE_CAP8_SPEED_RAMP_8A",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    true_4x = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_RAMP_24A_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    sources12 = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_SOURCES12_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    sources12_polish = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    final_polish = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_SOURCES12_FINAL_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    completion_bonus = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_COMPLETION_BONUS_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    dense2 = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_DENSE2_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    dense2_trail = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_DENSE2_TRAIL_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    sources16 = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_SOURCES16_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    best_select = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_BEST_SELECT_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )
    cap32 = build_research_experiment_plan(
        run_id="DISTANCE_CAP32_SPEED_SOURCES12_430",
        global_update_cap=None,
        num_envs=1,
        num_steps=None,
        wandb_mode="disabled",
    )

    four_parsed = parse_args(four_ant["common_args"])
    eight_parsed = parse_args(eight_ant["common_args"])
    ramp_parsed = parse_args(ramp["common_args"])
    true_4x_parsed = parse_args(true_4x["common_args"])

    assert four_ant["family"] == "efficiency_finetune"
    assert four_ant["stage_sizes"] == [25]
    assert four_ant["stages"][0]["max_steps"] == 550
    assert four_ant["stages"][0]["global_update_cap"] == 1000
    assert four_ant["source_checkpoint"].endswith(
        "DISTANCE_CAP4_SHARP/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert four_parsed.num_ants == 4
    assert four_parsed.step_penalty == 0.00035
    assert four_ant["evaluation"]["action_modes"][0]["move_temperature"] == 1.1

    assert eight_ant["family"] == "efficiency_finetune"
    assert eight_ant["stage_sizes"] == [25]
    assert eight_ant["stages"][0]["max_steps"] == 550
    assert eight_ant["stages"][0]["global_update_cap"] == 900
    assert eight_ant["source_checkpoint"].endswith(
        "DISTANCE_CAP4_SHARP/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert eight_parsed.num_ants == 8
    assert eight_parsed.step_penalty == 0.00016

    assert ramp["family"] == "efficiency_finetune"
    assert ramp["stage_sizes"] == [25, 25, 25]
    assert [stage["name"] for stage in ramp["stages"]] == [
        "25x25_900",
        "25x25_700",
        "25x25",
    ]
    assert [stage["max_steps"] for stage in ramp["stages"]] == [900, 700, 550]
    assert [stage["global_update_cap"] for stage in ramp["stages"]] == [300, 300, 300]
    assert ramp["final_checkpoint"].endswith(
        "DISTANCE_CAP8_SPEED_RAMP_8A/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert ramp_parsed.num_ants == 8
    assert ramp_parsed.step_penalty == 0.00012

    assert true_4x["family"] == "efficiency_finetune"
    assert true_4x["stage_sizes"] == [25, 25, 25]
    assert [stage["max_steps"] for stage in true_4x["stages"]] == [700, 550, 430]
    assert [stage["global_update_cap"] for stage in true_4x["stages"]] == [300, 300, 400]
    assert true_4x["final_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_RAMP_24A_430/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert true_4x_parsed.num_ants == 24
    assert true_4x_parsed.step_penalty == 0.000045
    assert [mode["move_temperature"] for mode in true_4x["evaluation"]["action_modes"]] == [
        0.9,
        1.1,
        1.3,
    ]

    assert sources12["family"] == "food_distribution"
    assert sources12["stage_sizes"] == [25, 25, 25]
    assert [stage["food_sources"] for stage in sources12["stages"]] == [12, 12, 12]
    assert [stage["max_steps"] for stage in sources12["stages"]] == [700, 550, 430]
    assert sources12["final_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_SOURCES12_430/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )

    assert sources12_polish["family"] == "food_distribution"
    assert sources12_polish["stage_sizes"] == [25]
    assert sources12_polish["stages"][0]["food_sources"] == 12
    assert sources12_polish["stages"][0]["max_steps"] == 430
    assert sources12_polish["stages"][0]["global_update_cap"] == 700
    assert sources12_polish["source_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_SOURCES12_430/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )

    final_parsed = parse_args(final_polish["common_args"])
    assert final_polish["family"] == "food_distribution"
    assert final_polish["stages"][0]["food_sources"] == 12
    assert final_polish["stages"][0]["max_steps"] == 430
    assert final_polish["stages"][0]["global_update_cap"] == 500
    assert final_polish["source_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert final_parsed.step_penalty == 0.000035

    completion_parsed = parse_args(completion_bonus["common_args"])
    assert completion_bonus["family"] == "reward_shaping"
    assert completion_bonus["stages"][0]["food_sources"] == 12
    assert completion_bonus["stages"][0]["max_steps"] == 430
    assert completion_bonus["source_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert completion_parsed.completion_bonus == 2.0

    dense2_parsed = parse_args(dense2["common_args"])
    assert dense2["family"] == "food_distribution"
    assert [stage["food_count"] for stage in dense2["stages"]] == [23, 23]
    assert [stage["food_sources"] for stage in dense2["stages"]] == [2, 2]
    assert [stage["max_steps"] for stage in dense2["stages"]] == [550, 430]
    assert dense2_parsed.num_ants == 24
    assert dense2_parsed.completion_bonus == 1.5

    dense2_trail_parsed = parse_args(dense2_trail["common_args"])
    assert dense2_trail["family"] == "memory_shaping"
    assert [stage["food_sources"] for stage in dense2_trail["stages"]] == [2, 2]
    assert dense2_trail_parsed.write_bits == 2
    assert dense2_trail_parsed.write_head_transfer == "neutral-new"
    assert dense2_trail_parsed.delivery_byte_trail_bonus == 0.2
    assert dense2_trail_parsed.byte_follow_bonus == 0.004

    sources16_parsed = parse_args(sources16["common_args"])
    assert sources16["family"] == "food_distribution"
    assert [stage["food_count"] for stage in sources16["stages"]] == [23, 23]
    assert [stage["food_sources"] for stage in sources16["stages"]] == [16, 16]
    assert [stage["max_steps"] for stage in sources16["stages"]] == [550, 430]
    assert sources16["source_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert sources16_parsed.num_ants == 24
    assert sources16_parsed.completion_bonus == 1.5

    best_select_parsed = parse_args(best_select["common_args"])
    assert best_select["family"] == "checkpoint_selection"
    assert best_select["stage_sizes"] == [25]
    assert best_select["stages"][0]["food_sources"] == 12
    assert best_select["stages"][0]["max_steps"] == 430
    assert best_select["stages"][0]["save_best_checkpoint"] is True
    assert best_select["stages"][0]["select_best_checkpoint"] is True
    assert best_select["final_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_BEST_SELECT_430/checkpoints/jax_mappo_forage_stage1_25x25_best.pkl"
    )
    assert best_select_parsed.num_ants == 24
    assert best_select_parsed.completion_bonus == 1.5
    assert best_select_parsed.log_interval == 10

    cap32_parsed = parse_args(cap32["common_args"])
    assert cap32["family"] == "food_distribution"
    assert cap32["stages"][0]["food_sources"] == 12
    assert cap32["stages"][0]["max_steps"] == 430
    assert cap32["stages"][0]["global_update_cap"] == 600
    assert cap32["source_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert cap32_parsed.num_ants == 32
    assert cap32_parsed.step_penalty == 0.00003


def test_research_loop_can_evaluate_checkpoint_action_modes(tmp_path: Path) -> None:
    plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_HYBRID_POLICY",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"checkpoint")
    plan["checkpoint"] = str(checkpoint)
    plan["final_checkpoint"] = str(checkpoint)
    plan["source_checkpoint"] = str(checkpoint)
    calls: list[dict[str, object]] = []

    def fake_evaluate_checkpoint(
        checkpoint_path: Path,
        *,
        num_episodes: int,
        seed_offset: int,
        action_mode: str,
        shuffle_positions: bool,
    ) -> dict[str, float]:
        calls.append(
            {
                "checkpoint_path": checkpoint_path,
                "num_episodes": num_episodes,
                "seed_offset": seed_offset,
                "action_mode": action_mode,
                "shuffle_positions": shuffle_positions,
            }
        )
        return {
            "eval_success_rate": 1.0,
            "eval_mean_delivered_food": 23.0,
            "eval_mean_delivered_fraction": 1.0,
            "eval_mean_episode_return": 23.0,
            "eval_mean_episode_length": 800.0,
        }

    summary = execute_research_experiment_plan(
        plan,
        evaluate_checkpoint=fake_evaluate_checkpoint,
        check_resources=False,
    )

    assert summary["mode"] == "checkpoint_evaluation"
    assert plan["total_train_env_steps"] == 0
    assert [call["action_mode"] for call in calls] == [
        "greedy_move_greedy_write",
        "greedy_move_sampled_write",
        "sampled_move_greedy_write",
        "sampled_move_sampled_write",
    ]
    assert calls[0]["num_episodes"] == 32
    assert summary["evaluation"]["sampled_move_greedy_write"]["eval_mean_delivered_food"] == 23.0

    temp_plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_TEMP_POLICY",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )

    assert temp_plan["mode"] == "checkpoint_evaluation"
    assert temp_plan["evaluation"]["action_modes"][0]["move_temperature"] == 0.5
    assert temp_plan["evaluation"]["action_modes"][-1]["move_temperature"] == 1.25

    fine_temp_plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_TEMP_FINE_POLICY",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )

    fine_modes = fine_temp_plan["evaluation"]["action_modes"]
    assert [mode["episodes"] for mode in fine_modes] == [48, 48, 48, 48, 48]
    assert len({mode["seed_offset"] for mode in fine_modes}) == 1

    confirm_plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_T125_CONFIRM_POLICY",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )

    confirm_modes = confirm_plan["evaluation"]["action_modes"]
    assert [mode["episodes"] for mode in confirm_modes] == [128, 128]
    assert confirm_modes[0]["move_temperature"] == 1.25
    assert confirm_modes[1]["action_mode"] == "sampled_move_sampled_write"

    confirm_grid_plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_TEMP_CONFIRM_GRID",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )

    grid_modes = confirm_grid_plan["evaluation"]["action_modes"]
    assert [mode["episodes"] for mode in grid_modes] == [96, 96, 96, 96, 96]
    assert len({mode["seed_offset"] for mode in grid_modes}) == 1

    high_grid_plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_TEMP_HIGH_GRID",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )

    high_modes = high_grid_plan["evaluation"]["action_modes"]
    assert [mode["move_temperature"] for mode in high_modes] == [1.25, 1.4, 1.6, 1.8, 2.0]
    assert [mode["episodes"] for mode in high_modes] == [64, 64, 64, 64, 64]

    top3_plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_SHARP_TEMP_TOP3_CONFIRM",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )

    top3_modes = top3_plan["evaluation"]["action_modes"]
    assert [mode["move_temperature"] for mode in top3_modes] == [1.1, 1.25, 1.4]
    assert [mode["episodes"] for mode in top3_modes] == [128, 128, 128]

    speed_temp_plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP16_SPEED_TEMP_GRID",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )

    speed_temp_modes = speed_temp_plan["evaluation"]["action_modes"]
    assert speed_temp_plan["mode"] == "checkpoint_evaluation"
    assert speed_temp_plan["source_checkpoint"].endswith(
        "DISTANCE_CAP16_SPEED_16A_550/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert [mode["move_temperature"] for mode in speed_temp_modes] == [
        0.9,
        1.1,
        1.3,
        1.5,
        1.8,
    ]
    assert len({mode["seed_offset"] for mode in speed_temp_modes}) == 1

    polish_temp_plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP24_SPEED_POLISH_FINE_TEMP",
        run_root=tmp_path / "loop",
        wandb_mode="disabled",
    )

    polish_temp_modes = polish_temp_plan["evaluation"]["action_modes"]
    assert polish_temp_plan["mode"] == "checkpoint_evaluation"
    assert polish_temp_plan["source_checkpoint"].endswith(
        "DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert [mode["move_temperature"] for mode in polish_temp_modes] == [
        0.95,
        1.0,
        1.05,
        1.1,
    ]
    assert [mode["episodes"] for mode in polish_temp_modes] == [96, 96, 96, 96]


def test_execute_research_loop_plan_runs_forage_and_writes_report_files(tmp_path: Path) -> None:
    plan = build_research_experiment_plan(
        run_id="VISION2",
        run_root=tmp_path / "loop",
        global_update_cap=1,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )
    calls: dict[str, object] = {}

    def fake_train_main(args: list[str], progress_callback=None) -> dict[str, float]:
        del args, progress_callback
        return {"global_step": 4.0}

    def fake_run_curriculum(**kwargs: object) -> dict[str, object]:
        calls.update(kwargs)
        checkpoint_dir = Path(str(kwargs["checkpoint_dir"]))
        final_checkpoint = checkpoint_dir / "jax_mappo_forage_stage1_25x25.pkl"
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        final_checkpoint.write_bytes(b"checkpoint")
        return {
            "stage_metrics": [
                {
                    "stage_name": "25x25",
                    "episode_return": 4.0,
                    "delivery_events": 64.0,
                    "pickup_events": 70.0,
                    "final_mean_remaining_food": 5.0,
                }
            ],
            "stage_checkpoint_paths": [final_checkpoint],
            "final_checkpoint_path": final_checkpoint,
            "final_train_metrics": {"global_step": 4.0},
        }

    def fake_evaluate_checkpoint(
        checkpoint_path: Path,
        *,
        num_episodes: int,
        seed_offset: int,
        deterministic: bool,
        shuffle_positions: bool,
    ) -> dict[str, float]:
        assert checkpoint_path.name == "jax_mappo_forage_stage1_25x25.pkl"
        assert num_episodes == 8
        assert seed_offset >= 1_000_000
        assert shuffle_positions is True
        return {
            "eval_success_rate": 0.0,
            "eval_mean_delivered_food": 3.0 if deterministic else 2.0,
            "eval_mean_delivered_fraction": 0.25,
            "eval_mean_episode_return": 2.5 if deterministic else 2.0,
            "eval_mean_episode_length": 100.0,
        }

    summary = execute_research_experiment_plan(
        plan,
        train_main=fake_train_main,
        run_curriculum=fake_run_curriculum,
        evaluate_checkpoint=fake_evaluate_checkpoint,
        check_resources=False,
    )

    assert summary["resumed"] is False
    assert "radius-2" in Path(summary["note_path"]).read_text(encoding="utf-8")
    assert Path(summary["plan_path"]).exists()
    assert Path(summary["summary_path"]).exists()
    assert Path(summary["evaluation_path"]).exists()
    assert calls["wandb_notes"].startswith("# VISION2")
    assert [path.name for path in calls["wandb_artifact_paths"]] == ["experiment.md", "plan.json"]
    assert summary["curriculum"]["stage_metrics"][0]["episode_return"] == 4.0
    assert summary["evaluation"]["deterministic"]["eval_mean_episode_return"] == 2.5


def test_execute_research_loop_plan_passes_source_checkpoint(tmp_path: Path) -> None:
    expected_source = Path(
        "runs/autoresearch/forage_loop/DISTANCE_CAP4/checkpoints/"
        "jax_mappo_forage_stage1_25x25.pkl"
    )
    plan = build_research_experiment_plan(
        run_id="DISTANCE_CAP4_DISTILL",
        run_root=tmp_path / "loop",
        global_update_cap=1,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )
    calls: dict[str, object] = {}

    def fake_train_main(args: list[str], progress_callback=None) -> dict[str, float]:
        del args, progress_callback
        return {"global_step": 4.0}

    def fake_run_curriculum(**kwargs: object) -> dict[str, object]:
        calls.update(kwargs)
        checkpoint_dir = Path(str(kwargs["checkpoint_dir"]))
        final_checkpoint = checkpoint_dir / "jax_mappo_forage_stage1_25x25.pkl"
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        final_checkpoint.write_bytes(b"checkpoint")
        return {
            "stage_metrics": [],
            "stage_checkpoint_paths": [final_checkpoint],
            "final_checkpoint_path": final_checkpoint,
            "final_train_metrics": {"global_step": 4.0},
        }

    summary = execute_research_experiment_plan(
        plan,
        train_main=fake_train_main,
        run_curriculum=fake_run_curriculum,
        evaluate_checkpoint=lambda *args, **kwargs: {
            "eval_success_rate": 0.0,
            "eval_mean_delivered_food": 0.0,
            "eval_mean_delivered_fraction": 0.0,
            "eval_mean_episode_return": 0.0,
            "eval_mean_episode_length": 100.0,
        },
        check_resources=False,
    )

    assert calls["initial_checkpoint"] == expected_source
    assert "Source checkpoint:" in Path(summary["note_path"]).read_text(encoding="utf-8")


def test_research_loop_rank_reads_target_stage(tmp_path: Path) -> None:
    matrix = load_research_loop_matrix()
    matrix["run_root"] = str(tmp_path / "runs")
    matrix["experiments"] = [
        {**matrix["experiments"][0], "run_dir": str(tmp_path / "runs" / "A")},
        {**matrix["experiments"][1], "run_dir": str(tmp_path / "runs" / "B")},
    ]
    matrix_path = tmp_path / "loop.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    for run_id, episode_return, deliveries in [("A", 2.0, 20.0), ("B", 5.0, 50.0)]:
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        write_json(
            run_dir / "summary.json",
            {
                "curriculum": {
                    "stage_metrics": [
                        {
                            "stage_name": "25x25",
                            "episode_return": episode_return,
                            "delivery_events": deliveries,
                            "pickup_events": deliveries + 1,
                            "final_mean_remaining_food": 4.0,
                        }
                    ]
                }
            },
        )

    ranked = rank_research_loop_runs(matrix_path=matrix_path)

    assert [row["id"] for row in ranked["ranked"]] == ["CAPACITY4", "DISTANCE_SHAPE"]
    assert ranked["missing"] == []


def test_cli_research_loop_plan_prints_self_contained_notes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(
        [
            "autoresearch",
            "loop-plan",
            "--id",
            "VISION2_CAP4",
            "--global-update-cap",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--wandb-mode",
            "disabled",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == "VISION2_CAP4"
    assert payload["mode"] == "forage_curriculum"
    assert "## Success Signal" in payload["notes_markdown"]


def test_cli_research_loop_run_uses_executable_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ant_byte_env.research_loop as research_loop_module

    def fake_run_research_experiment(**kwargs: object) -> dict[str, object]:
        assert kwargs["run_id"] == "CAPACITY4"
        assert kwargs["check_resources"] is False
        return {
            "id": "CAPACITY4",
            "summary_path": "runs/autoresearch/forage_loop/CAPACITY4/summary.json",
        }

    monkeypatch.setattr(
        research_loop_module,
        "run_research_experiment",
        fake_run_research_experiment,
    )

    exit_code = cli_main(
        [
            "autoresearch",
            "loop-run",
            "--id",
            "CAPACITY4",
            "--skip-resource-check",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == "CAPACITY4"


def test_cli_research_loop_auto_uses_controller(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ant_byte_env.research_loop as research_loop_module

    def fake_run_research_loop(**kwargs: object) -> dict[str, object]:
        assert kwargs["run_ids"] == ["DISTANCE_SHAPE", "CAPACITY4"]
        assert kwargs["max_runs"] == 2
        assert kwargs["min_disk_free_gb"] == 4.0
        return {
            "ledger_path": "runs/autoresearch/forage_loop/ledger.json",
            "results": [{"id": "DISTANCE_SHAPE"}],
        }

    monkeypatch.setattr(
        research_loop_module,
        "run_research_loop",
        fake_run_research_loop,
    )

    exit_code = cli_main(
        [
            "autoresearch",
            "loop-auto",
            "--ids",
            "DISTANCE_SHAPE",
            "CAPACITY4",
            "--max-runs",
            "2",
            "--min-disk-free-gb",
            "4.0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["results"][0]["id"] == "DISTANCE_SHAPE"


def test_cli_research_loop_run_reports_resource_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ant_byte_env.research_loop as research_loop_module

    def fake_run_research_experiment(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise AutoresearchResourceError("disk is too full")

    monkeypatch.setattr(
        research_loop_module,
        "run_research_experiment",
        fake_run_research_experiment,
    )

    exit_code = cli_main(["autoresearch", "loop-run", "--id", "CAPACITY4"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "disk is too full" in captured.err


def test_autoresearch_resource_guard_rejects_low_swap_when_memory_is_tight() -> None:
    with pytest.raises(AutoresearchResourceError, match="available RAM"):
        assert_autoresearch_resources_available(
            snapshot={
                "disk_free_gb": 20.0,
                "mem_available_gb": 1.5,
                "swap_free_gb": 0.1,
                "gpu_compute_memory_mb": 0,
            }
        )


def test_autoresearch_resource_guard_accepts_safe_snapshot() -> None:
    assert_autoresearch_resources_available(
        snapshot={
            "disk_free_gb": 20.0,
            "mem_available_gb": 8.0,
            "swap_free_gb": 1.0,
            "gpu_compute_memory_mb": 0,
        }
    )


def test_run_helpers_create_manifest_and_metrics(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path, "smoke")

    append_metrics(run_dir / "metrics.jsonl", {"episode_return": 1.25, "step": 1})
    write_json(run_dir / "summary.json", {"ok": True})

    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "media").is_dir()
    assert json.loads((run_dir / "summary.json").read_text()) == {"ok": True}
    assert json.loads((run_dir / "metrics.jsonl").read_text().strip())["episode_return"] == 1.25


def test_result_indexer_reads_vault_metadata(tmp_path: Path) -> None:
    entry_dir = tmp_path / "runs" / "vault" / "20260611T000000Z"
    entry_dir.mkdir(parents=True)
    write_json(
        entry_dir / "metadata.json",
        {
            "id": "20260611T000000Z",
            "source": "manual",
            "title": "Interesting run",
            "tags": ["forage"],
            "metrics": {"episode_return": 2.5},
            "files": ["rollout.mp4"],
            "notes": "good",
        },
    )

    payload = index_result_metadata(tmp_path / "runs", tmp_path / "index.json")

    assert payload["entry_count"] == 1
    assert payload["entries"][0]["title"] == "Interesting run"
    assert json.loads((tmp_path / "index.json").read_text())["entry_count"] == 1
