from pathlib import Path

import pytest

from ant_byte_env import experiments as experiment_config
from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.training.jax_mappo.cli import parse_args
from ant_byte_env.workflows import experiments


def test_load_jax_experiment_rejects_non_jax_configs(tmp_path: Path) -> None:
    config_path = tmp_path / "torch.json"
    config_path.write_text(
        '{"name": "torch_run", "backend": "torch", "args": {}, "metadata": {}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected a JAX experiment"):
        experiments.load_jax_experiment(config_path)


def test_resolve_project_path_preserves_absolute_paths(tmp_path: Path) -> None:
    absolute = tmp_path / "checkpoint.pkl"

    assert experiments.resolve_project_path(Path("/project"), absolute) == absolute
    assert experiments.resolve_project_path(Path("/project"), "runs/model.pkl") == (
        Path("/project") / "runs" / "model.pkl"
    )


def test_config_args_to_argv_preserves_wandb_tag_lists() -> None:
    argv = experiment_config.config_args_to_argv({"wandb_tags": ["jax", "50x50"]})

    assert argv == ["--wandb-tags", "jax", "50x50"]


def test_run_jax_smoke_uses_tiny_training_args() -> None:
    captured: list[str] = []

    def fake_train_main(argv: list[str]) -> dict[str, float]:
        captured.extend(argv)
        return {"loss": 0.0}

    assert experiments.run_jax_smoke(fake_train_main) == {"loss": 0.0}
    assert captured[captured.index("--total-timesteps") + 1] == "8"
    assert captured[captured.index("--hidden-size") + 1] == "16"
    assert "--quiet" in captured


def test_write_cost_experiment_uses_meaningful_bit_level_penalty() -> None:
    experiment = experiments.load_jax_experiment(
        Path(
            "experiments/"
            "exploration_to_forage_full_layout_8ants_half_food_50x50_"
            "shared_writes_write_cost.json"
        )
    )
    args = experiment.args

    assert args["load_model"].endswith(
        "best_full_layout_proximity_8ants_half_food_shared_writes.pkl"
    )
    assert args["per_ant_write_channels"] is False
    assert args["write_bit_penalty"] == pytest.approx(0.01)
    assert args["write_bit_penalty_decay"] == pytest.approx(0.5)
    assert args["write_bit_entropy_bonus"] == pytest.approx(0.0)

    common_args = workflows.config_common_args(
        args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )
    assert common_args[common_args.index("--write-bit-penalty") + 1] == "0.01"
    assert common_args[common_args.index("--write-bit-penalty-decay") + 1] == "0.5"

    full_value_penalty = sum(
        float(args["write_bit_penalty"])
        * (float(args["write_bit_penalty_decay"]) ** bit_index)
        for bit_index in range(int(args["write_bits"]))
    )
    full_episode_spam_cost = (
        full_value_penalty * int(args["num_ants"]) * int(args["max_steps"])
    )
    assert full_episode_spam_cost == pytest.approx(300.0)
    assert full_episode_spam_cost > float(args["food_count"]) * 2.0


def test_8bit_write_cost_experiment_uses_best_previous_shared_checkpoint() -> None:
    experiment = experiments.load_jax_experiment(
        Path(
            "experiments/"
            "exploration_to_forage_full_layout_8ants_half_food_50x50_"
            "shared_writes_write_cost_8bits.json"
        )
    )
    args = experiment.args
    metadata = experiment.metadata

    assert experiment.name == "fl50_8ants_half_food_shared_writes_write_cost_8bits_from_best"
    assert args["load_model"].endswith(
        "best_full_layout_proximity_8ants_half_food_shared_writes_write_cost.pkl"
    )
    assert args["save_best_model"].endswith(
        "best_full_layout_proximity_8ants_half_food_shared_writes_write_cost_8bits.pkl"
    )
    assert args["num_ants"] == 8
    assert args["write_bits"] == 8
    assert args["per_ant_write_channels"] is False
    assert args["write_head_transfer"] == "neutral-new"
    assert args["write_while_moving"] is True
    assert args["write_bit_penalty"] == pytest.approx(0.0002)
    assert args["write_bit_penalty_decay"] == pytest.approx(0.5)

    common_args = workflows.config_common_args(
        args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )
    assert common_args[common_args.index("--write-bits") + 1] == "8"
    assert common_args[common_args.index("--write-head-transfer") + 1] == "neutral-new"
    assert "--per-ant-write-channels" not in common_args

    full_value_penalty = sum(
        float(args["write_bit_penalty"])
        * (float(args["write_bit_penalty_decay"]) ** bit_index)
        for bit_index in range(int(args["write_bits"]))
    )
    full_episode_spam_cost = (
        full_value_penalty * int(args["num_ants"]) * int(args["max_steps"])
    )
    assert full_episode_spam_cost == pytest.approx(6.375)
    assert metadata["source_write_bits"] == 4
    assert metadata["target_write_bits"] == 8
    assert metadata["write_channel_policy"] == "shared_write_space"
    assert metadata["source_write_channel_policy"] == "shared_write_space"
    assert metadata["checkpoint_video_render_style"] == "big_scale_old_three_color"
    assert metadata["checkpoint_video_show_vision"] is False
    assert metadata["rollout_render_style"] == "big_scale_old_three_color"
    assert metadata["rollout_show_vision"] is False


def test_60_ant_8bit_experiment_starts_from_best_8bit_checkpoint() -> None:
    experiment = experiments.load_jax_experiment(
        Path(
            "experiments/"
            "exploration_to_forage_full_layout_60ants_half_food_50x50_"
            "shared_writes_write_cost_8bits_from_best.json"
        )
    )
    args = experiment.args
    metadata = experiment.metadata

    assert experiment.name == "fl50_60ants_half_food_sw_wc_8bits_from_8bit_best"
    assert args["load_model"].endswith(
        "best_full_layout_proximity_8ants_half_food_shared_writes_write_cost_8bits.pkl"
    )
    assert args["save_best_model"].endswith(
        "best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits.pkl"
    )
    assert args["num_ants"] == 60
    assert args["num_envs"] == 16
    assert args["total_timesteps"] == 81920000
    assert args["write_bits"] == 8
    assert args["per_ant_write_channels"] is False
    assert args["write_while_moving"] is True
    assert args["actor_vision_radius"] == 2
    assert args["agent_identity_types"] == 8
    assert args["hub_center_window_size"] == 24
    assert args["layout_margin"] == 0

    common_args = workflows.config_common_args(
        args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )
    assert common_args[common_args.index("--num-ants") + 1] == "60"
    assert common_args[common_args.index("--num-envs") + 1] == "16"
    assert common_args[common_args.index("--write-bits") + 1] == "8"
    assert common_args[common_args.index("--actor-vision-radius") + 1] == "2"
    assert common_args[common_args.index("--agent-identity-types") + 1] == "8"
    assert common_args[common_args.index("--hub-center-window-size") + 1] == "24"
    assert "--per-ant-write-channels" not in common_args

    assert metadata["source_num_ants"] == 8
    assert metadata["target_num_ants"] == 60
    assert metadata["hub_center_window"] == "24x24"
    assert metadata["source_identity_features"] == 8
    assert metadata["target_identity_features"] == 8
    assert metadata["checkpoint_video_interval_updates"] == 500
    assert metadata["checkpoint_video_render_style"] == "sprite"
    assert metadata["rollout_render_style"] == "sprite"
    assert metadata["actor_identity_policy"] == (
        "shared_actor_with_8_repeating_identity_types"
    )
    assert metadata["actor_identity_transfer"]["new_identity_features"] == 0
    assert metadata["actor_identity_transfer"]["identity_assignment"] == "ant_index_mod_8"


def test_60_ant_stabilization_experiment_preserves_best_policy_carefully() -> None:
    experiment = experiments.load_jax_experiment(
        Path(
            "experiments/"
            "exploration_to_forage_full_layout_60ants_half_food_50x50_"
            "shared_writes_write_cost_8bits_stabilize_from_60best.json"
        )
    )
    args = experiment.args
    metadata = experiment.metadata

    assert experiment.name == "fl50_60ants_half_food_sw_wc_8bits_stabilize_from_60best"
    assert args["load_model"].endswith(
        "best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits.pkl"
    )
    assert args["save_best_model"].endswith(
        "best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits_stabilized.pkl"
    )
    assert args["reset_optimizer_on_load"] is True
    assert args["ent_coef"] == pytest.approx(0.0)
    assert args["learning_rate"] == pytest.approx(0.000025)
    assert args["clip_coef"] == pytest.approx(0.05)
    assert args["max_grad_norm"] == pytest.approx(0.25)
    assert args["anneal_lr"] is True
    assert args["training_rollout_temperature"] == pytest.approx(0.525)
    assert args["best_model_metric"] == "eval_mean_delivered_food_per_1000_ant_steps"
    assert args["best_eval_interval"] == 10
    assert args["best_eval_move_temperature"] == pytest.approx(0.525)
    assert args["num_ants"] == 60
    assert args["actor_vision_radius"] == 2
    assert args["agent_identity_types"] == 8
    assert args["total_timesteps"] == 409600

    common_args = workflows.config_common_args(
        args,
        exclude=workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    )
    assert "--reset-optimizer-on-load" in common_args
    assert common_args[common_args.index("--ent-coef") + 1] == "0.0"
    assert common_args[common_args.index("--learning-rate") + 1] == "2.5e-05"
    assert common_args[common_args.index("--clip-coef") + 1] == "0.05"
    assert common_args[common_args.index("--max-grad-norm") + 1] == "0.25"

    assert metadata["source_num_ants"] == 60
    assert metadata["target_num_ants"] == 60
    assert metadata["optimizer_policy"] == "reset_adam_state_on_load"
    assert metadata["stage_update_multiplier"] == pytest.approx(0.0125)
    assert metadata["rollout_policy_temperature"] == pytest.approx(0.525)
    assert metadata["checkpoint_video_policy_temperature"] == pytest.approx(0.525)
    assert metadata["stabilization_controls"]["disable_entropy_bonus"] is True
    assert metadata["stabilization_controls"]["best_eval_move_temperature"] == pytest.approx(
        0.525
    )
    assert (
        metadata["stabilization_controls"]["best_model_metric"]
        == "eval_mean_delivered_food_per_1000_ant_steps"
    )
    assert metadata["deployment_temperature_validation"][
        "selected_move_temperature"
    ] == pytest.approx(0.525)


def test_250x250_distance_autocurriculum_healthy_reset_experiment_matches_culprit_fix() -> None:
    experiment = experiments.load_jax_experiment(
        Path("experiments/half_scale_distance_autocurriculum_250x250_healthy_reset.json")
    )
    args = experiment.args
    metadata = experiment.metadata

    assert experiment.name == "hs250_distance_auto_healthy_reset_lr25e6"
    assert args["critic_architecture"] == "set_cnn"
    assert args["learning_rate"] == pytest.approx(0.000025)
    assert args["distance_autocurriculum"] is True
    assert args["distance_autocurriculum_success_cookies"] == 0
    assert args["distance_progress_normalizer"] == "stage"
    assert args["reset_env_each_update"] is True
    assert args["reset_optimizer_on_load"] is True
    assert args["num_ants"] == 500
    assert args["width"] == 250
    assert args["height"] == 250
    assert args["total_timesteps"] == 640
    assert metadata["default_updates"] == 10
    assert metadata["healthy_control_first_update_deliveries"] == 785
    assert metadata["healthy_control_final_update_deliveries"] == 4822

    common_args = workflows.config_common_args(
        args,
        exclude=workflows.SINGLE_CHECKPOINT_ARG_EXCLUDES,
    )
    assert "--distance-autocurriculum" in common_args
    assert "--reset-env-each-update" in common_args
    assert "--reset-optimizer-on-load" in common_args
    assert common_args[common_args.index("--critic-architecture") + 1] == "set_cnn"
    assert common_args[common_args.index("--learning-rate") + 1] == "2.5e-05"
    assert common_args[common_args.index("--distance-progress-normalizer") + 1] == "stage"
    assert common_args[common_args.index("--distance-autocurriculum-success-cookies") + 1] == "0"
    assert "--run-dir" not in common_args
    assert "--save-model" not in common_args

    parsed = parse_args(
        [
            *common_args,
            "--total-timesteps",
            str(int(metadata["default_updates"]) * int(metadata["update_timesteps"])),
            "--run-dir",
            args["run_dir"],
            "--save-model",
            args["save_model"],
        ]
    )
    assert parsed.critic_architecture == "set_cnn"
    assert parsed.distance_autocurriculum_success_cookies == 0
    assert parsed.distance_progress_normalizer == "stage"
    assert parsed.reset_env_each_update is True
    assert parsed.reset_optimizer_on_load is True


def test_notebook_workflows_reexports_experiment_helpers() -> None:
    assert workflows.load_jax_experiment is experiments.load_jax_experiment
    assert workflows.resolve_project_path is experiments.resolve_project_path
    assert workflows.run_jax_smoke is experiments.run_jax_smoke
