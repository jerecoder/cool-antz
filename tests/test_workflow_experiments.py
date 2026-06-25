from pathlib import Path

import pytest

from ant_byte_env import notebook_workflows as workflows
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


def test_notebook_workflows_reexports_experiment_helpers() -> None:
    assert workflows.load_jax_experiment is experiments.load_jax_experiment
    assert workflows.resolve_project_path is experiments.resolve_project_path
    assert workflows.run_jax_smoke is experiments.run_jax_smoke
