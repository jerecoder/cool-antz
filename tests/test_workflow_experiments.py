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


def test_notebook_workflows_reexports_experiment_helpers() -> None:
    assert workflows.load_jax_experiment is experiments.load_jax_experiment
    assert workflows.resolve_project_path is experiments.resolve_project_path
    assert workflows.run_jax_smoke is experiments.run_jax_smoke
