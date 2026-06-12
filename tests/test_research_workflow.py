from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant_byte_env.cli import main as cli_main
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.rendering import infer_checkpoint_backend
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
            "experiments/communication_bits.json",
            "--dry-run",
            "--",
            "--total-timesteps",
            "8",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["backend"] == "jax"
    assert payload["experiment"] == "communication_bits"
    assert payload["resolved_args"]["load_model"].endswith(
        "runs/notebooks/forage_curriculum/checkpoints/jax_mappo_forage_stage1_15x15.pkl"
    )
    assert payload["resolved_args"]["total_timesteps"] == 8


def test_run_helpers_create_manifest_and_metrics(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path, "demo", run_id="fixed")

    write_json(run_dir / "config.json", {"args": {"seed": 3}})
    append_metrics(run_dir / "metrics.jsonl", {"update": 1, "loss": 0.5})

    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "media").is_dir()
    assert json.loads((run_dir / "config.json").read_text())["args"]["seed"] == 3
    assert json.loads((run_dir / "metrics.jsonl").read_text()) == {"loss": 0.5, "update": 1}


def test_result_indexer_reads_vault_metadata(tmp_path: Path) -> None:
    entry_dir = tmp_path / "runs" / "communication_bits" / "vault" / "20260611T000000Z"
    entry_dir.mkdir(parents=True)
    (entry_dir / "rollout.mp4").write_bytes(b"video")
    (entry_dir / "metadata.json").write_text(
        json.dumps(
            {
                "created_at": "2026-06-11T00:00:00Z",
                "title": "Best rollout",
                "description": "A curated run.",
                "assets": [{"filename": "rollout.mp4"}],
                "metadata": {"stage": "15x15"},
            }
        ),
        encoding="utf-8",
    )

    payload = index_result_metadata(tmp_path / "runs", tmp_path / "curated" / "index.json")

    assert payload["entry_count"] == 1
    assert payload["entries"][0]["title"] == "Best rollout"
    assert payload["entries"][0]["assets"][0]["size_bytes"] == 5


def test_render_helpers_are_notebook_independent() -> None:
    assert infer_checkpoint_backend(Path("policy.pt")) == "torch"
    assert infer_checkpoint_backend(Path("policy.pkl")) == "jax"
    with pytest.raises(ValueError, match="suffix"):
        infer_checkpoint_backend(Path("policy.bin"))
