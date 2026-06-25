from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant_byte_env.cli import main as cli_main
from ant_byte_env.workflows import map_ant


class FakeProgress:
    def update(self, value: int) -> None:
        del value

    def set_postfix(self, **kwargs: str) -> None:
        del kwargs

    def close(self) -> None:
        pass


def good_eval(checkpoint_path: Path, *, num_episodes: int) -> dict[str, object]:
    del checkpoint_path, num_episodes
    metrics = {
        "eval_success_rate": 1.0,
        "eval_mean_delivered_food": 1.0,
        "eval_mean_delivered_fraction": 1.0,
        "eval_mean_episode_return": 1.0,
        "eval_mean_episode_length": 1.0,
        "eval_mean_pickups": 1.0,
        "eval_mean_pickup_to_delivery_rate": 1.0,
        "eval_mean_write_action_rate": 0.1,
        "eval_mean_applied_nonzero_write_rate": 0.1,
    }
    return {
        "deterministic": {"metrics": dict(metrics)},
        "sampled": {"metrics": dict(metrics)},
    }


def bad_eval(checkpoint_path: Path, *, num_episodes: int) -> dict[str, object]:
    payload = good_eval(checkpoint_path, num_episodes=num_episodes)
    for mode in ("deterministic", "sampled"):
        metrics = payload[mode]["metrics"]
        metrics["eval_success_rate"] = 0.0
        metrics["eval_mean_delivered_fraction"] = 0.0
        metrics["eval_mean_pickup_to_delivery_rate"] = 0.0
    return payload


def sampled_bad_eval(checkpoint_path: Path, *, num_episodes: int) -> dict[str, object]:
    payload = good_eval(checkpoint_path, num_episodes=num_episodes)
    metrics = payload["sampled"]["metrics"]
    metrics["eval_success_rate"] = 0.0
    metrics["eval_mean_delivered_fraction"] = 0.0
    metrics["eval_mean_pickup_to_delivery_rate"] = 0.0
    return payload


def fake_train_factory(calls: list[list[str]]):
    def fake_train(argv: list[str], *, progress_callback):
        calls.append(list(argv))
        checkpoint_path = Path(argv[argv.index("--save-model") + 1])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        progress_callback(
            1,
            1,
            {
                "loss": 0.1,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clipfrac": 0.0,
                "grad_norm": 0.0,
                "episode_return": 1.0,
                "env_return": 1.0,
                "completed_episodes": 1.0,
                "terminated_episodes": 1.0,
                "truncated_episodes": 0.0,
                "global_step": 1.0,
                "learning_rate": 1e-4,
            },
        )
        return {"loss": 0.1, "episode_return": 1.0, "global_step": 1.0}

    return fake_train


def test_default_stage_construction_matches_historical_schedule(tmp_path: Path) -> None:
    args = map_ant.parse_args(["--run-dir", str(tmp_path)])

    stages = map_ant.build_curriculum_stages(args)

    assert [stage["name"] for stage in stages] == [
        "4x4_1_ants",
        "6x6_1_ants",
        "8x8_2_ants",
        "10x10_2_ants",
        "12x12_3_ants",
        "16x16_4_ants",
        "20x20_5_ants",
        "25x25_6_ants",
        "32x32_8_ants",
        "40x40_10_ants",
        "50x50_10_ants",
    ]
    assert [stage["food_count"] for stage in stages] == [
        2,
        4,
        6,
        8,
        10,
        14,
        18,
        23,
        30,
        38,
        48,
    ]
    assert [stage["food_sources"] for stage in stages] == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
    ]


def test_stage_parser_rejects_bad_food_source_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 2 comma-separated values"):
        map_ant.parse_args(
            [
                "--run-dir",
                str(tmp_path),
                "--stage-plan",
                "4:1,6:1",
                "--food-sources-by-stage",
                "1",
            ]
        )


def test_gate_eval_modes_reject_bad_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown gate eval mode"):
        map_ant.parse_args(
            [
                "--run-dir",
                str(tmp_path),
                "--stage-plan",
                "4:1",
                "--gate-eval-modes",
                "deterministic,random",
            ]
        )


def test_write_rate_gate_rejects_bad_min_max_band(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        map_ant.parse_args(
            [
                "--run-dir",
                str(tmp_path),
                "--stage-plan",
                "4:1",
                "--gate-min-applied-write-rate",
                "0.5",
                "--gate-max-applied-write-rate",
                "0.2",
            ]
        )


def test_curriculum_forwards_obs_canvas_and_mlp_critic(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    args = map_ant.parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--stage-plan",
            "12:3",
            "--obs-width",
            "50",
            "--obs-height",
            "50",
            "--gate-update-chunk-cap",
            "1",
            "--gate-max-stage-attempts",
            "1",
        ]
    )

    result = map_ant.execute_curriculum(
        args,
        train_main=fake_train_factory(calls),
        evaluate_modes=good_eval,
        progress_factory=lambda label, total: FakeProgress(),
    )

    assert result["status"] == "passed"
    train_argv = calls[0]
    assert train_argv[train_argv.index("--obs-width") + 1] == "50"
    assert train_argv[train_argv.index("--obs-height") + 1] == "50"
    assert train_argv[train_argv.index("--critic-architecture") + 1] == "mlp"
    assert "--random-food" in train_argv
    assert "--random-hub" in train_argv
    assert "--write-while-moving" in train_argv


def test_curriculum_passes_multiple_stages_and_persists_attempts(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    args = map_ant.parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--stage-plan",
            "4:1,6:1",
            "--gate-update-chunk-cap",
            "1",
            "--gate-max-stage-attempts",
            "1",
        ]
    )

    result = map_ant.execute_curriculum(
        args,
        train_main=fake_train_factory(calls),
        evaluate_modes=good_eval,
        progress_factory=lambda label, total: FakeProgress(),
    )

    assert result["status"] == "passed"
    assert len(calls) == 2
    assert (tmp_path / "checkpoints" / "4x4_1_ants" / "attempt_001.pkl").exists()
    assert (tmp_path / "checkpoints" / "6x6_1_ants" / "attempt_001.pkl").exists()
    history = (tmp_path / "gate_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 2
    state = json.loads((tmp_path / "curriculum_state.json").read_text(encoding="utf-8"))
    assert state["stages"]["6x6_1_ants"]["status"] == "passed"


def test_deterministic_only_gate_can_pass_with_sampled_failure(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    args = map_ant.parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--stage-plan",
            "4:1",
            "--gate-update-chunk-cap",
            "1",
            "--gate-max-stage-attempts",
            "1",
            "--gate-eval-modes",
            "deterministic",
        ]
    )

    result = map_ant.execute_curriculum(
        args,
        train_main=fake_train_factory(calls),
        evaluate_modes=sampled_bad_eval,
        progress_factory=lambda label, total: FakeProgress(),
    )

    assert result["status"] == "passed"
    history = (tmp_path / "gate_history.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(history[0])
    assert record["gate"]["modes"] == ["deterministic"]
    assert record["gate"]["passed"] is True
    assert record["sampled"]["eval_mean_delivered_fraction"] == 0.0


def test_failed_stage_can_resume_from_latest_checkpoint_with_new_hyperparams(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    args = map_ant.parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--stage-plan",
            "4:1,6:1",
            "--gate-update-chunk-cap",
            "1",
            "--gate-max-stage-attempts",
            "1",
        ]
    )

    def first_eval(checkpoint_path: Path, *, num_episodes: int) -> dict[str, object]:
        if "6x6_1_ants" in str(checkpoint_path):
            return bad_eval(checkpoint_path, num_episodes=num_episodes)
        return good_eval(checkpoint_path, num_episodes=num_episodes)

    first = map_ant.execute_curriculum(
        args,
        train_main=fake_train_factory(calls),
        evaluate_modes=first_eval,
        progress_factory=lambda label, total: FakeProgress(),
    )
    assert first["status"] == "failed"
    failed_checkpoint = tmp_path / "checkpoints" / "6x6_1_ants" / "attempt_001.pkl"
    assert failed_checkpoint.exists()

    retry_calls: list[list[str]] = []
    retry_args = map_ant.parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--stage-plan",
            "4:1,6:1",
            "--resume",
            "--gamma",
            "0.97",
            "--gate-update-chunk-cap",
            "1",
            "--gate-max-stage-attempts",
            "1",
        ]
    )
    second = map_ant.execute_curriculum(
        retry_args,
        train_main=fake_train_factory(retry_calls),
        evaluate_modes=good_eval,
        progress_factory=lambda label, total: FakeProgress(),
    )

    assert second["status"] == "passed"
    assert len(retry_calls) == 1
    retry_argv = retry_calls[0]
    assert retry_argv[retry_argv.index("--load-model") + 1] == str(failed_checkpoint)
    assert retry_argv[retry_argv.index("--gamma") + 1] == "0.97"
    assert (tmp_path / "checkpoints" / "6x6_1_ants" / "attempt_002.pkl").exists()


def test_train_config_dry_run_validates_map_ant_workflow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(
        [
            "train",
            "jax",
            "--config",
            "experiments/map_ant_gated_mlp_curriculum.json",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["workflow"] == "map_ant_gated_curriculum"
    assert payload["experiment"] == "map_ant_gated_mlp_curriculum"
    assert payload["resolved_args"]["critic_architecture"] == "mlp"
    assert payload["resolved_args"]["actor_vision_radius"] == 1
    assert payload["resolved_args"]["random_food"] is True
    assert payload["resolved_args"]["random_hub"] is True
    assert payload["stages"][-1]["name"] == "50x50_10_ants"
