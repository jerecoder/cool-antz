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
    assert plan["evaluation"]["sampled_episodes"] == 2
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
        assert num_episodes in {2, 4}
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
