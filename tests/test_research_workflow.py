from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant_byte_env.cli import main as cli_main
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.rendering import infer_checkpoint_backend
from ant_byte_env.results import index_result_metadata
from ant_byte_env.runs import append_metrics, prepare_run_dir, write_json
from ant_byte_env.autoresearch import (
    AutoresearchResourceError,
    assert_autoresearch_resources_available,
    build_communication_sweep_plan,
    execute_communication_sweep_plan,
    rank_communication_gate_probes,
)
from ant_byte_env.autocurriculum_autoresearch import (
    build_autocurriculum_sweep_plan,
    execute_autocurriculum_sweep_plan,
)
from ant_byte_env.forage_autoresearch import (
    build_forage_50x50_sweep_plan,
    execute_forage_50x50_sweep_plan,
)


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
        "runs/notebooks/forage_curriculum/checkpoints/jax_mappo_forage_stage1_25x25.pkl"
    )
    assert payload["resolved_args"]["obs_width"] == 50
    assert payload["resolved_args"]["obs_height"] == 50
    assert payload["resolved_args"]["write_while_moving"] is True
    assert payload["resolved_args"]["total_timesteps"] == 8


def test_direct_goal_baseline_config_resolves_final_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(
        [
            "train",
            "jax",
            "--config",
            "experiments/direct_goal_baseline.json",
            "--dry-run",
            "--",
            "--total-timesteps",
            "8",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    resolved = payload["resolved_args"]
    assert exit_code == 0
    assert payload["experiment"] == "direct_goal_baseline"
    assert resolved["width"] == 50
    assert resolved["height"] == 50
    assert resolved["obs_width"] == 50
    assert resolved["obs_height"] == 50
    assert resolved["actor_vision_radius"] == 1
    assert resolved["num_ants"] == 10
    assert resolved["write_bits"] == 5
    assert resolved["write_while_moving"] is True
    assert resolved["pickup_bonus"] == 0.0
    assert resolved["distance_bonus"] == 0.0


def test_communication_autoresearch_matrix_resolves_jax_args() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    matrix = json.loads(Path("autoresearch/communication_sweep.json").read_text())
    base_spec = load_experiment_config(Path(matrix["base_config"]))
    run_root = str(matrix["run_root"])

    assert matrix["screening_bit_stages"] == [2, 3]
    assert matrix["final_bit_stages"] == [2, 3, 5, 8]
    assert [entry["id"] for entry in matrix["phases"]["horizon"]] == ["H0", "H1", "H2", "H3"]
    assert [entry["id"] for entry in matrix["phases"]["reward"]] == ["R0", "R1", "R2", "R3"]
    assert [entry["id"] for entry in matrix["phases"]["transfer"]] == ["T0", "T1", "T2"]
    assert [entry["id"] for entry in matrix["phases"]["final"]] == ["F1", "F2", "F3"]
    assert [entry["id"] for entry in matrix["phases"]["promoted_validation"]] == [
        "PV1",
        "PV2",
        "PV3",
    ]
    assert [entry["id"] for entry in matrix["phases"]["polish_length"]] == [
        "PL0",
        "PL1",
        "PL2",
        "PL3",
        "PL4",
        "PL5",
        "PL6",
        "PL7",
        "PL8",
        "PL9",
        "PL10",
        "PL11",
        "PL12",
        "PL13",
        "PL14",
        "PL15",
    ]
    assert [entry["id"] for entry in matrix["phases"]["polish_refine"]] == [
        "PR1",
        "PR2",
        "PR3",
        "PR4",
        "PR5",
        "PR6",
        "PR7",
    ]
    assert [entry["id"] for entry in matrix["phases"]["polish_gate"]] == [
        "PG1",
        "PG2",
        "PG3",
    ]

    all_ids: set[str] = set()
    for entries in matrix["phases"].values():
        for entry in entries:
            assert entry["id"] not in all_ids
            all_ids.add(entry["id"])
            assert str(entry["run_dir"]).startswith(f"{run_root}/")
            assert str(entry["probe_output_dir"]).startswith(f"{entry['run_dir']}/")
            merged_args = {**base_spec.args, **entry["args"]}
            parsed = parse_args(config_args_to_argv(merged_args))
            assert parsed.random_food
            assert parsed.random_hub
            assert parsed.obs_width == 50
            assert parsed.obs_height == 50
            assert parsed.write_while_moving
            assert parsed.write_head_transfer in {"repeat", "reset", "neutral-new"}
            for post_stage in entry.get("post_stages", []):
                post_parsed = parse_args(
                    config_args_to_argv({**merged_args, **post_stage["args"]})
                )
                assert post_parsed.random_food
                assert post_parsed.random_hub
                assert post_parsed.obs_width == 50
                assert post_parsed.obs_height == 50
                assert post_parsed.write_while_moving

    for entry in matrix["phases"]["horizon"]:
        parsed = parse_args(config_args_to_argv({**base_spec.args, **entry["args"]}))
        assert (
            parsed.num_steps * parsed.num_envs * int(entry["global_update_cap"])
            == matrix["screening_env_steps_per_stage"]
        )


def test_forage_50x50_autoresearch_matrix_keeps_no_cheat_jax_args() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    matrix = json.loads(Path("autoresearch/forage_50x50_sweep.json").read_text())
    base_spec = load_experiment_config(Path(matrix["base_config"]))
    run_root = str(matrix["run_root"])

    assert matrix["constraints"]["num_ants"] == 1
    assert matrix["constraints"]["actor_vision_radius"] == 1
    assert matrix["screening_stage_sizes"][-1] == 50
    assert matrix["final_stage_sizes"] == list(range(4, 51))
    assert [entry["id"] for entry in matrix["phases"]["reward"]] == ["R0", "R1", "R2", "R3"]
    assert [entry["id"] for entry in matrix["phases"]["algorithm"]] == [
        "H0",
        "H1",
        "H2",
        "H3",
        "H4",
    ]
    assert [entry["id"] for entry in matrix["phases"]["memory"]] == ["M1", "M2", "M3", "M4"]
    assert [entry["id"] for entry in matrix["phases"]["architecture"]] == ["A0", "A1", "A2"]
    assert [entry["id"] for entry in matrix["phases"]["final"]] == ["F1", "F2", "F3"]

    all_ids: set[str] = set()
    for entries in matrix["phases"].values():
        for entry in entries:
            assert entry["id"] not in all_ids
            all_ids.add(entry["id"])
            assert str(entry["run_dir"]).startswith(f"{run_root}/")
            merged_args = {**base_spec.args, **entry["args"]}
            parsed = parse_args(config_args_to_argv(merged_args))
            assert parsed.num_ants == 1
            assert parsed.actor_vision_radius == 1
            assert parsed.obs_width == 50
            assert parsed.obs_height == 50
            if str(entry["id"]).startswith("M"):
                assert parsed.write_bits in {2, 3}
                assert parsed.write_bit_entropy_bonus >= 0.0
            else:
                assert parsed.write_bits == 1
            assert parsed.random_food
            assert parsed.random_hub
            assert parsed.write_while_moving


def test_autocurriculum_autoresearch_matrix_keeps_no_cheat_jax_args() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    matrix = json.loads(Path("autoresearch/autocurriculum_sweep.json").read_text())
    base_spec = load_experiment_config(Path(matrix["base_config"]))
    run_root = str(matrix["run_root"])

    assert matrix["constraints"]["num_ants"] == 1
    assert matrix["constraints"]["actor_vision_radius"] == 1
    assert matrix["constraints"]["write_bits"] == 1
    assert [entry["id"] for entry in matrix["phases"]["reward"]] == [
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
    ]
    assert [entry["id"] for entry in matrix["phases"]["algorithm"]] == [
        "H0",
        "H1",
        "H2",
        "H3",
    ]
    assert [entry["id"] for entry in matrix["phases"]["final"]] == ["F1"]
    reward_by_id = {entry["id"]: entry for entry in matrix["phases"]["reward"]}
    assert reward_by_id["R4"]["args"]["distance_bonus"] == 0.0
    assert reward_by_id["R4"]["args"]["write_bit_penalty"] > 0.0
    assert reward_by_id["R5"]["args"]["pickup_bonus"] < reward_by_id["R0"]["args"]["pickup_bonus"]
    assert reward_by_id["R6"]["args"]["write_while_moving"] is False
    assert reward_by_id["R7"]["args"]["stage_completion_bonus"] > 0.0
    for entry in [
        *matrix["phases"]["algorithm"],
        *matrix["phases"]["final"],
    ]:
        assert entry["args"]["distance_bonus"] == 0.0

    all_ids: set[str] = set()
    for entries in matrix["phases"].values():
        for entry in entries:
            assert entry["id"] not in all_ids
            all_ids.add(entry["id"])
            assert str(entry["run_dir"]).startswith(f"{run_root}/")
            merged_args = {**base_spec.args, **entry["args"]}
            parsed = parse_args(config_args_to_argv(merged_args))
            assert parsed.autocurriculum is True
            assert parsed.width == 50
            assert parsed.height == 50
            assert parsed.num_ants == 1
            assert parsed.actor_vision_radius == 1
            assert parsed.write_bits == 1
            assert parsed.random_food
            assert parsed.random_hub
            assert parsed.write_while_moving is (entry["id"] != "R6")


def test_autocurriculum_sweep_plan_builds_training_probe_and_wandb_notes() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    plan = build_autocurriculum_sweep_plan(
        phase="reward",
        run_id="R2",
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        probe_rollout_steps=12,
        probe_num_envs=2,
        render_rollout=False,
        wandb_mode="offline",
    )
    parsed = parse_args(plan["common_args"])

    assert plan["id"] == "R2"
    assert plan["phase"] == "reward"
    assert plan["global_update_cap"] == 2
    assert plan["update_timesteps"] == 4
    assert plan["total_train_env_steps"] == 8
    assert plan["checkpoint"].endswith("reward/R2/checkpoints/model.pkl")
    assert plan["probe"] == {"rollout_steps": 12, "num_envs": 2, "seed_offset": 3_000_000}
    assert plan["rollout"]["enabled"] is False
    assert plan["wandb"]["project"] == "cool-antz"
    assert plan["wandb"]["mode"] == "offline"
    assert "progress shaping" in plan["wandb"]["notes"]
    assert "--wandb-notes" in plan["training_argv"]
    assert parsed.autocurriculum is True
    assert parsed.num_ants == 1
    assert parsed.actor_vision_radius == 1
    assert parsed.write_bits == 1
    assert parsed.distance_bonus == 0.05


def test_autocurriculum_sweep_plan_rejects_cheating_actor_radius(tmp_path: Path) -> None:
    matrix = json.loads(Path("autoresearch/autocurriculum_sweep.json").read_text())
    matrix["phases"]["reward"][0]["args"]["actor_vision_radius"] = 2
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(ValueError, match="actor_vision_radius"):
        build_autocurriculum_sweep_plan(
            matrix_path=matrix_path,
            phase="reward",
            run_id="R0",
        )


def test_autocurriculum_sweep_plan_can_continue_from_checkpoint(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "source.pkl"

    plan = build_autocurriculum_sweep_plan(
        phase="reward",
        run_id="R0",
        run_root=tmp_path / "continued",
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        probe_rollout_steps=12,
        probe_num_envs=2,
        render_rollout=False,
        wandb_mode="offline",
        load_model=source_checkpoint,
    )

    assert plan["load_model"] == str(source_checkpoint)
    assert "--load-model" in plan["training_argv"]
    assert str(source_checkpoint) in plan["training_argv"]
    assert "Continuation source checkpoint" in plan["wandb"]["notes"]


def test_autocurriculum_sweep_plan_rejects_in_place_continuation() -> None:
    with pytest.raises(ValueError, match="load_model"):
        build_autocurriculum_sweep_plan(
            phase="reward",
            run_id="R0",
            load_model=Path("runs/autoresearch/autocurriculum/reward/R0/checkpoints/model.pkl"),
        )


def test_execute_autocurriculum_sweep_plan_runs_train_probe_and_render(
    tmp_path: Path,
) -> None:
    plan = build_autocurriculum_sweep_plan(
        phase="reward",
        run_id="R2",
        run_root=tmp_path / "autocurriculum",
        global_update_cap=1,
        num_envs=1,
        num_steps=4,
        probe_rollout_steps=12,
        probe_num_envs=2,
        render_rollout=True,
        max_render_frames=8,
        wandb_mode="disabled",
    )
    calls: dict[str, object] = {}

    def fake_train_main(argv: list[str]) -> dict[str, float]:
        calls["train_argv"] = argv
        checkpoint = Path(str(plan["checkpoint"]))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"global_step": 4.0, "autocurriculum_max_active_size": 5.0}

    def fake_probe_checkpoint(checkpoint_path: Path, **kwargs: object) -> dict[str, object]:
        calls["probe"] = {"checkpoint": checkpoint_path, **kwargs}
        return {
            "checkpoint": str(checkpoint_path),
            "rollout_steps": 12,
            "num_envs": 2,
            "env_steps": 24,
            "metrics": {
                "autocurriculum_max_active_size": 6.0,
                "delivery_events_per_1000_env_steps": 42.0,
            },
        }

    def fake_render_checkpoint(
        checkpoint_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        calls["render"] = {"checkpoint": checkpoint_path, "output": output_path, **kwargs}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    summary = execute_autocurriculum_sweep_plan(
        plan,
        train_main=fake_train_main,
        probe_checkpoint=fake_probe_checkpoint,
        render_checkpoint_fn=fake_render_checkpoint,
        check_resources=False,
    )

    assert summary["resumed"] is False
    assert calls["probe"]["rollout_steps"] == 12
    assert calls["probe"]["num_envs"] == 2
    assert calls["render"]["max_frames"] == 8
    assert calls["render"]["policy_temperature"] == 1.0
    assert summary["probe"]["metrics"]["autocurriculum_max_active_size"] == 6.0
    assert summary["rollout_path"].endswith("sampled_autocurriculum_rollout.mp4")
    assert summary["wandb"]["enabled"] is False
    assert json.loads(
        (tmp_path / "autocurriculum" / "reward" / "R2" / "sweep_summary.json").read_text()
    )["id"] == "R2"


def test_forage_50x50_memory_plan_keeps_actor_local_with_wider_self_memory() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    plan = build_forage_50x50_sweep_plan(
        phase="memory",
        run_id="M1",
        stage_sizes=[4, 8, 50],
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )
    parsed = parse_args(plan["common_args"])

    assert plan["phase"] == "memory"
    assert parsed.num_ants == 1
    assert parsed.actor_vision_radius == 1
    assert parsed.obs_width == 50
    assert parsed.obs_height == 50
    assert parsed.write_bits == 2
    assert parsed.write_bit_entropy_bonus == 0.02
    assert parsed.random_food
    assert parsed.random_hub
    assert "actor_observation" in plan["no_cheat_invariants"]


def test_forage_50x50_m4_uses_lower_noise_two_bit_memory() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    plan = build_forage_50x50_sweep_plan(
        phase="memory",
        run_id="M4",
        stage_sizes=[4, 8, 50],
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
    )
    parsed = parse_args(plan["common_args"])

    assert parsed.num_ants == 1
    assert parsed.actor_vision_radius == 1
    assert parsed.write_bits == 2
    assert parsed.hidden_size == 128
    assert parsed.write_bit_entropy_bonus == 0.005
    assert parsed.random_food
    assert parsed.random_hub


def test_forage_50x50_sweep_plan_builds_curriculum_with_wandb_milestones() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    plan = build_forage_50x50_sweep_plan(
        phase="reward",
        run_id="R1",
        stage_sizes=[4, 6, 50],
        global_update_cap=2,
        num_envs=1,
        num_steps=4,
        wandb_mode="offline",
    )

    assert plan["id"] == "R1"
    assert plan["stage_sizes"] == [4, 6, 50]
    assert [stage["name"] for stage in plan["stages"]] == ["4x4", "6x6", "50x50"]
    assert plan["global_update_cap"] == 2
    assert plan["update_timesteps_per_stage"] == 4
    assert plan["total_train_env_steps"] == 24
    assert plan["final_checkpoint"].endswith(
        "reward/R1/checkpoints/jax_mappo_forage_stage1_50x50.pkl"
    )
    assert plan["wandb"]["project"] == "cool-antz"
    assert plan["wandb"]["mode"] == "offline"
    assert plan["wandb"]["video_stage_names"] == ["25x25", "40x40", "50x50"]
    assert plan["no_cheat_invariants"]["actor_vision_radius"] == 1

    parsed = parse_args(plan["common_args"])
    assert parsed.num_ants == 1
    assert parsed.actor_vision_radius == 1
    assert parsed.obs_width == 50
    assert parsed.obs_height == 50
    assert parsed.pickup_bonus == 0.25
    assert parsed.distance_bonus == 0.02
    assert parsed.total_timesteps == 100_000


def test_forage_50x50_sweep_plan_rejects_cheating_actor_radius(tmp_path: Path) -> None:
    matrix = json.loads(Path("autoresearch/forage_50x50_sweep.json").read_text())
    matrix["phases"]["reward"][0]["args"]["actor_vision_radius"] = 2
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(ValueError, match="actor_vision_radius"):
        build_forage_50x50_sweep_plan(
            matrix_path=matrix_path,
            phase="reward",
            run_id="R0",
        )


def test_execute_forage_50x50_sweep_plan_runs_curriculum(tmp_path: Path) -> None:
    plan = build_forage_50x50_sweep_plan(
        phase="reward",
        run_id="R1",
        run_root=tmp_path / "forage",
        stage_sizes=[4, 5],
        global_update_cap=1,
        num_envs=1,
        num_steps=4,
        wandb_mode="disabled",
        wandb_video_stage_names=["5x5"],
    )
    curriculum_calls: list[dict[str, object]] = []

    def fake_train_main(args: list[str], progress_callback=None) -> dict[str, float]:
        del args, progress_callback
        return {"global_step": 1.0}

    def fake_run_curriculum(**kwargs: object) -> dict[str, object]:
        curriculum_calls.append(kwargs)
        checkpoint_dir = Path(str(kwargs["checkpoint_dir"]))
        final_checkpoint = checkpoint_dir / "jax_mappo_forage_stage1_5x5.pkl"
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        final_checkpoint.write_bytes(b"checkpoint")
        return {
            "stage_metrics": [{"stage_name": "5x5", "episode_return": 1.0}],
            "stage_checkpoint_paths": [checkpoint_dir / "jax_mappo_forage_stage1_5x5.pkl"],
            "final_checkpoint_path": final_checkpoint,
            "final_train_metrics": {"global_step": 1.0},
        }

    summary = execute_forage_50x50_sweep_plan(
        plan,
        train_main=fake_train_main,
        run_curriculum=fake_run_curriculum,
        check_resources=False,
    )

    assert len(curriculum_calls) == 1
    call = curriculum_calls[0]
    assert call["train_main"] is fake_train_main
    assert call["wandb_mode"] == "disabled"
    assert call["wandb_video_stage_names"] == ["5x5"]
    assert [stage["name"] for stage in call["stages"]] == ["4x4", "5x5"]
    assert summary["resumed"] is False
    assert summary["curriculum"]["final_checkpoint_path"].endswith(
        "jax_mappo_forage_stage1_5x5.pkl"
    )
    assert json.loads((tmp_path / "forage" / "reward" / "R1" / "sweep_plan.json").read_text())[
        "id"
    ] == "R1"
    assert json.loads(
        (tmp_path / "forage" / "reward" / "R1" / "sweep_summary.json").read_text()
    )["summary_path"].endswith("sweep_summary.json")


def test_communication_sweep_plan_builds_promoted_validation_post_stages() -> None:
    plan = build_communication_sweep_plan(
        phase="promoted_validation",
        run_id="PV1",
        probe_episodes=4,
        render_rollouts=False,
    )

    assert plan["bit_stages"] == [2, 3, 5, 8]
    assert plan["env_steps_per_stage"] == 12_800_000
    assert plan["total_train_env_steps"] == 60_800_000
    assert [command["stage_name"] for command in plan["train_commands"]] == [
        "2_bits",
        "3_bits",
        "5_bits",
        "8_bits",
        "8_bits_consolidated",
        "8_bits_polished",
    ]
    assert [command["stage_kind"] for command in plan["train_commands"]] == [
        "bit",
        "bit",
        "bit",
        "bit",
        "post",
        "post",
    ]
    consolidated = plan["train_commands"][4]
    polished = plan["train_commands"][5]
    assert consolidated["global_update_cap"] == 5000
    assert consolidated["env_steps"] == 6_400_000
    assert consolidated["source_checkpoint"] == plan["train_commands"][3]["checkpoint"]
    assert polished["global_update_cap"] == 2500
    assert polished["env_steps"] == 3_200_000
    assert polished["source_checkpoint"] == consolidated["checkpoint"]
    assert plan["probe_command"]["checkpoint"] == polished["checkpoint"]
    assert plan["probe_command"]["output_dir"].endswith("promoted/PV1/probe_eval4")


def test_communication_sweep_plan_builds_polish_length_probe() -> None:
    plan = build_communication_sweep_plan(
        phase="polish_length",
        run_id="PL0",
        probe_episodes=4,
        render_rollouts=False,
    )

    assert plan["bit_stages"] == [8]
    assert plan["global_update_cap"] == 5000
    assert plan["env_steps_per_stage"] == 6_400_000
    assert plan["total_train_env_steps"] == 6_400_000
    assert len(plan["train_commands"]) == 1
    command = plan["train_commands"][0]
    assert command["stage_name"] == "8_bits"
    assert command["write_bits"] == 8
    assert command["global_update_cap"] == 5000
    assert command["source_checkpoint"].endswith(
        "promoted/PV3/8_bits_consolidated/checkpoints/model.pkl"
    )
    assert command["checkpoint"].endswith("polish_length/PL0/8_bits/checkpoints/model.pkl")
    assert plan["probe_command"]["checkpoint"] == command["checkpoint"]
    assert plan["probe_command"]["output_dir"].endswith("polish_length/PL0/probe_eval4")

    short_plan = build_communication_sweep_plan(
        phase="polish_length",
        run_id="PL1",
        probe_episodes=4,
        render_rollouts=False,
    )
    assert short_plan["global_update_cap"] == 1250
    assert short_plan["env_steps_per_stage"] == 1_600_000
    assert short_plan["total_train_env_steps"] == 1_600_000
    assert short_plan["train_commands"][0]["source_checkpoint"].endswith(
        "promoted/PV3/8_bits_consolidated/checkpoints/model.pkl"
    )

    replicate_plan = build_communication_sweep_plan(
        phase="polish_length",
        run_id="PL2",
        probe_episodes=4,
        render_rollouts=False,
    )
    assert replicate_plan["global_update_cap"] == 1250
    assert replicate_plan["env_steps_per_stage"] == 1_600_000
    assert replicate_plan["train_commands"][0]["source_checkpoint"].endswith(
        "promoted/PV1/8_bits_consolidated/checkpoints/model.pkl"
    )
    assert replicate_plan["probe_command"]["output_dir"].endswith(
        "polish_length/PL2/probe_eval4"
    )

    seed_plan = build_communication_sweep_plan(
        phase="polish_length",
        run_id="PL4",
        probe_episodes=4,
        render_rollouts=False,
    )
    assert seed_plan["global_update_cap"] == 1250
    assert seed_plan["env_steps_per_stage"] == 1_600_000
    assert seed_plan["train_commands"][0]["source_checkpoint"].endswith(
        "promoted/PV3/8_bits_consolidated/checkpoints/model.pkl"
    )
    seed_indices = [
        index
        for index, value in enumerate(seed_plan["train_commands"][0]["training_argv"])
        if value == "--seed"
    ]
    assert seed_indices
    seed_index = seed_indices[-1]
    assert seed_plan["train_commands"][0]["training_argv"][seed_index + 1] == "804"
    assert seed_plan["probe_command"]["output_dir"].endswith(
        "polish_length/PL4/probe_eval4"
    )

    cross_source_plan = build_communication_sweep_plan(
        phase="polish_length",
        run_id="PL7",
        probe_episodes=4,
        render_rollouts=False,
    )
    assert cross_source_plan["global_update_cap"] == 1250
    assert cross_source_plan["train_commands"][0]["source_checkpoint"].endswith(
        "promoted/PV1/8_bits_consolidated/checkpoints/model.pkl"
    )
    cross_seed_indices = [
        index
        for index, value in enumerate(cross_source_plan["train_commands"][0]["training_argv"])
        if value == "--seed"
    ]
    assert cross_seed_indices
    cross_seed_index = cross_seed_indices[-1]
    assert cross_source_plan["train_commands"][0]["training_argv"][cross_seed_index + 1] == "805"
    assert cross_source_plan["probe_command"]["output_dir"].endswith(
        "polish_length/PL7/probe_eval4"
    )

    pv2_seed_plan = build_communication_sweep_plan(
        phase="polish_length",
        run_id="PL9",
        probe_episodes=4,
        render_rollouts=False,
    )
    assert pv2_seed_plan["global_update_cap"] == 1250
    assert pv2_seed_plan["train_commands"][0]["source_checkpoint"].endswith(
        "promoted/PV2/8_bits_consolidated/checkpoints/model.pkl"
    )
    pv2_seed_indices = [
        index
        for index, value in enumerate(pv2_seed_plan["train_commands"][0]["training_argv"])
        if value == "--seed"
    ]
    assert pv2_seed_indices
    pv2_seed_index = pv2_seed_indices[-1]
    assert pv2_seed_plan["train_commands"][0]["training_argv"][pv2_seed_index + 1] == "803"
    assert pv2_seed_plan["probe_command"]["output_dir"].endswith(
        "polish_length/PL9/probe_eval4"
    )

    pv1_seed_plan = build_communication_sweep_plan(
        phase="polish_length",
        run_id="PL12",
        probe_episodes=4,
        render_rollouts=False,
    )
    assert pv1_seed_plan["global_update_cap"] == 1250
    assert pv1_seed_plan["train_commands"][0]["source_checkpoint"].endswith(
        "promoted/PV1/8_bits_consolidated/checkpoints/model.pkl"
    )
    pv1_seed_indices = [
        index
        for index, value in enumerate(pv1_seed_plan["train_commands"][0]["training_argv"])
        if value == "--seed"
    ]
    assert pv1_seed_indices
    pv1_seed_index = pv1_seed_indices[-1]
    assert pv1_seed_plan["train_commands"][0]["training_argv"][pv1_seed_index + 1] == "802"
    assert pv1_seed_plan["probe_command"]["output_dir"].endswith(
        "polish_length/PL12/probe_eval4"
    )


def test_communication_sweep_plan_builds_staged_train_and_probe_commands() -> None:
    from ant_byte_env.training.jax_mappo.cli import parse_args

    plan = build_communication_sweep_plan(
        phase="horizon",
        run_id="H0",
        probe_episodes=2,
        render_rollouts=False,
    )
    base_spec = load_experiment_config(Path(plan["base_config"]))

    assert plan["bit_stages"] == [2, 3]
    assert plan["env_steps_per_stage"] == 2_560_000
    assert len(plan["train_commands"]) == 2
    first, second = plan["train_commands"]
    assert first["write_bits"] == 2
    assert second["write_bits"] == 3
    assert first["source_checkpoint"].endswith("jax_mappo_forage_stage1_25x25.pkl")
    assert second["source_checkpoint"] == first["checkpoint"]
    assert second["checkpoint"].endswith("horizon/H0/3_bits/checkpoints/model.pkl")
    assert plan["probe_command"]["checkpoint"] == second["checkpoint"]
    assert plan["probe_command"]["output_dir"].endswith("horizon/H0/probe")
    assert plan["probe_command"]["argv"][-1] == "--no-render"
    assert plan["probe_command"]["options"]["max_render_frames"] == 300

    for command in plan["train_commands"]:
        argv = command["argv"]
        assert argv[:5] == [
            "ant-byte",
            "train",
            "jax",
            "--config",
            "experiments/communication_bits.json",
        ]
        override_argv = argv[argv.index("--") + 1 :]
        parsed = parse_args([*config_args_to_argv(base_spec.args), *override_argv])
        assert parsed.total_timesteps == plan["env_steps_per_stage"]
        assert parsed.run_dir.as_posix() == command["run_dir"]
        assert parsed.load_model.as_posix() == command["source_checkpoint"]
        assert parsed.write_bits == command["write_bits"]
        assert parsed.obs_width == 50
        assert parsed.obs_height == 50
        assert parsed.write_while_moving
        direct_parsed = parse_args(command["training_argv"])
        assert direct_parsed.write_bits == command["write_bits"]
        assert direct_parsed.obs_width == 50
        assert direct_parsed.obs_height == 50
        assert direct_parsed.write_while_moving


def test_communication_sweep_plan_builds_probe_only_polish_gate() -> None:
    plan = build_communication_sweep_plan(
        phase="polish_gate",
        run_id="PG2",
        probe_episodes=16,
        render_rollouts=False,
    )

    assert plan["bit_stages"] == []
    assert plan["global_update_cap"] is None
    assert plan["env_steps_per_stage"] == 0
    assert plan["total_train_env_steps"] == 0
    assert plan["train_commands"] == []
    assert plan["probe_command"]["checkpoint"].endswith(
        "polish_length/PL10/8_bits/checkpoints/model.pkl"
    )
    assert plan["probe_command"]["output_dir"].endswith("polish_gate/PG2/probe_eval16")
    assert plan["probe_command"]["options"]["num_episodes"] == 16
    assert plan["probe_command"]["argv"][-1] == "--no-render"


def test_communication_sweep_plan_builds_polish_refine_probe() -> None:
    plan = build_communication_sweep_plan(
        phase="polish_refine",
        run_id="PR1",
        probe_episodes=4,
        render_rollouts=False,
    )

    assert plan["bit_stages"] == [8]
    assert plan["global_update_cap"] == 250
    assert plan["env_steps_per_stage"] == 320_000
    assert plan["total_train_env_steps"] == 320_000
    assert len(plan["train_commands"]) == 1
    command = plan["train_commands"][0]
    assert command["source_checkpoint"].endswith(
        "polish_length/PL5/8_bits/checkpoints/model.pkl"
    )
    assert command["checkpoint"].endswith("polish_refine/PR1/8_bits/checkpoints/model.pkl")
    assert plan["probe_command"]["output_dir"].endswith("polish_refine/PR1/probe_eval4")

    guarded_plan = build_communication_sweep_plan(
        phase="polish_refine",
        run_id="PR3",
        probe_episodes=4,
        render_rollouts=False,
    )
    guarded_argv = guarded_plan["train_commands"][0]["training_argv"]
    bit_entropy_index = [
        index for index, value in enumerate(guarded_argv) if value == "--write-bit-entropy-bonus"
    ][-1]
    ent_coef_index = [index for index, value in enumerate(guarded_argv) if value == "--ent-coef"][
        -1
    ]
    assert guarded_argv[bit_entropy_index + 1] == "0.02"
    assert guarded_argv[ent_coef_index + 1] == "0.001"

    pv1_plan = build_communication_sweep_plan(
        phase="polish_refine",
        run_id="PR4",
        probe_episodes=4,
        render_rollouts=False,
    )
    assert pv1_plan["train_commands"][0]["source_checkpoint"].endswith(
        "polish_length/PL7/8_bits/checkpoints/model.pkl"
    )

    pv2_plan = build_communication_sweep_plan(
        phase="polish_refine",
        run_id="PR7",
        probe_episodes=4,
        render_rollouts=False,
    )
    assert pv2_plan["train_commands"][0]["source_checkpoint"].endswith(
        "polish_length/PL10/8_bits/checkpoints/model.pkl"
    )
    pv2_argv = pv2_plan["train_commands"][0]["training_argv"]
    pv2_seed_index = [index for index, value in enumerate(pv2_argv) if value == "--seed"][-1]
    pv2_bit_entropy_index = [
        index for index, value in enumerate(pv2_argv) if value == "--write-bit-entropy-bonus"
    ][-1]
    assert pv2_argv[pv2_seed_index + 1] == "804"
    assert pv2_argv[pv2_bit_entropy_index + 1] == "0.02"


def test_communication_sweep_plan_can_override_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "smoke" / "communication_bits"
    plan = build_communication_sweep_plan(
        phase="horizon",
        run_id="H0",
        run_root=run_root,
        bit_stages=[2],
        global_update_cap=1,
        num_envs=1,
        num_steps=1,
        probe_episodes=1,
    )

    assert Path(plan["run_dir"]) == run_root / "horizon" / "H0"
    assert Path(plan["probe_output_dir"]) == run_root / "horizon" / "H0" / "probe"
    assert Path(plan["train_commands"][0]["run_dir"]) == run_root / "horizon" / "H0" / "2_bits"
    assert Path(plan["probe_command"]["output_dir"]) == run_root / "horizon" / "H0" / "probe"
    assert plan["probe_command"]["options"]["render_rollouts"] is False
    assert plan["probe_command"]["argv"][-1] == "--no-render"


def test_execute_communication_sweep_plan_runs_stages_and_probe(tmp_path: Path) -> None:
    matrix = json.loads(Path("autoresearch/communication_sweep.json").read_text())
    matrix["phases"]["horizon"][0]["run_dir"] = str(tmp_path / "H0")
    matrix["phases"]["horizon"][0]["probe_output_dir"] = str(tmp_path / "H0" / "probe")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    plan = build_communication_sweep_plan(
        matrix_path=matrix_path,
        phase="horizon",
        run_id="H0",
        bit_stages=[2],
        global_update_cap=1,
        num_envs=1,
        num_steps=1,
        probe_episodes=1,
        render_rollouts=False,
    )
    train_argvs: list[list[str]] = []
    probe_calls: list[tuple[Path, dict[str, object]]] = []

    def fake_train_main(argv: list[str]) -> dict[str, float]:
        train_argvs.append(argv)
        return {"global_step": float(len(train_argvs))}

    def fake_probe_checkpoint(checkpoint: Path, **kwargs: object) -> dict[str, object]:
        probe_calls.append((checkpoint, kwargs))
        return {"probe_path": str(Path(kwargs["output_dir"]) / "communication_probe.json")}

    summary = execute_communication_sweep_plan(
        plan,
        train_main=fake_train_main,
        probe_checkpoint=fake_probe_checkpoint,
        check_resources=False,
    )

    assert len(train_argvs) == 1
    assert train_argvs[0] == plan["train_commands"][0]["training_argv"]
    assert probe_calls == [
        (
            Path(plan["probe_command"]["checkpoint"]),
            {
                "output_dir": Path(plan["probe_command"]["output_dir"]),
                "num_episodes": 1,
                "render_rollouts": False,
                "max_render_frames": 300,
                "tile_size": 16,
            },
        )
    ]
    assert summary["stage_results"][0]["metrics"] == {"global_step": 1.0}
    assert summary["stage_results"][0]["resumed"] is False
    assert json.loads((tmp_path / "H0" / "sweep_plan.json").read_text())["id"] == "H0"
    assert json.loads((tmp_path / "H0" / "sweep_summary.json").read_text())[
        "summary_path"
    ].endswith("sweep_summary.json")


def test_execute_communication_sweep_plan_runs_probe_only_entry(tmp_path: Path) -> None:
    matrix = json.loads(Path("autoresearch/communication_sweep.json").read_text())
    entry = matrix["phases"]["polish_gate"][1]
    entry["run_dir"] = str(tmp_path / "PG2")
    entry["probe_output_dir"] = str(tmp_path / "PG2" / "probe_eval16")
    entry["probe_checkpoint"] = str(tmp_path / "selected" / "model.pkl")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    plan = build_communication_sweep_plan(
        matrix_path=matrix_path,
        phase="polish_gate",
        run_id="PG2",
        probe_episodes=16,
        render_rollouts=False,
    )
    probe_calls: list[tuple[Path, dict[str, object]]] = []

    def fake_train_main(argv: list[str]) -> dict[str, float]:
        raise AssertionError(f"probe-only entry should not train: {argv}")

    def fake_probe_checkpoint(checkpoint: Path, **kwargs: object) -> dict[str, object]:
        probe_calls.append((checkpoint, kwargs))
        return {"probe_path": str(Path(kwargs["output_dir"]) / "communication_probe.json")}

    summary = execute_communication_sweep_plan(
        plan,
        train_main=fake_train_main,
        probe_checkpoint=fake_probe_checkpoint,
        check_resources=False,
    )

    assert summary["stage_results"] == []
    assert probe_calls == [
        (
            tmp_path / "selected" / "model.pkl",
            {
                "output_dir": tmp_path / "PG2" / "probe_eval16",
                "num_episodes": 16,
                "render_rollouts": False,
                "max_render_frames": 300,
                "tile_size": 16,
            },
        )
    ]
    assert json.loads((tmp_path / "PG2" / "sweep_plan.json").read_text())["id"] == "PG2"
    assert json.loads((tmp_path / "PG2" / "sweep_summary.json").read_text())[
        "summary_path"
    ].endswith("sweep_summary.json")


def test_execute_communication_sweep_plan_resumes_existing_stages(tmp_path: Path) -> None:
    matrix = json.loads(Path("autoresearch/communication_sweep.json").read_text())
    matrix["phases"]["horizon"][0]["run_dir"] = str(tmp_path / "H0")
    matrix["phases"]["horizon"][0]["probe_output_dir"] = str(tmp_path / "H0" / "probe")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    plan = build_communication_sweep_plan(
        matrix_path=matrix_path,
        phase="horizon",
        run_id="H0",
        bit_stages=[2],
        global_update_cap=1,
        num_envs=1,
        num_steps=1,
        render_rollouts=False,
    )
    run_dir = Path(plan["train_commands"][0]["run_dir"])
    checkpoint = Path(plan["train_commands"][0]["checkpoint"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    write_json(run_dir / "summary.json", {"metrics": {"global_step": 99.0, "loss": 0.5}})
    train_calls: list[list[str]] = []

    def fake_train_main(argv: list[str]) -> dict[str, float]:
        train_calls.append(argv)
        return {"global_step": 1.0}

    def fake_probe_checkpoint(checkpoint: Path, **kwargs: object) -> dict[str, object]:
        return {"probe_path": str(Path(kwargs["output_dir"]) / "communication_probe.json")}

    summary = execute_communication_sweep_plan(
        plan,
        train_main=fake_train_main,
        probe_checkpoint=fake_probe_checkpoint,
        check_resources=False,
    )

    assert train_calls == []
    assert summary["stage_results"][0]["resumed"] is True
    assert summary["stage_results"][0]["metrics"] == {"global_step": 99.0, "loss": 0.5}


def test_cli_communication_plan_prints_staged_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(
        [
            "autoresearch",
            "communication-plan",
            "--phase",
            "horizon",
            "--id",
            "H1",
            "--probe-episodes",
            "2",
            "--no-render",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == "H1"
    assert payload["global_update_cap"] == 1250
    assert payload["env_steps_per_stage"] == 2_560_000
    assert len(payload["train_commands"]) == 2
    assert payload["probe_command"]["argv"][-1] == "--no-render"


def test_cli_forage_plan_prints_curriculum_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(
        [
            "autoresearch",
            "forage-plan",
            "--phase",
            "algorithm",
            "--id",
            "H1",
            "--stage-sizes",
            "4",
            "50",
            "--global-update-cap",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--wandb-mode",
            "offline",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == "H1"
    assert payload["phase"] == "algorithm"
    assert payload["stage_sizes"] == [4, 50]
    assert payload["total_train_env_steps"] == 16
    assert payload["wandb"]["project"] == "cool-antz"
    assert payload["wandb"]["mode"] == "offline"


def test_cli_autocurriculum_plan_prints_executable_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(
        [
            "autoresearch",
            "autocurriculum-plan",
            "--phase",
            "reward",
            "--id",
            "R2",
            "--global-update-cap",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--probe-rollout-steps",
            "12",
            "--probe-num-envs",
            "2",
            "--no-render",
            "--wandb-mode",
            "offline",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == "R2"
    assert payload["phase"] == "reward"
    assert payload["total_train_env_steps"] == 8
    assert payload["probe"]["rollout_steps"] == 12
    assert payload["rollout"]["enabled"] is False
    assert payload["wandb"]["mode"] == "offline"
    assert "No-cheat constraints" in payload["wandb"]["notes"]


def test_communication_rank_balances_sampled_and_deterministic_delivery(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    run_root = tmp_path / "runs"
    entries = []
    for run_id, seed, sampled_delivered, deterministic_delivered in [
        ("A", 804, 19.0, 11.0),
        ("B", 805, 17.0, 15.0),
        ("C", 803, 9.75, 16.5),
    ]:
        probe_output_dir = run_root / run_id / "probe"
        _write_fake_communication_probe(
            probe_output_dir / "communication_probe.json",
            sampled_delivered=sampled_delivered,
            deterministic_delivered=deterministic_delivered,
        )
        entries.append(
            {
                "id": run_id,
                "depends_on": "source",
                "run_dir": str(run_root / run_id),
                "probe_output_dir": str(probe_output_dir),
                "global_update_cap": 1,
                "args": {
                    "seed": seed,
                    "load_model": "source.pkl",
                },
            }
        )
    write_json(
        matrix_path,
        {
            "base_config": "experiments/communication_bits.json",
            "run_root": str(run_root),
            "phases": {"polish_length": entries},
        },
    )

    payload = rank_communication_gate_probes(
        matrix_path=matrix_path,
        phase="polish_length",
    )

    assert [candidate["id"] for candidate in payload["ranked"]] == ["B", "A", "C"]
    assert payload["ranked"][0]["gate_score"] == 16.0
    assert payload["ranked"][0]["min_delivered"] == 15.0
    assert payload["ranked"][0]["seed"] == 805
    assert payload["ranked"][1]["sampled"]["delivered"] == 19.0
    assert payload["missing"] == []


def test_communication_rank_reports_missing_artifacts(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    run_root = tmp_path / "runs"
    write_json(
        matrix_path,
        {
            "base_config": "experiments/communication_bits.json",
            "run_root": str(run_root),
            "phases": {
                "polish_length": [
                    {
                        "id": "A",
                        "run_dir": str(run_root / "A"),
                        "probe_output_dir": str(run_root / "A" / "probe"),
                        "global_update_cap": 1,
                        "args": {},
                    }
                ]
            },
        },
    )

    payload = rank_communication_gate_probes(
        matrix_path=matrix_path,
        phase="polish_length",
        run_ids=["A", "unknown"],
    )

    assert payload["ranked"] == []
    assert payload["missing"] == [
        {
            "id": "A",
            "probe_path": str(run_root / "A" / "probe" / "communication_probe.json"),
            "reason": "missing_probe",
        },
        {
            "id": "unknown",
            "probe_path": None,
            "reason": "unknown_id",
        },
    ]


def test_cli_communication_rank_prints_gate_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matrix_path = tmp_path / "matrix.json"
    run_root = tmp_path / "runs"
    probe_output_dir = run_root / "B" / "probe"
    _write_fake_communication_probe(
        probe_output_dir / "communication_probe.json",
        sampled_delivered=17.0,
        deterministic_delivered=15.0,
    )
    write_json(
        matrix_path,
        {
            "base_config": "experiments/communication_bits.json",
            "run_root": str(run_root),
            "phases": {
                "polish_length": [
                    {
                        "id": "B",
                        "run_dir": str(run_root / "B"),
                        "probe_output_dir": str(probe_output_dir),
                        "global_update_cap": 1,
                        "args": {"seed": 805},
                    }
                ]
            },
        },
    )

    exit_code = cli_main(
        [
            "autoresearch",
            "communication-rank",
            "--matrix",
            str(matrix_path),
            "--phase",
            "polish_length",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ranked"][0]["id"] == "B"
    assert payload["ranked"][0]["gate_score"] == 16.0
    assert payload["ranked"][0]["rank"] == 1


def test_cli_communication_run_uses_executable_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import ant_byte_env.autoresearch as autoresearch_module

    captured_plan: dict[str, object] = {}
    captured_check_resources: list[bool] = []
    captured_resume_completed: list[bool] = []

    def fake_execute_communication_sweep_plan(
        plan: dict[str, object],
        *,
        check_resources: bool,
        resume_completed: bool,
    ) -> dict[str, object]:
        captured_plan.update(plan)
        captured_check_resources.append(check_resources)
        captured_resume_completed.append(resume_completed)
        return {
            "phase": plan["phase"],
            "id": plan["id"],
            "summary_path": "runs/autoresearch/communication_bits/horizon/H0/sweep_summary.json",
            "stage_results": [],
            "probe": {},
        }

    monkeypatch.setattr(
        autoresearch_module,
        "execute_communication_sweep_plan",
        fake_execute_communication_sweep_plan,
    )

    exit_code = cli_main(
        [
            "autoresearch",
            "communication-run",
            "--phase",
            "horizon",
            "--id",
            "H0",
            "--run-root",
            str(tmp_path / "cli-smoke"),
            "--bit-stages",
            "2",
            "--global-update-cap",
            "1",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--no-render",
            "--skip-resource-check",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == "H0"
    assert captured_plan["bit_stages"] == [2]
    assert str(captured_plan["run_dir"]).startswith(str(tmp_path / "cli-smoke"))
    assert captured_plan["probe_command"]["options"]["max_render_frames"] == 300
    assert captured_plan["env_steps_per_stage"] == 1
    assert captured_check_resources == [False]
    assert captured_resume_completed == [True]


def test_cli_forage_run_uses_executable_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import ant_byte_env.forage_autoresearch as forage_autoresearch_module

    captured_plan: dict[str, object] = {}
    captured_check_resources: list[bool] = []
    captured_resume_completed: list[bool] = []

    def fake_execute_forage_50x50_sweep_plan(
        plan: dict[str, object],
        *,
        check_resources: bool,
        resume_completed: bool,
    ) -> dict[str, object]:
        captured_plan.update(plan)
        captured_check_resources.append(check_resources)
        captured_resume_completed.append(resume_completed)
        return {
            "phase": plan["phase"],
            "id": plan["id"],
            "summary_path": "runs/autoresearch/forage_50x50/reward/R1/sweep_summary.json",
            "resumed": False,
            "curriculum": {},
        }

    monkeypatch.setattr(
        forage_autoresearch_module,
        "execute_forage_50x50_sweep_plan",
        fake_execute_forage_50x50_sweep_plan,
    )

    exit_code = cli_main(
        [
            "autoresearch",
            "forage-run",
            "--phase",
            "reward",
            "--id",
            "R1",
            "--run-root",
            str(tmp_path / "cli-smoke"),
            "--stage-sizes",
            "4",
            "5",
            "--global-update-cap",
            "1",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--wandb-mode",
            "disabled",
            "--wandb-video-stages",
            "5x5",
            "--skip-resource-check",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == "R1"
    assert captured_plan["stage_sizes"] == [4, 5]
    assert str(captured_plan["run_dir"]).startswith(str(tmp_path / "cli-smoke"))
    assert captured_plan["wandb"]["mode"] == "disabled"
    assert captured_plan["wandb"]["video_stage_names"] == ["5x5"]
    assert captured_plan["total_train_env_steps"] == 8
    assert captured_check_resources == [False]
    assert captured_resume_completed == [True]


def test_cli_autocurriculum_run_uses_executable_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import ant_byte_env.autocurriculum_autoresearch as autocurriculum_module

    captured_plan: dict[str, object] = {}
    captured_check_resources: list[bool] = []
    captured_resume_completed: list[bool] = []

    def fake_execute_autocurriculum_sweep_plan(
        plan: dict[str, object],
        *,
        check_resources: bool,
        resume_completed: bool,
    ) -> dict[str, object]:
        captured_plan.update(plan)
        captured_check_resources.append(check_resources)
        captured_resume_completed.append(resume_completed)
        return {
            "phase": plan["phase"],
            "id": plan["id"],
            "summary_path": "runs/autoresearch/autocurriculum/reward/R2/sweep_summary.json",
            "resumed": False,
            "probe": {},
        }

    monkeypatch.setattr(
        autocurriculum_module,
        "execute_autocurriculum_sweep_plan",
        fake_execute_autocurriculum_sweep_plan,
    )

    exit_code = cli_main(
        [
            "autoresearch",
            "autocurriculum-run",
            "--phase",
            "reward",
            "--id",
            "R2",
            "--run-root",
            str(tmp_path / "cli-smoke"),
            "--global-update-cap",
            "1",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--probe-rollout-steps",
            "12",
            "--probe-num-envs",
            "2",
            "--load-model",
            str(tmp_path / "source.pkl"),
            "--no-render",
            "--wandb-mode",
            "disabled",
            "--skip-resource-check",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == "R2"
    assert str(captured_plan["run_dir"]).startswith(str(tmp_path / "cli-smoke"))
    assert captured_plan["probe"]["rollout_steps"] == 12
    assert captured_plan["wandb"]["mode"] == "disabled"
    assert captured_plan["total_train_env_steps"] == 4
    assert captured_plan["load_model"] == str(tmp_path / "source.pkl")
    assert captured_check_resources == [False]
    assert captured_resume_completed == [True]


def test_cli_communication_run_reports_resource_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ant_byte_env.autoresearch as autoresearch_module

    def fake_execute_communication_sweep_plan(
        plan: dict[str, object],
        *,
        check_resources: bool,
        resume_completed: bool,
    ) -> dict[str, object]:
        raise AutoresearchResourceError("Autoresearch resources look unsafe.\n- low swap")

    monkeypatch.setattr(
        autoresearch_module,
        "execute_communication_sweep_plan",
        fake_execute_communication_sweep_plan,
    )

    exit_code = cli_main(
        [
            "autoresearch",
            "communication-run",
            "--phase",
            "horizon",
            "--id",
            "H0",
            "--bit-stages",
            "2",
            "--global-update-cap",
            "1",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--no-render",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "low swap" in captured.err
    assert "Traceback" not in captured.err


def test_autoresearch_resource_guard_rejects_low_swap_when_memory_is_tight() -> None:
    with pytest.raises(RuntimeError, match="Autoresearch resources look unsafe"):
        assert_autoresearch_resources_available(
            {
                "disk_free_gb": 10.0,
                "mem_available_gb": 4.1,
                "swap_free_gb": 0.01,
                "gpu_compute_memory_mb": 0,
                "top_memory_processes": [],
            }
        )


def test_autoresearch_resource_guard_accepts_safe_snapshot() -> None:
    assert_autoresearch_resources_available(
        {
            "disk_free_gb": 10.0,
            "mem_available_gb": 8.0,
            "swap_free_gb": 1.0,
            "gpu_compute_memory_mb": 0,
            "top_memory_processes": [],
        }
    )


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


def test_cli_communication_probe_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "model.pkl"
    output_dir = tmp_path / "probe"

    def fake_probe_communication_checkpoint(actual_checkpoint_path: Path, **kwargs):
        assert actual_checkpoint_path == checkpoint_path
        assert kwargs["output_dir"] == output_dir
        assert kwargs["num_episodes"] == 2
        assert not kwargs["render_rollouts"]
        return {
            "probe_path": str(output_dir / "communication_probe.json"),
            "sampled": {"write_bit_entropy": 0.25},
            "deterministic": {"write_bit_entropy": 0.0},
        }

    import ant_byte_env.training.jax_mappo.probe as probe_module

    monkeypatch.setattr(
        probe_module,
        "probe_communication_checkpoint",
        fake_probe_communication_checkpoint,
    )

    exit_code = cli_main(
        [
            "probe",
            "communication",
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--num-episodes",
            "2",
            "--no-render",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "deterministic_write_bit_entropy": 0.0,
        "output": str(output_dir / "communication_probe.json"),
        "sampled_write_bit_entropy": 0.25,
    }


def _write_fake_communication_probe(
    path: Path,
    *,
    sampled_delivered: float,
    deterministic_delivered: float,
) -> None:
    path.parent.mkdir(parents=True)
    write_json(
        path,
        {
            "sampled": _fake_communication_probe_mode(sampled_delivered, entropy=0.2),
            "deterministic": _fake_communication_probe_mode(
                deterministic_delivered,
                entropy=0.3,
            ),
        },
    )


def _fake_communication_probe_mode(delivered: float, *, entropy: float) -> dict[str, object]:
    return {
        "delivery_metrics": {
            "mean_delivered_food": delivered,
            "mean_delivered_fraction": delivered / 23.0,
            "success_rate": 0.25,
            "mean_episode_length": 100.0,
        },
        "write_bit_entropy": entropy,
        "distinct_nonzero_values": [1, 2],
        "major_nonzero_values": [1, 2],
        "per_bit_entropy": [0.2, 0.0, 0.3],
    }
