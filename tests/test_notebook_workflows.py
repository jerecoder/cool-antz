from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ant_byte_env import notebook_workflows as workflows


def test_jax_notebook_runtime_sets_conservative_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    monkeypatch.delitem(sys.modules, "jax", raising=False)

    status = workflows.configure_jax_notebook_runtime(memory_fraction="0.33")

    assert status["jax_preallocate"] == "false"
    assert status["jax_memory_fraction"] == "0.33"
    assert status["jax_allocator"] == "platform"
    assert "memory_trimmed" in status
    assert status["disk_free_gb"] >= 0.0
    assert status["disk_used_percent"] >= 0.0
    assert "mem_available_gb" in status
    assert "swap_free_gb" in status


def test_jax_notebook_runtime_lowers_unsafe_existing_memory_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    monkeypatch.delitem(sys.modules, "jax", raising=False)

    status = workflows.configure_jax_notebook_runtime(memory_fraction="0.30")

    assert status["jax_preallocate"] == "false"
    assert status["jax_memory_fraction"] == "0.30"
    assert status["jax_allocator"] == "platform"


def test_notebook_resource_guard_rejects_unsafe_state() -> None:
    with pytest.raises(RuntimeError, match="Notebook resources look unsafe") as exc_info:
        workflows.assert_notebook_resources_available(
            {
                "disk_free_gb": 1.0,
                "mem_available_gb": 1.0,
                "swap_free_gb": 0.0,
                "gpu_compute_memory_mb": 4096,
                "top_memory_processes": [
                    {
                        "pid": 123,
                        "rss_mb": 1800.0,
                        "command": "python -m ipykernel_launcher --f=kernel-old.json",
                        "is_current_process": False,
                        "is_notebook_kernel": True,
                    }
                ],
            }
        )
    message = str(exc_info.value)
    assert "PID 123" in message
    assert "kill 123" in message
    assert "cleanup_notebook_artifacts" in message


def test_notebook_resource_guard_allows_low_swap_when_ram_is_available() -> None:
    workflows.assert_notebook_resources_available(
        {
            "disk_free_gb": 6.0,
            "mem_available_gb": 8.0,
            "swap_free_gb": 0.05,
            "gpu_compute_memory_mb": 0,
        }
    )


def test_notebook_resource_guard_rejects_gpu_reboot_recovery_action() -> None:
    with pytest.raises(RuntimeError, match="GPU recovery action is Reboot") as exc_info:
        workflows.assert_notebook_resources_available(
            {
                "disk_free_gb": 6.0,
                "mem_available_gb": 8.0,
                "swap_free_gb": 1.0,
                "gpu_compute_memory_mb": 0,
                "gpu_recovery_action": "Reboot",
            }
        )

    assert "reboot" in str(exc_info.value).lower()


def test_cleanup_notebook_artifacts_dry_run_and_delete(tmp_path: Path) -> None:
    cache_dir = tmp_path / "__pycache__"
    nested_checkpoint = tmp_path / "notebooks" / ".ipynb_checkpoints"
    keep_dir = tmp_path / "runs"
    cache_dir.mkdir()
    nested_checkpoint.mkdir(parents=True)
    keep_dir.mkdir()
    (cache_dir / "module.pyc").write_bytes(b"x" * 10)
    (nested_checkpoint / "draft.ipynb").write_bytes(b"y" * 7)
    (keep_dir / "model.pkl").write_bytes(b"z" * 100)

    dry_run = workflows.cleanup_notebook_artifacts(tmp_path, dry_run=True)

    assert dry_run["dry_run"] is True
    assert dry_run["candidate_count"] == 2
    assert dry_run["freed_bytes"] == 17
    assert cache_dir.exists()
    assert nested_checkpoint.exists()

    cleanup = workflows.cleanup_notebook_artifacts(tmp_path, dry_run=False)

    assert cleanup["dry_run"] is False
    assert cleanup["removed_count"] == 2
    assert cleanup["freed_bytes"] == 17
    assert not cache_dir.exists()
    assert not nested_checkpoint.exists()
    assert keep_dir.exists()


def test_forage_stage_generation_reaches_50x50() -> None:
    stages = workflows.build_forage_curriculum_stages((4, 50))

    assert stages[0] == {
        "name": "4x4",
        "width": 4,
        "height": 4,
        "food_count": 2,
        "food_sources": 1,
        "cookie_distance": 1,
        "max_steps": 64,
    }
    assert stages[-1]["name"] == "50x50"
    assert stages[-1]["food_count"] == 48
    assert stages[-1]["food_sources"] == 12
    assert stages[-1]["cookie_distance"] == 24
    assert stages[-1]["max_steps"] == 10000


def test_forage_food_sources_concentrate_total_food_budget() -> None:
    stages = workflows.build_forage_curriculum_stages((4, 6, 8, 25))

    assert [stage["food_count"] for stage in stages] == [2, 4, 6, 23]
    assert [stage["food_sources"] for stage in stages] == [1, 1, 2, 6]
    for stage in stages:
        assert 1 <= int(stage["food_sources"]) <= int(stage["food_count"])


def test_forage_common_args_use_largest_stage_padding_and_moving_writes() -> None:
    stages = workflows.build_forage_curriculum_stages((4, 50))

    args = workflows.build_forage_common_args(
        stages,
        num_envs=16,
        num_steps=80,
        actor_vision_radius=1,
        write_bits=1,
        gamma=0.97,
    )

    assert args[args.index("--obs-width") + 1] == "50"
    assert args[args.index("--obs-height") + 1] == "50"
    assert args[args.index("--num-envs") + 1] == "16"
    assert args[args.index("--num-steps") + 1] == "80"
    assert args[args.index("--gamma") + 1] == "0.97"
    assert "--random-food" in args
    assert "--random-hub" in args
    assert "--distance-bonus" not in args
    assert "--write-while-moving" in args

    stay_only_args = workflows.build_forage_common_args(
        stages,
        num_envs=16,
        num_steps=80,
        actor_vision_radius=1,
        write_bits=1,
        gamma=0.97,
        write_while_moving=False,
    )
    assert "--write-while-moving" not in stay_only_args


def test_config_common_args_excludes_stage_specific_keys() -> None:
    args = workflows.config_common_args(
        {
            "width": 25,
            "height": 25,
            "write_bits": 2,
            "write_bit_penalty": 0.0,
            "write_bit_penalty_decay": 0.5,
            "write_entropy_bonus": 0.1,
            "write_entropy_bonus_cap": 0.15,
            "write_bit_entropy_bonus": 0.5,
            "ent_coef": 0.02,
            "write_head_transfer": "neutral-new",
            "write_while_moving": True,
            "load_model": "source.pkl",
            "quiet": True,
        },
        exclude=workflows.COMMUNICATION_ARG_EXCLUDES,
    )

    assert "--width" in args
    assert "--height" in args
    assert "--quiet" in args
    assert args[args.index("--write-bit-penalty") + 1] == "0.0"
    assert args[args.index("--write-bit-penalty-decay") + 1] == "0.5"
    assert args[args.index("--write-entropy-bonus") + 1] == "0.1"
    assert args[args.index("--write-entropy-bonus-cap") + 1] == "0.15"
    assert args[args.index("--write-bit-entropy-bonus") + 1] == "0.5"
    assert args[args.index("--ent-coef") + 1] == "0.02"
    assert args[args.index("--write-head-transfer") + 1] == "neutral-new"
    assert "--write-while-moving" in args
    assert "--write-bits" not in args
    assert "--load-model" not in args


def test_ant_count_training_args_keep_25x25_task_with_50_padded_observations() -> None:
    args = workflows.ant_count_training_args(
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


def test_checkpoint_path_helpers_match_notebook_artifact_layout(tmp_path: Path) -> None:
    stages = workflows.build_forage_curriculum_stages((4,))

    assert workflows.forage_checkpoint_paths(tmp_path / "checkpoints", stages) == [
        tmp_path / "checkpoints" / "jax_mappo_forage_stage1_4x4.pkl"
    ]
    assert workflows.communication_checkpoint_paths(tmp_path, (2, 3)) == [
        tmp_path / "2_bits" / "checkpoints" / "model.pkl",
        tmp_path / "3_bits" / "checkpoints" / "model.pkl",
    ]
    assert workflows.ant_count_checkpoint_paths(tmp_path, (2, 4)) == [
        tmp_path / "2_ants" / "checkpoints" / "model.pkl",
        tmp_path / "4_ants" / "checkpoints" / "model.pkl",
    ]


def test_communication_consolidation_runs_single_stage_with_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeProgress:
        def update(self, value: int) -> None:
            del value

        def set_postfix(self, **kwargs: str) -> None:
            del kwargs

        def close(self) -> None:
            pass

    captured_args: list[str] = []

    def fake_train_main(args: list[str], progress_callback):
        captured_args.extend(args)
        progress_callback(1, 1, {"loss": 0.1, "episode_return": 2.0})
        return {"loss": 0.1, "episode_return": 2.0}

    monkeypatch.setattr(
        workflows,
        "stage_update_progress",
        lambda label, total_updates: FakeProgress(),
    )

    result = workflows.run_communication_consolidation(
        source_checkpoint=tmp_path / "source.pkl",
        target_bits=8,
        run_dir=tmp_path / "run",
        common_args=["--ent-coef", "0.02", "--write-bit-entropy-bonus", "0.5"],
        experiment_name="jax_mappo_communication",
        update_timesteps_per_stage=1280,
        global_update_cap=2500,
        train_main=fake_train_main,
        stage_name="8_bits_consolidated",
        extra_args={"ent_coef": 0.002, "write_bit_entropy_bonus": 0.05},
    )

    assert result["final_checkpoint"] == (
        tmp_path / "run" / "8_bits_consolidated" / "checkpoints" / "model.pkl"
    )
    assert captured_args[captured_args.index("--exp-name") + 1] == (
        "jax_mappo_communication_8_bits_consolidated"
    )
    assert captured_args[captured_args.index("--write-bits") + 1] == "8"
    assert captured_args[captured_args.index("--total-timesteps") + 1] == "3200000"
    assert captured_args[captured_args.index("--load-model") + 1] == str(tmp_path / "source.pkl")
    ent_coef_index = len(captured_args) - 1 - captured_args[::-1].index("--ent-coef")
    bit_bonus_index = (
        len(captured_args) - 1 - captured_args[::-1].index("--write-bit-entropy-bonus")
    )
    assert captured_args[ent_coef_index + 1] == "0.002"
    assert captured_args[bit_bonus_index + 1] == "0.05"


def test_vision_range_curriculum_trains_and_renders_each_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_postfixes: list[dict[str, str]] = []

    class FakeProgress:
        def update(self, value: int) -> None:
            del value

        def set_postfix(self, **kwargs: str) -> None:
            captured_postfixes.append(kwargs)

        def close(self) -> None:
            pass

    captured_train_args: list[list[str]] = []
    captured_renders: list[tuple[Path, Path, dict[str, object]]] = []

    def fake_train_main(args: list[str], progress_callback):
        captured_train_args.append(args)
        progress_callback(
            1,
            1,
            {
                "loss": 0.1,
                "policy_loss": 0.2,
                "value_loss": 0.3,
                "episode_return": 2.0,
                "recent_episode_return": 1.5,
                "env_return": 1.0,
            },
        )
        return {"loss": 0.1, "episode_return": 2.0}

    def fake_render_checkpoint(checkpoint: Path, output_path: Path, **kwargs: object) -> Path:
        captured_renders.append((checkpoint, output_path, kwargs))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"gif")
        return output_path

    monkeypatch.setattr(
        workflows,
        "stage_update_progress",
        lambda label, total_updates: FakeProgress(),
    )
    monkeypatch.setattr(workflows, "render_checkpoint", fake_render_checkpoint)

    result = workflows.run_vision_range_curriculum(
        vision_radii=(2, 1),
        run_dir=tmp_path / "run",
        common_args=["--width", "50", "--write-bits", "1"],
        experiment_name="vision_range",
        update_timesteps_per_stage=10,
        global_update_cap=3,
        train_main=fake_train_main,
        max_render_frames=7,
        tile_size=8,
    )

    first_checkpoint = tmp_path / "run" / "5x5" / "checkpoints" / "model.pkl"
    second_checkpoint = tmp_path / "run" / "3x3" / "checkpoints" / "model.pkl"
    assert result["stage_checkpoint_paths"] == [first_checkpoint, second_checkpoint]
    assert "--load-model" not in captured_train_args[0]
    assert captured_train_args[1][captured_train_args[1].index("--load-model") + 1] == str(
        first_checkpoint
    )
    assert [path.name for _, path, _ in captured_renders] == ["vision_5x5.gif", "vision_3x3.gif"]
    assert captured_renders[0][2]["max_frames"] == 7
    assert captured_renders[0][2]["tile_size"] == 8
    assert captured_postfixes[0]["ret"] == "2.000"
    assert captured_postfixes[0]["ret_avg"] == "1.500"
    assert result["stage_metrics"][0]["vision_side"] == 5
    assert result["stage_metrics"][1]["source_checkpoint"] == str(first_checkpoint)


def test_communication_post_stage_sequence_runs_enabled_stages_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeProgress:
        def update(self, value: int) -> None:
            del value

        def set_postfix(self, **kwargs: str) -> None:
            del kwargs

        def close(self) -> None:
            pass

    captured_loads: list[str] = []

    def fake_train_main(args: list[str], progress_callback):
        captured_loads.append(args[args.index("--load-model") + 1])
        progress_callback(1, 1, {"loss": 0.1, "episode_return": 2.0})
        return {"loss": 0.1, "episode_return": 2.0}

    monkeypatch.setattr(
        workflows,
        "stage_update_progress",
        lambda label, total_updates: FakeProgress(),
    )

    result = workflows.run_communication_post_stage_sequence(
        stage_configs={
            "consolidation": {
                "enabled": True,
                "stage_name": "8_bits_consolidated",
                "global_update_cap": 5000,
                "args": {"ent_coef": 0.002},
            },
            "disabled": {"enabled": False},
            "polish": {
                "enabled": True,
                "stage_name": "8_bits_polished",
                "global_update_cap": 2500,
                "args": {"ent_coef": 0.0},
            },
        },
        source_checkpoint=tmp_path / "source.pkl",
        target_bits=8,
        run_dir=tmp_path / "run",
        common_args=[],
        experiment_name="jax_mappo_communication",
        update_timesteps_per_stage=1280,
        train_main=fake_train_main,
    )

    consolidated = tmp_path / "run" / "8_bits_consolidated" / "checkpoints" / "model.pkl"
    polished = tmp_path / "run" / "8_bits_polished" / "checkpoints" / "model.pkl"
    assert captured_loads == [str(tmp_path / "source.pkl"), str(consolidated)]
    assert result["checkpoint_paths"] == [consolidated, polished]
    assert result["final_checkpoint"] == polished
    assert result["stage_results"]["disabled"] is None


def test_stage_validators_reject_non_increasing_curricula() -> None:
    with pytest.raises(ValueError, match="increasing"):
        workflows.validate_communication_stages([2, 2])

    with pytest.raises(ValueError, match="increasing"):
        workflows.validate_ant_count_stages(ant_stages=[2, 2], source_num_ants=1)

    with pytest.raises(ValueError, match="beyond"):
        workflows.validate_ant_count_stages(ant_stages=[1, 2], source_num_ants=1)


def test_rollout_wrappers_keep_distinct_stage_names(monkeypatch, tmp_path: Path) -> None:
    captured_paths: list[Path] = []

    def fake_render_rollout_suite(**kwargs):
        checkpoint = kwargs["checkpoint_paths"][0]
        captured_paths.append(
            kwargs["rollout_path_for_checkpoint"](checkpoint, kwargs["media_dir"])
        )
        return {"rollout_paths": captured_paths, "vault_entry_path": tmp_path / "vault"}

    monkeypatch.setattr(workflows, "render_rollout_suite", fake_render_rollout_suite)

    communication_result = workflows.render_communication_rollouts(
        experiment_config=tmp_path / "experiment.json",
        source_checkpoint=tmp_path / "source.pkl",
        run_dir=tmp_path,
        media_dir=tmp_path / "media",
        bit_stages=(3,),
        global_update_cap=10,
    )
    ant_result = workflows.render_ant_count_rollouts(
        experiment_config=tmp_path / "experiment.json",
        source_checkpoint=tmp_path / "source.pkl",
        run_dir=tmp_path,
        media_dir=tmp_path / "media",
        communication_bits=3,
        source_num_ants=1,
        ant_stages=(4,),
        global_update_cap=10,
    )

    assert communication_result["rollout_paths"][0].name == (
        "jax_mappo_25x25_3_bits_vision_rollout.mp4"
    )
    assert ant_result["rollout_paths"][1].name == "jax_mappo_25x25_3bits_4_ants_vision_rollout.mp4"


def test_rollout_suite_uses_full_episode_render_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "model.pkl"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
    captured_kwargs: list[dict[str, object]] = []

    def fake_render_checkpoint(
        checkpoint: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        del checkpoint
        captured_kwargs.append(kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    monkeypatch.setattr(workflows, "render_checkpoint", fake_render_checkpoint)

    workflows.render_rollout_suite(
        checkpoint_paths=[checkpoint_path],
        media_dir=tmp_path / "media",
        rollout_path_for_checkpoint=lambda checkpoint, media: media / f"{checkpoint.stem}.mp4",
        progress_desc="rendering",
        vault_dir=tmp_path / "vault",
        title="Preview",
        description="Notebook rollout",
        metadata={},
    )

    assert captured_kwargs == [
        {
            "backend": "jax",
            "seed_offset": workflows.NOTEBOOK_ROLLOUT_SEED_OFFSET,
            "reuse_existing": False,
            "max_frames": None,
            "tile_size": workflows.NOTEBOOK_ROLLOUT_TILE_SIZE,
            "policy_temperature": workflows.NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
        }
    ]


def test_rollout_suite_uses_distinct_seed_offsets_per_rollout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_paths = [
        tmp_path / "checkpoints" / "stage_a.pkl",
        tmp_path / "checkpoints" / "stage_b.pkl",
    ]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
    captured_offsets: list[int] = []
    captured_metadata: dict[str, object] = {}

    def fake_render_checkpoint(
        checkpoint: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        del checkpoint
        captured_offsets.append(int(kwargs["seed_offset"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    def fake_create_vault_entry(**kwargs: object) -> Path:
        captured_metadata.update(dict(kwargs["metadata"]))
        return tmp_path / "vault" / "entry"

    monkeypatch.setattr(workflows, "render_checkpoint", fake_render_checkpoint)
    monkeypatch.setattr(workflows, "create_vault_entry", fake_create_vault_entry)

    workflows.render_rollout_suite(
        checkpoint_paths=checkpoint_paths,
        media_dir=tmp_path / "media",
        rollout_path_for_checkpoint=lambda checkpoint, media: media / f"{checkpoint.stem}.mp4",
        progress_desc="rendering",
        vault_dir=tmp_path / "vault",
        title="Preview",
        description="Notebook rollout",
        metadata={},
    )

    assert captured_offsets == [
        workflows.NOTEBOOK_ROLLOUT_SEED_OFFSET,
        workflows.NOTEBOOK_ROLLOUT_SEED_OFFSET + 1,
    ]
    assert captured_metadata["rollout_seed_offsets"] == captured_offsets
    assert (
        captured_metadata["rollout_policy_temperature"]
        == workflows.NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE
    )
