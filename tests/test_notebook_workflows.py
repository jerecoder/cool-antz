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

    assert {
        key: stages[0][key]
        for key in (
            "name",
            "width",
            "height",
            "food_count",
            "food_sources",
            "cookie_distance",
            "max_steps",
        )
    } == {
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


def test_forage_training_profiles_give_large_maps_more_budget() -> None:
    stages = workflows.build_forage_curriculum_stages((4, 8, 15, 25, 50))

    assert [stage["global_update_cap"] for stage in stages] == [
        1500,
        1500,
        3000,
        6000,
        8000,
    ]
    assert [stage["num_steps"] for stage in stages] == [80, 80, 128, 160, 256]
    assert [stage["gamma"] for stage in stages] == [0.99, 0.99, 0.995, 0.995, 0.997]


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


def test_exploration_stage_generation_uses_full_grid_visit_objective() -> None:
    stages = workflows.build_exploration_curriculum_stages((4, 50))

    assert stages[0]["name"] == "4x4"
    assert stages[0]["food_count"] == 2
    assert stages[0]["food_sources"] == 1
    assert stages[0]["max_steps"] == 64
    assert stages[-1]["name"] == "50x50"
    assert stages[-1]["food_count"] == 48
    assert stages[-1]["food_sources"] == 12
    assert stages[-1]["max_steps"] == 10000


def test_exploration_common_args_disable_memory_and_cookie_objective() -> None:
    stages = workflows.build_exploration_curriculum_stages((4, 50))

    args = workflows.build_exploration_common_args(
        stages,
        num_envs=16,
        num_steps=80,
        actor_vision_radius=1,
        write_bits=1,
        gamma=0.97,
    )

    assert args[args.index("--obs-width") + 1] == "50"
    assert args[args.index("--obs-height") + 1] == "50"
    assert args[args.index("--actor-vision-radius") + 1] == "1"
    assert args[args.index("--write-bits") + 1] == "1"
    assert args[args.index("--reward-mode") + 1] == "explore"
    assert "--no-food-termination" in args
    assert "--write-action-ablation" in args
    assert "--random-food" in args
    assert "--actor-nearest-food-vector" not in args
    assert "--pickup-bonus" in args
    assert args[args.index("--pickup-bonus") + 1] == "0.0"


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


def test_forage_curriculum_honors_stage_budget_and_hyperparameter_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeProgress:
        def __init__(self, total_updates: int) -> None:
            self.total_updates = total_updates

        def update(self, value: int) -> None:
            del value

        def set_postfix(self, **kwargs: str) -> None:
            del kwargs

        def close(self) -> None:
            pass

    progress_totals: list[int] = []
    captured_args: list[list[str]] = []

    def fake_train_main(args: list[str], progress_callback):
        captured_args.append(args)
        total_timesteps = int(args[args.index("--total-timesteps") + 1])
        progress_callback(
            1,
            1,
            {
                "global_step": float(total_timesteps),
                "loss": 0.1,
                "episode_return": 1.0,
                "env_return": 0.0,
            },
        )
        checkpoint_path = Path(args[args.index("--save-model") + 1])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        return {"global_step": float(total_timesteps), "loss": 0.1, "episode_return": 1.0}

    def fake_progress(label: str, total_updates: int) -> FakeProgress:
        del label
        progress_totals.append(total_updates)
        return FakeProgress(total_updates)

    monkeypatch.setattr(workflows, "stage_update_progress", fake_progress)

    stages = workflows.build_forage_curriculum_stages((4, 25))
    result = workflows.run_forage_curriculum(
        stages=stages,
        checkpoint_dir=tmp_path / "checkpoints",
        common_args=["--num-envs", "16", "--num-steps", "80", "--gamma", "0.99"],
        update_timesteps_per_stage=1280,
        global_update_cap=4000,
        train_main=fake_train_main,
        wandb_video_max_frames=0,
    )

    assert progress_totals == [1500, 6000]
    assert [args[args.index("--total-timesteps") + 1] for args in captured_args] == [
        "1920000",
        "15360000",
    ]
    assert _last_arg(captured_args[0], "--num-steps") == "80"
    assert _last_arg(captured_args[1], "--num-steps") == "160"
    assert _last_arg(captured_args[0], "--gamma") == "0.99"
    assert _last_arg(captured_args[1], "--gamma") == "0.995"
    assert result["stage_metrics"][0]["curriculum_global_step"] == 1_920_000
    assert result["stage_metrics"][1]["curriculum_global_step"] == 17_280_000


def _last_arg(args: list[str], option: str) -> str:
    index = len(args) - 1 - args[::-1].index(option)
    return args[index + 1]


def test_autocurriculum_training_helper_passes_single_env_curriculum_args(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_train_main(argv: list[str], *, progress_callback):
        calls.append(argv)
        progress_callback(
            1,
            1,
            {
                "loss": 0.5,
                "episode_return": 1.0,
                "global_step": 16.0,
            },
        )
        return {"loss": 0.5, "episode_return": 1.0, "global_step": 16.0}

    result = workflows.run_autocurriculum_training(
        run_dir=tmp_path / "autocurriculum",
        common_args=[
            "--autocurriculum",
            "--width",
            "50",
            "--height",
            "50",
            "--food-count",
            "12",
            "--food-sources",
            "2",
        ],
        update_timesteps_per_stage=16,
        global_update_cap=1,
        train_main=fake_train_main,
    )

    assert calls
    train_args = calls[0]
    assert "--autocurriculum" in train_args
    assert train_args[train_args.index("--total-timesteps") + 1] == "16"
    assert train_args[train_args.index("--run-dir") + 1] == str(tmp_path / "autocurriculum")
    assert result["checkpoint_path"] == tmp_path / "autocurriculum" / "checkpoints" / "model.pkl"


def test_autocurriculum_progress_advances_by_reported_update_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeProgress:
        def __init__(self) -> None:
            self.updates: list[int] = []

        def update(self, value: int) -> None:
            self.updates.append(value)

        def set_postfix(self, **kwargs: str) -> None:
            del kwargs

        def close(self) -> None:
            pass

    progress = FakeProgress()

    def fake_train_main(argv: list[str], *, progress_callback):
        del argv
        for update_index in (1, 5, 10):
            progress_callback(
                update_index,
                10,
                {
                    "loss": 0.1,
                    "episode_return": 1.0,
                    "global_step": float(update_index),
                },
            )
        return {"loss": 0.1, "episode_return": 1.0, "global_step": 10.0}

    monkeypatch.setattr(
        workflows,
        "stage_update_progress",
        lambda label, total_updates: progress,
    )

    workflows.run_autocurriculum_training(
        run_dir=tmp_path / "autocurriculum",
        common_args=["--autocurriculum", "--width", "50", "--height", "50"],
        update_timesteps_per_stage=1,
        global_update_cap=10,
        train_main=fake_train_main,
    )

    assert progress.updates == [1, 4, 5]


def test_forage_curriculum_logs_wandb_metrics_and_stage_preview(
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

    class FakeTracker:
        instances: list["FakeTracker"] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.enabled = True
            self.metrics: list[tuple[dict[str, object], int | None]] = []
            self.videos: list[tuple[str, Path, int | None]] = []
            self.finished = False
            self.instances.append(self)

        def log_metrics(self, metrics: dict[str, object], *, step: int | None = None) -> None:
            self.metrics.append((metrics, step))

        def log_video(self, key: str, path: Path, *, step: int | None = None) -> None:
            self.videos.append((key, path, step))

        def finish(self) -> None:
            self.finished = True

    captured_render_kwargs: list[dict[str, object]] = []

    def fake_render_checkpoint(
        checkpoint: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        assert checkpoint.name == "jax_mappo_forage_stage1_4x4.pkl"
        captured_render_kwargs.append(kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    def fake_train_main(args: list[str], progress_callback):
        checkpoint_path = Path(args[args.index("--save-model") + 1])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        progress_callback(
            1,
            2,
            {
                "global_step": 128.0,
                "loss": 0.1,
                "episode_return": 2.0,
                "env_return": 1.0,
            },
        )
        progress_callback(
            2,
            2,
            {
                "global_step": 256.0,
                "loss": 0.05,
                "episode_return": 3.0,
                "env_return": 2.0,
            },
        )
        return {"global_step": 256.0, "loss": 0.05, "episode_return": 3.0}

    monkeypatch.setattr(
        workflows,
        "stage_update_progress",
        lambda label, total_updates: FakeProgress(),
    )
    monkeypatch.setattr(workflows, "WandbTracker", FakeTracker)
    monkeypatch.setattr(workflows, "render_checkpoint", fake_render_checkpoint)

    stages = workflows.build_forage_curriculum_stages((4,))
    result = workflows.run_forage_curriculum(
        stages=stages,
        checkpoint_dir=tmp_path / "checkpoints",
        common_args=["--num-envs", "2", "--num-steps", "64"],
        update_timesteps_per_stage=128,
        global_update_cap=2,
        train_main=fake_train_main,
        wandb_project="cool-antz",
        wandb_entity="team",
        wandb_group="forage_curriculum_50x50",
        wandb_run_name="phone",
        wandb_mode="offline",
        wandb_tags=["jax"],
        wandb_video_max_frames=600,
    )

    tracker = FakeTracker.instances[0]
    assert tracker.kwargs["project"] == "cool-antz"
    assert tracker.kwargs["entity"] == "team"
    assert tracker.kwargs["group"] == "forage_curriculum_50x50"
    assert tracker.kwargs["name"] == "phone"
    assert tracker.kwargs["mode"] == "offline"
    assert tracker.metrics[0][1] == 128
    assert tracker.metrics[1][1] == 256
    assert tracker.metrics[1][0]["stage_name"] == "4x4"
    assert tracker.metrics[1][0]["stage_update"] == 2
    assert tracker.videos == [
        (
            "videos/forage/4x4",
            tmp_path / "media" / "wandb_previews" / "jax_mappo_forage_stage1_4x4_preview.mp4",
            256,
        )
    ]
    assert captured_render_kwargs == [
        {
            "backend": "jax",
            "seed_offset": workflows.NOTEBOOK_ROLLOUT_SEED_OFFSET,
            "reuse_existing": False,
            "max_frames": 600,
            "tile_size": workflows.NOTEBOOK_ROLLOUT_TILE_SIZE,
            "policy_temperature": workflows.NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
        }
    ]
    assert result["final_checkpoint_path"] == (
        tmp_path / "checkpoints" / "jax_mappo_forage_stage1_4x4.pkl"
    )
    assert tracker.finished is True


def test_forage_curriculum_can_select_best_stage_checkpoint(
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

    captured_args: list[list[str]] = []

    def fake_train_main(args: list[str], progress_callback):
        captured_args.append(args)
        checkpoint_path = Path(args[args.index("--save-model") + 1])
        best_checkpoint_path = Path(args[args.index("--save-best-model") + 1])
        assert args[args.index("--best-model-metric") + 1] == "eval_mean_delivered_food"
        assert args[args.index("--best-model-mode") + 1] == "max"
        assert args[args.index("--best-model-selection") + 1] == "eval"
        assert args[args.index("--best-eval-episodes") + 1] == "16"
        assert args[args.index("--best-eval-interval") + 1] == "10"
        assert args[args.index("--best-eval-seed-offset") + 1] == "123000"
        assert args[args.index("--best-eval-action-mode") + 1] == (
            "sampled_move_greedy_write"
        )
        assert args[args.index("--best-eval-move-temperature") + 1] == "0.95"
        assert "--no-best-eval-shuffle-positions" in args
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"terminal")
        best_checkpoint_path.write_bytes(b"best")
        progress_callback(
            1,
            1,
            {
                "global_step": 64.0,
                "loss": 0.1,
                "episode_return": 2.0,
                "env_return": 1.0,
            },
        )
        return {"global_step": 64.0, "loss": 0.1, "episode_return": 2.0}

    monkeypatch.setattr(
        workflows,
        "stage_update_progress",
        lambda label, total_updates: FakeProgress(),
    )

    stage = {
        **workflows.build_forage_curriculum_stages((4,))[0],
        "save_best_checkpoint": True,
        "select_best_checkpoint": True,
        "best_checkpoint_selection": "eval",
        "best_checkpoint_metric": "eval_mean_delivered_food",
        "best_checkpoint_mode": "max",
        "best_eval_episodes": 16,
        "best_eval_interval": 10,
        "best_eval_seed_offset": 123000,
        "best_eval_action_mode": "sampled_move_greedy_write",
        "best_eval_move_temperature": 0.95,
        "best_eval_shuffle_positions": False,
    }
    result = workflows.run_forage_curriculum(
        stages=[stage],
        checkpoint_dir=tmp_path / "checkpoints",
        common_args=["--num-envs", "1", "--num-steps", "64"],
        update_timesteps_per_stage=64,
        global_update_cap=1,
        train_main=fake_train_main,
        wandb_video_max_frames=0,
    )

    terminal_path = tmp_path / "checkpoints" / "jax_mappo_forage_stage1_4x4.pkl"
    best_path = tmp_path / "checkpoints" / "jax_mappo_forage_stage1_4x4_best.pkl"
    assert captured_args[0][captured_args[0].index("--save-best-model") + 1] == str(
        best_path
    )
    assert result["terminal_stage_checkpoint_paths"] == [terminal_path]
    assert result["best_stage_checkpoint_paths"] == [best_path]
    assert result["stage_checkpoint_paths"] == [best_path]
    assert result["final_checkpoint_path"] == best_path
    assert result["stage_metrics"][0]["best_checkpoint"] == str(best_path)


def test_forage_curriculum_limits_wandb_previews_to_selected_stages(
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

    class FakeTracker:
        instances: list["FakeTracker"] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.enabled = True
            self.metrics: list[tuple[dict[str, object], int | None]] = []
            self.videos: list[tuple[str, Path, int | None]] = []
            self.instances.append(self)

        def log_metrics(self, metrics: dict[str, object], *, step: int | None = None) -> None:
            self.metrics.append((metrics, step))

        def log_video(self, key: str, path: Path, *, step: int | None = None) -> None:
            self.videos.append((key, path, step))

        def finish(self) -> None:
            pass

    rendered_checkpoints: list[str] = []

    def fake_render_checkpoint(
        checkpoint: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        del kwargs
        rendered_checkpoints.append(checkpoint.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    def fake_train_main(args: list[str], progress_callback):
        checkpoint_path = Path(args[args.index("--save-model") + 1])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        progress_callback(
            1,
            1,
            {
                "global_step": 64.0,
                "loss": 0.1,
                "episode_return": 1.0,
                "env_return": 0.0,
            },
        )
        return {"global_step": 64.0, "loss": 0.1, "episode_return": 1.0}

    monkeypatch.setattr(
        workflows,
        "stage_update_progress",
        lambda label, total_updates: FakeProgress(),
    )
    monkeypatch.setattr(workflows, "WandbTracker", FakeTracker)
    monkeypatch.setattr(workflows, "render_checkpoint", fake_render_checkpoint)

    workflows.run_forage_curriculum(
        stages=workflows.build_forage_curriculum_stages((4, 5)),
        checkpoint_dir=tmp_path / "checkpoints",
        common_args=["--num-envs", "1", "--num-steps", "64"],
        update_timesteps_per_stage=64,
        global_update_cap=1,
        train_main=fake_train_main,
        wandb_project="cool-antz",
        wandb_video_max_frames=600,
        wandb_video_stage_names=["5x5"],
    )

    tracker = FakeTracker.instances[0]
    assert [row["stage_name"] for row, _ in tracker.metrics] == ["4x4", "5x5"]
    assert tracker.videos == [
            (
                "videos/forage/5x5",
                tmp_path / "media" / "wandb_previews" / "jax_mappo_forage_stage1_5x5_preview.mp4",
                120064,
            )
        ]
    assert rendered_checkpoints == ["jax_mappo_forage_stage1_5x5.pkl"]
    assert tracker.kwargs["config"]["wandb_video_stage_names"] == ["5x5"]


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


def test_render_forage_rollouts_can_limit_stage_names(monkeypatch, tmp_path: Path) -> None:
    captured_checkpoint_paths: list[Path] = []
    captured_metadata: dict[str, object] = {}

    def fake_render_rollout_suite(**kwargs):
        captured_checkpoint_paths.extend(kwargs["checkpoint_paths"])
        captured_metadata.update(dict(kwargs["metadata"]))
        return {"rollout_paths": [], "vault_entry_path": tmp_path / "vault"}

    monkeypatch.setattr(workflows, "render_rollout_suite", fake_render_rollout_suite)

    workflows.render_forage_rollouts(
        run_dir=tmp_path / "run",
        checkpoint_dir=tmp_path / "checkpoints",
        media_dir=tmp_path / "media",
        stages=workflows.build_forage_curriculum_stages((4, 5, 8)),
        stage_names=("5x5", "8x8"),
        actor_vision_radius=1,
        write_bits=1,
        global_update_cap=8000,
    )

    assert captured_checkpoint_paths == [
        tmp_path / "checkpoints" / "jax_mappo_forage_stage1_5x5.pkl",
        tmp_path / "checkpoints" / "jax_mappo_forage_stage1_8x8.pkl",
    ]
    assert captured_metadata["stages"] == ["5x5", "8x8"]


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


def test_autocurriculum_rollout_reuses_existing_video_and_logs_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "model.pkl"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
    captured_render_kwargs: list[dict[str, object]] = []
    tracker_instances: list[object] = []

    def fake_render_checkpoint(
        checkpoint: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        del checkpoint
        captured_render_kwargs.append(kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    class FakeTracker:
        enabled = True

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.videos: list[tuple[str, Path, int | None]] = []
            self.finished = False
            tracker_instances.append(self)

        def log_video(self, key: str, path: Path, *, step=None, fps: int = 8) -> None:
            del fps
            self.videos.append((key, path, None if step is None else int(step)))

        def finish(self) -> None:
            self.finished = True

    monkeypatch.setattr(workflows, "render_checkpoint", fake_render_checkpoint)
    monkeypatch.setattr(workflows, "WandbTracker", FakeTracker)

    result = workflows.render_autocurriculum_rollout(
        run_dir=tmp_path / "run",
        checkpoint_path=checkpoint_path,
        media_dir=tmp_path / "run" / "media",
        global_update_cap=100,
        wandb_project="cool-antz",
        wandb_group="autocurriculum_50x50",
        wandb_run_name="autocurriculum_50x50_rollout",
        wandb_mode="offline",
        wandb_step=1280,
    )

    rollout_path = tmp_path / "run" / "media" / "jax_mappo_autocurriculum_rollout.mp4"
    assert captured_render_kwargs[0]["reuse_existing"] is True
    assert result["rollout_paths"] == [rollout_path]
    assert result["wandb_video_keys"] == ["videos/autocurriculum/rollout"]
    assert len(tracker_instances) == 1
    tracker = tracker_instances[0]
    assert tracker.kwargs["project"] == "cool-antz"
    assert tracker.kwargs["group"] == "autocurriculum_50x50"
    assert tracker.kwargs["name"] == "autocurriculum_50x50_rollout"
    assert tracker.videos == [("videos/autocurriculum/rollout", rollout_path, 1280)]
    assert tracker.finished is True


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
