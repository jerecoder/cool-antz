"""Reusable workflow helpers for the AntByte notebooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ant_byte_env import MAX_WRITE_BITS
from ant_byte_env.experiments import config_args_to_argv
from ant_byte_env.curricula.stages import (
    FORAGE_STAGE_SIZES,
    FORAGE_STAGE_TRAINING_PROFILE,
    FORAGE_WANDB_PREVIEW_STAGE_NAMES,
    EXPLORATION_TO_FORAGE_STAGE_SIZES,
    EXPLORATION_TO_FORAGE_WANDB_PREVIEW_STAGE_NAMES,
    EXPLORATION_TO_FORAGE_VISIT_REWARD_SCHEDULE,
    EXPLORATION_STAGE_SIZES,
    EXPLORATION_STAGE_TRAINING_PROFILE,
    EXPLORATION_WANDB_PREVIEW_STAGE_NAMES,
    MAZE_EXPLORATION_STAGE_SIZES,
    MAZE_EXPLORATION_WANDB_PREVIEW_STAGE_NAMES,
    CURRICULUM_BITES_PER_FOOD_SOURCE,
    EXPLORATION_MAX_STEPS_PER_CELL,
    curriculum_food_count,
    curriculum_food_sources,
    exploration_max_steps,
    exploration_to_forage_visit_reward_scale,
    forage_training_profile,
    build_forage_curriculum_stages,
    build_exploration_to_forage_curriculum_stages,
    build_food_source_curriculum_stages,
    build_food_cluster_curriculum_stages,
    food_source_curriculum_visit_reward_scale,
    build_exploration_curriculum_stages,
    build_maze_exploration_curriculum_stages,
    exploration_training_profile,
)
from ant_byte_env.runtime.resources import (
    DEFAULT_JAX_MEMORY_FRACTION,
    NOTEBOOK_SAFE_CLEANUP_DIR_NAMES,
    assert_notebook_resources_available,
    cleanup_notebook_artifacts,
    configure_jax_notebook_runtime,
    notebook_resource_snapshot,
    trim_current_process_memory,
)
from ant_byte_env.rendering import render_checkpoint
from ant_byte_env.vault import create_vault_entry
from ant_byte_env.wandb_tracking import WandbTracker
from ant_byte_env.workflows.args import (
    ANT_COUNT_ARG_EXCLUDES,
    AUTOCURRICULUM_ARG_EXCLUDES,
    COMMUNICATION_ARG_EXCLUDES,
    EXPLORATION_ARG_EXCLUDES,
    EXPLORATION_TO_FORAGE_ARG_EXCLUDES,
    SINGLE_CHECKPOINT_ARG_EXCLUDES,
    build_exploration_common_args,
    build_forage_common_args,
    build_maze_exploration_common_args,
    config_common_args,
    update_timesteps,
)
from ant_byte_env.workflows.ant_count import (
    ant_count_train_args,
    ant_count_training_args,
    strictly_increasing as _strictly_increasing,
    validate_ant_count_stages,
)
from ant_byte_env.workflows.checkpoints import (
    ant_count_checkpoint_paths,
    communication_checkpoint_paths,
    exploration_checkpoint_paths,
    forage_checkpoint_paths,
    maze_exploration_checkpoint_paths,
)
from ant_byte_env.workflows.cli import (
    WANDB_CLI_VALUE_ARGS as _WANDB_CLI_VALUE_ARGS,
    WANDB_CLI_VARARGS as _WANDB_CLI_VARARGS,
    argv_int as _argv_int,
    strip_wandb_cli_args as _strip_wandb_cli_args,
)
from ant_byte_env.workflows.experiments import (
    load_jax_experiment,
    resolve_project_path,
    run_jax_smoke,
)
from ant_byte_env.workflows.progress import (
    advance_progress_to as _advance_progress_to,
    stage_update_progress,
)
from ant_byte_env.workflows.previews import (
    render_forage_wandb_previews as _render_forage_wandb_previews,
    validate_wandb_preview_stage_names as _validate_wandb_preview_stage_names,
    validate_wandb_video_rollout_count as _validate_wandb_video_rollout_count,
    wandb_preview_enabled as _wandb_preview_enabled,
    wandb_preview_stage_enabled as _wandb_preview_stage_enabled,
    wandb_preview_video_key as _wandb_preview_video_key,
    wandb_video_seed_offset_base as _wandb_video_seed_offset_base,
)
from ant_byte_env.workflows.stage_profiles import (
    forage_stage_training_profiles as _forage_stage_training_profiles,
    forage_stage_update_timesteps as _forage_stage_update_timesteps,
)
from ant_byte_env.workflows.rollouts import (
    NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    NOTEBOOK_ROLLOUT_SEED_OFFSET,
    NOTEBOOK_ROLLOUT_TILE_SIZE,
    notebook_rollout_policy_temperature,
    render_jax_checkpoint_rollout,
    render_rollout_suite,
    validate_rollout_policy_temperature as _validate_rollout_policy_temperature,
)
from ant_byte_env.workflows.training import run_jax_checkpoint_training

def run_forage_curriculum(
    *,
    stages: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
    common_args: Sequence[str],
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
    initial_checkpoint: Path | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    wandb_tags: Sequence[str] | None = None,
    wandb_notes: str | None = None,
    wandb_artifact_paths: Sequence[Path] | None = None,
    wandb_artifact_prefix: str = "forage-curriculum",
    checkpoint_name_prefix: str = "jax_mappo_forage_stage1",
    wandb_video_key_prefix: str = "videos/forage",
    wandb_video_max_frames: int | None = 600,
    wandb_video_stage_names: Sequence[str] | None = FORAGE_WANDB_PREVIEW_STAGE_NAMES,
    wandb_video_policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    wandb_video_rollout_count: int = 1,
    wandb_video_seed_offset_base: int | None = None,
    checkpoint_video_interval_updates: int | None = None,
    checkpoint_video_max_frames: int | None = 600,
    checkpoint_video_tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    checkpoint_video_policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    checkpoint_video_rollout_count: int = 1,
    checkpoint_video_wandb_key_prefix: str | None = None,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    uses_default_wandb_video_stage_names = (
        wandb_video_stage_names is FORAGE_WANDB_PREVIEW_STAGE_NAMES
    )
    wandb_video_policy_temperature = _validate_rollout_policy_temperature(
        wandb_video_policy_temperature,
        name="wandb_video_policy_temperature",
    )
    wandb_video_rollout_count = _validate_wandb_video_rollout_count(
        wandb_video_rollout_count
    )
    wandb_video_seed_offset_base = _wandb_video_seed_offset_base(
        wandb_video_seed_offset_base
    )
    checkpoint_video_interval = (
        None
        if checkpoint_video_interval_updates is None
        else int(checkpoint_video_interval_updates)
    )
    if checkpoint_video_interval is not None and checkpoint_video_interval <= 0:
        raise ValueError("checkpoint_video_interval_updates must be positive.")
    checkpoint_video_policy_temperature = _validate_rollout_policy_temperature(
        checkpoint_video_policy_temperature,
        name="checkpoint_video_policy_temperature",
    )
    checkpoint_video_rollout_count = _validate_wandb_video_rollout_count(
        checkpoint_video_rollout_count
    )
    if (
        _wandb_preview_enabled(wandb_video_max_frames)
        and not uses_default_wandb_video_stage_names
    ):
        wandb_video_stage_names = _validate_wandb_preview_stage_names(
            stages,
            wandb_video_stage_names,
        )
    elif wandb_video_stage_names is not None:
        wandb_video_stage_names = tuple(str(name) for name in wandb_video_stage_names)
    stage_metrics: list[dict[str, Any]] = []
    stage_checkpoint_paths: list[Path] = []
    terminal_stage_checkpoint_paths: list[Path] = []
    best_stage_checkpoint_paths: list[Path] = []
    checkpoint_video_checkpoint_paths: list[Path] = []
    checkpoint_video_paths: list[Path] = []
    checkpoint_video_wandb_keys: list[str] = []
    previous_checkpoint = Path(initial_checkpoint) if initial_checkpoint is not None else None
    if previous_checkpoint is not None and not previous_checkpoint.exists():
        raise FileNotFoundError(f"initial forage checkpoint does not exist: {previous_checkpoint}")
    if wandb_project is not None and wandb_mode != "disabled":
        stage_common_args, stripped_stage_wandb_args = _strip_wandb_cli_args(common_args)
    else:
        stage_common_args = list(common_args)
        stripped_stage_wandb_args = []
    final_train_metrics: dict[str, float] = {}
    curriculum_step_base = 0
    tracker = WandbTracker(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        name=wandb_run_name,
        tags=wandb_tags,
        mode=wandb_mode,
        run_dir=checkpoint_dir.parent,
        notes=wandb_notes,
        config={
            "common_args": list(stage_common_args),
            "initial_checkpoint": None if previous_checkpoint is None else str(previous_checkpoint),
            "global_update_cap": int(global_update_cap),
            "checkpoint_name_prefix": str(checkpoint_name_prefix),
            "stages": [str(stage["name"]) for stage in stages],
            "update_timesteps_per_stage": int(update_timesteps_per_stage),
            "stripped_stage_wandb_args": stripped_stage_wandb_args,
            "stage_training_profiles": _forage_stage_training_profiles(
                stages,
                common_args=stage_common_args,
                fallback_update_timesteps=int(update_timesteps_per_stage),
                fallback_update_cap=int(global_update_cap),
            ),
            "wandb_video_max_frames": wandb_video_max_frames,
            "wandb_video_stage_names": (
                None
                if wandb_video_stage_names is None
                else [str(name) for name in wandb_video_stage_names]
            ),
            "wandb_video_policy_temperature": wandb_video_policy_temperature,
            "wandb_video_rollout_count": wandb_video_rollout_count,
            "wandb_video_seed_offset_base": wandb_video_seed_offset_base,
            "checkpoint_video_interval_updates": checkpoint_video_interval,
            "checkpoint_video_max_frames": checkpoint_video_max_frames,
            "checkpoint_video_policy_temperature": checkpoint_video_policy_temperature,
            "checkpoint_video_rollout_count": checkpoint_video_rollout_count,
            "checkpoint_video_wandb_key_prefix": checkpoint_video_wandb_key_prefix,
        },
    )
    if tracker.enabled:
        for artifact_path in wandb_artifact_paths or ():
            if artifact_path.exists():
                tracker.log_artifact(
                    f"{wandb_artifact_prefix}-{artifact_path.stem}",
                    artifact_path,
                    artifact_type="research-plan",
                    aliases=["latest"],
                )

    try:
        for stage_index, stage in enumerate(stages, start=1):
            stage_update_cap = int(stage.get("global_update_cap", global_update_cap))
            stage_update_timesteps = _forage_stage_update_timesteps(
                stage,
                common_args=stage_common_args,
                fallback_update_timesteps=int(update_timesteps_per_stage),
            )
            print(f"Training stage {stage_index}/{len(stages)}: {stage['name']}")
            print("First update for this shape may compile; progress starts after it returns.")
            checkpoint_path = checkpoint_dir / f"{checkpoint_name_prefix}_{stage['name']}.pkl"
            best_checkpoint_path = None
            if bool(
                stage.get(
                    "save_best_checkpoint",
                    stage.get("select_best_checkpoint", False),
                )
            ):
                best_checkpoint_path = (
                    Path(str(stage["best_checkpoint_path"]))
                    if "best_checkpoint_path" in stage
                    else checkpoint_dir / f"{checkpoint_name_prefix}_{stage['name']}_best.pkl"
                )
            progress = stage_update_progress(str(stage["name"]), stage_update_cap)
            last_progress_update = 0

            def record_progress(
                update_index: int,
                total_updates: int,
                metrics: dict[str, float],
            ) -> None:
                nonlocal last_progress_update
                curriculum_step = curriculum_step_base + int(
                    float(metrics.get("global_step", 0.0))
                )
                last_progress_update = _advance_progress_to(
                    progress,
                    update_index=update_index,
                    previous_update_index=last_progress_update,
                )
                progress.set_postfix(
                    loss=f"{metrics['loss']:.3f}",
                    ret=f"{metrics['episode_return']:.3f}",
                )
                row = {
                    **stage,
                    **metrics,
                    "stage_index": stage_index,
                    "stage_name": str(stage["name"]),
                    "stage_update": update_index,
                    "stage_total_updates": total_updates,
                    "global_update_cap": stage_update_cap,
                    "stage_update_timesteps": stage_update_timesteps,
                    "curriculum_global_step": curriculum_step,
                    "checkpoint": str(checkpoint_path),
                }
                if best_checkpoint_path is not None:
                    row["best_checkpoint"] = str(best_checkpoint_path)
                stage_metrics.append(row)
                tracker.log_metrics(row, step=curriculum_step)

            def record_checkpoint_video(
                *,
                update: int,
                metrics: dict[str, float],
                params: Any,
                opt_state: Any,
                args: Any,
                central_obs_dim: int,
                actor_obs_dim: int,
                run_name: str,
                global_step: int,
                **_: Any,
            ) -> None:
                if (
                    checkpoint_video_interval is None
                    or int(update) % checkpoint_video_interval != 0
                ):
                    return
                from ant_byte_env.training.jax_mappo.checkpointing import save_checkpoint

                checkpoint_file = (
                    checkpoint_dir
                    / f"{checkpoint_path.stem}_update_{int(update):06d}{checkpoint_path.suffix}"
                )
                checkpoint_metrics = {
                    **metrics,
                    "checkpoint_update": float(update),
                    "checkpoint_global_step": float(global_step),
                    "checkpoint_stage_index": float(stage_index),
                }
                save_checkpoint(
                    checkpoint_file,
                    params=params,
                    opt_state=opt_state,
                    args=args,
                    central_obs_dim=central_obs_dim,
                    actor_obs_dim=actor_obs_dim,
                    run_name=run_name,
                    metrics=checkpoint_metrics,
                )
                checkpoint_video_checkpoint_paths.append(checkpoint_file)
                for rollout_index in range(checkpoint_video_rollout_count):
                    rollout_suffix = (
                        ""
                        if checkpoint_video_rollout_count == 1
                        else f"_{rollout_index + 1:02d}"
                    )
                    rollout_path = render_checkpoint(
                        checkpoint_file,
                        checkpoint_dir.parent
                        / "media"
                        / "checkpoint_videos"
                        / f"{checkpoint_file.stem}_rollout{rollout_suffix}.mp4",
                        backend="jax",
                        reuse_existing=False,
                        seed_offset=(
                            NOTEBOOK_ROLLOUT_SEED_OFFSET
                            + int(update)
                            + rollout_index
                        ),
                        max_frames=checkpoint_video_max_frames,
                        tile_size=checkpoint_video_tile_size,
                        policy_temperature=checkpoint_video_policy_temperature,
                    )
                    checkpoint_video_paths.append(rollout_path)
                    if (
                        checkpoint_video_wandb_key_prefix is not None
                        and tracker.enabled
                    ):
                        video_key = (
                            f"{checkpoint_video_wandb_key_prefix.rstrip('/')}/"
                            f"{stage['name']}/update_{int(update):06d}"
                        )
                        if checkpoint_video_rollout_count > 1:
                            video_key = f"{video_key}/rollout_{rollout_index + 1:02d}"
                        tracker.log_video(
                            video_key,
                            rollout_path,
                            step=curriculum_step_base + int(global_step),
                        )
                        checkpoint_video_wandb_keys.append(video_key)

            train_args = [
                *stage_common_args,
                "--total-timesteps",
                str(stage_update_timesteps * stage_update_cap),
                "--width",
                str(stage["width"]),
                "--height",
                str(stage["height"]),
                "--food-count",
                str(stage["food_count"]),
                "--food-sources",
                str(stage["food_sources"]),
                "--cookie-distance",
                str(stage["cookie_distance"]),
                "--max-steps",
                str(stage["max_steps"]),
                "--save-model",
                str(checkpoint_path),
            ]
            if "num_steps" in stage:
                train_args.extend(["--num-steps", str(int(stage["num_steps"]))])
            if "gamma" in stage:
                train_args.extend(["--gamma", str(float(stage["gamma"]))])
            for stage_key, option in (
                ("visit_reward_scale", "--visit-reward-scale"),
                ("visit_reward_decay", "--visit-reward-decay"),
                ("view_reward_scale", "--view-reward-scale"),
                ("view_reward_decay", "--view-reward-decay"),
                ("border_view_penalty", "--border-view-penalty"),
                ("border_moat_penalty", "--border-moat-penalty"),
            ):
                if stage_key in stage:
                    train_args.extend([option, str(float(stage[stage_key]))])
            if "border_moat_width" in stage:
                train_args.extend(["--border-moat-width", str(int(stage["border_moat_width"]))])
            if "food_cluster_count" in stage:
                train_args.extend(
                    ["--food-cluster-count", str(int(stage["food_cluster_count"]))]
                )
            if "food_cluster_radius" in stage:
                train_args.extend(
                    ["--food-cluster-radius", str(int(stage["food_cluster_radius"]))]
                )
            if "random_ant_spawn_radius" in stage:
                train_args.extend(
                    [
                        "--random-ant-spawn-radius",
                        str(int(stage["random_ant_spawn_radius"])),
                    ]
                )
            if best_checkpoint_path is not None:
                train_args.extend(
                    [
                        "--save-best-model",
                        str(best_checkpoint_path),
                        "--best-model-metric",
                        str(stage.get("best_checkpoint_metric", "episode_return")),
                        "--best-model-mode",
                        str(stage.get("best_checkpoint_mode", "max")),
                        "--best-model-selection",
                        str(stage.get("best_checkpoint_selection", "train")),
                    ]
                )
                if "best_eval_episodes" in stage:
                    train_args.extend(
                        ["--best-eval-episodes", str(int(stage["best_eval_episodes"]))]
                    )
                if "best_eval_interval" in stage:
                    train_args.extend(
                        ["--best-eval-interval", str(int(stage["best_eval_interval"]))]
                    )
                if "best_eval_seed_offset" in stage:
                    train_args.extend(
                        [
                            "--best-eval-seed-offset",
                            str(int(stage["best_eval_seed_offset"])),
                        ]
                    )
                if "best_eval_action_mode" in stage:
                    train_args.extend(
                        ["--best-eval-action-mode", str(stage["best_eval_action_mode"])]
                    )
                if "best_eval_move_temperature" in stage:
                    train_args.extend(
                        [
                            "--best-eval-move-temperature",
                            str(float(stage["best_eval_move_temperature"])),
                        ]
                    )
                if "best_eval_write_temperature" in stage:
                    train_args.extend(
                        [
                            "--best-eval-write-temperature",
                            str(float(stage["best_eval_write_temperature"])),
                        ]
                    )
                if stage.get("best_eval_shuffle_positions") is False:
                    train_args.append("--no-best-eval-shuffle-positions")
            if previous_checkpoint is not None:
                train_args.extend(["--load-model", str(previous_checkpoint)])

            try:
                train_kwargs: dict[str, Any] = {"progress_callback": record_progress}
                if checkpoint_video_interval is not None:
                    train_kwargs["checkpoint_callback"] = record_checkpoint_video
                final_train_metrics = train_main(train_args, **train_kwargs)
            finally:
                progress.close()

            terminal_stage_checkpoint_paths.append(checkpoint_path)
            selected_checkpoint_path = checkpoint_path
            if bool(stage.get("select_best_checkpoint", False)):
                if best_checkpoint_path is None or not best_checkpoint_path.exists():
                    raise FileNotFoundError(
                        "stage requested best-checkpoint selection, but no best checkpoint "
                        f"was written for {stage['name']}"
                    )
                selected_checkpoint_path = best_checkpoint_path
            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                best_stage_checkpoint_paths.append(best_checkpoint_path)
            stage_checkpoint_paths.append(selected_checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            if selected_checkpoint_path != checkpoint_path:
                print(f"Selected best checkpoint {selected_checkpoint_path}")
            if (
                tracker.enabled
                and _wandb_preview_enabled(wandb_video_max_frames)
                and _wandb_preview_stage_enabled(stage["name"], wandb_video_stage_names)
            ):
                preview_paths = _render_forage_wandb_previews(
                    checkpoint_path=selected_checkpoint_path,
                    checkpoint_dir=checkpoint_dir,
                    stage_index=stage_index,
                    max_frames=wandb_video_max_frames,
                    policy_temperature=wandb_video_policy_temperature,
                    rollout_count=wandb_video_rollout_count,
                    seed_offset_base=wandb_video_seed_offset_base,
                )
                for preview_index, preview_path in enumerate(preview_paths):
                    tracker.log_video(
                        _wandb_preview_video_key(
                            prefix=wandb_video_key_prefix,
                            stage_name=stage["name"],
                            preview_index=preview_index,
                            preview_count=len(preview_paths),
                        ),
                        preview_path,
                        step=curriculum_step_base
                        + int(float(final_train_metrics.get("global_step", 0.0))),
                    )
            curriculum_step_base += stage_update_timesteps * stage_update_cap
            previous_checkpoint = selected_checkpoint_path
    finally:
        tracker.finish()

    return {
        "stage_metrics": stage_metrics,
        "stage_checkpoint_paths": stage_checkpoint_paths,
        "terminal_stage_checkpoint_paths": terminal_stage_checkpoint_paths,
        "best_stage_checkpoint_paths": best_stage_checkpoint_paths,
        "final_checkpoint_path": previous_checkpoint,
        "final_train_metrics": final_train_metrics,
        "checkpoint_video_checkpoint_paths": checkpoint_video_checkpoint_paths,
        "checkpoint_video_paths": checkpoint_video_paths,
        "checkpoint_video_wandb_keys": checkpoint_video_wandb_keys,
    }


def run_exploration_curriculum(
    *,
    stages: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
    common_args: Sequence[str],
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
    initial_checkpoint: Path | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    wandb_tags: Sequence[str] | None = None,
    wandb_notes: str | None = None,
    wandb_artifact_paths: Sequence[Path] | None = None,
    wandb_video_max_frames: int | None = 600,
    wandb_video_stage_names: Sequence[str] | None = EXPLORATION_WANDB_PREVIEW_STAGE_NAMES,
    wandb_video_policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
) -> dict[str, Any]:
    return run_forage_curriculum(
        stages=stages,
        checkpoint_dir=checkpoint_dir,
        common_args=common_args,
        update_timesteps_per_stage=update_timesteps_per_stage,
        global_update_cap=global_update_cap,
        train_main=train_main,
        initial_checkpoint=initial_checkpoint,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        wandb_tags=wandb_tags,
        wandb_notes=wandb_notes,
        wandb_artifact_paths=wandb_artifact_paths,
        wandb_artifact_prefix="exploration-curriculum",
        checkpoint_name_prefix="jax_mappo_explore",
        wandb_video_key_prefix="videos/exploration",
        wandb_video_max_frames=wandb_video_max_frames,
        wandb_video_stage_names=wandb_video_stage_names,
        wandb_video_policy_temperature=wandb_video_policy_temperature,
    )


def run_autocurriculum_training(
    *,
    run_dir: Path,
    common_args: Sequence[str],
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoints" / "model.pkl"
    stage_metrics: list[dict[str, Any]] = []
    progress = stage_update_progress("autocurriculum", global_update_cap)
    last_progress_update = 0

    def record_progress(
        update_index: int,
        total_updates: int,
        metrics: dict[str, float],
    ) -> None:
        nonlocal last_progress_update
        last_progress_update = _advance_progress_to(
            progress,
            update_index=update_index,
            previous_update_index=last_progress_update,
        )
        progress.set_postfix(
            loss=f"{metrics['loss']:.3f}",
            ret=f"{metrics['episode_return']:.3f}",
        )
        stage_metrics.append(
            {
                **metrics,
                "stage_update": int(update_index),
                "stage_total_updates": int(total_updates),
                "global_update_cap": int(global_update_cap),
                "checkpoint": str(checkpoint_path),
            }
        )

    train_args = [
        *common_args,
        "--total-timesteps",
        str(int(update_timesteps_per_stage) * int(global_update_cap)),
        "--run-dir",
        str(run_dir),
        "--save-model",
        str(checkpoint_path),
    ]
    try:
        final_train_metrics = train_main(train_args, progress_callback=record_progress)
    finally:
        progress.close()

    return {
        "checkpoint_path": checkpoint_path,
        "stage_metrics": stage_metrics,
        "final_train_metrics": final_train_metrics,
    }


def validate_communication_stages(bit_stages: Sequence[int]) -> None:
    if any(bits <= 1 or bits > MAX_WRITE_BITS for bits in bit_stages):
        raise ValueError(f"bit stages must contain integers from 2 to {MAX_WRITE_BITS}.")
    if not _strictly_increasing(bit_stages):
        raise ValueError("bit stages must be increasing.")


def run_communication_bit_curriculum(
    *,
    bit_stages: Sequence[int],
    source_checkpoint: Path,
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    validate_communication_stages(bit_stages)
    stage_metrics: list[dict[str, Any]] = []
    stage_checkpoint_paths: list[Path] = []
    previous_checkpoint = source_checkpoint
    final_train_metrics: dict[str, float] = {}

    for target_bits in bit_stages:
        stage_run_dir = run_dir / f"{target_bits}_bits"
        checkpoint_path = stage_run_dir / "checkpoints" / "model.pkl"
        print(f"Training communication stage: {target_bits} writable bits")
        print(f"Starting from: {previous_checkpoint}")
        progress = stage_update_progress(f"{target_bits} bits", global_update_cap)
        last_progress_update = 0

        def record_progress(
            update_index: int,
            total_updates: int,
            metrics: dict[str, float],
        ) -> None:
            nonlocal last_progress_update
            del total_updates
            last_progress_update = _advance_progress_to(
                progress,
                update_index=update_index,
                previous_update_index=last_progress_update,
            )
            progress.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                ret=f"{metrics['episode_return']:.3f}",
            )
            stage_metrics.append(
                {
                    "write_bits": target_bits,
                    **metrics,
                    "stage_update": update_index,
                    "global_update_cap": global_update_cap,
                    "checkpoint": str(checkpoint_path),
                    "source_checkpoint": str(previous_checkpoint),
                    "run_dir": str(stage_run_dir),
                }
            )

        train_args = [
            *common_args,
            "--exp-name",
            f"{experiment_name}_{target_bits}_bits",
            "--write-bits",
            str(target_bits),
            "--total-timesteps",
            str(update_timesteps_per_stage * global_update_cap),
            "--load-model",
            str(previous_checkpoint),
            "--run-dir",
            str(stage_run_dir),
        ]
        try:
            final_train_metrics = train_main(train_args, progress_callback=record_progress)
        finally:
            progress.close()

        stage_checkpoint_paths.append(checkpoint_path)
        print(f"Saved {target_bits}-bit checkpoint to {checkpoint_path}")
        previous_checkpoint = checkpoint_path

    return {
        "source_checkpoint": source_checkpoint,
        "stage_checkpoint_paths": stage_checkpoint_paths,
        "final_checkpoint": previous_checkpoint,
        "final_train_metrics": final_train_metrics,
    }


def run_communication_consolidation(
    *,
    source_checkpoint: Path,
    target_bits: int,
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
    stage_name: str = "8_bits_consolidated",
    extra_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(target_bits) <= 1 or int(target_bits) > MAX_WRITE_BITS:
        raise ValueError(f"target_bits must be an integer from 2 to {MAX_WRITE_BITS}.")
    if int(global_update_cap) <= 0:
        raise ValueError("global_update_cap must be positive.")

    stage_run_dir = run_dir / stage_name
    checkpoint_path = stage_run_dir / "checkpoints" / "model.pkl"
    stage_metrics: list[dict[str, Any]] = []
    progress = stage_update_progress(stage_name, global_update_cap)
    last_progress_update = 0
    print(f"Training communication consolidation: {stage_name}")
    print(f"Starting from: {source_checkpoint}")

    def record_progress(
        update_index: int,
        total_updates: int,
        metrics: dict[str, float],
    ) -> None:
        nonlocal last_progress_update
        del total_updates
        last_progress_update = _advance_progress_to(
            progress,
            update_index=update_index,
            previous_update_index=last_progress_update,
        )
        progress.set_postfix(
            loss=f"{metrics['loss']:.3f}",
            ret=f"{metrics['episode_return']:.3f}",
        )
        stage_metrics.append(
            {
                "write_bits": int(target_bits),
                **metrics,
                "stage_update": update_index,
                "global_update_cap": int(global_update_cap),
                "checkpoint": str(checkpoint_path),
                "source_checkpoint": str(source_checkpoint),
                "run_dir": str(stage_run_dir),
            }
        )

    train_args = [
        *common_args,
        *config_args_to_argv(dict(extra_args or {})),
        "--exp-name",
        f"{experiment_name}_{stage_name}",
        "--write-bits",
        str(int(target_bits)),
        "--total-timesteps",
        str(update_timesteps_per_stage * int(global_update_cap)),
        "--load-model",
        str(source_checkpoint),
        "--run-dir",
        str(stage_run_dir),
    ]
    try:
        final_train_metrics = train_main(train_args, progress_callback=record_progress)
    finally:
        progress.close()

    print(f"Saved consolidated checkpoint to {checkpoint_path}")
    return {
        "source_checkpoint": source_checkpoint,
        "stage_checkpoint_paths": [checkpoint_path],
        "final_checkpoint": checkpoint_path,
        "final_train_metrics": final_train_metrics,
        "stage_metrics": stage_metrics,
        "stage_name": stage_name,
    }


def run_communication_post_stage_sequence(
    *,
    stage_configs: Mapping[str, Mapping[str, Any]],
    source_checkpoint: Path,
    target_bits: int,
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    train_main: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    current_checkpoint = source_checkpoint
    stage_results: dict[str, dict[str, Any] | None] = {}
    checkpoint_paths: list[Path] = []

    for label, config in stage_configs.items():
        if not config.get("enabled", False):
            stage_results[label] = None
            continue

        result = run_communication_consolidation(
            source_checkpoint=current_checkpoint,
            target_bits=target_bits,
            run_dir=run_dir,
            common_args=common_args,
            experiment_name=experiment_name,
            update_timesteps_per_stage=update_timesteps_per_stage,
            global_update_cap=int(config.get("global_update_cap", 0)),
            train_main=train_main,
            stage_name=str(config.get("stage_name", f"{target_bits}_bits_{label}")),
            extra_args=dict(config.get("args", {})),
        )
        current_checkpoint = result["final_checkpoint"]
        checkpoint_paths.append(current_checkpoint)
        stage_results[label] = result

    return {
        "source_checkpoint": source_checkpoint,
        "checkpoint_paths": checkpoint_paths,
        "final_checkpoint": current_checkpoint,
        "stage_results": stage_results,
    }


def run_ant_count_curriculum(
    *,
    ant_stages: Sequence[int],
    source_checkpoint: Path,
    source_num_ants: int,
    communication_bits: int,
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    validate_ant_count_stages(ant_stages=ant_stages, source_num_ants=source_num_ants)
    stage_metrics: list[dict[str, Any]] = []
    stage_checkpoint_paths: list[Path] = []
    previous_checkpoint = source_checkpoint
    previous_num_ants = int(source_num_ants)
    final_train_metrics: dict[str, float] = {}

    for target_num_ants in ant_stages:
        stage_run_dir = run_dir / f"{target_num_ants}_ants"
        checkpoint_path = stage_run_dir / "checkpoints" / "model.pkl"
        warm_start_checkpoint = (
            stage_run_dir
            / "warm_start"
            / f"from_{previous_num_ants}_to_{target_num_ants}_ants.pkl"
        )
        stage_source_checkpoint = previous_checkpoint
        stage_source_num_ants = previous_num_ants

        print(f"Training ant-count stage: {target_num_ants} ants")
        print(f"Starting from: {stage_source_checkpoint}")

        warm_start_args = ant_count_train_args(
            common_args=common_args,
            experiment_name=experiment_name,
            target_num_ants=target_num_ants,
            communication_bits=communication_bits,
            update_timesteps_per_stage=update_timesteps_per_stage,
            global_update_cap=global_update_cap,
            load_model=stage_source_checkpoint,
            run_dir=stage_run_dir,
        )
        prepare_ant_count_checkpoint(
            stage_source_checkpoint,
            warm_start_checkpoint,
            warm_start_args,
            fallback_source_num_ants=source_num_ants,
            expected_write_bits=communication_bits,
        )

        progress = stage_update_progress(f"{target_num_ants} ants", global_update_cap)
        last_progress_update = 0

        def record_progress(
            update_index: int,
            total_updates: int,
            metrics: dict[str, float],
        ) -> None:
            nonlocal last_progress_update
            del total_updates
            last_progress_update = _advance_progress_to(
                progress,
                update_index=update_index,
                previous_update_index=last_progress_update,
            )
            progress.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                ret=f"{metrics['episode_return']:.3f}",
            )
            stage_metrics.append(
                {
                    "num_ants": target_num_ants,
                    "source_num_ants": stage_source_num_ants,
                    "write_bits": communication_bits,
                    **metrics,
                    "stage_update": update_index,
                    "global_update_cap": global_update_cap,
                    "checkpoint": str(checkpoint_path),
                    "source_checkpoint": str(stage_source_checkpoint),
                    "warm_start_checkpoint": str(warm_start_checkpoint),
                    "run_dir": str(stage_run_dir),
                }
            )

        train_args = ant_count_train_args(
            common_args=common_args,
            experiment_name=experiment_name,
            target_num_ants=target_num_ants,
            communication_bits=communication_bits,
            update_timesteps_per_stage=update_timesteps_per_stage,
            global_update_cap=global_update_cap,
            load_model=warm_start_checkpoint,
            run_dir=stage_run_dir,
        )
        try:
            final_train_metrics = train_main(train_args, progress_callback=record_progress)
        finally:
            progress.close()

        stage_checkpoint_paths.append(checkpoint_path)
        print(f"Saved {target_num_ants}-ant checkpoint to {checkpoint_path}")
        previous_checkpoint = checkpoint_path
        previous_num_ants = int(target_num_ants)

    return {
        "source_checkpoint": source_checkpoint,
        "stage_checkpoint_paths": stage_checkpoint_paths,
        "final_checkpoint": previous_checkpoint,
        "final_train_metrics": final_train_metrics,
    }


def expand_critic_input_for_ant_count(
    params: Any,
    *,
    source_num_ants: int,
    target_num_ants: int,
) -> Any:
    import jax.numpy as jnp

    from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams, LinearParams

    source_num_ants = int(source_num_ants)
    target_num_ants = int(target_num_ants)
    if source_num_ants <= 0 or target_num_ants <= 0:
        raise ValueError("ant counts must be positive.")

    first_layer = params.critic_body[0]
    old_weight = jnp.asarray(first_layer.weight)
    old_bias = jnp.asarray(first_layer.bias)
    source_ant_features = 3 * source_num_ants
    target_ant_features = 3 * target_num_ants
    if old_weight.shape[0] < source_ant_features:
        raise ValueError("source critic input is too small for its ant count.")

    tail_dim = old_weight.shape[0] - source_ant_features
    target_dim = target_ant_features + tail_dim
    if target_dim == old_weight.shape[0] and source_num_ants == target_num_ants:
        return params

    shared_ants = min(source_num_ants, target_num_ants)
    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)

    source_pos = slice(0, 2 * shared_ants)
    target_pos = slice(0, 2 * shared_ants)
    source_carry = slice(2 * source_num_ants, 2 * source_num_ants + shared_ants)
    target_carry = slice(2 * target_num_ants, 2 * target_num_ants + shared_ants)
    source_tail = slice(3 * source_num_ants, old_weight.shape[0])
    target_tail = slice(3 * target_num_ants, target_dim)

    new_weight = new_weight.at[target_pos, :].set(old_weight[source_pos, :])
    new_weight = new_weight.at[target_carry, :].set(old_weight[source_carry, :])
    new_weight = new_weight.at[target_tail, :].set(old_weight[source_tail, :])

    return JaxMAPPOParams(
        actor_body=params.actor_body,
        move_head=params.move_head,
        write_head=params.write_head,
        critic_body=(LinearParams(weight=new_weight, bias=old_bias), params.critic_body[1]),
        value_head=params.value_head,
    )


def training_dimensions(argv: Sequence[str]) -> tuple[Any, int, int]:
    import jax

    from ant_byte_env.training.jax_mappo.cli import parse_args
    from ant_byte_env.training.jax_mappo.env_factory import make_jax_mappo_env
    from ant_byte_env.training.jax_mappo.observations import (
        build_actor_observations,
        build_central_observations,
        food_observation_scale,
    )
    from ant_byte_env.training.jax_mappo.curriculum import reset_batch

    args = parse_args(list(argv))
    env = make_jax_mappo_env(args)
    _, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    central_obs = build_central_observations(
        obs,
        food_scale=food_scale,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        agent_identity_types=getattr(args, "agent_identity_types", None),
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    return args, int(central_obs.shape[-1]), int(actor_obs.shape[-1])


def prepare_ant_count_checkpoint(
    source_checkpoint: Path,
    warm_start_checkpoint: Path,
    target_argv: Sequence[str],
    *,
    fallback_source_num_ants: int,
    expected_write_bits: int,
) -> Path:
    from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint, save_checkpoint
    from ant_byte_env.training.jax_mappo.updates import init_adam_state
    from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training

    source_checkpoint = Path(source_checkpoint)
    warm_start_checkpoint = Path(warm_start_checkpoint)
    target_args, target_central_obs_dim, target_actor_obs_dim = training_dimensions(target_argv)
    checkpoint = read_checkpoint(source_checkpoint)
    if int(checkpoint["actor_obs_dim"]) != target_actor_obs_dim:
        checkpoint = load_checkpoint_for_training(
            source_checkpoint,
            central_obs_dim=int(checkpoint["central_obs_dim"]),
            actor_obs_dim=target_actor_obs_dim,
            target_write_bits=expected_write_bits,
            actor_vision_radius=target_args.actor_vision_radius,
            target_num_ants=target_args.num_ants,
            target_agent_identity_types=getattr(target_args, "agent_identity_types", None),
        )
    source_args = checkpoint.get("args", {})
    source_num_ants = int(source_args.get("num_ants", fallback_source_num_ants))
    source_write_bits = int(source_args.get("write_bits", expected_write_bits))

    if source_write_bits != expected_write_bits:
        raise ValueError(
            f"Expected a {expected_write_bits}-bit source checkpoint, got {source_write_bits}."
        )
    if int(checkpoint["actor_obs_dim"]) != target_actor_obs_dim:
        raise ValueError("Actor observation dimension transfer did not match this stage.")

    params = checkpoint["params"]
    if int(checkpoint["central_obs_dim"]) != target_central_obs_dim:
        params = expand_critic_input_for_ant_count(
            params,
            source_num_ants=source_num_ants,
            target_num_ants=target_args.num_ants,
        )
    if params.critic_body[0].weight.shape[0] != target_central_obs_dim:
        raise ValueError("Transferred critic input dimension does not match this stage.")

    save_checkpoint(
        warm_start_checkpoint,
        params=params,
        opt_state=init_adam_state(params),
        args=target_args,
        central_obs_dim=target_central_obs_dim,
        actor_obs_dim=target_actor_obs_dim,
        run_name=(
            f"{checkpoint.get('run_name', 'jax_mappo')}"
            f"__{target_args.num_ants}_ants_warm_start"
        ),
        metrics={
            **checkpoint.get("metrics", {}),
            "source_num_ants": float(source_num_ants),
            "target_num_ants": float(target_args.num_ants),
        },
    )
    return warm_start_checkpoint


def render_forage_rollouts(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    media_dir: Path,
    stages: Sequence[Mapping[str, Any]],
    actor_vision_radius: int,
    write_bits: int,
    global_update_cap: int,
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    stage_names: Sequence[str] | None = FORAGE_WANDB_PREVIEW_STAGE_NAMES,
) -> dict[str, Any]:
    selected_stages = _filter_stages_by_name(stages, stage_names)
    return render_rollout_suite(
        checkpoint_paths=forage_checkpoint_paths(checkpoint_dir, selected_stages),
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda checkpoint, media: (
            media / f"{checkpoint.stem}_rollout.mp4"
        ),
        progress_desc="rendering policies",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO curriculum policy rollouts",
        description="Rollout MP4 videos for each saved JAX MAPPO curriculum stage policy.",
        metadata={
            "stages": [stage["name"] for stage in selected_stages],
            "actor_vision_radius": actor_vision_radius,
            "write_bits": write_bits,
            "global_update_cap": global_update_cap,
        },
        max_frames=max_frames,
        tile_size=tile_size,
        policy_temperature=policy_temperature,
    )

def render_exploration_rollouts(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    media_dir: Path,
    stages: Sequence[Mapping[str, Any]],
    actor_vision_radius: int,
    write_bits: int,
    global_update_cap: int,
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    stage_names: Sequence[str] | None = EXPLORATION_WANDB_PREVIEW_STAGE_NAMES,
) -> dict[str, Any]:
    selected_stages = _filter_stages_by_name(stages, stage_names)
    return render_rollout_suite(
        checkpoint_paths=exploration_checkpoint_paths(checkpoint_dir, selected_stages),
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda checkpoint, media: (
            media / f"{checkpoint.stem}_rollout.mp4"
        ),
        progress_desc="rendering exploration policies",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO exploration curriculum policy rollouts",
        description="Rollout MP4 videos for each saved JAX MAPPO exploration stage policy.",
        metadata={
            "stages": [stage["name"] for stage in selected_stages],
            "actor_vision_radius": actor_vision_radius,
            "write_bits": write_bits,
            "global_update_cap": global_update_cap,
            "reward_mode": "explore",
        },
        max_frames=max_frames,
        tile_size=tile_size,
        policy_temperature=policy_temperature,
    )


def render_maze_exploration_rollouts(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    media_dir: Path,
    stages: Sequence[Mapping[str, Any]],
    actor_vision_radius: int,
    write_bits: int,
    global_update_cap: int,
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    stage_names: Sequence[str] | None = MAZE_EXPLORATION_WANDB_PREVIEW_STAGE_NAMES,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    wandb_tags: Sequence[str] | None = None,
    wandb_video_key_prefix: str | None = None,
    wandb_step: int | float | None = None,
) -> dict[str, Any]:
    selected_stages = _filter_stages_by_name(stages, stage_names)
    selected_stage_names = [str(stage["name"]) for stage in selected_stages]
    return render_rollout_suite(
        checkpoint_paths=maze_exploration_checkpoint_paths(checkpoint_dir, selected_stages),
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda checkpoint, media: (
            media / f"{checkpoint.stem}_rollout.mp4"
        ),
        progress_desc="rendering maze exploration policies",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO maze exploration curriculum policy rollouts",
        description=(
            "Rollout MP4 videos for each saved JAX MAPPO maze exploration stage policy."
        ),
        metadata={
            "stages": selected_stage_names,
            "actor_vision_radius": actor_vision_radius,
            "write_bits": write_bits,
            "global_update_cap": global_update_cap,
            "reward_mode": "explore",
            "maze_obstacles": True,
        },
        max_frames=max_frames,
        tile_size=tile_size,
        policy_temperature=policy_temperature,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        wandb_tags=wandb_tags,
        wandb_video_key_prefix=wandb_video_key_prefix,
        wandb_video_names=selected_stage_names,
        wandb_step=wandb_step,
    )


def _filter_stages_by_name(
    stages: Sequence[Mapping[str, Any]],
    stage_names: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    if stage_names is None:
        return list(stages)
    enabled_names = {str(stage_name) for stage_name in stage_names}
    return [stage for stage in stages if str(stage["name"]) in enabled_names]


def render_autocurriculum_rollout(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    media_dir: Path,
    global_update_cap: int,
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    reuse_existing: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    wandb_tags: Sequence[str] | None = None,
    wandb_step: int | float | None = None,
) -> dict[str, Any]:
    return render_rollout_suite(
        checkpoint_paths=[checkpoint_path],
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda _checkpoint, media: (
            media / "jax_mappo_autocurriculum_rollout.mp4"
        ),
        progress_desc="rendering autocurriculum policy",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO autocurriculum policy rollout",
        description="Rollout MP4 video for the single-env JAX MAPPO autocurriculum policy.",
        metadata={
            "global_update_cap": int(global_update_cap),
            "autocurriculum": True,
        },
        max_frames=max_frames,
        tile_size=tile_size,
        policy_temperature=policy_temperature,
        reuse_existing=reuse_existing,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        wandb_tags=wandb_tags,
        wandb_video_key_prefix="videos/autocurriculum",
        wandb_video_names=["rollout"],
        wandb_step=wandb_step,
    )


def render_communication_rollouts(
    *,
    experiment_config: Path,
    source_checkpoint: Path,
    run_dir: Path,
    media_dir: Path,
    bit_stages: Sequence[int],
    global_update_cap: int,
    extra_checkpoint_paths: Sequence[Path] = (),
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
) -> dict[str, Any]:
    checkpoint_paths = [
        *communication_checkpoint_paths(run_dir, bit_stages),
        *[Path(path) for path in extra_checkpoint_paths],
    ]
    return render_rollout_suite(
        checkpoint_paths=checkpoint_paths,
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda checkpoint, media: (
            media / f"jax_mappo_25x25_{checkpoint.parent.parent.name}_vision_rollout.mp4"
        ),
        progress_desc="rendering communication policies",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO communication-bit curriculum",
        description=(
            "Rollout MP4 videos for 25x25 JAX MAPPO policies trained with progressively "
            "larger writable communication alphabets."
        ),
        metadata={
            "experiment_config": str(experiment_config),
            "source_checkpoint": str(source_checkpoint),
            "bit_stages": list(bit_stages),
            "global_update_cap": global_update_cap,
            "extra_checkpoint_paths": [str(path) for path in extra_checkpoint_paths],
        },
        max_frames=max_frames,
        tile_size=tile_size,
        policy_temperature=policy_temperature,
    )


def render_ant_count_rollouts(
    *,
    experiment_config: Path,
    source_checkpoint: Path,
    run_dir: Path,
    media_dir: Path,
    communication_bits: int,
    source_num_ants: int,
    ant_stages: Sequence[int],
    global_update_cap: int,
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
) -> dict[str, Any]:
    return render_rollout_suite(
        checkpoint_paths=ant_count_checkpoint_paths(run_dir, ant_stages),
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda checkpoint, media: (
            media / f"jax_mappo_25x25_3bits_{checkpoint.parent.parent.name}_vision_rollout.mp4"
        ),
        progress_desc="rendering ant-count policies",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO ant-count curriculum",
        description=(
            "Rollout MP4 videos for 25x25, 3-bit JAX MAPPO policies trained with "
            "progressively larger ant teams."
        ),
        metadata={
            "experiment_config": str(experiment_config),
            "source_checkpoint": str(source_checkpoint),
            "communication_bits": communication_bits,
            "source_num_ants": source_num_ants,
            "ant_stages": list(ant_stages),
            "global_update_cap": global_update_cap,
        },
        max_frames=max_frames,
        tile_size=tile_size,
        policy_temperature=policy_temperature,
    )
