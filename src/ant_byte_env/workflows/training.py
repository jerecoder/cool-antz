"""Training orchestration helpers shared by notebook workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ant_byte_env.rendering import render_checkpoint
from ant_byte_env.workflows.progress import advance_progress_to, stage_update_progress
from ant_byte_env.workflows.rollouts import (
    NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    NOTEBOOK_ROLLOUT_SEED_OFFSET,
    NOTEBOOK_ROLLOUT_TILE_SIZE,
    validate_rollout_policy_temperature,
)


def run_jax_checkpoint_training(
    *,
    run_dir: Path,
    common_args: Sequence[str],
    update_timesteps: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
    checkpoint_name: str = "model.pkl",
    progress_label: str = "training",
    checkpoint_video_interval_updates: int | None = None,
    checkpoint_video_max_frames: int | None = 600,
    checkpoint_video_tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    checkpoint_video_policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    checkpoint_video_wandb_key_prefix: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    stage_metrics: list[dict[str, Any]] = []
    checkpoint_video_paths: list[Path] = []
    checkpoint_video_checkpoint_paths: list[Path] = []
    checkpoint_video_wandb_keys: list[str] = []
    checkpoint_video_interval = (
        None
        if checkpoint_video_interval_updates is None
        else int(checkpoint_video_interval_updates)
    )
    if checkpoint_video_interval is not None and checkpoint_video_interval <= 0:
        raise ValueError("checkpoint_video_interval_updates must be positive.")
    checkpoint_video_policy_temperature = validate_rollout_policy_temperature(
        checkpoint_video_policy_temperature,
        name="checkpoint_video_policy_temperature",
    )
    progress = stage_update_progress(progress_label, global_update_cap)
    last_progress_update = 0

    def record_progress(
        update_index: int,
        total_updates: int,
        metrics: dict[str, float],
    ) -> None:
        nonlocal last_progress_update
        last_progress_update = advance_progress_to(
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
        tracker: Any,
        global_step: int,
        **_: Any,
    ) -> None:
        if checkpoint_video_interval is None or int(update) % checkpoint_video_interval != 0:
            return
        from ant_byte_env.training.jax_mappo.checkpointing import save_checkpoint

        checkpoint_file = (
            checkpoint_path.parent
            / f"{checkpoint_path.stem}_update_{int(update):06d}{checkpoint_path.suffix}"
        )
        checkpoint_metrics = {
            **metrics,
            "checkpoint_update": float(update),
            "checkpoint_global_step": float(global_step),
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
        rollout_path = render_checkpoint(
            checkpoint_file,
            run_dir
            / "media"
            / "checkpoint_videos"
            / f"{checkpoint_file.stem}_rollout.mp4",
            backend="jax",
            reuse_existing=False,
            seed_offset=NOTEBOOK_ROLLOUT_SEED_OFFSET + int(update),
            max_frames=checkpoint_video_max_frames,
            tile_size=checkpoint_video_tile_size,
            policy_temperature=checkpoint_video_policy_temperature,
        )
        checkpoint_video_checkpoint_paths.append(checkpoint_file)
        checkpoint_video_paths.append(rollout_path)
        if checkpoint_video_wandb_key_prefix is not None and tracker.enabled:
            video_key = (
                f"{checkpoint_video_wandb_key_prefix.rstrip('/')}/update_{int(update):06d}"
            )
            tracker.log_video(video_key, rollout_path, step=global_step)
            checkpoint_video_wandb_keys.append(video_key)

    train_args = [
        *common_args,
        "--total-timesteps",
        str(int(update_timesteps) * int(global_update_cap)),
        "--run-dir",
        str(run_dir),
        "--save-model",
        str(checkpoint_path),
    ]
    try:
        train_kwargs: dict[str, Any] = {"progress_callback": record_progress}
        if checkpoint_video_interval is not None:
            train_kwargs["checkpoint_callback"] = record_checkpoint_video
        final_train_metrics = train_main(train_args, **train_kwargs)
    finally:
        progress.close()

    return {
        "checkpoint_path": checkpoint_path,
        "stage_metrics": stage_metrics,
        "final_train_metrics": final_train_metrics,
        "checkpoint_video_checkpoint_paths": checkpoint_video_checkpoint_paths,
        "checkpoint_video_paths": checkpoint_video_paths,
        "checkpoint_video_wandb_keys": checkpoint_video_wandb_keys,
    }


__all__ = ["run_jax_checkpoint_training"]
