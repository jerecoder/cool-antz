"""Rollout preview settings shared by notebook workflows."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ant_byte_env.rendering import render_checkpoint
from ant_byte_env.vault import create_vault_entry
from ant_byte_env.wandb_tracking import WandbTracker

NOTEBOOK_ROLLOUT_TILE_SIZE = 16
NOTEBOOK_ROLLOUT_SEED_OFFSET = 100_000
NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE = 1.0


def notebook_rollout_policy_temperature(
    metadata: Mapping[str, Any],
    *,
    key: str = "rollout_policy_temperature",
    default: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
) -> float:
    return validate_rollout_policy_temperature(
        metadata.get(key, default),
        name=key,
    )


def validate_rollout_policy_temperature(value: object, *, name: str) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative float.") from exc
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"{name} must be a non-negative float.")
    return temperature


def render_rollout_suite(
    *,
    checkpoint_paths: Sequence[Path],
    media_dir: Path,
    rollout_path_for_checkpoint: Callable[[Path, Path], Path],
    progress_desc: str,
    vault_dir: Path,
    title: str,
    description: str,
    metadata: Mapping[str, Any],
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
    reuse_existing: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    wandb_tags: Sequence[str] | None = None,
    wandb_video_key_prefix: str | None = None,
    wandb_video_names: Sequence[str] | None = None,
    wandb_step: int | float | None = None,
) -> dict[str, Any]:
    from tqdm.auto import tqdm

    policy_temperature = validate_rollout_policy_temperature(
        policy_temperature,
        name="policy_temperature",
    )
    media_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [Path(path) for path in checkpoint_paths]
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Train the missing policies before rendering:\n{missing_text}")

    rollout_paths = []
    rollout_seed_offsets = []
    for rollout_index, checkpoint in enumerate(tqdm(checkpoints, desc=progress_desc)):
        seed_offset = NOTEBOOK_ROLLOUT_SEED_OFFSET + rollout_index
        rollout_seed_offsets.append(seed_offset)
        rollout_paths.append(
            render_checkpoint(
                checkpoint,
                rollout_path_for_checkpoint(checkpoint, media_dir),
                backend="jax",
                seed_offset=seed_offset,
                reuse_existing=reuse_existing,
                max_frames=max_frames,
                tile_size=tile_size,
                policy_temperature=policy_temperature,
            )
        )
    wandb_video_keys = _log_rollout_videos_to_wandb(
        rollout_paths=rollout_paths,
        key_prefix=wandb_video_key_prefix,
        video_names=wandb_video_names,
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        run_name=wandb_run_name,
        mode=wandb_mode,
        tags=wandb_tags,
        run_dir=media_dir.parent,
        step=wandb_step,
        config={
            **metadata,
            "checkpoint_paths": [str(path) for path in checkpoints],
            "rollout_paths": [str(path) for path in rollout_paths],
            "render_max_frames": max_frames,
            "render_tile_size": tile_size,
            "rollout_policy_temperature": policy_temperature,
            "reuse_existing": reuse_existing,
        },
    )
    vault_entry_path = create_vault_entry(
        vault_dir=vault_dir,
        title=title,
        description=description,
        assets=rollout_paths,
        metadata={
            **metadata,
            "render_max_frames": max_frames,
            "render_tile_size": tile_size,
            "rollout_policy_temperature": policy_temperature,
            "reuse_existing": reuse_existing,
            "wandb_video_keys": wandb_video_keys,
            "rollout_seed_offsets": rollout_seed_offsets,
            "checkpoint_paths": [str(path) for path in checkpoints],
            "rollout_paths": [str(path) for path in rollout_paths],
        },
    )
    return {
        "rollout_paths": rollout_paths,
        "vault_entry_path": vault_entry_path,
        "wandb_video_keys": wandb_video_keys,
    }


def render_jax_checkpoint_rollout(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    media_dir: Path,
    rollout_filename: str,
    title: str,
    description: str,
    metadata: Mapping[str, Any],
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
    wandb_video_key: str | None = None,
    wandb_step: int | float | None = None,
) -> dict[str, Any]:
    policy_temperature = validate_rollout_policy_temperature(
        policy_temperature,
        name="policy_temperature",
    )
    media_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = render_checkpoint(
        checkpoint_path,
        media_dir / rollout_filename,
        backend="jax",
        reuse_existing=reuse_existing,
        max_frames=max_frames,
        tile_size=tile_size,
        policy_temperature=policy_temperature,
    )
    tracker = WandbTracker(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        name=wandb_run_name,
        mode=wandb_mode,
        tags=wandb_tags,
        run_dir=run_dir,
        config={"checkpoint_path": str(checkpoint_path), **dict(metadata)},
    )
    try:
        if tracker.enabled and wandb_video_key is not None:
            tracker.log_video(wandb_video_key, rollout_path, step=wandb_step)
    finally:
        tracker.finish()
    vault_entry_path = create_vault_entry(
        vault_dir=run_dir / "vault",
        title=title,
        description=description,
        assets=[rollout_path],
        metadata={
            "checkpoint_path": str(checkpoint_path),
            "rollout_path": str(rollout_path),
            "rollout_policy_temperature": policy_temperature,
            **dict(metadata),
        },
    )
    return {
        "rollout_path": rollout_path,
        "vault_entry_path": vault_entry_path,
        "wandb_video_key": wandb_video_key if tracker.enabled else None,
    }


def _log_rollout_videos_to_wandb(
    *,
    rollout_paths: Sequence[Path],
    key_prefix: str | None,
    video_names: Sequence[str] | None,
    project: str | None,
    entity: str | None,
    group: str | None,
    run_name: str | None,
    mode: str,
    tags: Sequence[str] | None,
    run_dir: Path,
    step: int | float | None,
    config: Mapping[str, Any],
) -> list[str]:
    if key_prefix is None:
        return []
    tracker = WandbTracker(
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        tags=tags,
        mode=mode,
        run_dir=run_dir,
        config=config,
    )
    logged_keys: list[str] = []
    try:
        for index, path in enumerate(rollout_paths):
            if video_names is not None and index < len(video_names):
                video_name = str(video_names[index])
            else:
                video_name = Path(path).stem
            video_key = f"{key_prefix.rstrip('/')}/{video_name}"
            tracker.log_video(video_key, Path(path), step=step)
            if tracker.enabled:
                logged_keys.append(video_key)
    finally:
        tracker.finish()
    return logged_keys


__all__ = [
    "NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE",
    "NOTEBOOK_ROLLOUT_SEED_OFFSET",
    "NOTEBOOK_ROLLOUT_TILE_SIZE",
    "notebook_rollout_policy_temperature",
    "render_jax_checkpoint_rollout",
    "render_rollout_suite",
    "validate_rollout_policy_temperature",
]
