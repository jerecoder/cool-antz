"""Reusable workflow helpers for the AntByte notebooks."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ant_byte_env import MAX_WRITE_BITS
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
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

FORAGE_STAGE_SIZES = tuple(range(4, 51))
FORAGE_STAGE_TRAINING_PROFILE = (
    {"max_size": 8, "global_update_cap": 1500, "num_steps": 80, "gamma": 0.99},
    {"max_size": 15, "global_update_cap": 3000, "num_steps": 128, "gamma": 0.995},
    {"max_size": 25, "global_update_cap": 6000, "num_steps": 160, "gamma": 0.995},
    {"max_size": 50, "global_update_cap": 8000, "num_steps": 256, "gamma": 0.997},
)
FORAGE_WANDB_PREVIEW_STAGE_NAMES = (
    "4x4",
    "8x8",
    "15x15",
    "20x20",
    "25x25",
    "30x30",
    "40x40",
    "50x50",
)
EXPLORATION_TO_FORAGE_STAGE_SIZES = (8, 12, 16, 20, 25, 30, 35, 40, 45, 50)
EXPLORATION_TO_FORAGE_WANDB_PREVIEW_STAGE_NAMES = (
    "8x8",
    "16x16",
    "25x25",
    "35x35",
    "50x50",
)
EXPLORATION_TO_FORAGE_VISIT_REWARD_SCHEDULE = (
    (8, 0.02),
    (12, 0.015),
    (16, 0.01),
    (20, 0.0075),
    (25, 0.005),
    (30, 0.003),
    (35, 0.002),
    (40, 0.0015),
    (45, 0.001),
    (50, 0.001),
)
EXPLORATION_STAGE_SIZES = tuple(range(4, 51))
EXPLORATION_STAGE_TRAINING_PROFILE = (
    {"max_size": 50, "global_update_cap": 1500, "num_steps": 80, "gamma": 0.99},
)
EXPLORATION_WANDB_PREVIEW_STAGE_NAMES: tuple[str, ...] | None = None
MAZE_EXPLORATION_STAGE_SIZES = tuple(range(10, 51))
MAZE_EXPLORATION_WANDB_PREVIEW_STAGE_NAMES: tuple[str, ...] | None = None
CURRICULUM_BITES_PER_FOOD_SOURCE = 4
EXPLORATION_MAX_STEPS_PER_CELL = 2.5
NOTEBOOK_ROLLOUT_TILE_SIZE = 16
NOTEBOOK_ROLLOUT_SEED_OFFSET = 100_000
NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE = 1.0
COMMUNICATION_ARG_EXCLUDES = {
    "exp_name",
    "write_bits",
    "total_timesteps",
    "save_model",
    "load_model",
    "run_dir",
}
AUTOCURRICULUM_ARG_EXCLUDES = {
    "total_timesteps",
    "save_model",
    "run_dir",
}
SINGLE_CHECKPOINT_ARG_EXCLUDES = {
    "total_timesteps",
    "save_model",
    "run_dir",
}
EXPLORATION_ARG_EXCLUDES = {
    "total_timesteps",
    "width",
    "height",
    "food_count",
    "food_sources",
    "food_cluster_count",
    "food_cluster_radius",
    "cookie_distance",
    "max_steps",
    "save_model",
    "load_model",
    "run_dir",
}
EXPLORATION_TO_FORAGE_ARG_EXCLUDES = EXPLORATION_ARG_EXCLUDES | {
    "gamma",
    "num_steps",
    "visit_reward_scale",
    "view_reward_scale",
    "cookies_per_source",
    "save_best_model",
    "best_model_metric",
    "best_model_mode",
    "best_model_selection",
    "best_eval_episodes",
    "best_eval_interval",
    "best_eval_seed_offset",
    "best_eval_action_mode",
    "best_eval_move_temperature",
    "best_eval_write_temperature",
    "best_eval_shuffle_positions",
}
ANT_COUNT_ARG_EXCLUDES = COMMUNICATION_ARG_EXCLUDES | {"num_ants"}
_WANDB_CLI_VALUE_ARGS = frozenset(
    {
        "--wandb-project",
        "--wandb-entity",
        "--wandb-group",
        "--wandb-run-name",
        "--wandb-notes",
        "--wandb-mode",
    }
)
_WANDB_CLI_VARARGS = frozenset({"--wandb-tags"})


def notebook_rollout_policy_temperature(
    metadata: Mapping[str, Any],
    *,
    key: str = "rollout_policy_temperature",
    default: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
) -> float:
    return _validate_rollout_policy_temperature(
        metadata.get(key, default),
        name=key,
    )


def _validate_rollout_policy_temperature(value: object, *, name: str) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative float.") from exc
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"{name} must be a non-negative float.")
    return temperature


def load_jax_experiment(config_path: Path) -> Any:
    experiment = load_experiment_config(config_path)
    if experiment.backend != "jax":
        raise ValueError(f"Expected a JAX experiment config, got {experiment.backend!r}.")
    return experiment


def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root / resolved


def config_common_args(training_args: Mapping[str, Any], *, exclude: Iterable[str]) -> list[str]:
    excluded = set(exclude)
    return config_args_to_argv(
        {key: value for key, value in training_args.items() if key not in excluded}
    )


def update_timesteps(*, num_envs: int, num_steps: int) -> int:
    return int(num_envs) * int(num_steps)


def curriculum_food_count(size: int) -> int:
    return 2 + max(0, int(size) - 4)


def curriculum_food_sources(size: int) -> int:
    food_count = curriculum_food_count(size)
    concentrated_sources = (
        food_count + CURRICULUM_BITES_PER_FOOD_SOURCE - 1
    ) // CURRICULUM_BITES_PER_FOOD_SOURCE
    return max(1, min(food_count, concentrated_sources))


def exploration_max_steps(size: int) -> int:
    return int(math.ceil(EXPLORATION_MAX_STEPS_PER_CELL * int(size) * int(size)))


def exploration_to_forage_visit_reward_scale(
    size: int,
    *,
    schedule: object = None,
    fallback: float = 0.0,
) -> float:
    """Return the small decaying new-cell reward for an exploration-to-forage stage."""

    fallback_scale = float(fallback)
    if not math.isfinite(fallback_scale) or fallback_scale < 0.0:
        raise ValueError("fallback visit reward scale must be a non-negative float.")
    normalized = _normalize_visit_reward_schedule(
        EXPLORATION_TO_FORAGE_VISIT_REWARD_SCHEDULE if schedule is None else schedule
    )
    if not normalized:
        return fallback_scale

    target_size = int(size)
    if target_size <= 0:
        raise ValueError("visit reward stage size must be positive.")
    previous_size, previous_scale = normalized[0]
    if target_size <= previous_size:
        return previous_scale
    for next_size, next_scale in normalized[1:]:
        if target_size == next_size:
            return next_scale
        if target_size < next_size:
            fraction = (target_size - previous_size) / (next_size - previous_size)
            return previous_scale + fraction * (next_scale - previous_scale)
        previous_size, previous_scale = next_size, next_scale
    return normalized[-1][1]


def _normalize_visit_reward_schedule(schedule: object) -> tuple[tuple[int, float], ...]:
    rows: list[tuple[int, float]] = []
    if isinstance(schedule, Mapping):
        items = tuple(schedule.items())
    else:
        try:
            items = tuple(schedule)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("visit reward schedule must be a mapping or sequence.") from exc

    for item in items:
        if isinstance(item, Mapping):
            size_value = item.get("size", item.get("max_size"))
            scale_value = item.get("scale", item.get("visit_reward_scale"))
        else:
            try:
                size_value, scale_value = item  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "visit reward schedule entries must be size/scale pairs."
                ) from exc
        try:
            stage_size = int(size_value)
            scale = float(scale_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("visit reward schedule entries must be numeric.") from exc
        if stage_size <= 0:
            raise ValueError("visit reward schedule sizes must be positive.")
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("visit reward schedule scales must be non-negative floats.")
        rows.append((stage_size, scale))

    normalized = tuple(sorted(rows, key=lambda row: row[0]))
    if any(left[0] == right[0] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("visit reward schedule sizes must be unique.")
    return normalized


def forage_training_profile(size: int) -> dict[str, int | float]:
    return _training_profile_for_size(int(size), FORAGE_STAGE_TRAINING_PROFILE)


def _training_profile_for_size(
    size: int,
    training_profile: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    for profile in training_profile:
        if int(size) <= int(profile["max_size"]):
            return {
                "global_update_cap": int(profile["global_update_cap"]),
                "num_steps": int(profile["num_steps"]),
                "gamma": float(profile["gamma"]),
            }
    last_profile = training_profile[-1]
    return {
        "global_update_cap": int(last_profile["global_update_cap"]),
        "num_steps": int(last_profile["num_steps"]),
        "gamma": float(last_profile["gamma"]),
    }


def build_forage_curriculum_stages(
    stage_sizes: Sequence[int] = FORAGE_STAGE_SIZES,
) -> list[dict[str, int | float | str]]:
    return [
        {
            "name": f"{size}x{size}",
            "width": int(size),
            "height": int(size),
            "food_count": curriculum_food_count(int(size)),
            "food_sources": curriculum_food_sources(int(size)),
            "cookie_distance": min(1 + (int(size) - 4) // 2, int(size) // 2),
            "max_steps": max(48, 4 * int(size) * int(size)),
            **forage_training_profile(int(size)),
        }
        for size in stage_sizes
    ]


def build_exploration_to_forage_curriculum_stages(
    final_args: Mapping[str, Any],
    stage_sizes: Sequence[int] = EXPLORATION_TO_FORAGE_STAGE_SIZES,
    *,
    visit_reward_schedule: object = None,
    stage_update_multiplier: float = 1.0,
) -> list[dict[str, int | float | str | bool]]:
    final_width = int(final_args.get("width", 50))
    final_height = int(final_args.get("height", final_width))
    if final_width != final_height:
        raise ValueError("exploration-to-forage curriculum currently expects square maps.")
    if not stage_sizes:
        raise ValueError("stage_sizes must not be empty.")
    stage_sizes = tuple(int(size) for size in stage_sizes)
    if stage_sizes[-1] != final_width:
        raise ValueError("last exploration-to-forage stage must match the final map size.")
    if any(size <= 1 for size in stage_sizes):
        raise ValueError("stage_sizes must be greater than one.")
    if any(left >= right for left, right in zip(stage_sizes, stage_sizes[1:])):
        raise ValueError("stage_sizes must be strictly increasing.")

    final_food_count = int(final_args["food_count"])
    final_food_sources = int(final_args["food_sources"])
    final_cookie_distance = int(final_args["cookie_distance"])
    final_max_steps = int(final_args["max_steps"])
    visit_reward_fallback = float(final_args.get("visit_reward_scale", 0.0))
    stage_update_multiplier = _validate_stage_update_multiplier(
        stage_update_multiplier
    )
    stages: list[dict[str, int | float | str | bool]] = []
    for size in stage_sizes:
        is_final = size == final_width
        food_count = final_food_count if is_final else min(final_food_count, curriculum_food_count(size))
        food_sources = (
            final_food_sources
            if is_final
            else min(
                food_count,
                final_food_sources,
                max(
                    1,
                    (food_count + CURRICULUM_BITES_PER_FOOD_SOURCE - 1)
                    // CURRICULUM_BITES_PER_FOOD_SOURCE,
                ),
            )
        )
        cookie_distance = (
            final_cookie_distance
            if is_final
            else min(final_cookie_distance, min(1 + (size - 4) // 2, size // 2))
        )
        max_steps = (
            final_max_steps
            if is_final
            else max(
                48,
                (final_max_steps * size * size + final_width * final_width - 1)
                // (final_width * final_width),
            )
        )
        training_profile = forage_training_profile(size)
        training_profile["global_update_cap"] = int(
            math.ceil(int(training_profile["global_update_cap"]) * stage_update_multiplier)
        )
        stage: dict[str, int | float | str | bool] = {
            "name": f"{size}x{size}",
            "width": size,
            "height": size,
            "food_count": food_count,
            "food_sources": food_sources,
            "cookie_distance": cookie_distance,
            "max_steps": max_steps,
            "visit_reward_scale": exploration_to_forage_visit_reward_scale(
                size,
                schedule=visit_reward_schedule,
                fallback=visit_reward_fallback,
            ),
            **training_profile,
        }
        if is_final and final_args.get("save_best_model"):
            stage.update(
                {
                    "save_best_checkpoint": True,
                    "select_best_checkpoint": True,
                    "best_checkpoint_path": str(final_args["save_best_model"]),
                    "best_checkpoint_metric": str(
                        final_args.get("best_model_metric", "episode_return")
                    ),
                    "best_checkpoint_mode": str(final_args.get("best_model_mode", "max")),
                    "best_checkpoint_selection": str(
                        final_args.get("best_model_selection", "train")
                    ),
                }
            )
            for source_key, stage_key in (
                ("best_eval_episodes", "best_eval_episodes"),
                ("best_eval_interval", "best_eval_interval"),
                ("best_eval_seed_offset", "best_eval_seed_offset"),
                ("best_eval_action_mode", "best_eval_action_mode"),
                ("best_eval_move_temperature", "best_eval_move_temperature"),
                ("best_eval_write_temperature", "best_eval_write_temperature"),
                ("best_eval_shuffle_positions", "best_eval_shuffle_positions"),
            ):
                if source_key in final_args:
                    stage[stage_key] = final_args[source_key]
        stages.append(stage)
    return stages


def build_food_source_curriculum_stages(
    final_args: Mapping[str, Any],
    source_counts: Sequence[int],
    *,
    visit_reward_schedule: object = None,
    view_reward_schedule: object = None,
    stage_update_multiplier: float = 1.0,
) -> list[dict[str, int | float | str | bool]]:
    """Build fixed-arena forage stages with decreasing food-source counts."""

    if not source_counts:
        raise ValueError("source_counts must not be empty.")
    source_counts = tuple(int(count) for count in source_counts)
    if any(count <= 0 for count in source_counts):
        raise ValueError("source_counts must be positive.")
    if any(left <= right for left, right in zip(source_counts, source_counts[1:])):
        raise ValueError("source_counts must be strictly decreasing.")

    width = int(final_args["width"])
    height = int(final_args.get("height", width))
    fixed_food_count = int(final_args["food_count"])
    cookies_per_source = int(final_args.get("cookies_per_source", 0))
    if cookies_per_source < 0:
        raise ValueError("cookies_per_source must be non-negative.")
    if any(count > fixed_food_count for count in source_counts):
        raise ValueError("source_counts must not exceed food_count.")

    visit_reward_fallback = float(final_args.get("visit_reward_scale", 0.0))
    view_reward_fallback = float(final_args.get("view_reward_scale", 0.0))
    stage_update_multiplier = _validate_stage_update_multiplier(
        stage_update_multiplier
    )
    base_profile = forage_training_profile(max(width, height))
    base_profile["global_update_cap"] = int(
        math.ceil(int(base_profile["global_update_cap"]) * stage_update_multiplier)
    )

    stages: list[dict[str, int | float | str | bool]] = []
    final_source_count = source_counts[-1]
    for source_count in source_counts:
        is_final = source_count == final_source_count
        stage: dict[str, int | float | str | bool] = {
            "name": f"{width}x{height}_sources_{source_count:02d}",
            "width": width,
            "height": height,
            "food_count": fixed_food_count,
            "food_sources": source_count,
            "cookie_distance": int(final_args["cookie_distance"]),
            "max_steps": int(final_args["max_steps"]),
            "visit_reward_scale": food_source_curriculum_visit_reward_scale(
                source_count,
                schedule=visit_reward_schedule,
                fallback=visit_reward_fallback,
            ),
            "view_reward_scale": food_source_curriculum_visit_reward_scale(
                source_count,
                schedule=view_reward_schedule,
                fallback=view_reward_fallback,
            ),
            "view_reward_decay": float(final_args.get("view_reward_decay", 1.0)),
            "border_view_penalty": float(final_args.get("border_view_penalty", 0.0)),
            "border_moat_width": int(final_args.get("border_moat_width", 0)),
            "border_moat_penalty": float(final_args.get("border_moat_penalty", 0.0)),
            **base_profile,
        }
        if is_final and final_args.get("save_best_model"):
            stage.update(
                {
                    "save_best_checkpoint": True,
                    "select_best_checkpoint": True,
                    "best_checkpoint_path": str(final_args["save_best_model"]),
                    "best_checkpoint_metric": str(
                        final_args.get("best_model_metric", "episode_return")
                    ),
                    "best_checkpoint_mode": str(final_args.get("best_model_mode", "max")),
                    "best_checkpoint_selection": str(
                        final_args.get("best_model_selection", "train")
                    ),
                }
            )
            for source_key, stage_key in (
                ("best_eval_episodes", "best_eval_episodes"),
                ("best_eval_interval", "best_eval_interval"),
                ("best_eval_seed_offset", "best_eval_seed_offset"),
                ("best_eval_action_mode", "best_eval_action_mode"),
                ("best_eval_move_temperature", "best_eval_move_temperature"),
                ("best_eval_write_temperature", "best_eval_write_temperature"),
                ("best_eval_shuffle_positions", "best_eval_shuffle_positions"),
            ):
                if source_key in final_args:
                    stage[stage_key] = final_args[source_key]
        stages.append(stage)
    return stages


def build_food_cluster_curriculum_stages(
    final_args: Mapping[str, Any],
    source_counts: Sequence[int],
    cluster_radii: Sequence[int],
    *,
    visit_reward_schedule: object = None,
    view_reward_schedule: object = None,
    stage_update_multiplier: float = 1.0,
) -> list[dict[str, int | float | str | bool]]:
    """Build fixed-arena stages with two macro food sources and shrinking footprints."""

    if not source_counts:
        raise ValueError("source_counts must not be empty.")
    source_counts = tuple(int(count) for count in source_counts)
    cluster_radii = tuple(int(radius) for radius in cluster_radii)
    if len(source_counts) != len(cluster_radii):
        raise ValueError("source_counts and cluster_radii must have the same length.")
    if any(count <= 0 for count in source_counts):
        raise ValueError("source_counts must be positive.")
    if any(radius < 0 for radius in cluster_radii):
        raise ValueError("cluster_radii must be non-negative.")
    if any(left <= right for left, right in zip(source_counts, source_counts[1:])):
        raise ValueError("source_counts must be strictly decreasing.")
    if any(left < right for left, right in zip(cluster_radii, cluster_radii[1:])):
        raise ValueError("cluster_radii must be non-increasing.")

    width = int(final_args["width"])
    height = int(final_args.get("height", width))
    fixed_food_count = int(final_args["food_count"])
    cluster_count = int(final_args.get("food_cluster_count", final_args["food_sources"]))
    if cluster_count <= 0:
        raise ValueError("food_cluster_count must be positive.")
    if any(count > fixed_food_count for count in source_counts):
        raise ValueError("source_counts must not exceed food_count.")
    if any(count < cluster_count for count in source_counts):
        raise ValueError("source_counts must be at least food_cluster_count.")
    for count, radius in zip(source_counts, cluster_radii):
        max_cluster_positions = cluster_count * (2 * radius + 1) ** 2
        if count > max_cluster_positions:
            raise ValueError("source_counts must fit inside each cluster footprint.")

    visit_reward_fallback = float(final_args.get("visit_reward_scale", 0.0))
    view_reward_fallback = float(final_args.get("view_reward_scale", 0.0))
    stage_update_multiplier = _validate_stage_update_multiplier(
        stage_update_multiplier
    )
    base_profile = forage_training_profile(max(width, height))
    base_profile["global_update_cap"] = int(
        math.ceil(int(base_profile["global_update_cap"]) * stage_update_multiplier)
    )

    stages: list[dict[str, int | float | str | bool]] = []
    final_source_count = source_counts[-1]
    for source_count, cluster_radius in zip(source_counts, cluster_radii):
        is_final = source_count == final_source_count
        stage: dict[str, int | float | str | bool] = {
            "name": (
                f"{width}x{height}_clusters_{cluster_count:02d}_"
                f"r{cluster_radius:02d}_sources_{source_count:03d}"
            ),
            "width": width,
            "height": height,
            "food_count": fixed_food_count,
            "food_sources": source_count,
            "food_cluster_count": cluster_count,
            "food_cluster_radius": cluster_radius,
            "cookie_distance": int(final_args["cookie_distance"]),
            "max_steps": int(final_args["max_steps"]),
            "visit_reward_scale": food_source_curriculum_visit_reward_scale(
                source_count,
                schedule=visit_reward_schedule,
                fallback=visit_reward_fallback,
            ),
            "view_reward_scale": food_source_curriculum_visit_reward_scale(
                source_count,
                schedule=view_reward_schedule,
                fallback=view_reward_fallback,
            ),
            "view_reward_decay": float(final_args.get("view_reward_decay", 1.0)),
            "border_view_penalty": float(final_args.get("border_view_penalty", 0.0)),
            "border_moat_width": int(final_args.get("border_moat_width", 0)),
            "border_moat_penalty": float(final_args.get("border_moat_penalty", 0.0)),
            **base_profile,
        }
        if is_final and final_args.get("save_best_model"):
            stage.update(
                {
                    "save_best_checkpoint": True,
                    "select_best_checkpoint": True,
                    "best_checkpoint_path": str(final_args["save_best_model"]),
                    "best_checkpoint_metric": str(
                        final_args.get("best_model_metric", "episode_return")
                    ),
                    "best_checkpoint_mode": str(final_args.get("best_model_mode", "max")),
                    "best_checkpoint_selection": str(
                        final_args.get("best_model_selection", "train")
                    ),
                }
            )
            for source_key, stage_key in (
                ("best_eval_episodes", "best_eval_episodes"),
                ("best_eval_interval", "best_eval_interval"),
                ("best_eval_seed_offset", "best_eval_seed_offset"),
                ("best_eval_action_mode", "best_eval_action_mode"),
                ("best_eval_move_temperature", "best_eval_move_temperature"),
                ("best_eval_write_temperature", "best_eval_write_temperature"),
                ("best_eval_shuffle_positions", "best_eval_shuffle_positions"),
            ):
                if source_key in final_args:
                    stage[stage_key] = final_args[source_key]
        stages.append(stage)
    return stages


def food_source_curriculum_visit_reward_scale(
    source_count: int,
    *,
    schedule: object,
    fallback: float,
) -> float:
    """Return a configured source-count-specific visit reward scale."""

    if fallback < 0.0:
        raise ValueError("fallback visit reward scale must be a non-negative float.")
    if schedule is None:
        return fallback
    normalized = _normalize_visit_reward_schedule(schedule)
    if not normalized:
        return fallback
    source_count = int(source_count)
    if source_count <= 0:
        raise ValueError("source count must be positive.")
    for scheduled_count, scale in normalized:
        if source_count == scheduled_count:
            return scale
    return fallback


def _validate_stage_update_multiplier(value: object) -> float:
    try:
        multiplier = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("stage update multiplier must be a positive float.") from exc
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("stage update multiplier must be a positive float.")
    return multiplier


def build_exploration_curriculum_stages(
    stage_sizes: Sequence[int] = EXPLORATION_STAGE_SIZES,
    *,
    training_profile: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, int | float | str]]:
    return [
        {
            "name": f"{size}x{size}",
            "width": int(size),
            "height": int(size),
            "food_count": curriculum_food_count(int(size)),
            "food_sources": curriculum_food_sources(int(size)),
            "cookie_distance": min(1 + (int(size) - 4) // 2, int(size) // 2),
            "max_steps": exploration_max_steps(int(size)),
            **exploration_training_profile(
                int(size),
                training_profile=training_profile,
            ),
        }
        for size in stage_sizes
    ]


def build_maze_exploration_curriculum_stages(
    stage_sizes: Sequence[int] = MAZE_EXPLORATION_STAGE_SIZES,
    *,
    training_profile: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, int | float | str]]:
    return build_exploration_curriculum_stages(
        stage_sizes,
        training_profile=training_profile,
    )


def exploration_training_profile(
    size: int,
    *,
    training_profile: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int | float]:
    return _training_profile_for_size(
        int(size),
        training_profile or EXPLORATION_STAGE_TRAINING_PROFILE,
    )


def build_forage_common_args(
    stages: Sequence[Mapping[str, Any]],
    *,
    num_envs: int,
    num_steps: int,
    actor_vision_radius: int,
    write_bits: int,
    gamma: float = 0.99,
    write_while_moving: bool = True,
    seed: int = 1,
) -> list[str]:
    max_width = max(int(stage["width"]) for stage in stages)
    max_height = max(int(stage["height"]) for stage in stages)
    args = [
        "--num-envs",
        str(num_envs),
        "--num-steps",
        str(num_steps),
        "--num-minibatches",
        "4",
        "--update-epochs",
        "4",
        "--gamma",
        str(float(gamma)),
        "--obs-width",
        str(max_width),
        "--obs-height",
        str(max_height),
        "--actor-vision-radius",
        str(actor_vision_radius),
        "--write-bits",
        str(write_bits),
        "--num-ants",
        "1",
        "--random-food",
        "--random-hub",
        "--pickup-bonus",
        "0.25",
        "--hidden-size",
        "128",
        "--seed",
        str(seed),
        "--quiet",
    ]
    if write_while_moving:
        args.append("--write-while-moving")
    return args


def build_exploration_common_args(
    stages: Sequence[Mapping[str, Any]],
    *,
    num_envs: int,
    num_steps: int,
    actor_vision_radius: int,
    write_bits: int,
    gamma: float = 0.99,
    seed: int = 1,
) -> list[str]:
    max_width = max(int(stage["width"]) for stage in stages)
    max_height = max(int(stage["height"]) for stage in stages)
    return [
        "--num-envs",
        str(num_envs),
        "--num-steps",
        str(num_steps),
        "--num-minibatches",
        "4",
        "--update-epochs",
        "4",
        "--gamma",
        str(float(gamma)),
        "--obs-width",
        str(max_width),
        "--obs-height",
        str(max_height),
        "--actor-vision-radius",
        str(actor_vision_radius),
        "--write-bits",
        str(write_bits),
        "--num-ants",
        "1",
        "--reward-mode",
        "explore",
        "--no-food-termination",
        "--terminate-on-full-coverage",
        "--write-action-ablation",
        "--random-food",
        "--random-hub",
        "--pickup-bonus",
        "0.0",
        "--hidden-size",
        "128",
        "--seed",
        str(seed),
        "--quiet",
    ]


def build_maze_exploration_common_args(
    stages: Sequence[Mapping[str, Any]],
    *,
    num_envs: int,
    num_steps: int,
    actor_vision_radius: int,
    write_bits: int,
    gamma: float = 0.99,
    seed: int = 1,
    maze_corridor_width: int = 3,
    maze_wall_width: int = 1,
    maze_seed: int = 0,
) -> list[str]:
    max_width = max(int(stage["width"]) for stage in stages)
    max_height = max(int(stage["height"]) for stage in stages)
    return [
        "--num-envs",
        str(num_envs),
        "--num-steps",
        str(num_steps),
        "--num-minibatches",
        "4",
        "--update-epochs",
        "4",
        "--gamma",
        str(float(gamma)),
        "--obs-width",
        str(max_width),
        "--obs-height",
        str(max_height),
        "--actor-vision-radius",
        str(actor_vision_radius),
        "--write-bits",
        str(write_bits),
        "--num-ants",
        "1",
        "--reward-mode",
        "explore",
        "--no-food-termination",
        "--terminate-on-full-coverage",
        "--write-while-moving",
        "--random-food",
        "--random-hub",
        "--maze-obstacles",
        "--maze-corridor-width",
        str(int(maze_corridor_width)),
        "--maze-wall-width",
        str(int(maze_wall_width)),
        "--maze-seed",
        str(int(maze_seed)),
        "--pickup-bonus",
        "0.0",
        "--hidden-size",
        "128",
        "--seed",
        str(seed),
        "--quiet",
    ]


def run_jax_smoke(train_main: Callable[..., dict[str, float]]) -> dict[str, float]:
    return train_main(
        [
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--max-steps",
            "8",
            "--write-bits",
            "1",
            "--hidden-size",
            "16",
            "--seed",
            "11",
            "--quiet",
        ]
    )


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
    checkpoint_video_policy_temperature = _validate_rollout_policy_temperature(
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


def _forage_stage_training_profiles(
    stages: Sequence[Mapping[str, Any]],
    *,
    common_args: Sequence[str],
    fallback_update_timesteps: int,
    fallback_update_cap: int,
) -> list[dict[str, Any]]:
    return [
        {
            "stage": str(stage["name"]),
            "global_update_cap": int(stage.get("global_update_cap", fallback_update_cap)),
            "num_steps": int(stage["num_steps"]) if "num_steps" in stage else None,
            "gamma": float(stage["gamma"]) if "gamma" in stage else None,
            "update_timesteps": _forage_stage_update_timesteps(
                stage,
                common_args=common_args,
                fallback_update_timesteps=fallback_update_timesteps,
            ),
        }
        for stage in stages
    ]


def _forage_stage_update_timesteps(
    stage: Mapping[str, Any],
    *,
    common_args: Sequence[str],
    fallback_update_timesteps: int,
) -> int:
    if "update_timesteps" in stage:
        return int(stage["update_timesteps"])
    if "num_steps" not in stage:
        return int(fallback_update_timesteps)
    num_envs = _argv_int(common_args, "--num-envs")
    if num_envs is None:
        return int(fallback_update_timesteps)
    return update_timesteps(num_envs=num_envs, num_steps=int(stage["num_steps"]))


def _argv_int(argv: Sequence[str], option: str) -> int | None:
    try:
        index = len(argv) - 1 - list(reversed(argv)).index(option)
    except ValueError:
        return None
    try:
        return int(argv[index + 1])
    except (IndexError, ValueError):
        return None


def _strip_wandb_cli_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    stripped: list[str] = []
    removed: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        value = str(values[index])
        if value in _WANDB_CLI_VALUE_ARGS:
            removed.append(value)
            index += 1
            if index < len(values):
                removed.append(str(values[index]))
                index += 1
            continue
        if value in _WANDB_CLI_VARARGS:
            removed.append(value)
            index += 1
            while index < len(values) and not str(values[index]).startswith("--"):
                removed.append(str(values[index]))
                index += 1
            continue
        stripped.append(value)
        index += 1
    return stripped, removed


def _advance_progress_to(
    progress: Any,
    *,
    update_index: int,
    previous_update_index: int,
) -> int:
    next_update_index = int(update_index)
    progress.update(max(0, next_update_index - int(previous_update_index)))
    return next_update_index


def _wandb_preview_enabled(max_frames: int | None) -> bool:
    return max_frames is None or int(max_frames) > 0


def _wandb_preview_stage_enabled(
    stage_name: object,
    enabled_stage_names: Sequence[str] | None,
) -> bool:
    if enabled_stage_names is None:
        return True
    return str(stage_name) in {str(name) for name in enabled_stage_names}


def _validate_wandb_preview_stage_names(
    stages: Sequence[Mapping[str, Any]],
    enabled_stage_names: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if enabled_stage_names is None:
        return None

    requested_stage_names = tuple(str(name) for name in enabled_stage_names)
    generated_stage_names = tuple(str(stage["name"]) for stage in stages)
    generated_stage_name_set = set(generated_stage_names)
    unknown_stage_names = tuple(
        name for name in requested_stage_names if name not in generated_stage_name_set
    )
    if unknown_stage_names:
        unknown_text = ", ".join(unknown_stage_names)
        available_text = ", ".join(generated_stage_names)
        raise ValueError(
            "wandb video stage names must match generated curriculum stage names; "
            f"unknown: {unknown_text}. Available stages: {available_text}"
        )
    return requested_stage_names


def _validate_wandb_video_rollout_count(value: object) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("wandb video rollout count must be a positive integer.") from exc
    if count < 1:
        raise ValueError("wandb video rollout count must be a positive integer.")
    return count


def _wandb_video_seed_offset_base(value: int | None) -> int:
    if value is None:
        return NOTEBOOK_ROLLOUT_SEED_OFFSET + int(time.time_ns() % 1_000_000_000)
    try:
        seed_offset_base = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("wandb video seed offset base must be non-negative.") from exc
    if seed_offset_base < 0:
        raise ValueError("wandb video seed offset base must be non-negative.")
    return seed_offset_base


def _wandb_preview_video_key(
    *,
    prefix: str,
    stage_name: object,
    preview_index: int,
    preview_count: int,
) -> str:
    stage_key = f"{prefix}/{stage_name}"
    if preview_count == 1:
        return stage_key
    return f"{stage_key}/rollout_{preview_index + 1:02d}"


def _render_forage_wandb_previews(
    *,
    checkpoint_path: Path,
    checkpoint_dir: Path,
    stage_index: int,
    max_frames: int | None,
    policy_temperature: float,
    rollout_count: int,
    seed_offset_base: int,
) -> list[Path]:
    media_dir = checkpoint_dir.parent / "media" / "wandb_previews"
    output_paths: list[Path] = []
    for preview_index in range(int(rollout_count)):
        suffix = (
            "_preview.mp4"
            if int(rollout_count) == 1
            else f"_preview_{preview_index + 1:02d}.mp4"
        )
        output_path = media_dir / f"{checkpoint_path.stem}{suffix}"
        seed_offset = (
            int(seed_offset_base)
            + (int(stage_index) - 1) * int(rollout_count)
            + preview_index
        )
        output_paths.append(
            render_checkpoint(
                checkpoint_path,
                output_path,
                backend="jax",
                seed_offset=seed_offset,
                reuse_existing=False,
                max_frames=max_frames,
                tile_size=NOTEBOOK_ROLLOUT_TILE_SIZE,
                policy_temperature=policy_temperature,
            )
        )
    return output_paths


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


def ant_count_training_args(
    base_args: Mapping[str, Any],
    *,
    communication_bits: int,
) -> dict[str, Any]:
    return {
        **base_args,
        "width": 25,
        "height": 25,
        "obs_width": 50,
        "obs_height": 50,
        "food_count": 23,
        "food_sources": 6,
        "cookie_distance": 11,
        "max_steps": 2500,
        "write_bits": int(communication_bits),
        "write_while_moving": True,
    }


def validate_ant_count_stages(*, ant_stages: Sequence[int], source_num_ants: int) -> None:
    if any(num_ants <= source_num_ants for num_ants in ant_stages):
        raise ValueError("ant stages must increase beyond the source checkpoint's ant count.")
    if not _strictly_increasing(ant_stages):
        raise ValueError("ant stages must be increasing.")


def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


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


def ant_count_train_args(
    *,
    common_args: Sequence[str],
    experiment_name: str,
    target_num_ants: int,
    communication_bits: int,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    load_model: Path,
    run_dir: Path,
) -> list[str]:
    return [
        *common_args,
        "--exp-name",
        f"{experiment_name}_{target_num_ants}_ants",
        "--write-bits",
        str(communication_bits),
        "--num-ants",
        str(target_num_ants),
        "--total-timesteps",
        str(update_timesteps_per_stage * global_update_cap),
        "--load-model",
        str(load_model),
        "--run-dir",
        str(run_dir),
    ]


def expand_critic_input_for_ant_count(
    params: Any,
    *,
    source_num_ants: int,
    target_num_ants: int,
) -> Any:
    import jax.numpy as jnp

    from ant_byte_env.training.jax_mappo.core import JaxMAPPOParams, LinearParams

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

    from ant_byte_env.jax_autocurriculum_env import JaxAntByteAutoCurriculumEnv
    from ant_byte_env.jax_env import JaxAntByteForagingEnv
    from ant_byte_env.training.jax_mappo.cli import parse_args
    from ant_byte_env.training.jax_mappo.core import (
        build_actor_observations,
        build_central_observations,
        food_observation_scale,
    )
    from ant_byte_env.training.jax_mappo.curriculum import reset_batch

    args = parse_args(list(argv))
    env_kwargs = {
        "width": args.width,
        "height": args.height,
        "num_ants": args.num_ants,
        "food_count": args.food_count,
        "food_source_count": args.food_sources,
        "max_steps": args.max_steps,
        "random_food": args.random_food,
        "random_hub": args.random_hub,
        "random_ant_spawn": args.random_ant_spawn,
        "random_ant_spawn_radius": args.random_ant_spawn_radius,
        "step_penalty": args.step_penalty,
        "completion_bonus": getattr(args, "completion_bonus", 0.0),
        "write_penalty": args.write_penalty,
        "write_bits": args.write_bits,
        "write_while_moving": args.write_while_moving,
        "per_ant_write_channels": bool(getattr(args, "per_ant_write_channels", False)),
    }
    if bool(getattr(args, "autocurriculum", False)):
        env = JaxAntByteAutoCurriculumEnv(
            **env_kwargs,
            start_size=args.autocurriculum_start_size,
            success_cookies=args.autocurriculum_success_cookies,
            actor_vision_radius=args.actor_vision_radius,
        )
    else:
        env = JaxAntByteForagingEnv(
            **env_kwargs,
            hub_center_window_size=int(getattr(args, "hub_center_window_size", 0)),
            terminate_on_food_delivery=bool(getattr(args, "food_termination", True)),
            terminate_on_full_coverage=bool(
                getattr(args, "terminate_on_full_coverage", False)
            ),
            maze_obstacles=bool(getattr(args, "maze_obstacles", False)),
            maze_corridor_width=int(getattr(args, "maze_corridor_width", 3)),
            maze_wall_width=int(getattr(args, "maze_wall_width", 1)),
            maze_seed=int(getattr(args, "maze_seed", 0)),
        )
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
    from ant_byte_env.training.jax_mappo.core import init_adam_state
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


def stage_update_progress(label: str, total_updates: int) -> Any:
    from tqdm.auto import tqdm

    return tqdm(
        range(1, int(total_updates) + 1),
        total=int(total_updates),
        desc=label,
        bar_format="{desc}: {n_fmt}/{total_fmt} updates |{bar}| {elapsed}<{remaining} {postfix}",
        leave=True,
    )


def forage_checkpoint_paths(
    checkpoint_dir: Path,
    stages: Sequence[Mapping[str, Any]],
) -> list[Path]:
    return [checkpoint_dir / f"jax_mappo_forage_stage1_{stage['name']}.pkl" for stage in stages]


def exploration_checkpoint_paths(
    checkpoint_dir: Path,
    stages: Sequence[Mapping[str, Any]],
) -> list[Path]:
    return [checkpoint_dir / f"jax_mappo_explore_{stage['name']}.pkl" for stage in stages]


def maze_exploration_checkpoint_paths(
    checkpoint_dir: Path,
    stages: Sequence[Mapping[str, Any]],
) -> list[Path]:
    return [
        checkpoint_dir / f"jax_mappo_maze_explore_{stage['name']}.pkl"
        for stage in stages
    ]


def communication_checkpoint_paths(run_dir: Path, bit_stages: Sequence[int]) -> list[Path]:
    return [run_dir / f"{bits}_bits" / "checkpoints" / "model.pkl" for bits in bit_stages]


def ant_count_checkpoint_paths(run_dir: Path, ant_stages: Sequence[int]) -> list[Path]:
    return [
        run_dir / f"{num_ants}_ants" / "checkpoints" / "model.pkl"
        for num_ants in ant_stages
    ]


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
    policy_temperature = _validate_rollout_policy_temperature(
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

    policy_temperature = _validate_rollout_policy_temperature(
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
