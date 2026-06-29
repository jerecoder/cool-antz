"""Curriculum stage builders for notebook and autoresearch workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "FORAGE_STAGE_SIZES",
    "FORAGE_STAGE_TRAINING_PROFILE",
    "FORAGE_WANDB_PREVIEW_STAGE_NAMES",
    "EXPLORATION_TO_FORAGE_STAGE_SIZES",
    "EXPLORATION_TO_FORAGE_WANDB_PREVIEW_STAGE_NAMES",
    "EXPLORATION_TO_FORAGE_VISIT_REWARD_SCHEDULE",
    "EXPLORATION_STAGE_SIZES",
    "EXPLORATION_STAGE_TRAINING_PROFILE",
    "EXPLORATION_WANDB_PREVIEW_STAGE_NAMES",
    "MAZE_EXPLORATION_STAGE_SIZES",
    "MAZE_EXPLORATION_WANDB_PREVIEW_STAGE_NAMES",
    "CURRICULUM_BITES_PER_FOOD_SOURCE",
    "EXPLORATION_MAX_STEPS_PER_CELL",
    "curriculum_food_count",
    "curriculum_food_sources",
    "exploration_max_steps",
    "exploration_to_forage_visit_reward_scale",
    "forage_training_profile",
    "build_forage_curriculum_stages",
    "build_exploration_to_forage_curriculum_stages",
    "build_food_source_curriculum_stages",
    "build_food_cluster_curriculum_stages",
    "food_source_curriculum_visit_reward_scale",
    "build_exploration_curriculum_stages",
    "build_maze_exploration_curriculum_stages",
    "exploration_training_profile",
]


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
            "random_food_same_distance": bool(
                final_args.get("random_food_same_distance", False)
            ),
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
