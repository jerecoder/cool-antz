"""Shared JAX MAPPO environment construction."""

from __future__ import annotations

from typing import Any

from ant_byte_env.jax_autocurriculum_env import JaxAntByteAutoCurriculumEnv
from ant_byte_env.jax_distance_autocurriculum_env import JaxAntByteDistanceCurriculumEnv
from ant_byte_env.jax_env import JaxAntByteForagingEnv

JaxMappoEnv = (
    JaxAntByteForagingEnv | JaxAntByteAutoCurriculumEnv | JaxAntByteDistanceCurriculumEnv
)


def make_jax_mappo_env(args: Any) -> JaxMappoEnv:
    """Build the cooperative JAX MAPPO env from parsed CLI/checkpoint args."""

    common_kwargs = {
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
        "actor_vision_radius": int(getattr(args, "actor_vision_radius", 1)),
        "step_penalty": args.step_penalty,
        "completion_bonus": args.completion_bonus,
        "write_penalty": args.write_penalty,
        "write_bits": args.write_bits,
        "write_while_moving": args.write_while_moving,
        "per_ant_write_channels": bool(getattr(args, "per_ant_write_channels", False)),
    }
    if bool(getattr(args, "distance_autocurriculum", False)):
        return JaxAntByteDistanceCurriculumEnv(
            **common_kwargs,
            start_distance=int(getattr(args, "distance_autocurriculum_start_distance", 2)),
            max_distance=int(getattr(args, "distance_autocurriculum_max_distance", 128)),
            distance_multiplier=int(getattr(args, "distance_autocurriculum_multiplier", 2)),
            success_cookies=(
                int(args.distance_autocurriculum_success_cookies)
                if int(getattr(args, "distance_autocurriculum_success_cookies", 0)) > 0
                else None
            ),
            layout_margin=int(getattr(args, "layout_margin", 0)),
            hub_center_window_size=int(getattr(args, "hub_center_window_size", 0)),
            maze_obstacles=bool(getattr(args, "maze_obstacles", False)),
            maze_corridor_width=int(getattr(args, "maze_corridor_width", 3)),
            maze_wall_width=int(getattr(args, "maze_wall_width", 1)),
            maze_seed=int(getattr(args, "maze_seed", 0)),
        )
    if bool(getattr(args, "autocurriculum", False)):
        return JaxAntByteAutoCurriculumEnv(
            **common_kwargs,
            start_size=args.autocurriculum_start_size,
            success_cookies=args.autocurriculum_success_cookies,
        )
    return JaxAntByteForagingEnv(
        **common_kwargs,
        lethal_food_count=int(getattr(args, "lethal_food_count", 0)),
        lethal_food_source_count=int(getattr(args, "lethal_food_sources", 0)),
        death_penalty=float(getattr(args, "death_penalty", 0.0)),
        layout_margin=int(getattr(args, "layout_margin", 0)),
        hub_center_window_size=int(getattr(args, "hub_center_window_size", 0)),
        terminate_on_food_delivery=bool(args.food_termination),
        terminate_on_full_coverage=bool(getattr(args, "terminate_on_full_coverage", False)),
        maze_obstacles=bool(getattr(args, "maze_obstacles", False)),
        maze_corridor_width=int(getattr(args, "maze_corridor_width", 3)),
        maze_wall_width=int(getattr(args, "maze_wall_width", 1)),
        maze_seed=int(getattr(args, "maze_seed", 0)),
    )
