"""CLI argument builders shared by AntByte workflow surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ant_byte_env.experiments import config_args_to_argv

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
VISION_RANGE_ARG_EXCLUDES = {
    "exp_name",
    "actor_vision_radius",
    "total_timesteps",
    "save_model",
    "load_model",
    "run_dir",
}


def config_common_args(
    training_args: Mapping[str, Any],
    *,
    exclude: Iterable[str],
) -> list[str]:
    excluded = set(exclude)
    return config_args_to_argv(
        {key: value for key, value in training_args.items() if key not in excluded}
    )


def update_timesteps(*, num_envs: int, num_steps: int) -> int:
    return int(num_envs) * int(num_steps)


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


__all__ = [
    "ANT_COUNT_ARG_EXCLUDES",
    "AUTOCURRICULUM_ARG_EXCLUDES",
    "COMMUNICATION_ARG_EXCLUDES",
    "EXPLORATION_ARG_EXCLUDES",
    "EXPLORATION_TO_FORAGE_ARG_EXCLUDES",
    "SINGLE_CHECKPOINT_ARG_EXCLUDES",
    "build_exploration_common_args",
    "build_forage_common_args",
    "build_maze_exploration_common_args",
    "config_common_args",
    "update_timesteps",
]
