"""CLI parsing for adversarial frozen-opponent JAX MAPPO."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ant_byte_env import DEFAULT_ACTOR_VISION_DEPTH, DEFAULT_WRITE_BITS, MAX_WRITE_BITS
from ant_byte_env.experiments import namespace_to_jsonable
from ant_byte_env.training.jax_mappo.adversarial.actions import ADVERSARIAL_ACTION_MODES

OPPONENT_ACTION_MODES = ADVERSARIAL_ACTION_MODES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train adversarial JAX MAPPO against a frozen opponent."
    )
    parser.add_argument("--exp-name", default="jax_mappo_adversarial_frozen")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1)

    parser.add_argument("--total-timesteps", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--training-rollout-temperature", type=float, default=1.0)

    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--num-ants-per-team", type=int, default=1)
    parser.add_argument("--food-count", type=int, default=2)
    parser.add_argument("--food-sources", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--delivery-limit", type=int, default=None)
    parser.add_argument("--actor-vision-radius", type=int, default=DEFAULT_ACTOR_VISION_DEPTH)
    parser.add_argument("--write-bits", type=int, default=DEFAULT_WRITE_BITS)
    parser.add_argument("--write-while-moving", action="store_true")
    parser.add_argument("--random-food", action="store_true")
    parser.add_argument("--random-hub", action="store_true")
    parser.add_argument("--layout-margin", type=int, default=0)
    parser.add_argument("--hub-center-window-size", type=int, default=0)
    parser.add_argument("--hub-pair-distance", type=int, default=0)
    parser.add_argument("--hub-pair-distance-min", type=int, default=0)
    parser.add_argument("--hub-pair-distance-max", type=int, default=0)
    parser.add_argument("--food-midpoint-window-size", type=int, default=0)
    parser.add_argument("--no-food-termination", dest="food_termination", action="store_false")
    parser.set_defaults(food_termination=True)

    parser.add_argument("--learner-team", type=int, choices=(0, 1), default=0)
    parser.add_argument("--learner-load-model", type=Path, default=None)
    parser.add_argument("--opponent-load-model", type=Path, default=None)
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument(
        "--opponent-action-mode",
        choices=OPPONENT_ACTION_MODES,
        default="deterministic",
    )
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-action-mode", choices=OPPONENT_ACTION_MODES, default="deterministic")
    parser.add_argument("--side-swap-eval", action="store_true")

    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    return _validate_args(parser.parse_args(argv))


def _validate_args(args: argparse.Namespace) -> argparse.Namespace:
    rollout_batch_size = int(args.num_envs) * int(args.num_steps)
    if args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("--num-envs and --num-steps must be positive.")
    if args.num_minibatches <= 0:
        raise ValueError("--num-minibatches must be positive.")
    if rollout_batch_size < args.num_minibatches:
        raise ValueError("--num-minibatches cannot exceed rollout batch size.")
    if rollout_batch_size % args.num_minibatches != 0:
        raise ValueError("--num-minibatches must evenly divide rollout batch size.")
    if args.update_epochs <= 0:
        raise ValueError("--update-epochs must be positive.")
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")
    if args.training_rollout_temperature <= 0.0:
        raise ValueError("--training-rollout-temperature must be positive.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.num_ants_per_team <= 0:
        raise ValueError("--num-ants-per-team must be positive.")
    if args.food_count < 0:
        raise ValueError("--food-count must be non-negative.")
    if args.food_sources <= 0:
        raise ValueError("--food-sources must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.actor_vision_radius < 0:
        raise ValueError("--actor-vision-radius must be non-negative.")
    _normalize_hub_distance_range(args)
    if args.food_midpoint_window_size < 0:
        raise ValueError("--food-midpoint-window-size must be non-negative.")
    if args.food_midpoint_window_size > max(args.width, args.height):
        raise ValueError("--food-midpoint-window-size must fit inside the map.")
    if args.write_bits <= 0 or args.write_bits > MAX_WRITE_BITS:
        raise ValueError(f"--write-bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    if args.delivery_limit is not None and args.delivery_limit <= 0:
        raise ValueError("--delivery-limit must be positive when provided.")
    if args.eval_episodes < 0:
        raise ValueError("--eval-episodes must be non-negative.")
    if (
        args.learner_load_model is None
        and args.resume_model is None
        and not args.allow_random_init
    ):
        raise ValueError(
            "--learner-load-model or --resume-model is required unless "
            "--allow-random-init is set."
        )
    args.num_ants = int(args.num_ants_per_team)
    args.critic_architecture = "mlp"
    return args


def _normalize_hub_distance_range(args: argparse.Namespace) -> None:
    diameter = int(args.width) + int(args.height) - 2
    fixed_distance = int(args.hub_pair_distance)
    min_distance = int(args.hub_pair_distance_min)
    max_distance = int(args.hub_pair_distance_max)
    if fixed_distance < 0 or min_distance < 0 or max_distance < 0:
        raise ValueError("hub pair distances must be non-negative.")
    if fixed_distance > 0 and (min_distance > 0 or max_distance > 0):
        raise ValueError(
            "use either --hub-pair-distance or --hub-pair-distance-min/max, not both."
        )
    if fixed_distance > 0:
        min_distance = fixed_distance
        max_distance = fixed_distance
    elif min_distance > 0 or max_distance > 0:
        if min_distance == 0:
            min_distance = 1
        if max_distance == 0:
            max_distance = min_distance
    if min_distance > max_distance:
        raise ValueError("--hub-pair-distance-min cannot exceed --hub-pair-distance-max.")
    if max_distance > diameter:
        raise ValueError("hub pair distance cannot exceed the map Manhattan diameter.")
    args.hub_pair_distance_min = min_distance
    args.hub_pair_distance_max = max_distance


def dry_run_payload(
    *,
    config_path: Path,
    experiment: str,
    argv: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "backend": "jax",
        "config": str(config_path),
        "experiment": experiment,
        "workflow": "adversarial_frozen_opponent",
        "argv": argv,
        "resolved_args": namespace_to_jsonable(args),
    }
