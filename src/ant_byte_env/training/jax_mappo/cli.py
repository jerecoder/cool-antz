"""Command-line parsing for JAX MAPPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ant_byte_env import DEFAULT_ACTOR_VISION_DEPTH, DEFAULT_WRITE_BITS, MAX_WRITE_BITS

WRITE_HEAD_TRANSFER_MODES = ("repeat", "reset", "neutral-new")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JAX MAPPO on AntByte forage.")
    parser.add_argument("--exp-name", type=str, default="jax_mappo_forage")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Report, print, and persist training metrics every N PPO updates.",
    )

    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)

    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--obs-width", type=int, default=None)
    parser.add_argument("--obs-height", type=int, default=None)
    parser.add_argument(
        "--autocurriculum",
        action="store_true",
        help="Grow the active square grid inside each episode after enough deliveries.",
    )
    parser.add_argument("--autocurriculum-start-size", type=int, default=4)
    parser.add_argument("--autocurriculum-success-cookies", type=int, default=6)
    parser.add_argument(
        "--actor-vision-radius",
        type=int,
        default=DEFAULT_ACTOR_VISION_DEPTH,
        help="Centered local actor-grid radius; the default radius 1 is a 3x3 grid.",
    )
    parser.add_argument("--num-ants", type=int, default=2)
    parser.add_argument("--food-count", type=int, default=4)
    parser.add_argument("--food-sources", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--write-penalty", type=float, default=0.0)
    parser.add_argument(
        "--write-bit-penalty",
        type=float,
        default=0.0,
        help=(
            "Trainer-side penalty for each set bit on applied write actions. Bit 0 costs "
            "this amount; higher bits are discounted by --write-bit-penalty-decay."
        ),
    )
    parser.add_argument(
        "--write-while-moving",
        action="store_true",
        help="Apply each write value after movement instead of only on stay/write actions.",
    )
    parser.add_argument(
        "--write-bit-penalty-decay",
        type=float,
        default=0.5,
        help="Geometric decay for higher write-bit penalties.",
    )
    parser.add_argument(
        "--write-entropy-bonus",
        type=float,
        default=0.0,
        help="Terminal bonus scale for normalized entropy over nonzero byte values.",
    )
    parser.add_argument(
        "--write-entropy-bonus-cap",
        type=float,
        default=0.15,
        help="Maximum terminal write-entropy bonus per environment.",
    )
    parser.add_argument(
        "--write-bit-entropy-bonus",
        type=float,
        default=0.0,
        help="Per-update bonus scale for balanced use of bits in applied write actions.",
    )
    parser.add_argument("--write-bits", type=int, default=DEFAULT_WRITE_BITS)
    parser.add_argument(
        "--write-head-transfer",
        choices=WRITE_HEAD_TRANSFER_MODES,
        default="repeat",
        help="How to initialize the write head when increasing write bits from a checkpoint.",
    )
    parser.add_argument("--cookie-distance", type=int, default=1)
    parser.add_argument("--random-food", action="store_true")
    parser.add_argument("--random-hub", action="store_true")
    parser.add_argument("--pickup-bonus", type=float, default=0.25)
    parser.add_argument(
        "--distance-bonus",
        type=float,
        default=0.0,
        help=(
            "Trainer-side normalized Manhattan progress bonus toward food while empty and "
            "toward the hub while carrying. Actor observations are unchanged."
        ),
    )
    parser.add_argument(
        "--stage-completion-bonus",
        type=float,
        default=0.0,
        help=(
            "Trainer-side bonus when an autocurriculum stage advances to the next active "
            "grid size. Actor observations are unchanged."
        ),
    )
    parser.add_argument(
        "--delivery-byte-trail-bonus",
        type=float,
        default=0.0,
        help=(
            "Trainer-side bonus per delivery scaled by pre-existing nonzero byte-trail "
            "tiles. Actor observations are unchanged."
        ),
    )
    parser.add_argument(
        "--delivery-byte-trail-target-tiles",
        type=float,
        default=8.0,
        help="Number of pre-existing nonzero byte tiles that saturates the delivery trail bonus.",
    )
    parser.add_argument(
        "--byte-follow-bonus",
        type=float,
        default=0.0,
        help=(
            "Trainer-side bonus when an empty ant moves onto a pre-existing byte tile "
            "and reduces nearest-food distance. Actor observations are unchanged."
        ),
    )
    parser.add_argument(
        "--carrying-byte-write-bonus",
        type=float,
        default=0.0,
        help=(
            "Trainer-side bonus for a carrying ant applying a nonzero write to a fresh "
            "non-food, non-hub tile. Actor observations are unchanged."
        ),
    )
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Write resolved config, metrics, summary, and default checkpoint under this run directory.",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None)

    args = parser.parse_args(argv)
    rollout_batch_size = args.num_envs * args.num_steps
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.num_minibatches <= 0:
        raise ValueError("--num-minibatches must be positive.")
    if rollout_batch_size < args.num_minibatches:
        raise ValueError("--num-minibatches cannot exceed rollout batch size.")
    if rollout_batch_size % args.num_minibatches != 0:
        raise ValueError("--num-minibatches must evenly divide rollout batch size.")
    if args.update_epochs <= 0:
        raise ValueError("--update-epochs must be positive.")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")
    if args.cookie_distance <= 0:
        raise ValueError("--cookie-distance must be positive.")
    if args.food_count > 0 and args.width * args.height <= 1:
        raise ValueError("food_count requires at least one non-hub tile.")
    if args.autocurriculum:
        if args.width != args.height:
            raise ValueError("--autocurriculum requires a square max grid.")
        if args.autocurriculum_start_size <= 0:
            raise ValueError("--autocurriculum-start-size must be positive.")
        if args.autocurriculum_start_size > args.width:
            raise ValueError("--autocurriculum-start-size must be no larger than --width.")
        if args.autocurriculum_success_cookies <= 0:
            raise ValueError("--autocurriculum-success-cookies must be positive.")
        if args.food_sources != 2:
            raise ValueError("--autocurriculum uses exactly two food sources.")
        if args.food_count != args.autocurriculum_success_cookies * args.food_sources:
            raise ValueError(
                "--food-count must equal --autocurriculum-success-cookies * --food-sources."
            )
    if args.obs_width is not None and args.obs_width < args.width:
        raise ValueError("--obs-width must be at least --width.")
    if args.obs_height is not None and args.obs_height < args.height:
        raise ValueError("--obs-height must be at least --height.")
    if args.actor_vision_radius < 0:
        raise ValueError("--actor-vision-radius must be non-negative.")
    if args.write_bits <= 0 or args.write_bits > MAX_WRITE_BITS:
        raise ValueError(f"--write-bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    if args.write_bit_penalty < 0.0:
        raise ValueError("--write-bit-penalty must be non-negative.")
    if not 0.0 <= args.write_bit_penalty_decay <= 1.0:
        raise ValueError("--write-bit-penalty-decay must be between 0 and 1.")
    if args.write_entropy_bonus < 0.0:
        raise ValueError("--write-entropy-bonus must be non-negative.")
    if args.write_entropy_bonus_cap < 0.0:
        raise ValueError("--write-entropy-bonus-cap must be non-negative.")
    if args.write_bit_entropy_bonus < 0.0:
        raise ValueError("--write-bit-entropy-bonus must be non-negative.")
    if args.distance_bonus < 0.0:
        raise ValueError("--distance-bonus must be non-negative.")
    if args.stage_completion_bonus < 0.0:
        raise ValueError("--stage-completion-bonus must be non-negative.")
    if args.delivery_byte_trail_bonus < 0.0:
        raise ValueError("--delivery-byte-trail-bonus must be non-negative.")
    if args.delivery_byte_trail_target_tiles <= 0.0:
        raise ValueError("--delivery-byte-trail-target-tiles must be positive.")
    if args.byte_follow_bonus < 0.0:
        raise ValueError("--byte-follow-bonus must be non-negative.")
    if args.carrying_byte_write_bonus < 0.0:
        raise ValueError("--carrying-byte-write-bonus must be non-negative.")
    return args
