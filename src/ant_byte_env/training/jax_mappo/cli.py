"""Command-line parsing for JAX MAPPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ant_byte_env import DEFAULT_ACTOR_VISION_DEPTH, DEFAULT_WRITE_BITS, MAX_WRITE_BITS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JAX MAPPO on AntByte forage.")
    parser.add_argument("--exp-name", type=str, default="jax_mappo_forage")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")

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
            "Trainer-side penalty for each set write bit. Bit 0 costs this amount; "
            "higher bits are discounted by --write-bit-penalty-decay."
        ),
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
    parser.add_argument("--write-bits", type=int, default=DEFAULT_WRITE_BITS)
    parser.add_argument("--cookie-distance", type=int, default=1)
    parser.add_argument("--random-food", action="store_true")
    parser.add_argument("--random-hub", action="store_true")
    parser.add_argument("--pickup-bonus", type=float, default=0.25)
    parser.add_argument("--distance-bonus", type=float, default=0.02)
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Write resolved config, metrics, summary, and default checkpoint under this run directory.",
    )

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
    if args.cookie_distance <= 0:
        raise ValueError("--cookie-distance must be positive.")
    if args.food_count > 0 and args.width * args.height <= 1:
        raise ValueError("food_count requires at least one non-hub tile.")
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
    return args
