"""Command-line parsing for JAX MAPPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ant_byte_env import DEFAULT_ACTOR_VISION_DEPTH, DEFAULT_WRITE_BITS, MAX_WRITE_BITS

WRITE_HEAD_TRANSFER_MODES = ("repeat", "reset", "neutral-new")
EVALUATION_ACTION_MODES = (
    "deterministic",
    "sampled",
    "greedy_move_greedy_write",
    "greedy_move_sampled_write",
    "sampled_move_greedy_write",
    "sampled_move_sampled_write",
    "greedy_move_zero_write",
    "sampled_move_zero_write",
)
REWARD_MODES = ("forage", "explore")


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
    parser.add_argument(
        "--deterministic-rollout",
        action="store_true",
        help="Collect PPO rollouts with greedy argmax actions instead of sampling.",
    )
    parser.add_argument(
        "--deterministic-rollout-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of rollout action slots forced to greedy argmax actions. "
            "Use 0 for sampled PPO and 1 for fully deterministic rollouts."
        ),
    )
    parser.add_argument(
        "--deterministic-move-rollout-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of rollout action slots where only the movement head is forced "
            "to greedy argmax while write actions stay sampled."
        ),
    )
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
    parser.add_argument(
        "--completion-bonus",
        type=float,
        default=0.0,
        help="One-time reward bonus when all food is delivered before truncation.",
    )
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
        "--reward-mode",
        choices=REWARD_MODES,
        default="forage",
        help=(
            "Training reward objective. 'forage' uses delivery/pickup shaping; "
            "'explore' rewards newly visited distinct cells and ignores cookies."
        ),
    )
    parser.add_argument(
        "--write-head-transfer",
        choices=WRITE_HEAD_TRANSFER_MODES,
        default="repeat",
        help="How to initialize the write head when increasing write bits from a checkpoint.",
    )
    parser.add_argument(
        "--write-action-ablation",
        action="store_true",
        help="Force executed write actions to zero while keeping the write head present.",
    )
    parser.add_argument("--cookie-distance", type=int, default=1)
    parser.add_argument("--random-food", action="store_true")
    parser.add_argument("--random-hub", action="store_true")
    parser.add_argument(
        "--no-food-termination",
        dest="food_termination",
        action="store_false",
        help="Do not terminate episodes when all cookies have been delivered.",
    )
    parser.set_defaults(food_termination=True)
    parser.add_argument(
        "--random-ant-spawn",
        action="store_true",
        help=(
            "Spawn ants at random non-hub, non-food tiles on reset instead of "
            "placing every ant on the colony hub."
        ),
    )
    parser.add_argument(
        "--random-ant-spawn-radius",
        type=int,
        default=None,
        help=(
            "When --random-ant-spawn is enabled, restrict random ant spawn tiles "
            "to this Chebyshev radius around the colony hub. Omit for full-map spawn."
        ),
    )
    parser.add_argument(
        "--actor-hub-vector",
        action="store_true",
        help=(
            "Append each ant's normalized relative vector to the colony hub to "
            "the actor observation. Useful when hubs and ant spawns are randomized."
        ),
    )
    parser.add_argument(
        "--actor-nearest-food-vector",
        action="store_true",
        help=(
            "Append each ant's normalized relative vector to the nearest current "
            "food tile to the actor observation."
        ),
    )
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
        "--visit-reward-scale",
        type=float,
        default=0.0,
        help=(
            "Trainer-side bonus for newly visited cells. The bonus is multiplied by "
            "(1 - visited_fraction) ** visit_reward_decay so it fades as coverage grows."
        ),
    )
    parser.add_argument(
        "--visit-reward-decay",
        type=float,
        default=1.0,
        help="Decay exponent for --visit-reward-scale as more of the map is visited.",
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
    parser.add_argument(
        "--save-best-model",
        type=Path,
        default=None,
        help=(
            "Optionally overwrite this checkpoint whenever --best-model-metric "
            "improves at a logged training update."
        ),
    )
    parser.add_argument(
        "--best-model-metric",
        type=str,
        default="episode_return",
        help="Logged metric used to choose --save-best-model checkpoints.",
    )
    parser.add_argument(
        "--best-model-mode",
        choices=["max", "min"],
        default="max",
        help="Whether larger or smaller --best-model-metric values are better.",
    )
    parser.add_argument(
        "--best-model-selection",
        choices=["train", "eval"],
        default="train",
        help=(
            "Choose --save-best-model from logged training metrics or from a "
            "small held-out evaluation run at reported updates."
        ),
    )
    parser.add_argument(
        "--best-eval-episodes",
        type=int,
        default=8,
        help="Held-out episodes per evaluation when --best-model-selection=eval.",
    )
    parser.add_argument(
        "--best-eval-interval",
        type=int,
        default=0,
        help=(
            "Run held-out best-checkpoint evaluation every N updates. "
            "0 reuses --log-interval. The first and final updates are always evaluated."
        ),
    )
    parser.add_argument(
        "--best-eval-seed-offset",
        type=int,
        default=1_000_000,
        help="Reset seed offset for held-out best-checkpoint evaluation.",
    )
    parser.add_argument(
        "--best-eval-action-mode",
        choices=EVALUATION_ACTION_MODES,
        default="sampled_move_greedy_write",
        help="Deployment action mode used by held-out best-checkpoint evaluation.",
    )
    parser.add_argument(
        "--best-eval-move-temperature",
        type=float,
        default=1.0,
        help="Movement-head temperature for held-out best-checkpoint evaluation.",
    )
    parser.add_argument(
        "--best-eval-write-temperature",
        type=float,
        default=1.0,
        help="Write-head temperature for held-out best-checkpoint evaluation.",
    )
    parser.add_argument(
        "--no-best-eval-shuffle-positions",
        dest="best_eval_shuffle_positions",
        action="store_false",
        help="Disable random food/hub layouts during held-out best-checkpoint evaluation.",
    )
    parser.set_defaults(best_eval_shuffle_positions=True)
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
    if (
        args.deterministic_rollout_fraction < 0.0
        or args.deterministic_rollout_fraction > 1.0
    ):
        raise ValueError("--deterministic-rollout-fraction must be between 0 and 1.")
    if (
        args.deterministic_move_rollout_fraction < 0.0
        or args.deterministic_move_rollout_fraction > 1.0
    ):
        raise ValueError(
            "--deterministic-move-rollout-fraction must be between 0 and 1."
        )
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")
    if args.save_best_model is not None and not args.best_model_metric:
        raise ValueError(
            "--best-model-metric must be non-empty when --save-best-model is set."
        )
    if args.best_model_selection == "eval" and args.save_best_model is None:
        raise ValueError("--best-model-selection eval requires --save-best-model.")
    if args.best_eval_episodes <= 0:
        raise ValueError("--best-eval-episodes must be positive.")
    if args.best_eval_interval < 0:
        raise ValueError("--best-eval-interval must be non-negative.")
    if args.best_eval_seed_offset < 0:
        raise ValueError("--best-eval-seed-offset must be non-negative.")
    if args.best_eval_move_temperature <= 0.0:
        raise ValueError("--best-eval-move-temperature must be positive.")
    if args.best_eval_write_temperature <= 0.0:
        raise ValueError("--best-eval-write-temperature must be positive.")
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
    if args.random_ant_spawn_radius is not None and args.random_ant_spawn_radius < 0:
        raise ValueError("--random-ant-spawn-radius must be non-negative.")
    if args.write_bits <= 0 or args.write_bits > MAX_WRITE_BITS:
        raise ValueError(f"--write-bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    if args.autocurriculum and not args.food_termination:
        raise ValueError("--no-food-termination is only supported for non-autocurriculum runs.")
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
    if args.visit_reward_scale < 0.0:
        raise ValueError("--visit-reward-scale must be non-negative.")
    if args.visit_reward_decay < 0.0:
        raise ValueError("--visit-reward-decay must be non-negative.")
    if args.completion_bonus < 0.0:
        raise ValueError("--completion-bonus must be non-negative.")
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
