"""Command-line parsing for JAX MAPPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ant_byte_env import DEFAULT_ACTOR_VISION_DEPTH, DEFAULT_WRITE_BITS, MAX_WRITE_BITS

WRITE_HEAD_TRANSFER_MODES = ("repeat", "reset", "neutral-new")
DISTANCE_PROGRESS_NORMALIZERS = ("map", "stage")
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
CRITIC_ARCHITECTURES = ("mlp", "structured_mlp", "strided_cnn", "set_cnn", "resnet_cnn")


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
    parser.add_argument(
        "--jax-parallelism",
        choices=("single", "data"),
        default="single",
        help=(
            "JAX execution strategy. 'single' preserves the single-device trainer; "
            "'data' shards environments across local devices."
        ),
    )
    parser.add_argument(
        "--jax-device-count",
        type=int,
        default=0,
        help="Number of local JAX devices for --jax-parallelism=data. 0 uses all.",
    )
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
        "--training-rollout-temperature",
        type=float,
        default=1.0,
        help=(
            "Positive softmax temperature used for PPO rollout action sampling and "
            "policy logprobs. Lower values make sampled training actions greedier."
        ),
    )
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
    parser.add_argument(
        "--reset-env-each-update",
        action="store_true",
        help="Reset all vectorized environments before each PPO rollout.",
    )
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--actor-max-grad-norm", type=float, default=None)
    parser.add_argument("--critic-max-grad-norm", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--critic-architecture",
        choices=CRITIC_ARCHITECTURES,
        default="mlp",
        help=(
            "Centralized value-function architecture. 'mlp' preserves the original "
            "dense critic; 'structured_mlp' splits flattened grid and entity features; "
            "'strided_cnn' uses a lightweight downsampling CNN; 'set_cnn' combines "
            "that spatial path with per-ant set pooling; 'resnet_cnn' uses a compact "
            "residual CNN critic."
        ),
    )

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
    parser.add_argument("--distance-autocurriculum", action="store_true")
    parser.add_argument("--distance-autocurriculum-start-distance", type=int, default=2)
    parser.add_argument("--distance-autocurriculum-max-distance", type=int, default=128)
    parser.add_argument("--distance-autocurriculum-multiplier", type=int, default=2)
    parser.add_argument("--distance-autocurriculum-success-cookies", type=int, default=0)
    parser.add_argument(
        "--actor-vision-radius",
        type=int,
        default=DEFAULT_ACTOR_VISION_DEPTH,
        help="Centered local actor-grid radius; the default radius 1 is a 3x3 grid.",
    )
    parser.add_argument("--num-ants", type=int, default=2)
    parser.add_argument("--agent-identity-types", type=int, default=None)
    parser.add_argument("--food-count", type=int, default=4)
    parser.add_argument("--food-sources", type=int, default=1)
    parser.add_argument("--lethal-food-count", type=int, default=0)
    parser.add_argument("--lethal-food-sources", type=int, default=0)
    parser.add_argument(
        "--food-cluster-count",
        type=int,
        default=0,
        help=(
            "When positive with --random-food, sample food-source tiles around this "
            "many random macro-source centers instead of uniformly across the map."
        ),
    )
    parser.add_argument(
        "--food-cluster-radius",
        type=int,
        default=0,
        help=(
            "Chebyshev radius around each macro-source center for clustered food "
            "source sampling. Use 0 for one source tile per macro-source center."
        ),
    )
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
        "--per-ant-write-channels",
        action="store_true",
        help=(
            "Treat write bits as ant-owned channels: every ant observes all bits, "
            "but ant i can only set or clear bit i modulo --write-bits on a tile."
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
    parser.add_argument(
        "--random-food-same-distance",
        action="store_true",
        help="Sample normal and lethal food sources on the same Chebyshev ring.",
    )
    parser.add_argument("--random-hub", action="store_true")
    parser.add_argument(
        "--no-food-termination",
        dest="food_termination",
        action="store_false",
        help="Do not terminate episodes when all cookies have been delivered.",
    )
    parser.set_defaults(food_termination=True)
    parser.add_argument(
        "--terminate-on-full-coverage",
        action="store_true",
        help="Terminate non-autocurriculum episodes after every map cell has been visited.",
    )
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
        "--layout-margin",
        type=int,
        default=0,
        help=(
            "Restrict random hub, food, and random ant spawn candidates to cells at "
            "least this many tiles from the map border when enough candidates exist."
        ),
    )
    parser.add_argument(
        "--hub-center-window-size",
        type=int,
        default=0,
        help=(
            "When positive with --random-hub, restrict colony sampling to this "
            "centered square window. For example, 4 on an 80x80 map samples x/y 38..41."
        ),
    )
    parser.add_argument(
        "--maze-obstacles",
        action="store_true",
        help="Generate fixed maze walls and block ant movement through them.",
    )
    parser.add_argument(
        "--maze-corridor-width",
        type=int,
        default=3,
        help="Open corridor thickness, in cells, for --maze-obstacles.",
    )
    parser.add_argument(
        "--maze-wall-width",
        type=int,
        default=1,
        help="Wall thickness, in cells, for --maze-obstacles.",
    )
    parser.add_argument(
        "--maze-seed",
        type=int,
        default=0,
        help="Seed for the generated maze layout.",
    )
    parser.add_argument(
        "--maze-layout-count",
        type=int,
        default=64,
        help="Number of obstacle layouts to pre-generate when obstacle sampling is enabled.",
    )
    parser.add_argument(
        "--random-wall-obstacles",
        action="store_true",
        help="Generate random straight and L-shaped wall segments as obstacles.",
    )
    parser.add_argument("--random-wall-count-min", type=int, default=1)
    parser.add_argument("--random-wall-count-max", type=int, default=3)
    parser.add_argument("--random-wall-length-min", type=int, default=4)
    parser.add_argument("--random-wall-length-max", type=int, default=14)
    parser.add_argument("--random-wall-width", type=int, default=1)
    parser.add_argument("--random-wall-l-turn-probability", type=float, default=0.5)
    parser.add_argument(
        "--random-wall-center-window-size",
        type=int,
        default=0,
        help=(
            "When positive with --random-wall-obstacles, prefer random wall starts "
            "inside this centered square window."
        ),
    )
    parser.add_argument("--lethal-food-min-distance", type=int, default=1)
    parser.add_argument(
        "--lethal-food-max-distance",
        type=int,
        default=0,
        help=(
            "When positive, prefer lethal food positions within this Chebyshev "
            "distance from the hub. Zero keeps full-map lethal sampling."
        ),
    )
    parser.add_argument("--pickup-bonus", type=float, default=0.25)
    parser.add_argument("--death-penalty", type=float, default=0.0)
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
        "--distance-progress-normalizer",
        choices=DISTANCE_PROGRESS_NORMALIZERS,
        default="map",
        help="Scale progress shaping by full-map span or current distance stage.",
    )
    parser.add_argument(
        "--carrying-hub-distance-bonus",
        type=float,
        default=0.0,
        help=(
            "Trainer-side normalized Manhattan progress bonus only while an ant remains "
            "carrying food and moves toward the hub. Actor observations are unchanged."
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
        "--view-reward-scale",
        type=float,
        default=0.0,
        help=(
            "Trainer-side bonus for newly viewed cells from the actor vision window. "
            "The bonus is multiplied by (1 - viewed_fraction) ** view_reward_decay."
        ),
    )
    parser.add_argument(
        "--view-reward-decay",
        type=float,
        default=1.0,
        help="Decay exponent for --view-reward-scale as more of the map is viewed.",
    )
    parser.add_argument(
        "--border-view-penalty",
        type=float,
        default=0.0,
        help="Trainer-side penalty per visible border cell in the actor vision window.",
    )
    parser.add_argument(
        "--border-moat-width",
        type=int,
        default=0,
        help="Outer-border moat width in cells for a soft near-border trainer penalty.",
    )
    parser.add_argument(
        "--border-moat-penalty",
        type=float,
        default=0.0,
        help="Trainer-side penalty scale for distance inside --border-moat-width.",
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
        "--reset-optimizer-on-load",
        action="store_true",
        help="Discard saved Adam state when loading a checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Write resolved config, metrics, summary, and default checkpoint under this run directory.",
    )
    parser.add_argument(
        "--layout-audit-dir",
        type=Path,
        default=None,
        help=(
            "Temporary debug folder for randomized training-layout JSONL records "
            "and optional PNG snapshots."
        ),
    )
    parser.add_argument(
        "--layout-audit-snapshot-interval",
        type=int,
        default=0,
        help=(
            "When --layout-audit-dir is set, write one map PNG after every N "
            "recorded layouts. 0 disables PNG snapshots."
        ),
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

    return _validate_args(parser.parse_args(argv))


def _validate_args(args: argparse.Namespace) -> argparse.Namespace:
    rollout_batch_size = args.num_envs * args.num_steps
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.jax_device_count < 0:
        raise ValueError("--jax-device-count must be non-negative.")
    if args.jax_parallelism == "data" and args.jax_device_count > 0:
        if args.num_envs % args.jax_device_count != 0:
            raise ValueError("--num-envs must be divisible by --jax-device-count.")
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
    if args.training_rollout_temperature <= 0.0:
        raise ValueError("--training-rollout-temperature must be positive.")
    if args.max_grad_norm < 0.0:
        raise ValueError("--max-grad-norm must be non-negative.")
    if args.actor_max_grad_norm is not None and args.actor_max_grad_norm < 0.0:
        raise ValueError("--actor-max-grad-norm must be non-negative.")
    if args.critic_max_grad_norm is not None and args.critic_max_grad_norm < 0.0:
        raise ValueError("--critic-max-grad-norm must be non-negative.")
    if args.layout_audit_snapshot_interval < 0:
        raise ValueError("--layout-audit-snapshot-interval must be non-negative.")
    if args.layout_audit_snapshot_interval > 0 and args.layout_audit_dir is None:
        raise ValueError(
            "--layout-audit-snapshot-interval requires --layout-audit-dir."
        )
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
    if args.food_cluster_count < 0:
        raise ValueError("--food-cluster-count must be non-negative.")
    if args.food_cluster_radius < 0:
        raise ValueError("--food-cluster-radius must be non-negative.")
    if args.food_cluster_count > 0:
        if not args.random_food:
            raise ValueError("--food-cluster-count requires --random-food.")
        if args.food_cluster_count > args.food_sources:
            raise ValueError("--food-cluster-count must be no larger than --food-sources.")
        max_cluster_positions = args.food_cluster_count * (
            2 * args.food_cluster_radius + 1
        ) ** 2
        if args.food_sources > max_cluster_positions:
            raise ValueError(
                "--food-sources must fit inside the requested food cluster footprint."
            )
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
    if args.distance_autocurriculum:
        if args.autocurriculum:
            raise ValueError("--distance-autocurriculum cannot be combined with --autocurriculum.")
        if args.distance_autocurriculum_start_distance <= 0:
            raise ValueError("--distance-autocurriculum-start-distance must be positive.")
        if (
            args.distance_autocurriculum_max_distance
            < args.distance_autocurriculum_start_distance
        ):
            raise ValueError(
                "--distance-autocurriculum-max-distance must be at least the start distance."
            )
        if args.distance_autocurriculum_multiplier <= 1:
            raise ValueError("--distance-autocurriculum-multiplier must be greater than 1.")
        if args.distance_autocurriculum_success_cookies < 0:
            raise ValueError("--distance-autocurriculum-success-cookies must be non-negative.")
        if args.distance_autocurriculum_success_cookies > args.food_count:
            raise ValueError(
                "--distance-autocurriculum-success-cookies cannot exceed --food-count."
            )
        if args.food_sources != 1:
            raise ValueError("--distance-autocurriculum currently uses exactly one food source.")
    if args.obs_width is not None and args.obs_width < args.width:
        raise ValueError("--obs-width must be at least --width.")
    if args.obs_height is not None and args.obs_height < args.height:
        raise ValueError("--obs-height must be at least --height.")
    if args.actor_vision_radius < 0:
        raise ValueError("--actor-vision-radius must be non-negative.")
    if args.agent_identity_types is not None and args.agent_identity_types <= 0:
        raise ValueError("--agent-identity-types must be positive.")
    if args.lethal_food_count < 0:
        raise ValueError("--lethal-food-count must be non-negative.")
    if args.lethal_food_sources < 0:
        raise ValueError("--lethal-food-sources must be non-negative.")
    if args.lethal_food_count > 0 and args.lethal_food_sources <= 0:
        raise ValueError("--lethal-food-sources must be positive when lethal food is used.")
    if args.lethal_food_count > 0 and (args.autocurriculum or args.distance_autocurriculum):
        raise ValueError("--lethal-food-count is only supported for non-curriculum JAX runs.")
    if args.lethal_food_count > 0 and not args.random_food:
        raise ValueError("--lethal-food-count requires --random-food for generated layouts.")
    if args.lethal_food_min_distance < 1:
        raise ValueError("--lethal-food-min-distance must be at least 1.")
    if args.lethal_food_max_distance < 0:
        raise ValueError("--lethal-food-max-distance must be non-negative.")
    if (
        args.lethal_food_max_distance > 0
        and args.lethal_food_max_distance < args.lethal_food_min_distance
    ):
        raise ValueError(
            "--lethal-food-max-distance must be at least --lethal-food-min-distance."
        )
    if args.random_food_same_distance and not args.random_food:
        raise ValueError("--random-food-same-distance requires --random-food.")
    if args.random_ant_spawn_radius is not None and args.random_ant_spawn_radius < 0:
        raise ValueError("--random-ant-spawn-radius must be non-negative.")
    if args.layout_margin < 0:
        raise ValueError("--layout-margin must be non-negative.")
    if args.layout_margin * 2 >= min(args.width, args.height):
        raise ValueError("--layout-margin must leave at least one interior cell.")
    if args.hub_center_window_size < 0:
        raise ValueError("--hub-center-window-size must be non-negative.")
    if args.hub_center_window_size > min(args.width, args.height):
        raise ValueError("--hub-center-window-size must fit inside the map.")
    if args.maze_corridor_width <= 0:
        raise ValueError("--maze-corridor-width must be positive.")
    if args.maze_wall_width <= 0:
        raise ValueError("--maze-wall-width must be positive.")
    if args.maze_layout_count <= 0:
        raise ValueError("--maze-layout-count must be positive.")
    if args.maze_obstacles and args.random_wall_obstacles:
        raise ValueError("--maze-obstacles and --random-wall-obstacles are mutually exclusive.")
    if args.random_wall_count_min <= 0:
        raise ValueError("--random-wall-count-min must be positive.")
    if args.random_wall_count_max < args.random_wall_count_min:
        raise ValueError("--random-wall-count-max must be at least --random-wall-count-min.")
    if args.random_wall_length_min <= 0:
        raise ValueError("--random-wall-length-min must be positive.")
    if args.random_wall_length_max < args.random_wall_length_min:
        raise ValueError("--random-wall-length-max must be at least --random-wall-length-min.")
    if args.random_wall_width <= 0:
        raise ValueError("--random-wall-width must be positive.")
    if args.random_wall_obstacles and args.random_wall_width >= min(args.width, args.height):
        raise ValueError("--random-wall-width must be smaller than both grid axes.")
    if not 0.0 <= args.random_wall_l_turn_probability <= 1.0:
        raise ValueError("--random-wall-l-turn-probability must be between 0 and 1.")
    if args.random_wall_center_window_size < 0:
        raise ValueError("--random-wall-center-window-size must be non-negative.")
    if args.random_wall_center_window_size > min(args.width, args.height):
        raise ValueError("--random-wall-center-window-size must fit inside the map.")
    if args.write_bits <= 0 or args.write_bits > MAX_WRITE_BITS:
        raise ValueError(f"--write-bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    if (args.autocurriculum or args.distance_autocurriculum) and not args.food_termination:
        raise ValueError("--no-food-termination is only supported for non-curriculum runs.")
    if (args.autocurriculum or args.distance_autocurriculum) and args.terminate_on_full_coverage:
        raise ValueError(
            "--terminate-on-full-coverage is only supported for non-curriculum runs."
        )
    if args.autocurriculum and (args.maze_obstacles or args.random_wall_obstacles):
        raise ValueError("Obstacle generation is only supported for non-autocurriculum runs.")
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
    if args.carrying_hub_distance_bonus < 0.0:
        raise ValueError("--carrying-hub-distance-bonus must be non-negative.")
    if args.visit_reward_scale < 0.0:
        raise ValueError("--visit-reward-scale must be non-negative.")
    if args.visit_reward_decay < 0.0:
        raise ValueError("--visit-reward-decay must be non-negative.")
    if args.view_reward_scale < 0.0:
        raise ValueError("--view-reward-scale must be non-negative.")
    if args.view_reward_decay < 0.0:
        raise ValueError("--view-reward-decay must be non-negative.")
    if args.border_view_penalty < 0.0:
        raise ValueError("--border-view-penalty must be non-negative.")
    if args.border_moat_width < 0:
        raise ValueError("--border-moat-width must be non-negative.")
    if args.border_moat_penalty < 0.0:
        raise ValueError("--border-moat-penalty must be non-negative.")
    if args.completion_bonus < 0.0:
        raise ValueError("--completion-bonus must be non-negative.")
    if args.death_penalty < 0.0:
        raise ValueError("--death-penalty must be non-negative.")
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
