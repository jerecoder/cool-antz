"""Gated map-and-ant curriculum workflow for historical JAX MAPPO runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ant_byte_env.experiments import config_args_to_argv, namespace_to_jsonable
from ant_byte_env.runs import ensure_run_structure, write_json

DEFAULT_STAGE_PLAN = "4:1,6:1,8:2,10:2,12:3,16:4,20:5,25:6,32:8,40:10,50:10"
DEFAULT_EXP_NAME = "jax_mappo_map_ant_gated_mlp"
GATE_FAILURE_EXIT_CODE = 2

TrainMain = Callable[..., dict[str, float]]
EvaluateModes = Callable[[Path], dict[str, Any]]


def add_map_ant_curriculum_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, default=Path("runs/map_ant_gated_mlp"))
    parser.add_argument("--stage-plan", default=DEFAULT_STAGE_PLAN)
    parser.add_argument("--food-sources-by-stage", default=None)
    parser.add_argument("--food-counts-by-stage", default=None)
    parser.add_argument("--final-food-count", type=int, default=48)
    parser.add_argument("--final-food-sources", type=int, default=2)
    parser.add_argument("--start-stage", default=None)
    parser.add_argument("--start-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-id", default=None)

    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--training-rollout-temperature", type=float, default=1.0)
    parser.add_argument("--deterministic-rollout", action="store_true")
    parser.add_argument("--deterministic-rollout-fraction", type=float, default=0.0)
    parser.add_argument("--deterministic-move-rollout-fraction", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--critic-architecture", choices=("mlp",), default="mlp")
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    parser.set_defaults(norm_adv=True)
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--verbose", dest="quiet", action="store_false")

    parser.add_argument("--actor-vision-radius", type=int, default=1)
    parser.add_argument("--obs-width", type=int, default=None)
    parser.add_argument("--obs-height", type=int, default=None)
    parser.add_argument("--write-bits", type=int, default=1)
    parser.add_argument("--write-while-moving", action="store_true", default=True)
    parser.add_argument(
        "--separate-write-action",
        dest="write_while_moving",
        action="store_false",
    )
    parser.add_argument("--write-head-transfer", default="repeat")
    parser.add_argument("--pickup-bonus", type=float, default=0.10)
    parser.add_argument("--completion-bonus", type=float, default=0.0)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--write-penalty", type=float, default=0.0)
    parser.add_argument("--write-bit-penalty", type=float, default=0.0)
    parser.add_argument("--write-bit-penalty-decay", type=float, default=0.5)
    parser.add_argument("--write-overwrite-penalty", type=float, default=0.0)
    parser.add_argument("--write-entropy-bonus", type=float, default=0.0)
    parser.add_argument("--write-entropy-bonus-cap", type=float, default=0.15)
    parser.add_argument("--write-bit-entropy-bonus", type=float, default=0.0)
    parser.add_argument("--visible-food-approach-bonus", type=float, default=0.0)
    parser.add_argument("--visible-food-stall-penalty", type=float, default=0.0)
    parser.add_argument("--carrying-hub-approach-bonus", type=float, default=0.0)
    parser.add_argument("--carrying-hub-stall-penalty", type=float, default=0.0)
    parser.add_argument("--random-food", action="store_true", default=True)
    parser.add_argument("--fixed-food", dest="random_food", action="store_false")
    parser.add_argument("--random-hub", action="store_true", default=True)
    parser.add_argument("--fixed-hub", dest="random_hub", action="store_false")
    parser.add_argument("--reset-opt-state-on-load", action="store_true")

    parser.add_argument("--gate-update-chunk-cap", type=int, default=500)
    parser.add_argument("--gate-max-stage-attempts", type=int, default=6)
    parser.add_argument("--gate-eval-num-episodes", type=int, default=16)
    parser.add_argument("--gate-min-delivered-fraction", type=float, default=0.95)
    parser.add_argument("--gate-min-success-rate", type=float, default=0.75)
    parser.add_argument("--gate-min-pickup-to-delivery", type=float, default=0.90)
    parser.add_argument("--gate-length-fraction", type=float, default=0.60)
    parser.add_argument("--gate-min-applied-write-rate", type=float, default=None)
    parser.add_argument("--gate-max-applied-write-rate", type=float, default=None)
    parser.add_argument("--gate-eval-modes", default="deterministic,sampled")

    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project-name", default="cool-antz")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a gated JAX MAPPO map-and-ant forage curriculum."
    )
    add_map_ant_curriculum_args(parser)
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    stages = parse_stage_plan(args.stage_plan)
    _parse_optional_int_list(args.food_counts_by_stage, expected_len=len(stages))
    _parse_optional_int_list(args.food_sources_by_stage, expected_len=len(stages))
    if args.start_checkpoint is not None and args.start_stage is None:
        raise ValueError("--start-checkpoint requires --start-stage.")
    if args.resume and (args.start_stage is not None or args.start_checkpoint is not None):
        raise ValueError("--resume cannot be combined with --start-stage or checkpoint.")
    if args.final_food_count <= 0 or args.final_food_sources <= 0:
        raise ValueError("final food counts and sources must be positive.")
    if args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("--num-envs and --num-steps must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")
    if args.training_rollout_temperature <= 0.0:
        raise ValueError("--training-rollout-temperature must be positive.")
    if not 0.0 <= args.deterministic_rollout_fraction <= 1.0:
        raise ValueError("--deterministic-rollout-fraction must be between 0 and 1.")
    if not 0.0 <= args.deterministic_move_rollout_fraction <= 1.0:
        raise ValueError("--deterministic-move-rollout-fraction must be between 0 and 1.")
    max_stage_width = max(int(stage["width"]) for stage in stages)
    max_stage_height = max(int(stage["height"]) for stage in stages)
    if args.obs_width is not None and args.obs_width < max_stage_width:
        raise ValueError("--obs-width must be at least the maximum stage width.")
    if args.obs_height is not None and args.obs_height < max_stage_height:
        raise ValueError("--obs-height must be at least the maximum stage height.")
    for name in (
        "completion_bonus",
        "visible_food_approach_bonus",
        "visible_food_stall_penalty",
        "carrying_hub_approach_bonus",
        "carrying_hub_stall_penalty",
        "write_overwrite_penalty",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.gate_update_chunk_cap <= 0 or args.gate_max_stage_attempts <= 0:
        raise ValueError("gate update caps and attempts must be positive.")
    if args.gate_eval_num_episodes <= 0:
        raise ValueError("--gate-eval-num-episodes must be positive.")
    if args.gate_min_applied_write_rate is not None and args.gate_min_applied_write_rate < 0.0:
        raise ValueError("--gate-min-applied-write-rate must be non-negative.")
    if args.gate_max_applied_write_rate is not None and args.gate_max_applied_write_rate < 0.0:
        raise ValueError("--gate-max-applied-write-rate must be non-negative.")
    if (
        args.gate_min_applied_write_rate is not None
        and args.gate_max_applied_write_rate is not None
        and args.gate_min_applied_write_rate > args.gate_max_applied_write_rate
    ):
        raise ValueError(
            "--gate-min-applied-write-rate cannot exceed --gate-max-applied-write-rate."
        )
    parse_gate_eval_modes(args.gate_eval_modes)


def parse_stage_plan(value: str) -> list[dict[str, int | str]]:
    stages: list[dict[str, int | str]] = []
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("stage plan entries must be formatted as SIZE:ANTS.")
        raw_size, raw_ants = item.split(":", maxsplit=1)
        try:
            size = int(raw_size)
            num_ants = int(raw_ants)
        except ValueError as exc:
            raise ValueError("stage plan sizes and ant counts must be integers.") from exc
        if size <= 1 or num_ants <= 0:
            raise ValueError("stage sizes must be >1 and ant counts must be positive.")
        stages.append(
            {
                "name": f"{size}x{size}_{num_ants}_ants",
                "width": size,
                "height": size,
                "num_ants": num_ants,
            }
        )
    if not stages:
        raise ValueError("stage plan must contain at least one stage.")
    return stages


def parse_gate_eval_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not modes:
        raise ValueError("--gate-eval-modes must contain at least one mode.")
    valid_modes = {"deterministic", "sampled"}
    invalid = sorted(set(modes) - valid_modes)
    if invalid:
        choices = ", ".join(sorted(valid_modes))
        raise ValueError(
            f"unknown gate eval mode(s): {', '.join(invalid)}; choices: {choices}."
        )
    if len(set(modes)) != len(modes):
        raise ValueError("--gate-eval-modes must not contain duplicates.")
    return modes


def build_curriculum_stages(args: argparse.Namespace) -> list[dict[str, int | str]]:
    stages = parse_stage_plan(args.stage_plan)
    max_size = max(int(stage["width"]) for stage in stages)
    food_counts = _parse_optional_int_list(args.food_counts_by_stage, expected_len=len(stages))
    food_sources = _parse_optional_int_list(args.food_sources_by_stage, expected_len=len(stages))
    built: list[dict[str, int | str]] = []
    for index, stage in enumerate(stages):
        size = int(stage["width"])
        food_count = (
            int(food_counts[index])
            if food_counts is not None
            else curriculum_food_count(
                size,
                final_size=max_size,
                final_food_count=int(args.final_food_count),
            )
        )
        source_count = (
            int(food_sources[index])
            if food_sources is not None
            else concentrated_food_sources(
                food_count,
                final_food_count=int(args.final_food_count),
                final_food_sources=int(args.final_food_sources),
            )
        )
        if source_count > food_count:
            raise ValueError("food source counts cannot exceed food counts.")
        built.append(
            {
                **stage,
                "food_count": food_count,
                "food_sources": source_count,
                "cookie_distance": min(1 + (size - 4) // 2, size // 2),
                "max_steps": max(48, 4 * size * size),
            }
        )
    return built


def curriculum_food_count(size: int, *, final_size: int, final_food_count: int) -> int:
    baseline = 2 + max(0, int(size) - 4)
    final_baseline = 2 + max(0, int(final_size) - 4)
    if int(size) == int(final_size):
        return int(final_food_count)
    scaled = round(int(final_food_count) * baseline / final_baseline)
    return max(1, int(scaled))


def concentrated_food_sources(
    food_count: int,
    *,
    final_food_count: int,
    final_food_sources: int,
) -> int:
    scaled = round(int(final_food_sources) * int(food_count) / int(final_food_count))
    return max(1, min(int(food_count), int(scaled)))


def execute_curriculum(
    args: argparse.Namespace,
    *,
    train_main: TrainMain | None = None,
    evaluate_modes: Callable[..., dict[str, Any]] | None = None,
    progress_factory: Callable[[str, int], Any] | None = None,
) -> dict[str, Any]:
    validate_args(args)
    if train_main is None:
        from ant_byte_env.training.jax_mappo.runner import main as train_main
    if evaluate_modes is None:
        evaluate_modes = evaluate_checkpoint_modes
    if progress_factory is None:
        from ant_byte_env.workflows.progress import stage_update_progress

        progress_factory = stage_update_progress

    run_dir = Path(args.run_dir)
    ensure_run_structure(run_dir)
    stages = build_curriculum_stages(args)
    write_json(run_dir / "curriculum_config.json", curriculum_config_payload(args, stages))
    state = load_or_create_state(run_dir, stages)
    start_index, source_checkpoint = resolve_start(args, stages, state)
    return run_stage_sequence(
        args=args,
        stages=stages,
        state=state,
        start_index=start_index,
        source_checkpoint=source_checkpoint,
        train_main=train_main,
        evaluate_modes=evaluate_modes,
        progress_factory=progress_factory,
    )


def run_stage_sequence(
    *,
    args: argparse.Namespace,
    stages: Sequence[Mapping[str, Any]],
    state: dict[str, Any],
    start_index: int,
    source_checkpoint: Path | None,
    train_main: TrainMain,
    evaluate_modes: Callable[..., dict[str, Any]],
    progress_factory: Callable[[str, int], Any],
) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    previous_checkpoint = source_checkpoint
    final_train_metrics: dict[str, float] = {}
    status = "passed"

    for stage_index in range(start_index, len(stages)):
        stage = dict(stages[stage_index])
        stage_state = ensure_stage_state(state, stage)
        stage_state["status"] = "running"
        state["status"] = "running"
        state["current_stage"] = stage["name"]
        state["current_stage_index"] = stage_index
        write_json(run_dir / "curriculum_state.json", state)

        stage_passed = False
        for local_attempt_index in range(int(args.gate_max_stage_attempts)):
            attempt_index = next_attempt_index(run_dir, stage, stage_state)
            checkpoint_path = attempt_checkpoint_path(run_dir, stage, attempt_index)
            attempt_run_dir = run_dir / "stage_runs" / str(stage["name"]) / f"attempt_{attempt_index:03d}"
            load_checkpoint = previous_checkpoint
            if local_attempt_index > 0 and stage_state.get("latest_checkpoint_path"):
                latest = Path(str(stage_state["latest_checkpoint_path"]))
                if latest.exists():
                    load_checkpoint = latest

            progress = progress_factory(
                f"{stage['name']} gate {attempt_index}",
                int(args.gate_update_chunk_cap),
            )
            stage_metrics: list[dict[str, Any]] = []

            def record_progress(
                update_index: int,
                total_updates: int,
                metrics: dict[str, float],
            ) -> None:
                del total_updates
                progress.update(1)
                progress.set_postfix(
                    loss=f"{metrics['loss']:.3f}",
                    ret=f"{metrics['episode_return']:.3f}",
                )
                stage_metrics.append(
                    {
                        **stage,
                        **metrics,
                        "attempt": attempt_index,
                        "stage_update": update_index,
                        "stage_cumulative_update": update_index,
                        "update_chunk_cap": int(args.gate_update_chunk_cap),
                        "checkpoint": str(checkpoint_path),
                        "source_checkpoint": str(load_checkpoint) if load_checkpoint else None,
                        "candidate_id": args.candidate_id,
                    }
                )

            train_argv = build_stage_train_argv(
                args=args,
                stages=stages,
                stage=stage,
                checkpoint_path=checkpoint_path,
                load_checkpoint=load_checkpoint,
                attempt_run_dir=attempt_run_dir,
            )
            try:
                final_train_metrics = train_main(
                    train_argv,
                    progress_callback=record_progress,
                )
            finally:
                progress.close()

            evaluation = evaluate_modes(
                checkpoint_path,
                num_episodes=int(args.gate_eval_num_episodes),
            )
            gate = evaluate_gate(args, stage=stage, evaluation=evaluation)
            gate_score = score_gate(gate)
            attempt_record = {
                **stage,
                "attempt": attempt_index,
                "candidate_id": args.candidate_id,
                "checkpoint": str(checkpoint_path),
                "source_checkpoint": str(load_checkpoint) if load_checkpoint else None,
                "run_dir": str(attempt_run_dir),
                "train_argv": train_argv,
                "train_metrics": final_train_metrics,
                "stage_metrics": stage_metrics,
                "gate_score": gate_score,
                "gate": gate,
                "deterministic": evaluation["deterministic"]["metrics"],
                "sampled": evaluation["sampled"]["metrics"],
            }
            append_jsonl(run_dir / "gate_history.jsonl", attempt_record)
            update_stage_state(stage_state, attempt_record)
            state["final_checkpoint_path"] = str(checkpoint_path)
            state["final_train_metrics"] = final_train_metrics

            if gate["passed"]:
                stage_state["status"] = "passed"
                stage_state["passed_checkpoint_path"] = str(checkpoint_path)
                previous_checkpoint = checkpoint_path
                stage_passed = True
                write_json(run_dir / "curriculum_state.json", state)
                break

            stage_state["status"] = "failed"
            write_json(run_dir / "curriculum_state.json", state)

        if not stage_passed:
            status = "failed"
            state["status"] = status
            state["current_stage"] = stage["name"]
            state["current_stage_index"] = stage_index
            write_curriculum_artifacts(run_dir, state, stages)
            return curriculum_result_payload(
                run_dir=run_dir,
                state=state,
                stages=stages,
                status=status,
            )

    state["status"] = status
    state["current_stage"] = None
    state["current_stage_index"] = None
    write_curriculum_artifacts(run_dir, state, stages)
    return curriculum_result_payload(
        run_dir=run_dir,
        state=state,
        stages=stages,
        status=status,
    )


def build_stage_train_argv(
    *,
    args: argparse.Namespace,
    stages: Sequence[Mapping[str, Any]],
    stage: Mapping[str, Any],
    checkpoint_path: Path,
    load_checkpoint: Path | None,
    attempt_run_dir: Path,
) -> list[str]:
    max_width = int(args.obs_width) if args.obs_width is not None else max(
        int(item["width"]) for item in stages
    )
    max_height = int(args.obs_height) if args.obs_height is not None else max(
        int(item["height"]) for item in stages
    )
    total_timesteps = int(args.num_envs) * int(args.num_steps) * int(args.gate_update_chunk_cap)
    payload: dict[str, Any] = {
        "exp_name": args.exp_name,
        "seed": args.seed,
        "quiet": args.quiet,
        "track": args.track,
        "wandb_project_name": args.wandb_project_name,
        "wandb_entity": args.wandb_entity,
        "wandb_group": args.wandb_group,
        "wandb_run_name": args.wandb_run_name,
        "wandb_tags": args.wandb_tags or None,
        "wandb_mode": args.wandb_mode,
        "total_timesteps": total_timesteps,
        "learning_rate": args.learning_rate,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "log_interval": args.log_interval,
        "anneal_lr": args.anneal_lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "num_minibatches": args.num_minibatches,
        "update_epochs": args.update_epochs,
        "norm_adv": args.norm_adv,
        "clip_coef": args.clip_coef,
        "ent_coef": args.ent_coef,
        "training_rollout_temperature": args.training_rollout_temperature,
        "deterministic_rollout": args.deterministic_rollout,
        "deterministic_rollout_fraction": args.deterministic_rollout_fraction,
        "deterministic_move_rollout_fraction": args.deterministic_move_rollout_fraction,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "hidden_size": args.hidden_size,
        "critic_architecture": args.critic_architecture,
        "width": stage["width"],
        "height": stage["height"],
        "obs_width": max_width,
        "obs_height": max_height,
        "actor_vision_radius": args.actor_vision_radius,
        "num_ants": stage["num_ants"],
        "food_count": stage["food_count"],
        "food_sources": stage["food_sources"],
        "max_steps": stage["max_steps"],
        "step_penalty": args.step_penalty,
        "completion_bonus": args.completion_bonus,
        "write_penalty": args.write_penalty,
        "write_bit_penalty": args.write_bit_penalty,
        "write_bit_penalty_decay": args.write_bit_penalty_decay,
        "write_overwrite_penalty": args.write_overwrite_penalty,
        "write_entropy_bonus": args.write_entropy_bonus,
        "write_entropy_bonus_cap": args.write_entropy_bonus_cap,
        "write_bit_entropy_bonus": args.write_bit_entropy_bonus,
        "visible_food_approach_bonus": args.visible_food_approach_bonus,
        "visible_food_stall_penalty": args.visible_food_stall_penalty,
        "carrying_hub_approach_bonus": args.carrying_hub_approach_bonus,
        "carrying_hub_stall_penalty": args.carrying_hub_stall_penalty,
        "write_bits": args.write_bits,
        "write_head_transfer": args.write_head_transfer,
        "cookie_distance": stage["cookie_distance"],
        "random_food": args.random_food,
        "random_hub": args.random_hub,
        "pickup_bonus": args.pickup_bonus,
        "save_model": checkpoint_path,
        "run_dir": attempt_run_dir,
        "write_while_moving": args.write_while_moving,
    }
    if load_checkpoint is not None:
        payload["load_model"] = load_checkpoint
        payload["reset_opt_state_on_load"] = args.reset_opt_state_on_load
    return config_args_to_argv(payload)


def evaluate_checkpoint_modes(
    checkpoint_path: Path,
    *,
    num_episodes: int,
) -> dict[str, Any]:
    from ant_byte_env.training.jax_mappo.evaluation import evaluate_checkpoint

    return {
        "deterministic": {
            "metrics": evaluate_checkpoint(
                checkpoint_path,
                num_episodes=num_episodes,
                deterministic=True,
            )
        },
        "sampled": {
            "metrics": evaluate_checkpoint(
                checkpoint_path,
                num_episodes=num_episodes,
                deterministic=False,
                seed_offset=2_000_000,
            )
        },
    }


def evaluate_gate(
    args: argparse.Namespace,
    *,
    stage: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    length_limit = float(stage["max_steps"]) * float(args.gate_length_fraction)
    for mode in parse_gate_eval_modes(args.gate_eval_modes):
        metrics = evaluation[mode]["metrics"]
        _add_min_check(
            checks,
            failures,
            mode=mode,
            metric="eval_mean_delivered_fraction",
            value=float(metrics.get("eval_mean_delivered_fraction", 0.0)),
            threshold=float(args.gate_min_delivered_fraction),
        )
        _add_min_check(
            checks,
            failures,
            mode=mode,
            metric="eval_success_rate",
            value=float(metrics.get("eval_success_rate", 0.0)),
            threshold=float(args.gate_min_success_rate),
        )
        _add_min_check(
            checks,
            failures,
            mode=mode,
            metric="eval_mean_pickup_to_delivery_rate",
            value=float(metrics.get("eval_mean_pickup_to_delivery_rate", 0.0)),
            threshold=float(args.gate_min_pickup_to_delivery),
        )
        _add_max_check(
            checks,
            failures,
            mode=mode,
            metric="eval_mean_episode_length",
            value=float(metrics.get("eval_mean_episode_length", float("inf"))),
            threshold=length_limit,
        )
        if args.gate_min_applied_write_rate is not None:
            _add_min_check(
                checks,
                failures,
                mode=mode,
                metric="eval_mean_applied_nonzero_write_rate",
                value=float(metrics.get("eval_mean_applied_nonzero_write_rate", 0.0)),
                threshold=float(args.gate_min_applied_write_rate),
            )
        if args.gate_max_applied_write_rate is not None:
            _add_max_check(
                checks,
                failures,
                mode=mode,
                metric="eval_mean_applied_nonzero_write_rate",
                value=float(metrics.get("eval_mean_applied_nonzero_write_rate", float("inf"))),
                threshold=float(args.gate_max_applied_write_rate),
            )
    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "modes": list(parse_gate_eval_modes(args.gate_eval_modes)),
    }


def _add_min_check(
    checks: list[dict[str, Any]],
    failures: list[str],
    *,
    mode: str,
    metric: str,
    value: float,
    threshold: float,
) -> None:
    passed = value >= threshold
    checks.append(
        {
            "mode": mode,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "comparison": ">=",
            "passed": passed,
        }
    )
    if not passed:
        failures.append(f"{mode}.{metric}")


def _add_max_check(
    checks: list[dict[str, Any]],
    failures: list[str],
    *,
    mode: str,
    metric: str,
    value: float,
    threshold: float,
) -> None:
    passed = value <= threshold
    checks.append(
        {
            "mode": mode,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "comparison": "<=",
            "passed": passed,
        }
    )
    if not passed:
        failures.append(f"{mode}.{metric}")


def resolve_start(
    args: argparse.Namespace,
    stages: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> tuple[int, Path | None]:
    if args.resume:
        return resolve_resume_start(stages, state)
    if args.start_stage is None:
        return 0, None

    stage_index = stage_index_by_name(stages, args.start_stage)
    if args.start_checkpoint is not None:
        return stage_index, args.start_checkpoint
    if stage_index == 0:
        return stage_index, None
    previous_state = stage_state_by_name(state, str(stages[stage_index - 1]["name"]))
    checkpoint = previous_state.get("passed_checkpoint_path")
    if not checkpoint:
        raise ValueError("--start-stage after the first stage requires --start-checkpoint.")
    return stage_index, Path(str(checkpoint))


def resolve_resume_start(
    stages: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> tuple[int, Path | None]:
    for index, stage in enumerate(stages):
        stage_state = stage_state_by_name(state, str(stage["name"]))
        if stage_state.get("status") != "passed":
            latest = stage_state.get("latest_checkpoint_path")
            if latest and Path(str(latest)).exists():
                return index, Path(str(latest))
            if index > 0:
                previous = stage_state_by_name(state, str(stages[index - 1]["name"]))
                passed = previous.get("passed_checkpoint_path")
                if passed:
                    return index, Path(str(passed))
            return index, None
    return len(stages), (
        Path(str(state["final_checkpoint_path"])) if state.get("final_checkpoint_path") else None
    )


def load_or_create_state(run_dir: Path, stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state_path = run_dir / "curriculum_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "status": "pending",
            "current_stage": None,
            "current_stage_index": None,
            "final_checkpoint_path": None,
            "final_train_metrics": {},
            "stages": {},
        }
    for stage in stages:
        ensure_stage_state(state, stage)
    return state


def ensure_stage_state(state: dict[str, Any], stage: Mapping[str, Any]) -> dict[str, Any]:
    stages_state = state.setdefault("stages", {})
    stage_name = str(stage["name"])
    if stage_name not in stages_state:
        stages_state[stage_name] = {
            "name": stage_name,
            "status": "pending",
            "attempt_count": 0,
            "latest_checkpoint_path": None,
            "best_checkpoint_path": None,
            "best_gate_score": -1.0,
            "passed_checkpoint_path": None,
            "last_failures": [],
        }
    return stages_state[stage_name]


def stage_state_by_name(state: Mapping[str, Any], stage_name: str) -> dict[str, Any]:
    return dict(state.get("stages", {}).get(stage_name, {}))


def update_stage_state(stage_state: dict[str, Any], attempt_record: Mapping[str, Any]) -> None:
    gate = attempt_record["gate"]
    checkpoint = str(attempt_record["checkpoint"])
    gate_score = float(attempt_record["gate_score"])
    stage_state["attempt_count"] = max(
        int(stage_state.get("attempt_count", 0)),
        int(attempt_record["attempt"]),
    )
    stage_state["latest_checkpoint_path"] = checkpoint
    stage_state["last_failures"] = list(gate["failures"])
    if gate_score >= float(stage_state.get("best_gate_score", -1.0)):
        stage_state["best_gate_score"] = gate_score
        stage_state["best_checkpoint_path"] = checkpoint


def next_attempt_index(
    run_dir: Path,
    stage: Mapping[str, Any],
    stage_state: Mapping[str, Any],
) -> int:
    attempt = int(stage_state.get("attempt_count", 0)) + 1
    while attempt_checkpoint_path(run_dir, stage, attempt).exists():
        attempt += 1
    return attempt


def attempt_checkpoint_path(
    run_dir: Path,
    stage: Mapping[str, Any],
    attempt_index: int,
) -> Path:
    return run_dir / "checkpoints" / str(stage["name"]) / f"attempt_{attempt_index:03d}.pkl"


def stage_index_by_name(stages: Sequence[Mapping[str, Any]], stage_name: str) -> int:
    for index, stage in enumerate(stages):
        if str(stage["name"]) == str(stage_name):
            return index
    choices = ", ".join(str(stage["name"]) for stage in stages)
    raise ValueError(f"unknown stage {stage_name!r}; choices: {choices}")


def curriculum_config_payload(
    args: argparse.Namespace,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "args": namespace_to_jsonable(args),
        "stages": [dict(stage) for stage in stages],
        "update_timesteps": int(args.num_envs) * int(args.num_steps),
        "env_steps_per_attempt": (
            int(args.num_envs) * int(args.num_steps) * int(args.gate_update_chunk_cap)
        ),
    }


def dry_run_payload(
    *,
    config_path: Path,
    experiment: str,
    argv: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stages = build_curriculum_stages(args)
    return {
        "backend": "jax",
        "workflow": "map_ant_gated_curriculum",
        "config": str(config_path),
        "experiment": experiment,
        "argv": list(argv),
        "resolved_args": namespace_to_jsonable(args),
        "stages": [dict(stage) for stage in stages],
        "gate_modes": list(parse_gate_eval_modes(args.gate_eval_modes)),
    }


def write_curriculum_artifacts(
    run_dir: Path,
    state: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
) -> None:
    write_json(run_dir / "curriculum_state.json", dict(state))
    write_json(
        run_dir / "curriculum_summary.json",
        curriculum_result_payload(
            run_dir=run_dir,
            state=state,
            stages=stages,
            status=str(state.get("status", "unknown")),
        ),
    )


def curriculum_result_payload(
    *,
    run_dir: Path,
    state: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "run_dir": str(run_dir),
        "config_path": str(run_dir / "curriculum_config.json"),
        "state_path": str(run_dir / "curriculum_state.json"),
        "summary_path": str(run_dir / "curriculum_summary.json"),
        "gate_history_path": str(run_dir / "gate_history.jsonl"),
        "final_checkpoint_path": state.get("final_checkpoint_path"),
        "current_stage": state.get("current_stage"),
        "current_stage_index": state.get("current_stage_index"),
        "stage_count": len(stages),
        "stages": list(state.get("stages", {}).values()),
    }


def score_gate(gate: Mapping[str, Any]) -> float:
    checks = list(gate.get("checks", []))
    if not checks:
        return 0.0
    return sum(1.0 for check in checks if check.get("passed")) / float(len(checks))


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _parse_optional_int_list(value: str | None, *, expected_len: int) -> list[int] | None:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(items) != expected_len:
        raise ValueError(f"expected {expected_len} comma-separated values.")
    try:
        parsed = [int(item) for item in items]
    except ValueError as exc:
        raise ValueError("comma-separated values must be integers.") from exc
    if any(item <= 0 for item in parsed):
        raise ValueError("comma-separated values must be positive.")
    return parsed


def run_cli_from_args(args: argparse.Namespace) -> int:
    validate_args(args)
    result = execute_curriculum(args)
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else GATE_FAILURE_EXIT_CODE


__all__ = [
    "DEFAULT_STAGE_PLAN",
    "GATE_FAILURE_EXIT_CODE",
    "add_map_ant_curriculum_args",
    "attempt_checkpoint_path",
    "build_curriculum_stages",
    "build_stage_train_argv",
    "concentrated_food_sources",
    "curriculum_food_count",
    "dry_run_payload",
    "evaluate_checkpoint_modes",
    "evaluate_gate",
    "execute_curriculum",
    "parse_args",
    "parse_gate_eval_modes",
    "parse_stage_plan",
    "run_cli_from_args",
    "score_gate",
]
