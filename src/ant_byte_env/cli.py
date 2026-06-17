"""Console entry point for AntByte research workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ant_byte_env.experiments import (
    load_experiment_config,
    namespace_to_jsonable,
    resolve_training_argv,
)
from ant_byte_env.rendering import render_checkpoint
from ant_byte_env.results import index_result_metadata
from ant_byte_env.runs import prepare_run_dir


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.command == "train":
        return _run_train(args, unknown)
    if args.command == "render":
        render_checkpoint(
            args.checkpoint,
            args.output,
            backend=args.backend,
            show_vision=not args.no_vision,
            reuse_existing=args.reuse_existing,
            max_frames=args.max_frames,
            tile_size=args.tile_size,
            policy_temperature=args.policy_temperature,
        )
        print(f"render saved to {args.output}")
        return 0
    if args.command == "results" and args.results_command == "index":
        payload = index_result_metadata(args.source, args.output)
        print(json.dumps({"output": str(args.output), "entry_count": payload["entry_count"]}))
        return 0
    if args.command == "probe" and args.probe_command == "communication":
        from ant_byte_env.training.jax_mappo.probe import probe_communication_checkpoint

        payload = probe_communication_checkpoint(
            args.checkpoint,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed_offset=args.seed_offset,
            render_rollouts=not args.no_render,
            max_render_frames=args.max_render_frames,
            tile_size=args.tile_size,
        )
        print(
            json.dumps(
                {
                    "output": payload["probe_path"],
                    "sampled_write_bit_entropy": payload["sampled"]["write_bit_entropy"],
                    "deterministic_write_bit_entropy": payload["deterministic"][
                        "write_bit_entropy"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "autoresearch" and args.autoresearch_command == "communication-plan":
        from ant_byte_env.autoresearch import build_communication_sweep_plan

        payload = build_communication_sweep_plan(
            matrix_path=args.matrix,
            phase=args.phase,
            run_id=args.run_id,
            run_root=args.run_root,
            bit_stages=args.bit_stages,
            global_update_cap=args.global_update_cap,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            probe_episodes=args.probe_episodes,
            render_rollouts=args.render_rollouts,
            max_render_frames=args.max_render_frames,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "autoresearch" and args.autoresearch_command == "communication-run":
        from ant_byte_env.autoresearch import (
            AutoresearchResourceError,
            build_communication_sweep_plan,
            execute_communication_sweep_plan,
        )

        plan = build_communication_sweep_plan(
            matrix_path=args.matrix,
            phase=args.phase,
            run_id=args.run_id,
            run_root=args.run_root,
            bit_stages=args.bit_stages,
            global_update_cap=args.global_update_cap,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            probe_episodes=args.probe_episodes,
            render_rollouts=args.render_rollouts,
            max_render_frames=args.max_render_frames,
        )
        try:
            payload = execute_communication_sweep_plan(
                plan,
                check_resources=not args.skip_resource_check,
                resume_completed=not args.rerun_completed,
            )
        except AutoresearchResourceError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "autoresearch" and args.autoresearch_command == "communication-rank":
        from ant_byte_env.autoresearch import rank_communication_gate_probes

        payload = rank_communication_gate_probes(
            matrix_path=args.matrix,
            phase=args.phase,
            run_ids=args.ids,
            probe_filename=args.probe_filename,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ant-byte")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run or validate a training config.")
    train.add_argument("backend", choices=["torch", "jax"])
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--run-root", type=Path, default=Path("runs"))

    render = subparsers.add_parser("render", help="Render a checkpoint rollout.")
    render.add_argument("--checkpoint", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--backend", choices=["torch", "jax"], default=None)
    render.add_argument(
        "--no-vision",
        action="store_true",
        help="Render without ant vision overlays.",
    )
    render.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip rendering when the output exists and is newer than the checkpoint.",
    )
    render.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the total rendered frame count, including the initial reset frame.",
    )
    render.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Override the render tile size in pixels.",
    )
    render.add_argument(
        "--policy-temperature",
        type=float,
        default=0.0,
        help="Use 0.0 for greedy rendering; any positive value samples from the policy.",
    )

    results = subparsers.add_parser("results", help="Manage curated result metadata.")
    result_subparsers = results.add_subparsers(dest="results_command", required=True)
    index = result_subparsers.add_parser("index", help="Index run metadata.")
    index.add_argument("source", type=Path)
    index.add_argument("output", type=Path)

    probe = subparsers.add_parser("probe", help="Run offline checkpoint probes.")
    probe_subparsers = probe.add_subparsers(dest="probe_command", required=True)
    communication = probe_subparsers.add_parser(
        "communication",
        help="Probe JAX communication-bit checkpoint behavior.",
    )
    communication.add_argument("--checkpoint", type=Path, required=True)
    communication.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/autoresearch/communication_bits"),
    )
    communication.add_argument("--num-episodes", type=int, default=4)
    communication.add_argument("--seed-offset", type=int, default=2_000_000)
    communication.add_argument("--no-render", action="store_true")
    communication.add_argument("--max-render-frames", type=int, default=None)
    communication.add_argument("--tile-size", type=int, default=16)

    autoresearch = subparsers.add_parser(
        "autoresearch",
        help="Prepare autoresearch experiment commands.",
    )
    autoresearch_subparsers = autoresearch.add_subparsers(
        dest="autoresearch_command",
        required=True,
    )
    communication_plan = autoresearch_subparsers.add_parser(
        "communication-plan",
        help="Print staged train/probe commands for a communication sweep entry.",
    )
    communication_plan.add_argument(
        "--matrix",
        type=Path,
        default=Path("autoresearch/communication_sweep.json"),
    )
    communication_plan.add_argument("--phase", required=True)
    communication_plan.add_argument("--id", dest="run_id", required=True)
    communication_plan.add_argument("--run-root", type=Path, default=None)
    communication_plan.add_argument("--bit-stages", type=int, nargs="+", default=None)
    communication_plan.add_argument("--global-update-cap", type=int, default=None)
    communication_plan.add_argument("--num-envs", type=int, default=None)
    communication_plan.add_argument("--num-steps", type=int, default=None)
    communication_plan.add_argument("--probe-episodes", type=int, default=1)
    communication_plan.add_argument(
        "--render-rollouts",
        dest="render_rollouts",
        action="store_true",
        default=False,
    )
    communication_plan.add_argument("--no-render", dest="render_rollouts", action="store_false")
    communication_plan.add_argument("--max-render-frames", type=int, default=300)
    communication_run = autoresearch_subparsers.add_parser(
        "communication-run",
        help="Execute staged training and probing for a communication sweep entry.",
    )
    communication_run.add_argument(
        "--matrix",
        type=Path,
        default=Path("autoresearch/communication_sweep.json"),
    )
    communication_run.add_argument("--phase", required=True)
    communication_run.add_argument("--id", dest="run_id", required=True)
    communication_run.add_argument("--run-root", type=Path, default=None)
    communication_run.add_argument("--bit-stages", type=int, nargs="+", default=None)
    communication_run.add_argument("--global-update-cap", type=int, default=None)
    communication_run.add_argument("--num-envs", type=int, default=None)
    communication_run.add_argument("--num-steps", type=int, default=None)
    communication_run.add_argument("--probe-episodes", type=int, default=1)
    communication_run.add_argument(
        "--render-rollouts",
        dest="render_rollouts",
        action="store_true",
        default=False,
    )
    communication_run.add_argument("--no-render", dest="render_rollouts", action="store_false")
    communication_run.add_argument("--max-render-frames", type=int, default=300)
    communication_run.add_argument("--rerun-completed", action="store_true")
    communication_run.add_argument("--skip-resource-check", action="store_true")
    communication_rank = autoresearch_subparsers.add_parser(
        "communication-rank",
        help="Rank completed communication probe artifacts by balanced delivery.",
    )
    communication_rank.add_argument(
        "--matrix",
        type=Path,
        default=Path("autoresearch/communication_sweep.json"),
    )
    communication_rank.add_argument("--phase", required=True)
    communication_rank.add_argument("--ids", nargs="+", default=None)
    communication_rank.add_argument(
        "--probe-filename",
        default="communication_probe.json",
        help="Probe JSON filename inside each matrix probe_output_dir.",
    )
    return parser


def _run_train(args: argparse.Namespace, overrides: list[str]) -> int:
    spec = load_experiment_config(args.config)
    if spec.backend != args.backend:
        raise ValueError(
            f"config backend {spec.backend!r} does not match selected backend {args.backend!r}."
        )

    training_argv = resolve_training_argv(args.config, overrides)
    parse_args = _backend_parse_args(args.backend)
    if not args.dry_run and "--run-dir" not in training_argv:
        run_dir = prepare_run_dir(args.run_root, spec.name)
        training_argv.extend(["--run-dir", str(run_dir)])

    parsed = parse_args(training_argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "config": str(args.config),
                    "experiment": spec.name,
                    "argv": training_argv,
                    "resolved_args": namespace_to_jsonable(parsed),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    runner = _backend_runner(args.backend)
    metrics = runner(training_argv)
    print(json.dumps(_jsonable_metrics(metrics), sort_keys=True))
    return 0


def _backend_parse_args(backend: str) -> Any:
    if backend == "torch":
        from ant_byte_env.training.torch_mappo.cli import parse_args

        return parse_args
    if backend == "jax":
        from ant_byte_env.training.jax_mappo.cli import parse_args

        return parse_args
    raise ValueError("backend must be 'torch' or 'jax'.")


def _backend_runner(backend: str) -> Any:
    if backend == "torch":
        from ant_byte_env.training.torch_mappo.runner import main

        return main
    if backend == "jax":
        from ant_byte_env.training.jax_mappo.runner import main

        return main
    raise ValueError("backend must be 'torch' or 'jax'.")


def _jsonable_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


if __name__ == "__main__":
    raise SystemExit(main())
