"""Console entry point for AntByte research workflows."""

from __future__ import annotations

import argparse
import json
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
        )
        print(f"render saved to {args.output}")
        return 0
    if args.command == "results" and args.results_command == "index":
        payload = index_result_metadata(args.source, args.output)
        print(json.dumps({"output": str(args.output), "entry_count": payload["entry_count"]}))
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
    render.add_argument("--no-vision", action="store_true", help="Render without ant vision overlays.")

    results = subparsers.add_parser("results", help="Manage curated result metadata.")
    result_subparsers = results.add_subparsers(dest="results_command", required=True)
    index = result_subparsers.add_parser("index", help="Index run metadata.")
    index.add_argument("source", type=Path)
    index.add_argument("output", type=Path)
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
