#!/usr/bin/env python3
"""Run the 50x50 actor-only critic ablation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ant_byte_env import write_value_count
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.notebook_workflows import training_dimensions
from ant_byte_env.runs import utc_run_id
from ant_byte_env.training.jax_mappo.core import init_agent_params
from ant_byte_env.training.jax_mappo.transfer import load_actor_from_checkpoint_for_training

CONFIGS = {
    "mlp": PROJECT_ROOT / "experiments" / "critic_ablation_50x50_actor_only_mlp.json",
    "strided_cnn": PROJECT_ROOT
    / "experiments"
    / "critic_ablation_50x50_actor_only_strided_cnn.json",
}
DEFAULT_SOURCE_CHECKPOINT = PROJECT_ROOT / "runs" / "ablation_sources" / "best_efficient_forage.pkl"
DEFAULT_SOURCE_ARTIFACT = (
    "jerefigueiredo-universidad-de-san-andr-s/"
    "cool-antz/critic-ablation-50x50-best-efficient-source:latest"
)


def _selected_configs(selection: str) -> list[Path]:
    if selection == "all":
        return [CONFIGS["mlp"], CONFIGS["strided_cnn"]]
    return [CONFIGS[selection]]


def _ensure_source_checkpoint(path: Path, artifact_name: str | None) -> Path:
    path = path.resolve()
    if path.exists():
        return path
    if artifact_name is None:
        raise FileNotFoundError(f"source checkpoint does not exist: {path}")
    import wandb

    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api().artifact(artifact_name)
    artifact_dir = Path(artifact.download(root=str(path.parent / "_artifact_download")))
    matches = sorted(artifact_dir.rglob("*.pkl"))
    if not matches:
        raise FileNotFoundError(f"artifact {artifact_name!r} did not contain a .pkl file")
    shutil.copy2(matches[0], path)
    return path


def _argv_for_config(
    config_path: Path,
    *,
    source_checkpoint: Path,
    run_root: Path,
    run_id: str,
    wandb_mode: str | None,
    smoke_updates: int | None,
) -> tuple[str, Path, list[str]]:
    spec = load_experiment_config(config_path)
    run_dir = run_root / spec.name / run_id
    argv = config_args_to_argv(spec.args)
    argv.extend(["--load-model", str(source_checkpoint)])
    argv.extend(["--run-dir", str(run_dir)])
    argv.extend(["--save-best-model", str(run_dir / "checkpoints" / "best.pkl")])
    if wandb_mode is not None:
        argv.extend(["--wandb-mode", wandb_mode])
    if smoke_updates is not None:
        update_timesteps = int(spec.args["num_envs"]) * int(spec.args["num_steps"])
        argv.extend(["--total-timesteps", str(update_timesteps * int(smoke_updates))])
        argv.extend(["--log-interval", "1", "--best-eval-interval", "1"])
    return spec.name, run_dir, argv


def _verify_actor_only_load(config_path: Path, argv: list[str]) -> dict[str, Any]:
    import jax

    args, central_obs_dim, actor_obs_dim = training_dimensions(argv)
    target_params = init_agent_params(
        jax.random.PRNGKey(args.seed),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
        critic_architecture=args.critic_architecture,
        critic_num_ants=args.num_ants,
        critic_obs_height=args.obs_height,
        critic_obs_width=args.obs_width,
    )
    loaded = load_actor_from_checkpoint_for_training(
        args.load_model,
        target_params=target_params,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        target_write_bits=args.write_bits,
        actor_vision_radius=args.actor_vision_radius,
        target_num_ants=args.num_ants,
        target_agent_identity_types=getattr(args, "agent_identity_types", None),
        write_head_transfer=args.write_head_transfer,
        target_critic_architecture=args.critic_architecture,
    )
    return {
        "config": str(config_path),
        "critic_architecture": args.critic_architecture,
        "central_obs_dim": central_obs_dim,
        "actor_obs_dim": actor_obs_dim,
        "source_critic_architecture": loaded["args"]["source_critic_architecture"],
        "target_critic_body": type(loaded["params"].critic_body).__name__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["all", "mlp", "strided_cnn"], default="all")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--source-artifact", default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument("--no-artifact-download", action="store_true")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "critic_ablation_50x50_actor_only")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--smoke-updates", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-load-only", action="store_true")
    args = parser.parse_args()

    if args.smoke_updates is not None and args.smoke_updates <= 0:
        parser.error("--smoke-updates must be positive")

    source_checkpoint = _ensure_source_checkpoint(
        args.source_checkpoint,
        None if args.no_artifact_download else args.source_artifact,
    )
    run_id = args.run_id or utc_run_id()
    plans = [
        _argv_for_config(
            config_path,
            source_checkpoint=source_checkpoint,
            run_root=args.run_root,
            run_id=run_id,
            wandb_mode=args.wandb_mode,
            smoke_updates=args.smoke_updates,
        )
        for config_path in _selected_configs(args.only)
    ]

    verification = [
        _verify_actor_only_load(config_path, argv)
        for config_path, (_, _, argv) in zip(_selected_configs(args.only), plans)
    ]
    if args.dry_run or args.verify_load_only:
        print(
            json.dumps(
                {
                    "source_checkpoint": str(source_checkpoint),
                    "run_id": run_id,
                    "verification": verification,
                    "plans": [
                        {"experiment": name, "run_dir": str(run_dir), "argv": argv}
                        for name, run_dir, argv in plans
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from ant_byte_env.training.jax_mappo import runner as jax_runner

    results = []
    for name, run_dir, argv in plans:
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = jax_runner.main(argv)
        results.append({"experiment": name, "run_dir": str(run_dir), "metrics": metrics})
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
