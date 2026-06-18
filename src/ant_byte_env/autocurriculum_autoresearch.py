"""Autoresearch helpers for the single-ant 50x50 autocurriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import jax

from ant_byte_env.autoresearch import (
    _resolve_matrix_path,
    _stage_metrics_from_run_dir,
    assert_autoresearch_resources_available,
)
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.rendering import render_checkpoint
from ant_byte_env.runs import write_json
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.cli import parse_args
from ant_byte_env.training.jax_mappo.core import (
    build_actor_observations,
    build_central_observations,
)
from ant_byte_env.training.jax_mappo.curriculum import reset_batch
from ant_byte_env.training.jax_mappo.rollout import collect_rollout
from ant_byte_env.training.jax_mappo.runner import (
    _autocurriculum_state_stats,
    _make_env,
    _rollout_stats,
)
from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training
from ant_byte_env.wandb_tracking import WandbTracker

DEFAULT_AUTOCURRICULUM_SWEEP_MATRIX = Path("autoresearch/autocurriculum_sweep.json")
AUTOCURRICULUM_ARG_EXCLUDES = {
    "total_timesteps",
    "save_model",
    "load_model",
    "run_dir",
    "wandb_project",
    "wandb_entity",
    "wandb_group",
    "wandb_run_name",
    "wandb_mode",
    "wandb_tags",
    "wandb_notes",
}
AUTOCURRICULUM_NO_CHEAT_NUM_ANTS = 1
AUTOCURRICULUM_NO_CHEAT_ACTOR_VISION_RADIUS = 1
AUTOCURRICULUM_NO_CHEAT_WRITE_BITS = 1
AUTOCURRICULUM_DISALLOWED_ACTOR_HINT_KEYS = {
    "actor_food_direction",
    "actor_food_distance",
    "actor_food_position",
    "actor_hub_direction",
    "actor_hub_distance",
    "actor_hub_position",
    "food_vector",
    "hub_vector",
    "oracle_food",
    "oracle_hub",
    "recurrent",
    "use_gru",
    "use_lstm",
}


def build_autocurriculum_sweep_plan(
    *,
    matrix_path: Path = DEFAULT_AUTOCURRICULUM_SWEEP_MATRIX,
    phase: str,
    run_id: str,
    run_root: Path | None = None,
    global_update_cap: int | None = None,
    num_envs: int | None = None,
    num_steps: int | None = None,
    probe_rollout_steps: int | None = None,
    probe_num_envs: int | None = None,
    render_rollout: bool | None = None,
    max_render_frames: int | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = None,
) -> dict[str, Any]:
    """Return one no-cheat autocurriculum experiment plan."""

    matrix = load_autocurriculum_sweep_matrix(matrix_path)
    entry = autocurriculum_sweep_entry(matrix, phase=phase, run_id=run_id)
    base_config = Path(str(matrix["base_config"]))
    base_spec = load_experiment_config(base_config)
    if base_spec.backend != "jax":
        raise ValueError("autocurriculum autoresearch requires a JAX base config.")

    entry_args = dict(entry.get("args", {}))
    merged_args = {**base_spec.args, **dict(matrix.get("default_args", {})), **entry_args}
    if num_envs is not None:
        merged_args["num_envs"] = int(num_envs)
    if num_steps is not None:
        merged_args["num_steps"] = int(num_steps)
    _assert_autocurriculum_no_cheat_args(merged_args)

    update_cap = (
        global_update_cap
        if global_update_cap is not None
        else entry.get("global_update_cap", matrix.get("global_update_cap"))
    )
    if update_cap is None:
        raise ValueError("global_update_cap is required for autocurriculum entries.")
    update_cap = int(update_cap)
    if update_cap <= 0:
        raise ValueError("global_update_cap must be positive.")

    run_dir = _resolve_matrix_path(
        Path(str(entry["run_dir"])),
        matrix_root=Path(str(matrix["run_root"])),
        override_root=run_root,
    )
    checkpoint_path = run_dir / "checkpoints" / "model.pkl"
    update_timesteps = int(merged_args["num_envs"]) * int(merged_args["num_steps"])
    total_timesteps = update_timesteps * update_cap
    merged_args["total_timesteps"] = total_timesteps
    wandb = _autocurriculum_wandb_config(
        matrix=matrix,
        entry=entry,
        phase=phase,
        run_id=run_id,
        project_override=wandb_project,
        mode_override=wandb_mode,
    )
    notes = _autocurriculum_wandb_notes(
        matrix=matrix,
        entry=entry,
        phase=phase,
        run_id=run_id,
    )
    common_args = config_args_to_argv(
        {
            key: value
            for key, value in merged_args.items()
            if key not in AUTOCURRICULUM_ARG_EXCLUDES
        }
    )
    _validate_autocurriculum_training_args(common_args)
    training_argv = [
        *common_args,
        "--total-timesteps",
        str(total_timesteps),
        "--run-dir",
        str(run_dir),
        "--save-model",
        str(checkpoint_path),
        *_wandb_argv(wandb, notes=notes),
    ]
    rollout_config = dict(matrix.get("rollout", {}))
    if render_rollout is not None:
        rollout_config["enabled"] = bool(render_rollout)
    if max_render_frames is not None:
        rollout_config["max_frames"] = int(max_render_frames)

    probe_steps = (
        int(probe_rollout_steps)
        if probe_rollout_steps is not None
        else int(entry.get("probe_rollout_steps", matrix.get("probe_rollout_steps", 1000)))
    )
    probe_envs = (
        int(probe_num_envs)
        if probe_num_envs is not None
        else int(entry.get("probe_num_envs", matrix.get("probe_num_envs", 16)))
    )
    if probe_steps <= 0 or probe_envs <= 0:
        raise ValueError("probe_rollout_steps and probe_num_envs must be positive.")

    return {
        "matrix_path": str(matrix_path),
        "phase": phase,
        "id": run_id,
        "description": entry.get("description", ""),
        "analysis": dict(matrix.get("analysis", {})),
        "base_config": str(base_config),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "global_update_cap": update_cap,
        "update_timesteps": update_timesteps,
        "total_train_env_steps": total_timesteps,
        "common_args": common_args,
        "training_argv": training_argv,
        "resolved_args": {
            key: value
            for key, value in merged_args.items()
            if key not in {"save_model", "load_model", "run_dir"}
        },
        "probe": {
            "rollout_steps": probe_steps,
            "num_envs": probe_envs,
            "seed_offset": int(entry.get("probe_seed_offset", 3_000_000)),
        },
        "rollout": {
            "enabled": bool(rollout_config.get("enabled", True)),
            "max_frames": (
                int(rollout_config["max_frames"])
                if rollout_config.get("max_frames") is not None
                else None
            ),
            "tile_size": (
                int(rollout_config["tile_size"])
                if rollout_config.get("tile_size") is not None
                else None
            ),
            "policy_temperature": float(rollout_config.get("policy_temperature", 1.0)),
            "output": str(run_dir / "media" / "sampled_autocurriculum_rollout.mp4"),
        },
        "wandb": {**wandb, "notes": notes},
        "no_cheat_invariants": {
            "num_ants": AUTOCURRICULUM_NO_CHEAT_NUM_ANTS,
            "actor_vision_radius": AUTOCURRICULUM_NO_CHEAT_ACTOR_VISION_RADIUS,
            "write_bits": AUTOCURRICULUM_NO_CHEAT_WRITE_BITS,
            "actor_observation": "local grid only; no food/hub coordinates or direction vectors",
            "central_critic": "may use padded global state for value learning",
        },
    }


def execute_autocurriculum_sweep_plan(
    plan: dict[str, Any],
    *,
    train_main: Callable[[list[str]], dict[str, float]] | None = None,
    probe_checkpoint: Callable[..., dict[str, Any]] | None = None,
    render_checkpoint_fn: Callable[..., Path] | None = None,
    check_resources: bool = True,
    resume_completed: bool = True,
) -> dict[str, Any]:
    """Execute one autocurriculum experiment and persist a compact summary."""

    if check_resources:
        assert_autoresearch_resources_available()
    if train_main is None:
        from ant_byte_env.training.jax_mappo.runner import main as train_main
    if probe_checkpoint is None:
        probe_checkpoint = probe_autocurriculum_checkpoint
    if render_checkpoint_fn is None:
        render_checkpoint_fn = render_checkpoint

    run_dir = Path(str(plan["run_dir"]))
    plan_path = run_dir / "sweep_plan.json"
    summary_path = run_dir / "sweep_summary.json"
    checkpoint_path = Path(str(plan["checkpoint"]))

    resumed = bool(resume_completed and checkpoint_path.exists())
    if not resumed:
        _clear_autocurriculum_run_outputs(plan)
    write_json(plan_path, plan)
    if resumed:
        train_metrics = _stage_metrics_from_run_dir(run_dir)
    else:
        train_metrics = train_main(list(plan["training_argv"]))

    probe_config = dict(plan["probe"])
    probe_payload = probe_checkpoint(
        checkpoint_path,
        rollout_steps=int(probe_config["rollout_steps"]),
        num_envs=int(probe_config["num_envs"]),
        seed_offset=int(probe_config.get("seed_offset", 3_000_000)),
    )
    rollout_path = _render_autocurriculum_rollout(
        plan=plan,
        checkpoint_path=checkpoint_path,
        render_checkpoint_fn=render_checkpoint_fn,
    )
    wandb_payload = _log_autocurriculum_outputs_to_wandb(
        plan=plan,
        train_metrics=train_metrics,
        probe_payload=probe_payload,
        rollout_path=rollout_path,
        checkpoint_path=checkpoint_path,
    )
    summary = {
        "plan_path": str(plan_path),
        "summary_path": str(summary_path),
        "phase": plan["phase"],
        "id": plan["id"],
        "run_dir": str(run_dir),
        "resumed": resumed,
        "checkpoint": str(checkpoint_path),
        "train_metrics": train_metrics,
        "probe": probe_payload,
        "rollout_path": str(rollout_path) if rollout_path is not None else None,
        "wandb": wandb_payload,
    }
    write_json(summary_path, summary)
    return summary


def probe_autocurriculum_checkpoint(
    checkpoint_path: Path,
    *,
    rollout_steps: int = 1000,
    num_envs: int = 16,
    seed_offset: int = 3_000_000,
) -> dict[str, Any]:
    """Run a fast fresh-start vectorized rollout probe for one checkpoint."""

    if rollout_steps <= 0 or num_envs <= 0:
        raise ValueError("rollout_steps and num_envs must be positive.")
    raw_checkpoint = read_checkpoint(checkpoint_path)
    args = _checkpoint_args_with_defaults(raw_checkpoint.get("args", {}))
    args.num_envs = int(num_envs)
    args.num_steps = int(rollout_steps)
    env = _make_env(args)
    states, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed + seed_offset))
    central_obs = build_central_observations(
        obs,
        food_scale=args.food_count,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    checkpoint = load_checkpoint_for_training(
        checkpoint_path,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        target_write_bits=args.write_bits,
        actor_vision_radius=args.actor_vision_radius,
    )
    rollout_fn = jax.jit(
        lambda rollout_key: collect_rollout(
            args=args,
            env=env,
            params=checkpoint["params"],
            states=states,
            obs=obs,
            key=rollout_key,
        )
    )
    final_states, _, rollout = rollout_fn(jax.random.PRNGKey(args.seed + seed_offset + 1))
    stats = {**_rollout_stats(rollout), **_autocurriculum_state_stats(final_states)}
    env_steps = int(num_envs) * int(rollout_steps)
    deliveries = float(stats.get("delivery_events", 0.0))
    completed_stages = float(stats.get("autocurriculum_completed_stages", 0.0))
    return {
        "checkpoint": str(checkpoint_path),
        "rollout_steps": int(rollout_steps),
        "num_envs": int(num_envs),
        "env_steps": env_steps,
        "metrics": {
            **stats,
            "delivery_events_per_1000_env_steps": deliveries / max(env_steps, 1) * 1000.0,
            "completed_stages_per_env": completed_stages / max(int(num_envs), 1),
        },
    }


def load_autocurriculum_sweep_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("autocurriculum sweep matrix must be a JSON object.")
    if "phases" not in payload or not isinstance(payload["phases"], dict):
        raise ValueError("autocurriculum sweep matrix requires a phases object.")
    return payload


def autocurriculum_sweep_entry(
    matrix: dict[str, Any],
    *,
    phase: str,
    run_id: str,
) -> dict[str, Any]:
    phases = matrix.get("phases", {})
    if phase not in phases:
        choices = ", ".join(sorted(phases))
        raise ValueError(f"unknown autocurriculum phase {phase!r}; choices: {choices}")
    for entry in phases[phase]:
        if str(entry.get("id")) == run_id:
            return dict(entry)
    choices = ", ".join(str(entry.get("id")) for entry in phases[phase])
    raise ValueError(f"unknown autocurriculum id {run_id!r}; choices: {choices}")


def _checkpoint_args_with_defaults(saved_args: dict[str, object]) -> argparse.Namespace:
    args = parse_args([])
    for key, value in saved_args.items():
        setattr(args, key, value)
    return args


def _validate_autocurriculum_training_args(common_args: list[str]) -> None:
    parse_args(list(common_args))


def _assert_autocurriculum_no_cheat_args(args: dict[str, Any]) -> None:
    if not bool(args.get("autocurriculum", False)):
        raise ValueError("autocurriculum autoresearch requires --autocurriculum.")
    if int(args.get("width", 0)) != 50 or int(args.get("height", 0)) != 50:
        raise ValueError("autocurriculum autoresearch is scoped to the 50x50 target.")
    if int(args.get("num_ants", AUTOCURRICULUM_NO_CHEAT_NUM_ANTS)) != 1:
        raise ValueError("autocurriculum autoresearch is scoped to exactly one ant.")
    if int(args.get("actor_vision_radius", 1)) != AUTOCURRICULUM_NO_CHEAT_ACTOR_VISION_RADIUS:
        raise ValueError("autocurriculum autoresearch must keep actor_vision_radius at 1.")
    if int(args.get("write_bits", 1)) != AUTOCURRICULUM_NO_CHEAT_WRITE_BITS:
        raise ValueError("autocurriculum screening keeps write_bits at 1.")
    disallowed = sorted(AUTOCURRICULUM_DISALLOWED_ACTOR_HINT_KEYS & set(args))
    if disallowed:
        names = ", ".join(disallowed)
        raise ValueError(f"autocurriculum autoresearch forbids actor oracle hints: {names}")


def _autocurriculum_wandb_config(
    *,
    matrix: dict[str, Any],
    entry: dict[str, Any],
    phase: str,
    run_id: str,
    project_override: str | None,
    mode_override: str | None,
) -> dict[str, Any]:
    base = dict(matrix.get("wandb", {}))
    entry_wandb = dict(entry.get("wandb", {}))
    payload = {**base, **entry_wandb}
    if project_override is not None:
        payload["project"] = project_override
    if mode_override is not None:
        payload["mode"] = mode_override
    payload.setdefault("project", None)
    payload.setdefault("entity", None)
    payload.setdefault("group", "autocurriculum_50x50_autoresearch")
    payload.setdefault("mode", "online")
    payload.setdefault("name", f"autocurriculum-{phase}-{run_id}")
    tags = [
        *[str(tag) for tag in base.get("tags", [])],
        *[str(tag) for tag in entry_wandb.get("tags", [])],
        phase,
        run_id,
    ]
    payload["tags"] = list(dict.fromkeys(tags))
    return {
        "project": payload.get("project"),
        "entity": payload.get("entity"),
        "group": payload.get("group"),
        "name": payload.get("name"),
        "mode": payload.get("mode"),
        "tags": payload.get("tags"),
    }


def _autocurriculum_wandb_notes(
    *,
    matrix: dict[str, Any],
    entry: dict[str, Any],
    phase: str,
    run_id: str,
) -> str:
    analysis = dict(matrix.get("analysis", {}))
    lines = [
        f"Autocurriculum autoresearch {phase}/{run_id}.",
        f"Hypothesis: {entry.get('description', '')}",
    ]
    for key in ("diagnosis", "reward_plan", "metric_plan"):
        if analysis.get(key):
            lines.append(f"{key.replace('_', ' ').title()}: {analysis[key]}")
    lines.append(
        "No-cheat constraints: one ant, actor vision radius 1, one write bit, "
        "feed-forward policy, no actor food/hub coordinates or direction vectors."
    )
    return "\n".join(lines)


def _wandb_argv(wandb: dict[str, Any], *, notes: str) -> list[str]:
    argv: list[str] = []
    if wandb.get("project") is not None:
        argv.extend(["--wandb-project", str(wandb["project"])])
    if wandb.get("entity") is not None:
        argv.extend(["--wandb-entity", str(wandb["entity"])])
    if wandb.get("group") is not None:
        argv.extend(["--wandb-group", str(wandb["group"])])
    if wandb.get("name") is not None:
        argv.extend(["--wandb-run-name", str(wandb["name"])])
    if wandb.get("mode") is not None:
        argv.extend(["--wandb-mode", str(wandb["mode"])])
    tags = list(wandb.get("tags") or [])
    if tags:
        argv.append("--wandb-tags")
        argv.extend(str(tag) for tag in tags)
    argv.extend(["--wandb-notes", notes])
    return argv


def _render_autocurriculum_rollout(
    *,
    plan: dict[str, Any],
    checkpoint_path: Path,
    render_checkpoint_fn: Callable[..., Path],
) -> Path | None:
    rollout = dict(plan["rollout"])
    if not bool(rollout.get("enabled", True)):
        return None
    output_path = Path(str(rollout["output"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return render_checkpoint_fn(
        checkpoint_path,
        output_path,
        backend="jax",
        reuse_existing=True,
        max_frames=rollout.get("max_frames"),
        tile_size=rollout.get("tile_size"),
        policy_temperature=float(rollout.get("policy_temperature", 1.0)),
    )


def _clear_autocurriculum_run_outputs(plan: dict[str, Any]) -> None:
    run_dir = Path(str(plan["run_dir"]))
    generated_paths = [
        run_dir / "config.json",
        run_dir / "metrics.jsonl",
        run_dir / "summary.json",
        run_dir / "sweep_summary.json",
        Path(str(plan["checkpoint"])),
        Path(str(plan["rollout"]["output"])),
    ]
    for path in generated_paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _log_autocurriculum_outputs_to_wandb(
    *,
    plan: dict[str, Any],
    train_metrics: dict[str, float],
    probe_payload: dict[str, Any],
    rollout_path: Path | None,
    checkpoint_path: Path,
) -> dict[str, Any]:
    wandb = dict(plan["wandb"])
    tracker = WandbTracker(
        project=wandb.get("project"),
        entity=wandb.get("entity"),
        group=wandb.get("group"),
        name=f"{wandb.get('name')}-outputs" if wandb.get("name") else None,
        tags=wandb.get("tags"),
        mode=str(wandb.get("mode", "online")),
        run_dir=Path(str(plan["run_dir"])),
        config={
            "phase": plan["phase"],
            "id": plan["id"],
            "description": plan.get("description", ""),
            "probe": plan["probe"],
            "rollout": plan["rollout"],
            "no_cheat_invariants": plan["no_cheat_invariants"],
        },
        notes=str(wandb.get("notes", "")),
    )
    step = train_metrics.get("global_step")
    probe_metrics = {
        f"probe/{key}": value for key, value in probe_payload.get("metrics", {}).items()
    }
    tracker.log_metrics(probe_metrics, step=step)
    if rollout_path is not None:
        tracker.log_video("videos/autocurriculum/sampled_rollout", rollout_path, step=step)
    tracker.log_artifact(
        f"autocurriculum-{plan['phase']}-{plan['id']}-checkpoint",
        checkpoint_path,
        artifact_type="model",
        aliases=[str(plan["phase"]), str(plan["id"]), "latest"],
    )
    tracker.finish()
    return {
        "enabled": tracker.enabled,
        "run_name": f"{wandb.get('name')}-outputs" if wandb.get("name") else None,
        "logged_probe_metric_count": len(probe_metrics),
        "logged_video": bool(tracker.enabled and rollout_path is not None),
        "logged_checkpoint_artifact": bool(tracker.enabled),
    }
