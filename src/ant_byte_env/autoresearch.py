"""Helpers for turning autoresearch matrices into runnable commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.runs import write_json

DEFAULT_COMMUNICATION_SWEEP_MATRIX = Path("autoresearch/communication_sweep.json")
AUTORESEARCH_MIN_DISK_FREE_GB = 5.0
AUTORESEARCH_MIN_MEM_AVAILABLE_GB = 4.0
AUTORESEARCH_MIN_SWAP_FREE_GB = 0.25
AUTORESEARCH_MAX_GPU_COMPUTE_MEMORY_MB = 1024


class AutoresearchResourceError(RuntimeError):
    """Raised when a long autoresearch run should not start on this machine."""


def build_communication_sweep_plan(
    *,
    matrix_path: Path = DEFAULT_COMMUNICATION_SWEEP_MATRIX,
    phase: str,
    run_id: str,
    bit_stages: list[int] | None = None,
    global_update_cap: int | None = None,
    num_envs: int | None = None,
    num_steps: int | None = None,
    probe_episodes: int = 4,
    render_rollouts: bool = True,
    probe_tile_size: int | None = 16,
) -> dict[str, Any]:
    """Return staged train commands and a probe command for one sweep entry."""

    if probe_episodes <= 0:
        raise ValueError("probe_episodes must be positive.")
    matrix = load_communication_sweep_matrix(matrix_path)
    entry = communication_sweep_entry(matrix, phase=phase, run_id=run_id)
    base_config = Path(matrix["base_config"])
    base_spec = load_experiment_config(base_config)
    entry_args = dict(entry.get("args", {}))
    merged_args = {**base_spec.args, **entry_args}
    if num_envs is not None:
        merged_args["num_envs"] = int(num_envs)
    if num_steps is not None:
        merged_args["num_steps"] = int(num_steps)

    update_cap = (
        global_update_cap if global_update_cap is not None else entry.get("global_update_cap")
    )
    if update_cap is None:
        raise ValueError("global_update_cap is required for dependent sweep entries.")
    update_cap = int(update_cap)
    if update_cap <= 0:
        raise ValueError("global_update_cap must be positive.")

    stages = bit_stages or _default_bit_stages(matrix, phase)
    if not stages:
        raise ValueError("bit_stages must not be empty.")
    if any(int(bits) <= 1 for bits in stages):
        raise ValueError("communication bit stages must be greater than one.")

    num_envs_value = int(merged_args["num_envs"])
    num_steps_value = int(merged_args["num_steps"])
    total_timesteps = num_envs_value * num_steps_value * update_cap
    run_dir = Path(str(entry["run_dir"]))
    probe_output_dir = Path(str(entry["probe_output_dir"]))
    previous_checkpoint = Path(str(merged_args["load_model"]))
    train_commands = []
    for stage_index, target_bits in enumerate(stages, start=1):
        stage_run_dir = run_dir / f"{int(target_bits)}_bits"
        checkpoint_path = stage_run_dir / "checkpoints" / "model.pkl"
        exp_name = f"{base_spec.args.get('exp_name', base_spec.name)}_{run_id}_{int(target_bits)}_bits"
        overrides = {
            **entry_args,
            "num_envs": num_envs_value,
            "num_steps": num_steps_value,
            "exp_name": exp_name,
            "write_bits": int(target_bits),
            "total_timesteps": total_timesteps,
            "load_model": str(previous_checkpoint),
            "run_dir": str(stage_run_dir),
        }
        override_argv = config_args_to_argv(overrides)
        training_argv = [*config_args_to_argv(base_spec.args), *override_argv]
        argv = [
            "ant-byte",
            "train",
            "jax",
            "--config",
            str(base_config),
            "--",
            *override_argv,
        ]
        train_commands.append(
            {
                "stage_index": stage_index,
                "write_bits": int(target_bits),
                "source_checkpoint": str(previous_checkpoint),
                "checkpoint": str(checkpoint_path),
                "run_dir": str(stage_run_dir),
                "training_argv": training_argv,
                "argv": argv,
            }
        )
        previous_checkpoint = checkpoint_path

    probe_argv = [
        "ant-byte",
        "probe",
        "communication",
        "--checkpoint",
        str(previous_checkpoint),
        "--output-dir",
        str(probe_output_dir),
        "--num-episodes",
        str(int(probe_episodes)),
    ]
    if not render_rollouts:
        probe_argv.append("--no-render")

    return {
        "matrix_path": str(matrix_path),
        "phase": phase,
        "id": run_id,
        "depends_on": entry.get("depends_on"),
        "base_config": str(base_config),
        "bit_stages": [int(bits) for bits in stages],
        "run_dir": str(run_dir),
        "probe_output_dir": str(probe_output_dir),
        "global_update_cap": update_cap,
        "env_steps_per_stage": total_timesteps,
        "train_commands": train_commands,
        "probe_command": {
            "checkpoint": str(previous_checkpoint),
            "output_dir": str(probe_output_dir),
            "options": {
                "num_episodes": int(probe_episodes),
                "render_rollouts": bool(render_rollouts),
                "tile_size": probe_tile_size,
            },
            "argv": probe_argv,
        },
    }


def execute_communication_sweep_plan(
    plan: dict[str, Any],
    *,
    train_main: Callable[[list[str]], dict[str, float]] | None = None,
    probe_checkpoint: Callable[..., dict[str, Any]] | None = None,
    check_resources: bool = True,
) -> dict[str, Any]:
    """Execute a communication sweep plan and persist a compact summary."""

    if check_resources:
        assert_autoresearch_resources_available()
    if train_main is None:
        from ant_byte_env.training.jax_mappo.runner import main as train_main
    if probe_checkpoint is None:
        from ant_byte_env.training.jax_mappo.probe import (
            probe_communication_checkpoint as probe_checkpoint,
        )

    run_dir = Path(str(plan["run_dir"]))
    plan_path = run_dir / "sweep_plan.json"
    summary_path = run_dir / "sweep_summary.json"
    write_json(plan_path, plan)

    stage_results = []
    for command in plan["train_commands"]:
        metrics = train_main(list(command["training_argv"]))
        stage_results.append(
            {
                "write_bits": int(command["write_bits"]),
                "run_dir": command["run_dir"],
                "checkpoint": command["checkpoint"],
                "metrics": metrics,
            }
        )

    probe_command = plan["probe_command"]
    probe_options = dict(probe_command.get("options", {}))
    probe_payload = probe_checkpoint(
        Path(str(probe_command["checkpoint"])),
        output_dir=Path(str(probe_command["output_dir"])),
        num_episodes=int(probe_options.get("num_episodes", 4)),
        render_rollouts=bool(probe_options.get("render_rollouts", True)),
        tile_size=probe_options.get("tile_size", 16),
    )
    summary = {
        "plan_path": str(plan_path),
        "summary_path": str(summary_path),
        "phase": plan["phase"],
        "id": plan["id"],
        "stage_results": stage_results,
        "probe": probe_payload,
    }
    write_json(summary_path, summary)
    return summary


def assert_autoresearch_resources_available(
    snapshot: dict[str, Any] | None = None,
    *,
    min_disk_free_gb: float = AUTORESEARCH_MIN_DISK_FREE_GB,
    min_mem_available_gb: float = AUTORESEARCH_MIN_MEM_AVAILABLE_GB,
    min_swap_free_gb: float = AUTORESEARCH_MIN_SWAP_FREE_GB,
    max_gpu_compute_memory_mb: int = AUTORESEARCH_MAX_GPU_COMPUTE_MEMORY_MB,
) -> None:
    """Fail before a long autoresearch run when local resources look unsafe."""

    from ant_byte_env.notebook_workflows import (
        assert_notebook_resources_available,
        notebook_resource_snapshot,
    )

    actual_snapshot = notebook_resource_snapshot() if snapshot is None else snapshot
    try:
        assert_notebook_resources_available(
            actual_snapshot,
            min_disk_free_gb=min_disk_free_gb,
            min_mem_available_gb=min_mem_available_gb,
            min_swap_free_gb=min_swap_free_gb,
            max_gpu_compute_memory_mb=max_gpu_compute_memory_mb,
        )
    except RuntimeError as exc:
        raise AutoresearchResourceError(
            "Autoresearch resources look unsafe for a communication sweep.\n"
            f"{exc}"
        ) from exc


def load_communication_sweep_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("communication sweep matrix must be a JSON object.")
    if "phases" not in payload or not isinstance(payload["phases"], dict):
        raise ValueError("communication sweep matrix requires a phases object.")
    return payload


def communication_sweep_entry(
    matrix: dict[str, Any],
    *,
    phase: str,
    run_id: str,
) -> dict[str, Any]:
    phases = matrix.get("phases", {})
    if phase not in phases:
        choices = ", ".join(sorted(phases))
        raise ValueError(f"unknown communication sweep phase {phase!r}; choices: {choices}")
    for entry in phases[phase]:
        if str(entry.get("id")) == run_id:
            return dict(entry)
    choices = ", ".join(str(entry.get("id")) for entry in phases[phase])
    raise ValueError(f"unknown communication sweep id {run_id!r}; choices: {choices}")


def _default_bit_stages(matrix: dict[str, Any], phase: str) -> list[int]:
    key = "final_bit_stages" if phase == "final" else "screening_bit_stages"
    return [int(bits) for bits in matrix.get(key, [])]
