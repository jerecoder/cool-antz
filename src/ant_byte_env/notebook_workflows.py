"""Reusable workflow helpers for the AntByte notebooks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import gc
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ant_byte_env import MAX_WRITE_BITS
from ant_byte_env.experiments import config_args_to_argv, load_experiment_config
from ant_byte_env.rendering import render_checkpoint
from ant_byte_env.vault import create_vault_entry

FORAGE_STAGE_SIZES = tuple(range(4, 51))
CURRICULUM_BITES_PER_FOOD_SOURCE = 4
NOTEBOOK_ROLLOUT_TILE_SIZE = 16
NOTEBOOK_ROLLOUT_SEED_OFFSET = 100_000
NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE = 0.0
DEFAULT_JAX_MEMORY_FRACTION = "0.35"
NOTEBOOK_SAFE_CLEANUP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".ruff_cache",
    }
)
COMMUNICATION_ARG_EXCLUDES = {
    "exp_name",
    "write_bits",
    "total_timesteps",
    "save_model",
    "load_model",
    "run_dir",
}
ANT_COUNT_ARG_EXCLUDES = COMMUNICATION_ARG_EXCLUDES | {"num_ants"}
VISION_RANGE_ARG_EXCLUDES = {
    "exp_name",
    "actor_vision_radius",
    "total_timesteps",
    "save_model",
    "load_model",
    "run_dir",
}


def configure_jax_notebook_runtime(
    *,
    memory_fraction: str = DEFAULT_JAX_MEMORY_FRACTION,
) -> dict[str, Any]:
    """Set conservative JAX runtime defaults before notebooks import JAX."""

    jax_already_imported = "jax" in sys.modules
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _set_jax_memory_fraction(str(memory_fraction), jax_already_imported=jax_already_imported)
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    memory_trimmed = trim_current_process_memory()
    snapshot = notebook_resource_snapshot()
    return {
        "jax_already_imported": jax_already_imported,
        "jax_preallocate": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "jax_memory_fraction": os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],
        "jax_allocator": os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"],
        "memory_trimmed": memory_trimmed,
        **snapshot,
    }


def assert_notebook_resources_available(
    snapshot: Mapping[str, Any],
    *,
    min_disk_free_gb: float = 3.0,
    min_mem_available_gb: float = 2.0,
    min_swap_free_gb: float = 0.25,
    max_gpu_compute_memory_mb: int = 1024,
) -> None:
    """Fail early when a notebook run is likely to destabilize the machine."""

    issues: list[str] = []
    mem_available_gb = float(snapshot.get("mem_available_gb", min_mem_available_gb))
    swap_free_gb = float(snapshot.get("swap_free_gb", min_swap_free_gb))
    if float(snapshot.get("disk_free_gb", min_disk_free_gb)) < min_disk_free_gb:
        issues.append(f"disk free is below {min_disk_free_gb:g} GB")
    if mem_available_gb < min_mem_available_gb:
        issues.append(f"available RAM is below {min_mem_available_gb:g} GB")
    if swap_free_gb < min_swap_free_gb and mem_available_gb < 2 * min_mem_available_gb:
        issues.append(
            f"free swap is below {min_swap_free_gb:g} GB while available RAM is low"
        )
    gpu_memory = snapshot.get("gpu_compute_memory_mb")
    if gpu_memory is not None and int(gpu_memory) > max_gpu_compute_memory_mb:
        issues.append(f"GPU compute memory already exceeds {max_gpu_compute_memory_mb} MB")
    recovery_action = str(snapshot.get("gpu_recovery_action", "")).strip()
    if recovery_action and recovery_action.lower() not in {"n/a", "none", "no action"}:
        issues.append(
            f"GPU recovery action is {recovery_action}; reboot before expecting CUDA/JAX GPU"
        )
    if issues:
        details = "\n- ".join(issues)
        recovery = _resource_recovery_text(snapshot)
        raise RuntimeError(
            "Notebook resources look unsafe. Stop old kernels/processes or free space, then "
            f"restart this kernel.\n- {details}{recovery}"
        )


def notebook_resource_snapshot(path: Path | str = ".") -> dict[str, Any]:
    """Return a lightweight disk/RAM/GPU snapshot for notebook preflight cells."""

    root = Path(path)
    disk = shutil.disk_usage(root)
    cleanup_candidates = _safe_cleanup_candidates(root)
    snapshot: dict[str, Any] = {
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "disk_used_percent": round(disk.used / disk.total * 100, 1),
        "current_pid": os.getpid(),
        "safe_cleanup_candidate_count": len(cleanup_candidates),
        "safe_cleanup_candidate_gb": round(
            sum(_path_size_bytes(candidate) for candidate in cleanup_candidates) / 1024**3,
            3,
        ),
        "top_memory_processes": _top_memory_processes(),
    }
    runs_path = root / "runs"
    if runs_path.exists():
        snapshot["runs_size_gb"] = round(_path_size_bytes(runs_path) / 1024**3, 3)
    snapshot.update(_linux_memory_snapshot())
    snapshot.update(_nvidia_health_snapshot())
    gpu_memory_mb = _nvidia_compute_memory_mb()
    if gpu_memory_mb is not None:
        snapshot["gpu_compute_memory_mb"] = gpu_memory_mb
    return snapshot


def cleanup_notebook_artifacts(
    project_root: Path | str = ".",
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove safe local notebook/cache artifacts without touching run outputs."""

    root = Path(project_root)
    candidates = _safe_cleanup_candidates(root)
    freed_bytes = sum(_path_size_bytes(candidate) for candidate in candidates)
    removed: list[str] = []
    if not dry_run:
        for candidate in candidates:
            if candidate.exists():
                shutil.rmtree(candidate)
                removed.append(str(candidate))
    return {
        "dry_run": dry_run,
        "candidate_count": len(candidates),
        "removed_count": len(removed),
        "freed_bytes": freed_bytes,
        "freed_gb": round(freed_bytes / 1024**3, 3),
        "paths": [str(candidate) for candidate in candidates],
    }


def trim_current_process_memory() -> bool:
    """Ask Python/libc to return free arenas to the OS before resource checks."""

    gc.collect()
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        return bool(libc.malloc_trim(0))
    except (AttributeError, OSError):
        return False


def _set_jax_memory_fraction(memory_fraction: str, *, jax_already_imported: bool) -> None:
    current = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    if current is None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = memory_fraction
        return
    if jax_already_imported:
        return
    current_value = _float_or_none(current)
    requested_value = _float_or_none(memory_fraction)
    if current_value is None or requested_value is None or current_value > requested_value:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = memory_fraction


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _resource_recovery_text(snapshot: Mapping[str, Any]) -> str:
    lines: list[str] = []
    top_processes = list(snapshot.get("top_memory_processes") or [])
    if top_processes:
        lines.append("")
        lines.append("Largest memory users:")
        for process in top_processes[:5]:
            marker = " current" if process.get("is_current_process") else ""
            kernel = " notebook-kernel" if process.get("is_notebook_kernel") else ""
            lines.append(
                "- PID {pid}: {rss_mb:g} MB{marker}{kernel} :: {command}".format(
                    pid=process.get("pid"),
                    rss_mb=float(process.get("rss_mb", 0.0)),
                    marker=marker,
                    kernel=kernel,
                    command=str(process.get("command", ""))[:140],
                )
            )
    stale_kernel_pids = [
        str(process["pid"])
        for process in top_processes
        if process.get("is_notebook_kernel") and not process.get("is_current_process")
    ]
    lines.append("")
    lines.append("Suggested recovery:")
    recovery_action = str(snapshot.get("gpu_recovery_action", "")).strip()
    if recovery_action and recovery_action.lower() not in {"n/a", "none", "no action"}:
        lines.append(
            f"- Reboot this machine; NVIDIA reports GPU Recovery Action: {recovery_action}."
        )
    if stale_kernel_pids:
        lines.append(
            "- Shut down stale Jupyter/VS Code kernels first. If those PIDs are stale, run: "
            f"kill {' '.join(stale_kernel_pids[:6])}"
        )
    lines.append(
        "- Clean safe local caches from a fresh cell or terminal: "
        "from ant_byte_env import notebook_workflows as workflows; "
        "workflows.cleanup_notebook_artifacts(PROJECT_ROOT, dry_run=False)"
    )
    if "runs_size_gb" in snapshot:
        lines.append(
            f"- runs/ currently uses {snapshot['runs_size_gb']} GB; archive or delete old runs "
            "only after keeping the checkpoints/media you need."
        )
    return "\n" + "\n".join(lines)


def _linux_memory_snapshot() -> dict[str, Any]:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return {}

    values: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        key, _, raw_value = line.partition(":")
        parts = raw_value.strip().split()
        if parts:
            values[key] = int(parts[0])
    return {
        "mem_available_gb": round(values.get("MemAvailable", 0) / 1024**2, 2),
        "swap_free_gb": round(values.get("SwapFree", 0) / 1024**2, 2),
    }


def _nvidia_health_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}

    raw = _parse_nvidia_smi_query(result.stdout)
    snapshot: dict[str, Any] = {}
    key_map = {
        "Driver Version": "gpu_driver_version",
        "CUDA Version": "gpu_cuda_version",
        "Product Name": "gpu_name",
        "GPU Recovery Action": "gpu_recovery_action",
    }
    for raw_key, snapshot_key in key_map.items():
        if raw_key in raw:
            snapshot[snapshot_key] = raw[raw_key]
    return snapshot


def _parse_nvidia_smi_query(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            parsed.setdefault(key.strip(), value.strip())
    return parsed


def _safe_cleanup_candidates(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name in NOTEBOOK_SAFE_CLEANUP_DIR_NAMES:
            candidates.append(path)
    return candidates


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _top_memory_processes(limit: int = 8) -> list[dict[str, Any]]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []

    current_uid = os.getuid() if hasattr(os, "getuid") else None
    current_pid = os.getpid()
    processes: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        status = _proc_status(entry)
        if not status:
            continue
        if current_uid is not None and _proc_uid(status) != current_uid:
            continue
        rss_kb = _status_kb(status.get("VmRSS", "0 kB"))
        if rss_kb <= 0:
            continue
        pid = int(entry.name)
        command_parts = _proc_cmdline(entry)
        command = " ".join(command_parts) if command_parts else status.get("Name", "")
        processes.append(
            {
                "pid": pid,
                "ppid": int(status.get("PPid", "0")),
                "rss_mb": round(rss_kb / 1024, 1),
                "command": command,
                "connection_file": _connection_file(command_parts),
                "is_current_process": pid == current_pid,
                "is_notebook_kernel": _is_notebook_kernel(command_parts, command),
            }
        )
    return sorted(processes, key=lambda process: float(process["rss_mb"]), reverse=True)[:limit]


def _proc_status(proc_entry: Path) -> dict[str, str]:
    try:
        lines = (proc_entry / "status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    status: dict[str, str] = {}
    for line in lines:
        key, _, value = line.partition(":")
        if key:
            status[key] = value.strip()
    return status


def _proc_uid(status: Mapping[str, str]) -> int | None:
    raw_uid = status.get("Uid", "")
    parts = raw_uid.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def _status_kb(value: str) -> int:
    parts = value.split()
    if not parts:
        return 0
    try:
        return int(parts[0])
    except ValueError:
        return 0


def _proc_cmdline(proc_entry: Path) -> list[str]:
    try:
        raw = (proc_entry / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _connection_file(command_parts: Sequence[str]) -> str | None:
    for index, part in enumerate(command_parts):
        if part.startswith("--f="):
            return part.removeprefix("--f=")
        if part == "--f" and index + 1 < len(command_parts):
            return command_parts[index + 1]
    return None


def _is_notebook_kernel(command_parts: Sequence[str], command: str) -> bool:
    return (
        "ipykernel_launcher" in command
        or any(part.startswith("--f=") and "kernel-" in part for part in command_parts)
    )


def _nvidia_compute_memory_mb() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    total = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            total += int(stripped)
    return total


def load_jax_experiment(config_path: Path) -> Any:
    experiment = load_experiment_config(config_path)
    if experiment.backend != "jax":
        raise ValueError(f"Expected a JAX experiment config, got {experiment.backend!r}.")
    return experiment


def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root / resolved


def config_common_args(training_args: Mapping[str, Any], *, exclude: Iterable[str]) -> list[str]:
    excluded = set(exclude)
    return config_args_to_argv(
        {key: value for key, value in training_args.items() if key not in excluded}
    )


def update_timesteps(*, num_envs: int, num_steps: int) -> int:
    return int(num_envs) * int(num_steps)


def curriculum_food_count(size: int) -> int:
    return 2 + max(0, int(size) - 4)


def curriculum_food_sources(size: int) -> int:
    food_count = curriculum_food_count(size)
    concentrated_sources = (
        food_count + CURRICULUM_BITES_PER_FOOD_SOURCE - 1
    ) // CURRICULUM_BITES_PER_FOOD_SOURCE
    return max(1, min(food_count, concentrated_sources))


def build_forage_curriculum_stages(
    stage_sizes: Sequence[int] = FORAGE_STAGE_SIZES,
) -> list[dict[str, int | str]]:
    return [
        {
            "name": f"{size}x{size}",
            "width": int(size),
            "height": int(size),
            "food_count": curriculum_food_count(int(size)),
            "food_sources": curriculum_food_sources(int(size)),
            "cookie_distance": min(1 + (int(size) - 4) // 2, int(size) // 2),
            "max_steps": max(48, 4 * int(size) * int(size)),
        }
        for size in stage_sizes
    ]


def build_forage_common_args(
    stages: Sequence[Mapping[str, Any]],
    *,
    num_envs: int,
    num_steps: int,
    actor_vision_radius: int,
    write_bits: int,
    gamma: float = 0.99,
    write_while_moving: bool = True,
    seed: int = 1,
) -> list[str]:
    max_width = max(int(stage["width"]) for stage in stages)
    max_height = max(int(stage["height"]) for stage in stages)
    args = [
        "--num-envs",
        str(num_envs),
        "--num-steps",
        str(num_steps),
        "--num-minibatches",
        "4",
        "--update-epochs",
        "4",
        "--gamma",
        str(float(gamma)),
        "--obs-width",
        str(max_width),
        "--obs-height",
        str(max_height),
        "--actor-vision-radius",
        str(actor_vision_radius),
        "--write-bits",
        str(write_bits),
        "--num-ants",
        "1",
        "--random-food",
        "--random-hub",
        "--pickup-bonus",
        "0.25",
        "--hidden-size",
        "128",
        "--seed",
        str(seed),
        "--quiet",
    ]
    if write_while_moving:
        args.append("--write-while-moving")
    return args


def run_jax_smoke(train_main: Callable[..., dict[str, float]]) -> dict[str, float]:
    return train_main(
        [
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--max-steps",
            "8",
            "--write-bits",
            "1",
            "--hidden-size",
            "16",
            "--seed",
            "11",
            "--quiet",
        ]
    )


def run_forage_curriculum(
    *,
    stages: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
    common_args: Sequence[str],
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_metrics: list[dict[str, Any]] = []
    stage_checkpoint_paths: list[Path] = []
    previous_checkpoint: Path | None = None
    final_train_metrics: dict[str, float] = {}

    for stage_index, stage in enumerate(stages, start=1):
        print(f"Training stage {stage_index}/{len(stages)}: {stage['name']}")
        print("First update for this shape may compile; progress starts after it returns.")
        checkpoint_path = checkpoint_dir / f"jax_mappo_forage_stage1_{stage['name']}.pkl"
        progress = stage_update_progress(str(stage["name"]), global_update_cap)

        def record_progress(
            update_index: int,
            total_updates: int,
            metrics: dict[str, float],
        ) -> None:
            del total_updates
            progress.update(1)
            postfix = {
                "loss": f"{metrics['loss']:.3f}",
                "ret": f"{metrics['episode_return']:.3f}",
            }
            if "recent_episode_return" in metrics:
                postfix["ret_avg"] = f"{metrics['recent_episode_return']:.3f}"
            progress.set_postfix(**postfix)
            stage_metrics.append(
                {
                    **stage,
                    **metrics,
                    "stage_update": update_index,
                    "global_update_cap": global_update_cap,
                    "checkpoint": str(checkpoint_path),
                }
            )

        train_args = [
            *common_args,
            "--total-timesteps",
            str(update_timesteps_per_stage * global_update_cap),
            "--width",
            str(stage["width"]),
            "--height",
            str(stage["height"]),
            "--food-count",
            str(stage["food_count"]),
            "--food-sources",
            str(stage["food_sources"]),
            "--cookie-distance",
            str(stage["cookie_distance"]),
            "--max-steps",
            str(stage["max_steps"]),
            "--save-model",
            str(checkpoint_path),
        ]
        if previous_checkpoint is not None:
            train_args.extend(["--load-model", str(previous_checkpoint)])

        try:
            final_train_metrics = train_main(train_args, progress_callback=record_progress)
        finally:
            progress.close()

        stage_checkpoint_paths.append(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        previous_checkpoint = checkpoint_path

    return {
        "stage_metrics": stage_metrics,
        "stage_checkpoint_paths": stage_checkpoint_paths,
        "final_checkpoint_path": previous_checkpoint,
        "final_train_metrics": final_train_metrics,
    }


def validate_vision_range_stages(vision_radii: Sequence[int]) -> None:
    if any(radius <= 0 for radius in vision_radii):
        raise ValueError("vision radii must be positive.")
    if not all(left > right for left, right in zip(vision_radii, vision_radii[1:])):
        raise ValueError("vision radii must be strictly decreasing.")


def vision_side(radius: int) -> int:
    return 2 * int(radius) + 1


def run_vision_range_curriculum(
    *,
    vision_radii: Sequence[int],
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
    render_rollouts: bool = True,
    max_render_frames: int | None = 300,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
) -> dict[str, Any]:
    validate_vision_range_stages(vision_radii)
    run_dir.mkdir(parents=True, exist_ok=True)
    media_dir = run_dir / "media"
    media_dir.mkdir(exist_ok=True)
    stage_metrics: list[dict[str, Any]] = []
    stage_checkpoint_paths: list[Path] = []
    rollout_paths: list[Path] = []
    previous_checkpoint: Path | None = None
    final_train_metrics: dict[str, float] = {}

    for stage_index, radius in enumerate(vision_radii):
        side = vision_side(int(radius))
        stage_name = f"{side}x{side}"
        stage_run_dir = run_dir / stage_name
        checkpoint_path = stage_run_dir / "checkpoints" / "model.pkl"
        print(f"Training vision stage {stage_index + 1}/{len(vision_radii)}: {stage_name}")
        if previous_checkpoint is not None:
            print(f"Starting from: {previous_checkpoint}")
        progress = stage_update_progress(stage_name, global_update_cap)

        def record_progress(
            update_index: int,
            total_updates: int,
            metrics: dict[str, float],
        ) -> None:
            del total_updates
            progress.update(1)
            postfix = {
                "loss": f"{metrics['loss']:.3f}",
                "ret": f"{metrics['episode_return']:.3f}",
            }
            if "recent_episode_return" in metrics:
                postfix["ret_avg"] = f"{metrics['recent_episode_return']:.3f}"
            progress.set_postfix(**postfix)
            stage_metrics.append(
                {
                    "actor_vision_radius": int(radius),
                    "vision_side": side,
                    **metrics,
                    "stage_update": update_index,
                    "global_update_cap": global_update_cap,
                    "checkpoint": str(checkpoint_path),
                    "source_checkpoint": str(previous_checkpoint)
                    if previous_checkpoint is not None
                    else None,
                    "run_dir": str(stage_run_dir),
                }
            )

        train_args = [
            *common_args,
            "--exp-name",
            f"{experiment_name}_{stage_name}",
            "--actor-vision-radius",
            str(int(radius)),
            "--total-timesteps",
            str(update_timesteps_per_stage * global_update_cap),
            "--run-dir",
            str(stage_run_dir),
        ]
        if previous_checkpoint is not None:
            train_args.extend(["--load-model", str(previous_checkpoint)])

        try:
            final_train_metrics = train_main(train_args, progress_callback=record_progress)
        finally:
            progress.close()

        stage_checkpoint_paths.append(checkpoint_path)
        if render_rollouts:
            rollout_path = media_dir / f"vision_{stage_name}.gif"
            rollout_paths.append(
                render_checkpoint(
                    checkpoint_path,
                    rollout_path,
                    backend="jax",
                    seed_offset=NOTEBOOK_ROLLOUT_SEED_OFFSET + stage_index,
                    reuse_existing=False,
                    max_frames=max_render_frames,
                    tile_size=tile_size,
                    policy_temperature=policy_temperature,
                )
            )
        previous_checkpoint = checkpoint_path

    return {
        "stage_metrics": stage_metrics,
        "stage_checkpoint_paths": stage_checkpoint_paths,
        "final_checkpoint": previous_checkpoint,
        "final_train_metrics": final_train_metrics,
        "rollout_paths": rollout_paths,
    }


def validate_communication_stages(bit_stages: Sequence[int]) -> None:
    if any(bits <= 1 or bits > MAX_WRITE_BITS for bits in bit_stages):
        raise ValueError(f"bit stages must contain integers from 2 to {MAX_WRITE_BITS}.")
    if not _strictly_increasing(bit_stages):
        raise ValueError("bit stages must be increasing.")


def run_communication_bit_curriculum(
    *,
    bit_stages: Sequence[int],
    source_checkpoint: Path,
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    validate_communication_stages(bit_stages)
    stage_metrics: list[dict[str, Any]] = []
    stage_checkpoint_paths: list[Path] = []
    previous_checkpoint = source_checkpoint
    final_train_metrics: dict[str, float] = {}

    for target_bits in bit_stages:
        stage_run_dir = run_dir / f"{target_bits}_bits"
        checkpoint_path = stage_run_dir / "checkpoints" / "model.pkl"
        print(f"Training communication stage: {target_bits} writable bits")
        print(f"Starting from: {previous_checkpoint}")
        progress = stage_update_progress(f"{target_bits} bits", global_update_cap)

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
                    "write_bits": target_bits,
                    **metrics,
                    "stage_update": update_index,
                    "global_update_cap": global_update_cap,
                    "checkpoint": str(checkpoint_path),
                    "source_checkpoint": str(previous_checkpoint),
                    "run_dir": str(stage_run_dir),
                }
            )

        train_args = [
            *common_args,
            "--exp-name",
            f"{experiment_name}_{target_bits}_bits",
            "--write-bits",
            str(target_bits),
            "--total-timesteps",
            str(update_timesteps_per_stage * global_update_cap),
            "--load-model",
            str(previous_checkpoint),
            "--run-dir",
            str(stage_run_dir),
        ]
        try:
            final_train_metrics = train_main(train_args, progress_callback=record_progress)
        finally:
            progress.close()

        stage_checkpoint_paths.append(checkpoint_path)
        print(f"Saved {target_bits}-bit checkpoint to {checkpoint_path}")
        previous_checkpoint = checkpoint_path

    return {
        "source_checkpoint": source_checkpoint,
        "stage_checkpoint_paths": stage_checkpoint_paths,
        "final_checkpoint": previous_checkpoint,
        "final_train_metrics": final_train_metrics,
    }


def run_communication_consolidation(
    *,
    source_checkpoint: Path,
    target_bits: int,
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
    stage_name: str = "8_bits_consolidated",
    extra_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(target_bits) <= 1 or int(target_bits) > MAX_WRITE_BITS:
        raise ValueError(f"target_bits must be an integer from 2 to {MAX_WRITE_BITS}.")
    if int(global_update_cap) <= 0:
        raise ValueError("global_update_cap must be positive.")

    stage_run_dir = run_dir / stage_name
    checkpoint_path = stage_run_dir / "checkpoints" / "model.pkl"
    stage_metrics: list[dict[str, Any]] = []
    progress = stage_update_progress(stage_name, global_update_cap)
    print(f"Training communication consolidation: {stage_name}")
    print(f"Starting from: {source_checkpoint}")

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
                "write_bits": int(target_bits),
                **metrics,
                "stage_update": update_index,
                "global_update_cap": int(global_update_cap),
                "checkpoint": str(checkpoint_path),
                "source_checkpoint": str(source_checkpoint),
                "run_dir": str(stage_run_dir),
            }
        )

    train_args = [
        *common_args,
        *config_args_to_argv(dict(extra_args or {})),
        "--exp-name",
        f"{experiment_name}_{stage_name}",
        "--write-bits",
        str(int(target_bits)),
        "--total-timesteps",
        str(update_timesteps_per_stage * int(global_update_cap)),
        "--load-model",
        str(source_checkpoint),
        "--run-dir",
        str(stage_run_dir),
    ]
    try:
        final_train_metrics = train_main(train_args, progress_callback=record_progress)
    finally:
        progress.close()

    print(f"Saved consolidated checkpoint to {checkpoint_path}")
    return {
        "source_checkpoint": source_checkpoint,
        "stage_checkpoint_paths": [checkpoint_path],
        "final_checkpoint": checkpoint_path,
        "final_train_metrics": final_train_metrics,
        "stage_metrics": stage_metrics,
        "stage_name": stage_name,
    }


def run_communication_post_stage_sequence(
    *,
    stage_configs: Mapping[str, Mapping[str, Any]],
    source_checkpoint: Path,
    target_bits: int,
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    train_main: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    current_checkpoint = source_checkpoint
    stage_results: dict[str, dict[str, Any] | None] = {}
    checkpoint_paths: list[Path] = []

    for label, config in stage_configs.items():
        if not config.get("enabled", False):
            stage_results[label] = None
            continue

        result = run_communication_consolidation(
            source_checkpoint=current_checkpoint,
            target_bits=target_bits,
            run_dir=run_dir,
            common_args=common_args,
            experiment_name=experiment_name,
            update_timesteps_per_stage=update_timesteps_per_stage,
            global_update_cap=int(config.get("global_update_cap", 0)),
            train_main=train_main,
            stage_name=str(config.get("stage_name", f"{target_bits}_bits_{label}")),
            extra_args=dict(config.get("args", {})),
        )
        current_checkpoint = result["final_checkpoint"]
        checkpoint_paths.append(current_checkpoint)
        stage_results[label] = result

    return {
        "source_checkpoint": source_checkpoint,
        "checkpoint_paths": checkpoint_paths,
        "final_checkpoint": current_checkpoint,
        "stage_results": stage_results,
    }


def ant_count_training_args(
    base_args: Mapping[str, Any],
    *,
    communication_bits: int,
) -> dict[str, Any]:
    return {
        **base_args,
        "width": 25,
        "height": 25,
        "obs_width": 50,
        "obs_height": 50,
        "food_count": 23,
        "food_sources": 6,
        "cookie_distance": 11,
        "max_steps": 2500,
        "write_bits": int(communication_bits),
        "write_while_moving": True,
    }


def validate_ant_count_stages(*, ant_stages: Sequence[int], source_num_ants: int) -> None:
    if any(num_ants <= source_num_ants for num_ants in ant_stages):
        raise ValueError("ant stages must increase beyond the source checkpoint's ant count.")
    if not _strictly_increasing(ant_stages):
        raise ValueError("ant stages must be increasing.")


def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def run_ant_count_curriculum(
    *,
    ant_stages: Sequence[int],
    source_checkpoint: Path,
    source_num_ants: int,
    communication_bits: int,
    run_dir: Path,
    common_args: Sequence[str],
    experiment_name: str,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    train_main: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    validate_ant_count_stages(ant_stages=ant_stages, source_num_ants=source_num_ants)
    stage_metrics: list[dict[str, Any]] = []
    stage_checkpoint_paths: list[Path] = []
    previous_checkpoint = source_checkpoint
    previous_num_ants = int(source_num_ants)
    final_train_metrics: dict[str, float] = {}

    for target_num_ants in ant_stages:
        stage_run_dir = run_dir / f"{target_num_ants}_ants"
        checkpoint_path = stage_run_dir / "checkpoints" / "model.pkl"
        warm_start_checkpoint = (
            stage_run_dir
            / "warm_start"
            / f"from_{previous_num_ants}_to_{target_num_ants}_ants.pkl"
        )
        stage_source_checkpoint = previous_checkpoint
        stage_source_num_ants = previous_num_ants

        print(f"Training ant-count stage: {target_num_ants} ants")
        print(f"Starting from: {stage_source_checkpoint}")

        warm_start_args = ant_count_train_args(
            common_args=common_args,
            experiment_name=experiment_name,
            target_num_ants=target_num_ants,
            communication_bits=communication_bits,
            update_timesteps_per_stage=update_timesteps_per_stage,
            global_update_cap=global_update_cap,
            load_model=stage_source_checkpoint,
            run_dir=stage_run_dir,
        )
        prepare_ant_count_checkpoint(
            stage_source_checkpoint,
            warm_start_checkpoint,
            warm_start_args,
            fallback_source_num_ants=source_num_ants,
            expected_write_bits=communication_bits,
        )

        progress = stage_update_progress(f"{target_num_ants} ants", global_update_cap)

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
                    "num_ants": target_num_ants,
                    "source_num_ants": stage_source_num_ants,
                    "write_bits": communication_bits,
                    **metrics,
                    "stage_update": update_index,
                    "global_update_cap": global_update_cap,
                    "checkpoint": str(checkpoint_path),
                    "source_checkpoint": str(stage_source_checkpoint),
                    "warm_start_checkpoint": str(warm_start_checkpoint),
                    "run_dir": str(stage_run_dir),
                }
            )

        train_args = ant_count_train_args(
            common_args=common_args,
            experiment_name=experiment_name,
            target_num_ants=target_num_ants,
            communication_bits=communication_bits,
            update_timesteps_per_stage=update_timesteps_per_stage,
            global_update_cap=global_update_cap,
            load_model=warm_start_checkpoint,
            run_dir=stage_run_dir,
        )
        try:
            final_train_metrics = train_main(train_args, progress_callback=record_progress)
        finally:
            progress.close()

        stage_checkpoint_paths.append(checkpoint_path)
        print(f"Saved {target_num_ants}-ant checkpoint to {checkpoint_path}")
        previous_checkpoint = checkpoint_path
        previous_num_ants = int(target_num_ants)

    return {
        "source_checkpoint": source_checkpoint,
        "stage_checkpoint_paths": stage_checkpoint_paths,
        "final_checkpoint": previous_checkpoint,
        "final_train_metrics": final_train_metrics,
    }


def ant_count_train_args(
    *,
    common_args: Sequence[str],
    experiment_name: str,
    target_num_ants: int,
    communication_bits: int,
    update_timesteps_per_stage: int,
    global_update_cap: int,
    load_model: Path,
    run_dir: Path,
) -> list[str]:
    return [
        *common_args,
        "--exp-name",
        f"{experiment_name}_{target_num_ants}_ants",
        "--write-bits",
        str(communication_bits),
        "--num-ants",
        str(target_num_ants),
        "--total-timesteps",
        str(update_timesteps_per_stage * global_update_cap),
        "--load-model",
        str(load_model),
        "--run-dir",
        str(run_dir),
    ]


def expand_critic_input_for_ant_count(
    params: Any,
    *,
    source_num_ants: int,
    target_num_ants: int,
) -> Any:
    import jax.numpy as jnp

    from ant_byte_env.training.jax_mappo.core import JaxMAPPOParams, LinearParams

    source_num_ants = int(source_num_ants)
    target_num_ants = int(target_num_ants)
    if source_num_ants <= 0 or target_num_ants <= 0:
        raise ValueError("ant counts must be positive.")
    if params.critic_conv:
        raise ValueError("Ant-count transfer is not supported with critic conv layers.")

    first_layer = params.critic_body[0]
    old_weight = jnp.asarray(first_layer.weight)
    old_bias = jnp.asarray(first_layer.bias)
    source_ant_features = 3 * source_num_ants
    target_ant_features = 3 * target_num_ants
    if old_weight.shape[0] < source_ant_features:
        raise ValueError("source critic input is too small for its ant count.")

    tail_dim = old_weight.shape[0] - source_ant_features
    target_dim = target_ant_features + tail_dim
    if target_dim == old_weight.shape[0] and source_num_ants == target_num_ants:
        return params

    shared_ants = min(source_num_ants, target_num_ants)
    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)

    source_pos = slice(0, 2 * shared_ants)
    target_pos = slice(0, 2 * shared_ants)
    source_carry = slice(2 * source_num_ants, 2 * source_num_ants + shared_ants)
    target_carry = slice(2 * target_num_ants, 2 * target_num_ants + shared_ants)
    source_tail = slice(3 * source_num_ants, old_weight.shape[0])
    target_tail = slice(3 * target_num_ants, target_dim)

    new_weight = new_weight.at[target_pos, :].set(old_weight[source_pos, :])
    new_weight = new_weight.at[target_carry, :].set(old_weight[source_carry, :])
    new_weight = new_weight.at[target_tail, :].set(old_weight[source_tail, :])

    return JaxMAPPOParams(
        actor_body=params.actor_body,
        move_head=params.move_head,
        write_head=params.write_head,
        critic_body=(LinearParams(weight=new_weight, bias=old_bias), *params.critic_body[1:]),
        value_head=params.value_head,
        actor_conv=params.actor_conv,
        critic_conv=params.critic_conv,
    )


def training_dimensions(argv: Sequence[str]) -> tuple[Any, int, int]:
    import jax

    from ant_byte_env.jax_env import JaxAntByteForagingEnv
    from ant_byte_env.training.jax_mappo.cli import parse_args
    from ant_byte_env.training.jax_mappo.core import (
        build_actor_observations,
        build_central_observations,
    )
    from ant_byte_env.training.jax_mappo.curriculum import reset_batch

    args = parse_args(list(argv))
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        step_penalty=args.step_penalty,
        write_penalty=args.write_penalty,
        write_bits=args.write_bits,
        write_while_moving=args.write_while_moving,
    )
    _, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
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
    return args, int(central_obs.shape[-1]), int(actor_obs.shape[-1])


def prepare_ant_count_checkpoint(
    source_checkpoint: Path,
    warm_start_checkpoint: Path,
    target_argv: Sequence[str],
    *,
    fallback_source_num_ants: int,
    expected_write_bits: int,
) -> Path:
    from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint, save_checkpoint
    from ant_byte_env.training.jax_mappo.core import init_adam_state
    from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training

    source_checkpoint = Path(source_checkpoint)
    warm_start_checkpoint = Path(warm_start_checkpoint)
    target_args, target_central_obs_dim, target_actor_obs_dim = training_dimensions(target_argv)
    checkpoint = read_checkpoint(source_checkpoint)
    if int(checkpoint["actor_obs_dim"]) != target_actor_obs_dim:
        checkpoint = load_checkpoint_for_training(
            source_checkpoint,
            central_obs_dim=int(checkpoint["central_obs_dim"]),
            actor_obs_dim=target_actor_obs_dim,
            target_write_bits=expected_write_bits,
            actor_vision_radius=target_args.actor_vision_radius,
        )
    source_args = checkpoint.get("args", {})
    source_num_ants = int(source_args.get("num_ants", fallback_source_num_ants))
    source_write_bits = int(source_args.get("write_bits", expected_write_bits))

    if source_write_bits != expected_write_bits:
        raise ValueError(
            f"Expected a {expected_write_bits}-bit source checkpoint, got {source_write_bits}."
        )
    if int(checkpoint["actor_obs_dim"]) != target_actor_obs_dim:
        raise ValueError("Actor observation dimension transfer did not match this stage.")

    params = checkpoint["params"]
    if int(checkpoint["central_obs_dim"]) != target_central_obs_dim:
        params = expand_critic_input_for_ant_count(
            params,
            source_num_ants=source_num_ants,
            target_num_ants=target_args.num_ants,
        )
    if params.critic_body[0].weight.shape[0] != target_central_obs_dim:
        raise ValueError("Transferred critic input dimension does not match this stage.")

    save_checkpoint(
        warm_start_checkpoint,
        params=params,
        opt_state=init_adam_state(params),
        args=target_args,
        central_obs_dim=target_central_obs_dim,
        actor_obs_dim=target_actor_obs_dim,
        run_name=(
            f"{checkpoint.get('run_name', 'jax_mappo')}"
            f"__{target_args.num_ants}_ants_warm_start"
        ),
        metrics={
            **checkpoint.get("metrics", {}),
            "source_num_ants": float(source_num_ants),
            "target_num_ants": float(target_args.num_ants),
        },
    )
    return warm_start_checkpoint


def stage_update_progress(label: str, total_updates: int) -> Any:
    from tqdm.auto import tqdm

    return tqdm(
        range(1, int(total_updates) + 1),
        total=int(total_updates),
        desc=label,
        bar_format="{desc}: {n_fmt}/{total_fmt} updates |{bar}| {elapsed}<{remaining} {postfix}",
        leave=True,
    )


def forage_checkpoint_paths(
    checkpoint_dir: Path,
    stages: Sequence[Mapping[str, Any]],
) -> list[Path]:
    return [checkpoint_dir / f"jax_mappo_forage_stage1_{stage['name']}.pkl" for stage in stages]


def communication_checkpoint_paths(run_dir: Path, bit_stages: Sequence[int]) -> list[Path]:
    return [run_dir / f"{bits}_bits" / "checkpoints" / "model.pkl" for bits in bit_stages]


def ant_count_checkpoint_paths(run_dir: Path, ant_stages: Sequence[int]) -> list[Path]:
    return [
        run_dir / f"{num_ants}_ants" / "checkpoints" / "model.pkl"
        for num_ants in ant_stages
    ]


def render_forage_rollouts(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    media_dir: Path,
    stages: Sequence[Mapping[str, Any]],
    actor_vision_radius: int,
    write_bits: int,
    global_update_cap: int,
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
) -> dict[str, Any]:
    return render_rollout_suite(
        checkpoint_paths=forage_checkpoint_paths(checkpoint_dir, stages),
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda checkpoint, media: (
            media / f"{checkpoint.stem}_rollout.mp4"
        ),
        progress_desc="rendering policies",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO curriculum policy rollouts",
        description="Rollout MP4 videos for each saved JAX MAPPO curriculum stage policy.",
        metadata={
            "stages": [stage["name"] for stage in stages],
            "actor_vision_radius": actor_vision_radius,
            "write_bits": write_bits,
            "global_update_cap": global_update_cap,
        },
        max_frames=max_frames,
        tile_size=tile_size,
    )


def render_communication_rollouts(
    *,
    experiment_config: Path,
    source_checkpoint: Path,
    run_dir: Path,
    media_dir: Path,
    bit_stages: Sequence[int],
    global_update_cap: int,
    extra_checkpoint_paths: Sequence[Path] = (),
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
) -> dict[str, Any]:
    checkpoint_paths = [
        *communication_checkpoint_paths(run_dir, bit_stages),
        *[Path(path) for path in extra_checkpoint_paths],
    ]
    return render_rollout_suite(
        checkpoint_paths=checkpoint_paths,
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda checkpoint, media: (
            media / f"jax_mappo_25x25_{checkpoint.parent.parent.name}_vision_rollout.mp4"
        ),
        progress_desc="rendering communication policies",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO communication-bit curriculum",
        description=(
            "Rollout MP4 videos for 25x25 JAX MAPPO policies trained with progressively "
            "larger writable communication alphabets."
        ),
        metadata={
            "experiment_config": str(experiment_config),
            "source_checkpoint": str(source_checkpoint),
            "bit_stages": list(bit_stages),
            "global_update_cap": global_update_cap,
            "extra_checkpoint_paths": [str(path) for path in extra_checkpoint_paths],
        },
        max_frames=max_frames,
        tile_size=tile_size,
    )


def render_ant_count_rollouts(
    *,
    experiment_config: Path,
    source_checkpoint: Path,
    run_dir: Path,
    media_dir: Path,
    communication_bits: int,
    source_num_ants: int,
    ant_stages: Sequence[int],
    global_update_cap: int,
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
) -> dict[str, Any]:
    return render_rollout_suite(
        checkpoint_paths=ant_count_checkpoint_paths(run_dir, ant_stages),
        media_dir=media_dir,
        rollout_path_for_checkpoint=lambda checkpoint, media: (
            media / f"jax_mappo_25x25_3bits_{checkpoint.parent.parent.name}_vision_rollout.mp4"
        ),
        progress_desc="rendering ant-count policies",
        vault_dir=run_dir / "vault",
        title="JAX MAPPO ant-count curriculum",
        description=(
            "Rollout MP4 videos for 25x25, 3-bit JAX MAPPO policies trained with "
            "progressively larger ant teams."
        ),
        metadata={
            "experiment_config": str(experiment_config),
            "source_checkpoint": str(source_checkpoint),
            "communication_bits": communication_bits,
            "source_num_ants": source_num_ants,
            "ant_stages": list(ant_stages),
            "global_update_cap": global_update_cap,
        },
        max_frames=max_frames,
        tile_size=tile_size,
    )


def render_rollout_suite(
    *,
    checkpoint_paths: Sequence[Path],
    media_dir: Path,
    rollout_path_for_checkpoint: Callable[[Path, Path], Path],
    progress_desc: str,
    vault_dir: Path,
    title: str,
    description: str,
    metadata: Mapping[str, Any],
    max_frames: int | None = None,
    tile_size: int | None = NOTEBOOK_ROLLOUT_TILE_SIZE,
    policy_temperature: float = NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE,
) -> dict[str, Any]:
    from tqdm.auto import tqdm

    media_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [Path(path) for path in checkpoint_paths]
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Train the missing policies before rendering:\n{missing_text}")

    rollout_paths = []
    rollout_seed_offsets = []
    for rollout_index, checkpoint in enumerate(tqdm(checkpoints, desc=progress_desc)):
        seed_offset = NOTEBOOK_ROLLOUT_SEED_OFFSET + rollout_index
        rollout_seed_offsets.append(seed_offset)
        rollout_paths.append(
            render_checkpoint(
                checkpoint,
                rollout_path_for_checkpoint(checkpoint, media_dir),
                backend="jax",
                seed_offset=seed_offset,
                reuse_existing=False,
                max_frames=max_frames,
                tile_size=tile_size,
                policy_temperature=policy_temperature,
            )
        )
    vault_entry_path = create_vault_entry(
        vault_dir=vault_dir,
        title=title,
        description=description,
        assets=rollout_paths,
        metadata={
            **metadata,
            "render_max_frames": max_frames,
            "render_tile_size": tile_size,
            "rollout_policy_temperature": policy_temperature,
            "rollout_seed_offsets": rollout_seed_offsets,
            "checkpoint_paths": [str(path) for path in checkpoints],
            "rollout_paths": [str(path) for path in rollout_paths],
        },
    )
    return {"rollout_paths": rollout_paths, "vault_entry_path": vault_entry_path}
