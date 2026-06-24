"""Runtime resource guards and cleanup helpers for notebooks and research loops."""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_JAX_MEMORY_FRACTION = "0.35"
NOTEBOOK_SAFE_CLEANUP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".ruff_cache",
    }
)

__all__ = [
    "DEFAULT_JAX_MEMORY_FRACTION",
    "NOTEBOOK_SAFE_CLEANUP_DIR_NAMES",
    "assert_notebook_resources_available",
    "cleanup_notebook_artifacts",
    "configure_jax_notebook_runtime",
    "notebook_resource_snapshot",
    "trim_current_process_memory",
]


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
    if swap_free_gb < min_swap_free_gb and mem_available_gb < min_mem_available_gb:
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
