"""CLI argv helpers for workflow orchestration."""

from __future__ import annotations

from collections.abc import Sequence

WANDB_CLI_VALUE_ARGS = frozenset(
    {
        "--wandb-project",
        "--wandb-entity",
        "--wandb-group",
        "--wandb-run-name",
        "--wandb-notes",
        "--wandb-mode",
    }
)
WANDB_CLI_VARARGS = frozenset({"--wandb-tags"})


def argv_int(argv: Sequence[str], option: str) -> int | None:
    try:
        index = len(argv) - 1 - list(reversed(argv)).index(option)
    except ValueError:
        return None
    try:
        return int(argv[index + 1])
    except (IndexError, ValueError):
        return None


def strip_wandb_cli_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    stripped: list[str] = []
    removed: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        value = str(values[index])
        if value in WANDB_CLI_VALUE_ARGS:
            removed.append(value)
            index += 1
            if index < len(values):
                removed.append(str(values[index]))
                index += 1
            continue
        if value in WANDB_CLI_VARARGS:
            removed.append(value)
            index += 1
            while index < len(values) and not str(values[index]).startswith("--"):
                removed.append(str(values[index]))
                index += 1
            continue
        stripped.append(value)
        index += 1
    return stripped, removed


__all__ = [
    "WANDB_CLI_VALUE_ARGS",
    "WANDB_CLI_VARARGS",
    "argv_int",
    "strip_wandb_cli_args",
]
