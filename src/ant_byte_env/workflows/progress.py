"""Progress helpers for staged workflow runs."""

from __future__ import annotations

from typing import Any


def advance_progress_to(
    progress: Any,
    *,
    update_index: int,
    previous_update_index: int,
) -> int:
    next_update_index = int(update_index)
    progress.update(max(0, next_update_index - int(previous_update_index)))
    return next_update_index


def stage_update_progress(label: str, total_updates: int) -> Any:
    from tqdm.auto import tqdm

    return tqdm(
        range(1, int(total_updates) + 1),
        total=int(total_updates),
        desc=label,
        bar_format="{desc}: {n_fmt}/{total_fmt} updates |{bar}| {elapsed}<{remaining} {postfix}",
        leave=True,
    )


__all__ = [
    "advance_progress_to",
    "stage_update_progress",
]
