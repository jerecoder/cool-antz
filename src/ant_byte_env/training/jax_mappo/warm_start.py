"""Warm-start helpers for communication-bit curriculum checkpoints."""

from __future__ import annotations

import numpy as np


def repeated_write_action_indices(old_bits: int, target_bits: int) -> np.ndarray:
    if old_bits <= 0:
        raise ValueError("old_bits must be positive.")
    if target_bits < old_bits:
        raise ValueError("target_bits must be at least old_bits.")
    old_count = 2**old_bits
    target_count = 2**target_bits
    return np.arange(target_count, dtype=np.int64) % old_count
