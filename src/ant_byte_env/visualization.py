"""Shared visualization overlays for AntByte renders."""

from __future__ import annotations

from typing import Mapping

import numpy as np

DEFAULT_VISION_COLORS = (
    (61, 220, 255),
    (255, 113, 206),
    (255, 226, 89),
    (93, 255, 139),
)


def draw_vision_squares(
    frame: np.ndarray,
    obs: Mapping[str, np.ndarray],
    *,
    tile_size: int,
    vision_radius: int,
    colors: tuple[tuple[int, int, int], ...] = DEFAULT_VISION_COLORS,
    border_px: int = 2,
    fill_alpha: float = 0.12,
) -> np.ndarray:
    """Return a copy of ``frame`` with each ant's local vision square overlaid."""

    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if vision_radius < 0:
        raise ValueError("vision_radius must be non-negative.")
    if border_px <= 0:
        raise ValueError("border_px must be positive.")
    if not 0.0 <= fill_alpha <= 1.0:
        raise ValueError("fill_alpha must be between 0 and 1.")
    if not colors:
        raise ValueError("at least one vision color is required.")

    output = frame.copy()
    grid_height, grid_width = obs["food"].shape
    frame_height, frame_width = output.shape[:2]

    for ant_index, position in enumerate(obs["ants_pos"]):
        x_pos, y_pos = int(position[0]), int(position[1])
        left_tile = max(0, x_pos - vision_radius)
        right_tile = min(grid_width, x_pos + vision_radius + 1)
        top_tile = max(0, y_pos - vision_radius)
        bottom_tile = min(grid_height, y_pos + vision_radius + 1)
        x0 = int(np.clip(left_tile * tile_size, 0, frame_width))
        x1 = int(np.clip(right_tile * tile_size, 0, frame_width))
        y0 = int(np.clip(top_tile * tile_size, 0, frame_height))
        y1 = int(np.clip(bottom_tile * tile_size, 0, frame_height))
        if x0 >= x1 or y0 >= y1:
            continue

        color = np.array(colors[ant_index % len(colors)], dtype=np.float32)
        region = output[y0:y1, x0:x1].astype(np.float32)
        output[y0:y1, x0:x1] = (
            region * (1.0 - fill_alpha) + color * fill_alpha
        ).astype(output.dtype)

        border = min(border_px, max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2))
        output[y0 : y0 + border, x0:x1] = color.astype(output.dtype)
        output[y1 - border : y1, x0:x1] = color.astype(output.dtype)
        output[y0:y1, x0 : x0 + border] = color.astype(output.dtype)
        output[y0:y1, x1 - border : x1] = color.astype(output.dtype)

    return output
