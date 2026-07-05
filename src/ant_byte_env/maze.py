"""Wide-corridor maze layout helpers for AntByte environments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def generate_wide_corridor_maze(
    *,
    width: int,
    height: int,
    corridor_width: int = 3,
    wall_width: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Return a boolean obstacle grid generated from a depth-first-search maze."""

    width = int(width)
    height = int(height)
    corridor_width = int(corridor_width)
    wall_width = int(wall_width)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if corridor_width <= 0:
        raise ValueError("corridor_width must be positive.")
    if wall_width <= 0:
        raise ValueError("wall_width must be positive.")

    rng = np.random.default_rng(int(seed))
    maze_width = _maze_axis_cell_count(
        width,
        corridor_width=corridor_width,
        wall_width=wall_width,
    )
    maze_height = _maze_axis_cell_count(
        height,
        corridor_width=corridor_width,
        wall_width=wall_width,
    )
    col_widths = _variable_corridor_widths(
        width,
        cell_count=maze_width,
        corridor_width=corridor_width,
        wall_width=wall_width,
        rng=rng,
    )
    row_heights = _variable_corridor_widths(
        height,
        cell_count=maze_height,
        corridor_width=corridor_width,
        wall_width=wall_width,
        rng=rng,
    )
    col_starts, col_ends = _axis_spans(col_widths, wall_width=wall_width)
    row_starts, row_ends = _axis_spans(row_heights, wall_width=wall_width)

    obstacles = np.ones((height, width), dtype=bool)
    visited = np.zeros((maze_height, maze_width), dtype=bool)
    start = (
        int(rng.integers(0, maze_height)),
        int(rng.integers(0, maze_width)),
    )
    stack: list[tuple[int, int]] = [start]
    visited[start] = True
    carved_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    _open_cell_block(
        obstacles,
        row=start[0],
        col=start[1],
        row_starts=row_starts,
        row_ends=row_ends,
        col_starts=col_starts,
        col_ends=col_ends,
    )

    while stack:
        row, col = stack[-1]
        directions = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        rng.shuffle(directions)
        neighbours = [
            (next_row, next_col)
            for next_row, next_col in directions
            if 0 <= next_row < maze_height
            and 0 <= next_col < maze_width
            and not visited[next_row, next_col]
        ]
        if not neighbours:
            stack.pop()
            continue
        next_row, next_col = neighbours[0]
        visited[next_row, next_col] = True
        _open_cell_block(
            obstacles,
            row=next_row,
            col=next_col,
            row_starts=row_starts,
            row_ends=row_ends,
            col_starts=col_starts,
            col_ends=col_ends,
        )
        _open_connection(
            obstacles,
            row_a=row,
            col_a=col,
            row_b=next_row,
            col_b=next_col,
            row_starts=row_starts,
            row_ends=row_ends,
            col_starts=col_starts,
            col_ends=col_ends,
        )
        carved_edges.add(_edge_key((row, col), (next_row, next_col)))
        stack.append((next_row, next_col))

    _open_extra_connections(
        obstacles,
        maze_width=maze_width,
        maze_height=maze_height,
        carved_edges=carved_edges,
        row_starts=row_starts,
        row_ends=row_ends,
        col_starts=col_starts,
        col_ends=col_ends,
        rng=rng,
    )
    return obstacles


def generate_random_wall_obstacles(
    *,
    width: int,
    height: int,
    wall_count_min: int = 1,
    wall_count_max: int = 3,
    wall_length_min: int = 4,
    wall_length_max: int = 14,
    wall_width: int = 1,
    l_turn_probability: float = 0.5,
    center_window_size: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Return a boolean obstacle grid with random straight and L-shaped walls."""

    width = int(width)
    height = int(height)
    wall_count_min = int(wall_count_min)
    wall_count_max = int(wall_count_max)
    wall_length_min = int(wall_length_min)
    wall_length_max = int(wall_length_max)
    wall_width = int(wall_width)
    l_turn_probability = float(l_turn_probability)
    center_window_size = int(center_window_size)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if wall_count_min <= 0:
        raise ValueError("wall_count_min must be positive.")
    if wall_count_max < wall_count_min:
        raise ValueError("wall_count_max must be at least wall_count_min.")
    if wall_length_min <= 0:
        raise ValueError("wall_length_min must be positive.")
    if wall_length_max < wall_length_min:
        raise ValueError("wall_length_max must be at least wall_length_min.")
    if wall_width <= 0:
        raise ValueError("wall_width must be positive.")
    if not 0.0 <= l_turn_probability <= 1.0:
        raise ValueError("l_turn_probability must be between 0 and 1.")
    if wall_width >= min(width, height):
        raise ValueError("wall_width must be smaller than both grid axes.")
    if center_window_size < 0:
        raise ValueError("center_window_size must be non-negative.")
    if center_window_size > min(width, height):
        raise ValueError("center_window_size must fit inside the grid.")

    rng = np.random.default_rng(int(seed))
    obstacles = np.zeros((height, width), dtype=bool)
    wall_count = int(rng.integers(wall_count_min, wall_count_max + 1))
    max_axis_length = max(width, height)
    placement_bounds = _center_window_bounds(
        width=width,
        height=height,
        window_size=center_window_size,
    )
    for _ in range(wall_count):
        horizontal = bool(rng.integers(0, 2))
        length = int(rng.integers(wall_length_min, wall_length_max + 1))
        endpoints = _paint_random_wall_segment(
            obstacles,
            horizontal=horizontal,
            length=min(length, max_axis_length),
            wall_width=wall_width,
            rng=rng,
            attach=None,
            placement_bounds=placement_bounds,
        )
        if rng.random() >= l_turn_probability:
            continue
        length = int(rng.integers(wall_length_min, wall_length_max + 1))
        _paint_random_wall_segment(
            obstacles,
            horizontal=not horizontal,
            length=min(length, max_axis_length),
            wall_width=wall_width,
            rng=rng,
            attach=endpoints[int(rng.integers(0, len(endpoints)))],
            placement_bounds=None,
        )

    if not np.any(~obstacles):
        raise ValueError("random wall obstacle layout must contain at least one open cell.")
    return obstacles


def _paint_random_wall_segment(
    obstacles: np.ndarray,
    *,
    horizontal: bool,
    length: int,
    wall_width: int,
    rng: np.random.Generator,
    attach: tuple[int, int] | None,
    placement_bounds: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    height, width = obstacles.shape
    length = max(1, int(length))
    wall_width = max(1, int(wall_width))
    if horizontal:
        length = min(length, width)
        if attach is None:
            x0 = _sample_wall_start(
                rng,
                axis_size=width,
                span=length,
                window_start=None if placement_bounds is None else placement_bounds[0],
                window_end=None if placement_bounds is None else placement_bounds[1],
            )
            y0 = _sample_wall_start(
                rng,
                axis_size=height,
                span=wall_width,
                window_start=None if placement_bounds is None else placement_bounds[2],
                window_end=None if placement_bounds is None else placement_bounds[3],
            )
        else:
            anchor_x, anchor_y = attach
            extend_right = bool(rng.integers(0, 2))
            x0 = anchor_x if extend_right else anchor_x - length + wall_width
            x0 = int(np.clip(x0, 0, max(width - length, 0)))
            y0 = int(np.clip(anchor_y, 0, max(height - wall_width, 0)))
        x1 = min(x0 + length, width)
        y1 = min(y0 + wall_width, height)
        obstacles[y0:y1, x0:x1] = True
        return (x0, y0), (max(x0, x1 - wall_width), y0)

    length = min(length, height)
    if attach is None:
        x0 = _sample_wall_start(
            rng,
            axis_size=width,
            span=wall_width,
            window_start=None if placement_bounds is None else placement_bounds[0],
            window_end=None if placement_bounds is None else placement_bounds[1],
        )
        y0 = _sample_wall_start(
            rng,
            axis_size=height,
            span=length,
            window_start=None if placement_bounds is None else placement_bounds[2],
            window_end=None if placement_bounds is None else placement_bounds[3],
        )
    else:
        anchor_x, anchor_y = attach
        extend_down = bool(rng.integers(0, 2))
        x0 = int(np.clip(anchor_x, 0, max(width - wall_width, 0)))
        y0 = anchor_y if extend_down else anchor_y - length + wall_width
        y0 = int(np.clip(y0, 0, max(height - length, 0)))
    x1 = min(x0 + wall_width, width)
    y1 = min(y0 + length, height)
    obstacles[y0:y1, x0:x1] = True
    return (x0, y0), (x0, max(y0, y1 - wall_width))


def _center_window_bounds(
    *,
    width: int,
    height: int,
    window_size: int,
) -> tuple[int, int, int, int] | None:
    if window_size <= 0:
        return None
    x_start = (width - window_size) // 2
    y_start = (height - window_size) // 2
    return x_start, x_start + window_size, y_start, y_start + window_size


def _sample_wall_start(
    rng: np.random.Generator,
    *,
    axis_size: int,
    span: int,
    window_start: int | None,
    window_end: int | None,
) -> int:
    max_start = max(axis_size - span, 0)
    if window_start is None or window_end is None:
        return int(rng.integers(0, max_start + 1))

    # Pick starts whose painted interval intersects the preferred centered window.
    low = max(0, int(window_start) - int(span) + 1)
    high = min(max_start, int(window_end) - 1)
    if high < low:
        return int(rng.integers(0, max_start + 1))
    return int(rng.integers(low, high + 1))


def open_flat_indices(obstacles: np.ndarray) -> np.ndarray:
    """Return sorted flattened indices for open cells."""

    obstacle_grid = np.asarray(obstacles, dtype=bool)
    return np.flatnonzero(~obstacle_grid.reshape(-1)).astype(np.int32)


def nearest_open_flat_lookup(obstacles: np.ndarray) -> np.ndarray:
    """Return a lookup mapping any flat grid index to its nearest open flat index."""

    obstacle_grid = np.asarray(obstacles, dtype=bool)
    height, width = obstacle_grid.shape
    open_indices = open_flat_indices(obstacle_grid)
    if open_indices.size == 0:
        raise ValueError("maze must contain at least one open cell.")

    open_x = open_indices % width
    open_y = open_indices // width
    lookup = np.empty(width * height, dtype=np.int32)
    for flat_index in range(width * height):
        x_pos = flat_index % width
        y_pos = flat_index // width
        distances = np.abs(open_x - x_pos) + np.abs(open_y - y_pos)
        lookup[flat_index] = int(open_indices[int(np.argmin(distances))])
    return lookup


def validate_open_positions(
    *,
    obstacles: np.ndarray,
    positions: Iterable[tuple[int, int]],
) -> None:
    obstacle_grid = np.asarray(obstacles, dtype=bool)
    height, width = obstacle_grid.shape
    for x_pos, y_pos in positions:
        if not (0 <= int(x_pos) < width and 0 <= int(y_pos) < height):
            raise ValueError(f"position {(int(x_pos), int(y_pos))!r} is outside the grid.")
        if obstacle_grid[int(y_pos), int(x_pos)]:
            raise ValueError(f"position {(int(x_pos), int(y_pos))!r} is blocked by a maze wall.")


def _open_cell_block(
    obstacles: np.ndarray,
    *,
    row: int,
    col: int,
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    col_starts: np.ndarray,
    col_ends: np.ndarray,
) -> None:
    obstacles[row_starts[row] : row_ends[row], col_starts[col] : col_ends[col]] = False


def _open_connection(
    obstacles: np.ndarray,
    *,
    row_a: int,
    col_a: int,
    row_b: int,
    col_b: int,
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    col_starts: np.ndarray,
    col_ends: np.ndarray,
) -> None:
    if row_a == row_b:
        y0, y1 = row_starts[row_a], row_ends[row_a]
        if col_a < col_b:
            x0, x1 = col_ends[col_a], col_starts[col_b]
        else:
            x0, x1 = col_ends[col_b], col_starts[col_a]
    else:
        x0, x1 = col_starts[col_a], col_ends[col_a]
        if row_a < row_b:
            y0, y1 = row_ends[row_a], row_starts[row_b]
        else:
            y0, y1 = row_ends[row_b], row_starts[row_a]
    obstacles[y0:y1, x0:x1] = False


def _maze_axis_cell_count(
    length: int,
    *,
    corridor_width: int,
    wall_width: int,
) -> int:
    stride = corridor_width + wall_width
    desired = max(2, int(np.ceil(max(1, length - wall_width) / stride)))
    max_cells = max(1, (length - wall_width) // (1 + wall_width))
    return min(desired, max_cells)


def _variable_corridor_widths(
    length: int,
    *,
    cell_count: int,
    corridor_width: int,
    wall_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    available = int(length) - int(wall_width) * (int(cell_count) + 1)
    if available <= 0:
        raise ValueError("maze dimensions are too small for the requested wall width.")
    base_width = max(1, min(int(corridor_width), available // int(cell_count)))
    widths = np.full(int(cell_count), base_width, dtype=np.int32)
    remaining = available - int(widths.sum())
    max_width = max(base_width, int(corridor_width) + 2)
    while remaining > 0:
        candidates = np.flatnonzero(widths < max_width)
        if candidates.size == 0:
            candidates = np.arange(widths.size)
        index = int(rng.choice(candidates))
        widths[index] += 1
        remaining -= 1
    return widths


def _axis_spans(widths: np.ndarray, *, wall_width: int) -> tuple[np.ndarray, np.ndarray]:
    starts: list[int] = []
    ends: list[int] = []
    cursor = int(wall_width)
    for width in widths:
        starts.append(cursor)
        cursor += int(width)
        ends.append(cursor)
        cursor += int(wall_width)
    return np.asarray(starts, dtype=np.int32), np.asarray(ends, dtype=np.int32)


def _open_extra_connections(
    obstacles: np.ndarray,
    *,
    maze_width: int,
    maze_height: int,
    carved_edges: set[tuple[tuple[int, int], tuple[int, int]]],
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    col_starts: np.ndarray,
    col_ends: np.ndarray,
    rng: np.random.Generator,
) -> None:
    for row in range(maze_height):
        for col in range(maze_width):
            for next_row, next_col in ((row + 1, col), (row, col + 1)):
                if next_row >= maze_height or next_col >= maze_width:
                    continue
                edge = _edge_key((row, col), (next_row, next_col))
                if edge in carved_edges or rng.random() >= 0.16:
                    continue
                _open_connection(
                    obstacles,
                    row_a=row,
                    col_a=col,
                    row_b=next_row,
                    col_b=next_col,
                    row_starts=row_starts,
                    row_ends=row_ends,
                    col_starts=col_starts,
                    col_ends=col_ends,
                )


def _edge_key(
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    first, second = sorted((a, b))
    return first, second
