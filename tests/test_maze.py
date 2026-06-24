from __future__ import annotations

import numpy as np

from ant_byte_env.maze import generate_wide_corridor_maze


def test_wide_corridor_maze_varies_on_small_maps() -> None:
    layouts = [
        generate_wide_corridor_maze(
            width=10,
            height=10,
            corridor_width=3,
            wall_width=1,
            seed=seed,
        )
        for seed in range(16)
    ]
    unique_layouts = {layout.tobytes() for layout in layouts}

    assert len(unique_layouts) >= 8
    for layout in layouts:
        assert np.any(layout)
        assert np.any(~layout)


def test_wide_corridor_maze_varies_corridor_widths() -> None:
    open_cell_counts = {
        int(
            np.sum(
                ~generate_wide_corridor_maze(
                    width=12,
                    height=12,
                    corridor_width=3,
                    wall_width=1,
                    seed=seed,
                )
            )
        )
        for seed in range(16)
    }

    assert len(open_cell_counts) >= 3
