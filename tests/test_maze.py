from __future__ import annotations

import numpy as np

from ant_byte_env.maze import generate_random_wall_obstacles, generate_wide_corridor_maze


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


def test_random_wall_obstacles_generate_variable_open_layouts() -> None:
    layouts = [
        generate_random_wall_obstacles(
            width=20,
            height=20,
            wall_count_min=2,
            wall_count_max=4,
            wall_length_min=5,
            wall_length_max=10,
            wall_width=2,
            l_turn_probability=1.0,
            seed=seed,
        )
        for seed in range(8)
    ]
    unique_layouts = {layout.tobytes() for layout in layouts}

    assert len(unique_layouts) >= 6
    for layout in layouts:
        assert np.any(layout)
        assert np.any(~layout)


def test_random_wall_obstacles_can_prefer_center_window() -> None:
    layout = generate_random_wall_obstacles(
        width=20,
        height=20,
        wall_count_min=4,
        wall_count_max=4,
        wall_length_min=4,
        wall_length_max=6,
        wall_width=1,
        l_turn_probability=0.0,
        center_window_size=8,
        seed=12,
    )
    center = layout[6:14, 6:14]

    assert np.any(center)
    assert np.any(~layout)
