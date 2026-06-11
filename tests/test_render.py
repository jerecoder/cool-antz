from __future__ import annotations

import numpy as np

from ant_byte_env.env import AntByteForagingEnv, facing_rotation_degrees, food_alpha


def test_rgb_array_render_returns_numpy_image() -> None:
    env = AntByteForagingEnv(
        width=4,
        height=3,
        num_ants=2,
        food_count=2,
        render_mode="rgb_array",
        tile_size=16,
    )
    env.reset(seed=21)

    frame = env.render()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (3 * 16, 4 * 16, 3)
    assert frame.dtype == np.uint8
    env.close()


def test_food_alpha_drops_as_food_source_is_depleted() -> None:
    assert food_alpha(remaining=4, initial=4) > food_alpha(remaining=2, initial=4)
    assert food_alpha(remaining=2, initial=4) > food_alpha(remaining=1, initial=4)
    assert food_alpha(remaining=0, initial=4) == 0


def test_facing_rotation_degrees_match_movement_actions() -> None:
    assert facing_rotation_degrees(2) == 0
    assert facing_rotation_degrees(1) == 90
    assert facing_rotation_degrees(4) == 180
    assert facing_rotation_degrees(3) == -90
