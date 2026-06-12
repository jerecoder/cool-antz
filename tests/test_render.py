from __future__ import annotations

import argparse

import numpy as np

from ant_byte_env.env import AntByteForagingEnv, facing_rotation_degrees, food_alpha
from ant_byte_env.rendering import _render_frame


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


def test_checkpoint_render_frame_overlays_ant_vision_square() -> None:
    env = AntByteForagingEnv(
        width=4,
        height=3,
        num_ants=1,
        food_count=0,
        render_mode="rgb_array",
        tile_size=10,
    )
    obs, _ = env.reset(seed=7, options={"hub_pos": (1, 1)})

    plain_frame = _render_frame(
        env,
        obs,
        args=argparse.Namespace(actor_vision_radius=1),
        show_vision=False,
    )
    vision_frame = _render_frame(
        env,
        obs,
        args=argparse.Namespace(actor_vision_radius=1),
        show_vision=True,
    )

    assert plain_frame.shape == vision_frame.shape
    assert not np.array_equal(plain_frame, vision_frame)
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
