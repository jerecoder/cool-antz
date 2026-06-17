from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

from ant_byte_env import cli as ant_cli
from ant_byte_env.env import AntByteForagingEnv, facing_rotation_degrees, food_alpha
from ant_byte_env.rendering import (
    _can_reuse_render,
    _deterministic_from_temperature,
    _env_from_args,
    _jax_render_reset_options,
    _render_frame,
    _render_step_count,
)


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
    assert not np.array_equal(plain_frame[0, 15], vision_frame[0, 15])
    assert np.array_equal(plain_frame[15, 15], vision_frame[15, 15])
    env.close()


def test_render_reuse_requires_nonempty_output_newer_than_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pkl"
    output_path = tmp_path / "rollout.mp4"
    checkpoint_path.write_bytes(b"checkpoint")
    output_path.write_bytes(b"mp4")
    os.utime(checkpoint_path, (100, 100))
    os.utime(output_path, (101, 101))

    assert _can_reuse_render(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        reuse_existing=True,
    )
    assert not _can_reuse_render(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        reuse_existing=False,
    )

    os.utime(checkpoint_path, (102, 102))
    assert not _can_reuse_render(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        reuse_existing=True,
    )

    output_path.write_bytes(b"")
    os.utime(output_path, (103, 103))
    assert not _can_reuse_render(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        reuse_existing=True,
    )


def test_render_step_count_caps_total_frames() -> None:
    args = argparse.Namespace(max_steps=10)

    assert _render_step_count(args, max_frames=None) == 10
    assert _render_step_count(args, max_frames=1) == 0
    assert _render_step_count(args, max_frames=4) == 3
    assert _render_step_count(args, max_frames=99) == 10

    with pytest.raises(ValueError, match="max_frames"):
        _render_step_count(args, max_frames=0)


def test_render_temperature_zero_is_deterministic_and_positive_samples() -> None:
    assert _deterministic_from_temperature(0.0)
    assert not _deterministic_from_temperature(0.5)
    with pytest.raises(ValueError, match="non-negative"):
        _deterministic_from_temperature(-0.5)


def test_jax_render_reset_options_respect_random_hub() -> None:
    args = argparse.Namespace(
        width=7,
        height=6,
        cookie_distance=2,
        random_food=False,
        random_hub=True,
    )

    first = _jax_render_reset_options(args, seed=123)
    second = _jax_render_reset_options(args, seed=123)

    assert first == second
    assert first is not None
    assert set(first) == {"hub_pos", "food_positions"}
    assert 0 <= first["hub_pos"][0] < args.width
    assert 0 <= first["hub_pos"][1] < args.height
    assert first["food_positions"][0] != first["hub_pos"]


def test_env_from_args_accepts_render_tile_size() -> None:
    args = argparse.Namespace(
        width=5,
        height=4,
        num_ants=1,
        food_count=2,
        food_sources=1,
        max_steps=12,
        random_food=True,
        step_penalty=0.0,
        write_penalty=0.0,
        write_bits=1,
    )

    env = _env_from_args(args, render_mode="rgb_array", tile_size=12)

    assert env.tile_size == 12
    env.close()


def test_render_cli_passes_render_policy_flags(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_render_checkpoint(*args, reuse_existing: bool, **kwargs) -> Path:
        del args
        captured["reuse_existing"] = reuse_existing
        captured["policy_temperature"] = kwargs["policy_temperature"]
        return tmp_path / "rollout.mp4"

    monkeypatch.setattr(ant_cli, "render_checkpoint", fake_render_checkpoint)

    exit_code = ant_cli.main(
        [
            "render",
            "--checkpoint",
            str(tmp_path / "model.pkl"),
            "--output",
            str(tmp_path / "rollout.mp4"),
            "--backend",
            "jax",
            "--reuse-existing",
            "--policy-temperature",
            "1.0",
        ]
    )

    assert exit_code == 0
    assert captured == {"reuse_existing": True, "policy_temperature": 1.0}


def test_food_alpha_drops_as_food_source_is_depleted() -> None:
    assert food_alpha(remaining=4, initial=4) > food_alpha(remaining=2, initial=4)
    assert food_alpha(remaining=2, initial=4) > food_alpha(remaining=1, initial=4)
    assert food_alpha(remaining=0, initial=4) == 0


def test_facing_rotation_degrees_match_movement_actions() -> None:
    assert facing_rotation_degrees(2) == 0
    assert facing_rotation_degrees(1) == 90
    assert facing_rotation_degrees(4) == 180
    assert facing_rotation_degrees(3) == -90
