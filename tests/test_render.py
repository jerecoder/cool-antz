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
    _compile_jax_action_selector,
    _deterministic_from_temperature,
    _env_from_args,
    _jax_render_food_scale,
    _jax_render_reset_options,
    _render_frame,
    _render_frame_limit,
    _render_step_count,
    _target_critic_architecture,
    render_checkpoint,
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


def test_render_frame_limit_keeps_total_video_budget() -> None:
    args = argparse.Namespace(max_steps=10)

    assert _render_frame_limit(args, max_frames=None) == 11
    assert _render_frame_limit(args, max_frames=1) == 1
    assert _render_frame_limit(args, max_frames=4) == 4

    with pytest.raises(ValueError, match="max_frames"):
        _render_frame_limit(args, max_frames=0)


def test_render_temperature_zero_is_deterministic_and_positive_samples() -> None:
    assert _deterministic_from_temperature(0.0)
    assert not _deterministic_from_temperature(0.5)
    with pytest.raises(ValueError, match="non-negative"):
        _deterministic_from_temperature(-0.5)


def test_jax_render_target_critic_architecture_defaults_to_mlp() -> None:
    assert _target_critic_architecture({}) == "mlp"
    assert _target_critic_architecture({"critic_architecture": "strided_cnn"}) == (
        "strided_cnn"
    )


def test_jax_render_action_selector_passes_critic_kwargs() -> None:
    class FakeJax:
        @staticmethod
        def jit(fn):
            return fn

    captured: dict[str, object] = {}

    def build_central_observations(obs_batch, **kwargs):
        captured["central_kwargs"] = kwargs
        return "central"

    def build_actor_observations(obs_batch, **kwargs):
        captured["actor_kwargs"] = kwargs
        return "actor"

    def get_action_and_value(
        params,
        actor_obs,
        central_obs,
        action_key,
        *,
        deterministic,
        **kwargs,
    ):
        captured["call"] = (
            params,
            actor_obs,
            central_obs,
            action_key,
            deterministic,
            kwargs,
        )
        return "actions", None, None, None

    selector = _compile_jax_action_selector(
        args=argparse.Namespace(
            write_bits=4,
            obs_width=50,
            obs_height=50,
            actor_vision_radius=2,
        ),
        params="params",
        deterministic=False,
        build_actor_observations=build_actor_observations,
        build_central_observations=build_central_observations,
        get_action_and_value=get_action_and_value,
        jax=FakeJax(),
        food_scale=125.0,
        critic_kwargs={
            "critic_architecture": "strided_cnn",
            "critic_num_ants": 4,
            "critic_obs_height": 50,
            "critic_obs_width": 50,
        },
    )

    assert selector({"obs": object()}, "key") == "actions"
    assert captured["call"] == (
        "params",
        "actor",
        "central",
        "key",
        False,
        {
            "critic_architecture": "strided_cnn",
            "critic_num_ants": 4,
            "critic_obs_height": 50,
            "critic_obs_width": 50,
        },
    )


def test_render_jax_checkpoint_uses_saved_critic_architecture(tmp_path: Path) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from ant_byte_env import write_value_count
    from ant_byte_env.jax_env import JaxAntByteForagingEnv
    from ant_byte_env.training.jax_mappo import (
        build_actor_observations,
        build_central_observations,
        init_adam_state,
        init_agent_params,
        parse_args,
        save_checkpoint,
    )

    args = parse_args(
        [
            "--critic-architecture",
            "strided_cnn",
            "--width",
            "8",
            "--height",
            "8",
            "--num-ants",
            "2",
            "--food-count",
            "2",
            "--food-sources",
            "1",
            "--cookie-distance",
            "2",
            "--max-steps",
            "4",
            "--write-bits",
            "2",
            "--hidden-size",
            "8",
            "--quiet",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        write_bits=args.write_bits,
    )
    _, obs, _ = env.reset(jax.random.PRNGKey(args.seed))
    obs_batch = {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}
    central_obs = build_central_observations(
        obs_batch,
        food_scale=args.food_count,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs_batch,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
        critic_architecture=args.critic_architecture,
        critic_num_ants=args.num_ants,
        critic_obs_height=args.obs_height or args.height,
        critic_obs_width=args.obs_width or args.width,
    )
    checkpoint_path = tmp_path / "strided_cnn.pkl"
    save_checkpoint(
        checkpoint_path,
        params=params,
        opt_state=init_adam_state(params),
        args=args,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        run_name="test",
        metrics={},
    )

    output_path = render_checkpoint(
        checkpoint_path,
        tmp_path / "rollout.mp4",
        backend="jax",
        max_frames=2,
        tile_size=4,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_jax_render_reset_options_leave_random_layout_to_env() -> None:
    args = argparse.Namespace(
        width=7,
        height=6,
        cookie_distance=2,
        random_food=True,
        random_hub=True,
    )

    assert _jax_render_reset_options(args, seed=123) is None

    args.random_food = False
    assert _jax_render_reset_options(args, seed=123) is None


def test_jax_render_reset_options_fix_nonrandom_layout() -> None:
    args = argparse.Namespace(
        width=7,
        height=6,
        cookie_distance=2,
        random_food=False,
        random_hub=False,
    )

    options = _jax_render_reset_options(args, seed=123)

    assert options == {"hub_pos": (3, 3), "food_positions": [(5, 3)]}


def test_jax_render_food_scale_uses_food_per_source() -> None:
    dense_args = argparse.Namespace(food_count=250, food_sources=250)
    sparse_args = argparse.Namespace(food_count=250, food_sources=2)
    legacy_args = argparse.Namespace(food_count=250)

    assert _jax_render_food_scale(dense_args) == 1.0
    assert _jax_render_food_scale(sparse_args) == 125.0
    assert _jax_render_food_scale(legacy_args) == 250.0


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


def test_env_from_args_passes_hub_center_window_size() -> None:
    args = argparse.Namespace(
        width=50,
        height=50,
        num_ants=2,
        food_count=0,
        food_sources=1,
        max_steps=12,
        random_food=True,
        random_hub=True,
        layout_margin=10,
        hub_center_window_size=4,
        random_wall_obstacles=True,
        random_wall_center_window_size=12,
        step_penalty=0.0,
        write_penalty=0.0,
        write_bits=1,
    )

    env = _env_from_args(args, render_mode="rgb_array", tile_size=12)
    obs, _ = env.reset(seed=3)
    hub_x, hub_y = obs["hub_pos"]

    assert env.hub_center_window_size == 4
    assert env.random_wall_obstacles is True
    assert env.random_wall_center_window_size == 12
    assert 23 <= int(hub_x) < 27
    assert 23 <= int(hub_y) < 27
    env.close()


def test_env_from_args_rejects_unsupported_jax_env_modes() -> None:
    base_args = dict(
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

    with pytest.raises(ValueError, match="distance-autocurriculum"):
        _env_from_args(
            argparse.Namespace(**base_args, distance_autocurriculum=True),
            render_mode="rgb_array",
        )
    with pytest.raises(ValueError, match="lethal-food"):
        _env_from_args(
            argparse.Namespace(**base_args, lethal_food_count=1),
            render_mode="rgb_array",
        )


def test_render_cli_passes_render_policy_flags(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_render_checkpoint(*args, reuse_existing: bool, **kwargs) -> Path:
        del args
        captured["reuse_existing"] = reuse_existing
        captured["seed_offset"] = kwargs["seed_offset"]
        captured["policy_temperature"] = kwargs["policy_temperature"]
        captured["action_mode"] = kwargs["action_mode"]
        captured["move_temperature"] = kwargs["move_temperature"]
        captured["write_temperature"] = kwargs["write_temperature"]
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
            "--seed-offset",
            "4600000",
            "--policy-temperature",
            "1.0",
            "--action-mode",
            "sampled_move_greedy_write",
            "--move-temperature",
            "0.95",
            "--write-temperature",
            "1.0",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "reuse_existing": True,
        "seed_offset": 4600000,
        "policy_temperature": 1.0,
        "action_mode": "sampled_move_greedy_write",
        "move_temperature": 0.95,
        "write_temperature": 1.0,
    }


def test_render_checkpoint_rejects_torch_action_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only supported for JAX"):
        render_checkpoint(
            tmp_path / "model.pt",
            tmp_path / "rollout.mp4",
            backend="torch",
            action_mode="sampled_move_greedy_write",
        )


def test_food_alpha_drops_as_food_source_is_depleted() -> None:
    assert food_alpha(remaining=4, initial=4) > food_alpha(remaining=2, initial=4)
    assert food_alpha(remaining=2, initial=4) > food_alpha(remaining=1, initial=4)
    assert food_alpha(remaining=0, initial=4) == 0


def test_facing_rotation_degrees_match_movement_actions() -> None:
    assert facing_rotation_degrees(2) == 0
    assert facing_rotation_degrees(1) == 90
    assert facing_rotation_degrees(4) == 180
    assert facing_rotation_degrees(3) == -90
