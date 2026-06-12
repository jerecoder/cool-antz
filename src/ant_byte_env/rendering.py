"""Checkpoint rendering helpers for CLI and notebooks."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from ant_byte_env import DEFAULT_WRITE_BITS, AntByteForagingEnv
from ant_byte_env.visualization import draw_vision_squares


def infer_checkpoint_backend(checkpoint_path: Path) -> str:
    suffix = Path(checkpoint_path).suffix.lower()
    if suffix == ".pt":
        return "torch"
    if suffix in {".pkl", ".pickle"}:
        return "jax"
    raise ValueError(f"cannot infer checkpoint backend from suffix: {checkpoint_path}")


def render_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    backend: str | None = None,
    seed_offset: int = 100_000,
    show_vision: bool = True,
) -> Path:
    actual_backend = backend or infer_checkpoint_backend(checkpoint_path)
    if actual_backend == "torch":
        return render_torch_checkpoint(
            checkpoint_path,
            output_path,
            seed_offset=seed_offset,
            show_vision=show_vision,
        )
    if actual_backend == "jax":
        return render_jax_checkpoint(
            checkpoint_path,
            output_path,
            seed_offset=seed_offset,
            show_vision=show_vision,
        )
    raise ValueError("backend must be 'torch' or 'jax'.")


def render_torch_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    seed_offset: int = 100_000,
    show_vision: bool = True,
) -> Path:
    import torch

    from ant_byte_env.training.torch_mappo import (
        MAPPOAgent,
        build_actor_observations,
        build_central_observations,
        build_curriculum_reset_options,
        flatten_agent_actions,
        obs_to_tensor,
        write_value_count,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = argparse.Namespace(**checkpoint["args"])
    if not hasattr(args, "write_bits"):
        args.write_bits = DEFAULT_WRITE_BITS

    agent = MAPPOAgent(
        central_obs_dim=int(checkpoint["central_obs_dim"]),
        actor_obs_dim=int(checkpoint["actor_obs_dim"]),
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    ).to(device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    env = _env_from_args(args, render_mode="rgb_array")
    writer = imageio.get_writer(output_path, fps=AntByteForagingEnv.metadata["render_fps"])
    try:
        obs, _ = env.reset(
            seed=args.seed + seed_offset,
            options=build_curriculum_reset_options(args, seed=args.seed + seed_offset),
        )
        writer.append_data(_render_frame(env, obs, args=args, show_vision=show_vision))
        for _ in range(args.max_steps):
            obs_batch = {key: value[np.newaxis, ...] for key, value in obs.items()}
            obs_tensor = obs_to_tensor(obs_batch, device)
            central_obs = build_central_observations(
                obs_tensor,
                food_scale=args.food_count,
                write_bits=args.write_bits,
                obs_width=args.obs_width,
                obs_height=args.obs_height,
            )
            actor_obs = build_actor_observations(
                obs_tensor,
                central_obs,
                food_scale=args.food_count,
                actor_vision_radius=args.actor_vision_radius,
                write_bits=args.write_bits,
                obs_width=args.obs_width,
                obs_height=args.obs_height,
            )
            with torch.no_grad():
                actions, _, _, _ = agent.get_action_and_value(
                    actor_obs,
                    central_obs,
                    deterministic=True,
                )
            obs, _, terminated, truncated, _ = env.step(flatten_agent_actions(actions).cpu().numpy()[0])
            writer.append_data(_render_frame(env, obs, args=args, show_vision=show_vision))
            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()
    return output_path


def render_jax_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    seed_offset: int = 100_000,
    show_vision: bool = True,
) -> Path:
    import jax
    import jax.numpy as jnp

    from ant_byte_env.training.jax_mappo import (
        build_actor_observations,
        build_central_observations,
        get_action_and_value,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(checkpoint_path).open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    args = argparse.Namespace(**checkpoint["args"])
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])

    env = _env_from_args(args, render_mode="rgb_array")
    writer = imageio.get_writer(output_path, fps=AntByteForagingEnv.metadata["render_fps"])
    try:
        obs, _ = env.reset(seed=args.seed + seed_offset, options=_jax_render_reset_options(args))
        writer.append_data(_render_frame(env, obs, args=args, show_vision=show_vision))
        key = jax.random.PRNGKey(args.seed + seed_offset)
        for _ in range(args.max_steps):
            key, action_key = jax.random.split(key)
            obs_batch = {name: jnp.expand_dims(jnp.asarray(value), axis=0) for name, value in obs.items()}
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
            actions, _, _, _ = get_action_and_value(
                params,
                actor_obs,
                central_obs,
                action_key,
                deterministic=True,
            )
            obs, _, terminated, truncated, _ = env.step(np.asarray(actions).reshape(-1))
            writer.append_data(_render_frame(env, obs, args=args, show_vision=show_vision))
            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()
    return output_path


def _render_frame(
    env: AntByteForagingEnv,
    obs: dict[str, np.ndarray],
    *,
    args: argparse.Namespace,
    show_vision: bool,
) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError("rgb_array rendering unexpectedly returned None.")
    if not show_vision:
        return frame
    return draw_vision_squares(
        frame,
        obs,
        tile_size=env.tile_size,
        vision_radius=int(getattr(args, "actor_vision_radius", 0)),
    )


def _env_from_args(args: argparse.Namespace, *, render_mode: str) -> AntByteForagingEnv:
    return AntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        step_penalty=args.step_penalty,
        write_penalty=args.write_penalty,
        write_bits=args.write_bits,
        render_mode=render_mode,
    )


def _jax_render_reset_options(args: argparse.Namespace) -> dict[str, tuple[int, int]] | None:
    if args.random_food:
        return None
    hub = (args.width // 2, args.height // 2)
    distance = min(args.cookie_distance, max(args.width, args.height))
    candidates = (
        (hub[0] + distance, hub[1]),
        (hub[0] - distance, hub[1]),
        (hub[0], hub[1] + distance),
        (hub[0], hub[1] - distance),
    )
    for candidate in candidates:
        if 0 <= candidate[0] < args.width and 0 <= candidate[1] < args.height:
            return {"hub_pos": hub, "food_positions": [candidate]}
    return {"hub_pos": hub}
