"""Checkpoint rendering helpers for CLI and notebooks."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np

from ant_byte_env import (
    DEFAULT_ACTOR_VISION_DEPTH,
    DEFAULT_WRITE_BITS,
    AntByteAutoCurriculumEnv,
    AntByteForagingEnv,
    MOVEMENT_ACTION_COUNT,
    actor_vision_patch_size,
)
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
    reuse_existing: bool = False,
    max_frames: int | None = None,
    tile_size: int | None = None,
    policy_temperature: float = 0.0,
) -> Path:
    actual_backend = backend or infer_checkpoint_backend(checkpoint_path)
    if actual_backend not in {"torch", "jax"}:
        raise ValueError("backend must be 'torch' or 'jax'.")
    if actual_backend == "torch" and action_mode is not None:
        raise ValueError("action_mode rendering is only supported for JAX checkpoints.")
    if _can_reuse_render(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        reuse_existing=reuse_existing,
    ):
        return output_path

    if actual_backend == "torch":
        return render_torch_checkpoint(
            checkpoint_path,
            output_path,
            seed_offset=seed_offset,
            show_vision=show_vision,
            reuse_existing=reuse_existing,
            max_frames=max_frames,
            tile_size=tile_size,
            policy_temperature=policy_temperature,
        )
    if actual_backend == "jax":
        return render_jax_checkpoint(
            checkpoint_path,
            output_path,
            seed_offset=seed_offset,
            show_vision=show_vision,
            reuse_existing=reuse_existing,
            max_frames=max_frames,
            tile_size=tile_size,
            policy_temperature=policy_temperature,
            action_mode=action_mode,
            move_temperature=move_temperature,
            write_temperature=write_temperature,
        )
    raise ValueError("backend must be 'torch' or 'jax'.")


def render_torch_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    seed_offset: int = 100_000,
    show_vision: bool = True,
    reuse_existing: bool = False,
    max_frames: int | None = None,
    tile_size: int | None = None,
    policy_temperature: float = 0.0,
    action_mode: str | None = None,
    move_temperature: float = 1.0,
    write_temperature: float = 1.0,
) -> Path:
    if _can_reuse_render(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        reuse_existing=reuse_existing,
    ):
        return output_path

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
    from ant_byte_env.training.torch_mappo.checkpointing import (
        adapt_agent_state_dict_for_actor_window,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = argparse.Namespace(**checkpoint["args"])
    if not hasattr(args, "write_bits"):
        args.write_bits = DEFAULT_WRITE_BITS
    if not hasattr(args, "actor_vision_radius"):
        args.actor_vision_radius = DEFAULT_ACTOR_VISION_DEPTH
    step_count = _render_step_count(args, max_frames=max_frames)
    actor_obs_dim = _actor_obs_dim_from_args(args)
    deterministic = _deterministic_from_temperature(policy_temperature)

    agent = MAPPOAgent(
        central_obs_dim=int(checkpoint["central_obs_dim"]),
        actor_obs_dim=actor_obs_dim,
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    ).to(device)
    agent.load_state_dict(
        adapt_agent_state_dict_for_actor_window(
            checkpoint["agent_state_dict"],
            saved_actor_dim=int(checkpoint["actor_obs_dim"]),
            actor_obs_dim=actor_obs_dim,
            write_bits=args.write_bits,
            actor_vision_radius=args.actor_vision_radius,
        )
    )
    agent.eval()

    env = _env_from_args(args, render_mode="rgb_array", tile_size=tile_size)
    writer = imageio.get_writer(output_path, fps=AntByteForagingEnv.metadata["render_fps"])
    try:
        obs, _ = env.reset(
            seed=args.seed + seed_offset,
            options=build_curriculum_reset_options(args, seed=args.seed + seed_offset),
        )
        writer.append_data(_render_frame(env, obs, args=args, show_vision=show_vision))
        with torch.inference_mode():
            for _ in range(step_count):
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
                actions, _, _, _ = agent.get_action_and_value(
                    actor_obs,
                    central_obs,
                    deterministic=deterministic,
                )
                obs, _, terminated, truncated, _ = env.step(
                    flatten_agent_actions(actions).cpu().numpy()[0]
                )
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
    reuse_existing: bool = False,
    max_frames: int | None = None,
    tile_size: int | None = None,
    policy_temperature: float = 0.0,
    action_mode: str | None = None,
    move_temperature: float = 1.0,
    write_temperature: float = 1.0,
) -> Path:
    if _can_reuse_render(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        reuse_existing=reuse_existing,
    ):
        return output_path

    import jax
    import jax.numpy as jnp

    from ant_byte_env.training.jax_mappo import (
        build_actor_observations,
        build_central_observations,
        get_action_and_value,
    )
    from ant_byte_env.training.jax_mappo.evaluation import (
        _evaluation_actions_for_mode,
        validate_evaluation_action_mode,
    )
    from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(checkpoint_path).open("rb") as checkpoint_file:
        raw_checkpoint = pickle.load(checkpoint_file)
    args = argparse.Namespace(**raw_checkpoint["args"])
    step_count = _render_step_count(args, max_frames=max_frames)
    deterministic = _deterministic_from_temperature(policy_temperature)
    resolved_action_mode = (
        None if action_mode is None else validate_evaluation_action_mode(action_mode)
    )

    env = _env_from_args(args, render_mode="rgb_array", tile_size=tile_size)
    writer = imageio.get_writer(output_path, fps=AntByteForagingEnv.metadata["render_fps"])
    try:
        reset_seed = args.seed + seed_offset
        obs, _ = env.reset(
            seed=reset_seed,
            options=_jax_render_reset_options(args, seed=reset_seed),
        )
        obs_batch = {
            name: jnp.expand_dims(jnp.asarray(value), axis=0)
            for name, value in obs.items()
        }
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
        checkpoint = load_checkpoint_for_training(
            Path(checkpoint_path),
            central_obs_dim=int(central_obs.shape[-1]),
            actor_obs_dim=int(actor_obs.shape[-1]),
            target_write_bits=args.write_bits,
            actor_vision_radius=args.actor_vision_radius,
        )
        params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
        select_action = _compile_jax_action_selector(
            args=args,
            params=params,
            deterministic=deterministic,
            build_actor_observations=build_actor_observations,
            build_central_observations=build_central_observations,
            get_action_and_value=get_action_and_value,
            evaluation_actions_for_mode=_evaluation_actions_for_mode,
            jax=jax,
            action_mode=resolved_action_mode,
            move_temperature=move_temperature,
            write_temperature=write_temperature,
        )
        writer.append_data(_render_frame(env, obs, args=args, show_vision=show_vision))
        key = jax.random.PRNGKey(args.seed + seed_offset)
        for _ in range(step_count):
            key, action_key = jax.random.split(key)
            obs_batch = {
                name: jnp.expand_dims(jnp.asarray(value), axis=0)
                for name, value in obs.items()
            }
            actions = select_action(obs_batch, action_key)
            obs, _, terminated, truncated, _ = env.step(np.asarray(actions).reshape(-1))
            writer.append_data(_render_frame(env, obs, args=args, show_vision=show_vision))
            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
    return output_path


def _can_reuse_render(
    *,
    checkpoint_path: Path,
    output_path: Path,
    reuse_existing: bool,
) -> bool:
    if not reuse_existing or not output_path.exists() or output_path.stat().st_size <= 0:
        return False
    return output_path.stat().st_mtime >= Path(checkpoint_path).stat().st_mtime


def _render_step_count(args: argparse.Namespace, *, max_frames: int | None) -> int:
    max_steps = int(args.max_steps)
    if max_frames is None:
        return max_steps
    frame_count = int(max_frames)
    if frame_count < 1:
        raise ValueError("max_frames must be at least 1.")
    return min(max_steps, frame_count - 1)


def _deterministic_from_temperature(policy_temperature: float) -> bool:
    temperature = float(policy_temperature)
    if temperature < 0.0:
        raise ValueError("policy_temperature must be non-negative.")
    return temperature == 0.0


def _actor_obs_dim_from_args(args: argparse.Namespace) -> int:
    patch_size = actor_vision_patch_size(int(args.actor_vision_radius))
    return patch_size * (int(args.write_bits) + 4) + MOVEMENT_ACTION_COUNT


def _compile_jax_action_selector(
    *,
    args: argparse.Namespace,
    params: Any,
    deterministic: bool,
    build_actor_observations: Callable[..., Any],
    build_central_observations: Callable[..., Any],
    get_action_and_value: Callable[..., Any],
    evaluation_actions_for_mode: Callable[..., Any] | None = None,
    jax: Any,
    action_mode: str | None = None,
    move_temperature: float = 1.0,
    write_temperature: float = 1.0,
) -> Callable[..., Any]:
    @jax.jit
    def select_action(obs_batch: dict[str, Any], action_key: Any) -> Any:
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
        if action_mode is None:
            actions, _, _, _ = get_action_and_value(
                params,
                actor_obs,
                central_obs,
                action_key,
                deterministic=deterministic,
            )
            return actions
        if evaluation_actions_for_mode is None:
            raise ValueError(
                "evaluation action selector is required for action_mode rendering."
            )
        return evaluation_actions_for_mode(
            params,
            actor_obs,
            central_obs,
            action_key,
            action_mode=action_mode,
            move_temperature=move_temperature,
            write_temperature=write_temperature,
        )

    return select_action


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


def _env_from_args(
    args: argparse.Namespace,
    *,
    render_mode: str,
    tile_size: int | None = None,
) -> AntByteForagingEnv | AntByteAutoCurriculumEnv:
    if bool(getattr(args, "autocurriculum", False)):
        env_kwargs: dict[str, Any] = {
            "start_size": int(getattr(args, "autocurriculum_start_size", 4)),
            "max_size": args.width,
            "cookies_per_stage": int(getattr(args, "autocurriculum_success_cookies", 6)),
            "num_ants": args.num_ants,
            "food_count": args.food_count,
            "food_source_count": args.food_sources,
            "max_steps": args.max_steps,
            "random_food": args.random_food,
            "random_hub": bool(getattr(args, "random_hub", False)),
            "step_penalty": args.step_penalty,
            "write_penalty": args.write_penalty,
            "write_bits": args.write_bits,
            "write_while_moving": bool(getattr(args, "write_while_moving", False)),
            "actor_vision_radius": int(getattr(args, "actor_vision_radius", 1)),
            "render_mode": render_mode,
        }
        if tile_size is not None:
            env_kwargs["tile_size"] = int(tile_size)
        return AntByteAutoCurriculumEnv(**env_kwargs)

    env_kwargs: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "num_ants": args.num_ants,
        "food_count": args.food_count,
        "food_source_count": args.food_sources,
        "max_steps": args.max_steps,
        "random_food": args.random_food,
        "random_hub": bool(getattr(args, "random_hub", False)),
        "step_penalty": args.step_penalty,
        "write_penalty": args.write_penalty,
        "write_bits": args.write_bits,
        "write_while_moving": bool(getattr(args, "write_while_moving", False)),
        "render_mode": render_mode,
    }
    if tile_size is not None:
        env_kwargs["tile_size"] = int(tile_size)
    return AntByteForagingEnv(**env_kwargs)


def _jax_render_reset_options(
    args: argparse.Namespace,
    *,
    seed: int | None = None,
) -> dict[str, tuple[int, int] | list[tuple[int, int]]] | None:
    if bool(getattr(args, "autocurriculum", False)):
        return None
    if bool(getattr(args, "random_hub", False)):
        rng = np.random.default_rng(seed)
        hub = (
            int(rng.integers(0, args.width)),
            int(rng.integers(0, args.height)),
        )
    else:
        hub = (args.width // 2, args.height // 2)
    if args.random_food:
        return {"hub_pos": hub} if bool(getattr(args, "random_hub", False)) else None
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
