"""MP4 rendering for timed-release cooperative JAX MAPPO checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env.env import (
    AntByteForagingEnv,
    CARRIED_FOOD_COLOR,
    CARRIED_FOOD_HIGHLIGHT,
    FOOD_COUNT_COLOR,
    OBSTACLE_COLOR,
    OBSTACLE_EDGE_COLOR,
    food_alpha,
    max_write_value,
    rotate_ant_sprite,
)
from ant_byte_env.sprites import load_sprites
from ant_byte_env.training.jax_mappo.curriculum import reset_batch
from ant_byte_env.training.jax_mappo.evaluation import (
    _evaluation_actions_for_mode,
    validate_evaluation_action_mode,
)
from ant_byte_env.training.jax_mappo.models import critic_forward_kwargs_from_args
from ant_byte_env.training.jax_mappo.observations import (
    build_actor_observations,
    build_central_observations,
    flatten_agent_actions,
    food_observation_scale,
)
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training
from ant_byte_env.training.jax_mappo.timed_release.evaluation import (
    _checkpoint_args_with_defaults,
    _checkpoint_observation_dims,
)
from ant_byte_env.training.jax_mappo.timed_release.env import (
    TimedReleaseJaxEnv,
    TimedReleaseState,
    make_timed_release_env,
)


RANK_COLORS = (
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (217, 119, 6),
    (147, 51, 234),
    (14, 116, 144),
    (225, 29, 72),
    (101, 163, 13),
)


def render_timed_release_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    seed_offset: int = 100_000,
    max_frames: int | None = 480,
    tile_size: int = 16,
    action_mode: str = "sampled_move_greedy_write",
    move_temperature: float = 0.75,
    write_temperature: float = 1.0,
) -> Path:
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("max_frames must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_checkpoint = read_checkpoint(Path(checkpoint_path))
    args = _checkpoint_args_with_defaults(raw_checkpoint.get("args", {}))
    frame_limit = int(args.max_steps) + 1 if max_frames is None else int(max_frames)
    resolved_action_mode = validate_evaluation_action_mode(action_mode)
    central_obs_dim, actor_obs_dim = _checkpoint_observation_dims(args)
    checkpoint = load_checkpoint_for_training(
        Path(checkpoint_path),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        target_write_bits=args.write_bits,
        actor_vision_radius=args.actor_vision_radius,
        target_num_ants=args.num_ants,
        target_agent_identity_types=getattr(args, "agent_identity_types", None),
        target_critic_architecture=getattr(args, "critic_architecture", "mlp"),
    )
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
    env = make_timed_release_env(args)
    eval_args = type(args)(**{**vars(args), "num_envs": 1})
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    step_fn = jax.jit(
        lambda state, obs, step_key: _render_step(
            env=env,
            args=eval_args,
            params=params,
            state=state,
            obs=obs,
            key=step_key,
            food_scale=food_scale,
            action_mode=resolved_action_mode,
            move_temperature=float(move_temperature),
            write_temperature=float(write_temperature),
        )
    )
    key = jax.random.PRNGKey(int(args.seed) + int(seed_offset))
    key, reset_key = jax.random.split(key)
    state, obs = reset_batch(args=eval_args, env=env, key=reset_key)
    writer = imageio.get_writer(output_path, fps=AntByteForagingEnv.metadata["render_fps"])
    try:
        writer.append_data(
            draw_timed_release_frame(
                obs,
                state=state,
                tile_size=tile_size,
                write_bits=int(args.write_bits),
            )
        )
        frames_written = 1
        while frames_written < frame_limit:
            key, step_key = jax.random.split(key)
            state, obs, _, terminated, truncated, _ = step_fn(state, obs, step_key)
            writer.append_data(
                draw_timed_release_frame(
                    obs,
                    state=state,
                    tile_size=tile_size,
                    write_bits=int(args.write_bits),
                )
            )
            frames_written += 1
            if bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0]):
                break
    finally:
        writer.close()
    return output_path


def _render_step(
    *,
    env: TimedReleaseJaxEnv,
    args: Any,
    params: Any,
    state: TimedReleaseState,
    obs: dict[str, jax.Array],
    key: jax.Array,
    food_scale: float,
    action_mode: str,
    move_temperature: float,
    write_temperature: float,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    central_obs = build_central_observations(
        obs,
        food_scale=food_scale,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        agent_identity_types=getattr(args, "agent_identity_types", None),
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actions = _evaluation_actions_for_mode(
        params,
        actor_obs,
        central_obs,
        key,
        action_mode=action_mode,
        move_temperature=move_temperature,
        write_temperature=write_temperature,
        **critic_forward_kwargs_from_args(args),
    )
    state, obs, reward, terminated, truncated, infos = jax.vmap(env.step)(
        state,
        flatten_agent_actions(actions),
    )
    return state, obs, reward, terminated, truncated, infos


def draw_timed_release_frame(
    obs: dict[str, Any],
    *,
    state: Any | None = None,
    tile_size: int = 16,
    write_bits: int = 1,
) -> np.ndarray:
    frame_obs = {name: np.asarray(value)[0] for name, value in obs.items()}
    food = frame_obs["food"]
    bytes_grid = frame_obs["bytes"]
    obstacles = frame_obs.get("obstacles", np.zeros_like(food, dtype=np.int8))
    hub_pos = frame_obs["hub_pos"]
    ants_pos = frame_obs["ants_pos"]
    ants_facing = frame_obs["ants_facing"]
    ants_carrying = frame_obs["ants_carrying"].astype(bool)
    active_mask = frame_obs.get("active_mask", np.ones((ants_pos.shape[0],), dtype=bool))
    active_mask = active_mask.astype(bool)
    initial_food = food
    delivered = 0
    step_count = 0
    if state is not None:
        initial_food = np.asarray(state.base.initial_food)[0]
        delivered = int(np.asarray(state.base.delivered_food)[0])
        step_count = int(np.asarray(state.base.step_count)[0])

    height, width = food.shape
    canvas = _pygame_surface((width * tile_size, height * tile_size))
    import pygame

    sprites = load_sprites(tile_size)
    font = pygame.font.Font(None, max(10, tile_size // 2))
    label_font = pygame.font.Font(None, max(14, int(tile_size * 0.65)))
    for y_pos in range(height):
        for x_pos in range(width):
            rect = pygame.Rect(x_pos * tile_size, y_pos * tile_size, tile_size, tile_size)
            canvas.blit(sprites["tile"], rect.topleft)
            if bool(obstacles[y_pos, x_pos]):
                pygame.draw.rect(canvas, OBSTACLE_COLOR, rect)
                pygame.draw.rect(canvas, OBSTACLE_EDGE_COLOR, rect, max(1, tile_size // 14))
                continue
            byte_value = int(bytes_grid[y_pos, x_pos])
            if byte_value > 0:
                ratio = byte_value / max(float(max_write_value(write_bits)), 1.0)
                overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
                overlay.fill((int(40 + 180 * ratio), 92, int(255 - 120 * ratio), 96))
                canvas.blit(overlay, rect.topleft)
                byte_label = font.render(str(byte_value), True, (24, 31, 36))
                canvas.blit(byte_label, (rect.x + 2, rect.y + 1))

    _blit_tile(canvas, sprites["hub"], hub_pos, tile_size)
    for y_pos in range(height):
        for x_pos in range(width):
            food_amount = int(food[y_pos, x_pos])
            if food_amount <= 0:
                continue
            food_sprite = sprites["food"].copy()
            food_sprite.set_alpha(food_alpha(food_amount, int(initial_food[y_pos, x_pos])))
            _blit_tile(canvas, food_sprite, np.array([x_pos, y_pos]), tile_size)
            if food_amount > 1:
                count_label = font.render(str(food_amount), True, FOOD_COUNT_COLOR)
                canvas.blit(
                    count_label,
                    (
                        x_pos * tile_size + tile_size // 2,
                        y_pos * tile_size + tile_size // 2,
                    ),
                )

    for rank, active in enumerate(active_mask):
        if not active:
            continue
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ant_sprite = rotate_ant_sprite(sprites["ant"], int(ants_facing[rank]))
        _blit_tile(canvas, ant_sprite, ants_pos[rank], tile_size)
        x_pos, y_pos = int(ants_pos[rank][0]), int(ants_pos[rank][1])
        badge_radius = max(4, tile_size // 4)
        badge_center = (
            x_pos * tile_size + badge_radius + 1,
            y_pos * tile_size + badge_radius + 1,
        )
        pygame.draw.circle(canvas, color, badge_center, badge_radius)
        rank_label = label_font.render(str(rank), True, (255, 255, 255))
        canvas.blit(rank_label, rank_label.get_rect(center=badge_center))
        if ants_carrying[rank]:
            center = (
                x_pos * tile_size + 3 * tile_size // 4,
                y_pos * tile_size + tile_size // 4,
            )
            pygame.draw.circle(canvas, CARRIED_FOOD_COLOR, center, max(3, tile_size // 7))
            pygame.draw.circle(
                canvas,
                CARRIED_FOOD_HIGHLIGHT,
                (center[0] - max(1, tile_size // 16), center[1] - max(1, tile_size // 16)),
                max(2, tile_size // 12),
            )

    active_count = int(np.sum(active_mask))
    panel_text = f"step {step_count}   active {active_count}/{len(active_mask)}   delivered {delivered}"
    panel_label = label_font.render(panel_text, True, (24, 31, 36))
    pad = max(4, tile_size // 4)
    panel = pygame.Surface(
        (panel_label.get_width() + 2 * pad, panel_label.get_height() + 2 * pad),
        pygame.SRCALPHA,
    )
    pygame.draw.rect(panel, (250, 248, 239, 232), panel.get_rect(), border_radius=4)
    pygame.draw.rect(panel, (54, 48, 38, 210), panel.get_rect(), 1, border_radius=4)
    panel.blit(panel_label, (pad, pad))
    canvas.blit(panel, (pad, pad))

    return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2)).copy()


def _pygame_surface(size: tuple[int, int]) -> Any:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    if not pygame.get_init():
        pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    return pygame.Surface(size)


def _blit_tile(
    canvas: Any,
    surface: Any,
    position: np.ndarray,
    tile_size: int,
) -> None:
    x_pos, y_pos = int(position[0]), int(position[1])
    canvas.blit(surface, (x_pos * tile_size, y_pos * tile_size))
