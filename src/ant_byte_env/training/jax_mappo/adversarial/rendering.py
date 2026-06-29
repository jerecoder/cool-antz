"""Rendering helpers for adversarial frozen-opponent rollouts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import jax
import numpy as np

from ant_byte_env.env import (
    AntByteForagingEnv,
    CARRIED_FOOD_COLOR,
    CARRIED_FOOD_HIGHLIGHT,
    FOOD_COUNT_COLOR,
    OBSTACLE_COLOR,
    OBSTACLE_EDGE_COLOR,
    food_alpha,
    rotate_ant_sprite,
)
from ant_byte_env.sprites import load_sprites
from ant_byte_env.training.jax_mappo.adversarial.actions import actions_from_logits
from ant_byte_env.training.jax_mappo.adversarial.checkpointing import (
    load_checkpoint_for_evaluation,
)
from ant_byte_env.training.jax_mappo.adversarial.env import reset_batch
from ant_byte_env.training.jax_mappo.adversarial.observations import (
    build_team_actor_observations,
)
from ant_byte_env.training.jax_mappo.adversarial.rollout import compose_team_actions
from ant_byte_env.training.jax_mappo.models import get_action_logits
from ant_byte_env.training.jax_mappo.observations import (
    flatten_agent_actions,
    food_observation_scale,
)

import pygame

TEAM_COLORS = ((37, 99, 235), (220, 38, 38))


def draw_adversarial_frame(
    obs: Mapping[str, Any],
    *,
    tile_size: int = 22,
    max_write_value: int = 1,
) -> np.ndarray:
    food = np.asarray(obs["food"])
    bytes_grid = np.asarray(obs["bytes"])
    hubs = np.asarray(obs["hub_pos"])
    ants = np.asarray(obs["ants_pos"])
    carrying = np.asarray(obs["ants_carrying"]).astype(bool)
    facing = np.asarray(obs.get("ants_facing", np.zeros((len(ants),), dtype=np.int32)))
    obstacles = np.asarray(obs.get("obstacles", np.zeros_like(food, dtype=bool))).astype(bool)
    initial_food = np.asarray(obs.get("initial_food", food))
    height, width = food.shape

    pygame.init()
    pygame.font.init()
    canvas = pygame.Surface((width * tile_size, height * tile_size))
    sprites = load_sprites(tile_size)
    font = pygame.font.Font(None, max(10, tile_size // 2))

    canvas.fill((215, 207, 181))

    for y in range(height):
        for x in range(width):
            rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
            canvas.blit(sprites["tile"], rect)
            if obstacles[y, x]:
                pygame.draw.rect(canvas, OBSTACLE_COLOR, rect)
                pygame.draw.rect(
                    canvas,
                    OBSTACLE_EDGE_COLOR,
                    rect,
                    max(1, tile_size // 14),
                )
                continue
            _draw_byte_overlay(
                canvas,
                font,
                rect,
                int(bytes_grid[y, x]),
                max_write_value=max_write_value,
            )

    for team, position in enumerate(hubs):
        _draw_team_hub(canvas, sprites["hub"], np.asarray(position), team, tile_size)

    for y in range(height):
        for x in range(width):
            food_amount = int(food[y, x])
            if food_amount <= 0:
                continue
            initial_amount = int(initial_food[y, x])
            food_sprite = sprites["food"].copy()
            food_sprite.set_alpha(food_alpha(food_amount, initial_amount))
            canvas.blit(food_sprite, (x * tile_size, y * tile_size))
            if food_amount > 1:
                label = font.render(str(food_amount), True, FOOD_COUNT_COLOR)
                canvas.blit(
                    label,
                    (
                        x * tile_size + tile_size // 2,
                        y * tile_size + tile_size // 2,
                    ),
                )

    ant_groups = _group_ants_by_position_and_team(ants)
    for (x, y, team), ant_indices in ant_groups.items():
        position = np.array([x, y], dtype=np.int32)
        _draw_team_ant(
            canvas,
            sprites["ant"],
            position,
            facing=int(facing[ant_indices[0]]),
            team=team,
            tile_size=tile_size,
        )
        if np.any(carrying[ant_indices]):
            _draw_carried_food_marker(canvas, position, tile_size)
        if len(ant_indices) > 1:
            _draw_stack_count_badge(
                canvas,
                font,
                position,
                count=len(ant_indices),
                team=team,
                tile_size=tile_size,
            )

    return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2)).copy()


def _group_ants_by_position_and_team(ants: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    ants_per_team = max(len(ants) // 2, 1)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for ant_index, (x_pos, y_pos) in enumerate(ants):
        team = min(ant_index // ants_per_team, 1)
        key = (int(x_pos), int(y_pos), team)
        groups.setdefault(key, []).append(ant_index)
    return groups


def _draw_byte_overlay(
    canvas: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    byte_value: int,
    *,
    max_write_value: int,
) -> None:
    if byte_value == 0:
        return

    ratio = byte_value / max(float(max_write_value), 1.0)
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    overlay.fill((int(40 + 180 * ratio), 92, int(255 - 120 * ratio), 96))
    canvas.blit(overlay, rect.topleft)
    label = font.render(str(byte_value), True, (24, 31, 36))
    canvas.blit(label, (rect.x + 2, rect.y + 1))


def _draw_team_hub(
    canvas: pygame.Surface,
    hub_sprite: pygame.Surface,
    position: np.ndarray,
    team: int,
    tile_size: int,
) -> None:
    color = TEAM_COLORS[team % len(TEAM_COLORS)]
    rect = _tile_rect(position, tile_size)
    pygame.draw.rect(
        canvas,
        color,
        rect.inflate(-2, -2),
        max(2, tile_size // 8),
        border_radius=3,
    )
    canvas.blit(hub_sprite, rect.topleft)
    pygame.draw.rect(
        canvas,
        color,
        rect.inflate(-4, -4),
        max(1, tile_size // 16),
        border_radius=3,
    )


def _draw_team_ant(
    canvas: pygame.Surface,
    ant_sprite: pygame.Surface,
    position: np.ndarray,
    *,
    facing: int,
    team: int,
    tile_size: int,
) -> None:
    color = TEAM_COLORS[team % len(TEAM_COLORS)]
    rect = _tile_rect(position, tile_size)
    center = rect.center
    pygame.draw.circle(
        canvas,
        color,
        center,
        max(4, tile_size // 3),
        max(2, tile_size // 10),
    )
    canvas.blit(rotate_ant_sprite(ant_sprite, facing), rect.topleft)


def _draw_carried_food_marker(
    canvas: pygame.Surface,
    position: np.ndarray,
    tile_size: int,
) -> None:
    x_pos, y_pos = int(position[0]), int(position[1])
    center = (
        x_pos * tile_size + 3 * tile_size // 4,
        y_pos * tile_size + tile_size // 4,
    )
    pygame.draw.circle(canvas, CARRIED_FOOD_COLOR, center, max(3, tile_size // 7))
    pygame.draw.circle(canvas, CARRIED_FOOD_HIGHLIGHT, center, max(1, tile_size // 12))


def _draw_stack_count_badge(
    canvas: pygame.Surface,
    font: pygame.font.Font,
    position: np.ndarray,
    *,
    count: int,
    team: int,
    tile_size: int,
) -> None:
    color = TEAM_COLORS[team % len(TEAM_COLORS)]
    x_pos, y_pos = int(position[0]), int(position[1])
    radius = max(4, tile_size // 5)
    center = (
        x_pos * tile_size + tile_size - radius - 1,
        y_pos * tile_size + tile_size - radius - 1,
    )
    pygame.draw.circle(canvas, color, center, radius)
    label = font.render(str(count), True, (255, 255, 255))
    label_rect = label.get_rect(center=center)
    canvas.blit(label, label_rect)


def _tile_rect(position: np.ndarray, tile_size: int) -> pygame.Rect:
    return pygame.Rect(
        int(position[0]) * tile_size,
        int(position[1]) * tile_size,
        tile_size,
        tile_size,
    )


def render_adversarial_rollout(
    checkpoint_path: Path,
    output_path: Path,
    *,
    argv: Sequence[str] | None = None,
    args: argparse.Namespace | None = None,
    max_frames: int | None = None,
    tile_size: int = 22,
    seed_offset: int = 900_000,
    deterministic: bool = True,
    action_mode: str | None = None,
) -> Path:
    bundle = load_checkpoint_for_evaluation(checkpoint_path, argv=argv, args=args)
    render_args = argparse.Namespace(**{**vars(bundle.args), "num_envs": 1})
    resolved_action_mode = action_mode or ("deterministic" if deterministic else "sampled")
    frame_limit = _frame_limit(render_args, max_frames=max_frames)
    key = jax.random.PRNGKey(int(render_args.seed) + int(seed_offset))
    states, obs = reset_batch(args=render_args, env=bundle.env, key=key)
    max_byte_value = (1 << int(render_args.write_bits)) - 1
    food_scale = food_observation_scale(
        food_count=render_args.food_count,
        food_sources=getattr(render_args, "food_sources", None),
    )
    learner_team = int(render_args.learner_team)
    opponent_team = 1 - learner_team

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=AntByteForagingEnv.metadata["render_fps"])
    try:
        writer.append_data(
            _frame_from_batched_obs(
                obs,
                states=states,
                tile_size=tile_size,
                max_write_value=max_byte_value,
            )
        )

        for _ in range(frame_limit - 1):
            key, learner_key, opponent_key = jax.random.split(key, 3)
            learner_actions = _model_actions(
                bundle.learner_params,
                obs,
                team=learner_team,
                key=learner_key,
                args=render_args,
                food_scale=food_scale,
                action_mode=resolved_action_mode,
            )
            opponent_actions = _model_actions(
                bundle.opponent_params,
                obs,
                team=opponent_team,
                key=opponent_key,
                args=render_args,
                food_scale=food_scale,
                action_mode=resolved_action_mode,
            )
            joint_actions = compose_team_actions(
                learner_actions,
                opponent_actions,
                learner_team=learner_team,
            )
            states, obs, _, terminated, truncated, _ = jax.vmap(bundle.env.step)(
                states,
                flatten_agent_actions(joint_actions),
            )
            writer.append_data(
                _frame_from_batched_obs(
                    obs,
                    states=states,
                    tile_size=tile_size,
                    max_write_value=max_byte_value,
                )
            )
            if bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0]):
                break
    finally:
        writer.close()
    return output_path


def _frame_limit(args: argparse.Namespace, *, max_frames: int | None) -> int:
    if max_frames is None:
        return int(args.max_steps) + 1
    frame_limit = int(max_frames)
    if frame_limit < 1:
        raise ValueError("max_frames must be at least 1.")
    return frame_limit


def _frame_from_batched_obs(
    obs: Mapping[str, Any],
    *,
    states: Any | None = None,
    tile_size: int,
    max_write_value: int = 1,
) -> np.ndarray:
    frame_obs = {name: np.asarray(value)[0] for name, value in obs.items()}
    if states is not None:
        frame_obs["initial_food"] = np.asarray(states.initial_food)[0]
    return draw_adversarial_frame(
        frame_obs,
        tile_size=tile_size,
        max_write_value=max_write_value,
    )


def _model_actions(
    params: Any,
    obs: Mapping[str, Any],
    *,
    team: int,
    key: jax.Array,
    args: argparse.Namespace,
    food_scale: float,
    action_mode: str,
) -> Any:
    actor_obs = build_team_actor_observations(
        obs,
        team=team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
    )
    move_logits, write_logits = get_action_logits(params, actor_obs)
    return actions_from_logits(
        move_logits,
        write_logits,
        key,
        action_mode=action_mode,
        move_temperature=float(args.training_rollout_temperature),
        write_temperature=float(args.training_rollout_temperature),
    )
