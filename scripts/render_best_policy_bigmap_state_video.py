#!/usr/bin/env python3
"""Render a best-policy big-map rollout and save replayable state data."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import imageio.v2 as imageio
import numpy as np

from ant_byte_env import MOVEMENT_ACTION_COUNT
from ant_byte_env.env import (
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    OLD_STYLE_BACKGROUND_COLOR,
    OLD_STYLE_CARRYING_ANT_COLOR,
    OLD_STYLE_COLONY_COLOR,
    OLD_STYLE_FOOD_CENTER_COLOR,
    OLD_STYLE_FOOD_OUTER_COLOR,
    OLD_STYLE_MARKER_CENTER_RADIUS_CELLS,
    OLD_STYLE_MARKER_RADIUS_CELLS,
    OLD_STYLE_NORMAL_ANT_COLOR,
)
from ant_byte_env.rendering import (
    _env_from_args,
    _jax_render_food_scale,
    _target_critic_architecture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="JAX checkpoint to render.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument(
        "--state-output",
        default=None,
        help="Compressed NPZ state archive. Defaults to OUTPUT with _states.npz.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="JSON sidecar path. Defaults to OUTPUT with .json suffix.",
    )
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--num-ants", type=int, default=None)
    parser.add_argument("--food-count", type=int, default=None)
    parser.add_argument("--food-sources", type=int, default=None)
    parser.add_argument("--inner-window-size", type=int, default=250)
    parser.add_argument("--layout-margin", type=int, default=None)
    parser.add_argument("--hub-center-window-size", type=int, default=None)
    parser.add_argument(
        "--hub-pos",
        default=None,
        help="Optional explicit hub position as x,y.",
    )
    parser.add_argument(
        "--food-position",
        action="append",
        default=None,
        help="Optional explicit food source as x,y. May be repeated.",
    )
    parser.add_argument("--seed-offset", type=int, default=214_000_000)
    parser.add_argument("--max-steps", type=int, default=120_000)
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--output-fps", type=int, default=160)
    parser.add_argument("--base-video-fps", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=1)
    parser.add_argument("--action-mode", default="sampled_move_greedy_write")
    parser.add_argument("--move-temperature", type=float, default=0.9)
    parser.add_argument("--write-temperature", type=float, default=1.0)
    parser.add_argument("--ffmpeg-crf", type=int, default=24)
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli_args = parse_args()
    checkpoint_path = _resolve_project_path(cli_args.checkpoint)
    output_path = _resolve_project_path(cli_args.output)
    state_output_path = (
        _resolve_project_path(cli_args.state_output)
        if cli_args.state_output
        else output_path.with_name(f"{output_path.stem}_states.npz")
    )
    metadata_path = (
        _resolve_project_path(cli_args.metadata_output)
        if cli_args.metadata_output
        else output_path.with_suffix(".json")
    )
    _check_outputs(
        (output_path, state_output_path, metadata_path),
        force=bool(cli_args.force),
    )

    _log("loading checkpoint")
    with checkpoint_path.open("rb") as checkpoint_file:
        raw_checkpoint = pickle.load(checkpoint_file)

    train_args = argparse.Namespace(**raw_checkpoint["args"])
    original_args = dict(raw_checkpoint["args"])
    _apply_bigmap_overrides(train_args, cli_args)

    food_scale = _jax_render_food_scale(train_args)
    actor_builder = NumpyActorObservationBuilder(
        num_ants=int(train_args.num_ants),
        actor_vision_radius=int(train_args.actor_vision_radius),
        write_bits=int(train_args.write_bits),
        food_scale=float(food_scale),
        agent_identity_types=getattr(train_args, "agent_identity_types", None),
    )
    if actor_builder.actor_obs_dim != int(raw_checkpoint["actor_obs_dim"]):
        raise ValueError(
            "actor observation dimension mismatch: "
            f"builder={actor_builder.actor_obs_dim}, "
            f"checkpoint={int(raw_checkpoint['actor_obs_dim'])}"
        )
    _log(f"prepared actor-only selector dim={actor_builder.actor_obs_dim}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    reset_seed = int(getattr(train_args, "seed", 0)) + int(cli_args.seed_offset)
    reset_options = _reset_options_from_cli(cli_args)
    frame_capacity = int(cli_args.max_steps) // int(cli_args.frame_stride) + 1
    state = StateRecorder(
        frame_capacity=frame_capacity,
        max_steps=int(cli_args.max_steps),
        num_ants=int(train_args.num_ants),
        width=int(train_args.width),
        height=int(train_args.height),
    )
    env = _env_from_args(
        train_args,
        render_mode="rgb_array",
        tile_size=int(cli_args.tile_size),
        render_style="big_scale_old_three_color",
    )
    writer = imageio.get_writer(
        output_path,
        fps=int(cli_args.output_fps),
        codec="libx264",
        macro_block_size=16,
        output_params=[
            "-crf",
            str(int(cli_args.ffmpeg_crf)),
            "-preset",
            str(cli_args.ffmpeg_preset),
        ],
    )

    start_time = time.monotonic()
    reward_total = 0.0
    steps = 0
    terminated = False
    truncated = False
    jax_devices: list[str] = []
    jax_module: Any | None = None
    action_mode = str(cli_args.action_mode)
    try:
        _log("resetting big-map environment")
        obs, _ = env.reset(seed=reset_seed, options=reset_options)
        _log("writing initial frame")
        state.capture_frame(step=0, obs=obs, env=env)
        writer.append_data(_render_frame(env))
        _log("importing jax and policy helpers")
        import jax
        import jax.numpy as jnp

        from ant_byte_env.training.jax_mappo.core import JaxMAPPOParams
        from ant_byte_env.training.jax_mappo.evaluation import (
            _evaluation_actions_for_mode,
            validate_evaluation_action_mode,
        )

        jax_module = jax
        action_mode = validate_evaluation_action_mode(action_mode)
        params = jax.tree_util.tree_map(jnp.asarray, raw_checkpoint["params"])
        actor_params = JaxMAPPOParams(
            actor_body=params.actor_body,
            move_head=params.move_head,
            write_head=params.write_head,
            critic_body=(),
            value_head=(),
        )

        @jax.jit
        def select_action(actor_obs: Any, action_key: Any) -> Any:
            return _evaluation_actions_for_mode(
                actor_params,
                actor_obs,
                jnp.zeros((actor_obs.shape[0], 1), dtype=jnp.float32),
                action_key,
                action_mode=action_mode,
                move_temperature=float(cli_args.move_temperature),
                write_temperature=float(cli_args.write_temperature),
            )

        jax_devices = [str(device) for device in jax.devices()]
        key = jax.random.PRNGKey(reset_seed)

        for step in range(1, int(cli_args.max_steps) + 1):
            key, action_key = jax.random.split(key)
            actor_obs = actor_builder.build(obs)
            if step == 1:
                _log("compiling first actor action")
            actions = np.asarray(
                select_action(jnp.asarray(actor_obs), action_key)
            ).reshape(int(train_args.num_ants), 2)
            if step == 1:
                _log("first actor action ready")
            state.capture_action(step_index=step - 1, actions=actions)
            obs, reward, terminated, truncated, _ = env.step(actions.reshape(-1))
            reward_total += float(reward)
            steps = step

            if step % int(cli_args.frame_stride) == 0:
                state.capture_frame(step=step, obs=obs, env=env)
                writer.append_data(_render_frame(env))
                if (
                    int(cli_args.progress_every) > 0
                    and state.frames_written % int(cli_args.progress_every) == 0
                ):
                    elapsed = time.monotonic() - start_time
                    print(
                        f"rendered {state.frames_written}/{frame_capacity} frames; "
                        f"step={step}; delivered={int(getattr(env, 'delivered_food', 0))}; "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()
        if jax_module is not None and hasattr(jax_module, "clear_caches"):
            jax_module.clear_caches()

    state.write(
        state_output_path,
        checkpoint_path=checkpoint_path,
        reset_seed=reset_seed,
        seed_offset=int(cli_args.seed_offset),
        action_mode=action_mode,
        move_temperature=float(cli_args.move_temperature),
        write_temperature=float(cli_args.write_temperature),
        frame_stride=int(cli_args.frame_stride),
        output_video=output_path,
    )
    elapsed = time.monotonic() - start_time
    metadata = _metadata(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        state_output_path=state_output_path,
        raw_checkpoint=raw_checkpoint,
        original_args=original_args,
        train_args=train_args,
        cli_args=cli_args,
        reset_seed=reset_seed,
        action_mode=action_mode,
        reward_total=reward_total,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        env=env,
        state=state,
        reset_options=reset_options,
        elapsed=elapsed,
        jax_devices=jax_devices,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {state_output_path}")
    print(f"wrote {metadata_path}")
    return 0


class NumpyActorObservationBuilder:
    def __init__(
        self,
        *,
        num_ants: int,
        actor_vision_radius: int,
        write_bits: int,
        food_scale: float,
        agent_identity_types: int | None,
    ) -> None:
        self.num_ants = int(num_ants)
        self.actor_vision_radius = int(actor_vision_radius)
        self.write_bits = int(write_bits)
        self.food_scale = float(food_scale)
        self.offsets = _base_offsets(self.actor_vision_radius)
        self.identity = _identity_features(self.num_ants, agent_identity_types)
        patch_size = self.offsets.shape[0]
        self.actor_obs_dim = (
            patch_size
            + patch_size
            + patch_size * self.write_bits
            + patch_size
            + patch_size
            + self.identity.shape[1]
            + 1
            + (MOVEMENT_ACTION_COUNT - 1)
        )

    def build(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ants_pos = np.asarray(obs["ants_pos"], dtype=np.int32)
        ants_facing = np.asarray(obs["ants_facing"], dtype=np.int32)
        positions, valid = _patch_positions(
            ants_pos,
            ants_facing,
            self.offsets,
            width=int(obs["food"].shape[1]),
            height=int(obs["food"].shape[0]),
        )
        local_food = _local_grid_values(obs["food"], positions, valid) / self.food_scale
        ants_count_scale = max(float(self.num_ants), 1.0)
        local_ants_count = (
            _local_grid_values(obs["ants_count"], positions, valid) / ants_count_scale
        )
        local_byte_bits = []
        bytes_grid = np.asarray(obs["bytes"], dtype=np.uint8)
        for bit_index in range(self.write_bits):
            bit_grid = ((bytes_grid >> bit_index) & np.uint8(1)).astype(np.float32)
            local_byte_bits.append(_local_grid_values(bit_grid, positions, valid))
        local_hub = _local_hub_values(obs["hub_pos"], positions, valid)
        local_border = (~valid).astype(np.float32)
        if "obstacles" in obs:
            local_obstacles = _local_grid_values(obs["obstacles"], positions, valid)
            local_border = np.maximum(local_border, local_obstacles)
        own_carrying = np.asarray(obs["ants_carrying"], dtype=np.float32).reshape(-1, 1)
        own_facing = _facing_one_hot(ants_facing)
        features = [
            local_food.astype(np.float32),
            local_ants_count.astype(np.float32),
            *[bit.astype(np.float32) for bit in local_byte_bits],
            local_hub.astype(np.float32),
            local_border.astype(np.float32),
            self.identity,
            own_carrying,
            own_facing,
        ]
        return np.concatenate(features, axis=-1).astype(np.float32, copy=False)[
            np.newaxis,
            ...,
        ]


class StateRecorder:
    def __init__(
        self,
        *,
        frame_capacity: int,
        max_steps: int,
        num_ants: int,
        width: int,
        height: int,
    ) -> None:
        coord_dtype = np.uint16 if max(width, height) <= np.iinfo(np.uint16).max else np.uint32
        self.actions = np.zeros((max_steps, num_ants, 2), dtype=np.uint8)
        self.frame_steps = np.zeros(frame_capacity, dtype=np.int32)
        self.frame_ant_positions = np.zeros((frame_capacity, num_ants, 2), dtype=coord_dtype)
        self.frame_ant_carrying = np.zeros((frame_capacity, num_ants), dtype=np.bool_)
        self.frame_hub_positions = np.zeros((frame_capacity, 2), dtype=coord_dtype)
        self.frame_delivered_food = np.zeros(frame_capacity, dtype=np.int32)
        self.frame_remaining_food = np.zeros(frame_capacity, dtype=np.int32)
        self.frame_carrying_ants = np.zeros(frame_capacity, dtype=np.int32)
        self.frame_nonzero_byte_tiles = np.zeros(frame_capacity, dtype=np.int32)
        self.frame_visited_cells = np.zeros(frame_capacity, dtype=np.int32)
        self.frames_written = 0
        self.steps_written = 0
        self.initial_food_yx: np.ndarray | None = None
        self.initial_food_amounts: np.ndarray | None = None
        self.final_byte_yx: np.ndarray | None = None
        self.final_byte_values: np.ndarray | None = None

    def capture_action(self, *, step_index: int, actions: np.ndarray) -> None:
        self.actions[int(step_index)] = np.asarray(actions, dtype=np.uint8)
        self.steps_written = max(self.steps_written, int(step_index) + 1)

    def capture_frame(
        self,
        *,
        step: int,
        obs: dict[str, np.ndarray],
        env: Any,
    ) -> None:
        index = self.frames_written
        self.frame_steps[index] = int(step)
        self.frame_ant_positions[index] = np.asarray(obs["ants_pos"])
        self.frame_ant_carrying[index] = np.asarray(obs["ants_carrying"], dtype=np.bool_)
        self.frame_hub_positions[index] = np.asarray(obs["hub_pos"])
        self.frame_delivered_food[index] = int(getattr(env, "delivered_food", 0))
        self.frame_remaining_food[index] = int(np.asarray(obs["food"]).sum())
        self.frame_carrying_ants[index] = int(np.asarray(obs["ants_carrying"]).sum())
        self.frame_nonzero_byte_tiles[index] = int(np.count_nonzero(obs["bytes"]))
        self.frame_visited_cells[index] = int(np.asarray(env.visited_cells).sum())
        if self.frames_written == 0:
            food_yx = np.argwhere(np.asarray(obs["food"]) > 0)
            self.initial_food_yx = food_yx.astype(self.frame_ant_positions.dtype)
            self.initial_food_amounts = np.asarray(obs["food"])[
                food_yx[:, 0], food_yx[:, 1]
            ].astype(np.int32)
        final_byte_yx = np.argwhere(np.asarray(obs["bytes"]) > 0)
        self.final_byte_yx = final_byte_yx.astype(self.frame_ant_positions.dtype)
        self.final_byte_values = np.asarray(obs["bytes"])[
            final_byte_yx[:, 0], final_byte_yx[:, 1]
        ].astype(np.uint8)
        self.frames_written += 1

    def write(
        self,
        path: Path,
        *,
        checkpoint_path: Path,
        reset_seed: int,
        seed_offset: int,
        action_mode: str,
        move_temperature: float,
        write_temperature: float,
        frame_stride: int,
        output_video: Path,
    ) -> None:
        count = self.frames_written
        np.savez_compressed(
            path,
            schema=np.asarray("cool_antz_bigmap_replay_state_v1"),
            checkpoint=np.asarray(_project_display_path(checkpoint_path)),
            output_video=np.asarray(_project_display_path(output_video)),
            reset_seed=np.asarray(reset_seed, dtype=np.int64),
            seed_offset=np.asarray(seed_offset, dtype=np.int64),
            action_mode=np.asarray(action_mode),
            move_temperature=np.asarray(move_temperature, dtype=np.float32),
            write_temperature=np.asarray(write_temperature, dtype=np.float32),
            frame_stride=np.asarray(frame_stride, dtype=np.int32),
            actions=self.actions[: self.steps_written],
            frame_steps=self.frame_steps[:count],
            frame_ant_positions=self.frame_ant_positions[:count],
            frame_ant_carrying=self.frame_ant_carrying[:count],
            frame_hub_positions=self.frame_hub_positions[:count],
            frame_delivered_food=self.frame_delivered_food[:count],
            frame_remaining_food=self.frame_remaining_food[:count],
            frame_carrying_ants=self.frame_carrying_ants[:count],
            frame_nonzero_byte_tiles=self.frame_nonzero_byte_tiles[:count],
            frame_visited_cells=self.frame_visited_cells[:count],
            initial_food_yx=np.zeros((0, 2), dtype=self.frame_ant_positions.dtype)
            if self.initial_food_yx is None
            else self.initial_food_yx,
            initial_food_amounts=np.zeros((0,), dtype=np.int32)
            if self.initial_food_amounts is None
            else self.initial_food_amounts,
            final_byte_yx=np.zeros((0, 2), dtype=self.frame_ant_positions.dtype)
            if self.final_byte_yx is None
            else self.final_byte_yx,
            final_byte_values=np.zeros((0,), dtype=np.uint8)
            if self.final_byte_values is None
            else self.final_byte_values,
        )


def _apply_bigmap_overrides(train_args: argparse.Namespace, cli_args: argparse.Namespace) -> None:
    inner_size = int(cli_args.inner_window_size)
    default_margin = (min(int(cli_args.width), int(cli_args.height)) - inner_size) // 2
    if default_margin < 0:
        raise ValueError("inner-window-size must fit inside width/height.")
    train_args.width = int(cli_args.width)
    train_args.height = int(cli_args.height)
    if cli_args.num_ants is not None:
        train_args.num_ants = int(cli_args.num_ants)
    if cli_args.food_count is not None:
        train_args.food_count = int(cli_args.food_count)
    if cli_args.food_sources is not None:
        train_args.food_sources = int(cli_args.food_sources)
    train_args.max_steps = int(cli_args.max_steps)
    train_args.layout_margin = (
        default_margin if cli_args.layout_margin is None else int(cli_args.layout_margin)
    )
    train_args.hub_center_window_size = (
        inner_size
        if cli_args.hub_center_window_size is None
        else int(cli_args.hub_center_window_size)
    )
    train_args.random_food = True
    train_args.random_hub = True
    if cli_args.food_position:
        train_args.random_food = False
    if cli_args.hub_pos:
        train_args.random_hub = False
    train_args.random_ant_spawn = False
    train_args.food_termination = False


def _reset_options_from_cli(
    cli_args: argparse.Namespace,
) -> dict[str, tuple[int, int] | list[tuple[int, int]]] | None:
    options: dict[str, tuple[int, int] | list[tuple[int, int]]] = {}
    if cli_args.hub_pos:
        options["hub_pos"] = _parse_xy(cli_args.hub_pos)
    if cli_args.food_position:
        options["food_positions"] = [
            _parse_xy(position) for position in cli_args.food_position
        ]
    return options or None


def _parse_xy(raw_value: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(raw_value).split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected x,y position, got {raw_value!r}")
    return int(parts[0]), int(parts[1])


def _base_offsets(radius: int) -> np.ndarray:
    axis = np.arange(-int(radius), int(radius) + 1, dtype=np.int32)
    offset_y = np.repeat(axis, 2 * int(radius) + 1)
    offset_x = np.tile(axis, 2 * int(radius) + 1)
    return np.stack([offset_x, offset_y], axis=-1)


def _patch_positions(
    ants_pos: np.ndarray,
    ants_facing: np.ndarray,
    base_offsets: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    offset_x = base_offsets[:, 0]
    offset_y = base_offsets[:, 1]
    offsets = np.broadcast_to(base_offsets, (ants_pos.shape[0], *base_offsets.shape)).copy()
    down_offsets = np.stack([-offset_y, offset_x], axis=-1)
    left_offsets = np.stack([-offset_x, -offset_y], axis=-1)
    up_offsets = np.stack([offset_y, -offset_x], axis=-1)
    offsets[ants_facing == MOVE_DOWN] = down_offsets
    offsets[ants_facing == MOVE_LEFT] = left_offsets
    offsets[ants_facing == MOVE_UP] = up_offsets
    unknown_facing = (
        (ants_facing != MOVE_UP)
        & (ants_facing != MOVE_RIGHT)
        & (ants_facing != MOVE_DOWN)
        & (ants_facing != MOVE_LEFT)
    )
    offsets[unknown_facing] = base_offsets
    positions = ants_pos[:, None, :] + offsets
    valid = (
        (positions[..., 0] >= 0)
        & (positions[..., 0] < int(width))
        & (positions[..., 1] >= 0)
        & (positions[..., 1] < int(height))
    )
    return positions, valid


def _local_grid_values(
    grid: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    clipped_x = np.clip(positions[..., 0], 0, grid.shape[1] - 1)
    clipped_y = np.clip(positions[..., 1], 0, grid.shape[0] - 1)
    values = np.asarray(grid)[clipped_y, clipped_x].astype(np.float32)
    return np.where(valid, values, 0.0).astype(np.float32)


def _local_hub_values(
    hub_pos: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    hub = np.asarray(hub_pos, dtype=np.int32)
    return (
        valid
        & (positions[..., 0] == int(hub[0]))
        & (positions[..., 1] == int(hub[1]))
    ).astype(np.float32)


def _facing_one_hot(ants_facing: np.ndarray) -> np.ndarray:
    facing_index = np.clip(
        np.asarray(ants_facing, dtype=np.int32) - 1,
        0,
        MOVEMENT_ACTION_COUNT - 2,
    )
    return np.eye(MOVEMENT_ACTION_COUNT - 1, dtype=np.float32)[facing_index]


def _identity_features(num_ants: int, agent_identity_types: int | None) -> np.ndarray:
    if int(num_ants) <= 1:
        return np.zeros((int(num_ants), 0), dtype=np.float32)
    identity_count = int(num_ants) if agent_identity_types is None else int(agent_identity_types)
    if identity_count <= 0:
        raise ValueError("agent_identity_types must be positive.")
    indices = np.arange(int(num_ants), dtype=np.int32) % identity_count
    return np.eye(identity_count, dtype=np.float32)[indices]


def _render_frame(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError("rgb_array rendering unexpectedly returned None.")
    return frame


def _metadata(
    *,
    checkpoint_path: Path,
    output_path: Path,
    state_output_path: Path,
    raw_checkpoint: dict[str, Any],
    original_args: dict[str, Any],
    train_args: argparse.Namespace,
    cli_args: argparse.Namespace,
    reset_seed: int,
    action_mode: str,
    reward_total: float,
    steps: int,
    terminated: bool,
    truncated: bool,
    env: Any,
    state: StateRecorder,
    reset_options: dict[str, tuple[int, int] | list[tuple[int, int]]] | None,
    elapsed: float,
    jax_devices: list[str],
) -> dict[str, Any]:
    food_positions = (
        []
        if state.initial_food_yx is None
        else [[int(x), int(y)] for y, x in state.initial_food_yx.tolist()]
    )
    hub = state.frame_hub_positions[0].astype(int).tolist() if state.frames_written else None
    food = food_positions[0] if food_positions else None
    manhattan_food_to_hub = (
        None
        if hub is None or food is None
        else int(abs(int(food[0]) - int(hub[0])) + abs(int(food[1]) - int(hub[1])))
    )
    inner_low = int(train_args.layout_margin)
    inner_high = int(train_args.width) - int(train_args.layout_margin) - 1
    return {
        "policy_choice": (
            "best 250x250 set_cnn actor rendered actor-only; critic/central obs skipped "
            "because sampled_move_greedy_write only uses actor logits"
        ),
        "checkpoint": _project_display_path(checkpoint_path),
        "checkpoint_run_name": raw_checkpoint.get("run_name"),
        "checkpoint_actor_obs_dim": int(raw_checkpoint["actor_obs_dim"]),
        "checkpoint_central_obs_dim": int(raw_checkpoint["central_obs_dim"]),
        "checkpoint_best_model_metric_value": raw_checkpoint.get("metrics", {}).get(
            "best_model_metric_value"
        ),
        "checkpoint_best_model_update": raw_checkpoint.get("metrics", {}).get(
            "best_model_update"
        ),
        "output": _project_display_path(output_path),
        "state_archive": _project_display_path(state_output_path),
        "state_archive_schema": "cool_antz_bigmap_replay_state_v1",
        "state_archive_contents": [
            "per-step uint8 move/write actions for exact replay",
            "render-frame ant positions/carrying snapshots",
            "render-frame hub, food, delivery, carrying, byte-count metrics",
            "initial food source sparse coordinates and final byte sparse coordinates",
        ],
        "render_style": "big_scale_old_three_color",
        "palette": _old_style_palette(),
        "width": int(train_args.width),
        "height": int(train_args.height),
        "tile_size": int(cli_args.tile_size),
        "num_ants": int(train_args.num_ants),
        "food_total": int(train_args.food_count),
        "food_sources": int(train_args.food_sources),
        "food_positions": food_positions,
        "hub": hub,
        "manhattan_food_to_hub": manhattan_food_to_hub,
        "inner_window_size": int(cli_args.inner_window_size),
        "inner_low": inner_low,
        "inner_high": inner_high,
        "food_inside_inner_window": _positions_inside(food_positions, inner_low, inner_high),
        "hub_inside_inner_window": (
            False if hub is None else _position_inside(hub, inner_low, inner_high)
        ),
        "seed_offset": int(cli_args.seed_offset),
        "reset_seed": int(reset_seed),
        "action_mode": action_mode,
        "move_temperature": float(cli_args.move_temperature),
        "write_temperature": float(cli_args.write_temperature),
        "requested_sim_steps": int(cli_args.max_steps),
        "steps": int(steps),
        "frame_stride": int(cli_args.frame_stride),
        "frames_written": int(state.frames_written),
        "video_fps": int(cli_args.base_video_fps),
        "output_fps": int(cli_args.output_fps),
        "post_speedup_from_base": float(cli_args.output_fps) / float(cli_args.base_video_fps),
        "effective_sim_steps_per_second": float(cli_args.output_fps)
        * float(cli_args.frame_stride),
        "encoded_duration_seconds": float(state.frames_written) / float(cli_args.output_fps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward_total": float(reward_total),
        "delivered_food": int(getattr(env, "delivered_food", 0)),
        "remaining_food": int(env.food.sum()),
        "carrying_ants": int(env.ants_carrying.sum()),
        "nonzero_byte_tiles": int(np.count_nonzero(env.bytes)),
        "visited_cells": int(env.visited_cells.sum()),
        "original_train_args": _jsonable_args(original_args),
        "rollout_overrides": {
            "width": int(train_args.width),
            "height": int(train_args.height),
            "max_steps": int(train_args.max_steps),
            "layout_margin": int(train_args.layout_margin),
            "hub_center_window_size": int(train_args.hub_center_window_size),
            "random_food": bool(train_args.random_food),
            "random_hub": bool(train_args.random_hub),
            "random_ant_spawn": bool(train_args.random_ant_spawn),
            "food_termination": bool(train_args.food_termination),
        },
        "reset_options": _jsonable_reset_options(reset_options),
        "critic_architecture": _target_critic_architecture(
            critic_forward_kwargs_from_safe_args(train_args)
        ),
        "jax_devices": jax_devices,
        "ffmpeg_crf": int(cli_args.ffmpeg_crf),
        "ffmpeg_preset": str(cli_args.ffmpeg_preset),
        "ffprobe": _ffprobe(output_path),
        "elapsed_render_seconds": round(float(elapsed), 3),
    }


def critic_forward_kwargs_from_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    from ant_byte_env.training.jax_mappo.core import critic_forward_kwargs_from_args

    return critic_forward_kwargs_from_args(args)


def _old_style_palette() -> dict[str, Any]:
    return {
        "background": [int(v) for v in OLD_STYLE_BACKGROUND_COLOR.tolist()],
        "normal_ant": list(OLD_STYLE_NORMAL_ANT_COLOR),
        "carrying_ant": list(OLD_STYLE_CARRYING_ANT_COLOR),
        "food_outer": list(OLD_STYLE_FOOD_OUTER_COLOR),
        "food_center": list(OLD_STYLE_FOOD_CENTER_COLOR),
        "colony": list(OLD_STYLE_COLONY_COLOR),
        "marker_radius_cells": int(OLD_STYLE_MARKER_RADIUS_CELLS),
        "marker_center_radius_cells": int(OLD_STYLE_MARKER_CENTER_RADIUS_CELLS),
        "byte_overlay": {
            "formula": (
                "rgb=(40+180*value/max_write,92,255-120*value/max_write), "
                "alpha=96/255"
            )
        },
    }


def _positions_inside(
    positions: list[list[int]],
    low: int,
    high: int,
) -> bool:
    return all(_position_inside(position, low, high) for position in positions)


def _position_inside(position: list[int], low: int, high: int) -> bool:
    return (
        int(position[0]) >= int(low)
        and int(position[0]) <= int(high)
        and int(position[1]) >= int(low)
        and int(position[1]) <= int(high)
    )


def _jsonable_args(args: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _jsonable_reset_options(
    reset_options: dict[str, tuple[int, int] | list[tuple[int, int]]] | None,
) -> dict[str, Any] | None:
    if reset_options is None:
        return None
    result: dict[str, Any] = {}
    if "hub_pos" in reset_options:
        result["hub_pos"] = [int(value) for value in reset_options["hub_pos"]]
    if "food_positions" in reset_options:
        result["food_positions"] = [
            [int(value) for value in position]
            for position in reset_options["food_positions"]
        ]
    return result


def _ffprobe(path: Path) -> dict[str, Any] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return json.loads(completed.stdout)


def _check_outputs(paths: tuple[Path, ...], *, force: bool) -> None:
    if force:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists: {rendered}")


def _log(message: str) -> None:
    print(f"[bigmap-render] {message}", flush=True)


def _resolve_project_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _project_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))
