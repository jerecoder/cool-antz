#!/usr/bin/env python3
"""Render a JAX MAPPO checkpoint as a channel-grid rollout video."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
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
from PIL import Image, ImageDraw, ImageFont

from ant_byte_env.rendering import (
    _compile_jax_action_selector,
    _env_from_args,
    _jax_render_food_scale,
    _jax_render_reset_options,
    _target_critic_architecture,
)


BIT_COLORS_RGB = (
    (20, 86, 210),
    (0, 145, 255),
    (77, 170, 255),
    (155, 210, 255),
    (90, 210, 135),
    (246, 210, 76),
    (244, 137, 73),
    (214, 79, 135),
)


class ChannelGridRenderer:
    """Build byte-channel frames without ant/food/hub markers."""

    def __init__(
        self,
        *,
        bit_colors: tuple[tuple[int, int, int], ...],
        labels: tuple[str, ...],
        left_size: int,
        panel_size: int,
        visual_easing_alpha: float,
    ) -> None:
        self.bit_colors = np.asarray(bit_colors, dtype=np.float32)
        self.labels = labels
        self.left_size = int(left_size)
        self.panel_size = int(panel_size)
        self.visual_easing_alpha = float(visual_easing_alpha)
        self.panel_columns = max(2, int(np.ceil(len(self.labels) / 2.0)))
        self.panel_rows = int(np.ceil(len(self.labels) / float(self.panel_columns)))
        self._smoothed_left: np.ndarray | None = None
        self._smoothed_bits: list[np.ndarray] | None = None
        self._font = _load_font(max(18, int(self.panel_size * 0.07)))

    @property
    def frame_size(self) -> tuple[int, int]:
        return (
            self.left_size + self.panel_size * self.panel_columns,
            max(self.left_size, self.panel_size * self.panel_rows),
        )

    def frame(self, byte_grid: np.ndarray) -> np.ndarray:
        current_left = self._normalized_grid(byte_grid)
        current_bits = self._bit_grids(byte_grid)
        if self._smoothed_left is None or self._smoothed_bits is None:
            self._smoothed_left = current_left
            self._smoothed_bits = current_bits
        else:
            alpha = self.visual_easing_alpha
            self._smoothed_left = alpha * current_left + (1.0 - alpha) * self._smoothed_left
            self._smoothed_bits = [
                alpha * current + (1.0 - alpha) * previous
                for current, previous in zip(current_bits, self._smoothed_bits)
            ]

        left = _resize_nearest(_uint8_rgb(self._smoothed_left), (self.left_size, self.left_size))
        right = Image.new(
            "RGB",
            (self.panel_size * self.panel_columns, self.panel_size * self.panel_rows),
            (0, 0, 0),
        )
        for index, panel in enumerate(self._smoothed_bits):
            panel_image = _resize_nearest(
                _uint8_rgb(panel),
                (self.panel_size, self.panel_size),
            )
            self._draw_label(panel_image, self.labels[index])
            x_pos = (index % self.panel_columns) * self.panel_size
            y_pos = (index // self.panel_columns) * self.panel_size
            right.paste(panel_image, (x_pos, y_pos))

        composite = Image.new("RGB", self.frame_size, (0, 0, 0))
        composite.paste(left, (0, 0))
        composite.paste(right, (self.left_size, 0))
        return np.asarray(composite)

    def _normalized_grid(self, byte_grid: np.ndarray) -> np.ndarray:
        bit_masks = (1 << np.arange(len(self.bit_colors), dtype=np.uint8)).reshape(1, 1, -1)
        active = (byte_grid[..., np.newaxis].astype(np.uint8) & bit_masks) != 0
        active_count = active.sum(axis=-1)
        color_sum = np.tensordot(active.astype(np.float32), self.bit_colors, axes=([2], [0]))
        normalized = np.zeros((*byte_grid.shape, 3), dtype=np.float32)
        nonzero = active_count > 0
        normalized[nonzero] = color_sum[nonzero] / active_count[nonzero, np.newaxis]
        return normalized

    def _bit_grids(self, byte_grid: np.ndarray) -> list[np.ndarray]:
        panels: list[np.ndarray] = []
        for index, color in enumerate(self.bit_colors):
            active = ((byte_grid.astype(np.uint8) & np.uint8(1 << index)) != 0)[..., np.newaxis]
            panels.append(np.where(active, color, 0.0).astype(np.float32))
        return panels

    def _draw_label(self, image: Image.Image, label: str) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        margin = max(8, int(self.panel_size * 0.035))
        text_bbox = draw.textbbox((0, 0), label, font=self._font)
        width = text_bbox[2] - text_bbox[0]
        height = text_bbox[3] - text_bbox[1]
        pad_x = max(7, int(self.panel_size * 0.018))
        pad_y = max(5, int(self.panel_size * 0.014))
        background = (
            margin,
            margin,
            margin + width + 2 * pad_x,
            margin + height + 2 * pad_y,
        )
        draw.rounded_rectangle(background, radius=5, fill=(0, 0, 0, 176))
        draw.text(
            (margin + pad_x, margin + pad_y),
            label,
            font=self._font,
            fill=(238, 247, 255, 255),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="JAX checkpoint to render.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Output JSON sidecar path. Defaults to the MP4 path with .json suffix.",
    )
    parser.add_argument("--seed-offset", type=int, default=210_002)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--action-mode",
        default="sampled_move_greedy_write",
        help="JAX evaluation action mode.",
    )
    parser.add_argument("--move-temperature", type=float, default=0.75)
    parser.add_argument("--write-temperature", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=0.0)
    parser.add_argument("--normal-fps", type=int, default=8)
    parser.add_argument("--base-output-fps", type=int, default=16)
    parser.add_argument("--speedup", type=float, default=3.0)
    parser.add_argument("--visual-easing-alpha", type=float, default=0.34)
    parser.add_argument("--left-size", type=int, default=800)
    parser.add_argument("--panel-size", type=int, default=400)
    parser.add_argument("--layout-margin", type=int, default=0)
    parser.add_argument("--hub-center-window-size", type=int, default=0)
    parser.add_argument(
        "--single-centered-ant-no-food",
        action="store_true",
        help="Render a one-ant centered rollout with an empty food grid.",
    )
    parser.add_argument("--ffmpeg-crf", type=int, default=24)
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli_args = parse_args()
    checkpoint_path = _resolve_project_path(cli_args.checkpoint)
    output_path = _resolve_project_path(cli_args.output)
    metadata_path = (
        _resolve_project_path(cli_args.metadata_output)
        if cli_args.metadata_output
        else output_path.with_suffix(".json")
    )
    if output_path.exists() and not cli_args.force:
        raise FileExistsError(f"output already exists: {output_path}")
    if metadata_path.exists() and not cli_args.force:
        raise FileExistsError(f"metadata output already exists: {metadata_path}")

    with checkpoint_path.open("rb") as checkpoint_file:
        raw_checkpoint = pickle.load(checkpoint_file)

    train_args = argparse.Namespace(**raw_checkpoint["args"])
    original_layout_margin = int(getattr(train_args, "layout_margin", 0))
    original_hub_center_window_size = int(getattr(train_args, "hub_center_window_size", 0))
    original_max_steps = int(getattr(train_args, "max_steps", cli_args.max_steps))
    original_num_ants = int(getattr(train_args, "num_ants", 1))
    original_food_count = int(getattr(train_args, "food_count", 0))
    original_food_sources = int(getattr(train_args, "food_sources", 1))
    original_food_termination = bool(getattr(train_args, "food_termination", True))
    original_random_food = bool(getattr(train_args, "random_food", False))
    original_random_hub = bool(getattr(train_args, "random_hub", False))
    original_random_ant_spawn = bool(getattr(train_args, "random_ant_spawn", False))
    train_args.layout_margin = int(cli_args.layout_margin)
    train_args.hub_center_window_size = int(cli_args.hub_center_window_size)
    train_args.max_steps = int(cli_args.max_steps)
    if cli_args.single_centered_ant_no_food:
        train_args.num_ants = 1
        train_args.food_count = 0
        train_args.food_sources = 1
        train_args.random_food = False
        train_args.random_hub = False
        train_args.random_ant_spawn = False
        train_args.food_termination = False

    frame_limit = _frame_limit(train_args.max_steps, cli_args.max_frames)
    output_fps = int(round(float(cli_args.base_output_fps) * float(cli_args.speedup)))
    write_bits = int(getattr(train_args, "write_bits", 4))
    if write_bits > len(BIT_COLORS_RGB):
        raise ValueError(
            f"this presentation preset supports at most {len(BIT_COLORS_RGB)} write bits"
        )
    labels = tuple(f"bit{index + 1}" for index in range(write_bits))
    renderer = ChannelGridRenderer(
        bit_colors=BIT_COLORS_RGB[: len(labels)],
        labels=labels,
        left_size=cli_args.left_size,
        panel_size=cli_args.panel_size,
        visual_easing_alpha=cli_args.visual_easing_alpha,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    reset_seed = int(getattr(train_args, "seed", 0)) + int(cli_args.seed_offset)
    env = _env_from_args(train_args, render_mode=None, tile_size=None)
    writer = imageio.get_writer(
        output_path,
        fps=output_fps,
        codec="libx264",
        macro_block_size=1,
        output_params=[
            "-crf",
            str(int(cli_args.ffmpeg_crf)),
            "-preset",
            str(cli_args.ffmpeg_preset),
        ],
    )

    start_time = time.monotonic()
    reward_total = 0.0
    frames_written = 0
    steps = 0
    terminated = False
    truncated = False
    initial_hub: list[int] | None = None
    initial_ant_positions: list[list[int]] = []
    source_positions: list[list[int]] = []
    action_mode = cli_args.action_mode
    jax_module: Any | None = None
    try:
        obs, _ = env.reset(
            seed=reset_seed,
            options=_jax_render_reset_options(train_args, seed=reset_seed),
        )
        initial_hub = _xy_list(obs["hub_pos"])
        initial_ant_positions = _positions_list(obs["ants_pos"])
        source_positions = _food_positions(obs["food"])
        writer.append_data(renderer.frame(obs["bytes"]))
        frames_written = 1

        import jax
        import jax.numpy as jnp

        from ant_byte_env.training.jax_mappo import (
            build_actor_observations,
            build_central_observations,
            get_action_and_value,
        )
        from ant_byte_env.training.jax_mappo.core import critic_forward_kwargs_from_args
        from ant_byte_env.training.jax_mappo.evaluation import (
            _evaluation_actions_for_mode,
            validate_evaluation_action_mode,
        )
        from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training

        jax_module = jax
        food_scale = _jax_render_food_scale(train_args)
        critic_kwargs = critic_forward_kwargs_from_args(train_args)
        action_mode = validate_evaluation_action_mode(cli_args.action_mode)

        obs_batch = _jax_obs_batch(obs, jnp)
        central_obs = build_central_observations(
            obs_batch,
            food_scale=food_scale,
            write_bits=train_args.write_bits,
            obs_width=train_args.obs_width,
            obs_height=train_args.obs_height,
        )
        actor_obs = build_actor_observations(
            obs_batch,
            food_scale=food_scale,
            actor_vision_radius=train_args.actor_vision_radius,
            write_bits=train_args.write_bits,
            obs_width=train_args.obs_width,
            obs_height=train_args.obs_height,
        )
        checkpoint = load_checkpoint_for_training(
            checkpoint_path,
            central_obs_dim=int(central_obs.shape[-1]),
            actor_obs_dim=int(actor_obs.shape[-1]),
            target_write_bits=int(train_args.write_bits),
            actor_vision_radius=int(train_args.actor_vision_radius),
            target_num_ants=int(getattr(train_args, "num_ants", 1)),
            target_agent_identity_types=getattr(train_args, "agent_identity_types", None),
            target_critic_architecture=_target_critic_architecture(critic_kwargs),
        )
        params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
        select_action = _compile_jax_action_selector(
            args=train_args,
            params=params,
            deterministic=float(cli_args.policy_temperature) == 0.0,
            build_actor_observations=build_actor_observations,
            build_central_observations=build_central_observations,
            get_action_and_value=get_action_and_value,
            evaluation_actions_for_mode=_evaluation_actions_for_mode,
            jax=jax,
            food_scale=food_scale,
            action_mode=action_mode,
            move_temperature=float(cli_args.move_temperature),
            write_temperature=float(cli_args.write_temperature),
            critic_kwargs=critic_kwargs,
        )

        key = jax.random.PRNGKey(reset_seed)
        while frames_written < frame_limit:
            key, action_key = jax.random.split(key)
            actions = select_action(_jax_obs_batch(obs, jnp), action_key)
            obs, reward, terminated, truncated, _ = env.step(np.asarray(actions).reshape(-1))
            reward_total += float(reward)
            steps += 1
            writer.append_data(renderer.frame(obs["bytes"]))
            frames_written += 1
            if cli_args.progress_every > 0 and frames_written % cli_args.progress_every == 0:
                print(
                    f"rendered {frames_written}/{frame_limit} frames; "
                    f"delivered={int(getattr(env, 'delivered_food', 0))}",
                    flush=True,
                )
            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()
        if jax_module is not None and hasattr(jax_module, "clear_caches"):
            jax_module.clear_caches()

    elapsed = time.monotonic() - start_time
    metadata = {
        "renderer": "normalized_channel_grid_no_markers_smooth_bit_labels_3x",
        "left_panel": "normalized_active_byte_channel_grid_only",
        "bitmap_panels": list(labels),
        "panel_label_policy": (
            "bit panels are labeled bit1..bitN; "
            "blue/channel prefixes intentionally omitted"
        ),
        "normalization": "mean of active channel colors per cell; zero active bits are black",
        "palette": "four shades of blue, one per byte channel",
        "visual_smoothing": {
            "full_step_capture": True,
            "base_output_fps": int(cli_args.base_output_fps),
            "playback_speed": float(cli_args.speedup),
            "visual_easing_alpha": float(cli_args.visual_easing_alpha),
            "note": "visual easing only; policy actions and environment states are unchanged",
        },
        "hidden_markers": ["ants", "colony", "cookies"],
        "visible_markers": [],
        "bit_colors_rgb": [list(color) for color in BIT_COLORS_RGB[: len(labels)]],
        "checkpoint": _project_display_path(checkpoint_path),
        "output": _project_display_path(output_path),
        "jax_devices": [str(device) for device in jax.devices()],
        "seed_offset": int(cli_args.seed_offset),
        "reset_seed": int(reset_seed),
        "action_mode": action_mode,
        "move_temperature": float(cli_args.move_temperature),
        "write_temperature": float(cli_args.write_temperature),
        "initial_hub": initial_hub,
        "final_hub": _xy_list(obs["hub_pos"]),
        "source_positions": source_positions,
        "initial_ant_positions": initial_ant_positions,
        "final_ant_positions": _positions_list(obs["ants_pos"]),
        "layout_margin": int(train_args.layout_margin),
        "hub_center_window_size": int(train_args.hub_center_window_size),
        "original_layout_margin": original_layout_margin,
        "original_hub_center_window_size": original_hub_center_window_size,
        "original_max_steps": original_max_steps,
        "original_num_ants": original_num_ants,
        "original_food_count": original_food_count,
        "original_food_sources": original_food_sources,
        "original_food_termination": original_food_termination,
        "original_random_food": original_random_food,
        "original_random_hub": original_random_hub,
        "original_random_ant_spawn": original_random_ant_spawn,
        "rollout_overrides": {
            "single_centered_ant_no_food": bool(cli_args.single_centered_ant_no_food),
            "centered_spawn": bool(cli_args.single_centered_ant_no_food),
            "no_cookies": bool(cli_args.single_centered_ant_no_food),
        },
        "max_steps": int(train_args.max_steps),
        "steps": int(steps),
        "frames_written": int(frames_written),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward_total": float(reward_total),
        "delivered_food": int(getattr(env, "delivered_food", 0)),
        "initial_food_total": int(
            getattr(env, "_initial_food_total", np.asarray(obs["food"]).sum())
        ),
        "remaining_grid_food": int(np.asarray(obs["food"]).sum()),
        "carrying_ants": int(np.asarray(obs["ants_carrying"]).sum()),
        "nonzero_byte_tiles": int(np.count_nonzero(obs["bytes"])),
        "final_bit_counts": _bit_counts(obs["bytes"], len(labels)),
        "frame_size": list(renderer.frame_size),
        "num_ants": int(getattr(train_args, "num_ants", 1)),
        "write_bits": int(getattr(train_args, "write_bits", len(labels))),
        "per_ant_write_channels": bool(getattr(train_args, "per_ant_write_channels", False)),
        "write_while_moving": bool(getattr(train_args, "write_while_moving", False)),
        "write_penalty": float(getattr(train_args, "write_penalty", 0.0)),
        "write_bit_penalty": float(getattr(train_args, "write_bit_penalty", 0.0)),
        "write_bit_penalty_decay": float(getattr(train_args, "write_bit_penalty_decay", 0.0)),
        "normal_fps": int(cli_args.normal_fps),
        "output_fps": int(output_fps),
        "speedup_from_base_video": float(cli_args.speedup),
        "frame_preservation": f"all generated frames retained at {output_fps} fps",
        "ffmpeg_crf": int(cli_args.ffmpeg_crf),
        "ffmpeg_preset": str(cli_args.ffmpeg_preset),
        "encoded_duration_seconds": frames_written / output_fps,
        "ffprobe": _ffprobe(output_path),
        "elapsed_render_seconds": round(elapsed, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {metadata_path}")
    return 0


def _frame_limit(max_steps: int, max_frames: int | None) -> int:
    if max_frames is not None:
        if int(max_frames) < 1:
            raise ValueError("max_frames must be at least 1.")
        return int(max_frames)
    return int(max_steps) + 1


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _project_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _jax_obs_batch(obs: dict[str, np.ndarray], jnp_module: Any) -> dict[str, Any]:
    return {
        name: jnp_module.expand_dims(jnp_module.asarray(value), axis=0)
        for name, value in obs.items()
    }


def _resize_nearest(image_array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(image_array, mode="RGB").resize(size, Image.Resampling.NEAREST)


def _uint8_rgb(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, 255).astype(np.uint8)


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _xy_list(value: np.ndarray) -> list[int]:
    array = np.asarray(value).astype(int).reshape(-1)
    return [int(array[0]), int(array[1])]


def _positions_list(value: np.ndarray) -> list[list[int]]:
    positions = np.asarray(value).astype(int).reshape(-1, 2)
    return [[int(x_pos), int(y_pos)] for x_pos, y_pos in positions]


def _food_positions(food_grid: np.ndarray) -> list[list[int]]:
    positions = np.argwhere(np.asarray(food_grid) > 0)
    return [[int(x_pos), int(y_pos)] for y_pos, x_pos in positions]


def _bit_counts(byte_grid: np.ndarray, bit_count: int) -> list[int]:
    grid = np.asarray(byte_grid, dtype=np.uint8)
    return [int(np.count_nonzero(grid & np.uint8(1 << index))) for index in range(bit_count)]


def _ffprobe(output_path: Path) -> dict[str, Any] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=width,height,nb_frames,r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(output_path),
    ]
    returncode, stdout, stderr = _spawn_capture(command)
    if returncode != 0:
        return {"error": stderr.strip()}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {"error": stdout.strip()}
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration": (payload.get("format") or {}).get("duration"),
        "nb_frames": stream.get("nb_frames"),
    }


def _spawn_capture(command: list[str]) -> tuple[int, str, str]:
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        file_actions = [
            (os.POSIX_SPAWN_DUP2, stdout_file.fileno(), 1),
            (os.POSIX_SPAWN_DUP2, stderr_file.fileno(), 2),
        ]
        pid = os.posix_spawnp(command[0], command, os.environ, file_actions=file_actions)
        _, status = os.waitpid(pid, 0)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr = stderr_file.read().decode("utf-8", errors="replace")
    return os.waitstatus_to_exitcode(status), stdout, stderr


if __name__ == "__main__":
    raise SystemExit(main())
