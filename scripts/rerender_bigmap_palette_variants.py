#!/usr/bin/env python3
"""Replay a saved big-map rollout and render semantic palette variants."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ant_byte_env.rendering import _env_from_args


@dataclass(frozen=True)
class Palette:
    name: str
    description: str
    background: tuple[int, int, int]
    byte_low: tuple[int, int, int]
    byte_high: tuple[int, int, int]
    byte_alpha: float
    hub: tuple[int, int, int]
    food_outer: tuple[int, int, int]
    food_center: tuple[int, int, int]
    normal_ant: tuple[int, int, int]
    carrying_ant: tuple[int, int, int]
    obstacle: tuple[int, int, int] = (52, 58, 64)
    marker_radius_cells: int = 3
    marker_center_radius_cells: int = 1


PALETTES: dict[str, Palette] = {
    "paper_ink": Palette(
        name="paper_ink",
        description="Light analytical map: blue trails, black hub/ants, red food.",
        background=(246, 247, 241),
        byte_low=(174, 210, 235),
        byte_high=(5, 84, 157),
        byte_alpha=0.72,
        hub=(20, 24, 31),
        food_outer=(255, 186, 161),
        food_center=(202, 43, 51),
        normal_ant=(26, 26, 26),
        carrying_ant=(250, 164, 58),
    ),
    "nocturne_neon": Palette(
        name="nocturne_neon",
        description="Dark neon read: cyan-violet trails, amber food, white carriers.",
        background=(8, 12, 18),
        byte_low=(31, 129, 167),
        byte_high=(185, 74, 255),
        byte_alpha=0.82,
        hub=(255, 72, 197),
        food_outer=(255, 217, 94),
        food_center=(255, 145, 42),
        normal_ant=(152, 226, 255),
        carrying_ant=(255, 255, 255),
    ),
    "okabe_signal": Palette(
        name="okabe_signal",
        description="Colorblind-safe signal palette using Okabe-Ito inspired hues.",
        background=(238, 238, 232),
        byte_low=(86, 180, 233),
        byte_high=(0, 114, 178),
        byte_alpha=0.78,
        hub=(204, 121, 167),
        food_outer=(240, 228, 66),
        food_center=(213, 94, 0),
        normal_ant=(0, 0, 0),
        carrying_ant=(0, 158, 115),
    ),
    "thermal_black": Palette(
        name="thermal_black",
        description="Black thermal map: ember trails, cyan hub, lime food.",
        background=(3, 5, 7),
        byte_low=(98, 23, 8),
        byte_high=(255, 222, 77),
        byte_alpha=0.86,
        hub=(0, 209, 255),
        food_outer=(158, 255, 133),
        food_center=(55, 210, 91),
        normal_ant=(242, 92, 84),
        carrying_ant=(255, 255, 255),
    ),
    "bathymetry": Palette(
        name="bathymetry",
        description="Oceanic depth palette: navy field, teal-to-sand trails.",
        background=(10, 31, 46),
        byte_low=(27, 121, 134),
        byte_high=(244, 204, 122),
        byte_alpha=0.80,
        hub=(235, 83, 111),
        food_outer=(206, 255, 214),
        food_center=(94, 218, 143),
        normal_ant=(170, 224, 234),
        carrying_ant=(255, 233, 161),
    ),
    "field_notes": Palette(
        name="field_notes",
        description="Terrain/field map: moss ground, indigo trails, coral food.",
        background=(39, 61, 48),
        byte_low=(91, 127, 179),
        byte_high=(188, 194, 255),
        byte_alpha=0.76,
        hub=(255, 235, 180),
        food_outer=(255, 157, 113),
        food_center=(225, 67, 75),
        normal_ant=(23, 26, 22),
        carrying_ant=(255, 216, 77),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Source rollout manifest JSON.")
    parser.add_argument(
        "--states",
        default=None,
        help="Source state archive. Defaults to manifest state_archive.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for palette renders.")
    parser.add_argument(
        "--palette",
        action="append",
        choices=sorted(PALETTES),
        default=None,
        help="Palette to render. Repeatable. Defaults to all palettes.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--contact-frame", type=int, default=250)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--ffmpeg-crf", type=int, default=24)
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = _resolve_project_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    states_path = (
        _resolve_project_path(args.states)
        if args.states
        else _resolve_project_path(str(manifest["state_archive"]))
    )
    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette_names = list(args.palette or PALETTES.keys())
    frame_steps = _frame_steps(states_path, max_frames=args.max_frames)
    if frame_steps.size == 0:
        raise ValueError("no frame steps selected")

    source_stem = Path(str(manifest["output"])).stem
    fps = int(args.fps or manifest.get("output_fps", 160))
    outputs = {
        name: output_dir / f"{source_stem}_{name}_{len(frame_steps)}f.mp4"
        for name in palette_names
    }
    metadata_outputs = {
        name: output_dir / f"{source_stem}_{name}_{len(frame_steps)}f.json"
        for name in palette_names
    }
    existing = [
        path
        for path in [*outputs.values(), *metadata_outputs.values()]
        if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError(
            "outputs already exist: " + ", ".join(str(path) for path in existing)
        )

    actions = np.load(states_path)["actions"]
    env_args = _env_args_from_manifest(manifest)
    reset_options = _reset_options_from_manifest(manifest)
    env = _env_from_args(
        env_args,
        render_mode=None,
        tile_size=int(manifest.get("tile_size", 1)),
    )
    writers = {
        name: imageio.get_writer(
            outputs[name],
            fps=fps,
            codec="libx264",
            macro_block_size=16,
            output_params=[
                "-crf",
                str(int(args.ffmpeg_crf)),
                "-preset",
                str(args.ffmpeg_preset),
            ],
        )
        for name in palette_names
    }
    contact_frames: dict[str, np.ndarray] = {}
    target_step_to_index = {int(step): index for index, step in enumerate(frame_steps.tolist())}
    next_target_index = 0
    target_steps = frame_steps.tolist()
    start_time = time.monotonic()

    try:
        obs, _ = env.reset(seed=int(manifest["reset_seed"]), options=reset_options)
        current_step = 0
        _write_if_target(
            env=env,
            palette_names=palette_names,
            writers=writers,
            contact_frames=contact_frames,
            current_step=current_step,
            target_step_to_index=target_step_to_index,
            contact_frame=int(args.contact_frame),
        )
        next_target_index = 1 if target_steps and target_steps[0] == 0 else 0

        final_step = int(target_steps[-1])
        for step_index in range(final_step):
            action = actions[step_index].reshape(-1)
            obs, _, terminated, truncated, _ = env.step(action)
            current_step = step_index + 1
            if (
                next_target_index < len(target_steps)
                and current_step == int(target_steps[next_target_index])
            ):
                _write_if_target(
                    env=env,
                    palette_names=palette_names,
                    writers=writers,
                    contact_frames=contact_frames,
                    current_step=current_step,
                    target_step_to_index=target_step_to_index,
                    contact_frame=int(args.contact_frame),
                )
                frame_count = next_target_index + 1
                if (
                    int(args.progress_every) > 0
                    and frame_count % int(args.progress_every) == 0
                ):
                    print(
                        f"rendered {frame_count}/{len(frame_steps)} frames; "
                        f"step={current_step}; delivered={int(env.delivered_food)}; "
                        f"elapsed={time.monotonic() - start_time:.1f}s",
                        flush=True,
                    )
                next_target_index += 1
            if terminated or truncated:
                break
    finally:
        for writer in writers.values():
            writer.close()
        env.close()

    contact_path = output_dir / f"{source_stem}_palette_contact_f{int(args.contact_frame):04d}.png"
    if contact_frames:
        _write_contact_sheet(contact_path, contact_frames)

    elapsed = time.monotonic() - start_time
    summary = {
        "source_manifest": _project_display_path(manifest_path),
        "source_states": _project_display_path(states_path),
        "source_video": manifest.get("output"),
        "frames_rendered": int(len(frame_steps)),
        "frame_steps_first": int(frame_steps[0]),
        "frame_steps_last": int(frame_steps[-1]),
        "fps": fps,
        "contact_sheet": _project_display_path(contact_path) if contact_frames else None,
        "final_delivered_food": int(env.delivered_food),
        "final_remaining_food": int(env.food.sum()),
        "elapsed_seconds": round(elapsed, 3),
        "palettes": {},
    }
    for name in palette_names:
        palette = PALETTES[name]
        metadata = {
            "palette": _palette_json(palette),
            "output": _project_display_path(outputs[name]),
            "source_manifest": _project_display_path(manifest_path),
            "source_states": _project_display_path(states_path),
            "frames_rendered": int(len(frame_steps)),
            "frame_steps_first": int(frame_steps[0]),
            "frame_steps_last": int(frame_steps[-1]),
            "fps": fps,
            "ffprobe": _ffprobe(outputs[name]),
            "final_delivered_food": int(env.delivered_food),
            "final_remaining_food": int(env.food.sum()),
        }
        metadata_outputs[name].write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["palettes"][name] = {
            "output": _project_display_path(outputs[name]),
            "metadata": _project_display_path(metadata_outputs[name]),
            "description": palette.description,
        }
    summary_path = output_dir / f"{source_stem}_palette_trials_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    if contact_frames:
        print(f"wrote {contact_path}")
    for name in palette_names:
        print(f"wrote {outputs[name]}")
    return 0


def _write_if_target(
    *,
    env: Any,
    palette_names: list[str],
    writers: dict[str, Any],
    contact_frames: dict[str, np.ndarray],
    current_step: int,
    target_step_to_index: dict[int, int],
    contact_frame: int,
) -> None:
    frame_index = target_step_to_index.get(int(current_step))
    if frame_index is None:
        return
    for name in palette_names:
        frame = render_palette_frame(env, PALETTES[name])
        writers[name].append_data(frame)
        if frame_index == int(contact_frame):
            contact_frames[name] = frame


def render_palette_frame(env: Any, palette: Palette) -> np.ndarray:
    grid = np.broadcast_to(
        np.asarray(palette.background, dtype=np.float32),
        (env.height, env.width, 3),
    ).copy()
    bytes_grid = env.bytes.astype(np.float32)
    byte_mask = bytes_grid > 0
    if np.any(byte_mask):
        ratio = bytes_grid / max(float(env.max_write_value), 1.0)
        low = np.asarray(palette.byte_low, dtype=np.float32)
        high = np.asarray(palette.byte_high, dtype=np.float32)
        overlay = low + (high - low) * ratio[..., None]
        alpha = float(palette.byte_alpha)
        grid[byte_mask] = grid[byte_mask] * (1.0 - alpha) + overlay[byte_mask] * alpha
    if np.any(env.obstacles):
        grid[env.obstacles] = np.asarray(palette.obstacle, dtype=np.float32)
    _draw_marker(
        grid,
        env.hub_pos,
        color=palette.hub,
        radius_cells=palette.marker_radius_cells,
    )
    for y_pos, x_pos in np.argwhere(env.food > 0):
        position = np.array([int(x_pos), int(y_pos)], dtype=np.int32)
        _draw_marker(
            grid,
            position,
            color=palette.food_outer,
            radius_cells=palette.marker_radius_cells,
        )
        _draw_marker(
            grid,
            position,
            color=palette.food_center,
            radius_cells=palette.marker_center_radius_cells,
        )
    normal = np.asarray(palette.normal_ant, dtype=np.float32)
    carrying = np.asarray(palette.carrying_ant, dtype=np.float32)
    for position, is_carrying in zip(env.ants_pos, env.ants_carrying):
        x_pos, y_pos = int(position[0]), int(position[1])
        if 0 <= x_pos < env.width and 0 <= y_pos < env.height:
            grid[y_pos, x_pos] = carrying if bool(is_carrying) else normal
    return np.clip(grid, 0, 255).astype(np.uint8)


def _draw_marker(
    grid: np.ndarray,
    position: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius_cells: int,
) -> None:
    x_pos, y_pos = int(position[0]), int(position[1])
    radius = int(radius_cells)
    x0 = max(0, x_pos - radius)
    x1 = min(grid.shape[1], x_pos + radius + 1)
    y0 = max(0, y_pos - radius)
    y1 = min(grid.shape[0], y_pos + radius + 1)
    if x0 < x1 and y0 < y1:
        grid[y0:y1, x0:x1] = np.asarray(color, dtype=np.float32)


def _write_contact_sheet(path: Path, frames: dict[str, np.ndarray]) -> None:
    tile_width = 360
    label_height = 30
    columns = 3
    rows = int(np.ceil(len(frames) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_width + label_height)), (24, 24, 24))
    font = ImageFont.load_default()
    for index, (name, frame) in enumerate(frames.items()):
        image = Image.fromarray(frame).resize((tile_width, tile_width), Image.Resampling.LANCZOS)
        x_pos = (index % columns) * tile_width
        y_pos = (index // columns) * (tile_width + label_height)
        sheet.paste(image, (x_pos, y_pos + label_height))
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((x_pos, y_pos, x_pos + tile_width, y_pos + label_height), fill=(20, 20, 20))
        draw.text((x_pos + 8, y_pos + 8), name, font=font, fill=(245, 245, 245))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _env_args_from_manifest(manifest: dict[str, Any]) -> argparse.Namespace:
    original_args = dict(manifest["original_train_args"])
    env_args = argparse.Namespace(**original_args)
    env_args.width = int(manifest["width"])
    env_args.height = int(manifest["height"])
    env_args.num_ants = int(manifest["num_ants"])
    env_args.food_count = int(manifest["food_total"])
    env_args.food_sources = int(manifest["food_sources"])
    env_args.max_steps = int(manifest["requested_sim_steps"])
    rollout_overrides = dict(manifest.get("rollout_overrides") or {})
    for key, value in rollout_overrides.items():
        setattr(env_args, key, value)
    env_args.width = int(manifest["width"])
    env_args.height = int(manifest["height"])
    env_args.num_ants = int(manifest["num_ants"])
    env_args.food_count = int(manifest["food_total"])
    env_args.food_sources = int(manifest["food_sources"])
    env_args.max_steps = int(manifest["requested_sim_steps"])
    return env_args


def _reset_options_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, tuple[int, int] | list[tuple[int, int]]] | None:
    raw = manifest.get("reset_options")
    if not raw:
        return None
    options: dict[str, tuple[int, int] | list[tuple[int, int]]] = {}
    if "hub_pos" in raw:
        options["hub_pos"] = tuple(int(value) for value in raw["hub_pos"])
    if "food_positions" in raw:
        options["food_positions"] = [
            tuple(int(value) for value in position)
            for position in raw["food_positions"]
        ]
    return options


def _frame_steps(states_path: Path, *, max_frames: int | None) -> np.ndarray:
    with np.load(states_path) as states:
        frame_steps = np.asarray(states["frame_steps"], dtype=np.int32)
    if max_frames is None:
        return frame_steps
    if int(max_frames) <= 0:
        raise ValueError("max-frames must be positive")
    return frame_steps[: int(max_frames)]


def _palette_json(palette: Palette) -> dict[str, Any]:
    return {
        "name": palette.name,
        "description": palette.description,
        "background": list(palette.background),
        "byte_low": list(palette.byte_low),
        "byte_high": list(palette.byte_high),
        "byte_alpha": float(palette.byte_alpha),
        "hub": list(palette.hub),
        "food_outer": list(palette.food_outer),
        "food_center": list(palette.food_center),
        "normal_ant": list(palette.normal_ant),
        "carrying_ant": list(palette.carrying_ant),
        "obstacle": list(palette.obstacle),
    }


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
    raise SystemExit(main())
