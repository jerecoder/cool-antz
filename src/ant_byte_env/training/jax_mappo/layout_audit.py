"""Temporary layout audit logging for randomized MAPPO training runs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _food_positions_from_grid(food_grid: np.ndarray) -> list[list[int]]:
    positions: list[list[int]] = []
    for y, x in np.argwhere(np.asarray(food_grid) > 0):
        positions.append([int(x), int(y)])
    return sorted(positions)


def _normalize_positions(
    positions: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[list[int]]:
    normalized: list[list[int]] = []
    for x_raw, y_raw in np.asarray(positions).reshape((-1, 2)):
        x = int(x_raw)
        y = int(y_raw)
        if 0 <= x < width and 0 <= y < height:
            normalized.append([x, y])
    return sorted(normalized)


def _layout_hash(
    *,
    width: int,
    height: int,
    hub_pos: list[int],
    food_positions: list[list[int]],
    obstacles: np.ndarray | None,
) -> tuple[str, str | None]:
    obstacle_hash = None
    obstacle_array = None if obstacles is None else np.asarray(obstacles, dtype=bool)
    if obstacle_array is not None and bool(np.any(obstacle_array)):
        obstacle_hash = hashlib.sha256(obstacle_array.tobytes()).hexdigest()[:16]
    payload: dict[str, Any] = {
        "food_positions": food_positions,
        "height": int(height),
        "hub_pos": hub_pos,
        "width": int(width),
    }
    if obstacle_hash is not None:
        payload["obstacle_hash"] = obstacle_hash
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return digest, obstacle_hash


def _tile_image(
    *,
    width: int,
    height: int,
    hub_pos: list[int],
    food_positions: list[list[int]],
    obstacles: np.ndarray | None,
) -> np.ndarray:
    tile_size = max(4, min(10, 480 // max(width, height, 1)))
    image = np.full(
        (height * tile_size, width * tile_size, 3),
        np.array([244, 241, 232], dtype=np.uint8),
        dtype=np.uint8,
    )
    grid_color = np.array([212, 207, 194], dtype=np.uint8)
    image[::tile_size, :, :] = grid_color
    image[:, ::tile_size, :] = grid_color

    obstacle_array = None if obstacles is None else np.asarray(obstacles, dtype=bool)
    if obstacle_array is not None and obstacle_array.shape == (height, width):
        for y, x in np.argwhere(obstacle_array):
            _paint_tile(
                image,
                x=int(x),
                y=int(y),
                tile_size=tile_size,
                color=(47, 52, 60),
            )
    for x, y in food_positions:
        _paint_tile(image, x=x, y=y, tile_size=tile_size, color=(215, 115, 45))
    _paint_tile(
        image,
        x=int(hub_pos[0]),
        y=int(hub_pos[1]),
        tile_size=tile_size,
        color=(42, 120, 180),
    )
    return image


def _paint_tile(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    tile_size: int,
    color: tuple[int, int, int],
) -> None:
    y0 = y * tile_size
    x0 = x * tile_size
    image[y0 : y0 + tile_size, x0 : x0 + tile_size, :] = np.asarray(
        color,
        dtype=np.uint8,
    )


class LayoutAuditTracker:
    """Append reset-layout records and occasional map snapshots to a local folder."""

    def __init__(
        self,
        *,
        audit_dir: Path | None,
        snapshot_interval: int,
        width: int,
        height: int,
        stage_name: str,
        run_name: str,
    ) -> None:
        self.audit_dir = None if audit_dir is None else Path(audit_dir)
        self.snapshot_interval = int(snapshot_interval)
        self.width = int(width)
        self.height = int(height)
        self.stage_name = str(stage_name)
        self.run_name = str(run_name)
        self._session_id = f"{int(time.time())}_{os.getpid()}"
        self._record_count = 0
        self._snapshot_count = 0
        self._same_env_repeats = 0
        self._seen_hashes: set[str] = set()
        self._last_hash_by_env: dict[int, str] = {}
        if self.enabled:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            (self.audit_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, args: Any, *, run_name: str) -> "LayoutAuditTracker":
        return cls(
            audit_dir=getattr(args, "layout_audit_dir", None),
            snapshot_interval=int(getattr(args, "layout_audit_snapshot_interval", 0)),
            width=int(args.width),
            height=int(args.height),
            stage_name=f"{int(args.width)}x{int(args.height)}",
            run_name=run_name,
        )

    @property
    def enabled(self) -> bool:
        return self.audit_dir is not None

    def observe_observations(
        self,
        *,
        obs: Mapping[str, Any],
        update: int,
        global_step: int,
        reason: str,
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        food = np.asarray(obs["food"])
        hubs = np.asarray(obs["hub_pos"])
        obstacles = None if "obstacles" not in obs else np.asarray(obs["obstacles"])
        if food.ndim == 2:
            food = food[None, ...]
        if hubs.ndim == 1:
            hubs = hubs[None, ...]
        if obstacles is not None and obstacles.ndim == 2:
            obstacles = obstacles[None, ...]
        env_count = min(int(food.shape[0]), int(hubs.shape[0]))
        for env_index in range(env_count):
            env_obstacles = None if obstacles is None else obstacles[env_index]
            self._record_layout(
                reason=reason,
                update=update,
                global_step=global_step,
                env_index=env_index,
                rollout_step_index=None,
                hub_pos=[int(hubs[env_index, 0]), int(hubs[env_index, 1])],
                food_positions=_food_positions_from_grid(food[env_index]),
                obstacles=env_obstacles,
            )
        return self.metrics()

    def observe_rollout_resets(
        self,
        *,
        rollout: Any,
        update: int,
        global_step: int,
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        dones = np.asarray(rollout.dones, dtype=bool)
        hubs = np.asarray(rollout.reset_hub_pos)
        foods = np.asarray(rollout.reset_food_positions)
        for rollout_step_index, env_index in zip(*np.nonzero(dones), strict=True):
            self._record_layout(
                reason="reset",
                update=update,
                global_step=global_step,
                env_index=int(env_index),
                rollout_step_index=int(rollout_step_index),
                hub_pos=[
                    int(hubs[rollout_step_index, env_index, 0]),
                    int(hubs[rollout_step_index, env_index, 1]),
                ],
                food_positions=_normalize_positions(
                    foods[rollout_step_index, env_index],
                    width=self.width,
                    height=self.height,
                ),
                obstacles=None,
            )
        return self.metrics()

    def metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return {
            "layout_audit_records": float(self._record_count),
            "layout_audit_unique_layouts": float(len(self._seen_hashes)),
            "layout_audit_same_env_repeats": float(self._same_env_repeats),
            "layout_audit_snapshots": float(self._snapshot_count),
        }

    def _record_layout(
        self,
        *,
        reason: str,
        update: int,
        global_step: int,
        env_index: int,
        rollout_step_index: int | None,
        hub_pos: list[int],
        food_positions: list[list[int]],
        obstacles: np.ndarray | None,
    ) -> None:
        layout_hash, obstacle_hash = _layout_hash(
            width=self.width,
            height=self.height,
            hub_pos=hub_pos,
            food_positions=food_positions,
            obstacles=obstacles,
        )
        previous_hash = self._last_hash_by_env.get(env_index)
        seen_before = layout_hash in self._seen_hashes
        changed_from_previous = previous_hash is None or previous_hash != layout_hash
        if previous_hash is not None and previous_hash == layout_hash:
            self._same_env_repeats += 1
        self._seen_hashes.add(layout_hash)
        self._last_hash_by_env[env_index] = layout_hash
        self._record_count += 1

        record: dict[str, Any] = {
            "audit_index": self._record_count,
            "changed_from_previous_for_env": changed_from_previous,
            "env_index": int(env_index),
            "food_positions": food_positions,
            "global_step": int(global_step),
            "height": self.height,
            "hub_pos": hub_pos,
            "layout_hash": layout_hash,
            "previous_layout_hash_for_env": previous_hash,
            "reason": str(reason),
            "rollout_step_index": rollout_step_index,
            "run_name": self.run_name,
            "seen_before": seen_before,
            "session_id": self._session_id,
            "stage_name": self.stage_name,
            "update": int(update),
            "width": self.width,
        }
        if obstacle_hash is not None:
            record["obstacle_hash"] = obstacle_hash
        if (
            self.snapshot_interval > 0
            and self._record_count % self.snapshot_interval == 0
        ):
            snapshot_path = self._write_snapshot(
                record=record,
                hub_pos=hub_pos,
                food_positions=food_positions,
                obstacles=obstacles,
            )
            record["snapshot_path"] = str(snapshot_path)
        records_path = self.audit_dir / "layout_records.jsonl"
        with records_path.open("a", encoding="utf-8") as records_file:
            records_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_snapshot(
        self,
        *,
        record: Mapping[str, Any],
        hub_pos: list[int],
        food_positions: list[list[int]],
        obstacles: np.ndarray | None,
    ) -> Path:
        import imageio.v2 as imageio

        self._snapshot_count += 1
        stage_name = _safe_name(self.stage_name)
        reason = _safe_name(str(record["reason"]))
        snapshot_path = (
            self.audit_dir
            / "snapshots"
            / (
                f"{self._session_id}_{int(record['audit_index']):06d}_{stage_name}_"
                f"{reason}_u{int(record['update']):06d}_env{int(record['env_index']):03d}_"
                f"{record['layout_hash']}.png"
            )
        )
        imageio.imwrite(
            snapshot_path,
            _tile_image(
                width=self.width,
                height=self.height,
                hub_pos=hub_pos,
                food_positions=food_positions,
                obstacles=obstacles,
            ),
        )
        return snapshot_path
