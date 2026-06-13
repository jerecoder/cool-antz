"""Observation builders shared by Torch MAPPO training and evaluation."""

from __future__ import annotations

import numpy as np
import torch

from ant_byte_env import DEFAULT_WRITE_BITS, MAX_WRITE_BITS, max_write_value
from ant_byte_env.env import (
    DEFAULT_ACTOR_VISION_DEPTH,
    DEFAULT_ACTOR_VISION_WIDTH,
    DEFAULT_FACING,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
)

TensorObs = dict[str, torch.Tensor]
NumpyObs = dict[str, np.ndarray]


def obs_to_tensor(obs: NumpyObs, device: torch.device) -> TensorObs:
    return {key: torch.as_tensor(value, device=device) for key, value in obs.items()}


def _position_scale(height: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [max(width - 1, 1), max(height - 1, 1)],
        dtype=torch.float32,
        device=device,
    )


def _normalize_positions(positions: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    scale = _position_scale(height, width, positions.device)
    return positions.float() / scale


def _resolve_observation_grid_shape(
    obs: TensorObs,
    *,
    obs_height: int | None,
    obs_width: int | None,
) -> tuple[int, int, int, int]:
    _, current_height, current_width = obs["food"].shape
    target_height = current_height if obs_height is None else obs_height
    target_width = current_width if obs_width is None else obs_width
    if target_height < current_height or target_width < current_width:
        raise ValueError(
            "Padded observation shape must be at least as large as the environment grid."
        )
    return current_height, current_width, target_height, target_width


def _pad_grid(grid: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if grid.shape[1:] == (height, width):
        return grid

    padded = torch.zeros(
        (grid.shape[0], height, width),
        dtype=grid.dtype,
        device=grid.device,
    )
    padded[:, : grid.shape[1], : grid.shape[2]] = grid
    return padded


def _ants_count_grid(obs: TensorObs, *, height: int, width: int) -> torch.Tensor:
    if "ants_count" in obs:
        return obs["ants_count"].float()

    ants_pos = obs["ants_pos"].long()
    batch_size = ants_pos.shape[0]
    counts = torch.zeros((batch_size, height, width), dtype=torch.float32, device=ants_pos.device)
    for batch_index in range(batch_size):
        for ant_x, ant_y in ants_pos[batch_index]:
            counts[batch_index, int(ant_y), int(ant_x)] += 1.0
    return counts


def build_central_observations(
    obs: TensorObs,
    *,
    food_scale: int,
    write_bits: int = DEFAULT_WRITE_BITS,
    obs_width: int | None = None,
    obs_height: int | None = None,
) -> torch.Tensor:
    """Flatten and normalize the full state used by the centralized critic."""

    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    food = obs["food"].float()
    bytes_grid = obs["bytes"].float()
    batch_size, _, _ = food.shape
    ant_count_scale = max(float(obs["ants_pos"].shape[1]), 1.0)
    current_height, current_width, target_height, target_width = (
        _resolve_observation_grid_shape(
            obs,
            obs_height=obs_height,
            obs_width=obs_width,
        )
    )
    ants_pos = _normalize_positions(
        obs["ants_pos"],
        height=target_height,
        width=target_width,
    )
    ants_count = _ants_count_grid(obs, height=current_height, width=current_width)
    hub_pos = _normalize_positions(
        obs["hub_pos"],
        height=target_height,
        width=target_width,
    )
    ants_carrying = obs["ants_carrying"].float()
    ants_count_norm = _pad_grid(
        ants_count / ant_count_scale,
        height=target_height,
        width=target_width,
    )
    food_norm = _pad_grid(
        food / max(float(food_scale), 1.0),
        height=target_height,
        width=target_width,
    )
    bytes_norm = _pad_grid(
        bytes_grid / max(float(max_write_value(write_bits)), 1.0),
        height=target_height,
        width=target_width,
    )
    grid_size = torch.tensor(
        [
            current_width / max(float(target_width), 1.0),
            current_height / max(float(target_height), 1.0),
        ],
        dtype=torch.float32,
        device=food.device,
    ).expand(batch_size, -1)

    return torch.cat(
        [
            ants_pos.reshape(batch_size, -1),
            ants_carrying.reshape(batch_size, -1),
            ants_count_norm.reshape(batch_size, -1),
            food_norm.reshape(batch_size, -1),
            bytes_norm.reshape(batch_size, -1),
            hub_pos.reshape(batch_size, -1),
            grid_size,
        ],
        dim=-1,
    )


def build_actor_observations(
    obs: TensorObs,
    central_obs: torch.Tensor | None = None,
    *,
    food_scale: int = 1,
    actor_vision_radius: int = DEFAULT_ACTOR_VISION_DEPTH,
    write_bits: int = DEFAULT_WRITE_BITS,
    obs_width: int | None = None,
    obs_height: int | None = None,
) -> torch.Tensor:
    """Create local per-ant actor observations for the decentralized policy."""

    del central_obs
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")

    del obs_width, obs_height
    food = obs["food"].float()
    ant_count_scale = max(float(obs["ants_pos"].shape[1]), 1.0)
    ants_count = _ants_count_grid(obs, height=food.shape[1], width=food.shape[2])
    own_carrying = obs["ants_carrying"].float().unsqueeze(-1)
    ants_facing = obs.get("ants_facing")
    if ants_facing is None:
        ants_facing = torch.full(
            obs["ants_pos"].shape[:2],
            DEFAULT_FACING,
            dtype=torch.long,
            device=obs["ants_pos"].device,
        )
    local_food = build_local_food_patches(
        food,
        obs["ants_pos"],
        radius=actor_vision_radius,
        ants_facing=ants_facing,
        food_scale=food_scale,
    )
    local_ants_count = build_local_grid_patches(
        ants_count,
        obs["ants_pos"],
        radius=actor_vision_radius,
        ants_facing=ants_facing,
    )
    local_ants_count = local_ants_count / ant_count_scale
    local_byte_bits = build_local_byte_bit_patches(
        obs["bytes"],
        obs["ants_pos"],
        radius=actor_vision_radius,
        ants_facing=ants_facing,
        write_bits=write_bits,
    )
    local_hub = build_local_hub_patches(
        obs["hub_pos"],
        obs["ants_pos"],
        ants_facing=ants_facing,
        grid_height=food.shape[1],
        grid_width=food.shape[2],
        radius=actor_vision_radius,
    )
    local_border = build_local_border_patches(
        obs["ants_pos"],
        ants_facing=ants_facing,
        grid_height=food.shape[1],
        grid_width=food.shape[2],
        radius=actor_vision_radius,
    )

    return torch.cat(
        [local_food, local_ants_count, local_byte_bits, local_hub, local_border, own_carrying],
        dim=-1,
    )


def build_local_food_patches(
    food: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    radius: int,
    ants_facing: torch.Tensor | None = None,
    food_scale: int,
) -> torch.Tensor:
    """Return flattened local food grids in front of each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    return build_local_grid_patches(
        food,
        ants_pos,
        radius=radius,
        ants_facing=ants_facing,
    ) / max(float(food_scale), 1.0)


def build_local_byte_bit_patches(
    bytes_grid: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    radius: int,
    ants_facing: torch.Tensor | None = None,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> torch.Tensor:
    """Return flattened local bit-plane patches for writable tile values."""

    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    bit_patches = []
    bytes_long = bytes_grid.long()
    for bit_index in range(write_bits):
        bit_grid = ((bytes_long >> bit_index) & 1).float()
        bit_patches.append(
            build_local_grid_patches(
                bit_grid,
                ants_pos,
                radius=radius,
                ants_facing=ants_facing,
            )
        )
    return torch.cat(bit_patches, dim=-1)


def build_local_grid_patches(
    grid: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    radius: int,
    ants_facing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return flattened 3-wide local grid patches in front of each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    batch_size, grid_height, grid_width = grid.shape
    num_agents = ants_pos.shape[1]
    patch_size = DEFAULT_ACTOR_VISION_WIDTH * radius
    patches = torch.zeros(
        (batch_size, num_agents, patch_size),
        dtype=torch.float32,
        device=grid.device,
    )
    if ants_facing is None:
        ants_facing = torch.full(
            ants_pos.shape[:2],
            DEFAULT_FACING,
            dtype=torch.long,
            device=ants_pos.device,
        )

    for batch_index in range(batch_size):
        for agent_index in range(num_agents):
            ant_x = int(ants_pos[batch_index, agent_index, 0])
            ant_y = int(ants_pos[batch_index, agent_index, 1])
            facing = int(ants_facing[batch_index, agent_index])
            forward_x, forward_y = _forward_delta(facing)
            right_x, right_y = -forward_y, forward_x
            patch_index = 0
            for depth in range(1, radius + 1):
                for lateral in range(
                    -(DEFAULT_ACTOR_VISION_WIDTH // 2),
                    DEFAULT_ACTOR_VISION_WIDTH // 2 + 1,
                ):
                    grid_x = ant_x + depth * forward_x + lateral * right_x
                    grid_y = ant_y + depth * forward_y + lateral * right_y
                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                        patches[batch_index, agent_index, patch_index] = grid[
                            batch_index,
                            grid_y,
                            grid_x,
                        ]
                    patch_index += 1

    return patches


def build_local_border_patches(
    ants_pos: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
    ants_facing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return flattened out-of-bounds masks for each forward vision patch."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    if ants_facing is None:
        ants_facing = torch.full(
            ants_pos.shape[:2],
            DEFAULT_FACING,
            dtype=torch.long,
            device=ants_pos.device,
        )
    positions = ants_pos.long().unsqueeze(2) + build_forward_vision_offsets(
        ants_facing.long(),
        depth=radius,
    )
    x_pos = positions[..., 0]
    y_pos = positions[..., 1]
    valid = (0 <= x_pos) & (x_pos < grid_width) & (0 <= y_pos) & (y_pos < grid_height)
    return (~valid).float()


def build_forward_vision_offsets(ants_facing: torch.Tensor, *, depth: int) -> torch.Tensor:
    if depth < 0:
        raise ValueError("depth must be non-negative.")

    device = ants_facing.device
    half_width = DEFAULT_ACTOR_VISION_WIDTH // 2
    depth_offsets = torch.repeat_interleave(
        torch.arange(1, depth + 1, dtype=torch.long, device=device),
        DEFAULT_ACTOR_VISION_WIDTH,
    )
    lateral_offsets = torch.tile(
        torch.arange(-half_width, half_width + 1, dtype=torch.long, device=device),
        (depth,),
    )
    facing = ants_facing.long()
    valid_facing = (
        (facing == MOVE_UP)
        | (facing == MOVE_RIGHT)
        | (facing == MOVE_DOWN)
        | (facing == MOVE_LEFT)
    )
    facing = torch.where(valid_facing, facing, torch.full_like(facing, DEFAULT_FACING))
    forward_x = torch.where(
        facing == MOVE_RIGHT,
        torch.ones_like(facing),
        torch.where(facing == MOVE_LEFT, -torch.ones_like(facing), torch.zeros_like(facing)),
    )
    forward_y = torch.where(
        facing == MOVE_DOWN,
        torch.ones_like(facing),
        torch.where(facing == MOVE_UP, -torch.ones_like(facing), torch.zeros_like(facing)),
    )
    right_x = -forward_y
    right_y = forward_x
    offset_x = forward_x.unsqueeze(-1) * depth_offsets + right_x.unsqueeze(-1) * lateral_offsets
    offset_y = forward_y.unsqueeze(-1) * depth_offsets + right_y.unsqueeze(-1) * lateral_offsets
    return torch.stack([offset_x, offset_y], dim=-1)


def _forward_delta(facing: int) -> tuple[int, int]:
    if facing == MOVE_RIGHT:
        return 1, 0
    if facing == MOVE_LEFT:
        return -1, 0
    if facing == MOVE_DOWN:
        return 0, 1
    if facing == MOVE_UP:
        return 0, -1
    return _forward_delta(DEFAULT_FACING)


def build_local_hub_patches(
    hub_pos: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
    ants_facing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return flattened local hub masks in front of each ant."""

    hub_grid = torch.zeros(
        (hub_pos.shape[0], grid_height, grid_width),
        dtype=torch.float32,
        device=hub_pos.device,
    )
    for batch_index in range(hub_pos.shape[0]):
        hub_x = int(hub_pos[batch_index, 0])
        hub_y = int(hub_pos[batch_index, 1])
        hub_grid[batch_index, hub_y, hub_x] = 1.0
    return build_local_grid_patches(
        hub_grid,
        ants_pos,
        radius=radius,
        ants_facing=ants_facing,
    )


def flatten_agent_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert joint movement/write actions to the env's interleaved action vector."""

    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError(f"joint actions must have shape (batch, ants, 2), got {actions.shape}.")
    batch_size, num_agents, _ = actions.shape
    flat_actions = torch.empty(
        (batch_size, num_agents * 2),
        dtype=torch.long,
        device=actions.device,
    )
    flat_actions[:, 0::2] = actions[..., 0].long()
    flat_actions[:, 1::2] = actions[..., 1].long()
    return flat_actions
