"""Observation builders shared by Torch MAPPO training and evaluation."""

from __future__ import annotations

import numpy as np
import torch

from ant_byte_env import DEFAULT_WRITE_BITS, MAX_WRITE_BITS, max_write_value

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
    actor_vision_radius: int = 2,
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
    local_food = build_local_food_patches(
        food,
        obs["ants_pos"],
        radius=actor_vision_radius,
        food_scale=food_scale,
    )
    local_ants_count = build_local_grid_patches(
        ants_count,
        obs["ants_pos"],
        radius=actor_vision_radius,
    )
    local_ants_count = local_ants_count / ant_count_scale
    local_byte_bits = build_local_byte_bit_patches(
        obs["bytes"],
        obs["ants_pos"],
        radius=actor_vision_radius,
        write_bits=write_bits,
    )
    local_hub = build_local_hub_patches(
        obs["hub_pos"],
        obs["ants_pos"],
        grid_height=food.shape[1],
        grid_width=food.shape[2],
        radius=actor_vision_radius,
    )
    local_border = build_local_border_patches(
        obs["ants_pos"],
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
    food_scale: int,
) -> torch.Tensor:
    """Return flattened local food grids centered on each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    return build_local_grid_patches(food, ants_pos, radius=radius) / max(float(food_scale), 1.0)


def build_local_byte_bit_patches(
    bytes_grid: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    radius: int,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> torch.Tensor:
    """Return flattened local bit-plane patches for writable tile values."""

    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    bit_patches = []
    bytes_long = bytes_grid.long()
    for bit_index in range(write_bits):
        bit_grid = ((bytes_long >> bit_index) & 1).float()
        bit_patches.append(build_local_grid_patches(bit_grid, ants_pos, radius=radius))
    return torch.cat(bit_patches, dim=-1)


def build_local_grid_patches(
    grid: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Return flattened local grid patches centered on each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    batch_size, grid_height, grid_width = grid.shape
    num_agents = ants_pos.shape[1]
    patch_width = 2 * radius + 1
    patches = torch.zeros(
        (batch_size, num_agents, patch_width, patch_width),
        dtype=torch.float32,
        device=grid.device,
    )

    for batch_index in range(batch_size):
        for agent_index in range(num_agents):
            ant_x = int(ants_pos[batch_index, agent_index, 0])
            ant_y = int(ants_pos[batch_index, agent_index, 1])
            for patch_y, grid_y in enumerate(range(ant_y - radius, ant_y + radius + 1)):
                if not 0 <= grid_y < grid_height:
                    continue
                for patch_x, grid_x in enumerate(range(ant_x - radius, ant_x + radius + 1)):
                    if 0 <= grid_x < grid_width:
                        patches[batch_index, agent_index, patch_y, patch_x] = grid[
                            batch_index,
                            grid_y,
                            grid_x,
                        ]

    return patches.reshape(batch_size, num_agents, -1)


def build_local_border_patches(
    ants_pos: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
) -> torch.Tensor:
    """Return flattened out-of-bounds masks centered on each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    offsets = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.long,
        device=ants_pos.device,
    )
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_pairs = torch.stack([offset_x.reshape(-1), offset_y.reshape(-1)], dim=-1)
    positions = ants_pos.long().unsqueeze(2) + offset_pairs.view(1, 1, -1, 2)
    x_pos = positions[..., 0]
    y_pos = positions[..., 1]
    valid = (0 <= x_pos) & (x_pos < grid_width) & (0 <= y_pos) & (y_pos < grid_height)
    return (~valid).float()


def build_local_hub_patches(
    hub_pos: torch.Tensor,
    ants_pos: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
) -> torch.Tensor:
    """Return flattened local hub masks centered on each ant."""

    hub_grid = torch.zeros(
        (hub_pos.shape[0], grid_height, grid_width),
        dtype=torch.float32,
        device=hub_pos.device,
    )
    for batch_index in range(hub_pos.shape[0]):
        hub_x = int(hub_pos[batch_index, 0])
        hub_y = int(hub_pos[batch_index, 1])
        hub_grid[batch_index, hub_y, hub_x] = 1.0
    return build_local_grid_patches(hub_grid, ants_pos, radius=radius)


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
