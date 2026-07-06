"""Observation builders for JAX MAPPO actors and centralized critics."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ant_byte_env import (
    DEFAULT_WRITE_BITS,
    MAX_WRITE_BITS,
    MOVEMENT_ACTION_COUNT,
    max_write_value,
)
from ant_byte_env.env import (
    DEFAULT_ACTOR_VISION_DEPTH,
    DEFAULT_FACING,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
)
from ant_byte_env.jax_env import JaxObs
from ant_byte_env.training.jax_mappo.types import CRITIC_AUX_FEATURE_DIM

def _normalize_positions(positions: jax.Array, *, height: int, width: int) -> jax.Array:
    scale = jnp.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=jnp.float32)
    return positions.astype(jnp.float32) / scale


def food_observation_scale(*, food_count: int | float, food_sources: int | None = None) -> float:
    if food_sources is None or int(food_sources) <= 0:
        return max(float(food_count), 1.0)
    return max(float(math.ceil(float(food_count) / float(food_sources))), 1.0)


def _facing_one_hot(ants_facing: jax.Array) -> jax.Array:
    facing_index = jnp.clip(
        ants_facing.astype(jnp.int32) - 1,
        0,
        MOVEMENT_ACTION_COUNT - 2,
    )
    return jax.nn.one_hot(
        facing_index,
        MOVEMENT_ACTION_COUNT - 1,
        dtype=jnp.float32,
    )


def _agent_identity_features(
    ants_pos: jax.Array,
    *,
    agent_identity_types: int | None = None,
) -> jax.Array:
    batch_size, num_agents = ants_pos.shape[:2]
    if num_agents <= 1:
        return jnp.zeros((batch_size, num_agents, 0), dtype=jnp.float32)
    if agent_identity_types is None:
        identity_count = num_agents
    else:
        identity_count = int(agent_identity_types)
        if identity_count <= 0:
            raise ValueError("agent_identity_types must be positive.")
    identity = jax.nn.one_hot(
        jnp.arange(num_agents) % identity_count,
        identity_count,
        dtype=jnp.float32,
    )
    return jnp.broadcast_to(
        identity[None, :, :],
        (batch_size, num_agents, identity_count),
    )


def _ants_facing_or_default(obs: JaxObs) -> jax.Array:
    ants_facing = obs.get("ants_facing")
    if ants_facing is None:
        return jnp.full(
            obs["ants_pos"].shape[:2],
            DEFAULT_FACING,
            dtype=jnp.int32,
        )
    return ants_facing.astype(jnp.int32)


def _resolve_observation_grid_shape(
    obs: JaxObs,
    *,
    obs_height: int | None,
    obs_width: int | None,
) -> tuple[int, int, int, int]:
    _, current_height, current_width = obs["food"].shape
    target_height = current_height if obs_height is None else int(obs_height)
    target_width = current_width if obs_width is None else int(obs_width)
    if target_height < current_height or target_width < current_width:
        raise ValueError("Padded observation shape must be at least the environment grid.")
    return current_height, current_width, target_height, target_width


def _active_grid_size(
    obs: JaxObs,
    *,
    fallback_height: int,
    fallback_width: int,
) -> jax.Array:
    active_grid_size = obs.get("active_grid_size")
    batch_size = obs["food"].shape[0]
    if active_grid_size is None:
        return jnp.broadcast_to(
            jnp.asarray([fallback_width, fallback_height], dtype=jnp.float32),
            (batch_size, 2),
        )
    return active_grid_size.astype(jnp.float32).reshape((batch_size, 2))


def _pad_grid(grid: jax.Array, *, height: int, width: int) -> jax.Array:
    if grid.shape[1:] == (height, width):
        return grid
    padded = jnp.zeros((grid.shape[0], height, width), dtype=grid.dtype)
    return padded.at[:, : grid.shape[1], : grid.shape[2]].set(grid)


def _obs_scalar_column(obs: JaxObs, key: str, *, batch_size: int) -> jax.Array:
    value = obs.get(key)
    if value is None:
        return jnp.zeros((batch_size, 1), dtype=jnp.float32)
    array = jnp.asarray(value, dtype=jnp.float32).reshape((batch_size, -1))
    return array[:, :1]


def _critic_aux_features(
    obs: JaxObs,
    *,
    food: jax.Array,
    bytes_grid: jax.Array,
    ants_pos: jax.Array,
    ants_carrying: jax.Array,
    hub_pos: jax.Array,
    food_scale: int | float,
    target_height: int,
    target_width: int,
) -> jax.Array:
    batch_size, current_height, current_width = food.shape
    map_distance = max(float(target_width + target_height - 2), 1.0)
    stage_distance = jnp.clip(
        _obs_scalar_column(
            obs,
            "distance_curriculum_stage_distance",
            batch_size=batch_size,
        )
        / map_distance,
        0.0,
        1.0,
    )
    stage_index = jnp.clip(
        _obs_scalar_column(
            obs,
            "distance_curriculum_stage_index",
            batch_size=batch_size,
        )
        / 16.0,
        0.0,
        1.0,
    )

    food_mass = jnp.sum(food, axis=(1, 2), keepdims=False)[:, None]
    remaining_food = jnp.clip(food_mass / max(float(food_scale), 1.0), 0.0, 1.0)
    carrier_fraction = jnp.mean(ants_carrying.astype(jnp.float32), axis=1, keepdims=True)
    nonzero_byte_fraction = (
        jnp.sum((bytes_grid > 0).astype(jnp.float32), axis=(1, 2), keepdims=False)[:, None]
        / max(float(current_height * current_width), 1.0)
    )

    normalized_ant_distances = (
        jnp.sum(jnp.abs(ants_pos - hub_pos[:, None, :]), axis=-1) / 2.0
    )
    mean_ant_hub_distance = jnp.mean(normalized_ant_distances, axis=1, keepdims=True)
    carrier_weights = ants_carrying.astype(jnp.float32)
    carrier_count = jnp.sum(carrier_weights, axis=1, keepdims=True)
    mean_carrier_hub_distance = (
        jnp.sum(normalized_ant_distances * carrier_weights, axis=1, keepdims=True)
        / jnp.maximum(carrier_count, 1.0)
    )
    mean_carrier_hub_distance = jnp.where(
        carrier_count > 0.0,
        mean_carrier_hub_distance,
        0.0,
    )

    x_coords = jnp.linspace(0.0, 1.0, current_width, dtype=jnp.float32)[None, None, :]
    y_coords = jnp.linspace(0.0, 1.0, current_height, dtype=jnp.float32)[None, :, None]
    safe_food_mass = jnp.maximum(food_mass, 1.0)
    food_centroid_x = (
        jnp.sum(food * x_coords, axis=(1, 2), keepdims=False)[:, None] / safe_food_mass
    )
    food_centroid_y = (
        jnp.sum(food * y_coords, axis=(1, 2), keepdims=False)[:, None] / safe_food_mass
    )
    has_food = food_mass > 0.0
    food_centroid_x = jnp.where(has_food, food_centroid_x, hub_pos[:, 0:1])
    food_centroid_y = jnp.where(has_food, food_centroid_y, hub_pos[:, 1:2])
    food_hub_abs_dx = jnp.abs(food_centroid_x - hub_pos[:, 0:1])
    food_hub_abs_dy = jnp.abs(food_centroid_y - hub_pos[:, 1:2])
    food_hub_manhattan = (food_hub_abs_dx + food_hub_abs_dy) / 2.0

    aux = jnp.concatenate(
        [
            stage_distance,
            stage_index,
            remaining_food,
            carrier_fraction,
            nonzero_byte_fraction,
            mean_ant_hub_distance,
            mean_carrier_hub_distance,
            food_centroid_x,
            food_centroid_y,
            food_hub_abs_dx,
            food_hub_abs_dy,
            food_hub_manhattan,
        ],
        axis=-1,
    ).astype(jnp.float32)
    if aux.shape[-1] != CRITIC_AUX_FEATURE_DIM:
        raise ValueError(f"critic aux features must have {CRITIC_AUX_FEATURE_DIM} columns.")
    return aux


def _ants_count_grid(obs: JaxObs, *, height: int, width: int) -> jax.Array:
    if "ants_count" in obs:
        return obs["ants_count"].astype(jnp.float32)

    ants_pos = obs["ants_pos"].astype(jnp.int32)
    batch_size = ants_pos.shape[0]
    batch_index = jnp.arange(batch_size)[:, None]
    return jnp.zeros((batch_size, height, width), dtype=jnp.float32).at[
        batch_index,
        ants_pos[..., 1],
        ants_pos[..., 0],
    ].add(1.0)


def build_central_observations(
    obs: JaxObs,
    *,
    food_scale: int,
    write_bits: int = DEFAULT_WRITE_BITS,
    obs_width: int | None = None,
    obs_height: int | None = None,
) -> jax.Array:
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    food = obs["food"].astype(jnp.float32)
    bytes_grid = obs["bytes"].astype(jnp.float32)
    batch_size = food.shape[0]
    ant_count_scale = max(float(obs["ants_pos"].shape[1]), 1.0)
    current_height, current_width, target_height, target_width = _resolve_observation_grid_shape(
        obs,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    active_grid_size = _active_grid_size(
        obs,
        fallback_height=current_height,
        fallback_width=current_width,
    )
    ants_pos = _normalize_positions(obs["ants_pos"], height=target_height, width=target_width)
    ants_count = _ants_count_grid(obs, height=current_height, width=current_width)
    hub_pos = _normalize_positions(obs["hub_pos"], height=target_height, width=target_width)
    ants_carrying = obs["ants_carrying"].astype(jnp.float32)
    ants_facing = _facing_one_hot(_ants_facing_or_default(obs))
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
    grid_size = active_grid_size / jnp.asarray(
        [max(float(target_width), 1.0), max(float(target_height), 1.0)],
        dtype=jnp.float32,
    )
    critic_aux = _critic_aux_features(
        obs,
        food=food,
        bytes_grid=bytes_grid,
        ants_pos=ants_pos,
        ants_carrying=ants_carrying,
        hub_pos=hub_pos,
        food_scale=food_scale,
        target_height=target_height,
        target_width=target_width,
    )
    return jnp.concatenate(
        [
            ants_pos.reshape(batch_size, -1),
            ants_carrying.reshape(batch_size, -1),
            ants_facing.reshape(batch_size, -1),
            ants_count_norm.reshape(batch_size, -1),
            food_norm.reshape(batch_size, -1),
            bytes_norm.reshape(batch_size, -1),
            hub_pos.reshape(batch_size, -1),
            grid_size,
            critic_aux,
        ],
        axis=-1,
    )


def build_actor_observations(
    obs: JaxObs,
    *,
    food_scale: int = 1,
    actor_vision_radius: int = DEFAULT_ACTOR_VISION_DEPTH,
    write_bits: int = DEFAULT_WRITE_BITS,
    agent_identity_types: int | None = None,
    obs_width: int | None = None,
    obs_height: int | None = None,
) -> jax.Array:
    del obs_width, obs_height
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")

    food = obs["food"].astype(jnp.float32)
    ant_count_scale = max(float(obs["ants_pos"].shape[1]), 1.0)
    ants_count = _ants_count_grid(obs, height=food.shape[1], width=food.shape[2])
    own_carrying = obs["ants_carrying"].astype(jnp.float32)[..., None]
    ants_facing = _ants_facing_or_default(obs)
    own_facing = _facing_one_hot(ants_facing)
    local_food = build_local_grid_patches(
        food,
        obs["ants_pos"],
        radius=actor_vision_radius,
        ants_facing=ants_facing,
    )
    local_food = local_food / max(float(food_scale), 1.0)
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
    active_grid_size = _active_grid_size(
        obs,
        fallback_height=food.shape[1],
        fallback_width=food.shape[2],
    )
    local_border = build_local_border_patches(
        obs["ants_pos"],
        ants_facing=ants_facing,
        grid_height=active_grid_size[:, 1],
        grid_width=active_grid_size[:, 0],
        radius=actor_vision_radius,
    )
    obstacles = obs.get("obstacles")
    if obstacles is None:
        obstacles = jnp.zeros_like(food, dtype=jnp.float32)
    local_obstacles = build_local_grid_patches(
        obstacles.astype(jnp.float32),
        obs["ants_pos"],
        radius=actor_vision_radius,
        ants_facing=ants_facing,
    )
    local_border = jnp.maximum(local_border, local_obstacles)
    features = [
        local_food,
        local_ants_count,
        local_byte_bits,
        local_hub,
        local_border,
        _agent_identity_features(
            obs["ants_pos"],
            agent_identity_types=agent_identity_types,
        ),
        own_carrying,
        own_facing,
    ]
    return jnp.concatenate(features, axis=-1)


def build_local_byte_bit_patches(
    bytes_grid: jax.Array,
    ants_pos: jax.Array,
    *,
    radius: int,
    ants_facing: jax.Array | None = None,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> jax.Array:
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    bytes_int = bytes_grid.astype(jnp.int32)
    bit_patches = []
    for bit_index in range(write_bits):
        bit_grid = ((bytes_int >> bit_index) & 1).astype(jnp.float32)
        bit_patches.append(
            build_local_grid_patches(
                bit_grid,
                ants_pos,
                radius=radius,
                ants_facing=ants_facing,
            )
        )
    return jnp.concatenate(bit_patches, axis=-1)


def build_local_grid_patches(
    grid: jax.Array,
    ants_pos: jax.Array,
    *,
    radius: int,
    ants_facing: jax.Array | None = None,
) -> jax.Array:
    """Return flattened facing-aware local grid patches around each ant."""

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    if ants_facing is None:
        ants_facing = jnp.full(ants_pos.shape[:2], DEFAULT_FACING, dtype=jnp.int32)

    batch_size, grid_height, grid_width = grid.shape
    offset_pairs = build_forward_vision_offsets(
        ants_facing.astype(jnp.int32),
        depth=radius,
    )
    positions = ants_pos.astype(jnp.int32)[:, :, None, :] + offset_pairs
    x_pos = positions[..., 0]
    y_pos = positions[..., 1]
    valid = (0 <= x_pos) & (x_pos < grid_width) & (0 <= y_pos) & (y_pos < grid_height)
    clipped_x = jnp.clip(x_pos, 0, grid_width - 1)
    clipped_y = jnp.clip(y_pos, 0, grid_height - 1)
    batch_index = jnp.arange(batch_size)[:, None, None]
    values = grid[batch_index, clipped_y, clipped_x].astype(jnp.float32)
    return jnp.where(valid, values, 0.0)


def build_local_border_patches(
    ants_pos: jax.Array,
    *,
    grid_height: int | jax.Array,
    grid_width: int | jax.Array,
    radius: int,
    ants_facing: jax.Array | None = None,
) -> jax.Array:
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    if ants_facing is None:
        ants_facing = jnp.full(ants_pos.shape[:2], DEFAULT_FACING, dtype=jnp.int32)
    offset_pairs = build_forward_vision_offsets(
        ants_facing.astype(jnp.int32),
        depth=radius,
    )
    positions = ants_pos.astype(jnp.int32)[:, :, None, :] + offset_pairs
    x_pos = positions[..., 0]
    y_pos = positions[..., 1]
    width_limit = _grid_limit_for_positions(grid_width)
    height_limit = _grid_limit_for_positions(grid_height)
    valid = (0 <= x_pos) & (x_pos < width_limit) & (0 <= y_pos) & (y_pos < height_limit)
    return jnp.where(valid, 0.0, 1.0).astype(jnp.float32)


def _grid_limit_for_positions(limit: int | jax.Array) -> jax.Array:
    limit_array = jnp.asarray(limit, dtype=jnp.int32)
    if limit_array.ndim == 0:
        return limit_array
    return limit_array.reshape((limit_array.shape[0], 1, 1))


def build_forward_vision_offsets(ants_facing: jax.Array, *, depth: int) -> jax.Array:
    if depth < 0:
        raise ValueError("depth must be non-negative.")

    axis = jnp.arange(-depth, depth + 1, dtype=jnp.int32)
    offset_y = jnp.repeat(axis, 2 * depth + 1)
    offset_x = jnp.tile(axis, 2 * depth + 1)
    offsets = jnp.stack([offset_x, offset_y], axis=-1)
    offset_x = offsets[:, 0]
    offset_y = offsets[:, 1]
    right_offsets = offsets
    down_offsets = jnp.stack([-offset_y, offset_x], axis=-1)
    left_offsets = jnp.stack([-offset_x, -offset_y], axis=-1)
    up_offsets = jnp.stack([offset_y, -offset_x], axis=-1)
    facing = jnp.where(
        (ants_facing == MOVE_UP)
        | (ants_facing == MOVE_RIGHT)
        | (ants_facing == MOVE_DOWN)
        | (ants_facing == MOVE_LEFT),
        ants_facing,
        DEFAULT_FACING,
    )
    facing = facing[..., None, None]
    expanded = jnp.broadcast_to(right_offsets, (*ants_facing.shape, offsets.shape[0], 2))
    expanded = jnp.where(facing == MOVE_DOWN, down_offsets, expanded)
    expanded = jnp.where(facing == MOVE_LEFT, left_offsets, expanded)
    return jnp.where(facing == MOVE_UP, up_offsets, expanded)


def build_local_hub_patches(
    hub_pos: jax.Array,
    ants_pos: jax.Array,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
    ants_facing: jax.Array | None = None,
) -> jax.Array:
    batch_size = hub_pos.shape[0]
    hub_grid = jnp.zeros((batch_size, grid_height, grid_width), dtype=jnp.float32)
    batch_index = jnp.arange(batch_size)
    hub_grid = hub_grid.at[batch_index, hub_pos[:, 1], hub_pos[:, 0]].set(1.0)
    return build_local_grid_patches(
        hub_grid,
        ants_pos,
        radius=radius,
        ants_facing=ants_facing,
    )


def flatten_agent_actions(actions: jax.Array) -> jax.Array:
    """Convert movement/write pairs to the env's interleaved action vector."""

    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError(f"joint actions must have shape (batch, ants, 2), got {actions.shape}.")
    return actions.astype(jnp.int32).reshape(actions.shape[0], actions.shape[1] * 2)
