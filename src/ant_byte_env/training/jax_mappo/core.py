"""Reusable pure JAX MAPPO pieces for AntByte training."""

from __future__ import annotations

import argparse
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env import (
    ACTION_STAY,
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


CRITIC_AUX_FEATURE_DIM = 12
CRITIC_GLOBAL_FEATURE_DIM = 4 + CRITIC_AUX_FEATURE_DIM
SET_CNN_ANT_FEATURE_DIM = 7


class LinearParams(NamedTuple):
    weight: jax.Array
    bias: jax.Array


class ConvParams(NamedTuple):
    kernel: jax.Array
    bias: jax.Array


class ResidualBlockParams(NamedTuple):
    first: ConvParams
    second: ConvParams


class ResNetCriticParams(NamedTuple):
    stem: ConvParams
    blocks_32: tuple[ResidualBlockParams, ResidualBlockParams]
    down_64: ConvParams
    blocks_64: tuple[ResidualBlockParams, ResidualBlockParams]
    down_96: ConvParams
    blocks_96: tuple[ResidualBlockParams, ResidualBlockParams]
    down_128: ConvParams
    blocks_128: tuple[ResidualBlockParams]
    spatial_dense: LinearParams
    entity_body: tuple[LinearParams, LinearParams]
    fusion_body: tuple[LinearParams, LinearParams]


class StridedCNNCriticParams(NamedTuple):
    conv_5x5: ConvParams
    conv_3x3_a: ConvParams
    conv_3x3_b: ConvParams
    spatial_dense: LinearParams
    entity_dense: LinearParams
    fusion_dense: LinearParams


class SetCNNCriticParams(NamedTuple):
    conv_5x5: ConvParams
    conv_3x3_a: ConvParams
    conv_3x3_b: ConvParams
    spatial_dense: LinearParams
    ant_encoder: tuple[LinearParams, LinearParams]
    global_dense: LinearParams
    fusion_body: tuple[LinearParams, LinearParams]


class StructuredMLPCriticParams(NamedTuple):
    grid_body: tuple[LinearParams, LinearParams]
    entity_body: tuple[LinearParams, LinearParams]
    fusion_body: tuple[LinearParams, LinearParams]


class JaxMAPPOParams(NamedTuple):
    actor_body: tuple[LinearParams, LinearParams]
    move_head: LinearParams
    write_head: LinearParams
    critic_body: Any
    value_head: LinearParams


class AdamState(NamedTuple):
    count: jax.Array
    m: JaxMAPPOParams
    v: JaxMAPPOParams


class Transition(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    logprobs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    values: jax.Array
    next_values: jax.Array
    env_rewards: jax.Array
    pickup_events: jax.Array
    delivery_events: jax.Array
    carrying_ants: jax.Array
    remaining_food: jax.Array
    active_size: jax.Array
    stage_advances: jax.Array
    stage_delivered_food: jax.Array
    newly_visited_cells: jax.Array
    visited_cell_count: jax.Array
    visited_cell_fraction: jax.Array
    newly_viewed_cells: jax.Array
    viewed_cell_count: jax.Array
    viewed_cell_fraction: jax.Array
    visible_border_cells: jax.Array
    border_moat_cost: jax.Array
    nonzero_byte_tiles: jax.Array
    nonzero_byte_fraction: jax.Array
    applied_nonzero_write_actions: jax.Array
    empty_nonzero_write_actions: jax.Array
    carrying_nonzero_write_actions: jax.Array
    empty_write_action_slots: jax.Array
    carrying_write_action_slots: jax.Array
    write_attempts: jax.Array
    overwrite_events: jax.Array
    reset_hub_pos: jax.Array
    reset_food_positions: jax.Array


class Rollout(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    logprobs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    values: jax.Array
    next_values: jax.Array
    env_rewards: jax.Array
    pickup_events: jax.Array
    delivery_events: jax.Array
    carrying_ants: jax.Array
    remaining_food: jax.Array
    active_size: jax.Array
    stage_advances: jax.Array
    stage_delivered_food: jax.Array
    newly_visited_cells: jax.Array
    visited_cell_count: jax.Array
    visited_cell_fraction: jax.Array
    newly_viewed_cells: jax.Array
    viewed_cell_count: jax.Array
    viewed_cell_fraction: jax.Array
    visible_border_cells: jax.Array
    border_moat_cost: jax.Array
    nonzero_byte_tiles: jax.Array
    nonzero_byte_fraction: jax.Array
    applied_nonzero_write_actions: jax.Array
    empty_nonzero_write_actions: jax.Array
    carrying_nonzero_write_actions: jax.Array
    empty_write_action_slots: jax.Array
    carrying_write_action_slots: jax.Array
    write_attempts: jax.Array
    overwrite_events: jax.Array
    reset_hub_pos: jax.Array
    reset_food_positions: jax.Array


class TrainingBatch(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    old_logprobs: jax.Array
    advantages: jax.Array
    returns: jax.Array


class UpdateMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clipfrac: jax.Array
    grad_norm: jax.Array
    actor_grad_norm: jax.Array
    critic_grad_norm: jax.Array


class GradientNorms(NamedTuple):
    global_norm: jax.Array
    actor_norm: jax.Array
    critic_norm: jax.Array


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

    return jnp.concatenate(
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


def init_layer(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    *,
    scale: float = np.sqrt(2.0),
) -> LinearParams:
    std = float(scale) / np.sqrt(max(float(in_dim), 1.0))
    return LinearParams(
        weight=jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * std,
        bias=jnp.zeros((out_dim,), dtype=jnp.float32),
    )


def init_conv_layer(
    key: jax.Array,
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    scale: float = np.sqrt(2.0),
) -> ConvParams:
    fan_in = max(float(kernel_size * kernel_size * in_channels), 1.0)
    std = float(scale) / np.sqrt(fan_in)
    return ConvParams(
        kernel=jax.random.normal(
            key,
            (kernel_size, kernel_size, in_channels, out_channels),
            dtype=jnp.float32,
        )
        * std,
        bias=jnp.zeros((out_channels,), dtype=jnp.float32),
    )


def init_residual_block(key: jax.Array, channels: int) -> ResidualBlockParams:
    first_key, second_key = jax.random.split(key)
    return ResidualBlockParams(
        first=init_conv_layer(first_key, channels, channels),
        second=init_conv_layer(second_key, channels, channels),
    )


def _critic_entity_dim(*, num_ants: int) -> int:
    return 7 * int(num_ants) + CRITIC_GLOBAL_FEATURE_DIM


def central_obs_dim_with_ants_count(
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> int:
    grid_area = int(obs_height) * int(obs_width)
    return 7 * int(num_ants) + 3 * grid_area + CRITIC_GLOBAL_FEATURE_DIM


def _strided_cnn_output_size(size: int) -> int:
    resolved = int(size)
    for _ in range(3):
        resolved = (resolved + 1) // 2
    return max(resolved, 1)


def _strided_cnn_flatten_dim(*, obs_height: int, obs_width: int) -> int:
    return (
        _strided_cnn_output_size(obs_height)
        * _strided_cnn_output_size(obs_width)
        * 64
    )


def init_resnet_cnn_critic(
    key: jax.Array,
    *,
    num_ants: int,
    spatial_channels: int = 4,
) -> tuple[ResNetCriticParams, LinearParams]:
    if num_ants <= 0:
        raise ValueError("critic_num_ants must be positive.")
    keys = jax.random.split(key, 17)
    critic_body = ResNetCriticParams(
        stem=init_conv_layer(keys[0], spatial_channels, 32),
        blocks_32=(
            init_residual_block(keys[1], 32),
            init_residual_block(keys[2], 32),
        ),
        down_64=init_conv_layer(keys[3], 32, 64),
        blocks_64=(
            init_residual_block(keys[4], 64),
            init_residual_block(keys[5], 64),
        ),
        down_96=init_conv_layer(keys[6], 64, 96),
        blocks_96=(
            init_residual_block(keys[7], 96),
            init_residual_block(keys[8], 96),
        ),
        down_128=init_conv_layer(keys[9], 96, 128),
        blocks_128=(init_residual_block(keys[10], 128),),
        spatial_dense=init_layer(keys[11], 256, 256),
        entity_body=(
            init_layer(keys[12], _critic_entity_dim(num_ants=num_ants), 128),
            init_layer(keys[13], 128, 128),
        ),
        fusion_body=(
            init_layer(keys[14], 384, 256),
            init_layer(keys[15], 256, 256),
        ),
    )
    return critic_body, init_layer(keys[16], 256, 1, scale=1.0)


def init_strided_cnn_critic(
    key: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
    spatial_channels: int = 4,
) -> tuple[StridedCNNCriticParams, LinearParams]:
    if num_ants <= 0:
        raise ValueError("critic_num_ants must be positive.")
    if obs_height <= 0 or obs_width <= 0:
        raise ValueError("critic_obs_height and critic_obs_width must be positive.")
    keys = jax.random.split(key, 7)
    critic_body = StridedCNNCriticParams(
        conv_5x5=init_conv_layer(keys[0], spatial_channels, 32, kernel_size=5),
        conv_3x3_a=init_conv_layer(keys[1], 32, 64),
        conv_3x3_b=init_conv_layer(keys[2], 64, 64),
        spatial_dense=init_layer(
            keys[3],
            _strided_cnn_flatten_dim(obs_height=obs_height, obs_width=obs_width),
            256,
        ),
        entity_dense=init_layer(keys[4], _critic_entity_dim(num_ants=num_ants), 128),
        fusion_dense=init_layer(keys[5], 384, 256),
    )
    return critic_body, init_layer(keys[6], 256, 1, scale=1.0)


def init_set_cnn_critic(
    key: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
    spatial_channels: int = 4,
) -> tuple[SetCNNCriticParams, LinearParams]:
    if num_ants <= 0:
        raise ValueError("critic_num_ants must be positive.")
    if obs_height <= 0 or obs_width <= 0:
        raise ValueError("critic_obs_height and critic_obs_width must be positive.")
    keys = jax.random.split(key, 10)
    critic_body = SetCNNCriticParams(
        conv_5x5=init_conv_layer(keys[0], spatial_channels, 32, kernel_size=5),
        conv_3x3_a=init_conv_layer(keys[1], 32, 64),
        conv_3x3_b=init_conv_layer(keys[2], 64, 64),
        spatial_dense=init_layer(
            keys[3],
            _strided_cnn_flatten_dim(obs_height=obs_height, obs_width=obs_width),
            256,
        ),
        ant_encoder=(
            init_layer(keys[4], SET_CNN_ANT_FEATURE_DIM, 64),
            init_layer(keys[5], 64, 64),
        ),
        global_dense=init_layer(keys[6], CRITIC_GLOBAL_FEATURE_DIM, 64),
        fusion_body=(
            init_layer(keys[7], 448, 256),
            init_layer(keys[8], 256, 256),
        ),
    )
    return critic_body, init_layer(keys[9], 256, 1, scale=1.0)


def init_structured_mlp_critic(
    key: jax.Array,
    *,
    grid_feature_dim: int,
    entity_feature_dim: int,
) -> tuple[StructuredMLPCriticParams, LinearParams]:
    if grid_feature_dim <= 0:
        raise ValueError("grid_feature_dim must be positive.")
    if entity_feature_dim <= 0:
        raise ValueError("entity_feature_dim must be positive.")
    keys = jax.random.split(key, 7)
    critic_body = StructuredMLPCriticParams(
        grid_body=(
            init_layer(keys[0], grid_feature_dim, 512),
            init_layer(keys[1], 512, 256),
        ),
        entity_body=(
            init_layer(keys[2], entity_feature_dim, 128),
            init_layer(keys[3], 128, 128),
        ),
        fusion_body=(
            init_layer(keys[4], 384, 256),
            init_layer(keys[5], 256, 256),
        ),
    )
    return critic_body, init_layer(keys[6], 256, 1, scale=1.0)


def init_agent_params(
    key: jax.Array,
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
    hidden_size: int = 128,
    write_value_count: int = 2,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
) -> JaxMAPPOParams:
    if write_value_count <= 0:
        raise ValueError("write_value_count must be positive.")
    architecture = str(critic_architecture)
    if architecture == "mlp":
        keys = jax.random.split(key, 7)
        critic_body: Any = (
            init_layer(keys[4], central_obs_dim, hidden_size),
            init_layer(keys[5], hidden_size, hidden_size),
        )
        value_head = init_layer(keys[6], hidden_size, 1, scale=1.0)
    elif architecture == "resnet_cnn":
        if critic_num_ants is None or critic_obs_height is None or critic_obs_width is None:
            raise ValueError(
                "resnet_cnn critic requires critic_num_ants, critic_obs_height, "
                "and critic_obs_width."
            )
        expected_dim = central_obs_dim_with_ants_count(
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
        if int(central_obs_dim) != expected_dim:
            raise ValueError(
                f"resnet_cnn critic expected central_obs_dim {expected_dim}, "
                f"got {central_obs_dim}."
            )
        keys = jax.random.split(key, 5)
        critic_body, value_head = init_resnet_cnn_critic(
            keys[4],
            num_ants=critic_num_ants,
        )
    elif architecture == "strided_cnn":
        if critic_num_ants is None or critic_obs_height is None or critic_obs_width is None:
            raise ValueError(
                "strided_cnn critic requires critic_num_ants, critic_obs_height, "
                "and critic_obs_width."
            )
        expected_dim = central_obs_dim_with_ants_count(
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
        if int(central_obs_dim) != expected_dim:
            raise ValueError(
                f"strided_cnn critic expected central_obs_dim {expected_dim}, "
                f"got {central_obs_dim}."
            )
        keys = jax.random.split(key, 5)
        critic_body, value_head = init_strided_cnn_critic(
            keys[4],
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
    elif architecture == "set_cnn":
        if critic_num_ants is None or critic_obs_height is None or critic_obs_width is None:
            raise ValueError(
                "set_cnn critic requires critic_num_ants, critic_obs_height, "
                "and critic_obs_width."
            )
        expected_dim = central_obs_dim_with_ants_count(
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
        if int(central_obs_dim) != expected_dim:
            raise ValueError(
                f"set_cnn critic expected central_obs_dim {expected_dim}, "
                f"got {central_obs_dim}."
            )
        keys = jax.random.split(key, 5)
        critic_body, value_head = init_set_cnn_critic(
            keys[4],
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
    elif architecture == "structured_mlp":
        if critic_num_ants is None or critic_obs_height is None or critic_obs_width is None:
            raise ValueError(
                "structured_mlp critic requires critic_num_ants, critic_obs_height, "
                "and critic_obs_width."
            )
        expected_dim = central_obs_dim_with_ants_count(
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
        if int(central_obs_dim) != expected_dim:
            raise ValueError(
                f"structured_mlp critic expected central_obs_dim {expected_dim}, "
                f"got {central_obs_dim}."
            )
        keys = jax.random.split(key, 5)
        grid_feature_dim = 3 * int(critic_obs_height) * int(critic_obs_width)
        critic_body, value_head = init_structured_mlp_critic(
            keys[4],
            grid_feature_dim=grid_feature_dim,
            entity_feature_dim=_critic_entity_dim(num_ants=critic_num_ants),
        )
    else:
        raise ValueError(
            "critic_architecture must be 'mlp', 'structured_mlp', 'strided_cnn', "
            "'set_cnn', or 'resnet_cnn'."
        )
    return JaxMAPPOParams(
        actor_body=(
            init_layer(keys[0], actor_obs_dim, hidden_size),
            init_layer(keys[1], hidden_size, hidden_size),
        ),
        move_head=init_layer(keys[2], hidden_size, MOVEMENT_ACTION_COUNT, scale=0.01),
        write_head=init_layer(keys[3], hidden_size, write_value_count, scale=0.01),
        critic_body=critic_body,
        value_head=value_head,
    )


def _linear(params: LinearParams, x: jax.Array) -> jax.Array:
    return x @ params.weight + params.bias


def _forward_body(layers: tuple[LinearParams, LinearParams], x: jax.Array) -> jax.Array:
    hidden = jnp.tanh(_linear(layers[0], x))
    return jnp.tanh(_linear(layers[1], hidden))


def _activation(x: jax.Array) -> jax.Array:
    return jax.nn.silu(x)


def _conv2d(params: ConvParams, x: jax.Array, *, stride: int = 1) -> jax.Array:
    output = jax.lax.conv_general_dilated(
        x,
        params.kernel,
        window_strides=(int(stride), int(stride)),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return output + params.bias


def _residual_block(params: ResidualBlockParams, x: jax.Array) -> jax.Array:
    residual = x
    hidden = _activation(_conv2d(params.first, x))
    hidden = _conv2d(params.second, hidden)
    return _activation(hidden + residual)


def _require_cnn_critic_field(
    name: str,
    value: int | None,
    *,
    critic_architecture: str = "resnet_cnn",
) -> int:
    if value is None:
        raise ValueError(f"{critic_architecture} critic requires {name}.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _require_structured_mlp_critic_field(name: str, value: int | None) -> int:
    if value is None:
        raise ValueError(f"structured_mlp critic requires {name}.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _split_central_observation_for_cnn(
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
    critic_architecture: str = "resnet_cnn",
) -> tuple[jax.Array, jax.Array, tuple[int, ...]]:
    leading_shape = central_obs.shape[:-1]
    flat = central_obs.reshape((-1, central_obs.shape[-1]))
    grid_area = int(obs_height) * int(obs_width)
    expected_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if int(central_obs.shape[-1]) != expected_dim:
        raise ValueError(
            f"{critic_architecture} critic expected central_obs_dim {expected_dim}, "
            f"got {central_obs.shape[-1]}."
        )

    ants_pos_width = 2 * int(num_ants)
    ants_carrying_width = int(num_ants)
    ants_facing_width = (MOVEMENT_ACTION_COUNT - 1) * int(num_ants)
    ants_pos_end = ants_pos_width
    ants_carrying_end = ants_pos_end + ants_carrying_width
    ants_facing_end = ants_carrying_end + ants_facing_width
    ants_count_end = ants_facing_end + grid_area
    food_end = ants_count_end + grid_area
    bytes_end = food_end + grid_area
    hub_end = bytes_end + 2

    ants_count = flat[:, ants_facing_end:ants_count_end].reshape(
        (-1, int(obs_height), int(obs_width))
    )
    food = flat[:, ants_count_end:food_end].reshape((-1, int(obs_height), int(obs_width)))
    bytes_grid = flat[:, food_end:bytes_end].reshape(
        (-1, int(obs_height), int(obs_width))
    )
    hub_pos = flat[:, bytes_end:hub_end]
    batch_index = jnp.arange(flat.shape[0])
    hub_x = jnp.clip(
        jnp.rint(hub_pos[:, 0] * max(int(obs_width) - 1, 1)).astype(jnp.int32),
        0,
        int(obs_width) - 1,
    )
    hub_y = jnp.clip(
        jnp.rint(hub_pos[:, 1] * max(int(obs_height) - 1, 1)).astype(jnp.int32),
        0,
        int(obs_height) - 1,
    )
    hub_grid = jnp.zeros(
        (flat.shape[0], int(obs_height), int(obs_width)),
        dtype=jnp.float32,
    ).at[batch_index, hub_y, hub_x].set(1.0)
    spatial = jnp.stack([ants_count, food, bytes_grid, hub_grid], axis=-1)
    entity = jnp.concatenate(
        [
            flat[:, :ants_pos_end],
            flat[:, ants_pos_end:ants_carrying_end],
            flat[:, ants_carrying_end:ants_facing_end],
            flat[:, bytes_end:],
        ],
        axis=-1,
    )
    return spatial, entity, leading_shape


def _split_central_observation_for_set_cnn(
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[int, ...]]:
    leading_shape = central_obs.shape[:-1]
    flat = central_obs.reshape((-1, central_obs.shape[-1]))
    grid_area = int(obs_height) * int(obs_width)
    expected_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if int(central_obs.shape[-1]) != expected_dim:
        raise ValueError(
            f"set_cnn critic expected central_obs_dim {expected_dim}, "
            f"got {central_obs.shape[-1]}."
        )

    ants_pos_width = 2 * int(num_ants)
    ants_carrying_width = int(num_ants)
    ants_facing_width = (MOVEMENT_ACTION_COUNT - 1) * int(num_ants)
    ants_pos_end = ants_pos_width
    ants_carrying_end = ants_pos_end + ants_carrying_width
    ants_facing_end = ants_carrying_end + ants_facing_width
    ants_count_end = ants_facing_end + grid_area
    food_end = ants_count_end + grid_area
    bytes_end = food_end + grid_area
    hub_end = bytes_end + 2

    ants_count = flat[:, ants_facing_end:ants_count_end].reshape(
        (-1, int(obs_height), int(obs_width))
    )
    food = flat[:, ants_count_end:food_end].reshape((-1, int(obs_height), int(obs_width)))
    bytes_grid = flat[:, food_end:bytes_end].reshape(
        (-1, int(obs_height), int(obs_width))
    )
    hub_pos = flat[:, bytes_end:hub_end]
    batch_index = jnp.arange(flat.shape[0])
    hub_x = jnp.clip(
        jnp.rint(hub_pos[:, 0] * max(int(obs_width) - 1, 1)).astype(jnp.int32),
        0,
        int(obs_width) - 1,
    )
    hub_y = jnp.clip(
        jnp.rint(hub_pos[:, 1] * max(int(obs_height) - 1, 1)).astype(jnp.int32),
        0,
        int(obs_height) - 1,
    )
    hub_grid = jnp.zeros(
        (flat.shape[0], int(obs_height), int(obs_width)),
        dtype=jnp.float32,
    ).at[batch_index, hub_y, hub_x].set(1.0)
    spatial = jnp.stack([ants_count, food, bytes_grid, hub_grid], axis=-1)

    ant_positions = flat[:, :ants_pos_end].reshape((-1, int(num_ants), 2))
    ant_carrying = flat[:, ants_pos_end:ants_carrying_end].reshape(
        (-1, int(num_ants), 1)
    )
    ant_facing = flat[:, ants_carrying_end:ants_facing_end].reshape(
        (-1, int(num_ants), MOVEMENT_ACTION_COUNT - 1)
    )
    ant_features = jnp.concatenate(
        [ant_positions, ant_carrying, ant_facing],
        axis=-1,
    )
    global_features = flat[:, bytes_end:]
    return spatial, ant_features, global_features, leading_shape


def _split_central_observation_for_structured_mlp(
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> tuple[jax.Array, jax.Array, tuple[int, ...]]:
    leading_shape = central_obs.shape[:-1]
    flat = central_obs.reshape((-1, central_obs.shape[-1]))
    grid_area = int(obs_height) * int(obs_width)
    expected_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if int(central_obs.shape[-1]) != expected_dim:
        raise ValueError(
            f"structured_mlp critic expected central_obs_dim {expected_dim}, "
            f"got {central_obs.shape[-1]}."
        )

    ants_pos_width = 2 * int(num_ants)
    ants_carrying_width = int(num_ants)
    ants_facing_width = (MOVEMENT_ACTION_COUNT - 1) * int(num_ants)
    ants_pos_end = ants_pos_width
    ants_carrying_end = ants_pos_end + ants_carrying_width
    ants_facing_end = ants_carrying_end + ants_facing_width
    ants_count_end = ants_facing_end + grid_area
    food_end = ants_count_end + grid_area
    bytes_end = food_end + grid_area

    grid_features = flat[:, ants_facing_end:bytes_end]
    entity_features = jnp.concatenate(
        [
            flat[:, :ants_facing_end],
            flat[:, bytes_end:],
        ],
        axis=-1,
    )
    return grid_features, entity_features, leading_shape


def _forward_resnet_cnn_critic(
    critic_body: ResNetCriticParams,
    value_head: LinearParams,
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> jax.Array:
    spatial, entity, leading_shape = _split_central_observation_for_cnn(
        central_obs,
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    hidden = _activation(_conv2d(critic_body.stem, spatial))
    for block in critic_body.blocks_32:
        hidden = _residual_block(block, hidden)
    hidden = _activation(_conv2d(critic_body.down_64, hidden, stride=2))
    for block in critic_body.blocks_64:
        hidden = _residual_block(block, hidden)
    hidden = _activation(_conv2d(critic_body.down_96, hidden, stride=2))
    for block in critic_body.blocks_96:
        hidden = _residual_block(block, hidden)
    hidden = _activation(_conv2d(critic_body.down_128, hidden, stride=2))
    for block in critic_body.blocks_128:
        hidden = _residual_block(block, hidden)

    pooled = jnp.concatenate(
        [
            jnp.mean(hidden, axis=(1, 2)),
            jnp.max(hidden, axis=(1, 2)),
        ],
        axis=-1,
    )
    spatial_embedding = _activation(_linear(critic_body.spatial_dense, pooled))
    entity_embedding = _activation(_linear(critic_body.entity_body[0], entity))
    entity_embedding = _activation(_linear(critic_body.entity_body[1], entity_embedding))
    fused = jnp.concatenate([spatial_embedding, entity_embedding], axis=-1)
    fused = _activation(_linear(critic_body.fusion_body[0], fused))
    fused = _activation(_linear(critic_body.fusion_body[1], fused))
    value = jnp.squeeze(_linear(value_head, fused), axis=-1)
    return value.reshape(leading_shape)


def _forward_strided_cnn_critic(
    critic_body: StridedCNNCriticParams,
    value_head: LinearParams,
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> jax.Array:
    spatial, entity, leading_shape = _split_central_observation_for_cnn(
        central_obs,
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
        critic_architecture="strided_cnn",
    )
    hidden = _activation(_conv2d(critic_body.conv_5x5, spatial, stride=2))
    hidden = _activation(_conv2d(critic_body.conv_3x3_a, hidden, stride=2))
    hidden = _activation(_conv2d(critic_body.conv_3x3_b, hidden, stride=2))

    spatial_features = hidden.reshape((hidden.shape[0], -1))
    spatial_embedding = _activation(
        _linear(critic_body.spatial_dense, spatial_features)
    )
    entity_embedding = _activation(_linear(critic_body.entity_dense, entity))
    fused = jnp.concatenate([spatial_embedding, entity_embedding], axis=-1)
    fused = _activation(_linear(critic_body.fusion_dense, fused))
    value = jnp.squeeze(_linear(value_head, fused), axis=-1)
    return value.reshape(leading_shape)


def _forward_set_cnn_critic(
    critic_body: SetCNNCriticParams,
    value_head: LinearParams,
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> jax.Array:
    spatial, ant_features, global_features, leading_shape = (
        _split_central_observation_for_set_cnn(
            central_obs,
            num_ants=num_ants,
            obs_height=obs_height,
            obs_width=obs_width,
        )
    )
    hidden = _activation(_conv2d(critic_body.conv_5x5, spatial, stride=2))
    hidden = _activation(_conv2d(critic_body.conv_3x3_a, hidden, stride=2))
    hidden = _activation(_conv2d(critic_body.conv_3x3_b, hidden, stride=2))
    spatial_features = hidden.reshape((hidden.shape[0], -1))
    spatial_embedding = _activation(
        _linear(critic_body.spatial_dense, spatial_features)
    )

    ant_hidden = _activation(_linear(critic_body.ant_encoder[0], ant_features))
    ant_hidden = _activation(_linear(critic_body.ant_encoder[1], ant_hidden))
    ant_embedding = jnp.concatenate(
        [
            jnp.mean(ant_hidden, axis=1),
            jnp.max(ant_hidden, axis=1),
        ],
        axis=-1,
    )
    global_embedding = _activation(_linear(critic_body.global_dense, global_features))
    fused = jnp.concatenate(
        [spatial_embedding, ant_embedding, global_embedding],
        axis=-1,
    )
    fused = _activation(_linear(critic_body.fusion_body[0], fused))
    fused = _activation(_linear(critic_body.fusion_body[1], fused))
    value = jnp.squeeze(_linear(value_head, fused), axis=-1)
    return value.reshape(leading_shape)


def _forward_structured_mlp_critic(
    critic_body: StructuredMLPCriticParams,
    value_head: LinearParams,
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> jax.Array:
    grid_features, entity_features, leading_shape = (
        _split_central_observation_for_structured_mlp(
            central_obs,
            num_ants=num_ants,
            obs_height=obs_height,
            obs_width=obs_width,
        )
    )
    grid_embedding = _forward_body(critic_body.grid_body, grid_features)
    entity_embedding = _forward_body(critic_body.entity_body, entity_features)
    fused = jnp.concatenate([grid_embedding, entity_embedding], axis=-1)
    fused = _forward_body(critic_body.fusion_body, fused)
    value = jnp.squeeze(_linear(value_head, fused), axis=-1)
    return value.reshape(leading_shape)


def critic_forward_kwargs_from_args(args: argparse.Namespace) -> dict[str, int | str]:
    architecture = str(getattr(args, "critic_architecture", "mlp"))
    if architecture == "mlp":
        return {}
    if architecture not in {"structured_mlp", "strided_cnn", "set_cnn", "resnet_cnn"}:
        raise ValueError(
            "critic_architecture must be 'mlp', 'structured_mlp', 'strided_cnn', "
            "'set_cnn', or 'resnet_cnn'."
        )
    return {
        "critic_architecture": architecture,
        "critic_num_ants": int(args.num_ants),
        "critic_obs_height": int(args.obs_height or args.height),
        "critic_obs_width": int(args.obs_width or args.width),
    }


def get_action_logits(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    hidden = _forward_body(params.actor_body, actor_obs)
    return _linear(params.move_head, hidden), _linear(params.write_head, hidden)


def get_value(
    params: JaxMAPPOParams,
    central_obs: jax.Array,
    *,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
) -> jax.Array:
    architecture = str(critic_architecture)
    if architecture == "mlp":
        hidden = _forward_body(params.critic_body, central_obs)
        return jnp.squeeze(_linear(params.value_head, hidden), axis=-1)
    if architecture == "resnet_cnn":
        return _forward_resnet_cnn_critic(
            params.critic_body,
            params.value_head,
            central_obs,
            num_ants=_require_cnn_critic_field("critic_num_ants", critic_num_ants),
            obs_height=_require_cnn_critic_field("critic_obs_height", critic_obs_height),
            obs_width=_require_cnn_critic_field("critic_obs_width", critic_obs_width),
        )
    if architecture == "strided_cnn":
        return _forward_strided_cnn_critic(
            params.critic_body,
            params.value_head,
            central_obs,
            num_ants=_require_cnn_critic_field(
                "critic_num_ants",
                critic_num_ants,
                critic_architecture="strided_cnn",
            ),
            obs_height=_require_cnn_critic_field(
                "critic_obs_height",
                critic_obs_height,
                critic_architecture="strided_cnn",
            ),
            obs_width=_require_cnn_critic_field(
                "critic_obs_width",
                critic_obs_width,
                critic_architecture="strided_cnn",
            ),
        )
    if architecture == "set_cnn":
        return _forward_set_cnn_critic(
            params.critic_body,
            params.value_head,
            central_obs,
            num_ants=_require_cnn_critic_field(
                "critic_num_ants",
                critic_num_ants,
                critic_architecture="set_cnn",
            ),
            obs_height=_require_cnn_critic_field(
                "critic_obs_height",
                critic_obs_height,
                critic_architecture="set_cnn",
            ),
            obs_width=_require_cnn_critic_field(
                "critic_obs_width",
                critic_obs_width,
                critic_architecture="set_cnn",
            ),
        )
    if architecture == "structured_mlp":
        return _forward_structured_mlp_critic(
            params.critic_body,
            params.value_head,
            central_obs,
            num_ants=_require_structured_mlp_critic_field(
                "critic_num_ants",
                critic_num_ants,
            ),
            obs_height=_require_structured_mlp_critic_field(
                "critic_obs_height",
                critic_obs_height,
            ),
            obs_width=_require_structured_mlp_critic_field(
                "critic_obs_width",
                critic_obs_width,
            ),
        )
    raise ValueError(
        "critic_architecture must be 'mlp', 'structured_mlp', 'strided_cnn', "
        "'set_cnn', or 'resnet_cnn'."
    )


def _categorical_log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def _logits_for_policy_temperature(
    logits: jax.Array,
    *,
    policy_temperature: float,
) -> jax.Array:
    temperature = float(policy_temperature)
    if temperature <= 0.0:
        raise ValueError("policy_temperature must be positive.")
    return logits / temperature


def evaluate_actions(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    central_obs: jax.Array,
    actions: jax.Array,
    *,
    policy_temperature: float = 1.0,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    move_logits, write_logits = get_action_logits(params, actor_obs)
    move_logits = _logits_for_policy_temperature(
        move_logits,
        policy_temperature=policy_temperature,
    )
    write_logits = _logits_for_policy_temperature(
        write_logits,
        policy_temperature=policy_temperature,
    )
    logprob = _categorical_log_prob(move_logits, actions[..., 0])
    logprob += _categorical_log_prob(write_logits, actions[..., 1])
    entropy = _categorical_entropy(move_logits) + _categorical_entropy(write_logits)
    value = get_value(
        params,
        central_obs,
        critic_architecture=critic_architecture,
        critic_num_ants=critic_num_ants,
        critic_obs_height=critic_obs_height,
        critic_obs_width=critic_obs_width,
    )
    return logprob, entropy, value


def get_action_and_value(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    central_obs: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool = False,
    policy_temperature: float = 1.0,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    move_logits, write_logits = get_action_logits(params, actor_obs)
    move_logits = _logits_for_policy_temperature(
        move_logits,
        policy_temperature=policy_temperature,
    )
    write_logits = _logits_for_policy_temperature(
        write_logits,
        policy_temperature=policy_temperature,
    )
    if deterministic:
        move_actions = jnp.argmax(move_logits, axis=-1)
        write_actions = jnp.argmax(write_logits, axis=-1)
    else:
        move_key, write_key = jax.random.split(key)
        move_actions = jax.random.categorical(move_key, move_logits, axis=-1)
        write_actions = jax.random.categorical(write_key, write_logits, axis=-1)
    actions = jnp.stack([move_actions, write_actions], axis=-1).astype(jnp.int32)
    logprob = _categorical_log_prob(move_logits, move_actions)
    logprob += _categorical_log_prob(write_logits, write_actions)
    entropy = _categorical_entropy(move_logits) + _categorical_entropy(write_logits)
    value = get_value(
        params,
        central_obs,
        critic_architecture=critic_architecture,
        critic_num_ants=critic_num_ants,
        critic_obs_height=critic_obs_height,
        critic_obs_width=critic_obs_width,
    )
    return actions, logprob, entropy, value


def _grid_values_at_positions(grid: jax.Array, positions: jax.Array) -> jax.Array:
    batch_index = jnp.arange(grid.shape[0])[:, None]
    x_pos = positions[..., 0].astype(jnp.int32)
    y_pos = positions[..., 1].astype(jnp.int32)
    return grid[batch_index, y_pos, x_pos]


def compute_forage_curriculum_rewards(
    *,
    previous_obs: JaxObs,
    next_obs: JaxObs,
    env_rewards: jax.Array,
    actions: jax.Array | None = None,
    pickup_bonus: float,
    distance_bonus: float = 0.0,
    distance_progress_normalizer: str = "map",
    carrying_hub_distance_bonus: float = 0.0,
    newly_visited_cells: jax.Array | None = None,
    visited_cell_fraction: jax.Array | None = None,
    visit_reward_scale: float = 0.0,
    visit_reward_decay: float = 1.0,
    newly_viewed_cells: jax.Array | None = None,
    viewed_cell_fraction: jax.Array | None = None,
    view_reward_scale: float = 0.0,
    view_reward_decay: float = 1.0,
    visible_border_cells: jax.Array | None = None,
    border_view_penalty: float = 0.0,
    border_moat_cost: jax.Array | None = None,
    border_moat_penalty: float = 0.0,
    stage_completion_events: jax.Array | None = None,
    stage_completion_bonus: float = 0.0,
    delivery_byte_trail_bonus: float = 0.0,
    delivery_byte_trail_target_tiles: float = 8.0,
    byte_follow_bonus: float = 0.0,
    carrying_byte_write_bonus: float = 0.0,
    write_while_moving: bool = False,
    per_ant_write_channels: bool = False,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> jax.Array:
    """Add trainer-side forage shaping without changing actor observations."""

    uses_delivery_byte_trail_bonus = delivery_byte_trail_bonus > 0.0
    uses_byte_follow_bonus = byte_follow_bonus > 0.0
    uses_carrying_byte_write_bonus = carrying_byte_write_bonus > 0.0
    uses_byte_shaping = (
        uses_delivery_byte_trail_bonus
        or uses_byte_follow_bonus
        or uses_carrying_byte_write_bonus
    )
    if uses_byte_shaping:
        previous_bytes = previous_obs["bytes"]
    else:
        previous_bytes = jnp.zeros_like(previous_obs["food"], dtype=jnp.uint8)
    _, previous_height, previous_width = previous_bytes.shape
    previous_active_size = (
        _active_grid_size(
            previous_obs,
            fallback_height=previous_height,
            fallback_width=previous_width,
        )
        if uses_delivery_byte_trail_bonus
        else jnp.zeros((previous_obs["food"].shape[0], 2), dtype=jnp.float32)
    )
    previous_carrying = previous_obs["ants_carrying"].astype(jnp.bool_)
    next_carrying = next_obs["ants_carrying"].astype(jnp.bool_)
    previous_positions = previous_obs["ants_pos"].astype(jnp.int32)
    next_positions = next_obs["ants_pos"].astype(jnp.int32)

    target_bytes = jnp.zeros(previous_carrying.shape, dtype=previous_bytes.dtype)
    if uses_byte_follow_bonus or uses_carrying_byte_write_bonus:
        target_bytes = _grid_values_at_positions(previous_bytes, next_positions)

    fresh_carrying_writes = jnp.zeros(previous_carrying.shape, dtype=jnp.bool_)
    if uses_carrying_byte_write_bonus:
        if actions is None:
            applied_write_values = jnp.zeros(previous_carrying.shape, dtype=jnp.uint32)
        else:
            applied_write_values = _applied_write_values(
                actions,
                write_while_moving=write_while_moving,
                per_ant_write_channels=per_ant_write_channels,
                write_bits=write_bits,
            )
        target_food = _grid_values_at_positions(previous_obs["food"], next_positions)
        target_is_hub = jnp.all(
            next_positions == previous_obs["hub_pos"][:, None, :],
            axis=-1,
        )
        fresh_carrying_writes = (
            previous_carrying
            & (applied_write_values > 0)
            & (target_bytes == 0)
            & (target_food <= 0)
            & jnp.logical_not(target_is_hub)
        )

    byte_follow_events = jnp.zeros(previous_carrying.shape, dtype=jnp.bool_)
    if uses_byte_follow_bonus:
        moved = jnp.any(next_positions != previous_positions, axis=-1)
        previous_food_distance = _nearest_food_distances(previous_obs["food"], previous_positions)
        next_food_distance = _nearest_food_distances(previous_obs["food"], next_positions)
        byte_follow_events = (
            jnp.logical_not(previous_carrying)
            & jnp.logical_not(next_carrying)
            & moved
            & (target_bytes > 0)
            & (next_food_distance < previous_food_distance)
        )

    def one_env(
        previous_carrying: jax.Array,
        next_carrying: jax.Array,
        env_reward: jax.Array,
        previous_bytes: jax.Array,
        active_size: jax.Array,
        fresh_carrying_writes: jax.Array,
        byte_follow_events: jax.Array,
    ) -> jax.Array:
        was_carrying = previous_carrying.astype(jnp.bool_)
        is_carrying = next_carrying.astype(jnp.bool_)
        pickups = jnp.logical_and(jnp.logical_not(was_carrying), is_carrying).astype(jnp.float32)
        deliveries = jnp.logical_and(was_carrying, jnp.logical_not(is_carrying)).astype(
            jnp.float32
        )
        shaped = env_reward.astype(jnp.float32)
        shaped += float(pickup_bonus) * jnp.sum(pickups)
        if delivery_byte_trail_bonus > 0.0:
            active_width, active_height = active_size.astype(jnp.float32)
            x_coords = jnp.arange(previous_width, dtype=jnp.float32)[None, :]
            y_coords = jnp.arange(previous_height, dtype=jnp.float32)[:, None]
            active_mask = (x_coords < active_width) & (y_coords < active_height)
            nonzero_trail_tiles = jnp.sum(
                jnp.logical_and(previous_bytes > 0, active_mask).astype(jnp.float32)
            )
            trail_fraction = jnp.minimum(
                nonzero_trail_tiles / max(float(delivery_byte_trail_target_tiles), 1.0),
                1.0,
            )
            shaped += (
                float(delivery_byte_trail_bonus)
                * jnp.sum(deliveries)
                * trail_fraction
            )
        if carrying_byte_write_bonus > 0.0:
            shaped += float(carrying_byte_write_bonus) * jnp.sum(
                fresh_carrying_writes.astype(jnp.float32)
            )
        if byte_follow_bonus > 0.0:
            shaped += float(byte_follow_bonus) * jnp.sum(
                byte_follow_events.astype(jnp.float32)
            )
        return shaped

    shaped_rewards = jax.vmap(one_env)(
        previous_carrying,
        next_carrying,
        env_rewards,
        previous_bytes,
        previous_active_size,
        fresh_carrying_writes,
        byte_follow_events,
    )
    if stage_completion_bonus > 0.0 and stage_completion_events is not None:
        shaped_rewards += float(stage_completion_bonus) * stage_completion_events.astype(
            jnp.float32
        )
    if visit_reward_scale > 0.0:
        visits = (
            jnp.zeros_like(shaped_rewards)
            if newly_visited_cells is None
            else newly_visited_cells.astype(jnp.float32)
        )
        coverage = (
            jnp.zeros_like(shaped_rewards)
            if visited_cell_fraction is None
            else visited_cell_fraction.astype(jnp.float32)
        )
        remaining_fraction = jnp.maximum(1.0 - coverage, 0.0)
        shaped_rewards += (
            float(visit_reward_scale)
            * visits
            * jnp.power(remaining_fraction, float(visit_reward_decay))
        )
    if view_reward_scale > 0.0:
        views = (
            jnp.zeros_like(shaped_rewards)
            if newly_viewed_cells is None
            else newly_viewed_cells.astype(jnp.float32)
        )
        viewed_coverage = (
            jnp.zeros_like(shaped_rewards)
            if viewed_cell_fraction is None
            else viewed_cell_fraction.astype(jnp.float32)
        )
        remaining_view_fraction = jnp.maximum(1.0 - viewed_coverage, 0.0)
        shaped_rewards += (
            float(view_reward_scale)
            * views
            * jnp.power(remaining_view_fraction, float(view_reward_decay))
        )
    if border_view_penalty > 0.0:
        border_cells = (
            jnp.zeros_like(shaped_rewards)
            if visible_border_cells is None
            else visible_border_cells.astype(jnp.float32)
        )
        shaped_rewards -= float(border_view_penalty) * border_cells
    if border_moat_penalty > 0.0:
        moat_cost = (
            jnp.zeros_like(shaped_rewards)
            if border_moat_cost is None
            else border_moat_cost.astype(jnp.float32)
        )
        shaped_rewards -= float(border_moat_penalty) * moat_cost
    if distance_bonus > 0.0:
        progress = _forage_distance_progress(
            previous_obs=previous_obs,
            next_obs=next_obs,
            normalizer_mode=distance_progress_normalizer,
        )
        shaped_rewards += float(distance_bonus) * progress
    if carrying_hub_distance_bonus > 0.0:
        carrying_progress = _carrying_hub_distance_progress(
            previous_obs=previous_obs,
            next_obs=next_obs,
            normalizer_mode=distance_progress_normalizer,
        )
        shaped_rewards += float(carrying_hub_distance_bonus) * carrying_progress
    return shaped_rewards


def _forage_distance_progress(
    *,
    previous_obs: JaxObs,
    next_obs: JaxObs,
    normalizer_mode: str = "map",
) -> jax.Array:
    previous_carrying = previous_obs["ants_carrying"].astype(jnp.bool_)
    next_carrying = next_obs["ants_carrying"].astype(jnp.bool_)
    same_target_mode = previous_carrying == next_carrying
    food = previous_obs["food"]
    previous_positions = previous_obs["ants_pos"]
    next_positions = next_obs["ants_pos"]
    previous_food_distance = _nearest_food_distances(food, previous_positions)
    next_food_distance = _nearest_food_distances(food, next_positions)
    previous_hub_distance = _hub_distances(previous_obs["hub_pos"], previous_positions)
    next_hub_distance = _hub_distances(previous_obs["hub_pos"], next_positions)
    previous_distance = jnp.where(
        previous_carrying,
        previous_hub_distance,
        previous_food_distance,
    )
    next_distance = jnp.where(
        previous_carrying,
        next_hub_distance,
        next_food_distance,
    )
    _, height, width = food.shape
    normalizer = _distance_progress_normalizer(
        previous_obs,
        height=height,
        width=width,
        mode=normalizer_mode,
    )
    progress = (previous_distance - next_distance) / normalizer
    progress = jnp.where(same_target_mode, progress, 0.0)
    return jnp.sum(progress, axis=-1).astype(jnp.float32)


def _carrying_hub_distance_progress(
    *,
    previous_obs: JaxObs,
    next_obs: JaxObs,
    normalizer_mode: str = "map",
) -> jax.Array:
    previous_carrying = previous_obs["ants_carrying"].astype(jnp.bool_)
    next_carrying = next_obs["ants_carrying"].astype(jnp.bool_)
    stayed_carrying = previous_carrying & next_carrying
    previous_distance = _hub_distances(previous_obs["hub_pos"], previous_obs["ants_pos"])
    next_distance = _hub_distances(previous_obs["hub_pos"], next_obs["ants_pos"])
    _, height, width = previous_obs["food"].shape
    normalizer = _distance_progress_normalizer(
        previous_obs,
        height=height,
        width=width,
        mode=normalizer_mode,
    )
    progress = (previous_distance - next_distance) / normalizer
    progress = jnp.where(stayed_carrying, progress, 0.0)
    return jnp.sum(progress, axis=-1).astype(jnp.float32)


def _distance_progress_normalizer(
    obs: JaxObs,
    *,
    height: int,
    width: int,
    mode: str,
) -> jax.Array:
    if mode == "map":
        return jnp.asarray(max(height + width - 2, 1), dtype=jnp.float32)
    if mode != "stage":
        raise ValueError("distance_progress_normalizer must be 'map' or 'stage'.")
    stage_distance = obs.get("distance_curriculum_stage_distance")
    if stage_distance is None:
        return jnp.asarray(max(height + width - 2, 1), dtype=jnp.float32)
    normalizer = jnp.asarray(stage_distance, dtype=jnp.float32)
    if normalizer.ndim == 0:
        normalizer = normalizer[None]
    while normalizer.ndim < 2:
        normalizer = normalizer[..., None]
    return jnp.maximum(normalizer, 1.0)


def _nearest_food_distances(food: jax.Array, ants_pos: jax.Array) -> jax.Array:
    food_mask = food > 0
    _, height, width = food_mask.shape
    x_coords = jnp.arange(width, dtype=jnp.float32)[None, None, None, :]
    y_coords = jnp.arange(height, dtype=jnp.float32)[None, None, :, None]
    ant_x = ants_pos[..., 0].astype(jnp.float32)[:, :, None, None]
    ant_y = ants_pos[..., 1].astype(jnp.float32)[:, :, None, None]
    distances = jnp.abs(ant_x - x_coords) + jnp.abs(ant_y - y_coords)
    large_distance = jnp.asarray(height + width + 1, dtype=jnp.float32)
    masked_distances = jnp.where(food_mask[:, None, :, :], distances, large_distance)
    nearest = jnp.min(masked_distances, axis=(-2, -1))
    has_food = jnp.any(food_mask, axis=(-2, -1))[:, None]
    return jnp.where(has_food, nearest, 0.0).astype(jnp.float32)


def _hub_distances(hub_pos: jax.Array, ants_pos: jax.Array) -> jax.Array:
    hub = hub_pos.astype(jnp.float32)[:, None, :]
    positions = ants_pos.astype(jnp.float32)
    return jnp.sum(jnp.abs(positions - hub), axis=-1).astype(jnp.float32)


def compute_write_bit_penalties(
    actions: jax.Array,
    *,
    write_bits: int,
    base_penalty: float,
    decay: float,
    write_while_moving: bool = False,
    per_ant_write_channels: bool = False,
) -> jax.Array:
    """Return per-env penalties for set write bits, with bit 0 most expensive."""

    if base_penalty <= 0.0:
        return jnp.zeros(actions.shape[0], dtype=jnp.float32)
    write_values = _applied_write_values(
        actions,
        write_while_moving=write_while_moving,
        per_ant_write_channels=per_ant_write_channels,
        write_bits=write_bits,
    )
    bit_indices = jnp.arange(int(write_bits), dtype=jnp.uint32)
    bit_mask = (write_values[..., None] >> bit_indices) & jnp.asarray(1, dtype=jnp.uint32)
    weights = float(base_penalty) * (float(decay) ** bit_indices.astype(jnp.float32))
    return jnp.sum(bit_mask.astype(jnp.float32) * weights, axis=(-1, -2))


def compute_terminal_write_entropy_bonus(
    next_obs: JaxObs,
    dones: jax.Array,
    *,
    write_bits: int,
    entropy_scale: float,
    max_bonus: float,
) -> jax.Array:
    """Return a capped terminal bonus for entropy over nonzero byte values."""

    nonzero_value_count = max_write_value(write_bits)
    if entropy_scale <= 0.0 or max_bonus <= 0.0 or nonzero_value_count <= 1:
        return jnp.zeros(next_obs["bytes"].shape[0], dtype=jnp.float32)
    next_bytes = next_obs["bytes"].astype(jnp.uint32)
    values = jnp.arange(1, nonzero_value_count + 1, dtype=jnp.uint32)
    counts = jnp.sum((next_bytes[..., None] == values).astype(jnp.float32), axis=(-2, -3))
    total = jnp.sum(counts, axis=-1, keepdims=True)
    probabilities = counts / jnp.maximum(total, 1.0)
    safe_probabilities = jnp.where(probabilities > 0.0, probabilities, 1.0)
    entropy = -jnp.sum(
        jnp.where(
            probabilities > 0.0,
            probabilities * jnp.log(safe_probabilities),
            0.0,
        ),
        axis=-1,
    )
    normalized_entropy = entropy / jnp.log(float(nonzero_value_count))
    raw_bonus = normalized_entropy * float(entropy_scale)
    capped_bonus = jnp.minimum(raw_bonus, float(max_bonus))
    return jnp.where(dones, capped_bonus, 0.0)


def compute_write_bit_entropy_bonus(
    actions: jax.Array,
    *,
    write_bits: int,
    entropy_scale: float,
    write_while_moving: bool = False,
    per_ant_write_channels: bool = False,
) -> jax.Array:
    """Return per-step rewards for balanced nonzero write-bit use in a rollout chunk."""

    if entropy_scale <= 0.0 or write_bits <= 0:
        return jnp.zeros(actions.shape[:2], dtype=jnp.float32)

    write_values = _applied_write_values(
        actions,
        write_while_moving=write_while_moving,
        per_ant_write_channels=per_ant_write_channels,
        write_bits=write_bits,
    )
    nonzero_writes = write_values > 0
    nonzero_count = jnp.sum(nonzero_writes.astype(jnp.float32), axis=(0, 2))
    bit_indices = jnp.arange(int(write_bits), dtype=jnp.uint32)
    bit_mask = ((write_values[..., None] >> bit_indices) & jnp.asarray(1, dtype=jnp.uint32))
    bit_counts = jnp.sum(bit_mask.astype(jnp.float32), axis=(0, 2))
    activation_rates = bit_counts / jnp.maximum(nonzero_count[:, None], 1.0)
    active_rates = (activation_rates > 0.0) & (activation_rates < 1.0)
    safe_rates = jnp.where(active_rates, activation_rates, 0.5)
    bit_entropy = -(
        safe_rates * jnp.log(safe_rates)
        + (1.0 - safe_rates) * jnp.log(1.0 - safe_rates)
    ) / jnp.log(2.0)
    bit_entropy = jnp.where(active_rates & (nonzero_count[:, None] > 0.0), bit_entropy, 0.0)
    per_env_bonus = float(entropy_scale) * jnp.mean(bit_entropy, axis=-1)
    return jnp.broadcast_to(
        per_env_bonus[None, :] / max(int(actions.shape[0]), 1),
        actions.shape[:2],
    ).astype(jnp.float32)


def _applied_write_values(
    actions: jax.Array,
    *,
    write_while_moving: bool = False,
    per_ant_write_channels: bool = False,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> jax.Array:
    """Return write values allowed by the current movement/write timing mode."""

    write_values = actions[..., 1].astype(jnp.uint32)
    if per_ant_write_channels:
        if int(write_bits) <= 0:
            raise ValueError("write_bits must be positive.")
        bit_indices = jnp.mod(
            jnp.arange(actions.shape[-2], dtype=jnp.uint32),
            jnp.asarray(int(write_bits), dtype=jnp.uint32),
        )
        ant_bits = jnp.left_shift(
            jnp.asarray(1, dtype=jnp.uint32),
            bit_indices,
        )
        write_values = write_values & ant_bits
    if write_while_moving:
        return write_values
    move_actions = actions[..., 0]
    return jnp.where(move_actions == ACTION_STAY, write_values, 0).astype(jnp.uint32)


def compute_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array | None = None,
    terminations: jax.Array | None = None,
    next_values: jax.Array | None = None,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    rewards = rewards.astype(jnp.float32)
    values = values.astype(jnp.float32)
    dones = dones.astype(jnp.float32)
    if terminations is None:
        terminations = dones
    terminations = terminations.astype(jnp.float32)
    if next_values is None:
        if next_value is None:
            raise ValueError("next_value or next_values must be provided.")
        next_values = jnp.concatenate(
            [values[1:], next_value.astype(jnp.float32)[None, ...]],
            axis=0,
        )
    next_values = next_values.astype(jnp.float32)

    def scan_step(
        last_gae: jax.Array,
        transition: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        reward, value, next_value_at_step, done, terminated = transition
        bootstrap_mask = 1.0 - terminated
        continuation_mask = 1.0 - done
        delta = reward + float(gamma) * next_value_at_step * bootstrap_mask - value
        advantage = delta + float(gamma) * float(gae_lambda) * continuation_mask * last_gae
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        scan_step,
        jnp.zeros_like(values[-1]),
        (
            rewards[::-1],
            values[::-1],
            next_values[::-1],
            dones[::-1],
            terminations[::-1],
        ),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def _flatten_rollout(rollout: Rollout, *, args: argparse.Namespace) -> TrainingBatch:
    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        terminations=rollout.terminations,
        next_values=rollout.next_values,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    batch_size = args.num_steps * args.num_envs
    return TrainingBatch(
        actor_obs=rollout.actor_obs.reshape(batch_size, args.num_ants, -1),
        central_obs=rollout.central_obs.reshape(batch_size, -1),
        actions=rollout.actions.reshape(batch_size, args.num_ants, 2),
        old_logprobs=rollout.logprobs.reshape(batch_size, args.num_ants),
        advantages=advantages.reshape(batch_size),
        returns=returns.reshape(batch_size),
    )


def _split_minibatches(batch: TrainingBatch, *, args: argparse.Namespace) -> TrainingBatch:
    minibatch_size = (args.num_steps * args.num_envs) // args.num_minibatches
    return jax.tree_util.tree_map(
        lambda value: value.reshape((args.num_minibatches, minibatch_size) + value.shape[1:]),
        batch,
    )


def _shuffle_batch(batch: TrainingBatch, *, key: jax.Array) -> TrainingBatch:
    batch_size = batch.advantages.shape[0]
    permutation = jax.random.permutation(key, batch_size)
    return jax.tree_util.tree_map(lambda value: value[permutation], batch)


def _global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _clip_tree_by_norm(tree: Any, *, norm: jax.Array, max_norm: float | None) -> Any:
    if max_norm is None or float(max_norm) <= 0.0:
        return tree
    scale = jnp.minimum(1.0, float(max_norm) / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda grad: grad * scale, tree)


def _clip_actor_critic_gradients(
    grads: JaxMAPPOParams,
    *,
    actor_max_grad_norm: float | None,
    critic_max_grad_norm: float | None,
) -> tuple[JaxMAPPOParams, GradientNorms]:
    actor_grads = (grads.actor_body, grads.move_head, grads.write_head)
    critic_grads = (grads.critic_body, grads.value_head)
    actor_norm = _global_norm(actor_grads)
    critic_norm = _global_norm(critic_grads)
    clipped_actor_body, clipped_move_head, clipped_write_head = _clip_tree_by_norm(
        actor_grads,
        norm=actor_norm,
        max_norm=actor_max_grad_norm,
    )
    clipped_critic_body, clipped_value_head = _clip_tree_by_norm(
        critic_grads,
        norm=critic_norm,
        max_norm=critic_max_grad_norm,
    )
    clipped_grads = JaxMAPPOParams(
        actor_body=clipped_actor_body,
        move_head=clipped_move_head,
        write_head=clipped_write_head,
        critic_body=clipped_critic_body,
        value_head=clipped_value_head,
    )
    return clipped_grads, GradientNorms(
        global_norm=_global_norm(grads),
        actor_norm=actor_norm,
        critic_norm=critic_norm,
    )


def init_adam_state(params: JaxMAPPOParams) -> AdamState:
    return AdamState(
        count=jnp.asarray(0, dtype=jnp.int32),
        m=jax.tree_util.tree_map(jnp.zeros_like, params),
        v=jax.tree_util.tree_map(jnp.zeros_like, params),
    )


def adam_update(
    params: JaxMAPPOParams,
    grads: JaxMAPPOParams,
    state: AdamState,
    *,
    learning_rate: float | jax.Array,
    max_grad_norm: float,
    actor_max_grad_norm: float | None = None,
    critic_max_grad_norm: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-5,
) -> tuple[JaxMAPPOParams, AdamState, GradientNorms]:
    if actor_max_grad_norm is None and critic_max_grad_norm is None:
        grad_norm = _global_norm(grads)
        actor_norm = _global_norm((grads.actor_body, grads.move_head, grads.write_head))
        critic_norm = _global_norm((grads.critic_body, grads.value_head))
        if max_grad_norm > 0:
            scale = jnp.minimum(1.0, float(max_grad_norm) / (grad_norm + 1e-6))
            grads = jax.tree_util.tree_map(lambda grad: grad * scale, grads)
        norms = GradientNorms(
            global_norm=grad_norm,
            actor_norm=actor_norm,
            critic_norm=critic_norm,
        )
    else:
        if actor_max_grad_norm is None:
            actor_max_grad_norm = max_grad_norm
        if critic_max_grad_norm is None:
            critic_max_grad_norm = max_grad_norm
        grads, norms = _clip_actor_critic_gradients(
            grads,
            actor_max_grad_norm=actor_max_grad_norm,
            critic_max_grad_norm=critic_max_grad_norm,
        )

    count = state.count + 1
    m = jax.tree_util.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, state.m, grads)
    v = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        state.v,
        grads,
    )
    m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**count), m)
    v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**count), v)
    next_params = jax.tree_util.tree_map(
        lambda param, mh, vh: param - learning_rate * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return next_params, AdamState(count=count, m=m, v=v), norms


def _ppo_loss(
    params: JaxMAPPOParams,
    batch: TrainingBatch,
    *,
    args: argparse.Namespace,
) -> tuple[jax.Array, UpdateMetrics]:
    new_logprobs, entropy, values = evaluate_actions(
        params,
        batch.actor_obs,
        batch.central_obs,
        batch.actions,
        policy_temperature=float(getattr(args, "training_rollout_temperature", 1.0)),
        **critic_forward_kwargs_from_args(args),
    )
    advantages = batch.advantages
    if args.norm_adv:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    agent_advantages = advantages[:, None]
    logratio = new_logprobs - batch.old_logprobs
    ratio = jnp.exp(logratio)

    policy_loss_1 = -agent_advantages * ratio
    policy_loss_2 = -agent_advantages * jnp.clip(
        ratio,
        1.0 - args.clip_coef,
        1.0 + args.clip_coef,
    )
    policy_loss = jnp.mean(jnp.maximum(policy_loss_1, policy_loss_2))
    value_loss = 0.5 * jnp.mean(jnp.square(values - batch.returns))
    entropy_mean = jnp.mean(entropy)
    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_mean
    approx_kl = jnp.mean((ratio - 1.0) - logratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > args.clip_coef).astype(jnp.float32))
    return loss, UpdateMetrics(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
        actor_grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
        critic_grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def update_agent(
    *,
    args: argparse.Namespace,
    params: JaxMAPPOParams,
    opt_state: AdamState,
    rollout: Rollout,
    learning_rate: float | jax.Array,
    key: jax.Array,
) -> tuple[JaxMAPPOParams, AdamState, UpdateMetrics]:
    batch = _flatten_rollout(rollout, args=args)

    def minibatch_step(
        carry: tuple[JaxMAPPOParams, AdamState],
        minibatch: TrainingBatch,
    ) -> tuple[tuple[JaxMAPPOParams, AdamState], UpdateMetrics]:
        current_params, current_opt_state = carry
        (loss, metrics), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(
            current_params,
            minibatch,
            args=args,
        )
        del loss
        next_params, next_opt_state, grad_norms = adam_update(
            current_params,
            grads,
            current_opt_state,
            learning_rate=learning_rate,
            max_grad_norm=args.max_grad_norm,
            actor_max_grad_norm=getattr(args, "actor_max_grad_norm", None),
            critic_max_grad_norm=getattr(args, "critic_max_grad_norm", None),
        )
        return (
            next_params,
            next_opt_state,
        ), metrics._replace(
            grad_norm=grad_norms.global_norm,
            actor_grad_norm=grad_norms.actor_norm,
            critic_grad_norm=grad_norms.critic_norm,
        )

    def epoch_step(
        carry: tuple[JaxMAPPOParams, AdamState],
        epoch_key: jax.Array,
    ) -> tuple[tuple[JaxMAPPOParams, AdamState], UpdateMetrics]:
        minibatches = _split_minibatches(_shuffle_batch(batch, key=epoch_key), args=args)
        next_carry, minibatch_metrics = jax.lax.scan(minibatch_step, carry, minibatches)
        mean_metrics = jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=0), minibatch_metrics)
        return next_carry, mean_metrics

    epoch_keys = jax.random.split(key, args.update_epochs)
    (params, opt_state), epoch_metrics = jax.lax.scan(
        epoch_step,
        (params, opt_state),
        epoch_keys,
    )
    final_metrics = jax.tree_util.tree_map(lambda value: value[-1], epoch_metrics)
    return params, opt_state, final_metrics
