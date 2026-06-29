"""Team-perspective observations for adversarial JAX MAPPO."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ant_byte_env import DEFAULT_WRITE_BITS, MAX_WRITE_BITS, max_write_value
from ant_byte_env.env import DEFAULT_ACTOR_VISION_DEPTH
from ant_byte_env.jax_env import JaxObs
from ant_byte_env.training.jax_mappo.observations import (
    _active_grid_size,
    _agent_identity_features,
    _ants_facing_or_default,
    _facing_one_hot,
    _normalize_positions,
    build_local_border_patches,
    build_local_byte_bit_patches,
    build_local_grid_patches,
)


def _team_slices(team: int, *, num_ants_per_team: int) -> tuple[slice, slice]:
    own = slice(0, num_ants_per_team) if int(team) == 0 else slice(num_ants_per_team, 2 * num_ants_per_team)
    opponent = slice(num_ants_per_team, 2 * num_ants_per_team) if int(team) == 0 else slice(0, num_ants_per_team)
    return own, opponent


def _team_parts(
    obs: JaxObs,
    *,
    team: int,
    num_ants_per_team: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    own, opponent = _team_slices(team, num_ants_per_team=num_ants_per_team)
    ants_facing = _ants_facing_or_default(obs)
    return (
        obs["ants_pos"][:, own, :],
        obs["ants_pos"][:, opponent, :],
        obs["ants_carrying"][:, own],
        obs["ants_carrying"][:, opponent],
        ants_facing[:, own],
        ants_facing[:, opponent],
    )


def _signed_ant_grid(
    obs: JaxObs,
    *,
    team: int,
    num_ants_per_team: int,
) -> jax.Array:
    food = obs["food"]
    own_pos, opponent_pos, *_ = _team_parts(
        obs,
        team=team,
        num_ants_per_team=num_ants_per_team,
    )
    batch_size, height, width = food.shape
    batch_index = jnp.arange(batch_size)[:, None]
    grid = jnp.zeros((batch_size, height, width), dtype=jnp.float32)
    grid = grid.at[batch_index, own_pos[..., 1], own_pos[..., 0]].add(1.0)
    return grid.at[batch_index, opponent_pos[..., 1], opponent_pos[..., 0]].add(-1.0)


def _signed_hub_grid(obs: JaxObs, *, team: int) -> jax.Array:
    food = obs["food"]
    hub_pos = obs["hub_pos"].astype(jnp.int32)
    batch_size, height, width = food.shape
    own_team = int(team)
    opponent_team = 1 - own_team
    batch_index = jnp.arange(batch_size)
    grid = jnp.zeros((batch_size, height, width), dtype=jnp.float32)
    grid = grid.at[
        batch_index,
        hub_pos[:, own_team, 1],
        hub_pos[:, own_team, 0],
    ].set(1.0)
    return grid.at[
        batch_index,
        hub_pos[:, opponent_team, 1],
        hub_pos[:, opponent_team, 0],
    ].set(-1.0)


def build_team_actor_observations(
    obs: JaxObs,
    *,
    team: int,
    num_ants_per_team: int,
    food_scale: int = 1,
    actor_vision_radius: int = DEFAULT_ACTOR_VISION_DEPTH,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> jax.Array:
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    food = obs["food"].astype(jnp.float32)
    own_pos, _, own_carrying, _, own_facing, _ = _team_parts(
        obs,
        team=team,
        num_ants_per_team=num_ants_per_team,
    )
    local_food = build_local_grid_patches(
        food,
        own_pos,
        radius=actor_vision_radius,
        ants_facing=own_facing,
    )
    local_food = local_food / max(float(food_scale), 1.0)
    local_ants = build_local_grid_patches(
        _signed_ant_grid(obs, team=team, num_ants_per_team=num_ants_per_team),
        own_pos,
        radius=actor_vision_radius,
        ants_facing=own_facing,
    )
    local_ants = local_ants / max(float(num_ants_per_team), 1.0)
    local_byte_bits = build_local_byte_bit_patches(
        obs["bytes"],
        own_pos,
        radius=actor_vision_radius,
        ants_facing=own_facing,
        write_bits=write_bits,
    )
    local_hub = build_local_grid_patches(
        _signed_hub_grid(obs, team=team),
        own_pos,
        radius=actor_vision_radius,
        ants_facing=own_facing,
    )
    active_grid_size = _active_grid_size(
        obs,
        fallback_height=food.shape[1],
        fallback_width=food.shape[2],
    )
    local_border = build_local_border_patches(
        own_pos,
        ants_facing=own_facing,
        grid_height=active_grid_size[:, 1],
        grid_width=active_grid_size[:, 0],
        radius=actor_vision_radius,
    )
    obstacles = obs.get("obstacles")
    if obstacles is None:
        obstacles = jnp.zeros_like(food, dtype=jnp.float32)
    local_obstacles = build_local_grid_patches(
        obstacles.astype(jnp.float32),
        own_pos,
        radius=actor_vision_radius,
        ants_facing=own_facing,
    )
    local_border = jnp.maximum(local_border, local_obstacles)
    return jnp.concatenate(
        [
            local_food,
            local_ants,
            local_byte_bits,
            local_hub,
            local_border,
            _agent_identity_features(own_pos),
            own_carrying.astype(jnp.float32)[..., None],
            _facing_one_hot(own_facing),
        ],
        axis=-1,
    )


def build_team_central_observations(
    obs: JaxObs,
    *,
    team: int,
    num_ants_per_team: int,
    food_scale: int,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> jax.Array:
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    food = obs["food"].astype(jnp.float32)
    batch_size, height, width = food.shape
    own_pos, opponent_pos, own_carrying, opponent_carrying, own_facing, opponent_facing = (
        _team_parts(obs, team=team, num_ants_per_team=num_ants_per_team)
    )
    hub_pos = obs["hub_pos"].astype(jnp.int32)
    own_team = int(team)
    opponent_team = 1 - own_team
    active_grid_size = _active_grid_size(obs, fallback_height=height, fallback_width=width)
    grid_size = active_grid_size / jnp.asarray(
        [max(float(width), 1.0), max(float(height), 1.0)],
        dtype=jnp.float32,
    )
    return jnp.concatenate(
        [
            _normalize_positions(own_pos, height=height, width=width).reshape(batch_size, -1),
            _normalize_positions(opponent_pos, height=height, width=width).reshape(
                batch_size,
                -1,
            ),
            own_carrying.astype(jnp.float32).reshape(batch_size, -1),
            opponent_carrying.astype(jnp.float32).reshape(batch_size, -1),
            _facing_one_hot(own_facing).reshape(batch_size, -1),
            _facing_one_hot(opponent_facing).reshape(batch_size, -1),
            (_signed_ant_grid(obs, team=team, num_ants_per_team=num_ants_per_team)
             / max(float(num_ants_per_team), 1.0)).reshape(batch_size, -1),
            (food / max(float(food_scale), 1.0)).reshape(batch_size, -1),
            (
                obs["bytes"].astype(jnp.float32)
                / max(float(max_write_value(write_bits)), 1.0)
            ).reshape(batch_size, -1),
            _normalize_positions(
                hub_pos[:, own_team, :],
                height=height,
                width=width,
            ).reshape(batch_size, -1),
            _normalize_positions(
                hub_pos[:, opponent_team, :],
                height=height,
                width=width,
            ).reshape(batch_size, -1),
            grid_size,
        ],
        axis=-1,
    )
