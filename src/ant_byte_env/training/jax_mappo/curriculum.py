"""Curriculum reset helpers for vectorized JAX MAPPO environments."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from ant_byte_env.jax_env import JaxAntByteForagingEnv, JaxAntState, JaxObs


def _fixed_cookie_positions(
    hub_positions: jax.Array,
    *,
    width: int,
    height: int,
    distance: int,
) -> jax.Array:
    offsets = jnp.asarray(
        [[distance, 0], [-distance, 0], [0, distance], [0, -distance]],
        dtype=jnp.int32,
    )
    candidates = hub_positions[:, None, :] + offsets[None, :, :]
    valid = (
        (0 <= candidates[..., 0])
        & (candidates[..., 0] < width)
        & (0 <= candidates[..., 1])
        & (candidates[..., 1] < height)
    )
    selected = candidates[jnp.arange(hub_positions.shape[0]), jnp.argmax(valid, axis=1)]
    fallback = jnp.where(
        jnp.any(hub_positions != jnp.asarray([0, 0], dtype=jnp.int32), axis=1)[:, None],
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([min(width - 1, 1), 0 if width > 1 else min(height - 1, 1)], dtype=jnp.int32),
    )
    return jnp.where(jnp.any(valid, axis=1)[:, None], selected, fallback)[:, None, :]


def reset_batch(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    key: jax.Array,
) -> tuple[JaxAntState, JaxObs]:
    reset_keys = jax.random.split(key, args.num_envs)
    if args.random_hub:
        hub_key_x, hub_key_y = jax.random.split(jax.random.fold_in(key, 11))
        hub_positions = jnp.stack(
            [
                jax.random.randint(hub_key_x, (args.num_envs,), 0, args.width),
                jax.random.randint(hub_key_y, (args.num_envs,), 0, args.height),
            ],
            axis=-1,
        ).astype(jnp.int32)
    else:
        hub_positions = jnp.broadcast_to(
            jnp.asarray([args.width // 2, args.height // 2], dtype=jnp.int32),
            (args.num_envs, 2),
        )

    if args.random_food:
        states, obs, _ = jax.vmap(lambda reset_key, hub: env.reset(reset_key, hub_pos=hub))(
            reset_keys,
            hub_positions,
        )
        return states, obs

    food_positions = _fixed_cookie_positions(
        hub_positions,
        width=args.width,
        height=args.height,
        distance=args.cookie_distance,
    )
    states, obs, _ = jax.vmap(
        lambda reset_key, hub, food: env.reset(
            reset_key,
            hub_pos=hub,
            food_positions=food,
        )
    )(reset_keys, hub_positions, food_positions)
    return states, obs
