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


def _flat_positions(flat_indices: jax.Array, *, width: int) -> jax.Array:
    return jnp.stack(
        [flat_indices % width, flat_indices // width],
        axis=-1,
    ).astype(jnp.int32)


def _batched_open_cell_mask(
    env: JaxAntByteForagingEnv,
    obstacles: jax.Array,
) -> jax.Array:
    if obstacles is None:
        return jnp.ones((1, env.width * env.height), dtype=jnp.bool_)
    return jnp.logical_not(obstacles.reshape((obstacles.shape[0], -1)))


def _prefer_layout_margin(
    *,
    env: JaxAntByteForagingEnv,
    flat_indices: jax.Array,
    candidate_mask: jax.Array,
    required_count: int = 1,
) -> jax.Array:
    flat_x = flat_indices % env.width
    flat_y = flat_indices // env.width
    interior_mask = candidate_mask & env._inside_layout_margin(flat_x, flat_y)[None, :]
    enough_interior = (
        jnp.sum(interior_mask.astype(jnp.int32), axis=1, keepdims=True)
        >= int(required_count)
    )
    return jnp.where(enough_interior, interior_mask, candidate_mask)


def _prefer_cookie_distance(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    flat_indices: jax.Array,
    hub_positions: jax.Array,
    candidate_mask: jax.Array,
    required_count: int = 1,
) -> jax.Array:
    if not bool(getattr(args, "random_food_same_distance", False)):
        return candidate_mask

    distance = int(getattr(args, "cookie_distance", 0))
    flat_x = flat_indices % env.width
    flat_y = flat_indices // env.width
    hub_x = hub_positions[:, 0:1]
    hub_y = hub_positions[:, 1:2]
    ring_mask = candidate_mask & (
        jnp.maximum(
            jnp.abs(flat_x[None, :] - hub_x),
            jnp.abs(flat_y[None, :] - hub_y),
        )
        == distance
    )
    enough_ring = (
        jnp.sum(ring_mask.astype(jnp.int32), axis=1, keepdims=True)
        >= int(required_count)
    )
    return jnp.where(enough_ring, ring_mask, candidate_mask)


def _prefer_hub_center_window(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    flat_indices: jax.Array,
    candidate_mask: jax.Array,
) -> jax.Array:
    window_size = int(getattr(args, "hub_center_window_size", 0))
    if window_size <= 0:
        return candidate_mask
    x_start = (env.width - window_size) // 2
    y_start = (env.height - window_size) // 2
    flat_x = flat_indices % env.width
    flat_y = flat_indices // env.width
    center_mask = candidate_mask & (
        (flat_x[None, :] >= x_start)
        & (flat_x[None, :] < x_start + window_size)
        & (flat_y[None, :] >= y_start)
        & (flat_y[None, :] < y_start + window_size)
    )
    return jnp.where(
        jnp.any(center_mask, axis=1, keepdims=True),
        center_mask,
        candidate_mask,
    )


def _sample_obstacle_grids(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    key: jax.Array,
    previous_obs: JaxObs | None,
) -> jax.Array:
    obstacle_bank = getattr(env, "obstacle_bank", None)
    if obstacle_bank is None:
        return jnp.zeros((args.num_envs, env.height, env.width), dtype=jnp.bool_)

    obstacle_bank = jnp.asarray(obstacle_bank, dtype=jnp.bool_)
    if obstacle_bank.shape[0] <= 1:
        return jnp.broadcast_to(obstacle_bank[0], (args.num_envs, env.height, env.width))

    candidate_mask = jnp.ones((args.num_envs, obstacle_bank.shape[0]), dtype=jnp.bool_)
    if previous_obs is not None and "obstacles" in previous_obs:
        previous_obstacles = previous_obs["obstacles"].astype(jnp.bool_)
        same_as_previous = jnp.all(
            obstacle_bank[None, :, :, :] == previous_obstacles[:, None, :, :],
            axis=(2, 3),
        )
        candidate_mask = jnp.where(
            jnp.any(jnp.logical_not(same_as_previous), axis=1, keepdims=True),
            jnp.logical_not(same_as_previous),
            candidate_mask,
        )

    scores = jax.random.uniform(key, candidate_mask.shape, dtype=jnp.float32)
    selected_layouts = jnp.argmax(jnp.where(candidate_mask, scores, -jnp.inf), axis=1)
    return obstacle_bank[selected_layouts]


def _sample_hub_positions(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    key: jax.Array,
    obstacles: jax.Array,
    previous_obs: JaxObs | None,
) -> jax.Array:
    if not args.random_hub:
        center = jnp.asarray([args.width // 2, args.height // 2], dtype=jnp.int32)
        return jax.vmap(
            lambda obstacle_grid: env._nearest_open_position(center, obstacle_grid)
        )(obstacles)

    flat_indices = jnp.arange(env.width * env.height, dtype=jnp.int32)
    open_mask = _batched_open_cell_mask(env, obstacles)
    candidate_mask = jnp.broadcast_to(open_mask, (args.num_envs, flat_indices.shape[0]))
    candidate_mask = _prefer_hub_center_window(
        args=args,
        env=env,
        flat_indices=flat_indices,
        candidate_mask=candidate_mask,
    )
    candidate_mask = _prefer_layout_margin(
        env=env,
        flat_indices=flat_indices,
        candidate_mask=candidate_mask,
    )
    if previous_obs is not None:
        previous_hub = previous_obs["hub_pos"].astype(jnp.int32)
        previous_hub_flat = previous_hub[:, 1] * env.width + previous_hub[:, 0]
        preferred_mask = candidate_mask & (flat_indices[None, :] != previous_hub_flat[:, None])
        candidate_mask = jnp.where(
            jnp.any(preferred_mask, axis=1, keepdims=True),
            preferred_mask,
            candidate_mask,
        )

    scores = jax.random.uniform(key, candidate_mask.shape, dtype=jnp.float32)
    selected_flat = jnp.argmax(jnp.where(candidate_mask, scores, -jnp.inf), axis=1)
    return _flat_positions(selected_flat, width=env.width)


def _sample_clustered_food_positions(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    key: jax.Array,
    candidate_mask: jax.Array,
    flat_indices: jax.Array,
    source_count: int | None = None,
) -> jax.Array:
    source_count = int(
        getattr(env, "source_count", 0) if source_count is None else source_count
    )
    cluster_count = int(getattr(args, "food_cluster_count", 0))
    cluster_radius = int(getattr(args, "food_cluster_radius", 0))
    if source_count <= 0:
        return jnp.zeros((args.num_envs, 0, 2), dtype=jnp.int32)
    if cluster_count <= 0:
        raise ValueError("clustered food sampling requires a positive cluster count.")

    flat_x = flat_indices % env.width
    flat_y = flat_indices // env.width
    env_indices = jnp.arange(args.num_envs, dtype=jnp.int32)
    centers: list[jax.Array] = []
    center_key, position_key = jax.random.split(key)
    center_keys = jax.random.split(center_key, cluster_count)
    min_center_gap = 2 * cluster_radius + 1
    for center_index in range(cluster_count):
        available_centers = candidate_mask
        if centers:
            separated = available_centers
            for center in centers:
                center_x = center % env.width
                center_y = center // env.width
                far_enough = (
                    jnp.maximum(
                        jnp.abs(flat_x[None, :] - center_x[:, None]),
                        jnp.abs(flat_y[None, :] - center_y[:, None]),
                    )
                    > min_center_gap
                )
                separated = separated & far_enough
            available_centers = jnp.where(
                jnp.any(separated, axis=1, keepdims=True),
                separated,
                available_centers,
            )
        scores = jax.random.uniform(center_keys[center_index], candidate_mask.shape)
        selected_center = jnp.argmax(
            jnp.where(available_centers, scores, -jnp.inf),
            axis=1,
        ).astype(jnp.int32)
        centers.append(selected_center)

    selected_mask = jnp.zeros_like(candidate_mask)
    selected_flats: list[jax.Array] = []
    position_keys = jax.random.split(position_key, cluster_count)
    for center_index, center in enumerate(centers):
        quota = source_count // cluster_count + int(center_index < source_count % cluster_count)
        if quota <= 0:
            continue
        center_x = center % env.width
        center_y = center // env.width
        in_cluster = (
            jnp.maximum(
                jnp.abs(flat_x[None, :] - center_x[:, None]),
                jnp.abs(flat_y[None, :] - center_y[:, None]),
            )
            <= cluster_radius
        )
        available = candidate_mask & jnp.logical_not(selected_mask)
        cluster_mask = available & in_cluster
        cluster_mask = jnp.where(
            jnp.sum(cluster_mask.astype(jnp.int32), axis=1, keepdims=True) >= quota,
            cluster_mask,
            available,
        )
        scores = jax.random.uniform(position_keys[center_index], candidate_mask.shape)
        _, selected = jax.lax.top_k(
            jnp.where(cluster_mask, scores, -jnp.inf),
            quota,
        )
        selected = selected.astype(jnp.int32)
        selected_flats.append(selected)
        selected_mask = selected_mask.at[env_indices[:, None], selected].set(True)

    return _flat_positions(jnp.concatenate(selected_flats, axis=1), width=env.width)


def _sample_food_positions(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    key: jax.Array,
    hub_positions: jax.Array,
    obstacles: jax.Array,
    previous_obs: JaxObs | None,
    previous_food: jax.Array | None,
    source_count: int | None = None,
    exclude_food: jax.Array | None = None,
) -> jax.Array:
    source_count = int(
        getattr(env, "source_count", 0) if source_count is None else source_count
    )
    if source_count <= 0:
        return jnp.zeros((args.num_envs, 0, 2), dtype=jnp.int32)

    flat_indices = jnp.arange(env.width * env.height, dtype=jnp.int32)
    open_mask = _batched_open_cell_mask(env, obstacles)
    hub_flat = hub_positions[:, 1] * env.width + hub_positions[:, 0]
    candidate_mask = jnp.broadcast_to(open_mask, (args.num_envs, flat_indices.shape[0]))
    candidate_mask = candidate_mask & (flat_indices[None, :] != hub_flat[:, None])
    candidate_mask = _prefer_layout_margin(
        env=env,
        flat_indices=flat_indices,
        candidate_mask=candidate_mask,
        required_count=source_count,
    )
    candidate_mask = _prefer_cookie_distance(
        args=args,
        env=env,
        flat_indices=flat_indices,
        hub_positions=hub_positions,
        candidate_mask=candidate_mask,
        required_count=source_count,
    )
    if exclude_food is not None:
        exclude_mask = exclude_food.reshape((args.num_envs, -1)) > 0
        preferred_mask = candidate_mask & jnp.logical_not(exclude_mask)
        enough_preferred = jnp.sum(preferred_mask, axis=1, keepdims=True) >= source_count
        candidate_mask = jnp.where(enough_preferred, preferred_mask, candidate_mask)

    previous_food_grid = previous_food
    if previous_food_grid is None and previous_obs is not None:
        previous_food_grid = previous_obs["food"]
    if previous_food_grid is not None:
        previous_food_mask = previous_food_grid.reshape((args.num_envs, -1)) > 0
        preferred_mask = candidate_mask & jnp.logical_not(previous_food_mask)
        enough_preferred = jnp.sum(preferred_mask, axis=1, keepdims=True) >= source_count
        candidate_mask = jnp.where(enough_preferred, preferred_mask, candidate_mask)

    if int(getattr(args, "food_cluster_count", 0)) > 0:
        return _sample_clustered_food_positions(
            args=args,
            env=env,
            key=key,
            candidate_mask=candidate_mask,
            flat_indices=flat_indices,
            source_count=source_count,
        )

    scores = jax.random.uniform(key, candidate_mask.shape, dtype=jnp.float32)
    _, selected_flat = jax.lax.top_k(
        jnp.where(candidate_mask, scores, -jnp.inf),
        source_count,
    )
    return _flat_positions(selected_flat.astype(jnp.int32), width=env.width)


def _positions_to_source_grid(
    positions: jax.Array,
    *,
    height: int,
    width: int,
) -> jax.Array:
    if positions.shape[1] == 0:
        return jnp.zeros((positions.shape[0], height, width), dtype=jnp.int32)
    env_indices = jnp.arange(positions.shape[0], dtype=jnp.int32)[:, None]
    grid = jnp.zeros((positions.shape[0], height, width), dtype=jnp.int32)
    return grid.at[
        env_indices,
        positions[..., 1],
        positions[..., 0],
    ].set(1)


def reset_batch(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    key: jax.Array,
    previous_obs: JaxObs | None = None,
    previous_food: jax.Array | None = None,
) -> tuple[JaxAntState, JaxObs]:
    reset_keys = jax.random.split(key, args.num_envs)
    if bool(getattr(args, "autocurriculum", False)) or bool(
        getattr(args, "distance_autocurriculum", False)
    ):
        states, obs, _ = jax.vmap(env.reset)(reset_keys)
        return states, obs

    maze_key, hub_key, food_key = jax.random.split(jax.random.fold_in(key, 11), 3)
    obstacles = _sample_obstacle_grids(
        args=args,
        env=env,
        key=maze_key,
        previous_obs=previous_obs,
    )
    hub_positions = _sample_hub_positions(
        args=args,
        env=env,
        key=hub_key,
        obstacles=obstacles,
        previous_obs=previous_obs,
    )

    if args.random_food:
        food_key, lethal_food_key = jax.random.split(food_key)
        food_positions = _sample_food_positions(
            args=args,
            env=env,
            key=food_key,
            hub_positions=hub_positions,
            obstacles=obstacles,
            previous_obs=previous_obs,
            previous_food=previous_food,
        )
        safe_source_grid = _positions_to_source_grid(
            food_positions,
            height=env.height,
            width=env.width,
        )
        lethal_food_positions = _sample_food_positions(
            args=args,
            env=env,
            key=lethal_food_key,
            hub_positions=hub_positions,
            obstacles=obstacles,
            previous_obs=previous_obs,
            previous_food=None,
            source_count=int(getattr(env, "lethal_source_count", 0)),
            exclude_food=safe_source_grid,
        )
        if int(getattr(env, "lethal_food_count", 0)) > 0:
            states, obs, _ = jax.vmap(
                lambda reset_key, hub, food, lethal_food, obstacle_grid: env.reset(
                    reset_key,
                    hub_pos=hub,
                    food_positions=food,
                    lethal_food_positions=lethal_food,
                    obstacles=obstacle_grid,
                )
            )(
                reset_keys,
                hub_positions,
                food_positions,
                lethal_food_positions,
                obstacles,
            )
        else:
            states, obs, _ = jax.vmap(
                lambda reset_key, hub, food, obstacle_grid: env.reset(
                    reset_key,
                    hub_pos=hub,
                    food_positions=food,
                    obstacles=obstacle_grid,
                )
            )(
                reset_keys,
                hub_positions,
                food_positions,
                obstacles,
            )
        return states, obs

    food_positions = _fixed_cookie_positions(
        hub_positions,
        width=args.width,
        height=args.height,
        distance=args.cookie_distance,
    )
    states, obs, _ = jax.vmap(
        lambda reset_key, hub, food, obstacle_grid: env.reset(
            reset_key,
            hub_pos=hub,
            food_positions=food,
            obstacles=obstacle_grid,
        )
    )(reset_keys, hub_positions, food_positions, obstacles)
    return states, obs
