"""Trainer-side reward shaping and write diagnostics for JAX MAPPO."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ant_byte_env import ACTION_STAY, DEFAULT_WRITE_BITS, max_write_value
from ant_byte_env.jax_env import JaxObs
from ant_byte_env.training.jax_mappo.observations import _active_grid_size

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

    uses_byte_shaping = (
        delivery_byte_trail_bonus > 0.0
        or byte_follow_bonus > 0.0
        or carrying_byte_write_bonus > 0.0
    )
    if uses_byte_shaping:
        previous_bytes = previous_obs["bytes"]
    else:
        previous_bytes = jnp.zeros_like(previous_obs["food"], dtype=jnp.uint8)
    _, previous_height, previous_width = previous_bytes.shape
    previous_active_size = _active_grid_size(
        previous_obs,
        fallback_height=previous_height,
        fallback_width=previous_width,
    )
    previous_carrying = previous_obs["ants_carrying"].astype(jnp.bool_)
    next_carrying = next_obs["ants_carrying"].astype(jnp.bool_)
    previous_positions = previous_obs["ants_pos"].astype(jnp.int32)
    next_positions = next_obs["ants_pos"].astype(jnp.int32)

    if actions is None:
        applied_write_values = jnp.zeros(previous_carrying.shape, dtype=jnp.uint32)
    else:
        applied_write_values = _applied_write_values(
            actions,
            write_while_moving=write_while_moving,
            per_ant_write_channels=per_ant_write_channels,
            write_bits=write_bits,
        )

    target_bytes = _grid_values_at_positions(previous_bytes, next_positions)
    target_food = _grid_values_at_positions(previous_obs["food"], next_positions)
    target_is_hub = jnp.all(next_positions == previous_obs["hub_pos"][:, None, :], axis=-1)
    fresh_carrying_writes = (
        previous_carrying
        & (applied_write_values > 0)
        & (target_bytes == 0)
        & (target_food <= 0)
        & jnp.logical_not(target_is_hub)
    )

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
        progress = _forage_distance_progress(previous_obs=previous_obs, next_obs=next_obs)
        shaped_rewards += float(distance_bonus) * progress
    if carrying_hub_distance_bonus > 0.0:
        carrying_progress = _carrying_hub_distance_progress(
            previous_obs=previous_obs,
            next_obs=next_obs,
        )
        shaped_rewards += float(carrying_hub_distance_bonus) * carrying_progress
    return shaped_rewards


def _forage_distance_progress(*, previous_obs: JaxObs, next_obs: JaxObs) -> jax.Array:
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
    height, width = food.shape[1:]
    normalizer = jnp.asarray(max(height + width - 2, 1), dtype=jnp.float32)
    progress = (previous_distance - next_distance) / normalizer
    progress = jnp.where(same_target_mode, progress, 0.0)
    return jnp.sum(progress, axis=-1).astype(jnp.float32)


def _carrying_hub_distance_progress(
    *,
    previous_obs: JaxObs,
    next_obs: JaxObs,
) -> jax.Array:
    previous_carrying = previous_obs["ants_carrying"].astype(jnp.bool_)
    next_carrying = next_obs["ants_carrying"].astype(jnp.bool_)
    stayed_carrying = previous_carrying & next_carrying
    previous_distance = _hub_distances(previous_obs["hub_pos"], previous_obs["ants_pos"])
    next_distance = _hub_distances(previous_obs["hub_pos"], next_obs["ants_pos"])
    _, height, width = previous_obs["food"].shape
    normalizer = jnp.asarray(max(height + width - 2, 1), dtype=jnp.float32)
    progress = (previous_distance - next_distance) / normalizer
    progress = jnp.where(stayed_carrying, progress, 0.0)
    return jnp.sum(progress, axis=-1).astype(jnp.float32)


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

