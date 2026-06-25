"""Actor-side JAX MAPPO checkpoint transfer helpers."""

from __future__ import annotations

import jax.numpy as jnp

from ant_byte_env import (
    DEFAULT_ACTOR_VISION_WIDTH,
    MOVEMENT_ACTION_COUNT,
    actor_vision_patch_size,
    write_value_count,
)
from ant_byte_env.training.jax_mappo.transfer_shapes import (
    FACING_FEATURE_COUNT,
    WRITE_HEAD_TRANSFER_MODES,
    actor_obs_dim_for_bits,
    agent_identity_feature_count,
    repeated_write_action_indices,
    source_actor_obs_dim,
    source_actor_patch_size,
    validate_write_head_transfer,
)
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams, LinearParams


def expand_params_for_write_bits(
    params: JaxMAPPOParams,
    *,
    old_bits: int,
    target_bits: int,
    actor_vision_radius: int,
    source_includes_ants_count: bool = True,
    source_includes_orientation: bool = False,
    source_includes_agent_identity: bool = False,
    source_layout: str = "centered",
    source_num_ants: int = 1,
    target_num_ants: int = 1,
    write_head_transfer: str = "repeat",
) -> JaxMAPPOParams:
    write_head_transfer = validate_write_head_transfer(write_head_transfer)
    source_identity_features = agent_identity_feature_count(
        source_num_ants,
        include_agent_identity=source_includes_agent_identity,
    )
    target_identity_features = agent_identity_feature_count(target_num_ants)
    if (
        target_bits == old_bits
        and source_includes_ants_count
        and source_includes_orientation
        and source_identity_features == target_identity_features
        and source_layout == "centered"
    ):
        return params
    if target_bits < old_bits:
        raise ValueError("target_bits must be at least old_bits.")
    return JaxMAPPOParams(
        actor_body=(
            expand_actor_input_layer(
                params.actor_body[0],
                old_bits=old_bits,
                target_bits=target_bits,
                actor_vision_radius=actor_vision_radius,
                source_includes_ants_count=source_includes_ants_count,
                source_includes_orientation=source_includes_orientation,
                source_includes_agent_identity=source_includes_agent_identity,
                source_layout=source_layout,
                source_num_ants=source_num_ants,
                target_num_ants=target_num_ants,
            ),
            params.actor_body[1],
        ),
        move_head=adapt_movement_head_layer(params.move_head),
        write_head=params.write_head
        if target_bits == old_bits
        else expand_write_head(
            params.write_head,
            old_bits=old_bits,
            target_bits=target_bits,
            transfer_mode=write_head_transfer,
        ),
        critic_body=params.critic_body,
        value_head=params.value_head,
    )


def resize_params_for_actor_vision_radius(
    params: JaxMAPPOParams,
    *,
    write_bits: int,
    source_actor_vision_radius: int,
    target_actor_vision_radius: int,
    source_num_ants: int = 1,
    target_num_ants: int = 1,
    source_includes_agent_identity: bool = True,
) -> JaxMAPPOParams:
    source_identity_features = agent_identity_feature_count(
        source_num_ants,
        include_agent_identity=source_includes_agent_identity,
    )
    target_identity_features = agent_identity_feature_count(target_num_ants)
    if (
        source_actor_vision_radius == target_actor_vision_radius
        and source_identity_features == target_identity_features
    ):
        return params
    return JaxMAPPOParams(
        actor_body=(
            resize_actor_input_layer_for_vision_radius(
                params.actor_body[0],
                write_bits=write_bits,
                source_actor_vision_radius=source_actor_vision_radius,
                target_actor_vision_radius=target_actor_vision_radius,
                source_num_ants=source_num_ants,
                target_num_ants=target_num_ants,
                source_includes_agent_identity=source_includes_agent_identity,
            ),
            params.actor_body[1],
        ),
        move_head=adapt_movement_head_layer(params.move_head),
        write_head=params.write_head,
        critic_body=params.critic_body,
        value_head=params.value_head,
    )


def adapt_movement_head(params: JaxMAPPOParams) -> tuple[JaxMAPPOParams, bool]:
    move_head = adapt_movement_head_layer(params.move_head)
    if move_head is params.move_head:
        return params, False
    return (
        JaxMAPPOParams(
            actor_body=params.actor_body,
            move_head=move_head,
            write_head=params.write_head,
            critic_body=params.critic_body,
            value_head=params.value_head,
        ),
        True,
    )


def adapt_movement_head_layer(layer: LinearParams) -> LinearParams:
    old_bias = jnp.asarray(layer.bias)
    old_count = int(old_bias.shape[0])
    if old_count == MOVEMENT_ACTION_COUNT:
        return layer
    legacy_turn_action_count = 4
    if old_count == legacy_turn_action_count:
        raise ValueError(
            "Legacy 4-action movement checkpoints cannot be automatically mapped "
            "onto the current cardinal movement action space."
        )
    raise ValueError(f"Checkpoint movement action count {old_count} does not match this run.")


def expand_actor_input_layer(
    layer: LinearParams,
    *,
    old_bits: int,
    target_bits: int,
    actor_vision_radius: int,
    source_includes_ants_count: bool = True,
    source_includes_orientation: bool = False,
    source_includes_agent_identity: bool = False,
    source_layout: str = "centered",
    source_num_ants: int = 1,
    target_num_ants: int = 1,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    target_patch_size = actor_vision_patch_size(actor_vision_radius)
    source_patch_size = source_actor_patch_size(
        actor_vision_radius=actor_vision_radius,
        source_layout=source_layout,
    )
    expected_old_dim = source_actor_obs_dim(
        write_bits=old_bits,
        actor_vision_radius=actor_vision_radius,
        num_ants=source_num_ants,
        include_ants_count=source_includes_ants_count,
        include_orientation=source_includes_orientation,
        include_agent_identity=source_includes_agent_identity,
        source_layout=source_layout,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=actor_vision_radius,
        num_ants=target_num_ants,
    )
    if old_weight.shape[0] != expected_old_dim:
        raise ValueError(
            f"Expected actor input dim {expected_old_dim}, got {old_weight.shape[0]}."
        )

    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    old_food = slice(0, source_patch_size)
    if source_includes_ants_count:
        old_ants_count = slice(source_patch_size, 2 * source_patch_size)
        old_bits_start = 2 * source_patch_size
        old_hub = slice(
            source_patch_size * (2 + old_bits),
            source_patch_size * (3 + old_bits),
        )
        old_border = slice(
            source_patch_size * (3 + old_bits),
            source_patch_size * (4 + old_bits),
        )
    else:
        old_ants_count = None
        old_bits_start = source_patch_size
        old_hub = slice(
            source_patch_size * (1 + old_bits),
            source_patch_size * (2 + old_bits),
        )
        old_border = slice(
            source_patch_size * (2 + old_bits),
            source_patch_size * (3 + old_bits),
        )
    old_bits_slices = [
        slice(
            old_bits_start + bit_index * source_patch_size,
            old_bits_start + (bit_index + 1) * source_patch_size,
        )
        for bit_index in range(old_bits)
    ]
    new_ants_count = slice(target_patch_size, 2 * target_patch_size)
    new_bits_slices = [
        slice(
            target_patch_size * (2 + bit_index),
            target_patch_size * (3 + bit_index),
        )
        for bit_index in range(old_bits)
    ]
    new_hub = slice(
        target_patch_size * (2 + target_bits),
        target_patch_size * (3 + target_bits),
    )
    new_border = slice(
        target_patch_size * (3 + target_bits),
        target_patch_size * (4 + target_bits),
    )
    old_tail_start = source_patch_size * (old_bits + (4 if source_includes_ants_count else 3))
    old_identity_width = agent_identity_feature_count(
        source_num_ants,
        include_agent_identity=source_includes_agent_identity,
    )
    old_identity = slice(old_tail_start, old_tail_start + old_identity_width)
    old_carrying = slice(old_identity.stop, old_identity.stop + 1)
    old_orientation = (
        slice(old_carrying.stop, old_carrying.stop + FACING_FEATURE_COUNT)
        if source_includes_orientation
        else None
    )
    new_tail_start = target_patch_size * (target_bits + 4)
    new_identity_width = agent_identity_feature_count(target_num_ants)
    new_identity = slice(new_tail_start, new_tail_start + new_identity_width)
    new_carrying = slice(new_identity.stop, new_identity.stop + 1)
    new_orientation = slice(
        new_carrying.stop,
        new_carrying.stop + FACING_FEATURE_COUNT,
    )

    new_weight = copy_actor_patch_channel(
        new_weight,
        old_weight,
        source=old_food,
        target=slice(0, target_patch_size),
        actor_vision_radius=actor_vision_radius,
        source_layout=source_layout,
    )
    if old_ants_count is not None:
        new_weight = copy_actor_patch_channel(
            new_weight,
            old_weight,
            source=old_ants_count,
            target=new_ants_count,
            actor_vision_radius=actor_vision_radius,
            source_layout=source_layout,
        )
    for old_bits_slice, new_bits_slice in zip(old_bits_slices, new_bits_slices):
        new_weight = copy_actor_patch_channel(
            new_weight,
            old_weight,
            source=old_bits_slice,
            target=new_bits_slice,
            actor_vision_radius=actor_vision_radius,
            source_layout=source_layout,
        )
    new_weight = copy_actor_patch_channel(
        new_weight,
        old_weight,
        source=old_hub,
        target=new_hub,
        actor_vision_radius=actor_vision_radius,
        source_layout=source_layout,
    )
    new_weight = copy_actor_patch_channel(
        new_weight,
        old_weight,
        source=old_border,
        target=new_border,
        actor_vision_radius=actor_vision_radius,
        source_layout=source_layout,
    )
    if old_identity_width > 0 and new_identity_width > 0:
        shared_identity_width = min(old_identity_width, new_identity_width)
        new_weight = new_weight.at[
            new_identity.start : new_identity.start + shared_identity_width,
            :,
        ].set(
            old_weight[
                old_identity.start : old_identity.start + shared_identity_width,
                :,
            ]
        )
    new_weight = new_weight.at[new_carrying, :].set(old_weight[old_carrying, :])
    if old_orientation is not None:
        new_weight = new_weight.at[new_orientation, :].set(old_weight[old_orientation, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def resize_actor_input_layer_for_vision_radius(
    layer: LinearParams,
    *,
    write_bits: int,
    source_actor_vision_radius: int,
    target_actor_vision_radius: int,
    source_num_ants: int = 1,
    target_num_ants: int = 1,
    source_includes_agent_identity: bool = True,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    source_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=source_actor_vision_radius,
        num_ants=source_num_ants,
        include_agent_identity=source_includes_agent_identity,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=target_actor_vision_radius,
        num_ants=target_num_ants,
    )
    if old_weight.shape[0] != source_dim:
        raise ValueError(f"Expected actor input dim {source_dim}, got {old_weight.shape[0]}.")

    source_patch_size = actor_vision_patch_size(source_actor_vision_radius)
    target_patch_size = actor_vision_patch_size(target_actor_vision_radius)
    channel_count = write_bits + 4
    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    for channel_index in range(channel_count):
        source = slice(
            channel_index * source_patch_size,
            (channel_index + 1) * source_patch_size,
        )
        target = slice(
            channel_index * target_patch_size,
            (channel_index + 1) * target_patch_size,
        )
        new_weight = copy_centered_actor_patch_channel_between_radii(
            new_weight,
            old_weight,
            source=source,
            target=target,
            source_actor_vision_radius=source_actor_vision_radius,
            target_actor_vision_radius=target_actor_vision_radius,
        )

    source_tail_start = channel_count * source_patch_size
    target_tail_start = channel_count * target_patch_size
    source_identity_width = agent_identity_feature_count(
        source_num_ants,
        include_agent_identity=source_includes_agent_identity,
    )
    target_identity_width = agent_identity_feature_count(target_num_ants)
    if source_identity_width > 0 and target_identity_width > 0:
        shared_identity_width = min(source_identity_width, target_identity_width)
        new_weight = new_weight.at[
            target_tail_start : target_tail_start + shared_identity_width,
            :,
        ].set(
            old_weight[
                source_tail_start : source_tail_start + shared_identity_width,
                :,
            ]
        )
    tail_width = 1 + FACING_FEATURE_COUNT
    target_tail = slice(
        target_tail_start + target_identity_width,
        target_tail_start + target_identity_width + tail_width,
    )
    source_tail = slice(
        source_tail_start + source_identity_width,
        source_tail_start + source_identity_width + tail_width,
    )
    new_weight = new_weight.at[target_tail, :].set(old_weight[source_tail, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def copy_centered_actor_patch_channel_between_radii(
    new_weight: jnp.ndarray,
    old_weight: jnp.ndarray,
    *,
    source: slice,
    target: slice,
    source_actor_vision_radius: int,
    target_actor_vision_radius: int,
) -> jnp.ndarray:
    source_width = 2 * source_actor_vision_radius + 1
    target_width = 2 * target_actor_vision_radius + 1
    shared_radius = min(source_actor_vision_radius, target_actor_vision_radius)
    for offset_y in range(-shared_radius, shared_radius + 1):
        for offset_x in range(-shared_radius, shared_radius + 1):
            source_index = (offset_y + source_actor_vision_radius) * source_width
            source_index += offset_x + source_actor_vision_radius
            target_index = (offset_y + target_actor_vision_radius) * target_width
            target_index += offset_x + target_actor_vision_radius
            new_weight = new_weight.at[target.start + target_index, :].set(
                old_weight[source.start + source_index, :]
            )
    return new_weight


def copy_actor_patch_channel(
    new_weight: jnp.ndarray,
    old_weight: jnp.ndarray,
    *,
    source: slice,
    target: slice,
    actor_vision_radius: int,
    source_layout: str,
) -> jnp.ndarray:
    if source_layout == "centered":
        return new_weight.at[target, :].set(old_weight[source, :])
    target_width = 2 * actor_vision_radius + 1
    source_index = 0
    first_depth = 0 if source_layout == "forward_current_row" else 1
    for depth in range(first_depth, actor_vision_radius + 1):
        for lateral in range(
            -(DEFAULT_ACTOR_VISION_WIDTH // 2),
            DEFAULT_ACTOR_VISION_WIDTH // 2 + 1,
        ):
            target_index = (lateral + actor_vision_radius) * target_width
            target_index += depth + actor_vision_radius
            new_weight = new_weight.at[target.start + target_index, :].set(
                old_weight[source.start + source_index, :]
            )
            source_index += 1
    return new_weight


def expand_write_head(
    layer: LinearParams,
    *,
    old_bits: int,
    target_bits: int,
    transfer_mode: str = "repeat",
) -> LinearParams:
    transfer_mode = validate_write_head_transfer(transfer_mode)
    old_weight = jnp.asarray(layer.weight)
    old_bias = jnp.asarray(layer.bias)
    old_count = write_value_count(old_bits)
    target_count = write_value_count(target_bits)
    if old_weight.shape[-1] != old_count:
        raise ValueError(f"Expected {old_count} old write logits, got {old_weight.shape[-1]}.")
    if old_bias.shape[0] != old_count:
        raise ValueError(f"Expected {old_count} old write biases, got {old_bias.shape[0]}.")
    if transfer_mode == "reset":
        return LinearParams(
            weight=jnp.zeros((old_weight.shape[0], target_count), dtype=old_weight.dtype),
            bias=jnp.zeros((target_count,), dtype=old_bias.dtype),
        )
    if transfer_mode == "neutral-new":
        old_columns = jnp.arange(old_count)
        new_weight = jnp.zeros((old_weight.shape[0], target_count), dtype=old_weight.dtype)
        new_bias = jnp.zeros((target_count,), dtype=old_bias.dtype)
        new_weight = new_weight.at[:, old_columns].set(old_weight)
        new_bias = new_bias.at[old_columns].set(old_bias)
        if target_count > old_count:
            new_columns = jnp.arange(old_count, target_count)
            mean_weight = jnp.mean(old_weight, axis=1, keepdims=True)
            mean_bias = jnp.mean(old_bias)
            new_weight = new_weight.at[:, new_columns].set(
                jnp.repeat(mean_weight, target_count - old_count, axis=1)
            )
            new_bias = new_bias.at[new_columns].set(mean_bias)
        return LinearParams(weight=new_weight, bias=new_bias)

    source_indices = jnp.asarray(repeated_write_action_indices(old_bits, target_bits))
    if source_indices.shape[0] != target_count:
        raise ValueError("write action transfer index count does not match target bits.")
    return LinearParams(weight=old_weight[:, source_indices], bias=old_bias[source_indices])


__all__ = [
    "WRITE_HEAD_TRANSFER_MODES",
    "adapt_movement_head",
    "adapt_movement_head_layer",
    "copy_actor_patch_channel",
    "copy_centered_actor_patch_channel_between_radii",
    "expand_actor_input_layer",
    "expand_params_for_write_bits",
    "expand_write_head",
    "resize_actor_input_layer_for_vision_radius",
    "resize_params_for_actor_vision_radius",
]
