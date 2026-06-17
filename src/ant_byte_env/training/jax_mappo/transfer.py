"""Current-format JAX checkpoint transfer helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ant_byte_env import (
    DEFAULT_ACTOR_VISION_WIDTH,
    DEFAULT_WRITE_BITS,
    MAX_WRITE_BITS,
    MOVEMENT_ACTION_COUNT,
    actor_vision_patch_size,
    write_value_count,
)
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.core import (
    JaxMAPPOParams,
    LinearParams,
    init_adam_state,
)

WRITE_HEAD_TRANSFER_MODES = ("repeat", "reset", "neutral-new")
FACING_FEATURE_COUNT = MOVEMENT_ACTION_COUNT - 1


def load_checkpoint_for_training(
    path: Path,
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
    target_write_bits: int,
    actor_vision_radius: int,
    write_head_transfer: str = "repeat",
) -> dict[str, Any]:
    write_head_transfer = validate_write_head_transfer(write_head_transfer)
    checkpoint = read_checkpoint(path)
    source_args = checkpoint.get("args", {})
    params = checkpoint["params"]
    params, params_changed = adapt_movement_head(params)
    source_write_bits = int(source_args.get("write_bits", DEFAULT_WRITE_BITS))
    if target_write_bits < source_write_bits:
        raise ValueError("Checkpoint actor observation dimension does not match this run.")
    if target_write_bits > MAX_WRITE_BITS:
        raise ValueError(f"target_write_bits must be at most {MAX_WRITE_BITS}.")
    if int(checkpoint["central_obs_dim"]) != central_obs_dim:
        params = expand_critic_input_for_ants_count(
            params,
            source_args=source_args,
            target_central_obs_dim=central_obs_dim,
        )
        params_changed = True
    if int(checkpoint["actor_obs_dim"]) == actor_obs_dim:
        if not params_changed:
            return checkpoint
        return {
            **checkpoint,
            "params": params,
            "opt_state": init_adam_state(params),
            "central_obs_dim": central_obs_dim,
            "actor_obs_dim": actor_obs_dim,
        }

    source_actor_vision_radius = int(source_args.get("actor_vision_radius", actor_vision_radius))
    if source_actor_vision_radius != actor_vision_radius:
        raise ValueError("Checkpoint actor vision radius does not match this run.")

    source_actor_obs_dim = int(checkpoint["actor_obs_dim"])
    source_shape = _actor_obs_source_shape(
        actor_obs_dim=source_actor_obs_dim,
        write_bits=source_write_bits,
        actor_vision_radius=actor_vision_radius,
    )
    if source_shape is None:
        raise ValueError("Checkpoint actor observation dimension does not match its write-bit config.")

    params = expand_params_for_write_bits(
        params,
        old_bits=source_write_bits,
        target_bits=target_write_bits,
        actor_vision_radius=actor_vision_radius,
        source_includes_ants_count=source_shape["include_ants_count"],
        source_includes_orientation=source_shape["include_orientation"],
        source_layout=source_shape["layout"],
        write_head_transfer=write_head_transfer,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_write_bits,
        actor_vision_radius=actor_vision_radius,
    )
    if target_dim != actor_obs_dim:
        raise ValueError("Transferred actor observation dimension does not match this run.")

    return {
        **checkpoint,
        "params": params,
        "opt_state": init_adam_state(params),
        "central_obs_dim": central_obs_dim,
        "actor_obs_dim": actor_obs_dim,
        "args": {
            **source_args,
            "write_bits": target_write_bits,
            "transfer_source_checkpoint": str(path),
            "write_head_transfer": write_head_transfer,
        },
    }


def validate_write_head_transfer(mode: str) -> str:
    if mode not in WRITE_HEAD_TRANSFER_MODES:
        choices = ", ".join(WRITE_HEAD_TRANSFER_MODES)
        raise ValueError(f"write_head_transfer must be one of: {choices}.")
    return mode


def actor_obs_dim_for_bits(
    *,
    write_bits: int,
    actor_vision_radius: int,
    include_ants_count: bool = True,
    include_orientation: bool = True,
    include_current_row: bool = True,
) -> int:
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    patch_size = actor_vision_patch_size(actor_vision_radius)
    if not include_current_row:
        patch_size = DEFAULT_ACTOR_VISION_WIDTH * actor_vision_radius
    grid_channels = write_bits + (4 if include_ants_count else 3)
    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
    return patch_size * grid_channels + 1 + orientation_features


def source_actor_patch_size(*, actor_vision_radius: int, source_layout: str) -> int:
    if source_layout == "centered":
        return actor_vision_patch_size(actor_vision_radius)
    if source_layout == "forward_current_row":
        return DEFAULT_ACTOR_VISION_WIDTH * (actor_vision_radius + 1)
    if source_layout == "forward_only":
        return DEFAULT_ACTOR_VISION_WIDTH * actor_vision_radius
    raise ValueError(f"Unsupported actor window layout: {source_layout}.")


def source_actor_obs_dim(
    *,
    write_bits: int,
    actor_vision_radius: int,
    include_ants_count: bool,
    include_orientation: bool,
    source_layout: str,
) -> int:
    patch_size = source_actor_patch_size(
        actor_vision_radius=actor_vision_radius,
        source_layout=source_layout,
    )
    grid_channels = write_bits + (4 if include_ants_count else 3)
    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
    return patch_size * grid_channels + 1 + orientation_features


def _actor_obs_source_shape(
    *,
    actor_obs_dim: int,
    write_bits: int,
    actor_vision_radius: int,
) -> dict[str, bool | str] | None:
    source_layouts = (
        ("centered", actor_vision_patch_size(actor_vision_radius), True),
        (
            "forward_current_row",
            DEFAULT_ACTOR_VISION_WIDTH * (actor_vision_radius + 1),
            True,
        ),
        ("forward_only", DEFAULT_ACTOR_VISION_WIDTH * actor_vision_radius, False),
    )
    for include_ants_count in (True, False):
        for include_orientation in (True, False):
            for layout, patch_size, include_current_row in source_layouts:
                grid_channels = write_bits + (4 if include_ants_count else 3)
                orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
                expected_dim = patch_size * grid_channels + 1 + orientation_features
                if actor_obs_dim == expected_dim:
                    return {
                        "include_ants_count": include_ants_count,
                        "include_orientation": include_orientation,
                        "include_current_row": include_current_row,
                        "layout": layout,
                    }
    return None


def central_obs_dim_with_ants_count(
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
    include_orientation: bool = True,
) -> int:
    grid_area = obs_height * obs_width
    orientation_features = FACING_FEATURE_COUNT * num_ants if include_orientation else 0
    return 3 * num_ants + orientation_features + 3 * grid_area + 4


def legacy_central_obs_dim(*, num_ants: int, obs_height: int, obs_width: int) -> int:
    grid_area = obs_height * obs_width
    return 3 * num_ants + 2 * grid_area + 4


def expand_critic_input_for_ants_count(
    params: JaxMAPPOParams,
    *,
    source_args: dict[str, Any],
    target_central_obs_dim: int,
) -> JaxMAPPOParams:
    source_num_ants = int(source_args.get("num_ants", 1))
    source_width = _source_grid_size(source_args, "width")
    source_height = _source_grid_size(source_args, "height")
    legacy_dim = legacy_central_obs_dim(
        num_ants=source_num_ants,
        obs_height=source_height,
        obs_width=source_width,
    )
    no_orientation_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=source_height,
        obs_width=source_width,
        include_orientation=False,
    )
    current_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=source_height,
        obs_width=source_width,
    )
    if current_dim != target_central_obs_dim:
        raise ValueError("Checkpoint central observation dimension does not match this run.")

    first_layer = params.critic_body[0]
    old_weight = jnp.asarray(first_layer.weight)
    if old_weight.shape[0] == legacy_dim:
        new_first_layer = expand_central_input_layer_for_ants_count(
            first_layer,
            num_ants=source_num_ants,
            obs_height=source_height,
            obs_width=source_width,
        )
    elif old_weight.shape[0] == no_orientation_dim:
        new_first_layer = expand_central_input_layer_for_orientation(
            first_layer,
            num_ants=source_num_ants,
            obs_height=source_height,
            obs_width=source_width,
        )
    else:
        raise ValueError("Checkpoint central observation dimension does not match this run.")

    return JaxMAPPOParams(
        actor_body=params.actor_body,
        move_head=adapt_movement_head_layer(params.move_head),
        write_head=params.write_head,
        critic_body=(new_first_layer, params.critic_body[1]),
        value_head=params.value_head,
    )


def expand_central_input_layer_for_ants_count(
    layer: LinearParams,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    grid_area = obs_height * obs_width
    legacy_dim = legacy_central_obs_dim(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    current_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if old_weight.shape[0] != legacy_dim:
        raise ValueError(f"Expected central input dim {legacy_dim}, got {old_weight.shape[0]}.")

    prefix_dim = 3 * num_ants
    orientation_dim = FACING_FEATURE_COUNT * num_ants
    new_prefix_dim = prefix_dim + orientation_dim
    old_food = slice(prefix_dim, prefix_dim + grid_area)
    old_bytes = slice(prefix_dim + grid_area, prefix_dim + 2 * grid_area)
    old_tail = slice(prefix_dim + 2 * grid_area, legacy_dim)
    new_food = slice(new_prefix_dim + grid_area, new_prefix_dim + 2 * grid_area)
    new_bytes = slice(new_prefix_dim + 2 * grid_area, new_prefix_dim + 3 * grid_area)
    new_tail = slice(new_prefix_dim + 3 * grid_area, current_dim)

    new_weight = jnp.zeros((current_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    new_weight = new_weight.at[:prefix_dim, :].set(old_weight[:prefix_dim, :])
    new_weight = new_weight.at[new_food, :].set(old_weight[old_food, :])
    new_weight = new_weight.at[new_bytes, :].set(old_weight[old_bytes, :])
    new_weight = new_weight.at[new_tail, :].set(old_weight[old_tail, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def expand_central_input_layer_for_orientation(
    layer: LinearParams,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    grid_area = obs_height * obs_width
    no_orientation_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
        include_orientation=False,
    )
    current_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if old_weight.shape[0] != no_orientation_dim:
        raise ValueError(
            f"Expected central input dim {no_orientation_dim}, got {old_weight.shape[0]}."
        )

    prefix_dim = 3 * num_ants
    orientation_dim = FACING_FEATURE_COUNT * num_ants
    old_maps_and_tail = slice(prefix_dim, no_orientation_dim)
    new_maps_and_tail = slice(prefix_dim + orientation_dim, current_dim)

    new_weight = jnp.zeros((current_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    new_weight = new_weight.at[:prefix_dim, :].set(old_weight[:prefix_dim, :])
    new_weight = new_weight.at[new_maps_and_tail, :].set(old_weight[old_maps_and_tail, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def _source_grid_size(source_args: dict[str, Any], axis: str) -> int:
    obs_name = f"obs_{axis}"
    if source_args.get(obs_name) is not None:
        return int(source_args[obs_name])
    return int(source_args[axis])


def expand_params_for_write_bits(
    params: JaxMAPPOParams,
    *,
    old_bits: int,
    target_bits: int,
    actor_vision_radius: int,
    source_includes_ants_count: bool = True,
    source_includes_orientation: bool = False,
    source_layout: str = "centered",
    write_head_transfer: str = "repeat",
) -> JaxMAPPOParams:
    write_head_transfer = validate_write_head_transfer(write_head_transfer)
    if (
        target_bits == old_bits
        and source_includes_ants_count
        and source_includes_orientation
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
                source_layout=source_layout,
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
    old_weight = jnp.asarray(layer.weight)
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
    source_layout: str = "centered",
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
        include_ants_count=source_includes_ants_count,
        include_orientation=source_includes_orientation,
        source_layout=source_layout,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=actor_vision_radius,
    )
    if old_weight.shape[0] != expected_old_dim:
        raise ValueError(f"Expected actor input dim {expected_old_dim}, got {old_weight.shape[0]}.")

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
    old_carrying = slice(old_tail_start, old_tail_start + 1)
    old_orientation = (
        slice(old_tail_start + 1, old_tail_start + 1 + FACING_FEATURE_COUNT)
        if source_includes_orientation
        else None
    )
    new_tail_start = target_patch_size * (target_bits + 4)
    new_carrying = slice(new_tail_start, new_tail_start + 1)
    new_orientation = slice(
        new_tail_start + 1,
        new_tail_start + 1 + FACING_FEATURE_COUNT,
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
    new_weight = new_weight.at[new_carrying, :].set(old_weight[old_carrying, :])
    if old_orientation is not None:
        new_weight = new_weight.at[new_orientation, :].set(old_weight[old_orientation, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


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


def repeated_write_action_indices(old_bits: int, target_bits: int) -> np.ndarray:
    if old_bits <= 0:
        raise ValueError("old_bits must be positive.")
    if target_bits < old_bits:
        raise ValueError("target_bits must be at least old_bits.")
    old_count = write_value_count(old_bits)
    target_count = write_value_count(target_bits)
    return np.arange(target_count, dtype=np.int64) % old_count
