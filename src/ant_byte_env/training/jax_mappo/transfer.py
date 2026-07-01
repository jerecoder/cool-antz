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
    ConvParams,
    JaxMAPPOParams,
    LinearParams,
    ResNetCriticParams,
    StridedCNNCriticParams,
    StructuredMLPCriticParams,
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
    target_num_ants: int = 1,
    write_head_transfer: str = "repeat",
    target_critic_architecture: str = "mlp",
) -> dict[str, Any]:
    write_head_transfer = validate_write_head_transfer(write_head_transfer)
    checkpoint = read_checkpoint(path)
    target_args = _checkpoint_args_mapping(checkpoint.get("args", {}))
    source_args = _checkpoint_args_mapping(
        checkpoint.get("transfer_source_args", target_args)
    )
    params = checkpoint["params"]
    params, params_changed = adapt_movement_head(params)
    source_critic_architecture = str(source_args.get("critic_architecture", "mlp"))
    target_critic_architecture = str(target_critic_architecture)
    if source_critic_architecture != target_critic_architecture:
        raise ValueError(
            "Checkpoint critic architecture does not match this run "
            f"({source_critic_architecture!r} != {target_critic_architecture!r})."
        )
    source_write_bits = int(source_args.get("write_bits", DEFAULT_WRITE_BITS))
    if target_write_bits < source_write_bits:
        raise ValueError("Checkpoint actor observation dimension does not match this run.")
    if target_write_bits > MAX_WRITE_BITS:
        raise ValueError(f"target_write_bits must be at most {MAX_WRITE_BITS}.")
    if int(checkpoint["central_obs_dim"]) != central_obs_dim:
        if target_critic_architecture == "mlp":
            params = expand_critic_input_for_ants_count(
                params,
                source_args=source_args,
                target_central_obs_dim=central_obs_dim,
                target_num_ants=target_num_ants,
            )
        else:
            params = resize_non_mlp_critic_for_ants_count(
                params,
                source_args=source_args,
                target_central_obs_dim=central_obs_dim,
                target_num_ants=target_num_ants,
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

    source_actor_obs_dim = int(checkpoint["actor_obs_dim"])
    source_actor_vision_radius = int(source_args.get("actor_vision_radius", actor_vision_radius))
    source_num_ants = int(source_args.get("num_ants", 1))
    source_shape = _actor_obs_source_shape(
        actor_obs_dim=source_actor_obs_dim,
        write_bits=source_write_bits,
        actor_vision_radius=source_actor_vision_radius,
        num_ants=source_num_ants,
    )
    if source_shape is None:
        raise ValueError(
            "Checkpoint actor observation dimension does not match its write-bit config."
        )

    params = expand_params_for_write_bits(
        params,
        old_bits=source_write_bits,
        target_bits=target_write_bits,
        actor_vision_radius=source_actor_vision_radius,
        source_includes_ants_count=source_shape["include_ants_count"],
        source_includes_dead_ants=source_shape["include_dead_ants"],
        source_includes_orientation=source_shape["include_orientation"],
        source_includes_agent_identity=source_shape["include_agent_identity"],
        source_layout=source_shape["layout"],
        source_num_ants=source_num_ants,
        target_num_ants=target_num_ants,
        target_includes_dead_ants=_actor_obs_includes_dead_ants(
            actor_obs_dim,
            write_bits=target_write_bits,
            actor_vision_radius=actor_vision_radius,
            num_ants=target_num_ants,
        ),
        write_head_transfer=write_head_transfer,
    )
    params = resize_params_for_actor_vision_radius(
        params,
        write_bits=target_write_bits,
        source_actor_vision_radius=source_actor_vision_radius,
        target_actor_vision_radius=actor_vision_radius,
        source_num_ants=target_num_ants,
        target_num_ants=target_num_ants,
        source_includes_agent_identity=True,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_write_bits,
        actor_vision_radius=actor_vision_radius,
        num_ants=target_num_ants,
        include_dead_ants=_actor_obs_includes_dead_ants(
            actor_obs_dim,
            write_bits=target_write_bits,
            actor_vision_radius=actor_vision_radius,
            num_ants=target_num_ants,
        ),
    )
    if target_dim + actor_vision_patch_size(actor_vision_radius) == actor_obs_dim:
        params = insert_dead_ant_actor_channel(
            params,
            actor_vision_radius=actor_vision_radius,
        )
        target_dim = actor_obs_dim
    if target_dim != actor_obs_dim:
        raise ValueError("Transferred actor observation dimension does not match this run.")

    return {
        **checkpoint,
        "params": params,
        "opt_state": init_adam_state(params),
        "central_obs_dim": central_obs_dim,
        "actor_obs_dim": actor_obs_dim,
        "args": {
            **target_args,
            "write_bits": target_write_bits,
            "actor_vision_radius": actor_vision_radius,
            "num_ants": target_num_ants,
            "transfer_source_checkpoint": str(path),
            "write_head_transfer": write_head_transfer,
        },
    }


def _checkpoint_args_mapping(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return dict(args)
    return dict(vars(args))


def _actor_obs_includes_dead_ants(
    actor_obs_dim: int,
    *,
    write_bits: int,
    actor_vision_radius: int,
    num_ants: int,
) -> bool:
    base_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=actor_vision_radius,
        num_ants=num_ants,
    )
    dead_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=actor_vision_radius,
        num_ants=num_ants,
        include_dead_ants=True,
    )
    if int(actor_obs_dim) == dead_dim:
        return True
    if int(actor_obs_dim) == base_dim:
        return False
    return False


def validate_write_head_transfer(mode: str) -> str:
    if mode not in WRITE_HEAD_TRANSFER_MODES:
        choices = ", ".join(WRITE_HEAD_TRANSFER_MODES)
        raise ValueError(f"write_head_transfer must be one of: {choices}.")
    return mode


def actor_obs_dim_for_bits(
    *,
    write_bits: int,
    actor_vision_radius: int,
    num_ants: int = 1,
    include_ants_count: bool = True,
    include_dead_ants: bool = False,
    include_orientation: bool = True,
    include_agent_identity: bool = True,
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
    if include_dead_ants:
        grid_channels += 1
    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
    identity_features = agent_identity_feature_count(
        num_ants,
        include_agent_identity=include_agent_identity,
    )
    return patch_size * grid_channels + identity_features + 1 + orientation_features


def agent_identity_feature_count(
    num_ants: int,
    *,
    include_agent_identity: bool = True,
) -> int:
    if not include_agent_identity:
        return 0
    count = int(num_ants)
    return count if count > 1 else 0


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
    num_ants: int,
    include_ants_count: bool,
    include_dead_ants: bool,
    include_orientation: bool,
    include_agent_identity: bool,
    source_layout: str,
) -> int:
    patch_size = source_actor_patch_size(
        actor_vision_radius=actor_vision_radius,
        source_layout=source_layout,
    )
    grid_channels = write_bits + (4 if include_ants_count else 3)
    if include_dead_ants:
        grid_channels += 1
    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
    identity_features = agent_identity_feature_count(
        num_ants,
        include_agent_identity=include_agent_identity,
    )
    return patch_size * grid_channels + identity_features + 1 + orientation_features


def _actor_obs_source_shape(
    *,
    actor_obs_dim: int,
    write_bits: int,
    actor_vision_radius: int,
    num_ants: int = 1,
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
        for include_dead_ants in (True, False):
            for include_orientation in (True, False):
                for include_agent_identity in (True, False):
                    for layout, patch_size, include_current_row in source_layouts:
                        grid_channels = write_bits + (4 if include_ants_count else 3)
                        if include_dead_ants:
                            grid_channels += 1
                        orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
                        identity_features = agent_identity_feature_count(
                            num_ants,
                            include_agent_identity=include_agent_identity,
                        )
                        expected_dim = (
                            patch_size * grid_channels
                            + identity_features
                            + 1
                            + orientation_features
                        )
                        if actor_obs_dim == expected_dim:
                            return {
                                "include_ants_count": include_ants_count,
                                "include_dead_ants": include_dead_ants,
                                "include_orientation": include_orientation,
                                "include_agent_identity": include_agent_identity,
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
    include_dead_ants: bool = False,
) -> int:
    grid_area = obs_height * obs_width
    orientation_features = FACING_FEATURE_COUNT * num_ants if include_orientation else 0
    grid_plane_count = 4 if include_dead_ants else 3
    return 3 * num_ants + orientation_features + grid_plane_count * grid_area + 4


def _central_dim_includes_dead_ants(
    central_obs_dim: int,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> bool:
    if int(central_obs_dim) == central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
        include_dead_ants=False,
    ):
        return False
    if int(central_obs_dim) == central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
        include_dead_ants=True,
    ):
        return True
    raise ValueError("Checkpoint central observation dimension does not match this run.")


def _critic_grid_map_names(*, include_dead_ants: bool) -> tuple[str, ...]:
    if include_dead_ants:
        return ("ants_count", "dead_ants_count", "food", "bytes")
    return ("ants_count", "food", "bytes")


def legacy_central_obs_dim(*, num_ants: int, obs_height: int, obs_width: int) -> int:
    grid_area = obs_height * obs_width
    return 3 * num_ants + 2 * grid_area + 4


def expand_critic_input_for_ants_count(
    params: JaxMAPPOParams,
    *,
    source_args: dict[str, Any],
    target_central_obs_dim: int,
    target_num_ants: int,
) -> JaxMAPPOParams:
    source_num_ants = int(source_args.get("num_ants", 1))
    source_width = _source_grid_size(source_args, "width")
    source_height = _source_grid_size(source_args, "height")
    target_include_dead_ants = _central_dim_includes_dead_ants(
        target_central_obs_dim,
        num_ants=target_num_ants,
        obs_height=source_height,
        obs_width=source_width,
    )
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
    current_dead_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=source_height,
        obs_width=source_width,
        include_dead_ants=True,
    )

    first_layer = params.critic_body[0]
    old_weight = jnp.asarray(first_layer.weight)
    if old_weight.shape[0] == legacy_dim:
        current_first_layer = expand_central_input_layer_for_ants_count(
            first_layer,
            num_ants=source_num_ants,
            obs_height=source_height,
            obs_width=source_width,
        )
        source_include_dead_ants = False
    elif old_weight.shape[0] == no_orientation_dim:
        current_first_layer = expand_central_input_layer_for_orientation(
            first_layer,
            num_ants=source_num_ants,
            obs_height=source_height,
            obs_width=source_width,
        )
        source_include_dead_ants = False
    elif old_weight.shape[0] == current_dim:
        current_first_layer = first_layer
        source_include_dead_ants = False
    elif old_weight.shape[0] == current_dead_dim:
        current_first_layer = first_layer
        source_include_dead_ants = True
    else:
        raise ValueError("Checkpoint central observation dimension does not match this run.")

    new_first_layer = resize_central_input_layer_for_num_ants(
        current_first_layer,
        source_num_ants=source_num_ants,
        target_num_ants=target_num_ants,
        obs_height=source_height,
        obs_width=source_width,
        source_include_dead_ants=source_include_dead_ants,
        target_include_dead_ants=target_include_dead_ants,
    )

    return JaxMAPPOParams(
        actor_body=params.actor_body,
        move_head=adapt_movement_head_layer(params.move_head),
        write_head=params.write_head,
        critic_body=(new_first_layer, params.critic_body[1]),
        value_head=params.value_head,
    )


def _infer_num_ants_for_current_central_dim(
    central_obs_dim: int,
    *,
    obs_height: int,
    obs_width: int,
) -> int:
    grid_area = obs_height * obs_width
    candidates = []
    for include_dead_ants in (False, True):
        grid_plane_count = 4 if include_dead_ants else 3
        non_ant_dim = grid_plane_count * grid_area + 4
        ant_dim = int(central_obs_dim) - non_ant_dim
        if ant_dim > 0 and ant_dim % 7 == 0:
            candidates.append(ant_dim // 7)
    if not candidates:
        raise ValueError("Checkpoint central observation dimension does not match this run.")
    return candidates[0]


def resize_central_input_layer_for_num_ants(
    layer: LinearParams,
    *,
    source_num_ants: int,
    target_num_ants: int,
    obs_height: int,
    obs_width: int,
    source_include_dead_ants: bool = False,
    target_include_dead_ants: bool = False,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    source_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
        include_dead_ants=source_include_dead_ants,
    )
    target_dim = central_obs_dim_with_ants_count(
        num_ants=target_num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
        include_dead_ants=target_include_dead_ants,
    )
    if old_weight.shape[0] != source_dim:
        raise ValueError(f"Expected central input dim {source_dim}, got {old_weight.shape[0]}.")
    if source_dim == target_dim:
        return layer

    grid_area = obs_height * obs_width
    source_prefix_dim = 3 * source_num_ants
    source_orientation_dim = FACING_FEATURE_COUNT * source_num_ants
    source_maps_start = source_prefix_dim + source_orientation_dim
    source_map_names = _critic_grid_map_names(include_dead_ants=source_include_dead_ants)
    source_tail = slice(source_maps_start + len(source_map_names) * grid_area, source_dim)

    target_prefix_dim = 3 * target_num_ants
    target_orientation_dim = FACING_FEATURE_COUNT * target_num_ants
    target_maps_start = target_prefix_dim + target_orientation_dim
    target_map_names = _critic_grid_map_names(include_dead_ants=target_include_dead_ants)
    target_tail = slice(target_maps_start + len(target_map_names) * grid_area, target_dim)

    shared_ants = min(source_num_ants, target_num_ants)
    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    new_weight = new_weight.at[: 2 * shared_ants, :].set(
        old_weight[: 2 * shared_ants, :]
    )
    new_weight = new_weight.at[
        2 * target_num_ants : 2 * target_num_ants + shared_ants,
        :,
    ].set(
        old_weight[
            2 * source_num_ants : 2 * source_num_ants + shared_ants,
            :,
        ]
    )
    new_weight = new_weight.at[
        target_prefix_dim : target_prefix_dim + FACING_FEATURE_COUNT * shared_ants,
        :,
    ].set(
        old_weight[
            source_prefix_dim : source_prefix_dim + FACING_FEATURE_COUNT * shared_ants,
            :,
        ]
    )
    for target_map_index, map_name in enumerate(target_map_names):
        if map_name not in source_map_names:
            continue
        source_map_index = source_map_names.index(map_name)
        source_map = slice(
            source_maps_start + source_map_index * grid_area,
            source_maps_start + (source_map_index + 1) * grid_area,
        )
        target_map = slice(
            target_maps_start + target_map_index * grid_area,
            target_maps_start + (target_map_index + 1) * grid_area,
        )
        new_weight = new_weight.at[target_map, :].set(old_weight[source_map, :])
    new_weight = new_weight.at[target_tail, :].set(old_weight[source_tail, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def resize_critic_entity_input_layer_for_num_ants(
    layer: LinearParams,
    *,
    source_num_ants: int,
    target_num_ants: int,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    source_dim = 7 * int(source_num_ants) + 4
    target_dim = 7 * int(target_num_ants) + 4
    if old_weight.shape[0] != source_dim:
        raise ValueError(f"Expected critic entity input dim {source_dim}, got {old_weight.shape[0]}.")
    if source_dim == target_dim:
        return layer

    shared_ants = min(int(source_num_ants), int(target_num_ants))
    source_pos_end = 2 * int(source_num_ants)
    source_carrying_end = 3 * int(source_num_ants)
    source_facing_end = 7 * int(source_num_ants)
    target_pos_end = 2 * int(target_num_ants)
    target_carrying_end = 3 * int(target_num_ants)
    target_facing_end = 7 * int(target_num_ants)

    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    new_weight = new_weight.at[: 2 * shared_ants, :].set(
        old_weight[: 2 * shared_ants, :]
    )
    new_weight = new_weight.at[target_pos_end : target_pos_end + shared_ants, :].set(
        old_weight[source_pos_end : source_pos_end + shared_ants, :]
    )
    new_weight = new_weight.at[
        target_carrying_end : target_carrying_end + FACING_FEATURE_COUNT * shared_ants,
        :,
    ].set(
        old_weight[
            source_carrying_end : source_carrying_end + FACING_FEATURE_COUNT * shared_ants,
            :,
        ]
    )
    new_weight = new_weight.at[target_facing_end:, :].set(old_weight[source_facing_end:, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def _spatial_conv_includes_dead_ants(layer: ConvParams) -> bool:
    input_channels = int(jnp.asarray(layer.kernel).shape[2])
    if input_channels == 4:
        return False
    if input_channels == 5:
        return True
    raise ValueError(
        "Non-MLP critic checkpoint central observation dimension does not match this run."
    )


def resize_spatial_conv_input_for_dead_ants(
    layer: ConvParams,
    *,
    source_include_dead_ants: bool,
    target_include_dead_ants: bool,
) -> ConvParams:
    source_channels = (*_critic_grid_map_names(include_dead_ants=source_include_dead_ants), "hub")
    target_channels = (*_critic_grid_map_names(include_dead_ants=target_include_dead_ants), "hub")
    old_kernel = jnp.asarray(layer.kernel)
    expected_source_channels = len(source_channels)
    if old_kernel.shape[2] != expected_source_channels:
        raise ValueError(
            f"Expected critic spatial input channels {expected_source_channels}, "
            f"got {old_kernel.shape[2]}."
        )
    if source_channels == target_channels:
        return layer

    new_kernel = jnp.zeros(
        (
            old_kernel.shape[0],
            old_kernel.shape[1],
            len(target_channels),
            old_kernel.shape[3],
        ),
        dtype=old_kernel.dtype,
    )
    for target_channel_index, channel_name in enumerate(target_channels):
        if channel_name not in source_channels:
            continue
        source_channel_index = source_channels.index(channel_name)
        new_kernel = new_kernel.at[:, :, target_channel_index, :].set(
            old_kernel[:, :, source_channel_index, :]
        )
    return ConvParams(kernel=new_kernel, bias=jnp.asarray(layer.bias))


def _structured_grid_layer_includes_dead_ants(
    layer: LinearParams,
    *,
    obs_height: int,
    obs_width: int,
) -> bool:
    grid_area = int(obs_height) * int(obs_width)
    input_dim = int(jnp.asarray(layer.weight).shape[0])
    if input_dim == 3 * grid_area:
        return False
    if input_dim == 4 * grid_area:
        return True
    raise ValueError(
        "Non-MLP critic checkpoint central observation dimension does not match this run."
    )


def resize_structured_grid_input_for_dead_ants(
    layer: LinearParams,
    *,
    obs_height: int,
    obs_width: int,
    source_include_dead_ants: bool,
    target_include_dead_ants: bool,
) -> LinearParams:
    grid_area = int(obs_height) * int(obs_width)
    source_maps = _critic_grid_map_names(include_dead_ants=source_include_dead_ants)
    target_maps = _critic_grid_map_names(include_dead_ants=target_include_dead_ants)
    old_weight = jnp.asarray(layer.weight)
    if old_weight.shape[0] != len(source_maps) * grid_area:
        raise ValueError(
            f"Expected critic grid input dim {len(source_maps) * grid_area}, "
            f"got {old_weight.shape[0]}."
        )
    if source_maps == target_maps:
        return layer

    new_weight = jnp.zeros(
        (len(target_maps) * grid_area, old_weight.shape[1]),
        dtype=old_weight.dtype,
    )
    for target_map_index, map_name in enumerate(target_maps):
        if map_name not in source_maps:
            continue
        source_map_index = source_maps.index(map_name)
        source_map = slice(
            source_map_index * grid_area,
            (source_map_index + 1) * grid_area,
        )
        target_map = slice(
            target_map_index * grid_area,
            (target_map_index + 1) * grid_area,
        )
        new_weight = new_weight.at[target_map, :].set(old_weight[source_map, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def resize_non_mlp_critic_for_ants_count(
    params: JaxMAPPOParams,
    *,
    source_args: dict[str, Any],
    target_central_obs_dim: int,
    target_num_ants: int,
) -> JaxMAPPOParams:
    source_num_ants = int(source_args.get("num_ants", 1))
    source_width = _source_grid_size(source_args, "width")
    source_height = _source_grid_size(source_args, "height")
    target_include_dead_ants = _central_dim_includes_dead_ants(
        target_central_obs_dim,
        num_ants=target_num_ants,
        obs_height=source_height,
        obs_width=source_width,
    )
    critic_body = params.critic_body
    if isinstance(critic_body, StridedCNNCriticParams):
        source_include_dead_ants = _spatial_conv_includes_dead_ants(
            critic_body.conv_5x5
        )
        resized_body = critic_body._replace(
            conv_5x5=resize_spatial_conv_input_for_dead_ants(
                critic_body.conv_5x5,
                source_include_dead_ants=source_include_dead_ants,
                target_include_dead_ants=target_include_dead_ants,
            ),
            entity_dense=resize_critic_entity_input_layer_for_num_ants(
                critic_body.entity_dense,
                source_num_ants=source_num_ants,
                target_num_ants=target_num_ants,
            )
        )
    elif isinstance(critic_body, ResNetCriticParams):
        source_include_dead_ants = _spatial_conv_includes_dead_ants(critic_body.stem)
        resized_first = resize_critic_entity_input_layer_for_num_ants(
            critic_body.entity_body[0],
            source_num_ants=source_num_ants,
            target_num_ants=target_num_ants,
        )
        resized_body = critic_body._replace(
            stem=resize_spatial_conv_input_for_dead_ants(
                critic_body.stem,
                source_include_dead_ants=source_include_dead_ants,
                target_include_dead_ants=target_include_dead_ants,
            ),
            entity_body=(resized_first, critic_body.entity_body[1])
        )
    elif isinstance(critic_body, StructuredMLPCriticParams):
        source_include_dead_ants = _structured_grid_layer_includes_dead_ants(
            critic_body.grid_body[0],
            obs_height=source_height,
            obs_width=source_width,
        )
        resized_grid_first = resize_structured_grid_input_for_dead_ants(
            critic_body.grid_body[0],
            obs_height=source_height,
            obs_width=source_width,
            source_include_dead_ants=source_include_dead_ants,
            target_include_dead_ants=target_include_dead_ants,
        )
        resized_first = resize_critic_entity_input_layer_for_num_ants(
            critic_body.entity_body[0],
            source_num_ants=source_num_ants,
            target_num_ants=target_num_ants,
        )
        resized_body = critic_body._replace(
            grid_body=(resized_grid_first, critic_body.grid_body[1]),
            entity_body=(resized_first, critic_body.entity_body[1])
        )
    else:
        raise ValueError(
            "Non-MLP critic checkpoint central observation dimension does not match this run."
        )
    return JaxMAPPOParams(
        actor_body=params.actor_body,
        move_head=adapt_movement_head_layer(params.move_head),
        write_head=params.write_head,
        critic_body=resized_body,
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
    source_includes_dead_ants: bool = False,
    source_includes_orientation: bool = False,
    source_includes_agent_identity: bool = False,
    source_layout: str = "centered",
    source_num_ants: int = 1,
    target_num_ants: int = 1,
    target_includes_dead_ants: bool = False,
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
        and source_includes_dead_ants == target_includes_dead_ants
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
                source_includes_dead_ants=source_includes_dead_ants,
                source_includes_orientation=source_includes_orientation,
                source_includes_agent_identity=source_includes_agent_identity,
                source_layout=source_layout,
                source_num_ants=source_num_ants,
                target_num_ants=target_num_ants,
                target_includes_dead_ants=target_includes_dead_ants,
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


def insert_dead_ant_actor_channel(
    params: JaxMAPPOParams,
    *,
    actor_vision_radius: int,
) -> JaxMAPPOParams:
    first_layer = params.actor_body[0]
    old_weight = jnp.asarray(first_layer.weight)
    patch_size = actor_vision_patch_size(actor_vision_radius)
    insert_at = 2 * patch_size
    zero_rows = jnp.zeros((patch_size, old_weight.shape[1]), dtype=old_weight.dtype)
    new_weight = jnp.concatenate(
        [
            old_weight[:insert_at],
            zero_rows,
            old_weight[insert_at:],
        ],
        axis=0,
    )
    return JaxMAPPOParams(
        actor_body=(
            LinearParams(weight=new_weight, bias=jnp.asarray(first_layer.bias)),
            params.actor_body[1],
        ),
        move_head=params.move_head,
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
    source_includes_dead_ants: bool = False,
    source_includes_orientation: bool = False,
    source_includes_agent_identity: bool = False,
    source_layout: str = "centered",
    source_num_ants: int = 1,
    target_num_ants: int = 1,
    target_includes_dead_ants: bool = False,
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
        include_dead_ants=source_includes_dead_ants,
        include_orientation=source_includes_orientation,
        include_agent_identity=source_includes_agent_identity,
        source_layout=source_layout,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=actor_vision_radius,
        num_ants=target_num_ants,
        include_dead_ants=target_includes_dead_ants,
    )
    if old_weight.shape[0] != expected_old_dim:
        raise ValueError(
            f"Expected actor input dim {expected_old_dim}, got {old_weight.shape[0]}."
        )

    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    old_food = slice(0, source_patch_size)
    if source_includes_ants_count:
        old_ants_count = slice(source_patch_size, 2 * source_patch_size)
        old_dead_ants_count = (
            slice(2 * source_patch_size, 3 * source_patch_size)
            if source_includes_dead_ants
            else None
        )
        old_bits_start = (2 + int(source_includes_dead_ants)) * source_patch_size
        old_hub = slice(
            source_patch_size * (2 + int(source_includes_dead_ants) + old_bits),
            source_patch_size * (3 + int(source_includes_dead_ants) + old_bits),
        )
        old_border = slice(
            source_patch_size * (3 + int(source_includes_dead_ants) + old_bits),
            source_patch_size * (4 + int(source_includes_dead_ants) + old_bits),
        )
    else:
        old_ants_count = None
        old_dead_ants_count = None
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
    new_dead_ants_count = (
        slice(2 * target_patch_size, 3 * target_patch_size)
        if target_includes_dead_ants
        else None
    )
    new_bits_start = (2 + int(target_includes_dead_ants)) * target_patch_size
    new_bits_slices = [
        slice(
            new_bits_start + bit_index * target_patch_size,
            new_bits_start + (bit_index + 1) * target_patch_size,
        )
        for bit_index in range(old_bits)
    ]
    new_hub = slice(
        target_patch_size * (2 + int(target_includes_dead_ants) + target_bits),
        target_patch_size * (3 + int(target_includes_dead_ants) + target_bits),
    )
    new_border = slice(
        target_patch_size * (3 + int(target_includes_dead_ants) + target_bits),
        target_patch_size * (4 + int(target_includes_dead_ants) + target_bits),
    )
    old_tail_start = source_patch_size * (
        old_bits
        + (4 if source_includes_ants_count else 3)
        + int(source_includes_dead_ants)
    )
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
    new_tail_start = target_patch_size * (
        target_bits + 4 + int(target_includes_dead_ants)
    )
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
    if old_dead_ants_count is not None and new_dead_ants_count is not None:
        new_weight = copy_actor_patch_channel(
            new_weight,
            old_weight,
            source=old_dead_ants_count,
            target=new_dead_ants_count,
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
    new_weight = new_weight.at[
        target_tail,
        :,
    ].set(
        old_weight[
            source_tail,
            :,
        ]
    )
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


def repeated_write_action_indices(old_bits: int, target_bits: int) -> np.ndarray:
    if old_bits <= 0:
        raise ValueError("old_bits must be positive.")
    if target_bits < old_bits:
        raise ValueError("target_bits must be at least old_bits.")
    old_count = write_value_count(old_bits)
    target_count = write_value_count(target_bits)
    return np.arange(target_count, dtype=np.int64) % old_count
