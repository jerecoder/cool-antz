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
    CRITIC_GLOBAL_FEATURE_DIM,
    JaxMAPPOParams,
    LinearParams,
    ResNetCriticParams,
    SetCNNCriticParams,
    StridedCNNCriticParams,
    StructuredMLPCriticParams,
    init_adam_state,
)

WRITE_HEAD_TRANSFER_MODES = ("repeat", "reset", "neutral-new")
FACING_FEATURE_COUNT = MOVEMENT_ACTION_COUNT - 1
LEGACY_GLOBAL_FEATURE_DIM = 4


def load_checkpoint_for_training(
    path: Path,
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
    target_write_bits: int,
    actor_vision_radius: int,
    target_num_ants: int = 1,
    target_agent_identity_types: int | None = None,
    write_head_transfer: str = "repeat",
    target_critic_architecture: str = "mlp",
    reset_optimizer: bool = False,
) -> dict[str, Any]:
    write_head_transfer = validate_write_head_transfer(write_head_transfer)
    checkpoint = read_checkpoint(path)
    source_args = checkpoint.get("args", {})
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
    target_args = {
        **source_args,
        "write_bits": target_write_bits,
        "actor_vision_radius": actor_vision_radius,
        "num_ants": target_num_ants,
        "agent_identity_types": target_agent_identity_types,
        "transfer_source_checkpoint": str(path),
        "write_head_transfer": write_head_transfer,
    }
    if int(checkpoint["central_obs_dim"]) != central_obs_dim:
        if target_critic_architecture == "mlp":
            params = expand_critic_input_for_ants_count(
                params,
                source_args=source_args,
                target_central_obs_dim=central_obs_dim,
            )
        else:
            params = resize_non_mlp_critic_for_ants_count(
                params,
                source_args=source_args,
                target_central_obs_dim=central_obs_dim,
            )
        params_changed = True
    if int(checkpoint["actor_obs_dim"]) == actor_obs_dim:
        if not params_changed and not reset_optimizer:
            return checkpoint
        return {
            **checkpoint,
            "params": params,
            "opt_state": init_adam_state(params),
            "central_obs_dim": central_obs_dim,
            "actor_obs_dim": actor_obs_dim,
            "args": target_args,
        }

    source_actor_obs_dim = int(checkpoint["actor_obs_dim"])
    source_actor_vision_radius = int(source_args.get("actor_vision_radius", actor_vision_radius))
    source_num_ants = int(source_args.get("num_ants", 1))
    source_agent_identity_types = source_args.get("agent_identity_types")
    source_shape = _actor_obs_source_shape(
        actor_obs_dim=source_actor_obs_dim,
        write_bits=source_write_bits,
        actor_vision_radius=source_actor_vision_radius,
        num_ants=source_num_ants,
        agent_identity_types=source_agent_identity_types,
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
        source_includes_orientation=source_shape["include_orientation"],
        source_includes_agent_identity=source_shape["include_agent_identity"],
        source_layout=source_shape["layout"],
        source_num_ants=source_num_ants,
        source_agent_identity_types=source_agent_identity_types,
        target_num_ants=target_num_ants,
        target_agent_identity_types=target_agent_identity_types,
        write_head_transfer=write_head_transfer,
    )
    params = resize_params_for_actor_vision_radius(
        params,
        write_bits=target_write_bits,
        source_actor_vision_radius=source_actor_vision_radius,
        target_actor_vision_radius=actor_vision_radius,
        source_num_ants=target_num_ants,
        target_num_ants=target_num_ants,
        source_agent_identity_types=target_agent_identity_types,
        target_agent_identity_types=target_agent_identity_types,
        source_includes_agent_identity=True,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_write_bits,
        actor_vision_radius=actor_vision_radius,
        num_ants=target_num_ants,
        agent_identity_types=target_agent_identity_types,
    )
    if target_dim != actor_obs_dim:
        raise ValueError("Transferred actor observation dimension does not match this run.")

    return {
        **checkpoint,
        "params": params,
        "opt_state": init_adam_state(params),
        "central_obs_dim": central_obs_dim,
        "actor_obs_dim": actor_obs_dim,
        "args": target_args,
    }


def load_actor_from_checkpoint_for_training(
    path: Path,
    *,
    target_params: JaxMAPPOParams,
    central_obs_dim: int,
    actor_obs_dim: int,
    target_write_bits: int,
    actor_vision_radius: int,
    target_num_ants: int = 1,
    target_agent_identity_types: int | None = None,
    write_head_transfer: str = "repeat",
    target_critic_architecture: str = "mlp",
) -> dict[str, Any]:
    raw_checkpoint = read_checkpoint(path)
    source_args = raw_checkpoint.get("args", {})
    source_critic_architecture = str(source_args.get("critic_architecture", "mlp"))
    actor_checkpoint = load_checkpoint_for_training(
        path,
        central_obs_dim=int(raw_checkpoint["central_obs_dim"]),
        actor_obs_dim=actor_obs_dim,
        target_write_bits=target_write_bits,
        actor_vision_radius=actor_vision_radius,
        target_num_ants=target_num_ants,
        target_agent_identity_types=target_agent_identity_types,
        write_head_transfer=write_head_transfer,
        target_critic_architecture=source_critic_architecture,
        reset_optimizer=True,
    )
    actor_params = actor_checkpoint["params"]
    params = target_params._replace(
        actor_body=actor_params.actor_body,
        move_head=actor_params.move_head,
        write_head=actor_params.write_head,
    )
    target_args = {
        **actor_checkpoint.get("args", {}),
        "transfer_source_checkpoint": str(path),
        "actor_only_transfer": True,
        "critic_architecture": str(target_critic_architecture),
        "source_critic_architecture": source_critic_architecture,
    }
    return {
        **actor_checkpoint,
        "params": params,
        "opt_state": init_adam_state(params),
        "central_obs_dim": central_obs_dim,
        "actor_obs_dim": actor_obs_dim,
        "args": target_args,
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
    num_ants: int = 1,
    include_ants_count: bool = True,
    include_orientation: bool = True,
    include_agent_identity: bool = True,
    include_current_row: bool = True,
    agent_identity_types: int | None = None,
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
    identity_features = agent_identity_feature_count(
        num_ants,
        include_agent_identity=include_agent_identity,
        agent_identity_types=agent_identity_types,
    )
    return patch_size * grid_channels + identity_features + 1 + orientation_features


def agent_identity_feature_count(
    num_ants: int,
    *,
    include_agent_identity: bool = True,
    agent_identity_types: int | None = None,
) -> int:
    if not include_agent_identity:
        return 0
    if int(num_ants) <= 1:
        return 0
    if agent_identity_types is None:
        return int(num_ants)
    count = int(agent_identity_types)
    if count <= 0:
        raise ValueError("agent_identity_types must be positive.")
    return count


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
    include_orientation: bool,
    include_agent_identity: bool,
    source_layout: str,
    agent_identity_types: int | None = None,
) -> int:
    patch_size = source_actor_patch_size(
        actor_vision_radius=actor_vision_radius,
        source_layout=source_layout,
    )
    grid_channels = write_bits + (4 if include_ants_count else 3)
    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
    identity_features = agent_identity_feature_count(
        num_ants,
        include_agent_identity=include_agent_identity,
        agent_identity_types=agent_identity_types,
    )
    return patch_size * grid_channels + identity_features + 1 + orientation_features


def _actor_obs_source_shape(
    *,
    actor_obs_dim: int,
    write_bits: int,
    actor_vision_radius: int,
    num_ants: int = 1,
    agent_identity_types: int | None = None,
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
            for include_agent_identity in (True, False):
                for layout, patch_size, include_current_row in source_layouts:
                    grid_channels = write_bits + (4 if include_ants_count else 3)
                    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
                    identity_features = agent_identity_feature_count(
                        num_ants,
                        include_agent_identity=include_agent_identity,
                        agent_identity_types=agent_identity_types,
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
    global_feature_dim: int = CRITIC_GLOBAL_FEATURE_DIM,
) -> int:
    grid_area = obs_height * obs_width
    orientation_features = FACING_FEATURE_COUNT * num_ants if include_orientation else 0
    return 3 * num_ants + orientation_features + 3 * grid_area + int(global_feature_dim)


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
        global_feature_dim=LEGACY_GLOBAL_FEATURE_DIM,
    )
    old_current_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=source_height,
        obs_width=source_width,
        global_feature_dim=LEGACY_GLOBAL_FEATURE_DIM,
    )
    current_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=source_height,
        obs_width=source_width,
    )
    target_num_ants = _infer_num_ants_for_current_central_dim(
        target_central_obs_dim,
        obs_height=source_height,
        obs_width=source_width,
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
    elif old_weight.shape[0] == no_orientation_dim:
        current_first_layer = expand_central_input_layer_for_orientation(
            first_layer,
            num_ants=source_num_ants,
            obs_height=source_height,
            obs_width=source_width,
        )
    elif old_weight.shape[0] == old_current_dim:
        current_first_layer = expand_central_input_layer_for_aux_features(
            first_layer,
            num_ants=source_num_ants,
            obs_height=source_height,
            obs_width=source_width,
        )
    elif old_weight.shape[0] == current_dim:
        current_first_layer = first_layer
    else:
        raise ValueError("Checkpoint central observation dimension does not match this run.")

    new_first_layer = resize_central_input_layer_for_num_ants(
        current_first_layer,
        source_num_ants=source_num_ants,
        target_num_ants=target_num_ants,
        obs_height=source_height,
        obs_width=source_width,
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
    for global_feature_dim in (CRITIC_GLOBAL_FEATURE_DIM, LEGACY_GLOBAL_FEATURE_DIM):
        non_ant_dim = 3 * grid_area + global_feature_dim
        ant_dim = int(central_obs_dim) - non_ant_dim
        if ant_dim > 0 and ant_dim % 7 == 0:
            return ant_dim // 7
    raise ValueError("Checkpoint central observation dimension does not match this run.")


def resize_central_input_layer_for_num_ants(
    layer: LinearParams,
    *,
    source_num_ants: int,
    target_num_ants: int,
    obs_height: int,
    obs_width: int,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    source_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    target_dim = central_obs_dim_with_ants_count(
        num_ants=target_num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if old_weight.shape[0] != source_dim:
        raise ValueError(f"Expected central input dim {source_dim}, got {old_weight.shape[0]}.")
    if source_dim == target_dim and int(source_num_ants) == int(target_num_ants):
        return layer

    grid_area = obs_height * obs_width
    source_prefix_dim = 3 * source_num_ants
    source_orientation_dim = FACING_FEATURE_COUNT * source_num_ants
    source_maps_start = source_prefix_dim + source_orientation_dim
    source_tail = slice(source_maps_start + 3 * grid_area, source_dim)

    target_prefix_dim = 3 * target_num_ants
    target_orientation_dim = FACING_FEATURE_COUNT * target_num_ants
    target_maps_start = target_prefix_dim + target_orientation_dim
    target_tail = slice(target_maps_start + 3 * grid_area, target_dim)

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
    new_weight = new_weight.at[
        target_maps_start : target_maps_start + 3 * grid_area,
        :,
    ].set(
        old_weight[
            source_maps_start : source_maps_start + 3 * grid_area,
            :,
        ]
    )
    new_weight = new_weight.at[target_tail, :].set(old_weight[source_tail, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def expand_central_input_layer_for_aux_features(
    layer: LinearParams,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    old_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
        global_feature_dim=LEGACY_GLOBAL_FEATURE_DIM,
    )
    current_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if old_weight.shape[0] != old_dim:
        raise ValueError(f"Expected central input dim {old_dim}, got {old_weight.shape[0]}.")
    if old_dim == current_dim:
        return layer

    new_weight = jnp.zeros((current_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    new_weight = new_weight.at[:old_dim, :].set(old_weight)
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def resize_critic_entity_input_layer_for_num_ants(
    layer: LinearParams,
    *,
    source_num_ants: int,
    target_num_ants: int,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    source_ant_dim = 7 * int(source_num_ants)
    source_tail_width = int(old_weight.shape[0]) - source_ant_dim
    target_dim = 7 * int(target_num_ants) + CRITIC_GLOBAL_FEATURE_DIM
    if source_tail_width <= 0:
        raise ValueError(
            "Checkpoint critic entity input dimension does not match its ant count "
            f"({old_weight.shape[0]} for {source_num_ants} ants)."
        )
    if old_weight.shape[0] == target_dim and int(source_num_ants) == int(target_num_ants):
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
    shared_tail_width = min(source_tail_width, CRITIC_GLOBAL_FEATURE_DIM)
    new_weight = new_weight.at[
        target_facing_end : target_facing_end + shared_tail_width,
        :,
    ].set(
        old_weight[
            source_facing_end : source_facing_end + shared_tail_width,
            :,
        ]
    )
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def resize_non_mlp_critic_for_ants_count(
    params: JaxMAPPOParams,
    *,
    source_args: dict[str, Any],
    target_central_obs_dim: int,
) -> JaxMAPPOParams:
    source_num_ants = int(source_args.get("num_ants", 1))
    source_width = _source_grid_size(source_args, "width")
    source_height = _source_grid_size(source_args, "height")
    target_num_ants = _infer_num_ants_for_current_central_dim(
        target_central_obs_dim,
        obs_height=source_height,
        obs_width=source_width,
    )
    critic_body = params.critic_body
    if isinstance(critic_body, StridedCNNCriticParams):
        resized_body = critic_body._replace(
            entity_dense=resize_critic_entity_input_layer_for_num_ants(
                critic_body.entity_dense,
                source_num_ants=source_num_ants,
                target_num_ants=target_num_ants,
            )
        )
    elif isinstance(critic_body, ResNetCriticParams):
        resized_first = resize_critic_entity_input_layer_for_num_ants(
            critic_body.entity_body[0],
            source_num_ants=source_num_ants,
            target_num_ants=target_num_ants,
        )
        resized_body = critic_body._replace(
            entity_body=(resized_first, critic_body.entity_body[1])
        )
    elif isinstance(critic_body, StructuredMLPCriticParams):
        resized_first = resize_critic_entity_input_layer_for_num_ants(
            critic_body.entity_body[0],
            source_num_ants=source_num_ants,
            target_num_ants=target_num_ants,
        )
        resized_body = critic_body._replace(
            entity_body=(resized_first, critic_body.entity_body[1])
        )
    elif isinstance(critic_body, SetCNNCriticParams):
        resized_body = critic_body
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
    new_tail_start = new_prefix_dim + 3 * grid_area
    copied_tail_width = min(old_tail.stop - old_tail.start, current_dim - new_tail_start)

    new_weight = jnp.zeros((current_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    new_weight = new_weight.at[:prefix_dim, :].set(old_weight[:prefix_dim, :])
    new_weight = new_weight.at[new_food, :].set(old_weight[old_food, :])
    new_weight = new_weight.at[new_bytes, :].set(old_weight[old_bytes, :])
    new_weight = new_weight.at[
        new_tail_start : new_tail_start + copied_tail_width,
        :,
    ].set(old_weight[old_tail.start : old_tail.start + copied_tail_width, :])
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
        global_feature_dim=LEGACY_GLOBAL_FEATURE_DIM,
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
    grid_and_old_tail_width = 3 * grid_area + LEGACY_GLOBAL_FEATURE_DIM
    old_maps_and_tail = slice(prefix_dim, prefix_dim + grid_and_old_tail_width)
    new_maps_and_tail = slice(
        prefix_dim + orientation_dim,
        prefix_dim + orientation_dim + grid_and_old_tail_width,
    )

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
    source_includes_agent_identity: bool = False,
    source_layout: str = "centered",
    source_num_ants: int = 1,
    source_agent_identity_types: int | None = None,
    target_num_ants: int = 1,
    target_agent_identity_types: int | None = None,
    write_head_transfer: str = "repeat",
) -> JaxMAPPOParams:
    write_head_transfer = validate_write_head_transfer(write_head_transfer)
    source_identity_features = agent_identity_feature_count(
        source_num_ants,
        include_agent_identity=source_includes_agent_identity,
        agent_identity_types=source_agent_identity_types,
    )
    target_identity_features = agent_identity_feature_count(
        target_num_ants,
        agent_identity_types=target_agent_identity_types,
    )
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
                source_agent_identity_types=source_agent_identity_types,
                target_num_ants=target_num_ants,
                target_agent_identity_types=target_agent_identity_types,
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
    source_agent_identity_types: int | None = None,
    target_agent_identity_types: int | None = None,
    source_includes_agent_identity: bool = True,
) -> JaxMAPPOParams:
    source_identity_features = agent_identity_feature_count(
        source_num_ants,
        include_agent_identity=source_includes_agent_identity,
        agent_identity_types=source_agent_identity_types,
    )
    target_identity_features = agent_identity_feature_count(
        target_num_ants,
        agent_identity_types=target_agent_identity_types,
    )
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
                source_agent_identity_types=source_agent_identity_types,
                target_agent_identity_types=target_agent_identity_types,
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
    source_includes_agent_identity: bool = False,
    source_layout: str = "centered",
    source_num_ants: int = 1,
    source_agent_identity_types: int | None = None,
    target_num_ants: int = 1,
    target_agent_identity_types: int | None = None,
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
        agent_identity_types=source_agent_identity_types,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=actor_vision_radius,
        num_ants=target_num_ants,
        agent_identity_types=target_agent_identity_types,
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
        agent_identity_types=source_agent_identity_types,
    )
    old_identity = slice(old_tail_start, old_tail_start + old_identity_width)
    old_carrying = slice(old_identity.stop, old_identity.stop + 1)
    old_orientation = (
        slice(old_carrying.stop, old_carrying.stop + FACING_FEATURE_COUNT)
        if source_includes_orientation
        else None
    )
    new_tail_start = target_patch_size * (target_bits + 4)
    new_identity_width = agent_identity_feature_count(
        target_num_ants,
        agent_identity_types=target_agent_identity_types,
    )
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
    source_agent_identity_types: int | None = None,
    target_agent_identity_types: int | None = None,
    source_includes_agent_identity: bool = True,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    source_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=source_actor_vision_radius,
        num_ants=source_num_ants,
        include_agent_identity=source_includes_agent_identity,
        agent_identity_types=source_agent_identity_types,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=target_actor_vision_radius,
        num_ants=target_num_ants,
        agent_identity_types=target_agent_identity_types,
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
        agent_identity_types=source_agent_identity_types,
    )
    target_identity_width = agent_identity_feature_count(
        target_num_ants,
        agent_identity_types=target_agent_identity_types,
    )
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
