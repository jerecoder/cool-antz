"""Critic-side JAX MAPPO checkpoint transfer helpers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ant_byte_env.training.jax_mappo.transfer_actor import adapt_movement_head_layer
from ant_byte_env.training.jax_mappo.transfer_shapes import (
    FACING_FEATURE_COUNT,
    central_obs_dim_with_ants_count,
    legacy_central_obs_dim,
)
from ant_byte_env.training.jax_mappo.types import (
    JaxMAPPOParams,
    LinearParams,
    ResNetCriticParams,
    StridedCNNCriticParams,
    StructuredMLPCriticParams,
)


def expand_critic_input_for_ants_count(
    params: JaxMAPPOParams,
    *,
    source_args: dict[str, Any],
    target_central_obs_dim: int,
) -> JaxMAPPOParams:
    source_num_ants = int(source_args.get("num_ants", 1))
    source_width = source_grid_size(source_args, "width")
    source_height = source_grid_size(source_args, "height")
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
    target_num_ants = infer_num_ants_for_current_central_dim(
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


def infer_num_ants_for_current_central_dim(
    central_obs_dim: int,
    *,
    obs_height: int,
    obs_width: int,
) -> int:
    grid_area = obs_height * obs_width
    non_ant_dim = 3 * grid_area + 4
    ant_dim = int(central_obs_dim) - non_ant_dim
    if ant_dim <= 0 or ant_dim % 7 != 0:
        raise ValueError("Checkpoint central observation dimension does not match this run.")
    return ant_dim // 7


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
    if source_dim == target_dim:
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


def resize_non_mlp_critic_for_ants_count(
    params: JaxMAPPOParams,
    *,
    source_args: dict[str, Any],
    target_central_obs_dim: int,
) -> JaxMAPPOParams:
    source_num_ants = int(source_args.get("num_ants", 1))
    source_width = source_grid_size(source_args, "width")
    source_height = source_grid_size(source_args, "height")
    target_num_ants = infer_num_ants_for_current_central_dim(
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


def source_grid_size(source_args: dict[str, Any], axis: str) -> int:
    obs_name = f"obs_{axis}"
    if source_args.get(obs_name) is not None:
        return int(source_args[obs_name])
    return int(source_args[axis])


__all__ = [
    "expand_central_input_layer_for_ants_count",
    "expand_central_input_layer_for_orientation",
    "expand_critic_input_for_ants_count",
    "infer_num_ants_for_current_central_dim",
    "resize_central_input_layer_for_num_ants",
    "resize_critic_entity_input_layer_for_num_ants",
    "resize_non_mlp_critic_for_ants_count",
    "source_grid_size",
]
