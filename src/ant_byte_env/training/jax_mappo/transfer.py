"""Current-format JAX checkpoint transfer helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ant_byte_env import DEFAULT_WRITE_BITS, MAX_WRITE_BITS, write_value_count
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.core import (
    JaxMAPPOParams,
    LinearParams,
    init_adam_state,
)


def load_checkpoint_for_training(
    path: Path,
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
    target_write_bits: int,
    actor_vision_radius: int,
) -> dict[str, Any]:
    checkpoint = read_checkpoint(path)
    source_args = checkpoint.get("args", {})
    params = checkpoint["params"]
    params_changed = False
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
        }

    source_write_bits = int(source_args.get("write_bits", DEFAULT_WRITE_BITS))
    source_actor_vision_radius = int(source_args.get("actor_vision_radius", actor_vision_radius))
    if source_actor_vision_radius != actor_vision_radius:
        raise ValueError("Checkpoint actor vision radius does not match this run.")
    if target_write_bits < source_write_bits:
        raise ValueError("Checkpoint actor observation dimension does not match this run.")
    if target_write_bits > MAX_WRITE_BITS:
        raise ValueError(f"target_write_bits must be at most {MAX_WRITE_BITS}.")

    expected_source_dim = actor_obs_dim_for_bits(
        write_bits=source_write_bits,
        actor_vision_radius=actor_vision_radius,
    )
    legacy_source_dim = actor_obs_dim_for_bits(
        write_bits=source_write_bits,
        actor_vision_radius=actor_vision_radius,
        include_ants_count=False,
    )
    source_actor_obs_dim = int(checkpoint["actor_obs_dim"])
    if source_actor_obs_dim not in {expected_source_dim, legacy_source_dim}:
        raise ValueError("Checkpoint actor observation dimension does not match its write-bit config.")

    params = expand_params_for_write_bits(
        params,
        old_bits=source_write_bits,
        target_bits=target_write_bits,
        actor_vision_radius=actor_vision_radius,
        source_includes_ants_count=source_actor_obs_dim == expected_source_dim,
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
        },
    }


def actor_obs_dim_for_bits(
    *,
    write_bits: int,
    actor_vision_radius: int,
    include_ants_count: bool = True,
) -> int:
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    patch_size = (2 * actor_vision_radius + 1) ** 2
    grid_channels = write_bits + (4 if include_ants_count else 3)
    return patch_size * grid_channels + 1


def central_obs_dim_with_ants_count(*, num_ants: int, obs_height: int, obs_width: int) -> int:
    grid_area = obs_height * obs_width
    return 3 * num_ants + 3 * grid_area + 4


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
    current_dim = central_obs_dim_with_ants_count(
        num_ants=source_num_ants,
        obs_height=source_height,
        obs_width=source_width,
    )
    if current_dim != target_central_obs_dim:
        raise ValueError("Checkpoint central observation dimension does not match this run.")

    first_layer = params.critic_body[0]
    old_weight = jnp.asarray(first_layer.weight)
    if old_weight.shape[0] != legacy_dim:
        raise ValueError("Checkpoint central observation dimension does not match this run.")

    new_first_layer = expand_central_input_layer_for_ants_count(
        first_layer,
        num_ants=source_num_ants,
        obs_height=source_height,
        obs_width=source_width,
    )
    return JaxMAPPOParams(
        actor_body=params.actor_body,
        move_head=params.move_head,
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
    old_food = slice(prefix_dim, prefix_dim + grid_area)
    old_bytes = slice(prefix_dim + grid_area, prefix_dim + 2 * grid_area)
    old_tail = slice(prefix_dim + 2 * grid_area, legacy_dim)
    new_food = slice(prefix_dim + grid_area, prefix_dim + 2 * grid_area)
    new_bytes = slice(prefix_dim + 2 * grid_area, prefix_dim + 3 * grid_area)
    new_tail = slice(prefix_dim + 3 * grid_area, current_dim)

    new_weight = jnp.zeros((current_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    new_weight = new_weight.at[:prefix_dim, :].set(old_weight[:prefix_dim, :])
    new_weight = new_weight.at[new_food, :].set(old_weight[old_food, :])
    new_weight = new_weight.at[new_bytes, :].set(old_weight[old_bytes, :])
    new_weight = new_weight.at[new_tail, :].set(old_weight[old_tail, :])
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
) -> JaxMAPPOParams:
    if target_bits == old_bits and source_includes_ants_count:
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
            ),
            params.actor_body[1],
        ),
        move_head=params.move_head,
        write_head=params.write_head
        if target_bits == old_bits
        else expand_write_head(
            params.write_head,
            old_bits=old_bits,
            target_bits=target_bits,
        ),
        critic_body=params.critic_body,
        value_head=params.value_head,
    )


def expand_actor_input_layer(
    layer: LinearParams,
    *,
    old_bits: int,
    target_bits: int,
    actor_vision_radius: int,
    source_includes_ants_count: bool = True,
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    patch_size = (2 * actor_vision_radius + 1) ** 2
    expected_old_dim = actor_obs_dim_for_bits(
        write_bits=old_bits,
        actor_vision_radius=actor_vision_radius,
        include_ants_count=source_includes_ants_count,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=actor_vision_radius,
    )
    if old_weight.shape[0] != expected_old_dim:
        raise ValueError(f"Expected actor input dim {expected_old_dim}, got {old_weight.shape[0]}.")

    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    old_food = slice(0, patch_size)
    if source_includes_ants_count:
        old_ants_count = slice(patch_size, 2 * patch_size)
        old_bits_slice = slice(2 * patch_size, patch_size * (2 + old_bits))
        old_hub = slice(patch_size * (2 + old_bits), patch_size * (3 + old_bits))
        old_border = slice(patch_size * (3 + old_bits), patch_size * (4 + old_bits))
    else:
        old_ants_count = None
        old_bits_slice = slice(patch_size, patch_size * (1 + old_bits))
        old_hub = slice(patch_size * (1 + old_bits), patch_size * (2 + old_bits))
        old_border = slice(patch_size * (2 + old_bits), patch_size * (3 + old_bits))
    new_ants_count = slice(patch_size, 2 * patch_size)
    new_bits_slice = slice(2 * patch_size, patch_size * (2 + old_bits))
    new_hub = slice(patch_size * (2 + target_bits), patch_size * (3 + target_bits))
    new_border = slice(patch_size * (3 + target_bits), patch_size * (4 + target_bits))

    new_weight = new_weight.at[old_food, :].set(old_weight[old_food, :])
    if old_ants_count is not None:
        new_weight = new_weight.at[new_ants_count, :].set(old_weight[old_ants_count, :])
    new_weight = new_weight.at[new_bits_slice, :].set(old_weight[old_bits_slice, :])
    new_weight = new_weight.at[new_hub, :].set(old_weight[old_hub, :])
    new_weight = new_weight.at[new_border, :].set(old_weight[old_border, :])
    new_weight = new_weight.at[-1, :].set(old_weight[-1, :])
    return LinearParams(weight=new_weight, bias=jnp.asarray(layer.bias))


def expand_write_head(layer: LinearParams, *, old_bits: int, target_bits: int) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    old_bias = jnp.asarray(layer.bias)
    old_count = write_value_count(old_bits)
    target_count = write_value_count(target_bits)
    if old_weight.shape[-1] != old_count:
        raise ValueError(f"Expected {old_count} old write logits, got {old_weight.shape[-1]}.")
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
