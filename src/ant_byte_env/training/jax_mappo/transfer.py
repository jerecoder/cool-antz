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
    if int(checkpoint["central_obs_dim"]) != central_obs_dim:
        raise ValueError("Checkpoint central observation dimension does not match this run.")
    if int(checkpoint["actor_obs_dim"]) == actor_obs_dim:
        return checkpoint

    source_args = checkpoint.get("args", {})
    source_write_bits = int(source_args.get("write_bits", DEFAULT_WRITE_BITS))
    source_actor_vision_radius = int(source_args.get("actor_vision_radius", actor_vision_radius))
    if source_actor_vision_radius != actor_vision_radius:
        raise ValueError("Checkpoint actor vision radius does not match this run.")
    if target_write_bits <= source_write_bits:
        raise ValueError("Checkpoint actor observation dimension does not match this run.")
    if target_write_bits > MAX_WRITE_BITS:
        raise ValueError(f"target_write_bits must be at most {MAX_WRITE_BITS}.")

    expected_source_dim = actor_obs_dim_for_bits(
        write_bits=source_write_bits,
        actor_vision_radius=actor_vision_radius,
    )
    if int(checkpoint["actor_obs_dim"]) != expected_source_dim:
        raise ValueError("Checkpoint actor observation dimension does not match its write-bit config.")

    params = expand_params_for_write_bits(
        checkpoint["params"],
        old_bits=source_write_bits,
        target_bits=target_write_bits,
        actor_vision_radius=actor_vision_radius,
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
        "actor_obs_dim": actor_obs_dim,
        "args": {
            **source_args,
            "write_bits": target_write_bits,
            "transfer_source_checkpoint": str(path),
        },
    }


def actor_obs_dim_for_bits(*, write_bits: int, actor_vision_radius: int) -> int:
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    patch_size = (2 * actor_vision_radius + 1) ** 2
    return patch_size * (write_bits + 3) + 1


def expand_params_for_write_bits(
    params: JaxMAPPOParams,
    *,
    old_bits: int,
    target_bits: int,
    actor_vision_radius: int,
) -> JaxMAPPOParams:
    if target_bits == old_bits:
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
            ),
            params.actor_body[1],
        ),
        move_head=params.move_head,
        write_head=expand_write_head(
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
) -> LinearParams:
    old_weight = jnp.asarray(layer.weight)
    patch_size = (2 * actor_vision_radius + 1) ** 2
    expected_old_dim = actor_obs_dim_for_bits(
        write_bits=old_bits,
        actor_vision_radius=actor_vision_radius,
    )
    target_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=actor_vision_radius,
    )
    if old_weight.shape[0] != expected_old_dim:
        raise ValueError(f"Expected actor input dim {expected_old_dim}, got {old_weight.shape[0]}.")

    new_weight = jnp.zeros((target_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    old_food = slice(0, patch_size)
    old_bits_slice = slice(patch_size, patch_size * (1 + old_bits))
    old_hub = slice(patch_size * (1 + old_bits), patch_size * (2 + old_bits))
    old_border = slice(patch_size * (2 + old_bits), patch_size * (3 + old_bits))
    new_hub = slice(patch_size * (1 + target_bits), patch_size * (2 + target_bits))
    new_border = slice(patch_size * (2 + target_bits), patch_size * (3 + target_bits))

    new_weight = new_weight.at[old_food, :].set(old_weight[old_food, :])
    new_weight = new_weight.at[old_bits_slice, :].set(old_weight[old_bits_slice, :])
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
