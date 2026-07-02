"""JAX MAPPO checkpoint loading and transfer orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ant_byte_env import DEFAULT_WRITE_BITS
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.transfer_actor import (
    adapt_movement_head,
    expand_params_for_write_bits,
    resize_params_for_actor_vision_radius,
)
from ant_byte_env.training.jax_mappo.transfer_critic import (
    expand_critic_input_for_ants_count,
    resize_non_mlp_critic_for_ants_count,
)
from ant_byte_env.training.jax_mappo.transfer_shapes import (
    MAX_WRITE_BITS,
    _actor_obs_source_shape,
    actor_obs_dim_for_bits,
    validate_write_head_transfer,
)
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams
from ant_byte_env.training.jax_mappo.updates import init_adam_state


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
        source_includes_orientation=source_shape["include_orientation"],
        source_includes_agent_identity=source_shape["include_agent_identity"],
        source_layout=source_shape["layout"],
        source_num_ants=source_num_ants,
        target_num_ants=target_num_ants,
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
            "actor_vision_radius": actor_vision_radius,
            "num_ants": target_num_ants,
            "transfer_source_checkpoint": str(path),
            "write_head_transfer": write_head_transfer,
        },
    }


def warm_start_actor_params(
    target_params: JaxMAPPOParams,
    path: Path,
    *,
    actor_obs_dim: int,
    target_write_bits: int,
    actor_vision_radius: int,
    target_num_ants: int = 1,
    write_head_transfer: str = "repeat",
) -> JaxMAPPOParams:
    """Copy a compatible checkpoint actor while keeping the target critic."""

    write_head_transfer = validate_write_head_transfer(write_head_transfer)
    checkpoint = read_checkpoint(path)
    source_args = checkpoint.get("args", {})
    source_params = checkpoint["params"]
    source_params, _ = adapt_movement_head(source_params)
    source_write_bits = int(source_args.get("write_bits", DEFAULT_WRITE_BITS))
    if target_write_bits < source_write_bits:
        raise ValueError("Checkpoint actor observation dimension does not match this run.")
    if target_write_bits > MAX_WRITE_BITS:
        raise ValueError(f"target_write_bits must be at most {MAX_WRITE_BITS}.")

    if int(checkpoint["actor_obs_dim"]) != actor_obs_dim:
        source_actor_obs_dim = int(checkpoint["actor_obs_dim"])
        source_actor_vision_radius = int(
            source_args.get("actor_vision_radius", actor_vision_radius)
        )
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
        source_params = expand_params_for_write_bits(
            source_params,
            old_bits=source_write_bits,
            target_bits=target_write_bits,
            actor_vision_radius=source_actor_vision_radius,
            source_includes_ants_count=source_shape["include_ants_count"],
            source_includes_orientation=source_shape["include_orientation"],
            source_includes_agent_identity=source_shape["include_agent_identity"],
            source_layout=source_shape["layout"],
            source_num_ants=source_num_ants,
            target_num_ants=target_num_ants,
            write_head_transfer=write_head_transfer,
        )
        source_params = resize_params_for_actor_vision_radius(
            source_params,
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
        )
        if target_dim != actor_obs_dim:
            raise ValueError("Transferred actor observation dimension does not match this run.")

    if int(checkpoint["actor_obs_dim"]) == actor_obs_dim:
        source_head_size = source_params.write_head.bias.shape[0]
        target_head_size = target_params.write_head.bias.shape[0]
        if int(source_head_size) != int(target_head_size):
            raise ValueError("Checkpoint write head does not match this run.")

    return JaxMAPPOParams(
        actor_body=source_params.actor_body,
        move_head=source_params.move_head,
        write_head=source_params.write_head,
        critic_body=target_params.critic_body,
        value_head=target_params.value_head,
    )


__all__ = ["load_checkpoint_for_training", "warm_start_actor_params"]
