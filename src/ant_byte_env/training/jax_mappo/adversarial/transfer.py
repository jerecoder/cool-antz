"""Actor-only warm-start helpers for adversarial MAPPO."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from ant_byte_env import write_value_count
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.transfer_actor import adapt_movement_head_layer
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams


def warm_start_actor_params(
    target_params: JaxMAPPOParams,
    checkpoint_path: Path,
    *,
    actor_obs_dim: int,
    target_write_bits: int,
) -> JaxMAPPOParams:
    """Copy actor weights from a compatible cooperative checkpoint.

    The adversarial critic and optimizer are intentionally not restored because
    their value target differs from cooperative foraging.
    """

    checkpoint = read_checkpoint(Path(checkpoint_path))
    source_params = checkpoint["params"]
    source_actor_dim = int(checkpoint.get("actor_obs_dim", -1))
    if source_actor_dim != int(actor_obs_dim):
        raise ValueError(
            "Checkpoint actor observation dimension does not match adversarial actor "
            f"({source_actor_dim} != {actor_obs_dim})."
        )
    source_write_head = source_params.write_head
    target_write_count = write_value_count(target_write_bits)
    if int(jnp.asarray(source_write_head.bias).shape[0]) != int(target_write_count):
        raise ValueError("Checkpoint write head does not match adversarial write bits.")
    return JaxMAPPOParams(
        actor_body=source_params.actor_body,
        move_head=adapt_movement_head_layer(source_params.move_head),
        write_head=source_params.write_head,
        critic_body=target_params.critic_body,
        value_head=target_params.value_head,
    )
