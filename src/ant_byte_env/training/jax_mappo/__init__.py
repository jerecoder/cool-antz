"""JAX MAPPO training utilities for AntByte."""

from ant_byte_env import write_value_count
from ant_byte_env.training.jax_mappo.checkpointing import (
    checkpoint_args,
    load_checkpoint,
    save_checkpoint,
)
from ant_byte_env.training.jax_mappo.cli import parse_args
from ant_byte_env.training.jax_mappo.core import (
    AdamState,
    JaxMAPPOParams,
    LinearParams,
    Rollout,
    TrainingBatch,
    Transition,
    UpdateMetrics,
    build_actor_observations,
    build_central_observations,
    build_local_border_patches,
    build_local_byte_bit_patches,
    build_local_grid_patches,
    build_local_hub_patches,
    compute_forage_curriculum_rewards,
    compute_gae,
    evaluate_actions,
    flatten_agent_actions,
    get_action_and_value,
    get_action_logits,
    get_value,
    init_adam_state,
    init_agent_params,
    init_layer,
    update_agent,
)
from ant_byte_env.training.jax_mappo.curriculum import reset_batch
from ant_byte_env.training.jax_mappo.rollout import collect_rollout
from ant_byte_env.training.jax_mappo.runner import main

__all__ = [
    "AdamState",
    "JaxMAPPOParams",
    "LinearParams",
    "Rollout",
    "TrainingBatch",
    "Transition",
    "UpdateMetrics",
    "build_actor_observations",
    "build_central_observations",
    "build_local_border_patches",
    "build_local_byte_bit_patches",
    "build_local_grid_patches",
    "build_local_hub_patches",
    "checkpoint_args",
    "collect_rollout",
    "compute_forage_curriculum_rewards",
    "compute_gae",
    "evaluate_actions",
    "flatten_agent_actions",
    "get_action_and_value",
    "get_action_logits",
    "get_value",
    "init_adam_state",
    "init_agent_params",
    "init_layer",
    "load_checkpoint",
    "main",
    "parse_args",
    "reset_batch",
    "save_checkpoint",
    "update_agent",
    "write_value_count",
]
