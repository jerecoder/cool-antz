"""JAX MAPPO training utilities for AntByte.

The exports are loaded lazily so pure helpers can be imported on machines that
do not have the optional JAX dependency installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AdamState": ("core", "AdamState"),
    "JaxMAPPOParams": ("core", "JaxMAPPOParams"),
    "LinearParams": ("core", "LinearParams"),
    "Rollout": ("core", "Rollout"),
    "TrainingBatch": ("core", "TrainingBatch"),
    "Transition": ("core", "Transition"),
    "UpdateMetrics": ("core", "UpdateMetrics"),
    "build_actor_observations": ("core", "build_actor_observations"),
    "build_central_observations": ("core", "build_central_observations"),
    "build_local_border_patches": ("core", "build_local_border_patches"),
    "build_local_byte_bit_patches": ("core", "build_local_byte_bit_patches"),
    "build_local_grid_patches": ("core", "build_local_grid_patches"),
    "build_local_hub_patches": ("core", "build_local_hub_patches"),
    "checkpoint_args": ("checkpointing", "checkpoint_args"),
    "collect_rollout": ("rollout", "collect_rollout"),
    "compute_forage_curriculum_rewards": ("core", "compute_forage_curriculum_rewards"),
    "compute_gae": ("core", "compute_gae"),
    "evaluate_actions": ("core", "evaluate_actions"),
    "evaluate_checkpoint": ("evaluation", "evaluate_checkpoint"),
    "evaluate_params": ("evaluation", "evaluate_params"),
    "flatten_agent_actions": ("core", "flatten_agent_actions"),
    "get_action_and_value": ("core", "get_action_and_value"),
    "get_action_logits": ("core", "get_action_logits"),
    "get_value": ("core", "get_value"),
    "init_adam_state": ("core", "init_adam_state"),
    "init_agent_params": ("core", "init_agent_params"),
    "init_layer": ("core", "init_layer"),
    "load_checkpoint": ("checkpointing", "load_checkpoint"),
    "load_checkpoint_for_training": ("transfer", "load_checkpoint_for_training"),
    "main": ("runner", "main"),
    "parse_args": ("cli", "parse_args"),
    "repeated_write_action_indices": ("transfer", "repeated_write_action_indices"),
    "reset_batch": ("curriculum", "reset_batch"),
    "save_checkpoint": ("checkpointing", "save_checkpoint"),
    "update_agent": ("core", "update_agent"),
}

__all__ = sorted([*_EXPORTS, "write_value_count"])


def __getattr__(name: str) -> Any:
    if name == "write_value_count":
        from ant_byte_env import write_value_count

        return write_value_count
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(f"ant_byte_env.training.jax_mappo.{module_name}")
    return getattr(module, attr_name)
