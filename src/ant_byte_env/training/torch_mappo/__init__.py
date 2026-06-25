"""Torch MAPPO training utilities for AntByte.

The package exposes the historical public names lazily so config validation can
import lightweight CLI helpers without requiring optional TorchRL dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "DEFAULT_VISION_COLORS": "ant_byte_env.training.torch_mappo.visualization",
    "JointMoveWriteCategorical": "ant_byte_env.training.torch_mappo.model",
    "MAPPOActorAdapter": "ant_byte_env.training.torch_mappo.model",
    "MAPPOAgent": "ant_byte_env.training.torch_mappo.model",
    "MAPPOCriticAdapter": "ant_byte_env.training.torch_mappo.model",
    "NumpyObs": "ant_byte_env.training.torch_mappo.observations",
    "TensorObs": "ant_byte_env.training.torch_mappo.observations",
    "build_actor_observations": "ant_byte_env.training.torch_mappo.observations",
    "build_central_observations": "ant_byte_env.training.torch_mappo.observations",
    "build_curriculum_reset_options": "ant_byte_env.training.torch_mappo.curriculum",
    "build_local_border_patches": "ant_byte_env.training.torch_mappo.observations",
    "build_local_byte_bit_patches": "ant_byte_env.training.torch_mappo.observations",
    "build_local_food_patches": "ant_byte_env.training.torch_mappo.observations",
    "build_local_grid_patches": "ant_byte_env.training.torch_mappo.observations",
    "build_local_hub_patches": "ant_byte_env.training.torch_mappo.observations",
    "checkpoint_args": "ant_byte_env.training.torch_mappo.checkpointing",
    "collect_rollout": "ant_byte_env.training.torch_mappo.rollout",
    "compute_forage_curriculum_rewards": "ant_byte_env.training.torch_mappo.curriculum",
    "draw_vision_squares": "ant_byte_env.training.torch_mappo.visualization",
    "evaluate_agent": "ant_byte_env.training.torch_mappo.evaluation",
    "evaluate_checkpoint": "ant_byte_env.training.torch_mappo.evaluation",
    "flatten_agent_actions": "ant_byte_env.training.torch_mappo.observations",
    "layer_init": "ant_byte_env.training.torch_mappo.model",
    "load_agent_checkpoint": "ant_byte_env.training.torch_mappo.checkpointing",
    "main": "ant_byte_env.training.torch_mappo.runner",
    "make_envs": "ant_byte_env.training.torch_mappo.rollout",
    "make_mappo_loss": "ant_byte_env.training.torch_mappo.model",
    "make_rollout_storage": "ant_byte_env.training.torch_mappo.rollout",
    "make_torchrl_actor": "ant_byte_env.training.torch_mappo.model",
    "make_torchrl_critic": "ant_byte_env.training.torch_mappo.model",
    "mastery_reached": "ant_byte_env.training.torch_mappo.evaluation",
    "obs_to_tensor": "ant_byte_env.training.torch_mappo.observations",
    "parse_args": "ant_byte_env.training.torch_mappo.cli",
    "reset_env": "ant_byte_env.training.torch_mappo.rollout",
    "rollout_storage_to_tensordict": "ant_byte_env.training.torch_mappo.rollout",
    "stack_obs": "ant_byte_env.training.torch_mappo.rollout",
    "update_agent": "ant_byte_env.training.torch_mappo.rollout",
    "write_value_count": "ant_byte_env",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
