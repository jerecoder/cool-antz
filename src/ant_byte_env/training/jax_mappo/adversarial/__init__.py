"""Experimental adversarial JAX MAPPO training lane."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "JaxAdversarialAntByteEnv": ("env", "JaxAdversarialAntByteEnv"),
    "JaxAdversarialAntInfo": ("env", "JaxAdversarialAntInfo"),
    "JaxAdversarialAntState": ("env", "JaxAdversarialAntState"),
    "build_team_actor_observations": ("observations", "build_team_actor_observations"),
    "build_team_central_observations": ("observations", "build_team_central_observations"),
    "collect_rollout": ("rollout", "collect_rollout"),
    "compose_team_actions": ("rollout", "compose_team_actions"),
    "draw_adversarial_frame": ("rendering", "draw_adversarial_frame"),
    "evaluate_checkpoint_matrix": ("checkpointing", "evaluate_checkpoint_matrix"),
    "evaluate_matrix": ("evaluation", "evaluate_matrix"),
    "init_adversarial_params": ("setup", "init_adversarial_params"),
    "load_checkpoint_for_evaluation": ("checkpointing", "load_checkpoint_for_evaluation"),
    "main": ("runner", "main"),
    "make_env": ("setup", "make_env"),
    "parse_args": ("cli", "parse_args"),
    "render_adversarial_rollout": ("rendering", "render_adversarial_rollout"),
    "reset_batch": ("env", "reset_batch"),
    "warm_start_actor_params": ("transfer", "warm_start_actor_params"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(f"ant_byte_env.training.jax_mappo.adversarial.{module_name}")
    return getattr(module, attr_name)
