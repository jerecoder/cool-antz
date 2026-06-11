"""Ant byte foraging Gymnasium environment."""

from gymnasium.envs.registration import register, registry

from ant_byte_env.agent import (
    AntAction,
    AntAgent,
    AntColonyController,
    AntContext,
    AntModel,
    AntTransition,
)
from ant_byte_env.env import AntByteForagingEnv

if "AntByteForaging-v0" not in registry:
    register(
        id="AntByteForaging-v0",
        entry_point="ant_byte_env.env:AntByteForagingEnv",
    )

__all__ = [
    "AntAction",
    "AntAgent",
    "AntByteForagingEnv",
    "AntColonyController",
    "AntContext",
    "AntModel",
    "AntTransition",
]
