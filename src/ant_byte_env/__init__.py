"""Ant byte foraging Gymnasium environment."""

from gymnasium.envs.registration import register, registry

from ant_byte_env.env import (
    DEFAULT_WRITE_BITS,
    MAX_WRITE_BITS,
    MAX_WRITE_VALUE,
    WRITE_VALUE_COUNT,
    AntByteForagingEnv,
    max_write_value,
    write_value_count,
)

if "AntByteForaging-v0" not in registry:
    register(
        id="AntByteForaging-v0",
        entry_point="ant_byte_env.env:AntByteForagingEnv",
    )

__all__ = [
    "AntByteForagingEnv",
    "DEFAULT_WRITE_BITS",
    "MAX_WRITE_BITS",
    "MAX_WRITE_VALUE",
    "WRITE_VALUE_COUNT",
    "max_write_value",
    "write_value_count",
]
