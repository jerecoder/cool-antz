"""Ant byte foraging Gymnasium environment."""

from gymnasium.envs.registration import register, registry

from ant_byte_env.env import (
    ACTION_FORWARD,
    ACTION_STAY,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    DEFAULT_ACTOR_VISION_DEPTH,
    DEFAULT_ACTOR_VISION_WIDTH,
    DEFAULT_WRITE_BITS,
    MAX_WRITE_BITS,
    MAX_WRITE_VALUE,
    MOVEMENT_ACTION_COUNT,
    WRITE_VALUE_COUNT,
    AntByteForagingEnv,
    actor_vision_patch_size,
    facing_delta,
    max_write_value,
    turn_facing,
    write_value_count,
)

if "AntByteForaging-v0" not in registry:
    register(
        id="AntByteForaging-v0",
        entry_point="ant_byte_env.env:AntByteForagingEnv",
    )

__all__ = [
    "AntByteForagingEnv",
    "ACTION_FORWARD",
    "ACTION_STAY",
    "ACTION_TURN_LEFT",
    "ACTION_TURN_RIGHT",
    "DEFAULT_ACTOR_VISION_DEPTH",
    "DEFAULT_ACTOR_VISION_WIDTH",
    "DEFAULT_WRITE_BITS",
    "MAX_WRITE_BITS",
    "MAX_WRITE_VALUE",
    "MOVEMENT_ACTION_COUNT",
    "WRITE_VALUE_COUNT",
    "actor_vision_patch_size",
    "facing_delta",
    "max_write_value",
    "turn_facing",
    "write_value_count",
]
