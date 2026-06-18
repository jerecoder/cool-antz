"""Ant byte foraging Gymnasium environment."""

from gymnasium.envs.registration import register, registry

from ant_byte_env.autocurriculum_env import AntByteAutoCurriculumEnv
from ant_byte_env.env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
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
    movement_facing,
    write_value_count,
)

if "AntByteForaging-v0" not in registry:
    register(
        id="AntByteForaging-v0",
        entry_point="ant_byte_env.env:AntByteForagingEnv",
    )

if "AntByteAutoCurriculum-v0" not in registry:
    register(
        id="AntByteAutoCurriculum-v0",
        entry_point="ant_byte_env.autocurriculum_env:AntByteAutoCurriculumEnv",
    )

__all__ = [
    "AntByteAutoCurriculumEnv",
    "AntByteForagingEnv",
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_STAY",
    "ACTION_UP",
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
    "movement_facing",
    "write_value_count",
]
