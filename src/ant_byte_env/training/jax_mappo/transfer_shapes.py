"""Shape and transfer-mode helpers for JAX MAPPO checkpoint adaptation."""

from __future__ import annotations

import numpy as np

from ant_byte_env import (
    DEFAULT_ACTOR_VISION_WIDTH,
    MAX_WRITE_BITS,
    MOVEMENT_ACTION_COUNT,
    actor_vision_patch_size,
    write_value_count,
)

WRITE_HEAD_TRANSFER_MODES = ("repeat", "reset", "neutral-new")
FACING_FEATURE_COUNT = MOVEMENT_ACTION_COUNT - 1


def validate_write_head_transfer(mode: str) -> str:
    if mode not in WRITE_HEAD_TRANSFER_MODES:
        choices = ", ".join(WRITE_HEAD_TRANSFER_MODES)
        raise ValueError(f"write_head_transfer must be one of: {choices}.")
    return mode


def actor_obs_dim_for_bits(
    *,
    write_bits: int,
    actor_vision_radius: int,
    num_ants: int = 1,
    include_ants_count: bool = True,
    include_orientation: bool = True,
    include_agent_identity: bool = True,
    include_current_row: bool = True,
) -> int:
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    patch_size = actor_vision_patch_size(actor_vision_radius)
    if not include_current_row:
        patch_size = DEFAULT_ACTOR_VISION_WIDTH * actor_vision_radius
    grid_channels = write_bits + (4 if include_ants_count else 3)
    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
    identity_features = agent_identity_feature_count(
        num_ants,
        include_agent_identity=include_agent_identity,
    )
    return patch_size * grid_channels + identity_features + 1 + orientation_features


def agent_identity_feature_count(
    num_ants: int,
    *,
    include_agent_identity: bool = True,
) -> int:
    if not include_agent_identity:
        return 0
    count = int(num_ants)
    return count if count > 1 else 0


def source_actor_patch_size(*, actor_vision_radius: int, source_layout: str) -> int:
    if source_layout == "centered":
        return actor_vision_patch_size(actor_vision_radius)
    if source_layout == "forward_current_row":
        return DEFAULT_ACTOR_VISION_WIDTH * (actor_vision_radius + 1)
    if source_layout == "forward_only":
        return DEFAULT_ACTOR_VISION_WIDTH * actor_vision_radius
    raise ValueError(f"Unsupported actor window layout: {source_layout}.")


def source_actor_obs_dim(
    *,
    write_bits: int,
    actor_vision_radius: int,
    num_ants: int,
    include_ants_count: bool,
    include_orientation: bool,
    include_agent_identity: bool,
    source_layout: str,
) -> int:
    patch_size = source_actor_patch_size(
        actor_vision_radius=actor_vision_radius,
        source_layout=source_layout,
    )
    grid_channels = write_bits + (4 if include_ants_count else 3)
    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
    identity_features = agent_identity_feature_count(
        num_ants,
        include_agent_identity=include_agent_identity,
    )
    return patch_size * grid_channels + identity_features + 1 + orientation_features


def _actor_obs_source_shape(
    *,
    actor_obs_dim: int,
    write_bits: int,
    actor_vision_radius: int,
    num_ants: int = 1,
) -> dict[str, bool | str] | None:
    source_layouts = (
        ("centered", actor_vision_patch_size(actor_vision_radius), True),
        (
            "forward_current_row",
            DEFAULT_ACTOR_VISION_WIDTH * (actor_vision_radius + 1),
            True,
        ),
        ("forward_only", DEFAULT_ACTOR_VISION_WIDTH * actor_vision_radius, False),
    )
    for include_ants_count in (True, False):
        for include_orientation in (True, False):
            for include_agent_identity in (True, False):
                for layout, patch_size, include_current_row in source_layouts:
                    grid_channels = write_bits + (4 if include_ants_count else 3)
                    orientation_features = FACING_FEATURE_COUNT if include_orientation else 0
                    identity_features = agent_identity_feature_count(
                        num_ants,
                        include_agent_identity=include_agent_identity,
                    )
                    expected_dim = (
                        patch_size * grid_channels
                        + identity_features
                        + 1
                        + orientation_features
                    )
                    if actor_obs_dim == expected_dim:
                        return {
                            "include_ants_count": include_ants_count,
                            "include_orientation": include_orientation,
                            "include_agent_identity": include_agent_identity,
                            "include_current_row": include_current_row,
                            "layout": layout,
                        }
    return None


def central_obs_dim_with_ants_count(
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
    include_orientation: bool = True,
) -> int:
    grid_area = obs_height * obs_width
    orientation_features = FACING_FEATURE_COUNT * num_ants if include_orientation else 0
    return 3 * num_ants + orientation_features + 3 * grid_area + 4


def legacy_central_obs_dim(*, num_ants: int, obs_height: int, obs_width: int) -> int:
    grid_area = obs_height * obs_width
    return 3 * num_ants + 2 * grid_area + 4


def repeated_write_action_indices(old_bits: int, target_bits: int) -> np.ndarray:
    if old_bits <= 0:
        raise ValueError("old_bits must be positive.")
    if target_bits < old_bits:
        raise ValueError("target_bits must be at least old_bits.")
    old_count = write_value_count(old_bits)
    target_count = write_value_count(target_bits)
    return np.arange(target_count, dtype=np.int64) % old_count
