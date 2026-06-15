"""Checkpoint I/O for Torch MAPPO."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ant_byte_env import (
    DEFAULT_ACTOR_VISION_WIDTH,
    MOVEMENT_ACTION_COUNT,
    actor_vision_patch_size,
)
from ant_byte_env.training.torch_mappo.model import MAPPOAgent


def checkpoint_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return checkpoint-safe CLI args made of torch safe-load primitives."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def load_agent_checkpoint(
    *,
    agent: MAPPOAgent,
    checkpoint_path: Path,
    central_obs_dim: int,
    actor_obs_dim: int,
    write_bits: int,
    actor_vision_radius: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_central_dim = int(checkpoint["central_obs_dim"])
    saved_actor_dim = int(checkpoint["actor_obs_dim"])
    if saved_central_dim != central_obs_dim:
        raise ValueError(
            "Checkpoint observation dimensions do not match this run. "
            "Use the same --obs-width, --obs-height, --num-ants, and --write-bits "
            "across curriculum stages."
        )

    agent_state_dict = adapt_agent_state_dict_for_actor_window(
        checkpoint["agent_state_dict"],
        saved_actor_dim=saved_actor_dim,
        actor_obs_dim=actor_obs_dim,
        write_bits=write_bits,
        actor_vision_radius=actor_vision_radius,
    )
    agent.load_state_dict(agent_state_dict)
    state_shapes_changed = _state_dict_shapes_changed(
        checkpoint["agent_state_dict"],
        agent_state_dict,
    )
    if saved_actor_dim == actor_obs_dim and not state_shapes_changed:
        return checkpoint

    adapted_checkpoint = {
        **checkpoint,
        "agent_state_dict": agent_state_dict,
        "actor_obs_dim": actor_obs_dim,
    }
    adapted_checkpoint.pop("optimizer_state_dict", None)
    return adapted_checkpoint


def _state_dict_shapes_changed(
    old_state_dict: dict[str, torch.Tensor],
    new_state_dict: dict[str, torch.Tensor],
) -> bool:
    return any(
        key in old_state_dict and old_state_dict[key].shape != value.shape
        for key, value in new_state_dict.items()
    )


def adapt_agent_state_dict_for_actor_window(
    state_dict: dict[str, torch.Tensor],
    *,
    saved_actor_dim: int,
    actor_obs_dim: int,
    write_bits: int,
    actor_vision_radius: int,
) -> dict[str, torch.Tensor]:
    adapted_state_dict = _adapt_movement_head_state_dict(state_dict)
    if saved_actor_dim == actor_obs_dim:
        return adapted_state_dict

    source_shape = _actor_obs_source_shape(
        actor_obs_dim=saved_actor_dim,
        write_bits=write_bits,
        actor_vision_radius=actor_vision_radius,
    )
    if source_shape is None:
        raise ValueError(
            "Checkpoint actor observation dimension does not match this run."
        )

    old_weight = adapted_state_dict["actor_body.0.weight"]
    target_patch_size = actor_vision_patch_size(actor_vision_radius)
    source_patch_size = _source_actor_patch_size(
        actor_vision_radius=actor_vision_radius,
        source_layout=str(source_shape["layout"]),
    )
    target_dim = target_patch_size * (write_bits + 4) + 1
    if target_dim != actor_obs_dim:
        raise ValueError("Target actor observation dimension does not match this run.")

    new_weight = torch.zeros(
        (old_weight.shape[0], actor_obs_dim),
        dtype=old_weight.dtype,
        device=old_weight.device,
    )
    source_slices = _actor_channel_slices(
        patch_size=source_patch_size,
        write_bits=write_bits,
        include_ants_count=source_shape["include_ants_count"],
    )
    target_slices = _actor_channel_slices(
        patch_size=target_patch_size,
        write_bits=write_bits,
        include_ants_count=True,
    )
    for name, source_slice in source_slices.items():
        if name == "ants_count" and source_slice is None:
            continue
        target_slice = target_slices[name]
        assert source_slice is not None
        new_weight = _copy_actor_patch_channel(
            new_weight,
            old_weight,
            source=source_slice,
            target=target_slice,
            actor_vision_radius=actor_vision_radius,
            source_layout=str(source_shape["layout"]),
        )
    new_weight[:, -1] = old_weight[:, -1]

    return {
        **adapted_state_dict,
        "actor_body.0.weight": new_weight,
    }


def _actor_obs_source_shape(
    *,
    actor_obs_dim: int,
    write_bits: int,
    actor_vision_radius: int,
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
        for layout, patch_size, include_current_row in source_layouts:
            channels = write_bits + (4 if include_ants_count else 3)
            if actor_obs_dim == patch_size * channels + 1:
                return {
                    "include_ants_count": include_ants_count,
                    "include_current_row": include_current_row,
                    "layout": layout,
                }
    return None


def _actor_channel_slices(
    *,
    patch_size: int,
    write_bits: int,
    include_ants_count: bool,
) -> dict[str, slice | None]:
    slices: dict[str, slice | None] = {"food": slice(0, patch_size)}
    if include_ants_count:
        slices["ants_count"] = slice(patch_size, 2 * patch_size)
        bits_start = 2 * patch_size
    else:
        slices["ants_count"] = None
        bits_start = patch_size
    for bit_index in range(write_bits):
        slices[f"bit_{bit_index}"] = slice(
            bits_start + bit_index * patch_size,
            bits_start + (bit_index + 1) * patch_size,
        )
    hub_start = bits_start + write_bits * patch_size
    slices["hub"] = slice(hub_start, hub_start + patch_size)
    slices["border"] = slice(hub_start + patch_size, hub_start + 2 * patch_size)
    return slices


def _copy_actor_patch_channel(
    new_weight: torch.Tensor,
    old_weight: torch.Tensor,
    *,
    source: slice,
    target: slice,
    actor_vision_radius: int,
    source_layout: str,
) -> torch.Tensor:
    if source_layout == "centered":
        new_weight[:, target] = old_weight[:, source]
        return new_weight

    target_width = 2 * actor_vision_radius + 1
    source_index = 0
    first_depth = 0 if source_layout == "forward_current_row" else 1
    for depth in range(first_depth, actor_vision_radius + 1):
        for lateral in range(
            -(DEFAULT_ACTOR_VISION_WIDTH // 2),
            DEFAULT_ACTOR_VISION_WIDTH // 2 + 1,
        ):
            target_index = (lateral + actor_vision_radius) * target_width
            target_index += depth + actor_vision_radius
            new_weight[:, target.start + target_index] = old_weight[
                :,
                source.start + source_index,
            ]
            source_index += 1
    return new_weight


def _source_actor_patch_size(*, actor_vision_radius: int, source_layout: str) -> int:
    if source_layout == "centered":
        return actor_vision_patch_size(actor_vision_radius)
    if source_layout == "forward_current_row":
        return DEFAULT_ACTOR_VISION_WIDTH * (actor_vision_radius + 1)
    if source_layout == "forward_only":
        return DEFAULT_ACTOR_VISION_WIDTH * actor_vision_radius
    raise ValueError(f"Unsupported actor window layout: {source_layout}.")


def _adapt_movement_head_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    weight = state_dict.get("move_head.weight")
    bias = state_dict.get("move_head.bias")
    if weight is None or bias is None:
        return state_dict
    old_count = int(bias.shape[0])
    if old_count == MOVEMENT_ACTION_COUNT:
        return state_dict
    legacy_turn_action_count = 4
    if old_count == legacy_turn_action_count:
        raise ValueError(
            "Legacy 4-action movement checkpoints cannot be automatically mapped "
            "onto the current cardinal movement action space."
        )
    raise ValueError(f"Checkpoint movement action count {old_count} does not match this run.")
