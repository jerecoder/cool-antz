"""Checkpoint I/O for Torch MAPPO."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

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
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_central_dim = int(checkpoint["central_obs_dim"])
    saved_actor_dim = int(checkpoint["actor_obs_dim"])
    if saved_central_dim != central_obs_dim or saved_actor_dim != actor_obs_dim:
        raise ValueError(
            "Checkpoint observation dimensions do not match this run. "
            "Use the same --obs-width, --obs-height, --num-ants, and --write-bits "
            "across curriculum stages."
        )

    agent.load_state_dict(checkpoint["agent_state_dict"])
    return checkpoint
