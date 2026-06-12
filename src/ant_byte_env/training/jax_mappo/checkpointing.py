"""Checkpoint I/O for JAX MAPPO."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env.training.jax_mappo.core import AdamState, JaxMAPPOParams


def checkpoint_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _numpy_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(value), tree)


def _jax_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, tree)


def save_checkpoint(
    path: Path,
    *,
    params: JaxMAPPOParams,
    opt_state: AdamState,
    args: argparse.Namespace,
    central_obs_dim: int,
    actor_obs_dim: int,
    run_name: str,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as checkpoint_file:
        pickle.dump(
            {
                "params": _numpy_tree(params),
                "opt_state": _numpy_tree(opt_state),
                "args": checkpoint_args(args),
                "central_obs_dim": int(central_obs_dim),
                "actor_obs_dim": int(actor_obs_dim),
                "run_name": run_name,
                "metrics": metrics,
            },
            checkpoint_file,
        )


def load_checkpoint(
    path: Path,
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
) -> dict[str, Any]:
    with path.open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    if int(checkpoint["central_obs_dim"]) != central_obs_dim:
        raise ValueError("Checkpoint central observation dimension does not match this run.")
    if int(checkpoint["actor_obs_dim"]) != actor_obs_dim:
        raise ValueError("Checkpoint actor observation dimension does not match this run.")
    checkpoint["params"] = _jax_tree(checkpoint["params"])
    checkpoint["opt_state"] = _jax_tree(checkpoint["opt_state"])
    return checkpoint
