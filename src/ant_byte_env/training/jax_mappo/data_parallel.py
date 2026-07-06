"""Data-parallel JAX MAPPO helpers.

This module keeps multi-device mechanics out of the stable single-device
runner path. The public helpers are intentionally small: resolve devices,
derive per-device args, merge sharded rollout data for logging, and run one
PPO update with gradients averaged across devices.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from ant_byte_env.training.jax_mappo.core import (
    AdamState,
    JaxMAPPOParams,
    Rollout,
    TrainingBatch,
    UpdateMetrics,
    _flatten_rollout,
    _ppo_loss,
    _shuffle_batch,
    _split_minibatches,
    adam_update,
)

DATA_PARALLEL_AXIS_NAME = "mappo_devices"


def resolve_data_parallel_devices(
    requested_count: int,
    *,
    available_devices: Sequence[jax.Device] | None = None,
) -> tuple[jax.Device, ...]:
    devices = tuple(jax.local_devices() if available_devices is None else available_devices)
    if requested_count < 0:
        raise ValueError("--jax-device-count must be non-negative.")
    if requested_count == 0:
        selected = devices
    else:
        if requested_count > len(devices):
            raise ValueError(
                f"--jax-device-count requested {requested_count} devices, "
                f"but JAX only sees {len(devices)} local device(s)."
            )
        selected = devices[:requested_count]
    if not selected:
        raise ValueError("JAX did not report any local devices.")
    return selected


def per_device_args(args: argparse.Namespace, *, device_count: int) -> argparse.Namespace:
    if device_count <= 0:
        raise ValueError("device_count must be positive.")
    if int(args.num_envs) % int(device_count) != 0:
        raise ValueError("--num-envs must be divisible by the data-parallel device count.")

    local_args = argparse.Namespace(**vars(args))
    local_args.num_envs = int(args.num_envs) // int(device_count)
    local_batch_size = int(local_args.num_envs) * int(local_args.num_steps)
    if local_batch_size < int(local_args.num_minibatches):
        raise ValueError("--num-minibatches cannot exceed per-device rollout batch size.")
    if local_batch_size % int(local_args.num_minibatches) != 0:
        raise ValueError(
            "--num-minibatches must evenly divide each data-parallel per-device batch."
        )
    return local_args


def replicate_tree(tree: Any, *, devices: Sequence[jax.Device]) -> Any:
    device_count = len(tuple(devices))
    if device_count <= 0:
        raise ValueError("Cannot replicate a tree with no target devices.")
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (device_count,) + value.shape),
        tree,
    )


def unreplicate_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: value[0], tree)


def merge_device_observations(obs: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        key: value.reshape((value.shape[0] * value.shape[1],) + value.shape[2:])
        for key, value in obs.items()
    }


def merge_device_rollout(rollout: Rollout) -> Rollout:
    def merge_leaf(value: jax.Array) -> jax.Array:
        transposed = jnp.swapaxes(value, 0, 1)
        return transposed.reshape(
            (transposed.shape[0], transposed.shape[1] * transposed.shape[2])
            + transposed.shape[3:]
        )

    return jax.tree_util.tree_map(merge_leaf, rollout)


def update_agent_data_parallel(
    *,
    args: argparse.Namespace,
    params: JaxMAPPOParams,
    opt_state: AdamState,
    rollout: Rollout,
    learning_rate: float | jax.Array,
    key: jax.Array,
    axis_name: str = DATA_PARALLEL_AXIS_NAME,
) -> tuple[JaxMAPPOParams, AdamState, UpdateMetrics]:
    batch = _flatten_rollout(rollout, args=args)

    def minibatch_step(
        carry: tuple[JaxMAPPOParams, AdamState],
        minibatch: TrainingBatch,
    ) -> tuple[tuple[JaxMAPPOParams, AdamState], UpdateMetrics]:
        current_params, current_opt_state = carry
        (loss, metrics), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(
            current_params,
            minibatch,
            args=args,
        )
        del loss
        averaged_grads = jax.lax.pmean(grads, axis_name=axis_name)
        averaged_metrics = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, axis_name=axis_name),
            metrics,
        )
        next_params, next_opt_state, grad_norm = adam_update(
            current_params,
            averaged_grads,
            current_opt_state,
            learning_rate=learning_rate,
            max_grad_norm=args.max_grad_norm,
            actor_max_grad_norm=args.actor_max_grad_norm,
            critic_max_grad_norm=args.critic_max_grad_norm,
        )
        return (next_params, next_opt_state), averaged_metrics._replace(
            grad_norm=grad_norm.global_norm,
            actor_grad_norm=grad_norm.actor_norm,
            critic_grad_norm=grad_norm.critic_norm,
        )

    def epoch_step(
        carry: tuple[JaxMAPPOParams, AdamState],
        epoch_key: jax.Array,
    ) -> tuple[tuple[JaxMAPPOParams, AdamState], UpdateMetrics]:
        minibatches = _split_minibatches(_shuffle_batch(batch, key=epoch_key), args=args)
        next_carry, minibatch_metrics = jax.lax.scan(
            minibatch_step,
            carry,
            minibatches,
        )
        mean_metrics = jax.tree_util.tree_map(
            lambda value: jnp.mean(value, axis=0),
            minibatch_metrics,
        )
        return next_carry, mean_metrics

    epoch_keys = jax.random.split(key, args.update_epochs)
    (params, opt_state), epoch_metrics = jax.lax.scan(
        epoch_step,
        (params, opt_state),
        epoch_keys,
    )
    final_metrics = jax.tree_util.tree_map(lambda value: value[-1], epoch_metrics)
    return params, opt_state, final_metrics
