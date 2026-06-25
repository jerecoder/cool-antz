"""GAE, optimizer, and PPO update helpers for JAX MAPPO."""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp

from ant_byte_env.training.jax_mappo.models import critic_forward_kwargs_from_args
from ant_byte_env.training.jax_mappo.policy import evaluate_actions
from ant_byte_env.training.jax_mappo.types import (
    AdamState,
    JaxMAPPOParams,
    Rollout,
    TrainingBatch,
    UpdateMetrics,
)

def compute_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array | None = None,
    terminations: jax.Array | None = None,
    next_values: jax.Array | None = None,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    rewards = rewards.astype(jnp.float32)
    values = values.astype(jnp.float32)
    dones = dones.astype(jnp.float32)
    if terminations is None:
        terminations = dones
    terminations = terminations.astype(jnp.float32)
    if next_values is None:
        if next_value is None:
            raise ValueError("next_value or next_values must be provided.")
        next_values = jnp.concatenate(
            [values[1:], next_value.astype(jnp.float32)[None, ...]],
            axis=0,
        )
    next_values = next_values.astype(jnp.float32)

    def scan_step(
        last_gae: jax.Array,
        transition: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        reward, value, next_value_at_step, done, terminated = transition
        bootstrap_mask = 1.0 - terminated
        continuation_mask = 1.0 - done
        delta = reward + float(gamma) * next_value_at_step * bootstrap_mask - value
        advantage = delta + float(gamma) * float(gae_lambda) * continuation_mask * last_gae
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        scan_step,
        jnp.zeros_like(values[-1]),
        (
            rewards[::-1],
            values[::-1],
            next_values[::-1],
            dones[::-1],
            terminations[::-1],
        ),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def _flatten_rollout(rollout: Rollout, *, args: argparse.Namespace) -> TrainingBatch:
    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        terminations=rollout.terminations,
        next_values=rollout.next_values,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    batch_size = args.num_steps * args.num_envs
    return TrainingBatch(
        actor_obs=rollout.actor_obs.reshape(batch_size, args.num_ants, -1),
        central_obs=rollout.central_obs.reshape(batch_size, -1),
        actions=rollout.actions.reshape(batch_size, args.num_ants, 2),
        old_logprobs=rollout.logprobs.reshape(batch_size, args.num_ants),
        advantages=advantages.reshape(batch_size),
        returns=returns.reshape(batch_size),
    )


def _split_minibatches(batch: TrainingBatch, *, args: argparse.Namespace) -> TrainingBatch:
    minibatch_size = (args.num_steps * args.num_envs) // args.num_minibatches
    return jax.tree_util.tree_map(
        lambda value: value.reshape((args.num_minibatches, minibatch_size) + value.shape[1:]),
        batch,
    )


def _shuffle_batch(batch: TrainingBatch, *, key: jax.Array) -> TrainingBatch:
    batch_size = batch.advantages.shape[0]
    permutation = jax.random.permutation(key, batch_size)
    return jax.tree_util.tree_map(lambda value: value[permutation], batch)


def _global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def init_adam_state(params: JaxMAPPOParams) -> AdamState:
    return AdamState(
        count=jnp.asarray(0, dtype=jnp.int32),
        m=jax.tree_util.tree_map(jnp.zeros_like, params),
        v=jax.tree_util.tree_map(jnp.zeros_like, params),
    )


def adam_update(
    params: JaxMAPPOParams,
    grads: JaxMAPPOParams,
    state: AdamState,
    *,
    learning_rate: float | jax.Array,
    max_grad_norm: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-5,
) -> tuple[JaxMAPPOParams, AdamState, jax.Array]:
    grad_norm = _global_norm(grads)
    if max_grad_norm > 0:
        scale = jnp.minimum(1.0, float(max_grad_norm) / (grad_norm + 1e-6))
        grads = jax.tree_util.tree_map(lambda grad: grad * scale, grads)

    count = state.count + 1
    m = jax.tree_util.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, state.m, grads)
    v = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        state.v,
        grads,
    )
    m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**count), m)
    v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**count), v)
    next_params = jax.tree_util.tree_map(
        lambda param, mh, vh: param - learning_rate * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return next_params, AdamState(count=count, m=m, v=v), grad_norm


def _ppo_loss(
    params: JaxMAPPOParams,
    batch: TrainingBatch,
    *,
    args: argparse.Namespace,
) -> tuple[jax.Array, UpdateMetrics]:
    new_logprobs, entropy, values = evaluate_actions(
        params,
        batch.actor_obs,
        batch.central_obs,
        batch.actions,
        policy_temperature=float(getattr(args, "training_rollout_temperature", 1.0)),
        **critic_forward_kwargs_from_args(args),
    )
    advantages = batch.advantages
    if args.norm_adv:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    agent_advantages = advantages[:, None]
    logratio = new_logprobs - batch.old_logprobs
    ratio = jnp.exp(logratio)

    policy_loss_1 = -agent_advantages * ratio
    policy_loss_2 = -agent_advantages * jnp.clip(
        ratio,
        1.0 - args.clip_coef,
        1.0 + args.clip_coef,
    )
    policy_loss = jnp.mean(jnp.maximum(policy_loss_1, policy_loss_2))
    value_loss = 0.5 * jnp.mean(jnp.square(values - batch.returns))
    entropy_mean = jnp.mean(entropy)
    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_mean
    approx_kl = jnp.mean((ratio - 1.0) - logratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > args.clip_coef).astype(jnp.float32))
    return loss, UpdateMetrics(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def update_agent(
    *,
    args: argparse.Namespace,
    params: JaxMAPPOParams,
    opt_state: AdamState,
    rollout: Rollout,
    learning_rate: float | jax.Array,
    key: jax.Array,
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
        next_params, next_opt_state, grad_norm = adam_update(
            current_params,
            grads,
            current_opt_state,
            learning_rate=learning_rate,
            max_grad_norm=args.max_grad_norm,
        )
        return (next_params, next_opt_state), metrics._replace(grad_norm=grad_norm)

    def epoch_step(
        carry: tuple[JaxMAPPOParams, AdamState],
        epoch_key: jax.Array,
    ) -> tuple[tuple[JaxMAPPOParams, AdamState], UpdateMetrics]:
        minibatches = _split_minibatches(_shuffle_batch(batch, key=epoch_key), args=args)
        next_carry, minibatch_metrics = jax.lax.scan(minibatch_step, carry, minibatches)
        mean_metrics = jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=0), minibatch_metrics)
        return next_carry, mean_metrics

    epoch_keys = jax.random.split(key, args.update_epochs)
    (params, opt_state), epoch_metrics = jax.lax.scan(
        epoch_step,
        (params, opt_state),
        epoch_keys,
    )
    final_metrics = jax.tree_util.tree_map(lambda value: value[-1], epoch_metrics)
    return params, opt_state, final_metrics
