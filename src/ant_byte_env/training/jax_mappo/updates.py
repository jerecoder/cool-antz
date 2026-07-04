"""GAE, optimizer, and PPO update helpers for JAX MAPPO."""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp

from ant_byte_env.training.jax_mappo.models import (
    critic_forward_kwargs_from_args,
    get_action_logits,
)
from ant_byte_env.training.jax_mappo.policy import (
    _logits_for_policy_temperature,
    evaluate_actions,
)
from ant_byte_env.training.jax_mappo.types import (
    AdamState,
    GradientNorms,
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
    agent_masks = getattr(rollout, "agent_masks", None)
    if agent_masks is None:
        agent_masks = jnp.ones_like(rollout.logprobs, dtype=jnp.float32)
    return TrainingBatch(
        actor_obs=rollout.actor_obs.reshape(batch_size, args.num_ants, -1),
        central_obs=rollout.central_obs.reshape(batch_size, -1),
        actions=rollout.actions.reshape(batch_size, args.num_ants, 2),
        agent_masks=agent_masks.reshape(batch_size, args.num_ants),
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


def _clip_tree_by_norm(tree: Any, *, norm: jax.Array, max_norm: float | None) -> Any:
    if max_norm is None or float(max_norm) <= 0.0:
        return tree
    scale = jnp.minimum(1.0, float(max_norm) / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda grad: grad * scale, tree)


def _clip_actor_critic_gradients(
    grads: JaxMAPPOParams,
    *,
    actor_max_grad_norm: float | None,
    critic_max_grad_norm: float | None,
) -> tuple[JaxMAPPOParams, GradientNorms]:
    actor_grads = (grads.actor_body, grads.move_head, grads.write_head)
    critic_grads = (grads.critic_body, grads.value_head)
    actor_norm = _global_norm(actor_grads)
    critic_norm = _global_norm(critic_grads)
    clipped_actor_body, clipped_move_head, clipped_write_head = _clip_tree_by_norm(
        actor_grads,
        norm=actor_norm,
        max_norm=actor_max_grad_norm,
    )
    clipped_critic_body, clipped_value_head = _clip_tree_by_norm(
        critic_grads,
        norm=critic_norm,
        max_norm=critic_max_grad_norm,
    )
    clipped_grads = JaxMAPPOParams(
        actor_body=clipped_actor_body,
        move_head=clipped_move_head,
        write_head=clipped_write_head,
        critic_body=clipped_critic_body,
        value_head=clipped_value_head,
    )
    return clipped_grads, GradientNorms(
        global_norm=_global_norm(grads),
        actor_norm=actor_norm,
        critic_norm=critic_norm,
    )


def _freeze_actor_grads(grads: JaxMAPPOParams) -> JaxMAPPOParams:
    return grads._replace(
        actor_body=jax.tree_util.tree_map(jnp.zeros_like, grads.actor_body),
        move_head=jax.tree_util.tree_map(jnp.zeros_like, grads.move_head),
        write_head=jax.tree_util.tree_map(jnp.zeros_like, grads.write_head),
    )


def _freeze_actor_opt_state(state: AdamState) -> AdamState:
    return state._replace(
        m=_freeze_actor_grads(state.m),
        v=_freeze_actor_grads(state.v),
    )


def _restore_actor_params(
    params: JaxMAPPOParams,
    actor_source: JaxMAPPOParams,
) -> JaxMAPPOParams:
    return params._replace(
        actor_body=actor_source.actor_body,
        move_head=actor_source.move_head,
        write_head=actor_source.write_head,
    )


def _categorical_kl(anchor_logits: jax.Array, current_logits: jax.Array) -> jax.Array:
    anchor_log_probs = jax.nn.log_softmax(anchor_logits, axis=-1)
    current_log_probs = jax.nn.log_softmax(current_logits, axis=-1)
    anchor_probs = jnp.exp(anchor_log_probs)
    return jnp.sum(anchor_probs * (anchor_log_probs - current_log_probs), axis=-1)


def behavior_anchor_kl(
    anchor_params: JaxMAPPOParams,
    current_params: JaxMAPPOParams,
    actor_obs: jax.Array,
    *,
    policy_temperature: float = 1.0,
) -> jax.Array:
    """Mean KL(anchor actor || current actor) for move and write heads."""
    anchor_move_logits, anchor_write_logits = get_action_logits(anchor_params, actor_obs)
    current_move_logits, current_write_logits = get_action_logits(current_params, actor_obs)
    anchor_move_logits = _logits_for_policy_temperature(
        anchor_move_logits,
        policy_temperature=policy_temperature,
    )
    anchor_write_logits = _logits_for_policy_temperature(
        anchor_write_logits,
        policy_temperature=policy_temperature,
    )
    current_move_logits = _logits_for_policy_temperature(
        current_move_logits,
        policy_temperature=policy_temperature,
    )
    current_write_logits = _logits_for_policy_temperature(
        current_write_logits,
        policy_temperature=policy_temperature,
    )
    move_kl = _categorical_kl(anchor_move_logits, current_move_logits)
    write_kl = _categorical_kl(anchor_write_logits, current_write_logits)
    return jnp.mean(move_kl + write_kl)


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
    actor_max_grad_norm: float | None = None,
    critic_max_grad_norm: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-5,
) -> tuple[JaxMAPPOParams, AdamState, GradientNorms]:
    if actor_max_grad_norm is None and critic_max_grad_norm is None:
        grad_norm = _global_norm(grads)
        actor_norm = _global_norm((grads.actor_body, grads.move_head, grads.write_head))
        critic_norm = _global_norm((grads.critic_body, grads.value_head))
        if max_grad_norm > 0:
            scale = jnp.minimum(1.0, float(max_grad_norm) / (grad_norm + 1e-6))
            grads = jax.tree_util.tree_map(lambda grad: grad * scale, grads)
        norms = GradientNorms(
            global_norm=grad_norm,
            actor_norm=actor_norm,
            critic_norm=critic_norm,
        )
    else:
        if actor_max_grad_norm is None:
            actor_max_grad_norm = max_grad_norm
        if critic_max_grad_norm is None:
            critic_max_grad_norm = max_grad_norm
        grads, norms = _clip_actor_critic_gradients(
            grads,
            actor_max_grad_norm=actor_max_grad_norm,
            critic_max_grad_norm=critic_max_grad_norm,
        )

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
    return next_params, AdamState(count=count, m=m, v=v), norms


def _ppo_loss(
    params: JaxMAPPOParams,
    batch: TrainingBatch,
    *,
    args: argparse.Namespace,
    behavior_anchor_params: JaxMAPPOParams | None = None,
) -> tuple[jax.Array, UpdateMetrics]:
    policy_temperature = float(getattr(args, "training_rollout_temperature", 1.0))
    new_logprobs, entropy, values = evaluate_actions(
        params,
        batch.actor_obs,
        batch.central_obs,
        batch.actions,
        policy_temperature=policy_temperature,
        **critic_forward_kwargs_from_args(args),
    )
    advantages = batch.advantages
    if args.norm_adv:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    agent_advantages = advantages[:, None]
    logratio = new_logprobs - batch.old_logprobs
    ratio = jnp.exp(logratio)
    agent_masks = batch.agent_masks.astype(jnp.float32)
    mask_denominator = jnp.maximum(jnp.sum(agent_masks), 1.0)

    def masked_mean(values: jax.Array) -> jax.Array:
        return jnp.sum(values * agent_masks) / mask_denominator

    policy_loss_1 = -agent_advantages * ratio
    policy_loss_2 = -agent_advantages * jnp.clip(
        ratio,
        1.0 - args.clip_coef,
        1.0 + args.clip_coef,
    )
    policy_loss = masked_mean(jnp.maximum(policy_loss_1, policy_loss_2))
    value_loss = 0.5 * jnp.mean(jnp.square(values - batch.returns))
    entropy_mean = masked_mean(entropy)
    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_mean
    behavior_kl = jnp.asarray(0.0, dtype=jnp.float32)
    behavior_anchor_coef = float(getattr(args, "behavior_anchor_coef", 0.0))
    if behavior_anchor_params is not None and behavior_anchor_coef != 0.0:
        behavior_kl = behavior_anchor_kl(
            behavior_anchor_params,
            params,
            batch.actor_obs,
            policy_temperature=policy_temperature,
        )
        loss = loss + behavior_anchor_coef * behavior_kl
    approx_kl = masked_mean((ratio - 1.0) - logratio)
    clipfrac = masked_mean((jnp.abs(ratio - 1.0) > args.clip_coef).astype(jnp.float32))
    return loss, UpdateMetrics(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        behavior_anchor_kl=behavior_kl,
        clipfrac=clipfrac,
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
        actor_grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
        critic_grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def update_agent(
    *,
    args: argparse.Namespace,
    params: JaxMAPPOParams,
    opt_state: AdamState,
    rollout: Rollout,
    learning_rate: float | jax.Array,
    key: jax.Array,
    behavior_anchor_params: JaxMAPPOParams | None = None,
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
            behavior_anchor_params=behavior_anchor_params,
        )
        del loss
        if getattr(args, "freeze_actor", False):
            grads = _freeze_actor_grads(grads)
            current_opt_state = _freeze_actor_opt_state(current_opt_state)
        next_params, next_opt_state, grad_norms = adam_update(
            current_params,
            grads,
            current_opt_state,
            learning_rate=learning_rate,
            max_grad_norm=args.max_grad_norm,
            actor_max_grad_norm=getattr(args, "actor_max_grad_norm", None),
            critic_max_grad_norm=getattr(args, "critic_max_grad_norm", None),
        )
        if getattr(args, "freeze_actor", False):
            next_params = _restore_actor_params(next_params, current_params)
            next_opt_state = _freeze_actor_opt_state(next_opt_state)
        return (
            next_params,
            next_opt_state,
        ), metrics._replace(
            grad_norm=grad_norms.global_norm,
            actor_grad_norm=grad_norms.actor_norm,
            critic_grad_norm=grad_norms.critic_norm,
        )

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
