"""Policy distribution helpers for JAX MAPPO."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ant_byte_env.training.jax_mappo.models import get_action_logits, get_value
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams

def _categorical_log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def _logits_for_policy_temperature(
    logits: jax.Array,
    *,
    policy_temperature: float,
) -> jax.Array:
    temperature = float(policy_temperature)
    if temperature <= 0.0:
        raise ValueError("policy_temperature must be positive.")
    return logits / temperature


def evaluate_actions(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    central_obs: jax.Array,
    actions: jax.Array,
    *,
    policy_temperature: float = 1.0,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
    critic_extra_entity_dim: int = 4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    move_logits, write_logits = get_action_logits(params, actor_obs)
    move_logits = _logits_for_policy_temperature(
        move_logits,
        policy_temperature=policy_temperature,
    )
    write_logits = _logits_for_policy_temperature(
        write_logits,
        policy_temperature=policy_temperature,
    )
    logprob = _categorical_log_prob(move_logits, actions[..., 0])
    logprob += _categorical_log_prob(write_logits, actions[..., 1])
    entropy = _categorical_entropy(move_logits) + _categorical_entropy(write_logits)
    value = get_value(
        params,
        central_obs,
        critic_architecture=critic_architecture,
        critic_num_ants=critic_num_ants,
        critic_obs_height=critic_obs_height,
        critic_obs_width=critic_obs_width,
        critic_extra_entity_dim=critic_extra_entity_dim,
    )
    return logprob, entropy, value


def get_action_and_value(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    central_obs: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool = False,
    policy_temperature: float = 1.0,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
    critic_extra_entity_dim: int = 4,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    move_logits, write_logits = get_action_logits(params, actor_obs)
    move_logits = _logits_for_policy_temperature(
        move_logits,
        policy_temperature=policy_temperature,
    )
    write_logits = _logits_for_policy_temperature(
        write_logits,
        policy_temperature=policy_temperature,
    )
    if deterministic:
        move_actions = jnp.argmax(move_logits, axis=-1)
        write_actions = jnp.argmax(write_logits, axis=-1)
    else:
        move_key, write_key = jax.random.split(key)
        move_actions = jax.random.categorical(move_key, move_logits, axis=-1)
        write_actions = jax.random.categorical(write_key, write_logits, axis=-1)
    actions = jnp.stack([move_actions, write_actions], axis=-1).astype(jnp.int32)
    logprob = _categorical_log_prob(move_logits, move_actions)
    logprob += _categorical_log_prob(write_logits, write_actions)
    entropy = _categorical_entropy(move_logits) + _categorical_entropy(write_logits)
    value = get_value(
        params,
        central_obs,
        critic_architecture=critic_architecture,
        critic_num_ants=critic_num_ants,
        critic_obs_height=critic_obs_height,
        critic_obs_width=critic_obs_width,
        critic_extra_entity_dim=critic_extra_entity_dim,
    )
    return actions, logprob, entropy, value
