"""Adversarial rollout containers."""

from __future__ import annotations

from typing import NamedTuple

import jax


class AdversarialTransition(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    joint_actions: jax.Array
    logprobs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    terminations: jax.Array
    values: jax.Array
    next_values: jax.Array
    env_rewards: jax.Array
    pickup_events: jax.Array
    delivery_events: jax.Array
    delivered_food: jax.Array
    remaining_food: jax.Array


class AdversarialRollout(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    joint_actions: jax.Array
    logprobs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    terminations: jax.Array
    values: jax.Array
    next_values: jax.Array
    env_rewards: jax.Array
    pickup_events: jax.Array
    delivery_events: jax.Array
    delivered_food: jax.Array
    remaining_food: jax.Array
