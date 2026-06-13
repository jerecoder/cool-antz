"""Reusable pure JAX MAPPO pieces for AntByte training."""

from __future__ import annotations

import argparse
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env import DEFAULT_WRITE_BITS, MAX_WRITE_BITS, max_write_value
from ant_byte_env.env import (
    DEFAULT_ACTOR_VISION_DEPTH,
    DEFAULT_ACTOR_VISION_WIDTH,
    DEFAULT_FACING,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
)
from ant_byte_env.jax_env import JaxObs


class LinearParams(NamedTuple):
    weight: jax.Array
    bias: jax.Array


class JaxMAPPOParams(NamedTuple):
    actor_body: tuple[LinearParams, LinearParams]
    move_head: LinearParams
    write_head: LinearParams
    critic_body: tuple[LinearParams, LinearParams]
    value_head: LinearParams


class AdamState(NamedTuple):
    count: jax.Array
    m: JaxMAPPOParams
    v: JaxMAPPOParams


class Transition(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    logprobs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    values: jax.Array
    env_rewards: jax.Array


class Rollout(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    logprobs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    values: jax.Array
    env_rewards: jax.Array
    next_value: jax.Array


class TrainingBatch(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    old_logprobs: jax.Array
    advantages: jax.Array
    returns: jax.Array


class UpdateMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clipfrac: jax.Array
    grad_norm: jax.Array


def _normalize_positions(positions: jax.Array, *, height: int, width: int) -> jax.Array:
    scale = jnp.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=jnp.float32)
    return positions.astype(jnp.float32) / scale


def _resolve_observation_grid_shape(
    obs: JaxObs,
    *,
    obs_height: int | None,
    obs_width: int | None,
) -> tuple[int, int, int, int]:
    _, current_height, current_width = obs["food"].shape
    target_height = current_height if obs_height is None else int(obs_height)
    target_width = current_width if obs_width is None else int(obs_width)
    if target_height < current_height or target_width < current_width:
        raise ValueError("Padded observation shape must be at least the environment grid.")
    return current_height, current_width, target_height, target_width


def _pad_grid(grid: jax.Array, *, height: int, width: int) -> jax.Array:
    if grid.shape[1:] == (height, width):
        return grid
    padded = jnp.zeros((grid.shape[0], height, width), dtype=grid.dtype)
    return padded.at[:, : grid.shape[1], : grid.shape[2]].set(grid)


def _ants_count_grid(obs: JaxObs, *, height: int, width: int) -> jax.Array:
    if "ants_count" in obs:
        return obs["ants_count"].astype(jnp.float32)

    ants_pos = obs["ants_pos"].astype(jnp.int32)
    batch_size = ants_pos.shape[0]
    batch_index = jnp.arange(batch_size)[:, None]
    return jnp.zeros((batch_size, height, width), dtype=jnp.float32).at[
        batch_index,
        ants_pos[..., 1],
        ants_pos[..., 0],
    ].add(1.0)


def build_central_observations(
    obs: JaxObs,
    *,
    food_scale: int,
    write_bits: int = DEFAULT_WRITE_BITS,
    obs_width: int | None = None,
    obs_height: int | None = None,
) -> jax.Array:
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    food = obs["food"].astype(jnp.float32)
    bytes_grid = obs["bytes"].astype(jnp.float32)
    batch_size = food.shape[0]
    ant_count_scale = max(float(obs["ants_pos"].shape[1]), 1.0)
    current_height, current_width, target_height, target_width = _resolve_observation_grid_shape(
        obs,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    ants_pos = _normalize_positions(obs["ants_pos"], height=target_height, width=target_width)
    ants_count = _ants_count_grid(obs, height=current_height, width=current_width)
    hub_pos = _normalize_positions(obs["hub_pos"], height=target_height, width=target_width)
    ants_carrying = obs["ants_carrying"].astype(jnp.float32)
    ants_count_norm = _pad_grid(
        ants_count / ant_count_scale,
        height=target_height,
        width=target_width,
    )
    food_norm = _pad_grid(
        food / max(float(food_scale), 1.0),
        height=target_height,
        width=target_width,
    )
    bytes_norm = _pad_grid(
        bytes_grid / max(float(max_write_value(write_bits)), 1.0),
        height=target_height,
        width=target_width,
    )
    grid_size = jnp.asarray(
        [
            current_width / max(float(target_width), 1.0),
            current_height / max(float(target_height), 1.0),
        ],
        dtype=jnp.float32,
    )
    grid_size = jnp.broadcast_to(grid_size, (batch_size, 2))
    return jnp.concatenate(
        [
            ants_pos.reshape(batch_size, -1),
            ants_carrying.reshape(batch_size, -1),
            ants_count_norm.reshape(batch_size, -1),
            food_norm.reshape(batch_size, -1),
            bytes_norm.reshape(batch_size, -1),
            hub_pos.reshape(batch_size, -1),
            grid_size,
        ],
        axis=-1,
    )


def build_actor_observations(
    obs: JaxObs,
    *,
    food_scale: int = 1,
    actor_vision_radius: int = DEFAULT_ACTOR_VISION_DEPTH,
    write_bits: int = DEFAULT_WRITE_BITS,
    obs_width: int | None = None,
    obs_height: int | None = None,
) -> jax.Array:
    del obs_width, obs_height
    if actor_vision_radius < 0:
        raise ValueError("actor_vision_radius must be non-negative.")

    food = obs["food"].astype(jnp.float32)
    ant_count_scale = max(float(obs["ants_pos"].shape[1]), 1.0)
    ants_count = _ants_count_grid(obs, height=food.shape[1], width=food.shape[2])
    own_carrying = obs["ants_carrying"].astype(jnp.float32)[..., None]
    ants_facing = obs.get("ants_facing")
    if ants_facing is None:
        ants_facing = jnp.full(
            obs["ants_pos"].shape[:2],
            DEFAULT_FACING,
            dtype=jnp.int32,
        )
    local_food = build_local_grid_patches(
        food,
        obs["ants_pos"],
        radius=actor_vision_radius,
        ants_facing=ants_facing,
    )
    local_food = local_food / max(float(food_scale), 1.0)
    local_ants_count = build_local_grid_patches(
        ants_count,
        obs["ants_pos"],
        radius=actor_vision_radius,
        ants_facing=ants_facing,
    )
    local_ants_count = local_ants_count / ant_count_scale
    local_byte_bits = build_local_byte_bit_patches(
        obs["bytes"],
        obs["ants_pos"],
        radius=actor_vision_radius,
        ants_facing=ants_facing,
        write_bits=write_bits,
    )
    local_hub = build_local_hub_patches(
        obs["hub_pos"],
        obs["ants_pos"],
        ants_facing=ants_facing,
        grid_height=food.shape[1],
        grid_width=food.shape[2],
        radius=actor_vision_radius,
    )
    local_border = build_local_border_patches(
        obs["ants_pos"],
        ants_facing=ants_facing,
        grid_height=food.shape[1],
        grid_width=food.shape[2],
        radius=actor_vision_radius,
    )
    return jnp.concatenate(
        [local_food, local_ants_count, local_byte_bits, local_hub, local_border, own_carrying],
        axis=-1,
    )


def build_local_byte_bit_patches(
    bytes_grid: jax.Array,
    ants_pos: jax.Array,
    *,
    radius: int,
    ants_facing: jax.Array | None = None,
    write_bits: int = DEFAULT_WRITE_BITS,
) -> jax.Array:
    if write_bits <= 0 or write_bits > MAX_WRITE_BITS:
        raise ValueError(f"write_bits must be an integer from 1 to {MAX_WRITE_BITS}.")

    bytes_int = bytes_grid.astype(jnp.int32)
    bit_patches = []
    for bit_index in range(write_bits):
        bit_grid = ((bytes_int >> bit_index) & 1).astype(jnp.float32)
        bit_patches.append(
            build_local_grid_patches(
                bit_grid,
                ants_pos,
                radius=radius,
                ants_facing=ants_facing,
            )
        )
    return jnp.concatenate(bit_patches, axis=-1)


def build_local_grid_patches(
    grid: jax.Array,
    ants_pos: jax.Array,
    *,
    radius: int,
    ants_facing: jax.Array | None = None,
) -> jax.Array:
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    batch_size, grid_height, grid_width = grid.shape
    if ants_facing is None:
        ants_facing = jnp.full(ants_pos.shape[:2], DEFAULT_FACING, dtype=jnp.int32)
    offset_pairs = build_forward_vision_offsets(
        ants_facing.astype(jnp.int32),
        depth=radius,
    )
    positions = ants_pos.astype(jnp.int32)[:, :, None, :] + offset_pairs
    x_pos = positions[..., 0]
    y_pos = positions[..., 1]
    valid = (0 <= x_pos) & (x_pos < grid_width) & (0 <= y_pos) & (y_pos < grid_height)
    clipped_x = jnp.clip(x_pos, 0, grid_width - 1)
    clipped_y = jnp.clip(y_pos, 0, grid_height - 1)
    batch_index = jnp.arange(batch_size)[:, None, None]
    values = grid[batch_index, clipped_y, clipped_x].astype(jnp.float32)
    return jnp.where(valid, values, 0.0)


def build_local_border_patches(
    ants_pos: jax.Array,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
    ants_facing: jax.Array | None = None,
) -> jax.Array:
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    if ants_facing is None:
        ants_facing = jnp.full(ants_pos.shape[:2], DEFAULT_FACING, dtype=jnp.int32)
    offset_pairs = build_forward_vision_offsets(
        ants_facing.astype(jnp.int32),
        depth=radius,
    )
    positions = ants_pos.astype(jnp.int32)[:, :, None, :] + offset_pairs
    x_pos = positions[..., 0]
    y_pos = positions[..., 1]
    valid = (0 <= x_pos) & (x_pos < grid_width) & (0 <= y_pos) & (y_pos < grid_height)
    return jnp.where(valid, 0.0, 1.0).astype(jnp.float32)


def build_forward_vision_offsets(ants_facing: jax.Array, *, depth: int) -> jax.Array:
    if depth < 0:
        raise ValueError("depth must be non-negative.")

    half_width = DEFAULT_ACTOR_VISION_WIDTH // 2
    depth_offsets = jnp.repeat(
        jnp.arange(1, depth + 1, dtype=jnp.int32),
        DEFAULT_ACTOR_VISION_WIDTH,
    )
    lateral_offsets = jnp.tile(
        jnp.arange(-half_width, half_width + 1, dtype=jnp.int32),
        depth,
    )
    facing = ants_facing.astype(jnp.int32)
    valid_facing = (
        (facing == MOVE_UP)
        | (facing == MOVE_RIGHT)
        | (facing == MOVE_DOWN)
        | (facing == MOVE_LEFT)
    )
    facing = jnp.where(valid_facing, facing, DEFAULT_FACING)
    forward_x = jnp.where(
        facing == MOVE_RIGHT,
        1,
        jnp.where(facing == MOVE_LEFT, -1, 0),
    )
    forward_y = jnp.where(
        facing == MOVE_DOWN,
        1,
        jnp.where(facing == MOVE_UP, -1, 0),
    )
    right_x = -forward_y
    right_y = forward_x
    offset_x = forward_x[..., None] * depth_offsets + right_x[..., None] * lateral_offsets
    offset_y = forward_y[..., None] * depth_offsets + right_y[..., None] * lateral_offsets
    return jnp.stack([offset_x, offset_y], axis=-1)


def build_local_hub_patches(
    hub_pos: jax.Array,
    ants_pos: jax.Array,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
    ants_facing: jax.Array | None = None,
) -> jax.Array:
    batch_size = hub_pos.shape[0]
    hub_grid = jnp.zeros((batch_size, grid_height, grid_width), dtype=jnp.float32)
    batch_index = jnp.arange(batch_size)
    hub_grid = hub_grid.at[batch_index, hub_pos[:, 1], hub_pos[:, 0]].set(1.0)
    return build_local_grid_patches(
        hub_grid,
        ants_pos,
        radius=radius,
        ants_facing=ants_facing,
    )


def flatten_agent_actions(actions: jax.Array) -> jax.Array:
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError(f"joint actions must have shape (batch, ants, 2), got {actions.shape}.")
    return actions.astype(jnp.int32).reshape(actions.shape[0], actions.shape[1] * 2)


def init_layer(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    *,
    scale: float = np.sqrt(2.0),
) -> LinearParams:
    std = float(scale) / np.sqrt(max(float(in_dim), 1.0))
    return LinearParams(
        weight=jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * std,
        bias=jnp.zeros((out_dim,), dtype=jnp.float32),
    )


def init_agent_params(
    key: jax.Array,
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
    hidden_size: int = 128,
    write_value_count: int = 2,
) -> JaxMAPPOParams:
    if write_value_count <= 0:
        raise ValueError("write_value_count must be positive.")
    keys = jax.random.split(key, 7)
    return JaxMAPPOParams(
        actor_body=(
            init_layer(keys[0], actor_obs_dim, hidden_size),
            init_layer(keys[1], hidden_size, hidden_size),
        ),
        move_head=init_layer(keys[2], hidden_size, 5, scale=0.01),
        write_head=init_layer(keys[3], hidden_size, write_value_count, scale=0.01),
        critic_body=(
            init_layer(keys[4], central_obs_dim, hidden_size),
            init_layer(keys[5], hidden_size, hidden_size),
        ),
        value_head=init_layer(keys[6], hidden_size, 1, scale=1.0),
    )


def _linear(params: LinearParams, x: jax.Array) -> jax.Array:
    return x @ params.weight + params.bias


def _forward_body(layers: tuple[LinearParams, LinearParams], x: jax.Array) -> jax.Array:
    hidden = jnp.tanh(_linear(layers[0], x))
    return jnp.tanh(_linear(layers[1], hidden))


def get_action_logits(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    hidden = _forward_body(params.actor_body, actor_obs)
    return _linear(params.move_head, hidden), _linear(params.write_head, hidden)


def get_value(params: JaxMAPPOParams, central_obs: jax.Array) -> jax.Array:
    hidden = _forward_body(params.critic_body, central_obs)
    return jnp.squeeze(_linear(params.value_head, hidden), axis=-1)


def _categorical_log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def evaluate_actions(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    central_obs: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    move_logits, write_logits = get_action_logits(params, actor_obs)
    logprob = _categorical_log_prob(move_logits, actions[..., 0])
    logprob += _categorical_log_prob(write_logits, actions[..., 1])
    entropy = _categorical_entropy(move_logits) + _categorical_entropy(write_logits)
    value = get_value(params, central_obs)
    return logprob, entropy, value


def get_action_and_value(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
    central_obs: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    move_logits, write_logits = get_action_logits(params, actor_obs)
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
    value = get_value(params, central_obs)
    return actions, logprob, entropy, value


def _nearest_food_distance(position: jax.Array, food_grid: jax.Array) -> tuple[jax.Array, jax.Array]:
    grid_height, grid_width = food_grid.shape
    y_index, x_index = jnp.indices((grid_height, grid_width), dtype=jnp.int32)
    distances = jnp.abs(x_index - position[0]) + jnp.abs(y_index - position[1])
    has_food = jnp.any(food_grid > 0)
    sentinel = jnp.asarray(grid_height + grid_width + 1, dtype=jnp.float32)
    best = jnp.min(jnp.where(food_grid > 0, distances.astype(jnp.float32), sentinel))
    return best, has_food


def _distance_to_hub(position: jax.Array, hub_pos: jax.Array) -> jax.Array:
    return jnp.sum(jnp.abs(position.astype(jnp.int32) - hub_pos.astype(jnp.int32))).astype(
        jnp.float32
    )


def compute_forage_curriculum_rewards(
    *,
    previous_obs: JaxObs,
    next_obs: JaxObs,
    env_rewards: jax.Array,
    pickup_bonus: float,
    distance_bonus: float,
) -> jax.Array:
    def one_env(
        previous_ants_pos: jax.Array,
        previous_carrying: jax.Array,
        previous_food: jax.Array,
        previous_hub: jax.Array,
        next_ants_pos: jax.Array,
        next_carrying: jax.Array,
        next_food: jax.Array,
        env_reward: jax.Array,
    ) -> jax.Array:
        was_carrying = previous_carrying.astype(jnp.bool_)
        is_carrying = next_carrying.astype(jnp.bool_)
        pickups = jnp.logical_and(jnp.logical_not(was_carrying), is_carrying).astype(jnp.float32)

        def one_ant(previous_pos: jax.Array, next_pos: jax.Array, carrying: jax.Array) -> jax.Array:
            hub_progress = _distance_to_hub(previous_pos, previous_hub)
            hub_progress -= _distance_to_hub(next_pos, previous_hub)
            previous_food_distance, previous_has_food = _nearest_food_distance(
                previous_pos,
                previous_food,
            )
            next_food_distance, next_has_food = _nearest_food_distance(next_pos, next_food)
            food_progress = previous_food_distance - next_food_distance
            food_progress = jnp.where(
                jnp.logical_and(previous_has_food, next_has_food),
                food_progress,
                0.0,
            )
            return jnp.where(carrying, hub_progress, food_progress)

        progress = jax.vmap(one_ant)(previous_ants_pos, next_ants_pos, was_carrying)
        shaped = env_reward.astype(jnp.float32)
        shaped += float(pickup_bonus) * jnp.sum(pickups)
        shaped += float(distance_bonus) * jnp.sum(progress)
        return shaped

    return jax.vmap(one_env)(
        previous_obs["ants_pos"],
        previous_obs["ants_carrying"],
        previous_obs["food"],
        previous_obs["hub_pos"],
        next_obs["ants_pos"],
        next_obs["ants_carrying"],
        next_obs["food"],
        env_rewards,
    )


def compute_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    rewards = rewards.astype(jnp.float32)
    values = values.astype(jnp.float32)
    dones = dones.astype(jnp.float32)

    def scan_step(
        carry: tuple[jax.Array, jax.Array],
        transition: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        last_gae, next_values = carry
        reward, value, done = transition
        nonterminal = 1.0 - done
        delta = reward + float(gamma) * next_values * nonterminal - value
        advantage = delta + float(gamma) * float(gae_lambda) * nonterminal * last_gae
        return (advantage, value), advantage

    (_, _), reversed_advantages = jax.lax.scan(
        scan_step,
        (jnp.zeros_like(next_value), next_value.astype(jnp.float32)),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def _flatten_rollout(rollout: Rollout, *, args: argparse.Namespace) -> TrainingBatch:
    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        next_value=rollout.next_value,
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


def _global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def init_adam_state(params: JaxMAPPOParams) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(count=jnp.asarray(0, dtype=jnp.int32), m=zeros, v=zeros)


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
) -> tuple[JaxMAPPOParams, AdamState, UpdateMetrics]:
    batch = _flatten_rollout(rollout, args=args)
    minibatches = _split_minibatches(batch, args=args)

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
        _: Any,
    ) -> tuple[tuple[JaxMAPPOParams, AdamState], UpdateMetrics]:
        next_carry, minibatch_metrics = jax.lax.scan(minibatch_step, carry, minibatches)
        mean_metrics = jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=0), minibatch_metrics)
        return next_carry, mean_metrics

    (params, opt_state), epoch_metrics = jax.lax.scan(
        epoch_step,
        (params, opt_state),
        None,
        length=args.update_epochs,
    )
    final_metrics = jax.tree_util.tree_map(lambda value: value[-1], epoch_metrics)
    return params, opt_state, final_metrics
