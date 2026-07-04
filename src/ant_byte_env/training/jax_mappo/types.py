"""Shared JAX MAPPO parameter and rollout containers."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

CRITIC_AUX_FEATURE_DIM = 12
CRITIC_GLOBAL_FEATURE_DIM = 4 + CRITIC_AUX_FEATURE_DIM
SET_CNN_ANT_FEATURE_DIM = 7


class LinearParams(NamedTuple):
    weight: jax.Array
    bias: jax.Array


class ConvParams(NamedTuple):
    kernel: jax.Array
    bias: jax.Array


class ResidualBlockParams(NamedTuple):
    first: ConvParams
    second: ConvParams


class ResNetCriticParams(NamedTuple):
    stem: ConvParams
    blocks_32: tuple[ResidualBlockParams, ResidualBlockParams]
    down_64: ConvParams
    blocks_64: tuple[ResidualBlockParams, ResidualBlockParams]
    down_96: ConvParams
    blocks_96: tuple[ResidualBlockParams, ResidualBlockParams]
    down_128: ConvParams
    blocks_128: tuple[ResidualBlockParams]
    spatial_dense: LinearParams
    entity_body: tuple[LinearParams, LinearParams]
    fusion_body: tuple[LinearParams, LinearParams]


class StridedCNNCriticParams(NamedTuple):
    conv_5x5: ConvParams
    conv_3x3_a: ConvParams
    conv_3x3_b: ConvParams
    spatial_dense: LinearParams
    entity_dense: LinearParams
    fusion_dense: LinearParams


class SetCNNCriticParams(NamedTuple):
    conv_5x5: ConvParams
    conv_3x3_a: ConvParams
    conv_3x3_b: ConvParams
    spatial_dense: LinearParams
    ant_encoder: tuple[LinearParams, LinearParams]
    global_dense: LinearParams
    fusion_body: tuple[LinearParams, LinearParams]


class StructuredMLPCriticParams(NamedTuple):
    grid_body: tuple[LinearParams, LinearParams]
    entity_body: tuple[LinearParams, LinearParams]
    fusion_body: tuple[LinearParams, LinearParams]


class JaxMAPPOParams(NamedTuple):
    actor_body: tuple[LinearParams, LinearParams]
    move_head: LinearParams
    write_head: LinearParams
    critic_body: Any
    value_head: LinearParams


class AdamState(NamedTuple):
    count: jax.Array
    m: JaxMAPPOParams
    v: JaxMAPPOParams


class Transition(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    agent_masks: jax.Array
    logprobs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    values: jax.Array
    next_values: jax.Array
    env_rewards: jax.Array
    pickup_events: jax.Array
    delivery_events: jax.Array
    carrying_ants: jax.Array
    remaining_food: jax.Array
    remaining_lethal_food: jax.Array
    death_events: jax.Array
    alive_ant_count: jax.Array
    dead_ant_count: jax.Array
    active_size: jax.Array
    stage_advances: jax.Array
    stage_delivered_food: jax.Array
    newly_visited_cells: jax.Array
    visited_cell_count: jax.Array
    visited_cell_fraction: jax.Array
    newly_viewed_cells: jax.Array
    viewed_cell_count: jax.Array
    viewed_cell_fraction: jax.Array
    visible_border_cells: jax.Array
    border_moat_cost: jax.Array
    nonzero_byte_tiles: jax.Array
    nonzero_byte_fraction: jax.Array
    applied_nonzero_write_actions: jax.Array
    empty_nonzero_write_actions: jax.Array
    carrying_nonzero_write_actions: jax.Array
    empty_write_action_slots: jax.Array
    carrying_write_action_slots: jax.Array
    write_attempts: jax.Array
    overwrite_events: jax.Array
    reset_hub_pos: jax.Array
    reset_food_positions: jax.Array


class Rollout(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    agent_masks: jax.Array
    logprobs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    values: jax.Array
    next_values: jax.Array
    env_rewards: jax.Array
    pickup_events: jax.Array
    delivery_events: jax.Array
    carrying_ants: jax.Array
    remaining_food: jax.Array
    remaining_lethal_food: jax.Array
    death_events: jax.Array
    alive_ant_count: jax.Array
    dead_ant_count: jax.Array
    active_size: jax.Array
    stage_advances: jax.Array
    stage_delivered_food: jax.Array
    newly_visited_cells: jax.Array
    visited_cell_count: jax.Array
    visited_cell_fraction: jax.Array
    newly_viewed_cells: jax.Array
    viewed_cell_count: jax.Array
    viewed_cell_fraction: jax.Array
    visible_border_cells: jax.Array
    border_moat_cost: jax.Array
    nonzero_byte_tiles: jax.Array
    nonzero_byte_fraction: jax.Array
    applied_nonzero_write_actions: jax.Array
    empty_nonzero_write_actions: jax.Array
    carrying_nonzero_write_actions: jax.Array
    empty_write_action_slots: jax.Array
    carrying_write_action_slots: jax.Array
    write_attempts: jax.Array
    overwrite_events: jax.Array
    reset_hub_pos: jax.Array
    reset_food_positions: jax.Array


class TrainingBatch(NamedTuple):
    actor_obs: jax.Array
    central_obs: jax.Array
    actions: jax.Array
    agent_masks: jax.Array
    old_logprobs: jax.Array
    advantages: jax.Array
    returns: jax.Array


class UpdateMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    behavior_anchor_kl: jax.Array
    clipfrac: jax.Array
    grad_norm: jax.Array
    actor_grad_norm: jax.Array
    critic_grad_norm: jax.Array


class GradientNorms(NamedTuple):
    global_norm: jax.Array
    actor_norm: jax.Array
    critic_norm: jax.Array
