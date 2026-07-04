"""Shared construction helpers for adversarial JAX MAPPO."""

from __future__ import annotations

from typing import Any

import jax

from ant_byte_env import write_value_count
from ant_byte_env.training.jax_mappo.adversarial.env import JaxAdversarialAntByteEnv
from ant_byte_env.training.jax_mappo.models import init_agent_params
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams


def make_env(args: Any) -> JaxAdversarialAntByteEnv:
    return JaxAdversarialAntByteEnv(
        width=args.width,
        height=args.height,
        num_ants_per_team=args.num_ants_per_team,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        random_hub=args.random_hub,
        layout_margin=args.layout_margin,
        hub_center_window_size=args.hub_center_window_size,
        hub_pair_distance=args.hub_pair_distance,
        hub_pair_distance_min=args.hub_pair_distance_min,
        hub_pair_distance_max=args.hub_pair_distance_max,
        food_midpoint_window_size=args.food_midpoint_window_size,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        write_while_moving=args.write_while_moving,
        terminate_on_food_delivery=args.food_termination,
        delivery_limit=args.delivery_limit,
    )


def init_adversarial_params(
    key: jax.Array,
    *,
    args: Any,
    central_obs_dim: int,
    actor_obs_dim: int,
) -> JaxMAPPOParams:
    return init_agent_params(
        key,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
        critic_architecture="mlp",
    )
