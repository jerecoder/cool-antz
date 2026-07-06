"""Compatibility facade for reusable pure JAX MAPPO pieces.

The implementation is split by ownership across sibling modules. Keep importing
from this module when compatibility with older notebooks or scripts matters.
"""

from __future__ import annotations

from ant_byte_env.training.jax_mappo.models import (
    _activation,
    _conv2d,
    _critic_entity_dim,
    _forward_body,
    _forward_resnet_cnn_critic,
    _forward_strided_cnn_critic,
    _forward_structured_mlp_critic,
    _linear,
    _require_cnn_critic_field,
    _require_structured_mlp_critic_field,
    _residual_block,
    _split_central_observation_for_cnn,
    _split_central_observation_for_structured_mlp,
    _strided_cnn_flatten_dim,
    _strided_cnn_output_size,
    central_obs_dim_with_ants_count,
    critic_forward_kwargs_from_args,
    get_action_logits,
    get_value,
    init_agent_params,
    init_conv_layer,
    init_layer,
    init_residual_block,
    init_resnet_cnn_critic,
    init_strided_cnn_critic,
    init_structured_mlp_critic,
)
from ant_byte_env.training.jax_mappo.observations import (
    _active_grid_size,
    _agent_identity_features,
    _ants_count_grid,
    _ants_facing_or_default,
    _facing_one_hot,
    _grid_limit_for_positions,
    _normalize_positions,
    _pad_grid,
    _resolve_observation_grid_shape,
    build_actor_observations,
    build_central_observations,
    build_forward_vision_offsets,
    build_local_border_patches,
    build_local_byte_bit_patches,
    build_local_grid_patches,
    build_local_hub_patches,
    flatten_agent_actions,
    food_observation_scale,
)
from ant_byte_env.training.jax_mappo.policy import (
    _categorical_entropy,
    _categorical_log_prob,
    _logits_for_policy_temperature,
    evaluate_actions,
    get_action_and_value,
)
from ant_byte_env.training.jax_mappo.rewards import (
    _applied_write_values,
    _carrying_hub_distance_progress,
    _forage_distance_progress,
    _grid_values_at_positions,
    _hub_distances,
    _nearest_food_distances,
    compute_forage_curriculum_rewards,
    compute_terminal_write_entropy_bonus,
    compute_write_bit_entropy_bonus,
    compute_write_bit_penalties,
)
from ant_byte_env.training.jax_mappo.types import (
    AdamState,
    ConvParams,
    JaxMAPPOParams,
    LinearParams,
    ResidualBlockParams,
    ResNetCriticParams,
    Rollout,
    StridedCNNCriticParams,
    StructuredMLPCriticParams,
    TrainingBatch,
    Transition,
    UpdateMetrics,
)
from ant_byte_env.training.jax_mappo.updates import (
    _flatten_rollout,
    _global_norm,
    _ppo_loss,
    _shuffle_batch,
    _split_minibatches,
    adam_update,
    compute_gae,
    init_adam_state,
    update_agent,
)
