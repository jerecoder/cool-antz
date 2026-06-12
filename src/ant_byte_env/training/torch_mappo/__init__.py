"""Torch MAPPO training utilities for AntByte."""

from ant_byte_env.training.torch_mappo.checkpointing import checkpoint_args, load_agent_checkpoint
from ant_byte_env.training.torch_mappo.cli import parse_args
from ant_byte_env.training.torch_mappo.curriculum import (
    build_curriculum_reset_options,
    compute_forage_curriculum_rewards,
)
from ant_byte_env.training.torch_mappo.evaluation import (
    evaluate_agent,
    evaluate_checkpoint,
    mastery_reached,
)
from ant_byte_env.training.torch_mappo.model import (
    JointMoveWriteCategorical,
    MAPPOAgent,
    MAPPOActorAdapter,
    MAPPOCriticAdapter,
    layer_init,
    make_mappo_loss,
    make_torchrl_actor,
    make_torchrl_critic,
)
from ant_byte_env.training.torch_mappo.observations import (
    NumpyObs,
    TensorObs,
    build_actor_observations,
    build_central_observations,
    build_local_border_patches,
    build_local_byte_bit_patches,
    build_local_food_patches,
    build_local_grid_patches,
    build_local_hub_patches,
    flatten_agent_actions,
    obs_to_tensor,
)
from ant_byte_env.training.torch_mappo.rollout import (
    collect_rollout,
    make_envs,
    make_rollout_storage,
    reset_env,
    rollout_storage_to_tensordict,
    stack_obs,
    update_agent,
)
from ant_byte_env.training.torch_mappo.runner import main
from ant_byte_env.training.torch_mappo.visualization import (
    DEFAULT_VISION_COLORS,
    draw_vision_squares,
)
from ant_byte_env import write_value_count

__all__ = [
    "DEFAULT_VISION_COLORS",
    "JointMoveWriteCategorical",
    "MAPPOActorAdapter",
    "MAPPOAgent",
    "MAPPOCriticAdapter",
    "NumpyObs",
    "TensorObs",
    "build_actor_observations",
    "build_central_observations",
    "build_curriculum_reset_options",
    "build_local_border_patches",
    "build_local_byte_bit_patches",
    "build_local_food_patches",
    "build_local_grid_patches",
    "build_local_hub_patches",
    "checkpoint_args",
    "collect_rollout",
    "compute_forage_curriculum_rewards",
    "draw_vision_squares",
    "evaluate_agent",
    "evaluate_checkpoint",
    "flatten_agent_actions",
    "layer_init",
    "load_agent_checkpoint",
    "main",
    "make_envs",
    "make_mappo_loss",
    "make_rollout_storage",
    "make_torchrl_actor",
    "make_torchrl_critic",
    "mastery_reached",
    "obs_to_tensor",
    "parse_args",
    "reset_env",
    "rollout_storage_to_tensordict",
    "stack_obs",
    "update_agent",
    "write_value_count",
]
