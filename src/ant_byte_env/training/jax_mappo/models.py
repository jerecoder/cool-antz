"""Network initialization and value/logit forward passes for JAX MAPPO."""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env import MOVEMENT_ACTION_COUNT
from ant_byte_env.training.jax_mappo.types import (
    CRITIC_GLOBAL_FEATURE_DIM,
    ConvParams,
    JaxMAPPOParams,
    LinearParams,
    ResidualBlockParams,
    ResNetCriticParams,
    SET_CNN_ANT_FEATURE_DIM,
    SetCNNCriticParams,
    StridedCNNCriticParams,
    StructuredMLPCriticParams,
)

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


def init_conv_layer(
    key: jax.Array,
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    scale: float = np.sqrt(2.0),
) -> ConvParams:
    fan_in = max(float(kernel_size * kernel_size * in_channels), 1.0)
    std = float(scale) / np.sqrt(fan_in)
    return ConvParams(
        kernel=jax.random.normal(
            key,
            (kernel_size, kernel_size, in_channels, out_channels),
            dtype=jnp.float32,
        )
        * std,
        bias=jnp.zeros((out_channels,), dtype=jnp.float32),
    )


def init_residual_block(key: jax.Array, channels: int) -> ResidualBlockParams:
    first_key, second_key = jax.random.split(key)
    return ResidualBlockParams(
        first=init_conv_layer(first_key, channels, channels),
        second=init_conv_layer(second_key, channels, channels),
    )


def _critic_entity_dim(*, num_ants: int) -> int:
    return 7 * int(num_ants) + CRITIC_GLOBAL_FEATURE_DIM


def central_obs_dim_with_ants_count(
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> int:
    grid_area = int(obs_height) * int(obs_width)
    return 7 * int(num_ants) + 3 * grid_area + CRITIC_GLOBAL_FEATURE_DIM


def _strided_cnn_output_size(size: int) -> int:
    resolved = int(size)
    for _ in range(3):
        resolved = (resolved + 1) // 2
    return max(resolved, 1)


def _strided_cnn_flatten_dim(*, obs_height: int, obs_width: int) -> int:
    return (
        _strided_cnn_output_size(obs_height)
        * _strided_cnn_output_size(obs_width)
        * 64
    )


def init_resnet_cnn_critic(
    key: jax.Array,
    *,
    num_ants: int,
    spatial_channels: int = 4,
) -> tuple[ResNetCriticParams, LinearParams]:
    if num_ants <= 0:
        raise ValueError("critic_num_ants must be positive.")
    keys = jax.random.split(key, 17)
    critic_body = ResNetCriticParams(
        stem=init_conv_layer(keys[0], spatial_channels, 32),
        blocks_32=(
            init_residual_block(keys[1], 32),
            init_residual_block(keys[2], 32),
        ),
        down_64=init_conv_layer(keys[3], 32, 64),
        blocks_64=(
            init_residual_block(keys[4], 64),
            init_residual_block(keys[5], 64),
        ),
        down_96=init_conv_layer(keys[6], 64, 96),
        blocks_96=(
            init_residual_block(keys[7], 96),
            init_residual_block(keys[8], 96),
        ),
        down_128=init_conv_layer(keys[9], 96, 128),
        blocks_128=(init_residual_block(keys[10], 128),),
        spatial_dense=init_layer(keys[11], 256, 256),
        entity_body=(
            init_layer(keys[12], _critic_entity_dim(num_ants=num_ants), 128),
            init_layer(keys[13], 128, 128),
        ),
        fusion_body=(
            init_layer(keys[14], 384, 256),
            init_layer(keys[15], 256, 256),
        ),
    )
    return critic_body, init_layer(keys[16], 256, 1, scale=1.0)


def init_strided_cnn_critic(
    key: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
    spatial_channels: int = 4,
) -> tuple[StridedCNNCriticParams, LinearParams]:
    if num_ants <= 0:
        raise ValueError("critic_num_ants must be positive.")
    if obs_height <= 0 or obs_width <= 0:
        raise ValueError("critic_obs_height and critic_obs_width must be positive.")
    keys = jax.random.split(key, 7)
    critic_body = StridedCNNCriticParams(
        conv_5x5=init_conv_layer(keys[0], spatial_channels, 32, kernel_size=5),
        conv_3x3_a=init_conv_layer(keys[1], 32, 64),
        conv_3x3_b=init_conv_layer(keys[2], 64, 64),
        spatial_dense=init_layer(
            keys[3],
            _strided_cnn_flatten_dim(obs_height=obs_height, obs_width=obs_width),
            256,
        ),
        entity_dense=init_layer(keys[4], _critic_entity_dim(num_ants=num_ants), 128),
        fusion_dense=init_layer(keys[5], 384, 256),
    )
    return critic_body, init_layer(keys[6], 256, 1, scale=1.0)


def init_set_cnn_critic(
    key: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
    spatial_channels: int = 4,
) -> tuple[SetCNNCriticParams, LinearParams]:
    if num_ants <= 0:
        raise ValueError("critic_num_ants must be positive.")
    if obs_height <= 0 or obs_width <= 0:
        raise ValueError("critic_obs_height and critic_obs_width must be positive.")
    keys = jax.random.split(key, 10)
    critic_body = SetCNNCriticParams(
        conv_5x5=init_conv_layer(keys[0], spatial_channels, 32, kernel_size=5),
        conv_3x3_a=init_conv_layer(keys[1], 32, 64),
        conv_3x3_b=init_conv_layer(keys[2], 64, 64),
        spatial_dense=init_layer(
            keys[3],
            _strided_cnn_flatten_dim(obs_height=obs_height, obs_width=obs_width),
            256,
        ),
        ant_encoder=(
            init_layer(keys[4], SET_CNN_ANT_FEATURE_DIM, 64),
            init_layer(keys[5], 64, 64),
        ),
        global_dense=init_layer(keys[6], CRITIC_GLOBAL_FEATURE_DIM, 64),
        fusion_body=(
            init_layer(keys[7], 448, 256),
            init_layer(keys[8], 256, 256),
        ),
    )
    return critic_body, init_layer(keys[9], 256, 1, scale=1.0)


def init_structured_mlp_critic(
    key: jax.Array,
    *,
    grid_feature_dim: int,
    entity_feature_dim: int,
) -> tuple[StructuredMLPCriticParams, LinearParams]:
    if grid_feature_dim <= 0:
        raise ValueError("grid_feature_dim must be positive.")
    if entity_feature_dim <= 0:
        raise ValueError("entity_feature_dim must be positive.")
    keys = jax.random.split(key, 7)
    critic_body = StructuredMLPCriticParams(
        grid_body=(
            init_layer(keys[0], grid_feature_dim, 512),
            init_layer(keys[1], 512, 256),
        ),
        entity_body=(
            init_layer(keys[2], entity_feature_dim, 128),
            init_layer(keys[3], 128, 128),
        ),
        fusion_body=(
            init_layer(keys[4], 384, 256),
            init_layer(keys[5], 256, 256),
        ),
    )
    return critic_body, init_layer(keys[6], 256, 1, scale=1.0)


def init_agent_params(
    key: jax.Array,
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
    hidden_size: int = 128,
    write_value_count: int = 2,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
) -> JaxMAPPOParams:
    if write_value_count <= 0:
        raise ValueError("write_value_count must be positive.")
    architecture = str(critic_architecture)
    if architecture == "mlp":
        keys = jax.random.split(key, 7)
        critic_body: Any = (
            init_layer(keys[4], central_obs_dim, hidden_size),
            init_layer(keys[5], hidden_size, hidden_size),
        )
        value_head = init_layer(keys[6], hidden_size, 1, scale=1.0)
    elif architecture == "resnet_cnn":
        if critic_num_ants is None or critic_obs_height is None or critic_obs_width is None:
            raise ValueError(
                "resnet_cnn critic requires critic_num_ants, critic_obs_height, "
                "and critic_obs_width."
            )
        expected_dim = central_obs_dim_with_ants_count(
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
        if int(central_obs_dim) != expected_dim:
            raise ValueError(
                f"resnet_cnn critic expected central_obs_dim {expected_dim}, "
                f"got {central_obs_dim}."
            )
        keys = jax.random.split(key, 5)
        critic_body, value_head = init_resnet_cnn_critic(
            keys[4],
            num_ants=critic_num_ants,
        )
    elif architecture == "strided_cnn":
        if critic_num_ants is None or critic_obs_height is None or critic_obs_width is None:
            raise ValueError(
                "strided_cnn critic requires critic_num_ants, critic_obs_height, "
                "and critic_obs_width."
            )
        expected_dim = central_obs_dim_with_ants_count(
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
        if int(central_obs_dim) != expected_dim:
            raise ValueError(
                f"strided_cnn critic expected central_obs_dim {expected_dim}, "
                f"got {central_obs_dim}."
            )
        keys = jax.random.split(key, 5)
        critic_body, value_head = init_strided_cnn_critic(
            keys[4],
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
    elif architecture == "set_cnn":
        if critic_num_ants is None or critic_obs_height is None or critic_obs_width is None:
            raise ValueError(
                "set_cnn critic requires critic_num_ants, critic_obs_height, "
                "and critic_obs_width."
            )
        expected_dim = central_obs_dim_with_ants_count(
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
        if int(central_obs_dim) != expected_dim:
            raise ValueError(
                f"set_cnn critic expected central_obs_dim {expected_dim}, "
                f"got {central_obs_dim}."
            )
        keys = jax.random.split(key, 5)
        critic_body, value_head = init_set_cnn_critic(
            keys[4],
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
    elif architecture == "structured_mlp":
        if critic_num_ants is None or critic_obs_height is None or critic_obs_width is None:
            raise ValueError(
                "structured_mlp critic requires critic_num_ants, critic_obs_height, "
                "and critic_obs_width."
            )
        expected_dim = central_obs_dim_with_ants_count(
            num_ants=critic_num_ants,
            obs_height=critic_obs_height,
            obs_width=critic_obs_width,
        )
        if int(central_obs_dim) != expected_dim:
            raise ValueError(
                f"structured_mlp critic expected central_obs_dim {expected_dim}, "
                f"got {central_obs_dim}."
            )
        keys = jax.random.split(key, 5)
        grid_feature_dim = 3 * int(critic_obs_height) * int(critic_obs_width)
        critic_body, value_head = init_structured_mlp_critic(
            keys[4],
            grid_feature_dim=grid_feature_dim,
            entity_feature_dim=_critic_entity_dim(num_ants=critic_num_ants),
        )
    else:
        raise ValueError(
            "critic_architecture must be 'mlp', 'structured_mlp', 'strided_cnn', "
            "'set_cnn', or 'resnet_cnn'."
        )
    return JaxMAPPOParams(
        actor_body=(
            init_layer(keys[0], actor_obs_dim, hidden_size),
            init_layer(keys[1], hidden_size, hidden_size),
        ),
        move_head=init_layer(keys[2], hidden_size, MOVEMENT_ACTION_COUNT, scale=0.01),
        write_head=init_layer(keys[3], hidden_size, write_value_count, scale=0.01),
        critic_body=critic_body,
        value_head=value_head,
    )


def _linear(params: LinearParams, x: jax.Array) -> jax.Array:
    return x @ params.weight + params.bias


def _forward_body(layers: tuple[LinearParams, LinearParams], x: jax.Array) -> jax.Array:
    hidden = jnp.tanh(_linear(layers[0], x))
    return jnp.tanh(_linear(layers[1], hidden))


def _activation(x: jax.Array) -> jax.Array:
    return jax.nn.silu(x)


def _conv2d(params: ConvParams, x: jax.Array, *, stride: int = 1) -> jax.Array:
    output = jax.lax.conv_general_dilated(
        x,
        params.kernel,
        window_strides=(int(stride), int(stride)),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return output + params.bias


def _residual_block(params: ResidualBlockParams, x: jax.Array) -> jax.Array:
    residual = x
    hidden = _activation(_conv2d(params.first, x))
    hidden = _conv2d(params.second, hidden)
    return _activation(hidden + residual)


def _require_cnn_critic_field(
    name: str,
    value: int | None,
    *,
    critic_architecture: str = "resnet_cnn",
) -> int:
    if value is None:
        raise ValueError(f"{critic_architecture} critic requires {name}.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _require_structured_mlp_critic_field(name: str, value: int | None) -> int:
    if value is None:
        raise ValueError(f"structured_mlp critic requires {name}.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _split_central_observation_for_cnn(
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
    critic_architecture: str = "resnet_cnn",
) -> tuple[jax.Array, jax.Array, tuple[int, ...]]:
    leading_shape = central_obs.shape[:-1]
    flat = central_obs.reshape((-1, central_obs.shape[-1]))
    grid_area = int(obs_height) * int(obs_width)
    expected_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if int(central_obs.shape[-1]) != expected_dim:
        raise ValueError(
            f"{critic_architecture} critic expected central_obs_dim {expected_dim}, "
            f"got {central_obs.shape[-1]}."
        )

    ants_pos_width = 2 * int(num_ants)
    ants_carrying_width = int(num_ants)
    ants_facing_width = (MOVEMENT_ACTION_COUNT - 1) * int(num_ants)
    ants_pos_end = ants_pos_width
    ants_carrying_end = ants_pos_end + ants_carrying_width
    ants_facing_end = ants_carrying_end + ants_facing_width
    ants_count_end = ants_facing_end + grid_area
    food_end = ants_count_end + grid_area
    bytes_end = food_end + grid_area
    hub_end = bytes_end + 2

    ants_count = flat[:, ants_facing_end:ants_count_end].reshape(
        (-1, int(obs_height), int(obs_width))
    )
    food = flat[:, ants_count_end:food_end].reshape((-1, int(obs_height), int(obs_width)))
    bytes_grid = flat[:, food_end:bytes_end].reshape(
        (-1, int(obs_height), int(obs_width))
    )
    hub_pos = flat[:, bytes_end:hub_end]
    batch_index = jnp.arange(flat.shape[0])
    hub_x = jnp.clip(
        jnp.rint(hub_pos[:, 0] * max(int(obs_width) - 1, 1)).astype(jnp.int32),
        0,
        int(obs_width) - 1,
    )
    hub_y = jnp.clip(
        jnp.rint(hub_pos[:, 1] * max(int(obs_height) - 1, 1)).astype(jnp.int32),
        0,
        int(obs_height) - 1,
    )
    hub_grid = jnp.zeros(
        (flat.shape[0], int(obs_height), int(obs_width)),
        dtype=jnp.float32,
    ).at[batch_index, hub_y, hub_x].set(1.0)
    spatial = jnp.stack([ants_count, food, bytes_grid, hub_grid], axis=-1)
    entity = jnp.concatenate(
        [
            flat[:, :ants_pos_end],
            flat[:, ants_pos_end:ants_carrying_end],
            flat[:, ants_carrying_end:ants_facing_end],
            flat[:, bytes_end:],
        ],
        axis=-1,
    )
    return spatial, entity, leading_shape


def _split_central_observation_for_structured_mlp(
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> tuple[jax.Array, jax.Array, tuple[int, ...]]:
    leading_shape = central_obs.shape[:-1]
    flat = central_obs.reshape((-1, central_obs.shape[-1]))
    grid_area = int(obs_height) * int(obs_width)
    expected_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if int(central_obs.shape[-1]) != expected_dim:
        raise ValueError(
            f"structured_mlp critic expected central_obs_dim {expected_dim}, "
            f"got {central_obs.shape[-1]}."
        )

    ants_pos_width = 2 * int(num_ants)
    ants_carrying_width = int(num_ants)
    ants_facing_width = (MOVEMENT_ACTION_COUNT - 1) * int(num_ants)
    ants_pos_end = ants_pos_width
    ants_carrying_end = ants_pos_end + ants_carrying_width
    ants_facing_end = ants_carrying_end + ants_facing_width
    ants_count_end = ants_facing_end + grid_area
    food_end = ants_count_end + grid_area
    bytes_end = food_end + grid_area

    grid_features = flat[:, ants_facing_end:bytes_end]
    entity_features = jnp.concatenate(
        [
            flat[:, :ants_facing_end],
            flat[:, bytes_end:],
        ],
        axis=-1,
    )
    return grid_features, entity_features, leading_shape


def _split_central_observation_for_set_cnn(
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[int, ...]]:
    leading_shape = central_obs.shape[:-1]
    flat = central_obs.reshape((-1, central_obs.shape[-1]))
    grid_area = int(obs_height) * int(obs_width)
    expected_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    if int(central_obs.shape[-1]) != expected_dim:
        raise ValueError(
            f"set_cnn critic expected central_obs_dim {expected_dim}, "
            f"got {central_obs.shape[-1]}."
        )

    ants_pos_width = 2 * int(num_ants)
    ants_carrying_width = int(num_ants)
    ants_facing_width = (MOVEMENT_ACTION_COUNT - 1) * int(num_ants)
    ants_pos_end = ants_pos_width
    ants_carrying_end = ants_pos_end + ants_carrying_width
    ants_facing_end = ants_carrying_end + ants_facing_width
    ants_count_end = ants_facing_end + grid_area
    food_end = ants_count_end + grid_area
    bytes_end = food_end + grid_area
    hub_end = bytes_end + 2
    global_end = bytes_end + CRITIC_GLOBAL_FEATURE_DIM

    ants_count = flat[:, ants_facing_end:ants_count_end].reshape(
        (-1, int(obs_height), int(obs_width))
    )
    food = flat[:, ants_count_end:food_end].reshape((-1, int(obs_height), int(obs_width)))
    bytes_grid = flat[:, food_end:bytes_end].reshape(
        (-1, int(obs_height), int(obs_width))
    )
    hub_pos = flat[:, bytes_end:hub_end]
    batch_index = jnp.arange(flat.shape[0])
    hub_x = jnp.clip(
        jnp.rint(hub_pos[:, 0] * max(int(obs_width) - 1, 1)).astype(jnp.int32),
        0,
        int(obs_width) - 1,
    )
    hub_y = jnp.clip(
        jnp.rint(hub_pos[:, 1] * max(int(obs_height) - 1, 1)).astype(jnp.int32),
        0,
        int(obs_height) - 1,
    )
    hub_grid = jnp.zeros(
        (flat.shape[0], int(obs_height), int(obs_width)),
        dtype=jnp.float32,
    ).at[batch_index, hub_y, hub_x].set(1.0)
    spatial = jnp.stack([ants_count, food, bytes_grid, hub_grid], axis=-1)
    ant_features = jnp.concatenate(
        [
            flat[:, :ants_pos_end].reshape((-1, int(num_ants), 2)),
            flat[:, ants_pos_end:ants_carrying_end].reshape((-1, int(num_ants), 1)),
            flat[:, ants_carrying_end:ants_facing_end].reshape(
                (-1, int(num_ants), MOVEMENT_ACTION_COUNT - 1)
            ),
        ],
        axis=-1,
    )
    global_features = flat[:, bytes_end:global_end]
    return spatial, ant_features, global_features, leading_shape


def _forward_resnet_cnn_critic(
    critic_body: ResNetCriticParams,
    value_head: LinearParams,
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> jax.Array:
    spatial, entity, leading_shape = _split_central_observation_for_cnn(
        central_obs,
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    hidden = _activation(_conv2d(critic_body.stem, spatial))
    for block in critic_body.blocks_32:
        hidden = _residual_block(block, hidden)
    hidden = _activation(_conv2d(critic_body.down_64, hidden, stride=2))
    for block in critic_body.blocks_64:
        hidden = _residual_block(block, hidden)
    hidden = _activation(_conv2d(critic_body.down_96, hidden, stride=2))
    for block in critic_body.blocks_96:
        hidden = _residual_block(block, hidden)
    hidden = _activation(_conv2d(critic_body.down_128, hidden, stride=2))
    for block in critic_body.blocks_128:
        hidden = _residual_block(block, hidden)

    pooled = jnp.concatenate(
        [
            jnp.mean(hidden, axis=(1, 2)),
            jnp.max(hidden, axis=(1, 2)),
        ],
        axis=-1,
    )
    spatial_embedding = _activation(_linear(critic_body.spatial_dense, pooled))
    entity_embedding = _activation(_linear(critic_body.entity_body[0], entity))
    entity_embedding = _activation(_linear(critic_body.entity_body[1], entity_embedding))
    fused = jnp.concatenate([spatial_embedding, entity_embedding], axis=-1)
    fused = _activation(_linear(critic_body.fusion_body[0], fused))
    fused = _activation(_linear(critic_body.fusion_body[1], fused))
    value = jnp.squeeze(_linear(value_head, fused), axis=-1)
    return value.reshape(leading_shape)


def _forward_strided_cnn_critic(
    critic_body: StridedCNNCriticParams,
    value_head: LinearParams,
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> jax.Array:
    spatial, entity, leading_shape = _split_central_observation_for_cnn(
        central_obs,
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
        critic_architecture="strided_cnn",
    )
    hidden = _activation(_conv2d(critic_body.conv_5x5, spatial, stride=2))
    hidden = _activation(_conv2d(critic_body.conv_3x3_a, hidden, stride=2))
    hidden = _activation(_conv2d(critic_body.conv_3x3_b, hidden, stride=2))

    spatial_features = hidden.reshape((hidden.shape[0], -1))
    spatial_embedding = _activation(
        _linear(critic_body.spatial_dense, spatial_features)
    )
    entity_embedding = _activation(_linear(critic_body.entity_dense, entity))
    fused = jnp.concatenate([spatial_embedding, entity_embedding], axis=-1)
    fused = _activation(_linear(critic_body.fusion_dense, fused))
    value = jnp.squeeze(_linear(value_head, fused), axis=-1)
    return value.reshape(leading_shape)


def _forward_structured_mlp_critic(
    critic_body: StructuredMLPCriticParams,
    value_head: LinearParams,
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> jax.Array:
    grid_features, entity_features, leading_shape = (
        _split_central_observation_for_structured_mlp(
            central_obs,
            num_ants=num_ants,
            obs_height=obs_height,
            obs_width=obs_width,
        )
    )
    grid_embedding = _forward_body(critic_body.grid_body, grid_features)
    entity_embedding = _forward_body(critic_body.entity_body, entity_features)
    fused = jnp.concatenate([grid_embedding, entity_embedding], axis=-1)
    fused = _forward_body(critic_body.fusion_body, fused)
    value = jnp.squeeze(_linear(value_head, fused), axis=-1)
    return value.reshape(leading_shape)


def _forward_set_cnn_critic(
    critic_body: SetCNNCriticParams,
    value_head: LinearParams,
    central_obs: jax.Array,
    *,
    num_ants: int,
    obs_height: int,
    obs_width: int,
) -> jax.Array:
    spatial, ant_features, global_features, leading_shape = (
        _split_central_observation_for_set_cnn(
            central_obs,
            num_ants=num_ants,
            obs_height=obs_height,
            obs_width=obs_width,
        )
    )
    hidden = _activation(_conv2d(critic_body.conv_5x5, spatial, stride=2))
    hidden = _activation(_conv2d(critic_body.conv_3x3_a, hidden, stride=2))
    hidden = _activation(_conv2d(critic_body.conv_3x3_b, hidden, stride=2))
    spatial_features = hidden.reshape((hidden.shape[0], -1))
    spatial_embedding = _activation(
        _linear(critic_body.spatial_dense, spatial_features)
    )

    ant_hidden = _activation(_linear(critic_body.ant_encoder[0], ant_features))
    ant_hidden = _activation(_linear(critic_body.ant_encoder[1], ant_hidden))
    ant_embedding = jnp.concatenate(
        [
            jnp.mean(ant_hidden, axis=1),
            jnp.max(ant_hidden, axis=1),
        ],
        axis=-1,
    )
    global_embedding = _activation(_linear(critic_body.global_dense, global_features))
    fused = jnp.concatenate(
        [spatial_embedding, ant_embedding, global_embedding],
        axis=-1,
    )
    fused = _activation(_linear(critic_body.fusion_body[0], fused))
    fused = _activation(_linear(critic_body.fusion_body[1], fused))
    value = jnp.squeeze(_linear(value_head, fused), axis=-1)
    return value.reshape(leading_shape)


def critic_forward_kwargs_from_args(args: argparse.Namespace) -> dict[str, int | str]:
    architecture = str(getattr(args, "critic_architecture", "mlp"))
    if architecture == "mlp":
        return {}
    if architecture not in {"structured_mlp", "strided_cnn", "set_cnn", "resnet_cnn"}:
        raise ValueError(
            "critic_architecture must be 'mlp', 'structured_mlp', 'strided_cnn', "
            "'set_cnn', or 'resnet_cnn'."
        )
    return {
        "critic_architecture": architecture,
        "critic_num_ants": int(args.num_ants),
        "critic_obs_height": int(args.obs_height or args.height),
        "critic_obs_width": int(args.obs_width or args.width),
    }


def get_action_logits(
    params: JaxMAPPOParams,
    actor_obs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    hidden = _forward_body(params.actor_body, actor_obs)
    return _linear(params.move_head, hidden), _linear(params.write_head, hidden)


def get_value(
    params: JaxMAPPOParams,
    central_obs: jax.Array,
    *,
    critic_architecture: str = "mlp",
    critic_num_ants: int | None = None,
    critic_obs_height: int | None = None,
    critic_obs_width: int | None = None,
) -> jax.Array:
    architecture = str(critic_architecture)
    if architecture == "mlp":
        hidden = _forward_body(params.critic_body, central_obs)
        return jnp.squeeze(_linear(params.value_head, hidden), axis=-1)
    if architecture == "resnet_cnn":
        return _forward_resnet_cnn_critic(
            params.critic_body,
            params.value_head,
            central_obs,
            num_ants=_require_cnn_critic_field("critic_num_ants", critic_num_ants),
            obs_height=_require_cnn_critic_field("critic_obs_height", critic_obs_height),
            obs_width=_require_cnn_critic_field("critic_obs_width", critic_obs_width),
        )
    if architecture == "strided_cnn":
        return _forward_strided_cnn_critic(
            params.critic_body,
            params.value_head,
            central_obs,
            num_ants=_require_cnn_critic_field(
                "critic_num_ants",
                critic_num_ants,
                critic_architecture="strided_cnn",
            ),
            obs_height=_require_cnn_critic_field(
                "critic_obs_height",
                critic_obs_height,
                critic_architecture="strided_cnn",
            ),
            obs_width=_require_cnn_critic_field(
                "critic_obs_width",
                critic_obs_width,
                critic_architecture="strided_cnn",
            ),
        )
    if architecture == "set_cnn":
        return _forward_set_cnn_critic(
            params.critic_body,
            params.value_head,
            central_obs,
            num_ants=_require_cnn_critic_field(
                "critic_num_ants",
                critic_num_ants,
                critic_architecture="set_cnn",
            ),
            obs_height=_require_cnn_critic_field(
                "critic_obs_height",
                critic_obs_height,
                critic_architecture="set_cnn",
            ),
            obs_width=_require_cnn_critic_field(
                "critic_obs_width",
                critic_obs_width,
                critic_architecture="set_cnn",
            ),
        )
    if architecture == "structured_mlp":
        return _forward_structured_mlp_critic(
            params.critic_body,
            params.value_head,
            central_obs,
            num_ants=_require_structured_mlp_critic_field(
                "critic_num_ants",
                critic_num_ants,
            ),
            obs_height=_require_structured_mlp_critic_field(
                "critic_obs_height",
                critic_obs_height,
            ),
            obs_width=_require_structured_mlp_critic_field(
                "critic_obs_width",
                critic_obs_width,
            ),
        )
    raise ValueError(
        "critic_architecture must be 'mlp', 'structured_mlp', 'strided_cnn', "
        "'set_cnn', or 'resnet_cnn'."
    )
