from __future__ import annotations

import argparse

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ant_byte_env import (
    ACTION_FORWARD,
    ACTION_TURN_LEFT,
    MOVEMENT_ACTION_COUNT,
    actor_vision_patch_size,
)
from ant_byte_env.jax_env import JaxAntByteForagingEnv
from ant_byte_env.training.jax_mappo import (
    LinearParams,
    build_actor_observations,
    build_central_observations,
    build_local_grid_patches,
    collect_rollout,
    compute_gae,
    evaluate_params,
    flatten_agent_actions,
    get_action_and_value,
    init_adam_state,
    init_agent_params,
    load_checkpoint_for_training,
    main,
    parse_args,
    repeated_write_action_indices,
    reset_batch,
    save_checkpoint,
    write_value_count,
)
from ant_byte_env.training.jax_mappo.transfer import (
    actor_obs_dim_for_bits,
    central_obs_dim_with_ants_count,
    legacy_central_obs_dim,
)


def _batched_reset_obs() -> dict[str, jax.Array]:
    env = JaxAntByteForagingEnv(
        width=4,
        height=3,
        num_ants=2,
        food_count=3,
        random_food=False,
    )
    _, obs, _ = env.reset(
        jax.random.PRNGKey(123),
        hub_pos=jnp.array([0, 0], dtype=jnp.int32),
        food_positions=jnp.array([[1, 0]], dtype=jnp.int32),
    )
    return {key: jnp.expand_dims(value, axis=0) for key, value in obs.items()}


def _rollout_args(extra: list[str] | None = None) -> argparse.Namespace:
    return parse_args(
        [
            "--total-timesteps",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "5",
            "--height",
            "3",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--cookie-distance",
            "2",
            "--max-steps",
            "8",
            "--hidden-size",
            "8",
            "--seed",
            "7",
            "--quiet",
            *(extra or []),
        ]
    )


def _params_for_args(args: argparse.Namespace, env: JaxAntByteForagingEnv):
    states, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
    central_obs = build_central_observations(
        obs,
        food_scale=args.food_count,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )
    return params, states, obs


def test_jax_observation_builders_match_mappo_shapes() -> None:
    obs = _batched_reset_obs()

    central_obs = build_central_observations(obs, food_scale=3)
    actor_obs = build_actor_observations(obs, food_scale=3)

    assert central_obs.shape == (1, 46)
    assert actor_obs.shape == (1, 2, 31)
    assert bool(jnp.all(central_obs >= 0.0))
    assert bool(jnp.all(central_obs <= 1.0))
    assert bool(jnp.all(actor_obs >= 0.0))
    assert bool(jnp.all(actor_obs <= 1.0))


def test_jax_observation_builders_preserve_food_amounts() -> None:
    obs = _batched_reset_obs()

    central_obs = build_central_observations(obs, food_scale=3)
    actor_obs = build_actor_observations(obs, food_scale=3, actor_vision_radius=1)

    ants_pos_width = 4
    ants_carrying_width = 2
    ants_count_width = 12
    food_cell_index = 1
    central_food_value = central_obs[
        0,
        ants_pos_width + ants_carrying_width + ants_count_width + food_cell_index,
    ]
    local_food_patch_index = 1

    assert float(central_food_value) == 1.0
    assert float(central_obs[0, ants_pos_width + ants_carrying_width]) == 1.0
    assert float(actor_obs[0, 0, local_food_patch_index]) == 1.0
    np.testing.assert_allclose(
        np.asarray(actor_obs[0, 0, 12:15]),
        np.array([1, 0, 0.0], dtype=np.float32),
    )


def test_jax_actor_observation_exposes_border_mask() -> None:
    obs = _batched_reset_obs()

    actor_obs = build_actor_observations(obs, food_scale=3, actor_vision_radius=1)

    assert actor_obs.shape == (1, 2, 16)
    np.testing.assert_allclose(
        np.asarray(actor_obs[0, 0, 12:15]),
        np.array([1, 0, 0.0], dtype=np.float32),
    )


def test_jax_actor_vision_patch_is_three_wide_and_two_deep_in_front() -> None:
    grid = jnp.asarray(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
                [40.0, 41.0, 42.0, 43.0, 44.0],
            ]
        ],
        dtype=jnp.float32,
    )
    ants_pos = jnp.asarray([[[2, 2]]], dtype=jnp.int32)
    ants_facing = jnp.asarray([[2]], dtype=jnp.int32)

    patch = build_local_grid_patches(
        grid,
        ants_pos,
        radius=2,
        ants_facing=ants_facing,
    )

    assert patch.shape == (1, 1, 6)
    np.testing.assert_allclose(
        np.asarray(patch[0, 0]),
        np.array([13.0, 23.0, 33.0, 14.0, 24.0, 34.0], dtype=np.float32),
    )


def test_jax_agent_samples_joint_actions_for_configured_write_bits() -> None:
    obs = _batched_reset_obs()
    central_obs = build_central_observations(obs, food_scale=3, write_bits=3)
    actor_obs = build_actor_observations(obs, food_scale=3, write_bits=3)
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=16,
        write_value_count=write_value_count(3),
    )

    actions, logprob, entropy, value = get_action_and_value(
        params,
        actor_obs,
        central_obs,
        jax.random.PRNGKey(1),
    )
    flat_actions = flatten_agent_actions(actions)

    assert actor_obs.shape == (1, 2, 43)
    assert actions.shape == (1, 2, 2)
    assert logprob.shape == (1, 2)
    assert entropy.shape == (1, 2)
    assert value.shape == (1,)
    assert flat_actions.shape == (1, 4)
    assert bool(jnp.all((0 <= actions[..., 0]) & (actions[..., 0] < MOVEMENT_ACTION_COUNT)))
    assert bool(jnp.all((0 <= actions[..., 1]) & (actions[..., 1] <= 7)))


def test_jax_checkpoint_transfer_expands_write_bits(tmp_path) -> None:
    source_bits = 1
    target_bits = 3
    radius = 1
    hidden_size = 8
    central_obs_dim = 12
    source_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=source_bits,
        actor_vision_radius=radius,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=target_bits,
        actor_vision_radius=radius,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(source_bits),
    )
    source_path = tmp_path / "one_bit.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=source_bits,
            actor_vision_radius=radius,
            save_model=source_path,
        ),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=source_actor_obs_dim,
        run_name="one_bit",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=target_bits,
        actor_vision_radius=radius,
    )

    transferred = checkpoint["params"]
    assert transferred.actor_body[0].weight.shape == (target_actor_obs_dim, hidden_size)
    assert transferred.write_head.weight.shape[-1] == write_value_count(target_bits)
    np.testing.assert_array_equal(
        np.asarray(repeated_write_action_indices(source_bits, target_bits)),
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.write_head.weight[:, :2]),
        np.asarray(source_params.write_head.weight),
    )
    assert checkpoint["opt_state"].count.shape == ()


def test_jax_checkpoint_transfer_adds_ants_count_planes_from_legacy_checkpoint(tmp_path) -> None:
    write_bits = 1
    radius = 1
    hidden_size = 8
    num_ants = 2
    obs_height = 3
    obs_width = 4
    legacy_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
        include_ants_count=False,
    )
    target_actor_obs_dim = actor_obs_dim_for_bits(
        write_bits=write_bits,
        actor_vision_radius=radius,
    )
    legacy_central_dim = legacy_central_obs_dim(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    target_central_dim = central_obs_dim_with_ants_count(
        num_ants=num_ants,
        obs_height=obs_height,
        obs_width=obs_width,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=legacy_central_dim,
        actor_obs_dim=legacy_actor_obs_dim,
        hidden_size=hidden_size,
        write_value_count=write_value_count(write_bits),
    )
    source_path = tmp_path / "legacy_without_ant_counts.pkl"
    save_checkpoint(
        source_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(
            write_bits=write_bits,
            actor_vision_radius=radius,
            width=obs_width,
            height=obs_height,
            obs_width=obs_width,
            obs_height=obs_height,
            num_ants=num_ants,
            save_model=source_path,
        ),
        central_obs_dim=legacy_central_dim,
        actor_obs_dim=legacy_actor_obs_dim,
        run_name="legacy",
        metrics={},
    )

    checkpoint = load_checkpoint_for_training(
        source_path,
        central_obs_dim=target_central_dim,
        actor_obs_dim=target_actor_obs_dim,
        target_write_bits=write_bits,
        actor_vision_radius=radius,
    )

    transferred = checkpoint["params"]
    patch_size = actor_vision_patch_size(radius)
    grid_area = obs_height * obs_width
    central_prefix = 3 * num_ants
    assert checkpoint["central_obs_dim"] == target_central_dim
    assert checkpoint["actor_obs_dim"] == target_actor_obs_dim
    assert transferred.critic_body[0].weight.shape == (target_central_dim, hidden_size)
    assert transferred.actor_body[0].weight.shape == (target_actor_obs_dim, hidden_size)
    np.testing.assert_allclose(
        np.asarray(transferred.critic_body[0].weight[central_prefix : central_prefix + grid_area]),
        np.zeros((grid_area, hidden_size), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.actor_body[0].weight[patch_size : 2 * patch_size]),
        np.zeros((patch_size, hidden_size), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(transferred.actor_body[0].weight[:patch_size]),
        np.asarray(source_params.actor_body[0].weight[:patch_size]),
    )


def test_jax_gae_respects_done_boundaries() -> None:
    rewards = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    values = jnp.zeros((2, 1), dtype=jnp.float32)
    dones = jnp.array([[False], [True]])
    next_value = jnp.zeros((1,), dtype=jnp.float32)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(np.asarray(advantages), np.array([[3.0], [2.0]]))
    np.testing.assert_allclose(np.asarray(returns), np.array([[3.0], [2.0]]))


def test_jax_rollout_carries_unfinished_state_between_calls() -> None:
    args = _rollout_args()
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)

    states, obs, first_rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(1),
    )
    states, obs, second_rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(2),
    )

    assert not bool(np.asarray(first_rollout.dones)[0, 0])
    assert not bool(np.asarray(second_rollout.dones)[0, 0])
    assert int(np.asarray(states.step_count)[0]) == 2


def test_jax_rollout_auto_resets_completed_envs() -> None:
    args = _rollout_args(
        [
            "--food-count",
            "0",
            "--width",
            "2",
            "--height",
            "1",
            "--cookie-distance",
            "1",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, states, obs = _params_for_args(args, env)

    states, _, rollout = collect_rollout(
        args=args,
        env=env,
        params=params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(3),
    )

    assert bool(np.asarray(rollout.dones)[0, 0])
    assert int(np.asarray(states.step_count)[0]) == 0


def test_jax_evaluate_params_reports_delivery_metrics() -> None:
    args = parse_args(
        [
            "--total-timesteps",
            "2",
            "--num-envs",
            "1",
            "--num-steps",
            "1",
            "--num-minibatches",
            "1",
            "--width",
            "3",
            "--height",
            "1",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--cookie-distance",
            "1",
            "--max-steps",
            "4",
            "--actor-vision-radius",
            "1",
            "--hidden-size",
            "1",
            "--seed",
            "19",
            "--quiet",
        ]
    )
    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        write_bits=args.write_bits,
    )
    params, _, obs = _params_for_args(args, env)
    central_obs = build_central_observations(obs, food_scale=args.food_count)
    actor_obs = build_actor_observations(
        obs,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
    )
    params = _scripted_delivery_params(
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
    )

    metrics = evaluate_params(params=params, args=args, num_episodes=3)

    assert metrics["eval_success_rate"] == 1.0
    assert metrics["eval_mean_delivered_food"] == 1.0
    assert metrics["eval_mean_delivered_fraction"] == 1.0
    assert metrics["eval_mean_episode_return"] == 1.0
    assert metrics["eval_mean_episode_length"] == 4.0


def _scripted_delivery_params(*, central_obs_dim: int, actor_obs_dim: int):
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        hidden_size=1,
        write_value_count=write_value_count(1),
    )
    patch_size = actor_vision_patch_size(1)
    local_food_center_index = patch_size // 2
    local_hub_center_index = patch_size * 3 + patch_size // 2
    actor_input_weight = jnp.zeros_like(params.actor_body[0].weight)
    actor_input_weight = actor_input_weight.at[local_food_center_index, 0].set(10.0)
    actor_input_weight = actor_input_weight.at[local_hub_center_index, 0].set(10.0)
    actor_hidden_weight = jnp.zeros_like(params.actor_body[1].weight).at[0, 0].set(5.0)
    move_weight = jnp.zeros_like(params.move_head.weight).at[0, ACTION_FORWARD].set(5.0)
    move_bias = jnp.zeros_like(params.move_head.bias).at[ACTION_TURN_LEFT].set(2.0)
    return params._replace(
        actor_body=(
            LinearParams(
                weight=actor_input_weight,
                bias=jnp.zeros_like(params.actor_body[0].bias),
            ),
            LinearParams(
                weight=actor_hidden_weight,
                bias=jnp.zeros_like(params.actor_body[1].bias),
            ),
        ),
        move_head=LinearParams(weight=move_weight, bias=move_bias),
    )


@pytest.mark.parametrize("write_bits", ["0", "9"])
def test_jax_parse_args_rejects_invalid_write_bits(write_bits: str) -> None:
    with pytest.raises(ValueError, match="write-bits"):
        parse_args(["--write-bits", write_bits])


def test_tiny_jax_mappo_training_run_completes() -> None:
    progress_updates = []

    metrics = main(
        [
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--num-steps",
            "4",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "1",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "8",
            "--hidden-size",
            "16",
            "--seed",
            "7",
            "--quiet",
        ],
        progress_callback=lambda update, total, row: progress_updates.append(
            (update, total, row["global_step"])
        ),
    )

    assert metrics["global_step"] == 8
    assert progress_updates == [(1, 2, 4.0), (2, 2, 8.0)]
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
