from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ant_byte_env.jax_env import JaxAntByteForagingEnv
from train_mappo_jax import (
    build_actor_observations,
    build_central_observations,
    compute_gae,
    flatten_agent_actions,
    get_action_and_value,
    init_agent_params,
    main,
    parse_args,
    write_value_count,
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


def test_jax_observation_builders_match_mappo_shapes() -> None:
    obs = _batched_reset_obs()

    central_obs = build_central_observations(obs, food_scale=3)
    actor_obs = build_actor_observations(obs, food_scale=3)

    assert central_obs.shape == (1, 34)
    assert actor_obs.shape == (1, 2, 101)
    assert bool(jnp.all(central_obs >= 0.0))
    assert bool(jnp.all(central_obs <= 1.0))
    assert bool(jnp.all(actor_obs >= 0.0))
    assert bool(jnp.all(actor_obs <= 1.0))


def test_jax_actor_observation_exposes_border_mask() -> None:
    obs = _batched_reset_obs()

    actor_obs = build_actor_observations(obs, food_scale=3, actor_vision_radius=1)

    assert actor_obs.shape == (1, 2, 37)
    np.testing.assert_allclose(
        np.asarray(actor_obs[0, 0, 27:36]),
        np.array([1, 1, 1, 1, 0, 0, 1, 0, 0.0], dtype=np.float32),
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

    assert actor_obs.shape == (1, 2, 151)
    assert actions.shape == (1, 2, 2)
    assert logprob.shape == (1, 2)
    assert entropy.shape == (1, 2)
    assert value.shape == (1,)
    assert flat_actions.shape == (1, 4)
    assert bool(jnp.all((0 <= actions[..., 0]) & (actions[..., 0] <= 4)))
    assert bool(jnp.all((0 <= actions[..., 1]) & (actions[..., 1] <= 7)))


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
