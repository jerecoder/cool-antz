import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ant_byte_env import ACTION_LEFT, ACTION_RIGHT, ACTION_STAY
from ant_byte_env.cli import main as cli_main
from ant_byte_env.training.jax_mappo.checkpointing import save_checkpoint
from ant_byte_env.training.jax_mappo.models import init_agent_params
from ant_byte_env.training.jax_mappo.observations import (
    build_actor_observations,
    build_central_observations,
)
from ant_byte_env.training.jax_mappo.timed_release import runner as timed_runner
from ant_byte_env.training.jax_mappo.timed_release.cli import parse_args
from ant_byte_env.training.jax_mappo.timed_release.env import (
    TimedReleaseJaxEnv,
    make_timed_release_env,
)
from ant_byte_env.training.jax_mappo.timed_release.evaluation import evaluate_checkpoint
from ant_byte_env.training.jax_mappo.timed_release.rendering import draw_timed_release_frame
from ant_byte_env.training.jax_mappo.types import TrainingBatch
from ant_byte_env.training.jax_mappo.updates import _ppo_loss, init_adam_state
from ant_byte_env.jax_env import JaxAntByteForagingEnv


def _small_args(**overrides):
    values = {
        "width": 5,
        "height": 5,
        "obs_width": None,
        "obs_height": None,
        "num_ants": 3,
        "food_count": 1,
        "food_sources": 1,
        "max_steps": 10,
        "random_food": False,
        "random_hub": False,
        "random_ant_spawn": False,
        "random_ant_spawn_radius": None,
        "layout_margin": 0,
        "hub_center_window_size": 0,
        "actor_vision_radius": 1,
        "step_penalty": 0.0,
        "completion_bonus": 0.0,
        "write_penalty": 0.0,
        "write_bits": 1,
        "write_while_moving": True,
        "per_ant_write_channels": False,
        "food_termination": True,
        "terminate_on_full_coverage": False,
        "maze_obstacles": False,
        "maze_corridor_width": 3,
        "maze_wall_width": 1,
        "maze_seed": 0,
        "release_interval": 2,
        "initial_active_ants": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_timed_release_env_releases_fixed_ranks_and_hides_inactive_counts() -> None:
    env = make_timed_release_env(_small_args(num_ants=3, release_interval=2))
    state, obs, _ = env.reset(
        jax.random.PRNGKey(0),
        hub_pos=jnp.array([2, 2], dtype=jnp.int32),
        food_positions=jnp.array([[4, 4]], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(obs["active_mask"]), [True, False, False])
    assert int(np.asarray(obs["ants_count"])[2, 2]) == 1

    move_all_right = jnp.array(
        [ACTION_RIGHT, 0, ACTION_RIGHT, 1, ACTION_RIGHT, 1],
        dtype=jnp.int32,
    )
    state, obs, _, _, _, _ = env.step(state, move_all_right)
    np.testing.assert_array_equal(np.asarray(obs["active_mask"]), [True, False, False])
    np.testing.assert_array_equal(np.asarray(state.base.ants_pos[1]), [2, 2])
    assert int(np.asarray(obs["ants_count"])[2, 2]) == 0

    state, obs, _, _, _, _ = env.step(state, move_all_right)
    np.testing.assert_array_equal(np.asarray(obs["active_mask"]), [True, True, False])
    np.testing.assert_array_equal(np.asarray(state.base.ants_pos[1]), [2, 2])
    assert int(np.asarray(obs["ants_count"])[2, 2]) == 1


def test_timed_release_active_ant_can_pickup_write_and_deliver() -> None:
    env = make_timed_release_env(_small_args(num_ants=2, release_interval=100))
    state, obs, _ = env.reset(
        jax.random.PRNGKey(1),
        hub_pos=jnp.array([2, 2], dtype=jnp.int32),
        food_positions=jnp.array([[3, 2]], dtype=jnp.int32),
    )

    state, obs, _, _, _, info = env.step(
        state,
        jnp.array([ACTION_RIGHT, 1, ACTION_RIGHT, 1], dtype=jnp.int32),
    )
    assert bool(np.asarray(state.base.ants_carrying[0]))
    assert float(np.asarray(info.pickup_events_per_ant[0])) == 1.0
    assert int(np.asarray(info.num_writes)) == 0

    state, obs, _, terminated, _, info = env.step(
        state,
        jnp.array([ACTION_LEFT, 1, ACTION_RIGHT, 1], dtype=jnp.int32),
    )
    assert int(np.asarray(state.base.delivered_food)) == 1
    assert float(np.asarray(info.delivery_events_per_ant[0])) == 1.0
    assert bool(np.asarray(terminated))


def test_timed_release_all_ants_active_by_expected_step() -> None:
    env = make_timed_release_env(_small_args(num_ants=8, release_interval=150))
    assert int(np.asarray(jnp.sum(env.active_mask_for_step(jnp.asarray(1049))))) == 7
    assert int(np.asarray(jnp.sum(env.active_mask_for_step(jnp.asarray(1050))))) == 8


def test_agent_masks_exclude_inactive_policy_slots_from_ppo_loss() -> None:
    params = init_agent_params(
        jax.random.PRNGKey(0),
        central_obs_dim=3,
        actor_obs_dim=4,
        hidden_size=8,
        write_value_count=2,
    )
    args = argparse.Namespace(
        norm_adv=False,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        training_rollout_temperature=1.0,
        critic_architecture="mlp",
        critic_num_ants=2,
        critic_obs_height=5,
        critic_obs_width=5,
    )
    base = TrainingBatch(
        actor_obs=jnp.zeros((2, 2, 4), dtype=jnp.float32),
        central_obs=jnp.zeros((2, 3), dtype=jnp.float32),
        actions=jnp.zeros((2, 2, 2), dtype=jnp.int32),
        agent_masks=jnp.array([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        old_logprobs=jnp.zeros((2, 2), dtype=jnp.float32),
        advantages=jnp.ones((2,), dtype=jnp.float32),
        returns=jnp.zeros((2,), dtype=jnp.float32),
    )
    changed_inactive = base._replace(
        actor_obs=base.actor_obs.at[:, 1, :].set(99.0),
        actions=base.actions.at[:, 1, :].set(jnp.array([ACTION_RIGHT, 1])),
        old_logprobs=base.old_logprobs.at[:, 1].set(42.0),
    )

    loss_a, metrics_a = _ppo_loss(params, base, args=args)
    loss_b, metrics_b = _ppo_loss(params, changed_inactive, args=args)

    assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-6)
    assert float(metrics_a.entropy) == pytest.approx(float(metrics_b.entropy), abs=1e-6)
    assert float(metrics_a.approx_kl) == pytest.approx(float(metrics_b.approx_kl), abs=1e-6)


def test_timed_release_config_dry_run_validates_workflow(capsys) -> None:
    exit_code = cli_main(
        [
            "train",
            "jax",
            "--config",
            "experiments/timed_release_roles_8ants_shared_writes.json",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["workflow"] == "timed_release_roles"
    assert payload["resolved_args"]["release_interval"] == 150
    assert payload["resolved_args"]["critic_architecture"] == "strided_cnn"
    assert payload["resolved_args"]["actor_only_warm_start"] is False
    assert payload["release_schedule"]["release_steps"][-1] == 1050


def test_tiny_timed_release_training_and_eval_from_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "source.pkl"
    args = parse_args(
        [
            "--total-timesteps",
            "4",
            "--num-envs",
            "1",
            "--num-steps",
            "2",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "2",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "6",
            "--write-bits",
            "1",
            "--hidden-size",
            "8",
            "--release-interval",
            "2",
            "--initial-active-ants",
            "1",
            "--critic-architecture",
            "mlp",
            "--quiet",
        ]
    )
    env = make_timed_release_env(args)
    state, obs, _ = env.reset(jax.random.PRNGKey(3))
    obs = {name: jnp.expand_dims(value, axis=0) for name, value in obs.items()}
    central_obs = build_central_observations(obs, food_scale=1, write_bits=1)
    actor_obs = build_actor_observations(obs, food_scale=1, actor_vision_radius=1, write_bits=1)
    params = init_agent_params(
        jax.random.PRNGKey(4),
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        hidden_size=8,
        write_value_count=2,
    )
    save_checkpoint(
        checkpoint_path,
        params=params,
        opt_state=init_adam_state(params),
        args=args,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
        run_name="tiny_timed_release_source",
        metrics={},
    )

    metrics = timed_runner.main(
        [
            "--total-timesteps",
            "4",
            "--num-envs",
            "1",
            "--num-steps",
            "2",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--width",
            "4",
            "--height",
            "4",
            "--num-ants",
            "2",
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "6",
            "--write-bits",
            "1",
            "--hidden-size",
            "8",
            "--release-interval",
            "2",
            "--initial-active-ants",
            "1",
            "--critic-architecture",
            "mlp",
            "--load-model",
            str(checkpoint_path),
            "--save-model",
            str(tmp_path / "trained.pkl"),
            "--quiet",
        ]
    )

    assert "mean_active_ants" in metrics
    eval_metrics = evaluate_checkpoint(tmp_path / "trained.pkl", num_episodes=1)
    assert "eval_rank_0_mean_pickups" in eval_metrics
    assert "eval_rank_1_mean_release_to_pickup_latency" in eval_metrics


def test_timed_release_frame_hides_unreleased_ant() -> None:
    obs = {
        "food": np.zeros((1, 4, 4), dtype=np.int32),
        "bytes": np.zeros((1, 4, 4), dtype=np.uint8),
        "obstacles": np.zeros((1, 4, 4), dtype=np.int8),
        "hub_pos": np.array([[0, 0]], dtype=np.int32),
        "ants_pos": np.array([[[1, 1], [3, 3]]], dtype=np.int32),
        "ants_carrying": np.zeros((1, 2), dtype=np.int8),
        "ants_facing": np.full((1, 2), ACTION_RIGHT, dtype=np.int32),
        "active_mask": np.array([[True, False]], dtype=bool),
    }
    moved_inactive = {**obs, "ants_pos": np.array([[[1, 1], [2, 3]]], dtype=np.int32)}
    active_inactive = {**obs, "active_mask": np.array([[True, True]], dtype=bool)}

    hidden_a = draw_timed_release_frame(obs, tile_size=10, write_bits=1)
    hidden_b = draw_timed_release_frame(moved_inactive, tile_size=10, write_bits=1)
    visible_b = draw_timed_release_frame(active_inactive, tile_size=10, write_bits=1)

    np.testing.assert_array_equal(hidden_a, hidden_b)
    assert not np.array_equal(hidden_a, visible_b)
