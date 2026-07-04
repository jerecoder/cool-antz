from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ant_byte_env import ACTION_LEFT, ACTION_RIGHT, ACTION_STAY
from ant_byte_env.cli import main as cli_main
from ant_byte_env.jax_env import JaxAntByteForagingEnv
from ant_byte_env.training.jax_mappo import (
    build_actor_observations,
    init_adam_state,
    init_agent_params,
    save_checkpoint,
    update_agent,
    write_value_count,
)
from ant_byte_env.training.jax_mappo.checkpointing import load_checkpoint, read_checkpoint
from ant_byte_env.training.jax_mappo.models import (
    critic_forward_kwargs_from_args,
    get_value,
)
from ant_byte_env.training.jax_mappo.adversarial.cli import parse_args
from ant_byte_env.training.jax_mappo.adversarial.checkpointing import (
    evaluate_checkpoint_matrix,
    load_checkpoint_for_evaluation,
)
from ant_byte_env.training.jax_mappo.adversarial.actions import actions_from_logits
from ant_byte_env.training.jax_mappo.adversarial.env import (
    JaxAdversarialAntByteEnv,
    reset_batch,
)
from ant_byte_env.training.jax_mappo.adversarial.evaluation import evaluate_matrix
from ant_byte_env.training.jax_mappo.adversarial.observations import (
    build_team_actor_observations,
    build_team_central_observations,
)
from ant_byte_env.training.jax_mappo.adversarial.rollout import (
    collect_rollout,
    compose_team_actions,
)
from ant_byte_env.training.jax_mappo.adversarial.rendering import (
    _group_ants_by_position_and_team,
    draw_adversarial_frame,
    render_adversarial_rollout,
)
from ant_byte_env.training.jax_mappo.adversarial.runner import main as adversarial_main
from ant_byte_env.training.jax_mappo.adversarial.setup import init_adversarial_params
from ant_byte_env.training.jax_mappo.adversarial.transfer import warm_start_actor_params
from ant_byte_env.training.jax_mappo.updates import behavior_anchor_kl


def _env(*, max_steps: int = 10, food_count: int = 2) -> JaxAdversarialAntByteEnv:
    return JaxAdversarialAntByteEnv(
        width=5,
        height=3,
        num_ants_per_team=1,
        food_count=food_count,
        food_source_count=max(food_count, 1),
        max_steps=max_steps,
        random_food=False,
        random_hub=False,
        actor_vision_radius=2,
        write_bits=1,
        write_while_moving=True,
    )


def _reset(env: JaxAdversarialAntByteEnv):
    return env.reset(
        jax.random.PRNGKey(0),
        hub_pos=jnp.array([[0, 1], [4, 1]], dtype=jnp.int32),
        food_positions=jnp.array([[1, 1], [3, 1]], dtype=jnp.int32),
    )


def _action(team0_move: int, team1_move: int = ACTION_STAY) -> jax.Array:
    return jnp.array([team0_move, 0, team1_move, 0], dtype=jnp.int32)


def _batched(obs: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {key: value[None, ...] for key, value in obs.items()}


def _small_argv(extra: list[str] | None = None) -> list[str]:
    return [
        "--allow-random-init",
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
        "--num-ants-per-team",
        "1",
        "--food-count",
        "2",
        "--food-sources",
        "2",
        "--max-steps",
        "4",
        "--hidden-size",
        "8",
        "--quiet",
        *(extra or []),
    ]


def _small_args(extra: list[str] | None = None) -> argparse.Namespace:
    return parse_args(_small_argv(extra))


def _params_for_args(args: argparse.Namespace, env: JaxAdversarialAntByteEnv):
    states, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
    central_obs = build_team_central_observations(
        obs,
        team=args.learner_team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=args.food_count,
        write_bits=args.write_bits,
    )
    actor_obs = build_team_actor_observations(
        obs,
        team=args.learner_team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
    )
    params = init_agent_params(
        jax.random.PRNGKey(3),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )
    opponent_params = init_agent_params(
        jax.random.PRNGKey(4),
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )
    return params, opponent_params, states, obs, central_obs.shape[-1], actor_obs.shape[-1]


def test_adversarial_reset_creates_two_hubs_and_equal_teams() -> None:
    env = _env()
    _, obs, info = _reset(env)

    np.testing.assert_array_equal(np.asarray(obs["hub_pos"]), np.array([[0, 1], [4, 1]]))
    np.testing.assert_array_equal(np.asarray(obs["ants_pos"]), np.array([[0, 1], [4, 1]]))
    assert tuple(np.asarray(info.delivered_food)) == (0, 0)


def test_adversarial_reset_can_control_hub_distance_range_and_midpoint_food() -> None:
    env = JaxAdversarialAntByteEnv(
        width=15,
        height=9,
        num_ants_per_team=1,
        food_count=4,
        food_source_count=2,
        max_steps=10,
        random_food=False,
        random_hub=False,
        hub_pair_distance_min=3,
        hub_pair_distance_max=5,
        food_midpoint_window_size=3,
        actor_vision_radius=2,
        write_bits=1,
    )

    _, obs, _ = env.reset(jax.random.PRNGKey(4))

    hubs = np.asarray(obs["hub_pos"])
    assert int(np.sum(np.abs(hubs[0] - hubs[1]))) == 4
    midpoint = np.mean(hubs, axis=0)
    food_xy = np.argwhere(np.asarray(obs["food"]) > 0)[:, ::-1]
    assert np.all(food_xy[:, 0] >= midpoint[0] - 1)
    assert np.all(food_xy[:, 0] < midpoint[0] + 2)
    assert np.all(food_xy[:, 1] >= midpoint[1] - 1)
    assert np.all(food_xy[:, 1] < midpoint[1] + 2)


def test_adversarial_random_hubs_respect_distance_range() -> None:
    env = JaxAdversarialAntByteEnv(
        width=15,
        height=9,
        num_ants_per_team=1,
        food_count=4,
        food_source_count=2,
        max_steps=10,
        random_food=True,
        random_hub=True,
        hub_pair_distance_min=3,
        hub_pair_distance_max=5,
        food_midpoint_window_size=5,
        actor_vision_radius=2,
        write_bits=1,
    )

    distances = []
    for seed in range(6):
        _, obs, _ = env.reset(jax.random.PRNGKey(seed))
        hubs = np.asarray(obs["hub_pos"])
        distances.append(int(np.sum(np.abs(hubs[0] - hubs[1]))))

    assert min(distances) >= 3
    assert max(distances) <= 5


def test_team_deliveries_increment_only_own_counter_and_rewards_are_opposed() -> None:
    env = _env()
    state, _, _ = _reset(env)

    state, *_ = env.step(state, _action(ACTION_RIGHT))
    state, _, reward, terminated, _, info = env.step(state, _action(ACTION_LEFT))

    np.testing.assert_array_equal(np.asarray(state.delivered_food), np.array([1, 0]))
    np.testing.assert_array_equal(np.asarray(info.delivery_events), np.array([1, 0]))
    np.testing.assert_array_equal(np.asarray(reward), np.array([1.0, -1.0]))
    assert not bool(terminated)


def test_opponent_hub_does_not_accept_learner_delivery() -> None:
    env = _env()
    state, _, _ = _reset(env)

    for move in (ACTION_RIGHT, ACTION_RIGHT, ACTION_RIGHT, ACTION_RIGHT):
        state, _, _, _, _, _ = env.step(state, _action(move))

    np.testing.assert_array_equal(np.asarray(state.ants_pos[0]), np.array([4, 1]))
    np.testing.assert_array_equal(np.asarray(state.delivered_food), np.array([0, 0]))
    assert bool(np.asarray(state.ants_carrying[0]))


def test_termination_waits_for_carried_food_before_food_completion() -> None:
    env = _env(food_count=1)
    state, _, _ = env.reset(
        jax.random.PRNGKey(1),
        hub_pos=jnp.array([[0, 1], [4, 1]], dtype=jnp.int32),
        food_positions=jnp.array([[1, 1]], dtype=jnp.int32),
    )

    state, _, _, terminated, _, _ = env.step(state, _action(ACTION_RIGHT))
    assert not bool(terminated)

    _, _, reward, terminated, _, _ = env.step(state, _action(ACTION_LEFT))
    assert bool(terminated)
    np.testing.assert_array_equal(np.asarray(reward), np.array([1.0, -1.0]))


def test_team_actor_observation_matches_cooperative_actor_shape_and_uses_signed_locals() -> None:
    adversarial_env = _env()
    _, adversarial_obs, _ = _reset(adversarial_env)
    cooperative_env = JaxAntByteForagingEnv(
        width=5,
        height=3,
        num_ants=1,
        food_count=2,
        food_source_count=2,
        actor_vision_radius=4,
        write_bits=1,
    )
    _, cooperative_obs, _ = cooperative_env.reset(
        jax.random.PRNGKey(2),
        hub_pos=jnp.array([0, 1], dtype=jnp.int32),
        food_positions=jnp.array([[1, 1], [3, 1]], dtype=jnp.int32),
    )

    actor_obs = build_team_actor_observations(
        _batched(adversarial_obs),
        team=0,
        num_ants_per_team=1,
        food_scale=2,
        actor_vision_radius=4,
        write_bits=1,
    )
    cooperative_actor_obs = build_actor_observations(
        _batched(cooperative_obs),
        food_scale=2,
        actor_vision_radius=4,
        write_bits=1,
    )

    assert actor_obs.shape == cooperative_actor_obs.shape
    patch_size = 81
    local_ants = np.asarray(actor_obs[0, 0, patch_size : 2 * patch_size])
    local_hubs = np.asarray(actor_obs[0, 0, 3 * patch_size : 4 * patch_size])
    assert local_ants.max() > 0.0
    assert local_ants.min() < 0.0
    assert local_hubs.max() > 0.0
    assert local_hubs.min() < 0.0


def test_team_actor_observation_preserves_cooperative_ant_count_scale() -> None:
    env = JaxAdversarialAntByteEnv(
        width=7,
        height=5,
        num_ants_per_team=8,
        food_count=2,
        food_source_count=2,
        actor_vision_radius=2,
        write_bits=1,
    )
    _, obs, _ = env.reset(
        jax.random.PRNGKey(3),
        hub_pos=jnp.array([[1, 2], [5, 2]], dtype=jnp.int32),
        food_positions=jnp.array([[2, 2], [4, 2]], dtype=jnp.int32),
    )

    actor_obs = build_team_actor_observations(
        _batched(obs),
        team=0,
        num_ants_per_team=8,
        food_scale=2,
        actor_vision_radius=2,
        write_bits=1,
    )

    patch_size = 25
    local_ants = np.asarray(actor_obs[0, 0, patch_size : 2 * patch_size])
    assert np.isclose(local_ants.max(), 1.0)


def test_compose_team_actions_respects_environment_team_order() -> None:
    learner = jnp.array([[[ACTION_RIGHT, 0]]], dtype=jnp.int32)
    opponent = jnp.array([[[ACTION_LEFT, 0]]], dtype=jnp.int32)

    team0_joint = compose_team_actions(learner, opponent, learner_team=0)
    team1_joint = compose_team_actions(learner, opponent, learner_team=1)

    np.testing.assert_array_equal(np.asarray(team0_joint[0]), np.array([[ACTION_RIGHT, 0], [ACTION_LEFT, 0]]))
    np.testing.assert_array_equal(np.asarray(team1_joint[0]), np.array([[ACTION_LEFT, 0], [ACTION_RIGHT, 0]]))


def test_hybrid_adversarial_action_mode_samples_move_and_greedy_write() -> None:
    move_logits = jnp.zeros((1, 1, 5), dtype=jnp.float32)
    write_logits = jnp.array([[[0.0, 1.0, 4.0, 2.0]]], dtype=jnp.float32)

    actions = actions_from_logits(
        move_logits,
        write_logits,
        jax.random.PRNGKey(9),
        action_mode="sampled_move_greedy_write",
        move_temperature=0.75,
    )

    assert actions.shape == (1, 1, 2)
    assert int(actions[0, 0, 1]) == 2


def test_behavior_anchor_kl_uses_move_and_write_heads() -> None:
    args = _small_args(["--training-rollout-temperature", "0.75"])
    env = _env(max_steps=args.max_steps)
    params, _, _, _, _, actor_dim = _params_for_args(args, env)
    actor_obs = jnp.zeros((2, args.num_ants_per_team, actor_dim), dtype=jnp.float32)

    identical_kl = behavior_anchor_kl(
        params,
        params,
        actor_obs,
        policy_temperature=args.training_rollout_temperature,
    )
    move_perturbed = params._replace(
        move_head=params.move_head._replace(
            bias=params.move_head.bias.at[0].add(1.0),
        ),
    )
    write_perturbed = params._replace(
        write_head=params.write_head._replace(
            bias=params.write_head.bias.at[1].add(1.0),
        ),
    )

    move_kl = behavior_anchor_kl(
        params,
        move_perturbed,
        actor_obs,
        policy_temperature=args.training_rollout_temperature,
    )
    write_kl = behavior_anchor_kl(
        params,
        write_perturbed,
        actor_obs,
        policy_temperature=args.training_rollout_temperature,
    )

    assert float(identical_kl) == pytest.approx(0.0, abs=1e-7)
    assert float(move_kl) > 0.0
    assert float(write_kl) > 0.0


def test_update_agent_accepts_behavior_anchor_params_and_reports_kl() -> None:
    args = _small_args()
    args.behavior_anchor_coef = 0.5
    env = _env(max_steps=args.max_steps)
    params, opponent_params, states, obs, _, _ = _params_for_args(args, env)
    anchor_params = params._replace(
        write_head=params.write_head._replace(
            bias=params.write_head.bias.at[1].add(1.0),
        ),
    )
    _, _, rollout = collect_rollout(
        args=args,
        env=env,
        learner_params=params,
        opponent_params=opponent_params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(19),
    )

    _, _, metrics = update_agent(
        args=args,
        params=params,
        opt_state=init_adam_state(params),
        rollout=rollout,
        learning_rate=args.learning_rate,
        key=jax.random.PRNGKey(20),
        behavior_anchor_params=anchor_params,
    )

    assert float(metrics.behavior_anchor_kl) > 0.0


def test_checkpoint_round_trips_behavior_anchor_and_accepts_legacy_payload(
    tmp_path: Path,
) -> None:
    args = _small_args()
    env = _env(max_steps=args.max_steps)
    params, _, _, _, central_dim, actor_dim = _params_for_args(args, env)
    anchor_params = params._replace(
        move_head=params.move_head._replace(
            bias=params.move_head.bias.at[0].add(1.0),
        ),
    )
    checkpoint_path = tmp_path / "anchor.pkl"

    save_checkpoint(
        checkpoint_path,
        params=params,
        opt_state=init_adam_state(params),
        args=args,
        central_obs_dim=central_dim,
        actor_obs_dim=actor_dim,
        run_name="anchored",
        metrics={},
        behavior_anchor_params=anchor_params,
    )

    checkpoint = read_checkpoint(checkpoint_path)
    loaded = load_checkpoint(
        checkpoint_path,
        central_obs_dim=central_dim,
        actor_obs_dim=actor_dim,
    )
    assert checkpoint["behavior_anchor_params"] is not None
    assert loaded["behavior_anchor_params"] is not None
    for actual, expected in zip(
        jax.tree_util.tree_leaves(checkpoint["behavior_anchor_params"]),
        jax.tree_util.tree_leaves(anchor_params),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))

    legacy_path = tmp_path / "legacy.pkl"
    with legacy_path.open("wb") as checkpoint_file:
        pickle.dump(
            {
                "params": jax.tree_util.tree_map(np.asarray, params),
                "opt_state": jax.tree_util.tree_map(np.asarray, init_adam_state(params)),
                "args": {},
                "central_obs_dim": central_dim,
                "actor_obs_dim": actor_dim,
                "run_name": "legacy",
                "metrics": {},
            },
            checkpoint_file,
        )

    legacy_checkpoint = read_checkpoint(legacy_path)
    legacy_loaded = load_checkpoint(
        legacy_path,
        central_obs_dim=central_dim,
        actor_obs_dim=actor_dim,
    )
    assert legacy_checkpoint["behavior_anchor_params"] is None
    assert legacy_loaded["behavior_anchor_params"] is None


def test_adversarial_parse_args_accepts_best_model_options(tmp_path: Path) -> None:
    best_path = tmp_path / "best.pkl"
    anchor_path = tmp_path / "anchor.pkl"

    args = _small_args(
        [
            "--behavior-anchor-coef",
            "0.01",
            "--behavior-anchor-model",
            str(anchor_path),
            "--save-best-model",
            str(best_path),
            "--best-model-metric",
            "delivery_event_difference",
            "--best-model-mode",
            "max",
            "--best-model-selection",
            "eval",
            "--best-eval-episodes",
            "3",
            "--best-eval-interval",
            "5",
        ]
    )

    assert args.behavior_anchor_coef == pytest.approx(0.01)
    assert args.behavior_anchor_model == anchor_path
    assert args.save_best_model == best_path
    assert args.best_model_metric == "delivery_event_difference"
    assert args.best_model_mode == "max"
    assert args.best_model_selection == "eval"
    assert args.best_eval_episodes == 3
    assert args.best_eval_interval == 5


def test_adversarial_parse_args_accepts_cnn_critic_and_wandb_options() -> None:
    args = _small_args(
        [
            "--critic-architecture",
            "strided_cnn",
            "--wandb-project",
            "cool-antz",
            "--wandb-entity",
            "team",
            "--wandb-group",
            "adversarial",
            "--wandb-run-name",
            "cnn-probe",
            "--wandb-notes",
            "tiny adversarial cnn smoke",
            "--wandb-mode",
            "offline",
            "--wandb-tags",
            "jax",
            "cnn",
        ]
    )

    assert args.critic_architecture == "strided_cnn"
    assert args.num_ants == args.num_ants_per_team
    assert args.critic_num_ants == 2 * args.num_ants_per_team
    assert args.critic_extra_entity_dim == 6
    assert args.obs_height == args.height
    assert args.obs_width == args.width
    assert args.wandb_project == "cool-antz"
    assert args.wandb_entity == "team"
    assert args.wandb_group == "adversarial"
    assert args.wandb_run_name == "cnn-probe"
    assert args.wandb_notes == "tiny adversarial cnn smoke"
    assert args.wandb_mode == "offline"
    assert args.wandb_tags == ["jax", "cnn"]


def test_adversarial_strided_cnn_critic_uses_both_teams_shape() -> None:
    args = _small_args(["--critic-architecture", "strided_cnn"])
    env = _env(max_steps=args.max_steps)
    _, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
    central_obs = build_team_central_observations(
        obs,
        team=args.learner_team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=args.food_count,
        write_bits=args.write_bits,
    )
    actor_obs = build_team_actor_observations(
        obs,
        team=args.learner_team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
    )

    params = init_adversarial_params(
        jax.random.PRNGKey(30),
        args=args,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
    )
    values = get_value(params, central_obs, **critic_forward_kwargs_from_args(args))

    assert values.shape == (args.num_envs,)


def test_actor_warm_start_copies_actor_only(tmp_path: Path) -> None:
    args = _small_args()
    env = _env(max_steps=args.max_steps)
    _, _, _, obs, central_dim, actor_dim = _params_for_args(args, env)
    del obs
    source_params = init_agent_params(
        jax.random.PRNGKey(10),
        central_obs_dim=7,
        actor_obs_dim=actor_dim,
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )
    target_params = init_agent_params(
        jax.random.PRNGKey(11),
        central_obs_dim=central_dim,
        actor_obs_dim=actor_dim,
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )
    checkpoint_path = tmp_path / "source.pkl"
    save_checkpoint(
        checkpoint_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(source="cooperative"),
        central_obs_dim=7,
        actor_obs_dim=actor_dim,
        run_name="source",
        metrics={},
    )

    warmed = warm_start_actor_params(
        target_params,
        checkpoint_path,
        actor_obs_dim=actor_dim,
        target_write_bits=args.write_bits,
    )

    for actual, expected in zip(
        jax.tree_util.tree_leaves(warmed.actor_body),
        jax.tree_util.tree_leaves(source_params.actor_body),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    for actual, expected in zip(
        jax.tree_util.tree_leaves(warmed.critic_body),
        jax.tree_util.tree_leaves(target_params.critic_body),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_actor_warm_start_preserves_fresh_cnn_critic(tmp_path: Path) -> None:
    args = _small_args(["--critic-architecture", "strided_cnn"])
    env = _env(max_steps=args.max_steps)
    _, obs = reset_batch(args=args, env=env, key=jax.random.PRNGKey(args.seed))
    central_obs = build_team_central_observations(
        obs,
        team=args.learner_team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=args.food_count,
        write_bits=args.write_bits,
    )
    actor_obs = build_team_actor_observations(
        obs,
        team=args.learner_team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
    )
    source_params = init_agent_params(
        jax.random.PRNGKey(31),
        central_obs_dim=7,
        actor_obs_dim=int(actor_obs.shape[-1]),
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )
    target_params = init_adversarial_params(
        jax.random.PRNGKey(32),
        args=args,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
    )
    checkpoint_path = tmp_path / "mlp_source.pkl"
    save_checkpoint(
        checkpoint_path,
        params=source_params,
        opt_state=init_adam_state(source_params),
        args=argparse.Namespace(source="mlp_source"),
        central_obs_dim=7,
        actor_obs_dim=int(actor_obs.shape[-1]),
        run_name="source",
        metrics={},
    )

    warmed = warm_start_actor_params(
        target_params,
        checkpoint_path,
        actor_obs_dim=int(actor_obs.shape[-1]),
        target_write_bits=args.write_bits,
    )

    for actual, expected in zip(
        jax.tree_util.tree_leaves(warmed.actor_body),
        jax.tree_util.tree_leaves(source_params.actor_body),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    for actual, expected in zip(
        jax.tree_util.tree_leaves(warmed.critic_body),
        jax.tree_util.tree_leaves(target_params.critic_body),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_one_update_adversarial_training_leaves_frozen_opponent_params_unchanged() -> None:
    args = _small_args()
    env = _env(max_steps=args.max_steps)
    params, opponent_params, states, obs, _, _ = _params_for_args(args, env)
    opponent_before = jax.tree_util.tree_map(lambda value: value.copy(), opponent_params)

    _, _, rollout = collect_rollout(
        args=args,
        env=env,
        learner_params=params,
        opponent_params=opponent_params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(20),
    )
    update_agent(
        args=args,
        params=params,
        opt_state=init_adam_state(params),
        rollout=rollout,
        learning_rate=args.learning_rate,
        key=jax.random.PRNGKey(21),
    )

    for actual, expected in zip(
        jax.tree_util.tree_leaves(opponent_params),
        jax.tree_util.tree_leaves(opponent_before),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_freeze_actor_update_only_changes_critic_params() -> None:
    train_args = _small_args()
    freeze_args = _small_args(["--freeze-actor"])
    env = _env(max_steps=train_args.max_steps)
    params, opponent_params, states, obs, _, _ = _params_for_args(train_args, env)

    _, _, rollout = collect_rollout(
        args=train_args,
        env=env,
        learner_params=params,
        opponent_params=opponent_params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(22),
    )
    trained_params, trained_opt_state, _ = update_agent(
        args=train_args,
        params=params,
        opt_state=init_adam_state(params),
        rollout=rollout,
        learning_rate=train_args.learning_rate,
        key=jax.random.PRNGKey(23),
    )
    _, _, freeze_rollout = collect_rollout(
        args=freeze_args,
        env=env,
        learner_params=trained_params,
        opponent_params=opponent_params,
        states=states,
        obs=obs,
        key=jax.random.PRNGKey(24),
    )
    updated_params, updated_opt_state, _ = update_agent(
        args=freeze_args,
        params=trained_params,
        opt_state=trained_opt_state,
        rollout=freeze_rollout,
        learning_rate=freeze_args.learning_rate,
        key=jax.random.PRNGKey(25),
    )

    for actual, expected in zip(
        jax.tree_util.tree_leaves(
            (
                updated_params.actor_body,
                updated_params.move_head,
                updated_params.write_head,
            )
        ),
        jax.tree_util.tree_leaves(
            (trained_params.actor_body, trained_params.move_head, trained_params.write_head)
        ),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    for value in jax.tree_util.tree_leaves(
        (
            updated_opt_state.m.actor_body,
            updated_opt_state.m.move_head,
            updated_opt_state.m.write_head,
            updated_opt_state.v.actor_body,
            updated_opt_state.v.move_head,
            updated_opt_state.v.write_head,
        )
    ):
        np.testing.assert_allclose(np.asarray(value), np.zeros_like(np.asarray(value)))

    critic_changed = any(
        not np.allclose(np.asarray(actual), np.asarray(expected))
        for actual, expected in zip(
            jax.tree_util.tree_leaves((updated_params.critic_body, updated_params.value_head)),
            jax.tree_util.tree_leaves((trained_params.critic_body, trained_params.value_head)),
        )
    )
    assert critic_changed


def test_evaluation_matrix_reports_required_matchups() -> None:
    args = _small_args(["--eval-episodes", "1"])
    env = _env(max_steps=args.max_steps)
    params, opponent_params, *_ = _params_for_args(args, env)
    progress_events = []

    metrics = evaluate_matrix(
        params=params,
        opponent_params=opponent_params,
        args=args,
        env=env,
        progress_callback=lambda name, episode, total, row: progress_events.append(
            (name, episode, total, row)
        ),
    )

    assert "eval_frozen_vs_frozen_mean_delivery_difference" in metrics
    assert "eval_learner_vs_frozen_mean_own_deliveries" in metrics
    assert "eval_frozen_vs_learner_mean_opponent_deliveries" in metrics
    assert "eval_random_vs_frozen_win_rate" in metrics
    assert "eval_learner_vs_random_mean_episode_length" in metrics
    assert "eval_learner_vs_frozen_mean_hub_pair_distance" in metrics
    assert "eval_learner_vs_frozen_mean_food_midpoint_distance" in metrics
    assert "eval_side_swapped_score_gap" in metrics
    assert "eval_learner_vs_frozen_side_swap_adjusted_delivery_difference" in metrics
    assert metrics[
        "eval_learner_vs_frozen_side_swap_adjusted_delivery_difference"
    ] == pytest.approx(
        metrics["eval_learner_vs_frozen_mean_delivery_difference"]
        - 0.5 * metrics["eval_side_swapped_score_gap"]
    )
    assert [event[0] for event in progress_events] == [
        "frozen_vs_frozen",
        "learner_vs_frozen",
        "frozen_vs_learner",
        "random_vs_frozen",
        "learner_vs_random",
    ]
    assert all(event[1] == 1 and event[2] == 1 for event in progress_events)
    assert all("delivery_difference" in event[3] for event in progress_events)


def test_evaluation_matrix_reports_step_progress_when_requested() -> None:
    args = _small_args(["--eval-episodes", "1", "--max-steps", "3"])
    env = _env(max_steps=args.max_steps)
    params, opponent_params, *_ = _params_for_args(args, env)
    progress_events = []

    evaluate_matrix(
        params=params,
        opponent_params=opponent_params,
        args=args,
        env=env,
        progress_callback=lambda name, episode, total, row: progress_events.append(
            (name, episode, total, row)
        ),
        progress_step_interval=1,
    )

    assert any(event[3].get("event") == "step" for event in progress_events)
    assert [event[3].get("event") for event in progress_events if event[3].get("event") == "episode"] == [
        "episode",
        "episode",
        "episode",
        "episode",
        "episode",
    ]
    assert all("max_steps" in event[3] for event in progress_events)


def test_evaluation_matrix_accepts_fixed_layout_positions() -> None:
    args = _small_args(["--eval-episodes", "1"])
    env = _env(max_steps=args.max_steps)
    params, opponent_params, *_ = _params_for_args(args, env)

    metrics = evaluate_matrix(
        params=params,
        opponent_params=opponent_params,
        args=args,
        env=env,
        fixed_hub_positions=[[1, 1], [3, 1]],
        fixed_food_positions=[[2, 1]],
    )

    assert metrics["eval_learner_vs_frozen_mean_hub_pair_distance"] == pytest.approx(2.0)
    assert metrics["eval_learner_vs_frozen_mean_food_midpoint_distance"] == pytest.approx(0.0)


def test_evaluation_uses_actor_only_actions() -> None:
    args = _small_args(["--eval-episodes", "1", "--max-steps", "2"])
    env = _env(max_steps=args.max_steps)
    params, opponent_params, *_ = _params_for_args(args, env)
    broken_critic_params = params._replace(critic_body=())
    broken_opponent_critic_params = opponent_params._replace(critic_body=())

    metrics = evaluate_matrix(
        params=broken_critic_params,
        opponent_params=broken_opponent_critic_params,
        args=args,
        env=env,
    )

    assert "eval_learner_vs_frozen_mean_delivery_difference" in metrics


def test_checkpoint_evaluation_and_render_helpers_use_saved_adversarial_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "adversarial_run"
    argv = _small_argv(["--run-dir", str(run_dir)])

    adversarial_main(argv)

    checkpoint_path = run_dir / "checkpoints" / "model.pkl"
    bundle = load_checkpoint_for_evaluation(checkpoint_path, argv=argv)
    assert bundle.args.run_dir == run_dir
    assert bundle.learner_params is not None
    assert bundle.opponent_params is not None

    metrics = evaluate_checkpoint_matrix(
        checkpoint_path,
        argv=argv,
        eval_episodes=1,
        eval_max_steps=2,
        progress_step_interval=1,
    )
    assert "eval_learner_vs_frozen_mean_delivery_difference" in metrics

    rollout_path = render_adversarial_rollout(
        checkpoint_path,
        tmp_path / "rollout.mp4",
        argv=argv,
        max_frames=3,
        tile_size=8,
    )
    assert rollout_path.exists()
    assert rollout_path.stat().st_size > 0

    fixed_rollout_path = render_adversarial_rollout(
        checkpoint_path,
        tmp_path / "fixed_rollout.mp4",
        argv=argv,
        max_frames=2,
        tile_size=8,
        fixed_hub_positions=[[1, 1], [3, 1]],
        fixed_food_positions=[[2, 1]],
    )
    assert fixed_rollout_path.exists()
    assert fixed_rollout_path.stat().st_size > 0


def test_adversarial_runner_resumes_checkpoint_between_food_stages(tmp_path: Path) -> None:
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    stage1_argv = _small_argv(["--run-dir", str(stage1_dir)])

    adversarial_main(stage1_argv)
    stage1_checkpoint = stage1_dir / "checkpoints" / "model.pkl"

    stage2_argv = [
        arg for arg in _small_argv(
            [
                "--run-dir",
                str(stage2_dir),
                "--resume-model",
                str(stage1_checkpoint),
                "--opponent-load-model",
                str(stage1_checkpoint),
                "--food-count",
                "1",
                "--food-sources",
                "1",
            ]
        )
        if arg != "--allow-random-init"
    ]
    metrics = adversarial_main(stage2_argv)

    assert stage1_checkpoint.exists()
    assert (stage2_dir / "checkpoints" / "model.pkl").exists()
    assert "loss" in metrics


def test_adversarial_runner_saves_best_checkpoint_from_training_metrics(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "model.pkl"
    best_path = tmp_path / "model_best.pkl"

    metrics = adversarial_main(
        _small_argv(
            [
                "--save-model",
                str(final_path),
                "--save-best-model",
                str(best_path),
                "--best-model-metric",
                "episode_return",
            ]
        )
    )

    assert final_path.exists()
    assert best_path.exists()
    checkpoint = read_checkpoint(best_path)
    assert checkpoint["args"]["save_best_model"] == str(best_path)
    assert checkpoint["metrics"]["best_model_selection"] == "train"
    assert checkpoint["metrics"]["best_model_metric_value"] == checkpoint["metrics"][
        "episode_return"
    ]
    assert checkpoint["metrics"]["best_model_update"] in {1.0, 2.0}
    assert metrics["best_model_metric_value"] == checkpoint["metrics"][
        "best_model_metric_value"
    ]


def test_adversarial_runner_can_select_best_checkpoint_by_eval_matrix(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "adversarial_run"
    best_path = run_dir / "checkpoints" / "best.pkl"

    metrics = adversarial_main(
        _small_argv(
            [
                "--run-dir",
                str(run_dir),
                "--save-best-model",
                str(best_path),
                "--best-model-selection",
                "eval",
                "--best-model-metric",
                "eval_learner_vs_frozen_side_swap_adjusted_delivery_difference",
                "--best-eval-episodes",
                "1",
                "--best-eval-interval",
                "1",
            ]
        )
    )

    checkpoint = read_checkpoint(best_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert checkpoint["metrics"]["best_model_selection"] == "eval"
    assert checkpoint["metrics"]["best_model_metric_value"] == checkpoint["metrics"][
        "eval_learner_vs_frozen_side_swap_adjusted_delivery_difference"
    ]
    assert "eval_frozen_vs_frozen_mean_delivery_difference" in checkpoint["metrics"]
    assert summary["best_checkpoint_path"] == str(best_path)
    assert summary["best_checkpoint_selection"] == "eval"
    assert summary["best_checkpoint_metrics"]["best_model_metric_value"] == metrics[
        "best_model_metric_value"
    ]


def test_adversarial_runner_saves_and_resumes_behavior_anchor(
    tmp_path: Path,
) -> None:
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    stage1_argv = _small_argv(
        [
            "--run-dir",
            str(stage1_dir),
            "--behavior-anchor-coef",
            "0.01",
        ]
    )

    adversarial_main(stage1_argv)
    stage1_checkpoint = stage1_dir / "checkpoints" / "model.pkl"
    stage1_payload = read_checkpoint(stage1_checkpoint)
    assert stage1_payload["behavior_anchor_params"] is not None

    stage2_argv = _small_argv(
        [
            "--run-dir",
            str(stage2_dir),
            "--resume-model",
            str(stage1_checkpoint),
            "--opponent-load-model",
            str(stage1_checkpoint),
            "--behavior-anchor-coef",
            "0.01",
        ]
    )
    metrics = adversarial_main(stage2_argv)

    stage2_payload = read_checkpoint(stage2_dir / "checkpoints" / "model.pkl")
    assert stage2_payload["behavior_anchor_params"] is not None
    assert "behavior_anchor_kl" in metrics
    for actual, expected in zip(
        jax.tree_util.tree_leaves(stage2_payload["behavior_anchor_params"]),
        jax.tree_util.tree_leaves(stage1_payload["behavior_anchor_params"]),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_adversarial_frame_uses_shared_sprite_renderer() -> None:
    env = _env()
    state, obs, _ = _reset(env)
    obs = {**obs, "initial_food": state.initial_food}
    frame = draw_adversarial_frame(
        {name: np.asarray(value) for name, value in obs.items()},
        tile_size=12,
        max_write_value=1,
    )

    assert frame.shape == (env.height * 12, env.width * 12, 3)
    assert np.any(np.all(frame == np.array([37, 99, 235], dtype=np.uint8), axis=-1))
    assert np.any(np.all(frame == np.array([220, 38, 38], dtype=np.uint8), axis=-1))
    assert not np.all(frame == frame[0, 0])


def test_adversarial_frame_colors_trained_learner_by_role() -> None:
    tile_size = 14
    obs = {
        "food": np.zeros((4, 4), dtype=np.int32),
        "bytes": np.zeros((4, 4), dtype=np.int32),
        "hub_pos": np.array([[1, 1], [2, 1]], dtype=np.int32),
        "ants_pos": np.array([[1, 2], [2, 2]], dtype=np.int32),
        "ants_carrying": np.zeros((2,), dtype=bool),
        "ants_facing": np.full((2,), ACTION_RIGHT, dtype=np.int32),
    }
    red = np.array([220, 38, 38], dtype=np.uint8)
    blue = np.array([37, 99, 235], dtype=np.uint8)

    learner_team0 = draw_adversarial_frame(
        obs,
        tile_size=tile_size,
        learner_team=0,
        show_role_legend=False,
    )
    team0_tile = learner_team0[
        tile_size : 2 * tile_size,
        tile_size : 2 * tile_size,
    ]
    team1_tile = learner_team0[
        tile_size : 2 * tile_size,
        2 * tile_size : 3 * tile_size,
    ]
    assert np.sum(np.all(team0_tile == red, axis=-1)) > np.sum(
        np.all(team0_tile == blue, axis=-1)
    )
    assert np.sum(np.all(team1_tile == blue, axis=-1)) > np.sum(
        np.all(team1_tile == red, axis=-1)
    )

    learner_team1 = draw_adversarial_frame(
        obs,
        tile_size=tile_size,
        learner_team=1,
        show_role_legend=False,
    )
    team0_tile = learner_team1[
        tile_size : 2 * tile_size,
        tile_size : 2 * tile_size,
    ]
    team1_tile = learner_team1[
        tile_size : 2 * tile_size,
        2 * tile_size : 3 * tile_size,
    ]
    assert np.sum(np.all(team0_tile == blue, axis=-1)) > np.sum(
        np.all(team0_tile == red, axis=-1)
    )
    assert np.sum(np.all(team1_tile == red, axis=-1)) > np.sum(
        np.all(team1_tile == blue, axis=-1)
    )


def test_adversarial_frame_groups_stacked_team_ants() -> None:
    ants = np.array([[0, 1], [0, 1], [4, 1], [4, 1]], dtype=np.int32)

    groups = _group_ants_by_position_and_team(ants)

    assert groups[(0, 1, 0)] == [0, 1]
    assert groups[(4, 1, 1)] == [2, 3]


def test_adversarial_runner_one_update_smoke() -> None:
    metrics = adversarial_main(
        [
            "--allow-random-init",
            "--total-timesteps",
            "1",
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
            "--food-count",
            "1",
            "--food-sources",
            "1",
            "--max-steps",
            "3",
            "--hidden-size",
            "8",
            "--quiet",
        ]
    )

    assert "loss" in metrics
    assert "own_delivery_events" in metrics


def test_adversarial_workflow_config_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli_main(
        [
            "train",
            "jax",
            "--config",
            "experiments/adversarial_frozen_opponent_probe.json",
            "--dry-run",
        ]
    )

    assert result == 0
    assert "adversarial_frozen_opponent" in capsys.readouterr().out
