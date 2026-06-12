from __future__ import annotations

import numpy as np
import pytest
import torch

from ant_byte_env import AntByteForagingEnv
from ant_byte_env.training.torch_mappo import (
    JointMoveWriteCategorical,
    MAPPOAgent,
    build_actor_observations,
    build_central_observations,
    build_curriculum_reset_options,
    build_local_food_patches,
    compute_forage_curriculum_rewards,
    evaluate_agent,
    evaluate_checkpoint,
    flatten_agent_actions,
    draw_vision_squares,
    main,
    mastery_reached,
    obs_to_tensor,
    parse_args,
    reset_env,
    write_value_count,
)


def _batched_reset_obs() -> dict[str, np.ndarray]:
    env = AntByteForagingEnv(
        width=4,
        height=3,
        num_ants=2,
        food_count=3,
        random_food=False,
    )
    obs, _ = env.reset(seed=123, options={"hub_pos": (0, 0), "food_positions": [(1, 0)]})
    env.close()
    return {key: value[np.newaxis, ...] for key, value in obs.items()}


def test_observation_builders_create_stable_mappo_tensors() -> None:
    obs = obs_to_tensor(_batched_reset_obs(), torch.device("cpu"))

    central_obs = build_central_observations(obs, food_scale=3)
    actor_obs = build_actor_observations(obs, central_obs, food_scale=3)

    assert central_obs.shape == (1, 34)
    assert actor_obs.shape == (1, 2, 101)
    assert torch.all(central_obs >= 0.0)
    assert torch.all(central_obs <= 1.0)
    assert torch.all(actor_obs >= 0.0)
    assert torch.all(actor_obs <= 1.0)


def test_observation_builders_can_pad_to_larger_curriculum_map() -> None:
    obs = obs_to_tensor(_batched_reset_obs(), torch.device("cpu"))

    central_obs = build_central_observations(
        obs,
        food_scale=3,
        obs_width=6,
        obs_height=6,
    )
    actor_obs = build_actor_observations(
        obs,
        central_obs,
        food_scale=3,
        obs_width=6,
        obs_height=6,
    )

    assert central_obs.shape == (1, 82)
    assert actor_obs.shape == (1, 2, 101)
    torch.testing.assert_close(central_obs[:, -2:], torch.tensor([[4 / 6, 3 / 6]]))


def test_actor_observation_contains_local_grids_border_mask_and_carrying_flag() -> None:
    obs = obs_to_tensor(_batched_reset_obs(), torch.device("cpu"))
    central_obs = build_central_observations(obs, food_scale=3)

    actor_obs = build_actor_observations(
        obs,
        central_obs,
        food_scale=3,
        actor_vision_radius=1,
    )

    assert actor_obs.shape == (1, 2, 37)
    torch.testing.assert_close(
        actor_obs[0, 0, :9],
        torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0.0]),
    )
    torch.testing.assert_close(actor_obs[0, 0, 9:18], torch.zeros(9))
    torch.testing.assert_close(
        actor_obs[0, 0, 18:27],
        torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0.0]),
    )
    torch.testing.assert_close(
        actor_obs[0, 0, 27:36],
        torch.tensor([1, 1, 1, 1, 0, 0, 1, 0, 0.0]),
    )
    torch.testing.assert_close(
        actor_obs[0, 0, 36:],
        torch.tensor([0.0]),
    )


def test_actor_observation_exposes_one_local_write_bit_patch() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    obs, _ = env.reset(seed=129, options={"hub_pos": (1, 1)})
    obs["bytes"][1, 2] = 1
    env.close()
    obs_tensor = obs_to_tensor(
        {key: value[np.newaxis, ...] for key, value in obs.items()},
        torch.device("cpu"),
    )
    central_obs = build_central_observations(obs_tensor, food_scale=1)

    actor_obs = build_actor_observations(
        obs_tensor,
        central_obs,
        food_scale=1,
        actor_vision_radius=1,
    )

    patch_size = 9
    byte_bits = actor_obs[0, 0, patch_size : 2 * patch_size]
    expected_bit = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0.0])
    torch.testing.assert_close(byte_bits, expected_bit)


def test_actor_observation_write_bits_controls_local_bit_planes() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0, write_bits=3)
    obs, _ = env.reset(seed=129, options={"hub_pos": (1, 1)})
    obs["bytes"][1, 2] = 5
    env.close()
    obs_tensor = obs_to_tensor(
        {key: value[np.newaxis, ...] for key, value in obs.items()},
        torch.device("cpu"),
    )
    central_obs = build_central_observations(
        obs_tensor,
        food_scale=1,
        write_bits=3,
    )

    actor_obs = build_actor_observations(
        obs_tensor,
        central_obs,
        food_scale=1,
        actor_vision_radius=1,
        write_bits=3,
    )

    patch_size = 9
    bit_patches = actor_obs[0, 0, patch_size : 4 * patch_size].reshape(3, patch_size)
    expected_bit = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0.0])
    assert actor_obs.shape == (1, 1, 55)
    torch.testing.assert_close(bit_patches[0], expected_bit)
    torch.testing.assert_close(bit_patches[1], torch.zeros(patch_size))
    torch.testing.assert_close(bit_patches[2], expected_bit)


def test_actor_observation_exposes_colony_in_local_hub_grid() -> None:
    env = AntByteForagingEnv(
        width=4,
        height=4,
        num_ants=1,
        food_count=1,
        random_food=False,
    )
    obs, _ = env.reset(seed=127, options={"hub_pos": (2, 1), "food_positions": [(3, 1)]})
    env.close()
    obs_tensor = obs_to_tensor(
        {key: value[np.newaxis, ...] for key, value in obs.items()},
        torch.device("cpu"),
    )
    central_obs = build_central_observations(obs_tensor, food_scale=1)

    actor_obs = build_actor_observations(
        obs_tensor,
        central_obs,
        food_scale=1,
        actor_vision_radius=1,
    )

    patch_size = 9
    torch.testing.assert_close(
        actor_obs[0, 0, 2 * patch_size : 3 * patch_size],
        torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0.0]),
    )


def test_local_food_patches_pad_map_edges() -> None:
    food = torch.tensor([[[0.0, 2.0], [3.0, 0.0]]])
    ants_pos = torch.tensor([[[0, 0], [1, 1]]])

    patches = build_local_food_patches(food, ants_pos, radius=1, food_scale=3)

    assert patches.shape == (1, 2, 9)
    torch.testing.assert_close(
        patches[0, 0],
        torch.tensor([0, 0, 0, 0, 0, 2 / 3, 0, 1, 0.0]),
    )
    torch.testing.assert_close(
        patches[0, 1],
        torch.tensor([0, 2 / 3, 0, 1, 0, 0, 0, 0, 0.0]),
    )


def test_agent_samples_and_scores_joint_move_write_actions() -> None:
    obs = obs_to_tensor(_batched_reset_obs(), torch.device("cpu"))
    central_obs = build_central_observations(obs, food_scale=3)
    actor_obs = build_actor_observations(obs, central_obs, food_scale=3)
    agent = MAPPOAgent(
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=32,
    )

    actions, logprob, entropy, value = agent.get_action_and_value(actor_obs, central_obs)
    rescored_actions, rescored_logprob, _, rescored_value = agent.get_action_and_value(
        actor_obs,
        central_obs,
        actions,
    )

    assert actions.shape == (1, 2, 2)
    assert torch.all((0 <= actions[..., 0]) & (actions[..., 0] <= 4))
    assert torch.all((0 <= actions[..., 1]) & (actions[..., 1] <= 1))
    assert logprob.shape == (1, 2)
    assert entropy.shape == (1, 2)
    assert value.shape == (1,)
    torch.testing.assert_close(rescored_actions, actions)
    torch.testing.assert_close(rescored_logprob, logprob)
    torch.testing.assert_close(rescored_value, value)


def test_agent_write_head_size_follows_write_bits() -> None:
    obs = obs_to_tensor(_batched_reset_obs(), torch.device("cpu"))
    central_obs = build_central_observations(obs, food_scale=3, write_bits=3)
    actor_obs = build_actor_observations(obs, central_obs, food_scale=3, write_bits=3)
    agent = MAPPOAgent(
        central_obs_dim=central_obs.shape[-1],
        actor_obs_dim=actor_obs.shape[-1],
        hidden_size=32,
        write_value_count=write_value_count(3),
    )

    move_logits, write_logits = agent.get_action_logits(actor_obs)
    actions, _, _, _ = agent.get_action_and_value(actor_obs, central_obs)

    assert actor_obs.shape == (1, 2, 151)
    assert move_logits.shape == (1, 2, 5)
    assert write_logits.shape == (1, 2, 8)
    assert torch.all((0 <= actions[..., 1]) & (actions[..., 1] <= 7))


def test_joint_move_write_distribution_scores_independent_heads() -> None:
    move_logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5]]])
    write_logits = torch.zeros((1, 1, 2))
    distribution = JointMoveWriteCategorical(
        move_logits=move_logits,
        write_logits=write_logits,
    )
    actions = torch.tensor([[[4, 1]]])

    logprob = distribution.log_prob(actions)
    entropy = distribution.entropy()

    expected_logprob = torch.distributions.Categorical(
        logits=move_logits
    ).log_prob(actions[..., 0]) + torch.distributions.Categorical(
        logits=write_logits
    ).log_prob(actions[..., 1])
    torch.testing.assert_close(logprob, expected_logprob)
    assert entropy.shape == (1, 1)
    torch.testing.assert_close(distribution.deterministic_sample, torch.tensor([[[4, 0]]]))


def test_flatten_agent_actions_interleaves_movement_and_write_bits() -> None:
    actions = torch.tensor([[[1, 1], [4, 0]]], dtype=torch.long)

    flat_actions = flatten_agent_actions(actions)

    torch.testing.assert_close(flat_actions, torch.tensor([[1, 1, 4, 0]]))


def test_draw_vision_squares_overlays_clipped_ant_windows() -> None:
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    obs = {
        "ants_pos": np.array([[0, 0], [3, 2]], dtype=np.int32),
        "food": np.zeros((3, 4), dtype=np.int32),
    }

    overlay = draw_vision_squares(
        frame,
        obs,
        tile_size=10,
        vision_radius=1,
        colors=((10, 20, 30), (200, 210, 220)),
        border_px=1,
        fill_alpha=0.0,
    )

    assert np.all(frame == 0)
    np.testing.assert_array_equal(overlay[0, 0], np.array([10, 20, 30], dtype=np.uint8))
    np.testing.assert_array_equal(overlay[29, 39], np.array([200, 210, 220], dtype=np.uint8))
    assert overlay.sum() > 0


def test_draw_vision_squares_rejects_invalid_settings() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    obs = {
        "ants_pos": np.array([[0, 0]], dtype=np.int32),
        "food": np.zeros((1, 1), dtype=np.int32),
    }

    for kwargs in (
        {"tile_size": 0, "vision_radius": 1},
        {"tile_size": 10, "vision_radius": -1},
        {"tile_size": 10, "vision_radius": 1, "border_px": 0},
        {"tile_size": 10, "vision_radius": 1, "fill_alpha": 1.5},
        {"tile_size": 10, "vision_radius": 1, "colors": ()},
    ):
        with pytest.raises(ValueError):
            draw_vision_squares(frame, obs, **kwargs)


def test_parse_args_rejects_negative_actor_vision_radius() -> None:
    with pytest.raises(ValueError, match="actor-vision-radius"):
        parse_args(["--actor-vision-radius", "-1"])


@pytest.mark.parametrize("write_bits", ["0", "9"])
def test_parse_args_rejects_invalid_write_bits(write_bits: str) -> None:
    with pytest.raises(ValueError, match="write-bits"):
        parse_args(["--write-bits", write_bits])


def test_forage_curriculum_rewards_pickup_and_target_progress() -> None:
    env = AntByteForagingEnv(width=4, height=3, num_ants=1, food_count=1)
    previous_obs, _ = env.reset(
        seed=5,
        options={"hub_pos": (0, 0), "food_positions": [(1, 0)]},
    )
    next_obs, env_reward, _, _, _ = env.step(np.array([2, 0], dtype=np.int64))
    env.close()

    shaped_rewards = compute_forage_curriculum_rewards(
        previous_obs={key: value[np.newaxis, ...] for key, value in previous_obs.items()},
        next_obs={key: value[np.newaxis, ...] for key, value in next_obs.items()},
        env_rewards=np.array([env_reward], dtype=np.float32),
        pickup_bonus=0.25,
        distance_bonus=0.02,
    )

    np.testing.assert_allclose(shaped_rewards, np.array([0.27], dtype=np.float32))


def test_random_hub_and_cookie_sources_are_seed_reproducible() -> None:
    args = parse_args(
        [
            "--width",
            "8",
            "--height",
            "8",
            "--num-ants",
            "1",
            "--food-count",
            "9",
            "--food-sources",
            "3",
            "--random-food",
            "--random-hub",
        ]
    )
    env_a = AntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        random_food=args.random_food,
    )
    env_b = AntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        random_food=args.random_food,
    )

    obs_a, _ = reset_env(env_a, seed=123, args=args)
    obs_b, _ = reset_env(env_b, seed=123, args=args)
    obs_c, _ = reset_env(env_b, seed=124, args=args)

    np.testing.assert_array_equal(obs_a["hub_pos"], obs_b["hub_pos"])
    np.testing.assert_array_equal(obs_a["food"], obs_b["food"])
    assert np.count_nonzero(obs_a["food"]) == 3
    assert obs_a["food"][obs_a["hub_pos"][1], obs_a["hub_pos"][0]] == 0
    assert (
        not np.array_equal(obs_a["hub_pos"], obs_c["hub_pos"])
        or not np.array_equal(obs_a["food"], obs_c["food"])
    )

    env_a.close()
    env_b.close()


def test_random_hub_options_include_colony_without_fixed_cookie_positions() -> None:
    args = parse_args(
        [
            "--width",
            "6",
            "--height",
            "7",
            "--random-food",
            "--random-hub",
        ]
    )

    options = build_curriculum_reset_options(args, seed=5)

    assert options is not None
    assert set(options) == {"hub_pos"}
    assert 0 <= options["hub_pos"][0] < 6
    assert 0 <= options["hub_pos"][1] < 7


class ScriptedForageAgent:
    def get_action_and_value(
        self,
        actor_obs: torch.Tensor,
        central_obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del action, deterministic
        del central_obs
        patch_size = (actor_obs.shape[-1] - 1) // 4
        patch_width = int(patch_size**0.5)
        center_index = patch_size // 2
        carrying = actor_obs[:, :, -1] > 0.5
        hub_patch = actor_obs[:, :, 2 * patch_size : 3 * patch_size]
        hub_at_center = hub_patch[:, :, center_index] > 0.5
        hub_left = hub_patch[:, :, center_index - 1] > 0.5
        hub_right = hub_patch[:, :, center_index + 1] > 0.5
        return_to_hub = torch.where(
            hub_at_center,
            torch.zeros_like(carrying, dtype=torch.long),
            torch.where(hub_left, torch.full_like(carrying, 4, dtype=torch.long), 2),
        )
        return_to_hub = torch.where(hub_right, torch.full_like(return_to_hub, 2), return_to_hub)
        movement = torch.where(carrying, return_to_hub, 2).long()
        write = torch.zeros_like(movement)
        actions = torch.stack([movement, write], dim=-1)
        zeros = torch.zeros_like(movement, dtype=torch.float32)
        values = torch.zeros(actor_obs.shape[0], dtype=torch.float32)
        assert patch_width == 2 * 2 + 1
        return actions, zeros, zeros, values


def test_evaluate_agent_measures_true_delivery_mastery() -> None:
    args = parse_args(
        [
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
            "--max-steps",
            "4",
            "--no-cuda",
            "--quiet",
        ]
    )

    metrics = evaluate_agent(
        agent=ScriptedForageAgent(),
        args=args,
        device=torch.device("cpu"),
        num_episodes=3,
    )

    assert metrics["eval_success_rate"] == 1.0
    assert metrics["eval_mean_delivered_fraction"] == 1.0
    assert mastery_reached(
        metrics,
        min_success_rate=1.0,
        min_delivered_fraction=1.0,
    )


def test_tiny_mappo_training_run_completes() -> None:
    metrics = main(
        [
            "--total-timesteps",
            "4",
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
            "--no-cuda",
            "--quiet",
        ]
    )

    assert metrics["global_step"] == 4
    assert np.isfinite(metrics["loss"])


def test_saved_checkpoint_uses_torch_safe_metadata(tmp_path) -> None:
    checkpoint_path = tmp_path / "mappo.pt"

    main(
        [
            "--total-timesteps",
            "4",
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
            "--max-steps",
            "8",
            "--hidden-size",
            "16",
            "--seed",
            "13",
            "--no-cuda",
            "--quiet",
            "--save-model",
            str(checkpoint_path),
        ]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert checkpoint["args"]["save_model"] == str(checkpoint_path)
    assert checkpoint["central_obs_dim"] > 0
    assert checkpoint["actor_obs_dim"] > 0
    assert "optimizer_state_dict" in checkpoint

    eval_metrics = evaluate_checkpoint(
        checkpoint_path,
        num_episodes=1,
        device=torch.device("cpu"),
    )
    assert "eval_success_rate" in eval_metrics
    assert "eval_mean_delivered_fraction" in eval_metrics


def test_checkpoint_can_resume_on_larger_padded_map(tmp_path) -> None:
    checkpoint_path = tmp_path / "mappo.pt"
    common_args = [
        "--total-timesteps",
        "4",
        "--num-envs",
        "1",
        "--num-steps",
        "4",
        "--num-minibatches",
        "1",
        "--update-epochs",
        "1",
        "--obs-width",
        "5",
        "--obs-height",
        "5",
        "--num-ants",
        "1",
        "--food-count",
        "1",
        "--max-steps",
        "8",
        "--hidden-size",
        "16",
        "--seed",
        "17",
        "--no-cuda",
        "--quiet",
    ]

    main(
        [
            *common_args,
            "--width",
            "4",
            "--height",
            "4",
            "--save-model",
            str(checkpoint_path),
        ]
    )

    metrics = main(
        [
            *common_args,
            "--width",
            "5",
            "--height",
            "5",
            "--load-model",
            str(checkpoint_path),
            "--save-model",
            str(checkpoint_path),
        ]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert metrics["global_step"] == 4
    assert checkpoint["central_obs_dim"] == 57
    assert checkpoint["actor_obs_dim"] == 101
    assert checkpoint["args"]["width"] == 5
