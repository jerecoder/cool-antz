from __future__ import annotations

import numpy as np
import pytest

from ant_byte_env import AntAction, AntAgent, AntByteForagingEnv, AntColonyController
from ant_byte_env.agent import FOOD_BASE_CODE, HUB_CODE, OUT_OF_BOUNDS_CODE


class RecordingModel:
    def __init__(self, action: AntAction) -> None:
        self.action = action
        self.calls = []
        self.transitions = []

    def act(self, observation, context):
        self.calls.append((observation.copy(), context))
        return self.action

    def observe(self, transition):
        self.transitions.append(transition)


class PredictModel:
    def predict(self, observation, deterministic=False):
        return np.array([3, 17], dtype=np.int64), None


def test_local_observation_matrix_is_centered_on_ant() -> None:
    env = AntByteForagingEnv(width=5, height=5, num_ants=1, food_count=3)
    obs, _ = env.reset(seed=7, options={"hub_pos": (2, 2), "food_positions": [(3, 2)]})
    obs["bytes"][2, 1] = 88

    model = RecordingModel(AntAction(move=2, write_byte=9))
    agent = AntAgent(ant_id=0, model=model, observation_size=3)
    action = agent.act(obs, info={"step_count": 0})

    assert action == AntAction(move=2, write_byte=9)
    local_obs, context = model.calls[0]
    assert local_obs.shape == (3, 3)
    assert local_obs[1, 1] == HUB_CODE
    assert local_obs[1, 2] == FOOD_BASE_CODE + 3
    assert local_obs[1, 0] == 88
    assert context.position == (2, 2)
    assert context.hub_position == (2, 2)
    assert context.carrying_food is False
    env.close()


def test_local_observation_matrix_marks_out_of_bounds() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    obs, _ = env.reset(seed=11, options={"hub_pos": (0, 0)})

    model = RecordingModel(AntAction(move=0, write_byte=0))
    agent = AntAgent(ant_id=0, model=model, observation_size=3)
    agent.act(obs)

    local_obs, _ = model.calls[0]
    assert local_obs[0, 0] == OUT_OF_BOUNDS_CODE
    assert local_obs[1, 1] == HUB_CODE
    env.close()


def test_agent_supports_predict_style_models() -> None:
    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    obs, _ = env.reset(seed=13)

    agent = AntAgent(ant_id=0, model=PredictModel(), observation_size=3)

    assert agent.act(obs) == AntAction(move=3, write_byte=17)
    env.close()


def test_colony_controller_builds_actions_and_sends_training_feedback() -> None:
    env = AntByteForagingEnv(width=4, height=4, num_ants=2, food_count=0)
    obs, info = env.reset(seed=17, options={"hub_pos": (0, 0)})
    models = [
        RecordingModel(AntAction(move=2, write_byte=5)),
        RecordingModel(AntAction(move=3, write_byte=8)),
    ]
    controller = AntColonyController(
        [
            AntAgent(ant_id=0, model=models[0], observation_size=3),
            AntAgent(ant_id=1, model=models[1], observation_size=3),
        ]
    )

    action = controller.act(obs, info=info)
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    controller.observe(
        next_obs=next_obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=step_info,
    )

    np.testing.assert_array_equal(action, np.array([2, 5, 3, 8], dtype=np.int64))
    assert len(models[0].transitions) == 1
    assert models[0].transitions[0].action == AntAction(move=2, write_byte=5)
    assert models[0].transitions[0].next_observation.shape == (3, 3)
    assert len(models[1].transitions) == 1
    env.close()


def test_agent_rejects_invalid_observation_size_and_actions() -> None:
    with pytest.raises(ValueError):
        AntAgent(ant_id=0, model=RecordingModel(AntAction(move=0)), observation_size=4)

    env = AntByteForagingEnv(width=3, height=3, num_ants=1, food_count=0)
    obs, _ = env.reset(seed=19)
    agent = AntAgent(ant_id=0, model=RecordingModel(AntAction(move=8)), observation_size=3)

    with pytest.raises(ValueError):
        agent.act(obs)
    env.close()
