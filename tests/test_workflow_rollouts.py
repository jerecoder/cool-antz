import pytest

from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.workflows import rollouts


def test_notebook_rollout_policy_temperature_comes_from_metadata() -> None:
    assert rollouts.notebook_rollout_policy_temperature({}) == (
        rollouts.NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE
    )
    assert rollouts.notebook_rollout_policy_temperature(
        {"rollout_policy_temperature": 0.0}
    ) == 0.0
    assert rollouts.notebook_rollout_policy_temperature(
        {"rollout_policy_temperature": "0.75"}
    ) == 0.75
    with pytest.raises(ValueError, match="rollout_policy_temperature"):
        rollouts.notebook_rollout_policy_temperature(
            {"rollout_policy_temperature": -0.1}
        )


def test_notebook_workflows_reexports_rollout_settings() -> None:
    assert workflows.NOTEBOOK_ROLLOUT_TILE_SIZE == rollouts.NOTEBOOK_ROLLOUT_TILE_SIZE
    assert workflows.NOTEBOOK_ROLLOUT_SEED_OFFSET == rollouts.NOTEBOOK_ROLLOUT_SEED_OFFSET
    assert workflows.NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE == (
        rollouts.NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE
    )
    assert workflows.notebook_rollout_policy_temperature is (
        rollouts.notebook_rollout_policy_temperature
    )
