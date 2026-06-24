from __future__ import annotations

from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.runtime import resources


def test_notebook_workflows_reexports_runtime_resource_helpers() -> None:
    assert workflows.configure_jax_notebook_runtime is resources.configure_jax_notebook_runtime
    assert workflows.assert_notebook_resources_available is (
        resources.assert_notebook_resources_available
    )
    assert workflows.cleanup_notebook_artifacts is resources.cleanup_notebook_artifacts
