from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.workflows import args as workflow_args


def test_notebook_workflows_reexports_common_arg_helpers() -> None:
    assert workflows.build_forage_common_args is workflow_args.build_forage_common_args
    assert workflows.build_exploration_common_args is (
        workflow_args.build_exploration_common_args
    )
    assert workflows.config_common_args is workflow_args.config_common_args


def test_notebook_workflows_reexports_common_arg_exclude_sets() -> None:
    assert workflows.COMMUNICATION_ARG_EXCLUDES is workflow_args.COMMUNICATION_ARG_EXCLUDES
    assert workflows.EXPLORATION_TO_FORAGE_ARG_EXCLUDES is (
        workflow_args.EXPLORATION_TO_FORAGE_ARG_EXCLUDES
    )
