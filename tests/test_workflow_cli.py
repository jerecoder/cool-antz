from ant_byte_env import notebook_workflows as workflows
from ant_byte_env.workflows import cli


def test_argv_int_reads_last_option_value() -> None:
    assert cli.argv_int(["--num-envs", "1", "--num-envs", "16"], "--num-envs") == 16
    assert cli.argv_int(["--num-envs"], "--num-envs") is None
    assert cli.argv_int(["--num-envs", "many"], "--num-envs") is None
    assert cli.argv_int([], "--num-envs") is None


def test_strip_wandb_cli_args_removes_scalar_and_vararg_options() -> None:
    stripped, removed = cli.strip_wandb_cli_args(
        [
            "--num-envs",
            "1",
            "--wandb-mode",
            "online",
            "--wandb-tags",
            "legacy",
            "stage",
            "--gamma",
            "0.99",
        ]
    )

    assert stripped == ["--num-envs", "1", "--gamma", "0.99"]
    assert removed == [
        "--wandb-mode",
        "online",
        "--wandb-tags",
        "legacy",
        "stage",
    ]


def test_notebook_workflows_reexports_cli_helpers_for_internal_compatibility() -> None:
    assert workflows._argv_int is cli.argv_int
    assert workflows._strip_wandb_cli_args is cli.strip_wandb_cli_args
