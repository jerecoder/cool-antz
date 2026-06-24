from ant_byte_env.workflows.stage_profiles import (
    forage_stage_training_profiles,
    forage_stage_update_timesteps,
)


def test_forage_stage_update_timesteps_prefers_stage_override() -> None:
    assert (
        forage_stage_update_timesteps(
            {"name": "8x8", "update_timesteps": 512, "num_steps": 16},
            common_args=["--num-envs", "4"],
            fallback_update_timesteps=128,
        )
        == 512
    )


def test_forage_stage_update_timesteps_uses_num_envs_and_stage_num_steps() -> None:
    assert (
        forage_stage_update_timesteps(
            {"name": "8x8", "num_steps": 16},
            common_args=["--num-envs", "4"],
            fallback_update_timesteps=128,
        )
        == 64
    )


def test_forage_stage_training_profiles_preserve_training_overrides() -> None:
    assert forage_stage_training_profiles(
        [
            {"name": "4x4"},
            {"name": "8x8", "global_update_cap": 7, "num_steps": 16, "gamma": 0.97},
        ],
        common_args=["--num-envs", "4"],
        fallback_update_timesteps=128,
        fallback_update_cap=3,
    ) == [
        {
            "stage": "4x4",
            "global_update_cap": 3,
            "num_steps": None,
            "gamma": None,
            "update_timesteps": 128,
        },
        {
            "stage": "8x8",
            "global_update_cap": 7,
            "num_steps": 16,
            "gamma": 0.97,
            "update_timesteps": 64,
        },
    ]
