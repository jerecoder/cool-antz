from __future__ import annotations

import numpy as np

from ant_byte_env import AntByteForagingEnv


def test_same_seed_gives_same_initial_state() -> None:
    env_a = AntByteForagingEnv(width=8, height=8, num_ants=3, food_count=6)
    env_b = AntByteForagingEnv(width=8, height=8, num_ants=3, food_count=6)

    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)

    assert info_a == info_b
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key])

    env_a.close()
    env_b.close()
