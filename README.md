# Ant Byte Foraging

`AntByteForagingEnv` is a Gymnasium gridworld where a centralized controller
moves several ants, lets each ant read the byte stored on its current tile, and
lets each ant overwrite one tile byte per step. Ants collect food from the grid
and deliver it back to a hub.

The environment is rendered with Pygame and supports `render_mode="human"` and
`render_mode="rgb_array"`.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Basic Usage

```python
import gymnasium as gym

env = gym.make("AntByteForaging-v0", render_mode="rgb_array")
obs, info = env.reset(seed=123)

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
frame = env.render()
env.close()
```

You can also import the class directly:

```python
from ant_byte_env import AntByteForagingEnv
```

## Action Space

For `num_ants = N`, the action space is:

```python
spaces.MultiDiscrete([5, 256] * N)
```

Each ant receives a pair `(move_i, write_byte_i)`.

Movement actions:

- `0`: stay
- `1`: up
- `2`: right
- `3`: down
- `4`: left

The write action is an integer from `0` to `255`. It is written to the ant's
tile after movement, pickup, and delivery are processed.

## Observation Space

The environment is fully observable for now:

```python
spaces.Dict({
    "ants_pos": Box(low=0, high=max(width, height), shape=(N, 2), dtype=np.int32),
    "ants_carrying": MultiBinary(N),
    "food": Box(low=0, high=food_count, shape=(height, width), dtype=np.int32),
    "bytes": Box(low=0, high=255, shape=(height, width), dtype=np.uint8),
    "hub_pos": Box(low=0, high=max(width, height), shape=(2,), dtype=np.int32),
})
```

TODO: add an optional local partial-observation mode for decentralized policies.

## Rewards

- `+1` when an ant picks up food.
- `+10` when an ant delivers food to the hub.
- `-0.01` per step per ant.
- `-write_penalty` per write when `write_penalty > 0`.

Episodes terminate when all food has been delivered. They truncate when
`max_steps` is reached.

## Tile Bytes

Every tile stores one unsigned byte. At reset all bytes are `0`. During each
step, ants are processed in index order. If several ants write to the same tile
in the same step, later ants overwrite earlier ants, and the overwrite count is
reported in `info["num_overwrites"]`.

## Random Rollout

```bash
./launch_random_rollout.py
```

or:

```bash
python examples/random_rollout.py
```

The launcher opens a Pygame window, samples random actions, prints delivered and
remaining food counts, and closes the environment cleanly. Use `--seed` for a
repeatable random rollout:

```bash
./launch_random_rollout.py --seed 123
```

## Assets

The repository includes generated placeholder sprites in
`ant_byte_env/assets/`. They are simple CC0-compatible project assets so the
environment works out of the box. See `ASSET_CREDITS.md` for exact asset names,
authors, source, and license.

To replace them with real art, drop files named `ant.png`, `food.png`,
`hub.png`, and `tile.png` into `ant_byte_env/assets/`.

## Stable-Baselines3 Later

This environment uses a `Dict` observation space and a `MultiDiscrete` action
space. For Stable-Baselines3 experiments, start with `MultiInputPolicy`, wrap or
flatten observations if needed, and keep the centralized action vector shape
`(2 * num_ants,)`.

## Tests

```bash
python -m pytest
```
