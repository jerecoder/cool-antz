# Timed-Release Cooperative Roles Experiment

This experiment tests whether a cooperative single-hub policy develops clearer
roles when ants are released from the hub in a fixed order instead of all being
active from step 0.

The core idea is simple: keep the actor observation shape compatible with the
8-ant shared-writes checkpoint, but change the episode dynamics so rank 0 starts
alone and the rest of the ranks join one at a time. The existing 8-way actor
identity one-hot is interpreted as release rank. No new actor input features are
added in V1.

## Research Question

The source cooperative policy can forage with 8 ants, local vision, and shared
write bits. What is less clear is whether the ants use the identity one-hot only
as an arbitrary slot label, or whether it can support stable role specialization.

Timed release creates a cleaner pressure:

- rank 0 is the first explorer;
- later ranks observe a world that may already contain trails, discovered food,
  and partial hub-local write patterns;
- each rank has a fixed release identity without changing the actor network.

If the setup works, role metrics should reveal differences such as early ranks
exploring more unique cells, later ranks reaching food faster, or some ranks
writing more useful trails.

## Current Config

Main config:

```text
experiments/timed_release_roles_8ants_shared_writes.json
```

Notebook:

```text
notebooks/timed_release/roles.ipynb
```

Workflow routing:

```text
metadata.workflow = "timed_release_roles"
```

Source checkpoint:

```text
runs/notebooks/exploration_to_forage_proximity_sources_full_layout_50x50_8ants_half_food_2src_shared_writes_from_64env_best/checkpoints/best_full_layout_proximity_8ants_half_food_shared_writes.pkl
```

## Environment

The timed-release lane wraps the normal cooperative `JaxAntByteForagingEnv`
instead of changing it globally.

V1 release schedule:

| Rank | Release Step |
| ---: | ---: |
| 0 | 0 |
| 1 | 150 |
| 2 | 300 |
| 3 | 450 |
| 4 | 600 |
| 5 | 750 |
| 6 | 900 |
| 7 | 1050 |

Episode length remains `2000` steps, so every rank has time to act after release.

Inactive ants:

- remain at the hub;
- are hidden from local actor-visible ant counts;
- cannot move;
- cannot write;
- cannot pick up food;
- cannot deliver;
- are excluded from PPO actor loss, entropy, KL, and clip fraction.

Value loss and GAE remain env-level. Normal cooperative rollouts use all-one
agent masks, so the shared JAX MAPPO path keeps its old behavior.

## Hyperparameters

The experiment is warm-started from the 8-ant, 50x50, shared-writes cooperative
checkpoint and keeps the source recipe where possible:

- map: `50x50`
- ants: `8`
- food: `125`
- sources: `2`
- `write_bits = 4`
- `actor_vision_radius = 2`
- shared writes
- `write_while_moving = true`
- critic: `strided_cnn`
- `gamma = 0.997`
- `pickup_bonus = 0.05`
- max episode steps: `2000`
- best checkpoint metric: `eval_mean_delivered_fraction`
- eval/render action mode: `sampled_move_greedy_write`

L4 runtime profile:

- `num_envs = 128`
- `num_steps = 256`
- `total_timesteps = 65536000`
- base updates: `500`
- tuned continuation updates from the saved global-best checkpoint: `2000`
- chunk size in notebook: `500` updates
- best-eval checkpointing during continuation: every chunk
- W&B target: `jerefigueiredo-universidad-de-san-andr-s/cool-antz`
- tuned learning rate: `1e-4`
- tuned write penalty: `0.0001`
- tuned best-eval episodes: `16`

The source run used a larger `64 x 256` rollout profile and a `strided_cnn`
critic. On the L4 instance, a short shape sweep selected `128 x 256`: it keeps
the source rollout horizon and spans the `150`-step release interval while
running near the measured throughput ceiling for this implementation. The local
profile intentionally does not preserve the source run's `20000`-update budget;
this is the GPU-backed role-probe profile. The first notebook probe covered
updates `0-500`; the later continuation reached a complete terminal checkpoint
at update `4500`, but 16-episode eval favored the saved global-best checkpoint
from the earlier part of the run. The tuned profile now starts from that saved
global best, lowers the PPO learning rate, adds a tiny write penalty, and uses
16-episode best-eval selection.

Measured steady-state throughput, excluding first-update compile/autotune:

| Profile | Env steps/update | Env steps/s | GPU memory |
| --- | ---: | ---: | ---: |
| `256 x 128` | `32768` | `47607` | `17116 MiB` |
| `128 x 256` | `32768` | `46746` | `17116 MiB` |
| `192 x 128` | `24576` | `46682` | `8924 MiB` |
| `96 x 384` | `36864` | `45883` | `16860 MiB` |
| `96 x 256` | `24576` | `44435` | `8668 MiB` |
| `64 x 256` | `16384` | `40961` | `8668 MiB` |

Checkpoint policy:

- load the source cooperative actor body, movement head, and write head;
- load the source `strided_cnn` critic;
- load the source optimizer state.

Policy temperature:

- training movement sampling temperature: `0.75`
- best-eval movement sampling temperature: `0.52`
- write temperature: `1.0`, but writes are greedy in `sampled_move_greedy_write`

So training keeps the source shared-writes recipe's movement stochasticity, while
held-out eval and rendering use sharper sampled movement for inspection. Writes
remain deterministic in the main eval/render mode.

## Notebook Flow

The notebook is intentionally thin. Core environment, evaluation, and rendering
logic lives in source files under:

```text
src/ant_byte_env/training/jax_mappo/timed_release/
```

Recommended first run:

```python
RUN_TRAINING = True
MAX_CHUNKS_TO_RUN = 1
```

This runs one chunk, then evaluation/rendering can be used to check
that the release mechanics and role metrics look sane.

The notebook can run in-loop best-checkpoint evaluation during chunk training.
For continuation, this is enabled by the experiment metadata. Training itself
continues from the terminal chunk checkpoint, while the notebook's
`ACTIVE_CHECKPOINT` switches to the best-eval checkpoint when one has been saved
so the evaluation and rendering cells inspect the strongest model seen so far.
Because notebook chunks are now `500` updates, each continuation chunk lands on
the global `500`-update best-eval cadence. Each scoring chunk writes a candidate
best checkpoint first; the notebook only promotes it to the global best
checkpoint if it beats the saved global best metric.

After the first sanity pass, or for the current tuned L4 continuation:

```python
MAX_CHUNKS_TO_RUN = None
```

That lets the notebook train the configured tuned continuation budget. For the
tuned run, the notebook writes under a fresh run directory and compares candidate
best checkpoints against the source best checkpoint before promoting them.

## Evaluation Metrics

Evaluation reports global delivery performance plus per-rank role diagnostics.

Global metrics include:

- delivered food;
- delivered fraction;
- episode return;
- episode length;
- active ant-steps;
- delivered food per 1000 active ant-steps;
- total pickups.

Per-rank metrics include:

- pickups;
- deliveries;
- writes;
- first pickup step;
- first delivery step;
- unique cells visited;
- release-to-pickup latency.

The key thing to watch is not only whether total delivery improves. We also want
to know whether the ranks diverge in useful ways. For example, rank 0 might visit
more cells, rank 1 might have lower release-to-pickup latency after following
early writes, or later ranks might specialize in hub-adjacent delivery.

## Rendering

Rendering outputs MP4, not GIF.

Default render settings:

- action mode: `sampled_move_greedy_write`
- move temperature: `0.52`
- max frames: `480`
- tile size: `16`

Unreleased ants are hidden. Active ants are drawn with rank labels so the video
can be inspected for release-order behavior.

Because movement is sampled, two renders with different seed offsets can differ.
For the main experiment this is desirable because it matches the source policy
usage. A fully greedy render can be useful as a diagnostic, but should not be the
main signal.

## What Would Count As Progress

Promising signs:

- total delivery does not collapse relative to the warm-start checkpoint;
- early ranks explore more unique cells;
- later ranks show lower release-to-pickup latency;
- write counts differ by rank in a stable way;
- rendered behavior shows later ants benefiting from earlier paths or byte
  markings;
- delivered food per 1000 active ant-steps improves over the warm-start
  baseline.

Warning signs:

- rank 0 gets stuck and later ranks merely recover;
- all ranks behave identically despite staggered release;
- delivery drops heavily after training;
- later ranks ignore earlier writes;
- role metrics are dominated by release time rather than behavior.

## Validation

Config dry-run:

```bash
PYTHONPATH=src:. python3 -m ant_byte_env.cli train jax \
  --config experiments/timed_release_roles_8ants_shared_writes.json \
  --dry-run
```

Targeted tests added with the implementation:

```bash
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_train_mappo_jax.py \
  tests/test_jax_timed_release_mappo.py \
  tests/test_render.py
```

Implementation validation previously passed with `160` tests.

## Implementation Files

Timed-release source package:

- `src/ant_byte_env/training/jax_mappo/timed_release/env.py`
- `src/ant_byte_env/training/jax_mappo/timed_release/rollout.py`
- `src/ant_byte_env/training/jax_mappo/timed_release/runner.py`
- `src/ant_byte_env/training/jax_mappo/timed_release/evaluation.py`
- `src/ant_byte_env/training/jax_mappo/timed_release/rendering.py`
- `src/ant_byte_env/training/jax_mappo/timed_release/cli.py`
- `src/ant_byte_env/training/jax_mappo/timed_release/types.py`

Shared MAPPO changes:

- `src/ant_byte_env/training/jax_mappo/types.py`
- `src/ant_byte_env/training/jax_mappo/rollout.py`
- `src/ant_byte_env/training/jax_mappo/updates.py`

Tests:

- `tests/test_jax_timed_release_mappo.py`

## Future Controls

Good next controls after the first run:

1. Randomized release order while preserving rank identity.
2. Same release schedule but zero writes.
3. Same release schedule but full greedy movement.
4. Compare against the original all-active cooperative checkpoint in identical
   evaluation layouts.
5. Release interval sweep, for example `75`, `150`, and `300` steps.

Do not expand actor observations until the fixed-rank, shape-compatible version
has been audited.
