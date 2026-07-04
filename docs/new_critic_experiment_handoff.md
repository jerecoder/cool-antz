# New Critic Experiment Handoff

Use this as the opening context for a fresh Codex session.

## Objective

Redo the two main experiment families with the new JAX MAPPO critic baseline:

1. The sparse-reward direct baseline on the hard final target, `50x50` with
   `10` ants, to see whether the stronger critic changes from-scratch
   convergence.
2. The map/ant curriculum that starts at `4x4_1_ant`, increases map size and ant
   count, and uses mastery gates instead of fixed update budgets. For this pass,
   stop at `12x12_3_ants`, verify mastery there, then decide how to continue.

The immediate goal is to compare behavior against the earlier runs and see
whether the stronger critic helps the ants learn cleaner pickup, delivery, and
sparse-useful writing. Writing is part of the task, not a nuisance variable to
remove: keep writable bits enabled and judge whether the ants learn useful
communication.

## Current Branch Context

- Repo: `/home/juan/Documents/rl/cool-antz`
- Branch: `research/direct-goal-repro-sweep`
- Last committed baseline before this handoff: `24f925b Add conv critic baseline for JAX MAPPO`
- Selected main-utility port commit: `5ccced7 Port JAX MAPPO eval and rollout controls`
- Do not merge `origin/main` wholesale. Only selected utility pieces were ported.
- Do not expand `--actor-vision-radius`; keep radius `1` unless the user explicitly changes that constraint.
- Do not disable writing. Keep the direct sparse baseline's original writable
  bits and keep `--write-bits 1` for the map/ant curriculum; do not use
  zero-write modes for training, gating, evaluation, rendering, diagnostics, or
  rescue runs.
- Do not make the environment easier with fixed hub/food placement for the main comparison.

## New Critic Baseline

The JAX MAPPO critic now uses the requested grid/scalar split when the central
observation has a real grid shape:

```text
50 x 50 x C
Conv 5x5, 32, stride 2
Conv 3x3, 64, stride 2
Conv 3x3, 64, stride 2
Flatten
Dense 256

Concat scalar/entity MLP 128

Dense 256
Dense 1
```

Implementation entrypoint:

- `src/ant_byte_env/training/jax_mappo/core.py`
- Constants: `CRITIC_GRID_CHANNELS`, `CRITIC_CONV_CHANNELS`,
  `CRITIC_GRID_EMBED_DIM`, `CRITIC_SCALAR_EMBED_DIM`,
  `CRITIC_JOINT_HIDDEN_DIM`
- Runner passes `central_grid_shape=(obs_height, obs_width, 3)` so padded
  curriculum observations can use the same critic shape across stages.

## Useful Main-Branch Pieces Ported

The following selected utilities from `origin/main` are now available without
bringing over the full main branch:

- Training rollout controls:
  - `--training-rollout-temperature`
  - `--deterministic-rollout`
  - `--deterministic-rollout-fraction`
  - `--deterministic-move-rollout-fraction`
- Split-head evaluation/render modes:
  - `deterministic`
  - `sampled`
  - `greedy_move_greedy_write`
  - `greedy_move_sampled_write`
  - `sampled_move_greedy_write`
  - `sampled_move_sampled_write`
- Eval-selected best checkpointing in the trainer:
  - `--save-best-model`
  - `--best-model-metric`
  - `--best-model-mode`
  - `--best-model-selection train|eval`
  - `--best-eval-*`
- `--log-interval` for throttled print/metrics/W&B logging.

The map/ant curriculum wrapper forwards the rollout controls and `--log-interval`.
The curriculum gate still defaults to deterministic plus sampled evaluation.
Use only writing-enabled split-head modes for diagnosis and videos unless the
user explicitly wants to change the gate. Zero-write modes answer the wrong
question for this experiment and should not be used.

## What We Learned Before This Baseline

- Fixed update budgets were misleading; the curriculum should advance only after
  a stage passes mastery gates.
- Rendering showed the core behavioral problem better than metrics alone.
- Bad attractor 1: too much writing, creating staircase or wedge-like marker spam.
- Bad attractor 2: almost no writing, with repeated exploration loops.
- Do not "fix" write-bit hacking by turning writing off. The point is to shape
  or select for useful communication, not to remove the communication channel.
- In videos, ants sometimes see food in the actor observation but choose a loop
  beside it instead of stepping onto the food cell. Pickup itself works when an
  ant enters a food cell.
- Keep random food and random hub for the real comparison; fixed positions can
  overfit.

More history is in:

- `vault/vault.md`
- `docs/map_ant_autoresearch_log.md`

## Recommended Next Session Plan

1. Audit the current worktree first.
   - Preserve unrelated local files such as `.vscode/`.
   - The old notebook may have metadata-only changes; do not stage them unless
     the user asks.

2. Update the existing notebooks in place rather than creating new notebook copies.
   - Use `notebooks/train_jax_direct_goal_baseline.ipynb` for the direct
     sparse-reward `50x50`, `10`-ant baseline rerun.
   - Use `notebooks/train_jax_10_ant_map_curriculum.ipynb` for the gated
     map/ant curriculum rerun from `4x4_1_ant` through `12x12_3_ants`.
   - It is fine to overwrite/simplify the old notebook content; the notebooks
     should be current launcher/review surfaces for this new-critic rerun, not
     archives of every past experiment.
   - Use fresh run directories for the new-critic reruns so their artifacts do
     not mix with older results.

3. Rerun the direct sparse-reward baseline with the new critic.
   - Target: `50x50`, `10` ants, sparse environment delivery reward, random food,
     random hub.
   - Keep `--actor-vision-radius 1`.
   - Keep the original direct-baseline communication/action setup unless the
     user explicitly changes it.
   - Render deterministic, sampled, and at least one split-head diagnostic mode
     that still allows writing, especially `greedy_move_sampled_write` or
     `sampled_move_greedy_write`.

4. Then rerun the gated map/ant curriculum toward `12x12_3_ants`.
   - This is the second major experiment family, not a small direct baseline.
   - Keep mastery gates; do not return to the old fixed-1000-updates-per-stage
     setup.
   - Even though this pass stops at `12x12_3_ants`, keep `--obs-width 50` and
     `--obs-height 50` so the conv critic grid shape stays compatible with a
     later continuation to larger maps.
   - Suggested first comparison schedule:

```bash
PYTHONPATH=src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
/home/juan/miniconda3/envs/cool-antz/bin/python \
  -m ant_byte_env.training.jax_mappo.map_ant_curriculum \
  --run-dir runs/autoresearch/map_ant_curriculum/conv_critic_12x12_3ants_baseline \
  --stage-plan 4:1,6:1,8:2,10:3,12:3 \
  --food-counts-by-stage 2,4,6,8,10 \
  --food-sources-by-stage 1,1,1,1,1 \
  --obs-width 50 \
  --obs-height 50 \
  --actor-vision-radius 1 \
  --random-food \
  --random-hub \
  --write-bits 1 \
  --write-while-moving \
  --num-envs 8 \
  --num-steps 256 \
  --num-minibatches 4 \
  --update-epochs 4 \
  --gate-update-chunk-cap 500 \
  --gate-max-stage-attempts 6 \
  --gate-eval-num-episodes 16 \
  --gate-eval-modes deterministic,sampled
```

5. If behavior collapses again, tune only available training/reward knobs.
   - Reasonable knobs: `gamma`, `ent-coef`, `learning-rate`, step penalty,
     pickup/completion bonus, write penalties, visible-food/carrying-hub shaping,
     rollout temperature, and deterministic rollout mix.
   - Do not tune actor vision radius.
   - Do not switch to fixed hub/food for the main result.

## Validation Command

Use this focused check after edits:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
/home/juan/miniconda3/envs/cool-antz/bin/python \
  -m pytest tests/test_train_mappo_jax.py tests/test_render.py tests/test_map_ant_curriculum.py -q
```

At the time this handoff was created, that focused suite passed.
