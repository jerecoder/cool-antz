# Vision-Range 51x51 Codex Autoresearch

You are Codex acting as the researcher for a small Karpathy-style autoresearch
loop. Keep the loop plain: one hypothesis, one JSON mutation, one notebook run,
one measured result, then decide what to try next.

## Objective

Maximize notebook `ret` for the `51x51` actor-vision stage in
`vision_range_curriculum`.

In this notebook, `51x51` means:

- `actor_vision_radius = 25`
- `vision_side = 51`
- stage directory: `runs/notebooks/vision_range_curriculum/51x51`

Use `trajectory_return` when it is present. In this checkout, the JAX runner may
write the same notebook `ret` as `episode_return`; treat `episode_return` as the
primary fallback. Use `env_return` only as supporting context.

## Operating Contract

The research action is to edit only:

```text
experiments/vision_range_curriculum.json
```

Do not edit source files, tests, notebooks, docs, or create a standalone tuner
unless the user explicitly asks. Generated training artifacts under `runs/` are
normal evidence and are allowed.

Run training/evaluation only through the notebook:

```bash
python -m jupyter nbconvert \
  --to notebook \
  --execute notebooks/train_jax_vision_range_curriculum.ipynb \
  --output-dir runs/autoresearch/vision_range_curriculum/notebook_exec \
  --output latest.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Do not bypass the notebook with a direct `ant-byte train` command for scoring.
Shell commands that inspect JSON, summaries, logs, or git status are fine.

## Result Extraction

After each notebook run, score the trial from:

```text
runs/notebooks/vision_range_curriculum/51x51/summary.json
```

The exact value to compare is:

```bash
jq '.metrics.trajectory_return // .metrics.episode_return // .metrics.env_return' \
  runs/notebooks/vision_range_curriculum/51x51/summary.json
```

Also inspect the last few rows of:

```text
runs/notebooks/vision_range_curriculum/51x51/metrics.jsonl
```

This helps distinguish one lucky final update from a run that improved
consistently.

## Loop

1. Read the current JSON and latest `51x51` result, if any.
2. State a short hypothesis before changing anything.
3. Make one dominant JSON change, or one tightly related family of changes.
4. Run the notebook command above from the repo root.
5. Extract the `51x51` ret from `summary.json`.
6. Compare against the previous best.
7. Decide: keep, mutate, or revert the JSON change.
8. Stop when `VISION_RANGE_TARGET_RET` is reached, or when
   `VISION_RANGE_MAX_TRIALS` nonzero trials have been exhausted.

If `VISION_RANGE_MAX_TRIALS=0`, run until the target is reached or a real
blocker appears.

## JSON Knobs

Prefer tuning training hyperparameters and budget first:

- `num_envs`
- `num_steps`
- `num_minibatches`
- `update_epochs`
- `hidden_size`
- `seed`
- `metadata.global_update_cap`

You may add standard JAX MAPPO CLI args to the JSON when useful, such as:

- `learning_rate`
- `gamma`
- `gae_lambda`
- `clip_eps`
- `ent_coef`
- `vf_coef`
- `max_grad_norm`
- `anneal_lr`

Keep the fixed task fixed unless you have a specific reason:

- `width = 50`
- `height = 50`
- `obs_width = 50`
- `obs_height = 50`
- `num_ants = 1`
- `actor_vision_radius` in `args` is stage-overridden by the notebook, so do
  not treat it as the tuning knob.

For fast 51x51 screening, it is acceptable to set:

```json
"vision_radii": [25],
"vision_sides": [51]
```

That avoids spending each trial on smaller-vision stages when the objective is
only the `51x51` return. If you later want a full curriculum validation, restore
the full decreasing schedule with `25` first.

`render_max_frames` does not affect ret; lower it only to reduce GIF overhead.

## Trial Notes

Keep notes in the Codex transcript. Each trial note should include:

- changed JSON fields
- hypothesis
- notebook command outcome
- `51x51` ret
- previous best ret
- verdict

Be honest about failures. If the notebook fails because of CUDA, memory, JSON
shape, or dependency issues, inspect the error and either make a smaller JSON
trial or report the blocker.
