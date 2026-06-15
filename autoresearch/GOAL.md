# Goal: Make Communication Bits Useful

## Objective

Improve the JAX communication-bit curriculum so wider write alphabets become
useful for forage behavior, not merely available. Success means the trained
policy preserves strong forage performance while using multiple writable bits
or write symbols in a measurable, repeatable way.

This goal is intentionally practical: find better training values first, then
promote the winning configuration into the main communication curriculum.

## Success Criteria

A candidate configuration passes if it satisfies all of these:

- Forage delivery is at least 90% of the terminal-entropy baseline.
- Sampled rollouts use at least two nonzero write values, each above 5% of
  nonzero writes.
- At least two writable bits have activation rates between 5% and 95%.
- Deterministic rollout is not collapsed to only write value `1`.
- The result is reproducible across at least three final seeds.

Primary ranking:

1. Reject any config that harms forage delivery by more than 10%.
2. Among remaining configs, prefer higher sampled write bit entropy.
3. Tie-break with deterministic rollout diversity and simpler reward shaping.

## Current Diagnosis

The current 25x25 communication setup uses `num_steps = 80` and
`max_steps = 2500`. PPO therefore updates after 80 sequential steps per
environment, while a full time-limit episode can span roughly 31 PPO updates.

The existing terminal write-entropy bonus is probably too sparse: most updates
do not include an episode end, so the write head receives many gradients from
ordinary forage behavior and warm-start bias before seeing entropy credit.

Checkpoint probes also show that policies trained through wider alphabets still
mostly write `1`, so the first research target is to make the write signal
dense enough to learn.

## Variables To Tune

Tune these variables deliberately, one family at a time:

- Rollout horizon: `num_steps`.
- Parallelism: `num_envs`, mostly to keep batch size and memory manageable.
- Reward shaping:
  - terminal final-grid entropy over nonzero bytes,
  - chunk-local write-action bit entropy,
  - optional combination of both.
- PPO exploration: `ent_coef`.
- Write-head transfer mode when increasing write bits.

Keep these fixed during screening unless a run fails technically:

- `width = 25`
- `height = 25`
- `obs_width = 25`
- `obs_height = 25`
- `actor_vision_radius = 1`
- `max_steps = 2500`
- `num_minibatches = 4`
- `update_epochs = 4`
- `learning_rate = 2.5e-4`
- `hidden_size = 128`
- staged warm start from the 25x25 forage checkpoint

## Required Implementation Support

Before running the full sweep, add two pieces of support code.

### Communication Probe

Add an offline checkpoint probe that reports:

- sampled write-action histogram,
- deterministic write-action histogram,
- final-grid byte histogram,
- per-bit activation rates,
- write bit entropy,
- distinct nonzero values,
- delivery metrics,
- rollout artifact paths.

Write probe outputs under `runs/autoresearch/communication_bits/`.

### Chunk-Local Bit Entropy Reward

Add a trainer flag:

```text
--write-bit-entropy-bonus
```

Default: `0.0`.

Reward definition:

- Compute binary entropy for each write bit from write actions inside the PPO
  rollout chunk.
- Exclude write value `0` from the activation denominator where practical, so
  "writing nothing" does not create fake entropy.
- Normalize each bit entropy to `[0, 1]`.
- Average across writable bits.
- Scale by `write_bit_entropy_bonus`.
- Distribute the shaped bonus uniformly across the rollout rewards for that
  environment.

The goal is to provide a communication signal every update, not only at episode
termination.

### Write-Head Transfer Modes

Add a transfer option for increasing write bits:

```text
--write-head-transfer {repeat,reset,neutral-new}
```

Modes:

- `repeat`: current modulo-repeat behavior.
- `reset`: reset the expanded write head to near-uniform logits when increasing
  write bits.
- `neutral-new`: preserve old write actions and initialize newly available
  write actions to the mean old write-head column.

## Experiment Matrix

Screening runs use stages `[2, 3]`. Keep enough env steps per stage to compare
meaningfully, but do not run the full `[2, 3, 5, 8]` curriculum until the
screening winner is chosen.

### 1. Horizon Sweep

Use the current terminal entropy reward and current transfer behavior.

| ID | `num_steps` | `num_envs` | Reward | Transfer |
| --- | ---: | ---: | --- | --- |
| H0 | 80 | 16 | terminal entropy `0.1`, cap `0.15` | repeat |
| H1 | 256 | 8 | terminal entropy `0.1`, cap `0.15` | repeat |
| H2 | 512 | 8 | terminal entropy `0.1`, cap `0.15` | repeat |
| H3 | 1024 | 4 | terminal entropy `0.1`, cap `0.15` | repeat |

Decision rule:

- Reject configs with delivery more than 10% below H0.
- Choose the remaining config with the best sampled write bit entropy.

### 2. Reward Sweep

Use the selected horizon from phase 1.

| ID | Terminal entropy | Bit entropy bonus | `ent_coef` | Transfer |
| --- | ---: | ---: | ---: | --- |
| R0 | `0.1`, cap `0.15` | `0.0` | `0.01` | repeat |
| R1 | `0.0` | `0.25` | `0.01` | repeat |
| R2 | `0.0` | `0.5` | `0.01` | repeat |
| R3 | `0.0` | `1.0` | `0.01` | repeat |
| R4 | `0.05`, cap `0.10` | `0.5` | `0.01` | repeat |
| R5 | `0.0` | `0.5` | `0.02` | repeat |

Decision rule:

- Reject configs with delivery more than 10% below R0.
- Choose the remaining config with the best sampled write bit entropy.
- Tie-break by deterministic rollout using at least two nonzero write values.

### 3. Transfer Sweep

Use the selected horizon and reward from phases 1 and 2.

| ID | Transfer mode |
| --- | --- |
| T0 | repeat |
| T1 | reset |
| T2 | neutral-new |

Decision rule:

- Prefer `reset` or `neutral-new` only if they improve bit usage without
  reducing delivery by more than 10%.

### 4. Final Confirmation

Run the selected config on full stages `[2, 3, 5, 8]`.

| ID | Seed | Env steps per stage |
| --- | ---: | ---: |
| F1 | 1 | `12_800_000` |
| F2 | 2 | `12_800_000` |
| F3 | 3 | `12_800_000` |

Promote the final config only if all success criteria pass.

## Test Plan

Add tests for:

- bit entropy reward:
  - all zeros gives zero bonus,
  - always `1` gives zero or near-zero bit entropy,
  - balanced bit usage gives positive normalized bonus,
  - reward scales with rollout horizon as intended;
- CLI validation for new reward and transfer flags;
- write-head transfer modes and output shapes;
- tiny JAX smoke training with `--write-bit-entropy-bonus`;
- communication probe output schema on a tiny checkpoint/run.

Keep existing focused suites passing:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jerefigo/miniconda3/envs/tp1-rl/bin/python -m pytest \
  tests/test_train_mappo_jax.py tests/test_notebook_workflows.py -q
```

## Artifact Layout

Use this run root:

```text
runs/autoresearch/communication_bits/
```

Recommended structure:

```text
runs/autoresearch/communication_bits/
  horizon/H0/
  horizon/H1/
  reward/R2/
  transfer/T1/
  final/F1/
```

Each run directory should contain:

- resolved config,
- metrics,
- checkpoint,
- probe JSON,
- deterministic rollout,
- sampled rollout,
- short verdict note.

## Promotion Rule

Only update the durable communication curriculum after final confirmation.
When promoting, update:

- `experiments/communication_bits.json`,
- notebook defaults if needed,
- autoresearch idea/verdict notes,
- tests covering any new public flags.

Do not promote a config merely because it writes diverse symbols. It must keep
or improve useful forage behavior.
