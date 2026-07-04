# Map-Ant 12x12 Conv-Critic Autoresearch

This loop tests whether the new JAX MAPPO critic helps the ants master the
honest map/ant curriculum from `4x4_1_ant` through `12x12_3_ants`.

## Constraints

- Keep actor vision radius at `1`.
- Keep food and hub placement randomized.
- Keep `--obs-width 50 --obs-height 50` for critic compatibility.
- Keep `--write-bits 1 --write-while-moving`; no zero-write ablations.
- Tune hyperparameters, gates, and reward shaping only.

## Commands

Plan one candidate:

```bash
PYTHONPATH=src python3 -m ant_byte_env.cli autoresearch map-ant-plan \
  --phase screen --id M000-control
```

Run the screen phase:

```bash
./scripts/run_map_ant_12x12_sweep.sh
```

Rank completed screen runs:

```bash
PYTHONPATH=src python3 -m ant_byte_env.cli autoresearch map-ant-rank \
  --phase screen
```

Run promotion seeds after choosing the top two screen candidates:

```bash
PHASE=promotion ./scripts/run_map_ant_12x12_sweep.sh \
  M020-visible-carry_seed2 M020-visible-carry_seed3
```

Current mutation:

```bash
PHASE=mutation ./scripts/run_map_ant_12x12_sweep.sh \
  M021-visible-carry-writeband-from-M020a3
```

This continues from `M020-visible-carry` attempt 3 and adds light write,
write-bit, and overwrite penalties. It is meant to recover from deterministic
marker spam while keeping `--write-bits 1 --write-while-moving`.

## Review Rule

The ranker uses existing task gates: delivery fraction, success rate,
pickup-to-delivery, episode length, and write-rate band. A ranked winner is not
declared successful until its rendered rollouts show writing that plausibly
serves as a food-to-hub trail rather than marker spam or no communication.

The screen phase uses `8` envs x `64` rollout steps for CPU-friendly conv-critic
throughput. It validates every `20` updates for up to `60` attempts per stage;
mutation validates every `20` updates for up to `150` attempts; promotion
validates every `100` updates for up to `30` attempts.
This keeps the screen phase at `1200` updates per stage while creating
checkpoints often enough to continue from a mastered or near-mastered stage.
