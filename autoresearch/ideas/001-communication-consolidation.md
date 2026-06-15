# Communication Consolidation Result

## Summary

The first final confirmation of H0 + R5 + T0 made write symbols diverse under
sampling, but deterministic rollouts still had poor forage behavior. A short
post-8-bit consolidation fine-tune from the final checkpoints fixed much of
that failure mode.

Best screened setting:

- Start from an 8-bit final checkpoint.
- Train only the 8-bit stage for `2_500` updates (`3_200_000` env steps).
- Use `write_bit_entropy_bonus = 0.05`.
- Use `ent_coef = 0.002`.
- Disable terminal write entropy: `write_entropy_bonus = 0.0`,
  `write_entropy_bonus_cap = 0.0`.

## Evidence

The original final seeds had weak deterministic delivery:

| Run | Sampled delivered | Sampled bit entropy | Deterministic delivered | Deterministic bit entropy |
| --- | ---: | ---: | ---: | ---: |
| F1 | 2.0 | 0.905 | 1.0 | 0.146 |
| F2 | 4.0 | 0.672 | 0.0 | 0.031 |
| F3 | 2.0 | 0.804 | 3.0 | 0.120 |

Single-episode consolidation screening from F3:

| ID | `ent_coef` | bit bonus | Sampled delivered | Deterministic delivered | Deterministic bit entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 | 0.0 | 0.0 | 19.0 | 8.0 | 0.040 |
| C1 | 0.002 | 0.0 | 12.0 | 0.0 | 0.010 |
| C2 | 0.0 | 0.05 | 11.0 | 3.0 | 0.235 |
| C3 | 0.002 | 0.05 | 21.0 | 11.0 | 0.237 |

Four-episode C3-style replication across final checkpoints:

| Run | Source | Sampled delivered | Sampled major symbols | Deterministic delivered | Deterministic major symbols |
| --- | --- | ---: | ---: | ---: | ---: |
| K1 | F1 | 6.75 | 6 | 9.5 | 3 |
| K2 | F2 | 12.0 | 2 | 7.5 | 2 |
| K3 | F3 | 13.25 | 4 | 12.25 | 6 |

Aggregate K-run means:

- sampled delivered: `10.67 / 23`
- deterministic delivered: `9.75 / 23`
- sampled write bit entropy: `0.536`
- deterministic write bit entropy: `0.355`
- sampled major nonzero symbols: `4.0`
- deterministic major nonzero symbols: `3.67`

## Interpretation

The useful pattern is not more entropy during the whole curriculum. It is:

1. Use the chunk-local bit entropy curriculum to make wider alphabets reachable.
2. After the 8-bit stage, run a short consolidation stage with much weaker
   communication pressure.
3. Keep a small bit bonus during consolidation; pure PPO entropy alone hurt
   deterministic behavior.

This suggests the final policy needs a settle phase where forage reward can
choose which communication symbols remain useful.

## Next Step

Promote consolidation into the durable communication curriculum as an explicit
post-8-bit stage. Do not replace the staged curriculum with C3 settings from
the start; the evidence only supports C3-style fine-tuning after an already
trained 8-bit communication checkpoint.
