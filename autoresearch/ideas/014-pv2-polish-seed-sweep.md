# PV2 Polish Seed Sweep

## Summary

PL9, PL10, and PL11 tested whether the weak PV2 short-polish result could be
fixed by changing only the final polish seed. It can. PL10, seed `804`, is the
best PV2 short-polish candidate so far.

All runs used:

- source checkpoint:
  `runs/autoresearch/communication_bits/promoted/PV2/8_bits_consolidated/checkpoints/model.pkl`
- `global_update_cap = 1_250`
- `write_entropy_bonus = 0.0`
- `write_bit_entropy_bonus = 0.0`
- `ent_coef = 0.0`
- `pickup_bonus = 0.25`
- `distance_bonus = 0.02`

Commands:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL9 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL10 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL11 \
  --probe-episodes 4 \
  --no-render
```

PL10 was then confirmed with a larger probe:

```bash
PYTHONPATH=src ant-byte probe communication \
  --checkpoint runs/autoresearch/communication_bits/polish_length/PL10/8_bits/checkpoints/model.pkl \
  --output-dir runs/autoresearch/communication_bits/polish_length/PL10/probe_eval16 \
  --num-episodes 16 \
  --no-render
```

## Four-Episode Result

| Run | Seed | Mode | Delivered | Fraction | Success | Bit entropy | Distinct values | Major values | Mid bits |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PL3 | 802 | sampled | 12.00 | 0.522 | 0.000 | 0.524 | 25 | 2, 4, 5, 214 | 7 |
| PL3 | 802 | deterministic | 10.25 | 0.446 | 0.000 | 0.108 | 7 | 4, 5 | 3 |
| PL8 | 805 | sampled | 11.25 | 0.489 | 0.000 | 0.513 | 35 | 2, 4, 5 | 8 |
| PL8 | 805 | deterministic | 8.75 | 0.380 | 0.000 | 0.332 | 11 | 2, 5 | 3 |
| PL9 | 803 | sampled | 15.75 | 0.685 | 0.000 | 0.450 | 27 | 2, 4, 5 | 7 |
| PL9 | 803 | deterministic | 12.50 | 0.543 | 0.000 | 0.331 | 8 | 2, 4, 5 | 3 |
| PL10 | 804 | sampled | 17.50 | 0.761 | 0.250 | 0.306 | 22 | 2, 4, 5 | 6 |
| PL10 | 804 | deterministic | 13.00 | 0.565 | 0.000 | 0.233 | 9 | 2, 4, 5 | 4 |
| PL11 | 806 | sampled | 10.75 | 0.467 | 0.000 | 0.248 | 22 | 2, 4, 5 | 6 |
| PL11 | 806 | deterministic | 11.75 | 0.511 | 0.250 | 0.207 | 7 | 2, 5 | 3 |

Four-episode PV2 seed-sweep aggregate:

| Mode | Delivered mean | Delivered min | Bit entropy mean |
| --- | ---: | ---: | ---: |
| sampled | 13.45 | 10.75 | 0.408 |
| deterministic | 11.25 | 8.75 | 0.242 |

## PL10 Confirmation

| Run | Episodes | Mode | Delivered | Fraction | Success | Bit entropy | Distinct values | Major values | Mid bits |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PL10 | 16 | sampled | 14.25 | 0.620 | 0.062 | 0.285 | 33 | 2, 4, 5 | 5 |
| PL10 | 16 | deterministic | 13.75 | 0.598 | 0.000 | 0.241 | 9 | 2, 4, 5 | 4 |

## Current Per-Source Best

The best confirmed short-polish seed per promoted consolidated checkpoint is:

| Source | Run | Seed | Eval16 sampled delivered | Eval16 deterministic delivered |
| --- | --- | ---: | ---: | ---: |
| PV1 | PL7 | 805 | 14.94 | 12.19 |
| PV2 | PL10 | 804 | 14.25 | 13.75 |
| PV3 | PL5 | 805 | 16.44 | 15.25 |

Per-source-best eval16 aggregate:

| Mode | Delivered mean | Delivered min |
| --- | ---: | ---: |
| sampled | 15.21 | 14.25 |
| deterministic | 13.73 | 12.19 |

For comparison, the promoted `2_500`-update polish four-episode aggregate was
`12.42` sampled delivered and `13.58` deterministic delivered.

## Interpretation

PV2 was not fundamentally incompatible with short polish. The bad fixed-805
result was mostly a polish-seed issue. Seed `804` is the best PV2 option so
far, and the per-source-best trio now improves sampled delivery substantially
while roughly matching promoted deterministic delivery.

The tradeoff remains deterministic communication diversity: the per-source-best
policies deliver well, but deterministic major values are still relatively
compact compared with the promoted recipe.

## Next Step

Turn this into a first-class selection-gate candidate:

1. Add a `polish_gate` or similar matrix phase that represents the chosen
   per-source best checkpoints PL7, PL10, and PL5.
2. Run a combined comparison note using eval16 probes for promoted polish vs
   per-source-best short polish.
3. If the comparison still holds, test a practical gate rule: run multiple
   short-polish seeds and choose by a cheap four-episode validation probe.
