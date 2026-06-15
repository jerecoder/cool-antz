# PV3 Short Polish Seed Sweep

## Summary

PL4, PL5, and PL6 tested whether PL1's strong `1_250`-update short polish from
the PV3 consolidated checkpoint was reproducible under different polish seeds.
The result is seed-sensitive, but PL5 is a strong confirmed candidate.

All runs used:

- source checkpoint:
  `runs/autoresearch/communication_bits/promoted/PV3/8_bits_consolidated/checkpoints/model.pkl`
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
  --id PL4 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL5 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL6 \
  --probe-episodes 4 \
  --no-render
```

PL5 was then confirmed with a larger probe:

```bash
PYTHONPATH=src ant-byte probe communication \
  --checkpoint runs/autoresearch/communication_bits/polish_length/PL5/8_bits/checkpoints/model.pkl \
  --output-dir runs/autoresearch/communication_bits/polish_length/PL5/probe_eval16 \
  --num-episodes 16 \
  --no-render
```

## Four-Episode Result

| Run | Seed | Mode | Delivered | Fraction | Success | Bit entropy | Distinct values | Major values | Mid bits |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PL1 | 803 | sampled | 17.50 | 0.761 | 0.250 | 0.526 | 32 | 1, 2, 5, 58 | 7 |
| PL1 | 803 | deterministic | 11.50 | 0.500 | 0.000 | 0.468 | 8 | 1, 2, 5, 7, 58 | 6 |
| PL4 | 804 | sampled | 5.00 | 0.217 | 0.000 | 0.753 | 36 | 1, 2, 32, 48, 58 | 8 |
| PL4 | 804 | deterministic | 10.50 | 0.457 | 0.000 | 0.694 | 11 | 1, 2, 31, 232 | 8 |
| PL5 | 805 | sampled | 15.00 | 0.652 | 0.000 | 0.651 | 39 | 1, 2, 48, 58, 196 | 8 |
| PL5 | 805 | deterministic | 18.25 | 0.793 | 0.250 | 0.278 | 11 | 1, 2 | 2 |
| PL6 | 806 | sampled | 11.25 | 0.489 | 0.000 | 0.580 | 37 | 1, 2, 3, 5, 128, 196 | 8 |
| PL6 | 806 | deterministic | 9.25 | 0.402 | 0.000 | 0.485 | 11 | 1, 2, 3, 166 | 5 |

Four-seed aggregate over PL1, PL4, PL5, and PL6:

| Mode | Delivered mean | Delivered min | Bit entropy mean | Success mean |
| --- | ---: | ---: | ---: | ---: |
| sampled | 12.19 | 5.00 | 0.628 | 0.062 |
| deterministic | 12.38 | 9.25 | 0.481 | 0.062 |

## PL5 Confirmation

| Run | Episodes | Mode | Delivered | Fraction | Success | Bit entropy | Distinct values | Major values | Mid bits |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PL5 | 16 | sampled | 16.44 | 0.715 | 0.125 | 0.615 | 51 | 1, 2, 48, 58 | 8 |
| PL5 | 16 | deterministic | 15.25 | 0.663 | 0.188 | 0.326 | 13 | 1, 2 | 4 |
| PL1 | 16 | sampled | 16.31 | 0.709 | 0.062 | 0.489 | 50 | 1, 2, 5, 58 | 7 |
| PL1 | 16 | deterministic | 13.81 | 0.601 | 0.188 | 0.452 | 9 | 1, 2, 5, 7 | 6 |

## Interpretation

PL5 is the best short-polish candidate so far. It beats PL1 on both sampled and
deterministic 16-episode delivery, and sampled behavior still uses multiple
major symbols and all writable bits.

The larger lesson is that write entropy alone is not enough. PL4 has the
highest entropy in the four-episode sweep but the worst sampled delivery. The
forage polish stage can either stabilize useful communication or create noisy
symbol use depending on seed.

The promoted `2_500`-update polish is still the safer default across sources.
PL5 suggests that a seed/checkpoint selection gate may outperform a fixed final
polish recipe.

## Next Step

Test whether polish seed `805` is generally good or only good for PV3:

- run a PV1 consolidated short polish with seed `805`;
- run a PV2 consolidated short polish with seed `805`;
- compare against PL2 and PL3, which used source-matched seeds `801` and `802`.

If seed `805` also improves PV1/PV2, promote a fixed polish seed. If not, test
a small validation gate that chooses between multiple polish seeds per
consolidated checkpoint.
