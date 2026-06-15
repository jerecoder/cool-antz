# Fixed Seed 805 Cross-Source Check

## Summary

PL7 and PL8 tested whether the strong PL5 polish seed, `805`, generalizes from
the PV3 consolidated checkpoint to the PV1 and PV2 consolidated checkpoints.

It partly generalizes. Seed `805` improves PV1 strongly and keeps PV3 strong,
but PV2 remains the weak source. The fixed-seed recipe is competitive with the
promoted `2_500`-update polish, especially for sampled delivery, but it does
not clearly dominate because PV2 deterministic delivery and deterministic bit
diversity remain weaker.

## Runs

| Run | Source checkpoint | Polish seed | Updates |
| --- | --- | ---: | ---: |
| PL7 | `promoted/PV1/8_bits_consolidated` | 805 | 1,250 |
| PL8 | `promoted/PV2/8_bits_consolidated` | 805 | 1,250 |
| PL5 | `promoted/PV3/8_bits_consolidated` | 805 | 1,250 |

Commands:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL7 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL8 \
  --probe-episodes 4 \
  --no-render
```

Additional confirmation probes:

```bash
PYTHONPATH=src ant-byte probe communication \
  --checkpoint runs/autoresearch/communication_bits/polish_length/PL7/8_bits/checkpoints/model.pkl \
  --output-dir runs/autoresearch/communication_bits/polish_length/PL7/probe_eval16 \
  --num-episodes 16 \
  --no-render

PYTHONPATH=src ant-byte probe communication \
  --checkpoint runs/autoresearch/communication_bits/polish_length/PL8/8_bits/checkpoints/model.pkl \
  --output-dir runs/autoresearch/communication_bits/polish_length/PL8/probe_eval16 \
  --num-episodes 16 \
  --no-render
```

## Four-Episode Result

| Run | Source | Mode | Delivered | Fraction | Success | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PL2 | PV1, source seed | sampled | 8.25 | 0.359 | 0.000 | 0.284 | 24 | 1, 2 | 6 |
| PL2 | PV1, source seed | deterministic | 9.75 | 0.424 | 0.000 | 0.252 | 7 | 1, 2 | 3 |
| PL7 | PV1, seed 805 | sampled | 17.00 | 0.739 | 0.250 | 0.270 | 13 | 1, 2 | 2 |
| PL7 | PV1, seed 805 | deterministic | 15.00 | 0.652 | 0.250 | 0.236 | 4 | 1, 2 | 2 |
| PL3 | PV2, source seed | sampled | 12.00 | 0.522 | 0.000 | 0.524 | 25 | 2, 4, 5, 214 | 7 |
| PL3 | PV2, source seed | deterministic | 10.25 | 0.446 | 0.000 | 0.108 | 7 | 4, 5 | 3 |
| PL8 | PV2, seed 805 | sampled | 11.25 | 0.489 | 0.000 | 0.513 | 35 | 2, 4, 5 | 8 |
| PL8 | PV2, seed 805 | deterministic | 8.75 | 0.380 | 0.000 | 0.332 | 11 | 2, 5 | 3 |
| PL5 | PV3, seed 805 | sampled | 15.00 | 0.652 | 0.000 | 0.651 | 39 | 1, 2, 48, 58, 196 | 8 |
| PL5 | PV3, seed 805 | deterministic | 18.25 | 0.793 | 0.250 | 0.278 | 11 | 1, 2 | 2 |

Four-episode aggregate:

| Recipe | Sampled delivered | Sampled min | Deterministic delivered | Deterministic min | Sampled entropy | Deterministic entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted 2,500 polish | 12.42 | 10.00 | 13.58 | 11.50 | 0.383 | 0.497 |
| source-seed 1,250 polish | 11.75 | 8.25 | 12.75 | 9.75 | 0.486 | 0.213 |
| fixed-805 1,250 polish | 14.42 | 11.25 | 14.00 | 8.75 | 0.478 | 0.282 |

## Sixteen-Episode Confirmation

| Run | Source | Mode | Delivered | Fraction | Success | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PL7 | PV1, seed 805 | sampled | 14.94 | 0.649 | 0.250 | 0.246 | 14 | 1, 2 | 2 |
| PL7 | PV1, seed 805 | deterministic | 12.19 | 0.530 | 0.188 | 0.247 | 4 | 1, 2 | 2 |
| PL8 | PV2, seed 805 | sampled | 10.25 | 0.446 | 0.000 | 0.548 | 39 | 2, 4, 5 | 8 |
| PL8 | PV2, seed 805 | deterministic | 11.88 | 0.516 | 0.125 | 0.390 | 12 | 2, 4, 5 | 4 |
| PL5 | PV3, seed 805 | sampled | 16.44 | 0.715 | 0.125 | 0.615 | 51 | 1, 2, 48, 58 | 8 |
| PL5 | PV3, seed 805 | deterministic | 15.25 | 0.663 | 0.188 | 0.326 | 13 | 1, 2 | 4 |

Fixed-805 eval16 aggregate:

| Mode | Delivered mean | Delivered min | Bit entropy mean |
| --- | ---: | ---: | ---: |
| sampled | 13.88 | 10.25 | 0.470 |
| deterministic | 13.10 | 11.88 | 0.321 |

## Interpretation

Seed `805` is a useful polish seed but not a full solution. It raises sampled
delivery over the promoted recipe and keeps the deterministic minimum slightly
above the promoted four-episode minimum, but deterministic average delivery is
still a bit lower and deterministic communication entropy is lower.

The main bottleneck is PV2. PL8's larger probe improved deterministic delivery
relative to the four-episode read, but sampled delivery remained weak.

## Next Step

Run a PV2-specific short-polish seed sweep. Keep the source checkpoint fixed at
`promoted/PV2/8_bits_consolidated`, and test nearby seeds that have not been
used on PV2 yet:

- seed `803`
- seed `804`
- seed `806`

If one PV2 seed improves delivery without collapsing sampled communication, the
next candidate is a per-source validation gate over short-polish seeds.
