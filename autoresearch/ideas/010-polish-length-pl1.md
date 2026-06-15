# Polish Length PL1

## Summary

PL1 tested a shorter `1_250`-update forage polish from the PV3 consolidated
checkpoint. This outperformed the `5_000`-update PL0 run and produced the best
sampled delivery seen in the promoted-validation family so far.

Command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL1 \
  --probe-episodes 4 \
  --no-render
```

Additional confirmation probe:

```bash
PYTHONPATH=src ant-byte probe communication \
  --checkpoint runs/autoresearch/communication_bits/polish_length/PL1/8_bits/checkpoints/model.pkl \
  --output-dir runs/autoresearch/communication_bits/polish_length/PL1/probe_eval16 \
  --num-episodes 16 \
  --no-render
```

Source checkpoint:

```text
runs/autoresearch/communication_bits/promoted/PV3/8_bits_consolidated/checkpoints/model.pkl
```

## Result

| Run | Mode | Episodes | Delivered | Fraction | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PV3 2,500 polish | sampled | 4 | 14.00 | 0.609 | 0.492 | 24 | 1, 2, 48 | 8 |
| PV3 2,500 polish | deterministic | 4 | 11.50 | 0.500 | 0.535 | 8 | 1, 2, 48, 58 | 5 |
| PL0 5,000 polish | sampled | 4 | 12.25 | 0.533 | 0.596 | 32 | 1, 2, 58, 128 | 8 |
| PL0 5,000 polish | deterministic | 4 | 6.75 | 0.293 | 0.654 | 13 | 1, 2, 58, 128 | 7 |
| PL1 1,250 polish | sampled | 4 | 17.50 | 0.761 | 0.526 | 32 | 1, 2, 5, 58 | 7 |
| PL1 1,250 polish | deterministic | 4 | 11.50 | 0.500 | 0.468 | 8 | 1, 2, 5, 7, 58 | 6 |
| PL1 1,250 polish | sampled | 16 | 16.31 | 0.709 | 0.489 | 50 | 1, 2, 5, 58 | 7 |
| PL1 1,250 polish | deterministic | 16 | 13.81 | 0.601 | 0.452 | 9 | 1, 2, 5, 7 | 6 |

## Interpretation

Shorter polish looks better than both the original `2_500`-update polish and
the longer `5_000`-update polish for this source checkpoint. PL1 preserves
multi-symbol communication while improving sampled delivery, and the larger
probe suggests the improvement is not just one lucky four-episode sample.

This is still not a promotion result. It is one source checkpoint, so the next
test is replication from the other promoted consolidated checkpoints.

## Next Step

Run PL2 and PL3: `1_250`-update polish replications from PV1 and PV2
consolidated checkpoints, keeping reward shaping fixed.
