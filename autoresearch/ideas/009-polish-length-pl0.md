# Polish Length PL0

## Summary

PL0 tested whether extending the PV3 polish stage from `2_500` to `5_000`
updates improves forage delivery. It did not. Longer polish increased bit usage
but reduced both sampled and deterministic delivery.

Command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL0 \
  --probe-episodes 4 \
  --no-render
```

Source checkpoint:

```text
runs/autoresearch/communication_bits/promoted/PV3/8_bits_consolidated/checkpoints/model.pkl
```

## Result

| Run | Mode | Delivered | Fraction | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| PV3 2,500 polish | sampled | 14.00 | 0.609 | 0.492 | 24 | 1, 2, 48 | 5 |
| PV3 2,500 polish | deterministic | 11.50 | 0.500 | 0.535 | 8 | 1, 2, 48, 58 | 5 |
| PL0 5,000 polish | sampled | 12.25 | 0.533 | 0.596 | 32 | 1, 2, 58, 128 | 8 |
| PL0 5,000 polish | deterministic | 6.75 | 0.293 | 0.654 | 13 | 1, 2, 58, 128 | 6 |

## Interpretation

More polish is not automatically better. PL0 shifted the policy toward even
broader write-bit usage, but that extra communication entropy did not translate
to food delivery. Deterministic delivery regressed sharply.

The polish-length optimum is likely at or below `2_500` updates for this source
checkpoint.

## Next Step

Run PL1: a shorter `1_250`-update polish from the same PV3 consolidated
checkpoint, keeping reward shaping and seed fixed.
