# Promoted Validation PV2

## Summary

PV2 completed the full promoted sequence and broadly replicated PV1: the final
polished checkpoint has useful deterministic forage behavior and multiple major
write symbols, while sampled rollout remains weaker than the earlier P0 polish
screen.

Command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase promoted_validation \
  --id PV2 \
  --probe-episodes 4 \
  --no-render
```

Final checkpoint:

```text
runs/autoresearch/communication_bits/promoted/PV2/8_bits_polished/checkpoints/model.pkl
```

## Four-Episode Probe Results

| Run | Mode | Delivered | Fraction | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| PV1 polished | sampled | 13.25 | 0.576 | 0.247 | 15 | 1, 2 | 2 |
| PV1 polished | deterministic | 15.50 | 0.674 | 0.634 | 8 | 1, 2, 206 | 6 |
| PV2 polished | sampled | 10.00 | 0.435 | 0.410 | 15 | 2, 4, 5 | 3 |
| PV2 polished | deterministic | 13.75 | 0.598 | 0.321 | 8 | 2, 4, 5 | 3 |

## Interpretation

The promoted sequence has now produced decent deterministic delivery in two
independent seeds. PV2 is weaker than PV1 on both sampled and deterministic
delivery, but it still avoids the old deterministic collapse and uses three
major nonzero write symbols.

The repeated weakness is sampled forage quality. The next tuning target should
not be raw symbol diversity; it should be improving sampled delivery while
preserving the deterministic multi-symbol behavior.

## Next Step

Run PV3 before changing the recipe. If PV3 also lands in the same range, record
the three-seed aggregate and then test a final-polish mutation, such as a longer
polish stage or a slightly lower sampled-action entropy pressure late in
training.
