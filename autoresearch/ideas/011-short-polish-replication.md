# Short Polish Replication

## Summary

PL1 showed that `1_250` updates of forage polish can be very strong from the
PV3 consolidated checkpoint. PL2 and PL3 tested the same short polish from the
PV1 and PV2 consolidated checkpoints. The improvement did not replicate.

Commands:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL2 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL3 \
  --probe-episodes 4 \
  --no-render
```

## Result

| Run | Source | Mode | Delivered | Fraction | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| PV1 2,500 polish | PV1 consolidated | sampled | 13.25 | 0.576 | 0.247 | 15 | 1, 2 | 2 |
| PV1 2,500 polish | PV1 consolidated | deterministic | 15.50 | 0.674 | 0.634 | 8 | 1, 2, 206 | 6 |
| PV2 2,500 polish | PV2 consolidated | sampled | 10.00 | 0.435 | 0.410 | 15 | 2, 4, 5 | 6 |
| PV2 2,500 polish | PV2 consolidated | deterministic | 13.75 | 0.598 | 0.321 | 8 | 2, 4, 5 | 3 |
| PV3 2,500 polish | PV3 consolidated | sampled | 14.00 | 0.609 | 0.492 | 24 | 1, 2, 48 | 8 |
| PV3 2,500 polish | PV3 consolidated | deterministic | 11.50 | 0.500 | 0.535 | 8 | 1, 2, 48, 58 | 5 |
| PL1 1,250 polish | PV3 consolidated | sampled | 17.50 | 0.761 | 0.526 | 32 | 1, 2, 5, 58 | 7 |
| PL1 1,250 polish | PV3 consolidated | deterministic | 11.50 | 0.500 | 0.468 | 8 | 1, 2, 5, 7, 58 | 6 |
| PL2 1,250 polish | PV1 consolidated | sampled | 8.25 | 0.359 | 0.284 | 24 | 1, 2 | 6 |
| PL2 1,250 polish | PV1 consolidated | deterministic | 9.75 | 0.424 | 0.252 | 7 | 1, 2 | 3 |
| PL3 1,250 polish | PV2 consolidated | sampled | 12.00 | 0.522 | 0.524 | 25 | 2, 4, 5, 214 | 7 |
| PL3 1,250 polish | PV2 consolidated | deterministic | 10.25 | 0.446 | 0.108 | 7 | 4, 5 | 3 |

Aggregate over the three promoted checkpoints:

| Recipe | Sampled delivered | Deterministic delivered | Sampled bit entropy | Deterministic bit entropy |
| --- | ---: | ---: | ---: | ---: |
| 2,500-update polish | 12.42 | 13.58 | 0.383 | 0.497 |
| 1,250-update polish | 12.58 | 10.50 | 0.445 | 0.276 |

## Interpretation

The shorter polish is not a robust promotion candidate. It slightly improves
mean sampled delivery because PL1 is excellent, but it damages deterministic
delivery on average and collapses communication diversity in PL2/PL3.

Current best robust recipe remains the promoted `2_500`-update polish. The next
useful direction is not simply shortening polish globally. Better candidates
are source-aware polish selection, improving the consolidated checkpoint before
polish, or adding a small validation gate that chooses between `1_250` and
`2_500` polish per seed.

## Next Step

Treat PL1 as a clue rather than a winner. The next experiment should test
whether the PV3 gain comes from checkpoint state, polish seed, or probe noise.
Run either:

- a 16-episode probe for PL2 and PL3 if cheap confirmation is needed, or
- a same-source polish-seed sweep from the PV3 consolidated checkpoint to test
  whether short-polish performance is reproducible.
