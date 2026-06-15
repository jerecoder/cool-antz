# Polish Gate Eval16 Comparison

## Summary

The first-class `polish_gate` candidates confirm the best short-polish models
under the same sixteen-episode probe protocol as the promoted baseline.

Plain-name mapping:

- `PG1` means `PL7`: PV1 consolidated checkpoint, short polish seed `805`.
- `PG2` means `PL10`: PV2 consolidated checkpoint, short polish seed `804`.
- `PG3` means `PL5`: PV3 consolidated checkpoint, short polish seed `805`.

These are not new training runs. They are probe-only gate entries that evaluate
already-trained checkpoints with `--num-episodes 16 --no-render`.

## Commands

Probe-only gate runs:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_gate \
  --id PG1 \
  --probe-episodes 16 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_gate \
  --id PG2 \
  --probe-episodes 16 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_gate \
  --id PG3 \
  --probe-episodes 16 \
  --no-render
```

Promoted baseline confirmation probes:

```bash
PYTHONPATH=src ant-byte probe communication \
  --checkpoint runs/autoresearch/communication_bits/promoted/PV1/8_bits_polished/checkpoints/model.pkl \
  --output-dir runs/autoresearch/communication_bits/promoted/PV1/probe_eval16 \
  --num-episodes 16 \
  --no-render

PYTHONPATH=src ant-byte probe communication \
  --checkpoint runs/autoresearch/communication_bits/promoted/PV2/8_bits_polished/checkpoints/model.pkl \
  --output-dir runs/autoresearch/communication_bits/promoted/PV2/probe_eval16 \
  --num-episodes 16 \
  --no-render

PYTHONPATH=src ant-byte probe communication \
  --checkpoint runs/autoresearch/communication_bits/promoted/PV3/8_bits_polished/checkpoints/model.pkl \
  --output-dir runs/autoresearch/communication_bits/promoted/PV3/probe_eval16 \
  --num-episodes 16 \
  --no-render
```

## Results

Polish-gate candidates:

| Candidate | Source | Mode | Delivered | Fraction | Success | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PG1 | PL7 / PV1 seed 805 | sampled | 14.94 | 0.649 | 0.250 | 0.246 | 14 | 1, 2 | 2 |
| PG1 | PL7 / PV1 seed 805 | deterministic | 12.19 | 0.530 | 0.188 | 0.247 | 4 | 1, 2 | 2 |
| PG2 | PL10 / PV2 seed 804 | sampled | 14.25 | 0.620 | 0.062 | 0.285 | 33 | 2, 4, 5 | 5 |
| PG2 | PL10 / PV2 seed 804 | deterministic | 13.75 | 0.598 | 0.000 | 0.241 | 9 | 2, 4, 5 | 4 |
| PG3 | PL5 / PV3 seed 805 | sampled | 16.44 | 0.715 | 0.125 | 0.615 | 51 | 1, 2, 48, 58 | 8 |
| PG3 | PL5 / PV3 seed 805 | deterministic | 15.25 | 0.663 | 0.188 | 0.326 | 13 | 1, 2 | 4 |

Previously promoted baseline:

| Candidate | Source | Mode | Delivered | Fraction | Success | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PV1 | promoted full polish | sampled | 13.31 | 0.579 | 0.062 | 0.257 | 20 | 1, 2 | 2 |
| PV1 | promoted full polish | deterministic | 13.38 | 0.582 | 0.125 | 0.507 | 8 | 1, 2, 206 | 6 |
| PV2 | promoted full polish | sampled | 14.94 | 0.649 | 0.062 | 0.395 | 20 | 2, 4, 5 | 5 |
| PV2 | promoted full polish | deterministic | 12.06 | 0.524 | 0.062 | 0.314 | 9 | 2, 4, 5 | 3 |
| PV3 | promoted full polish | sampled | 13.12 | 0.571 | 0.062 | 0.551 | 31 | 1, 2, 48 | 8 |
| PV3 | promoted full polish | deterministic | 12.31 | 0.535 | 0.062 | 0.517 | 11 | 1, 2, 48, 58 | 6 |

Aggregate:

| Set | Mode | Delivered mean | Delivered min | Delivered max | Bit entropy mean |
| --- | --- | ---: | ---: | ---: | ---: |
| polish gate | sampled | 15.21 | 14.25 | 16.44 | 0.382 |
| polish gate | deterministic | 13.73 | 12.19 | 15.25 | 0.271 |
| promoted baseline | sampled | 13.79 | 13.12 | 14.94 | 0.401 |
| promoted baseline | deterministic | 12.58 | 12.06 | 13.38 | 0.446 |

## Interpretation

The short-polish gate improves forage delivery over the previously promoted
full-polish models on this eval16 protocol:

- sampled delivery improves by `+1.42` food on average.
- deterministic delivery improves by `+1.15` food on average.
- the worst sampled short-polish gate candidate, `14.25`, is better than two of
  the three promoted sampled candidates.
- `PG3` / `PL5` is the strongest single checkpoint: `16.44` sampled and `15.25`
  deterministic delivered food out of `23`.

The remaining tradeoff is deterministic communication diversity. The promoted
baseline has higher deterministic bit entropy on average (`0.446` vs `0.271`),
while the short-polish gate has better task performance. `PG3` is the best
balanced current checkpoint because sampled communication remains broad
(`0.615` entropy, `51` distinct nonzero values, all `8` mid-entropy bits), even
though deterministic major symbols are still mostly compact.

## Next Step

Treat per-source short-polish selection as the current best recipe. The next
research step is to make the gate practical:

1. For each consolidated checkpoint, run several cheap short-polish seeds.
2. Select the seed with a small validation probe, probably four episodes.
3. Confirm only the selected seed with a sixteen-episode probe.

This tests whether the good result is reproducible as a selection procedure
rather than hand-picked after looking at many probes.
