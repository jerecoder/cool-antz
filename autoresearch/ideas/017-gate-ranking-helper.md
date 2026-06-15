# Gate Ranking Helper

## Summary

The autoresearch CLI now has a reusable communication probe ranking command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-rank \
  --phase polish_length
```

It ranks completed probe artifacts with:

```text
gate_score = mean(sampled_delivered, deterministic_delivered)
safety_metric = min(sampled_delivered, deterministic_delivered)
```

This matches the lesson from the PV1 seed sweep: a sampled-only or
deterministic-only selector can choose brittle seeds.

## Short-Polish Eval4 Ranking

Command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-rank \
  --phase polish_length
```

Top twelve completed candidates:

| Rank | Run | Seed | Source | Gate score | Min delivered | Sampled | Deterministic | Mean bit entropy |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PL5 | 805 | PV3 consolidated | 16.62 | 15.00 | 15.00 | 18.25 | 0.465 |
| 2 | PL7 | 805 | PV1 consolidated | 16.00 | 15.00 | 17.00 | 15.00 | 0.253 |
| 3 | PL10 | 804 | PV2 consolidated | 15.25 | 13.00 | 17.50 | 13.00 | 0.270 |
| 4 | PL14 | 804 | PV1 consolidated | 15.00 | 11.00 | 19.00 | 11.00 | 0.236 |
| 5 | PL1 | 803 | PV3 consolidated | 14.50 | 11.50 | 17.50 | 11.50 | 0.497 |
| 6 | PL9 | 803 | PV2 consolidated | 14.12 | 12.50 | 15.75 | 12.50 | 0.391 |
| 7 | PL13 | 803 | PV1 consolidated | 13.12 | 9.75 | 9.75 | 16.50 | 0.207 |
| 8 | PL12 | 802 | PV1 consolidated | 12.75 | 11.75 | 13.75 | 11.75 | 0.227 |
| 9 | PL11 | 806 | PV2 consolidated | 11.25 | 10.75 | 10.75 | 11.75 | 0.228 |
| 10 | PL3 | 802 | PV2 consolidated | 11.12 | 10.25 | 12.00 | 10.25 | 0.316 |
| 11 | PL15 | 806 | PV1 consolidated | 10.62 | 8.50 | 12.75 | 8.50 | 0.283 |
| 12 | PL6 | 806 | PV3 consolidated | 10.25 | 9.25 | 11.25 | 9.25 | 0.532 |

There were no missing eval4 artifacts for the matrix entries.

## Polish-Gate Eval16 Ranking

Command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-rank \
  --phase polish_gate
```

| Rank | Gate | Source | Gate score | Min delivered | Sampled | Deterministic | Mean bit entropy |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PG3 | PL5 | 15.84 | 15.25 | 16.44 | 15.25 | 0.470 |
| 2 | PG2 | PL10 | 14.00 | 13.75 | 14.25 | 13.75 | 0.263 |
| 3 | PG1 | PL7 | 13.56 | 12.19 | 14.94 | 12.19 | 0.247 |

There were no missing eval16 artifacts for the polish-gate entries.

## Interpretation

The balanced gate score selects the same per-source trio already identified by
manual analysis:

- PV1: PL7, seed `805`
- PV2: PL10, seed `804`
- PV3: PL5, seed `805`

It also shows why PL14 and PL13 should not be promoted directly despite strong
single-mode results: their safety metric is much lower than PL7.

## Next Step

Use the ranking helper as the default selection mechanism for future short
polish sweeps. The next useful experiment is to test whether the selected
short-polish checkpoints benefit from a very small additional balanced polish,
without reintroducing the longer full-polish collapse. Start with the best
checkpoint, PL5 / PG3, and keep the probe no-render and sequential.
