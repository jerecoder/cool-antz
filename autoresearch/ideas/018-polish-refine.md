# Polish Refine

## Summary

PR1, PR2, and PR3 tested whether the current best short-polish checkpoint,
PL5 / PG3, benefits from a tiny additional polish stage.

It does not. None of the refine candidates beat the PL5 eval4 baseline. Extra
polish from PL5 appears harmful or at least not useful under these settings.

## Runs

All runs started from:

```text
runs/autoresearch/communication_bits/polish_length/PL5/8_bits/checkpoints/model.pkl
```

| Run | Updates | Write bit entropy bonus | Entropy coef | Probe |
| --- | ---: | ---: | ---: | --- |
| PR1 | 250 | 0.00 | 0.000 | eval4 |
| PR2 | 500 | 0.00 | 0.000 | eval4 |
| PR3 | 250 | 0.02 | 0.001 | eval4 |

Commands:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_refine \
  --id PR1 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_refine \
  --id PR2 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_refine \
  --id PR3 \
  --probe-episodes 4 \
  --no-render
```

Ranking command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-rank \
  --phase polish_refine
```

## Results

| Run | Gate score | Min delivered | Sampled | Deterministic | Mean bit entropy | Sampled major values | Deterministic major values |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| PL5 baseline | 16.62 | 15.00 | 15.00 | 18.25 | 0.465 | 1, 2, 48, 58, 196 | 1, 2 |
| PR2 | 15.12 | 13.50 | 13.50 | 16.75 | 0.471 | 1, 2, 48, 196 | 1, 2 |
| PR3 | 14.88 | 14.75 | 14.75 | 15.00 | 0.514 | 1, 2, 48, 58, 96 | 1, 2, 152 |
| PR1 | 9.50 | 8.25 | 8.25 | 10.75 | 0.605 | 1, 2, 48, 58 | 1, 2, 48, 58 |

## Interpretation

The short-polish checkpoint is already in a good local region. Continuing
polish from that checkpoint can preserve or even increase write-symbol entropy
while damaging delivered food. That means the failure is not just communication
collapse; the task policy itself is being moved away from the useful behavior.

PR3 is the most balanced refine candidate by safety metric, but it still loses
to the PL5 baseline on gate score and deterministic delivery. PR2 is the best
refine by gate score, but it also loses to PL5.

## Next Step

Do not promote extra PL5 refine stages. Keep PL5 / PG3 as the current best
single checkpoint.

The next useful direction is to improve the weaker selected sources, especially
PV1/PL7 and PV2/PL10, or to test whether the seed-gate recipe generalizes to a
new consolidated source. Avoid spending more runs on extra polish from PL5
unless the reward or optimizer setup changes substantially.
