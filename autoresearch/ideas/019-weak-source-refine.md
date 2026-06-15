# Weak-Source Refine

## Summary

PR4 through PR7 tested whether tiny extra polish helps the weaker selected
short-polish sources:

- PV1 selected checkpoint: PL7
- PV2 selected checkpoint: PL10

It does not. Combined with PR1 through PR3 from PL5, all tested extra-refine
variants are worse than their source baselines by the balanced gate score.

## Runs

| Run | Source | Updates | Write bit entropy bonus | Entropy coef | Probe |
| --- | --- | ---: | ---: | ---: | --- |
| PR4 | PL7 | 250 | 0.00 | 0.000 | eval4 |
| PR5 | PL7 | 250 | 0.02 | 0.001 | eval4 |
| PR6 | PL10 | 250 | 0.00 | 0.000 | eval4 |
| PR7 | PL10 | 250 | 0.02 | 0.001 | eval4 |

Commands:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_refine \
  --id PR4 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_refine \
  --id PR5 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_refine \
  --id PR6 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_refine \
  --id PR7 \
  --probe-episodes 4 \
  --no-render
```

Ranking command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-rank \
  --phase polish_refine
```

## Results

Refine candidates:

| Rank | Run | Source | Gate score | Min delivered | Sampled | Deterministic | Mean bit entropy |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PR2 | PL5 | 15.12 | 13.50 | 13.50 | 16.75 | 0.471 |
| 2 | PR3 | PL5 | 14.88 | 14.75 | 14.75 | 15.00 | 0.514 |
| 3 | PR5 | PL7 | 14.12 | 8.25 | 20.00 | 8.25 | 0.211 |
| 4 | PR1 | PL5 | 9.50 | 8.25 | 8.25 | 10.75 | 0.605 |
| 5 | PR6 | PL10 | 9.12 | 7.75 | 10.50 | 7.75 | 0.191 |
| 6 | PR4 | PL7 | 8.62 | 5.75 | 11.50 | 5.75 | 0.216 |
| 7 | PR7 | PL10 | 8.00 | 6.25 | 9.75 | 6.25 | 0.278 |

Source baselines:

| Source | Gate score | Min delivered | Sampled | Deterministic | Mean bit entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| PL5 | 16.62 | 15.00 | 15.00 | 18.25 | 0.465 |
| PL7 | 16.00 | 15.00 | 17.00 | 15.00 | 0.253 |
| PL10 | 15.25 | 13.00 | 17.50 | 13.00 | 0.270 |

## Interpretation

Extra refine is not just failing on the strongest checkpoint. It also fails on
the weaker selected sources:

- PL7 + guarded refine produces a high sampled score (`20.00`) but collapses
  deterministic delivery (`8.25`), so it is brittle.
- PL10 refine is bad in both pure and guarded variants.
- The bit-entropy guard does not reliably preserve useful task behavior.

The selected short-polish checkpoints should be treated as endpoints, not as
starting points for another polish tail.

## Next Step

Stop testing extra refine tails unless the objective or optimizer changes.

The next improvement path should happen before the selected checkpoint, not
after it. Good candidates:

1. Improve the consolidation checkpoint that the seed gate starts from.
2. Try a new consolidated source seed and run the cheap short-polish seed gate.
3. Test whether a different validation metric predicts eval16 better than the
   current eval4 balanced score.
