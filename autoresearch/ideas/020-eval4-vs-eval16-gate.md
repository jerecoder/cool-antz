# Eval4 Versus Eval16 Gate

## Summary

The balanced eval4 gate score is useful for screening, but it should not be the
final promotion metric. On short-polish candidates that have both eval4 and
eval16 probes, eval4 score correlates well with eval16 score, but individual
candidates can still move by more than two delivered food.

## Compared Candidates

Candidates with both probes:

| Run | Eval4 score | Eval4 min | Eval4 sampled | Eval4 deterministic | Eval16 score | Eval16 min | Eval16 sampled | Eval16 deterministic | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PL1 | 14.50 | 11.50 | 17.50 | 11.50 | 15.06 | 13.81 | 16.31 | 13.81 | +0.56 |
| PL10 | 15.25 | 13.00 | 17.50 | 13.00 | 14.00 | 13.75 | 14.25 | 13.75 | -1.25 |
| PL2 | 9.00 | 8.25 | 8.25 | 9.75 | 9.19 | 8.94 | 9.44 | 8.94 | +0.19 |
| PL3 | 11.12 | 10.25 | 12.00 | 10.25 | 12.50 | 10.88 | 14.12 | 10.88 | +1.38 |
| PL5 | 16.62 | 15.00 | 15.00 | 18.25 | 15.84 | 15.25 | 16.44 | 15.25 | -0.78 |
| PL7 | 16.00 | 15.00 | 17.00 | 15.00 | 13.56 | 12.19 | 14.94 | 12.19 | -2.44 |
| PL8 | 10.00 | 8.75 | 11.25 | 8.75 | 11.06 | 10.25 | 10.25 | 11.88 | +1.06 |

Correlation:

| Metric | Pearson correlation |
| --- | ---: |
| Balanced gate score | 0.912 |
| Min delivered safety metric | 0.816 |

## Interpretation

Eval4 is good enough to reject clearly bad candidates and to pick likely
finalists. It correctly keeps the strong PL5/PL7/PL10 group above the weak
PL2/PL3/PL8 group.

Eval4 is not stable enough for final ordering:

- PL7 drops from `16.00` eval4 score to `13.56` eval16 score.
- PL10 drops from `15.25` to `14.00`.
- PL3 improves from `11.12` to `12.50`.

The selection procedure should therefore stay two-stage:

1. use eval4 for cheap seed screening;
2. use eval16 before promotion or final comparison.

## Next Step

Keep the current gate rule, but treat eval4 as a screening filter only. A
candidate should not be promoted unless it also passes eval16.

For the next substantial improvement, prefer a new consolidated source or an
improved consolidation recipe. Post-selection refine is now a low-priority
branch because every tested refine candidate underperformed its source
checkpoint.
