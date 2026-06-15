# Promoted Validation Aggregate

## Summary

PV1, PV2, and PV3 completed the full promoted communication sequence. The
recipe consistently avoids the old deterministic write-symbol collapse and
keeps multiple major nonzero symbols, but forage delivery is still only
midrange.

Sequence:

1. staged communication bits `2 -> 3 -> 5 -> 8`
2. `5_000`-update consolidation with small bit-entropy pressure
3. `2_500`-update forage polish with communication entropy off

## Four-Episode Probe Results

| Run | Mode | Delivered | Fraction | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| PV1 | sampled | 13.25 | 0.576 | 0.247 | 15 | 1, 2 | 2 |
| PV1 | deterministic | 15.50 | 0.674 | 0.634 | 8 | 1, 2, 206 | 6 |
| PV2 | sampled | 10.00 | 0.435 | 0.410 | 15 | 2, 4, 5 | 3 |
| PV2 | deterministic | 13.75 | 0.598 | 0.321 | 8 | 2, 4, 5 | 3 |
| PV3 | sampled | 14.00 | 0.609 | 0.492 | 24 | 1, 2, 48 | 5 |
| PV3 | deterministic | 11.50 | 0.500 | 0.535 | 8 | 1, 2, 48, 58 | 5 |

Aggregate:

- sampled delivery mean: `12.42 / 23`, min: `10.0 / 23`
- deterministic delivery mean: `13.58 / 23`, min: `11.5 / 23`
- sampled bit entropy mean: `0.383`
- deterministic bit entropy mean: `0.497`
- every sampled and deterministic probe used at least two major nonzero values

## Interpretation

The promoted recipe passes the communication-use part of the goal: all three
seeds use multiple major nonzero symbols and several writable bits. It does not
yet pass the forage-performance target. The repeated weakness is not symbol
diversity; it is converting the communication policy into more delivered food.

The final polish stage remains the most plausible lever because the post-8-bit
stages consistently improve task behavior after the high-entropy curriculum.

## Next Step

Run PL0: a longer `5_000`-update polish stage from the completed PV3
consolidated checkpoint. Keep the same polish reward shaping and seed as PV3 so
the main changed variable is polish duration.
