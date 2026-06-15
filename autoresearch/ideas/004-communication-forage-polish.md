# Communication Forage Polish

## Summary

A short forage-focused polish stage after long consolidation improved food
delivery beyond the 5,000-update consolidation checkpoint.

Current promoted post-curriculum sequence:

1. 8-bit consolidation: `5_000` updates, `write_bit_entropy_bonus = 0.05`,
   `ent_coef = 0.002`.
2. 8-bit polish: `2_500` updates, no write entropy bonus, no PPO entropy bonus,
   keep the existing pickup and distance shaping.

## Evidence

P0 starts from U2 and trains for `2_500` updates with:

- `write_bit_entropy_bonus = 0.0`
- `ent_coef = 0.0`
- `pickup_bonus = 0.25`
- `distance_bonus = 0.02`

Single-episode polish screen:

| ID | Pickup bonus | Distance bonus | Sampled delivered | Deterministic delivered | Deterministic symbols |
| --- | ---: | ---: | ---: | ---: | ---: |
| P0 | 0.25 | 0.02 | 10.0 | 19.0 | 6 |
| P1 | 0.25 | 0.0 | 11.0 | 0.0 | 7 |
| P2 | 0.0 | 0.0 | 5.0 | 12.0 | 13 |

Four-episode P0 probe:

- sampled delivered: `17.0 / 23`
- sampled write bit entropy: `0.415`
- sampled distinct nonzero values: `31`
- deterministic delivered: `15.0 / 23`
- deterministic write bit entropy: `0.219`
- deterministic distinct nonzero values: `9`

## Interpretation

Removing communication entropy pressure after consolidation helps forage
performance, but removing distance shaping is harmful for deterministic rollout.
The existing pickup and distance shaping should remain during the final polish.

P0 improves sampled delivery versus U2 and keeps deterministic delivery near
the best U2/V1 range while preserving several nonzero communication symbols.

## Next Step

Validate the full promoted notebook sequence end-to-end when wall-clock time is
available, then run a four-episode probe on `8_bits_polished`.
