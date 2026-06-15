# Communication Consolidation Duration

## Summary

A follow-up duration sweep found that the C3-style consolidation stage improves
further when extended from `2_500` to `5_000` updates.

Current promoted setting:

- Start from the 8-bit communication checkpoint.
- Train only the 8-bit consolidation stage.
- Use `5_000` updates (`6_400_000` env steps with `16 * 80` steps/update).
- Use `write_bit_entropy_bonus = 0.05`.
- Use `ent_coef = 0.002`.
- Keep terminal write entropy disabled.

## Evidence

Single-episode duration sweep from the F3 8-bit final checkpoint:

| ID | Updates | Sampled delivered | Sampled bit entropy | Deterministic delivered | Deterministic bit entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| U0 | 625 | 3.0 | 0.354 | 11.0 | 0.175 |
| U1 | 1,250 | 2.0 | 0.434 | 8.0 | 0.435 |
| K3 | 2,500 | 13.0 | 0.602 | 11.0 | 0.348 |
| U2 | 5,000 | 9.0 | 0.539 | 16.0 | 0.309 |

Four-episode probe for U2:

- sampled delivered: `9.75 / 23`
- sampled write bit entropy: `0.560`
- sampled distinct nonzero values: `51`
- deterministic delivered: `15.5 / 23`
- deterministic write bit entropy: `0.375`
- deterministic distinct nonzero values: `11`

## Interpretation

The longer consolidation stage gives forage reward more time to select useful
symbols after the high-entropy curriculum has made the action space reachable.
The 5,000-update result improves deterministic delivery without collapsing the
write channel to a single symbol.

## Next Step

Run the promoted config end-to-end from the notebook or experiment entrypoint
when enough wall-clock time is available, then compare the resulting
`8_bits_consolidated` checkpoint against U2 with a multi-episode probe.
