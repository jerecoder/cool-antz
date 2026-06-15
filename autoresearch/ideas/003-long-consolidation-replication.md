# Long Consolidation Replication

## Summary

The `5_000`-update consolidation setting replicated across all three final
communication checkpoints. This supports keeping the promoted notebook/config
consolidation length at `5_000` updates.

Setting:

- Source checkpoints: final `F1`, `F2`, `F3` 8-bit policies.
- Consolidation length: `5_000` updates.
- `write_bit_entropy_bonus = 0.05`
- `ent_coef = 0.002`
- terminal write entropy disabled.

## Four-Episode Probe Results

| Run | Source | Sampled delivered | Sampled bit entropy | Deterministic delivered | Deterministic bit entropy |
| --- | --- | ---: | ---: | ---: | ---: |
| V1 | F1 | 6.25 | 0.364 | 13.5 | 0.570 |
| V2 | F2 | 10.25 | 0.242 | 10.5 | 0.283 |
| U2 | F3 | 9.75 | 0.560 | 15.5 | 0.375 |

Aggregate deterministic behavior:

- delivered: `13.17 / 23` mean, `10.5 / 23` min
- distinct nonzero values: `9.0` mean, `6` min
- major nonzero symbols: `3.67` mean, `3` min
- mid-activation writable bits: `3.67` mean, `3` min
- write bit entropy: `0.409` mean

Aggregate sampled behavior:

- delivered: `8.75 / 23` mean
- distinct nonzero values: `33.0` mean
- write bit entropy: `0.389` mean

## Interpretation

Long consolidation consistently repairs deterministic delivery compared with
the pre-consolidation final checkpoints while preserving multi-symbol writes.
The sampled policy is still weaker and noisier than deterministic, so current
promotion should optimize for deterministic rollout behavior.

## Next Step

The next improvement target is full task success, not further symbol diversity:
try a forage-focused final polish after long consolidation, or evaluate whether
larger food-delivery reward / lower distance shaping improves completing all
food within the episode limit.
