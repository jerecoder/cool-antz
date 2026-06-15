# PV1 Polish Seed Gate

## Summary

PL12, PL13, PL14, and PL15 completed the missing PV1 short-polish seed sweep.
Together with the earlier PV1 runs, this gives cheap four-episode probes for
PV1 consolidated polish seeds `801` through `806`, except that seed `805` was
already represented by PL7.

The current PV1 best remains PL7, seed `805`. It wins by both balanced average
delivery and worst-case delivery across sampled and deterministic probes.

## Runs

All runs used:

- source checkpoint:
  `runs/autoresearch/communication_bits/promoted/PV1/8_bits_consolidated/checkpoints/model.pkl`
- `global_update_cap = 1_250`
- `write_entropy_bonus = 0.0`
- `write_bit_entropy_bonus = 0.0`
- `ent_coef = 0.0`
- `pickup_bonus = 0.25`
- `distance_bonus = 0.02`

Commands:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL12 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL13 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL14 \
  --probe-episodes 4 \
  --no-render

PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase polish_length \
  --id PL15 \
  --probe-episodes 4 \
  --no-render
```

## Results

| Run | Seed | Sampled delivered | Deterministic delivered | Mean delivered | Min delivered | Sampled entropy | Deterministic entropy | Sampled major values | Deterministic major values |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| PL2 | 801 | 8.25 | 9.75 | 9.00 | 8.25 | 0.284 | 0.252 | 1, 2 | 1, 2 |
| PL12 | 802 | 13.75 | 11.75 | 12.75 | 11.75 | 0.244 | 0.209 | 1, 2 | 1, 2 |
| PL13 | 803 | 9.75 | 16.50 | 13.12 | 9.75 | 0.154 | 0.259 | 1, 2 | 1, 2 |
| PL14 | 804 | 19.00 | 11.00 | 15.00 | 11.00 | 0.273 | 0.199 | 1, 2 | 1, 2 |
| PL7 | 805 | 17.00 | 15.00 | 16.00 | 15.00 | 0.270 | 0.236 | 1, 2 | 1, 2 |
| PL15 | 806 | 12.75 | 8.50 | 10.62 | 8.50 | 0.303 | 0.263 | 1, 2 | 1, 2 |

Best by metric:

| Metric | Winner |
| --- | --- |
| Sampled delivered only | PL14 / seed 804 |
| Deterministic delivered only | PL13 / seed 803 |
| Mean of sampled and deterministic delivered | PL7 / seed 805 |
| Minimum of sampled and deterministic delivered | PL7 / seed 805 |

## Interpretation

PV1 polish is strongly seed-sensitive. A one-mode validation gate would be
fragile: sampled-only selection would choose PL14, while deterministic-only
selection would choose PL13. Both are lopsided.

PL7 is the robust PV1 choice because it does not maximize only one rollout
mode; it keeps both sampled and deterministic delivery high. This supports a
balanced gate score such as:

```text
gate_score = mean(sampled_delivered, deterministic_delivered)
```

with `min(sampled_delivered, deterministic_delivered)` tracked as a safety
metric to reject brittle candidates.

## Current Gate Recipe

The current per-source short-polish selections are:

| Source | Selected run | Seed | Eval4 sampled | Eval4 deterministic | Eval16 sampled | Eval16 deterministic |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PV1 | PL7 | 805 | 17.00 | 15.00 | 14.94 | 12.19 |
| PV2 | PL10 | 804 | 17.50 | 13.00 | 14.25 | 13.75 |
| PV3 | PL5 | 805 | 15.00 | 18.25 | 16.44 | 15.25 |

## Next Step

Formalize the balanced gate score in code so future runs can be ranked from
probe artifacts without hand-building tables. Then use it to summarize all
available short-polish candidates across PV1, PV2, and PV3.
