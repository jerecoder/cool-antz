# Promoted Validation PV1

## Summary

PV1 completed the full promoted sequence end-to-end from the 25x25 forage
checkpoint. The result supports the consolidation-plus-polish structure, but it
needs replication before promotion is considered robust.

Command:

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase promoted_validation \
  --id PV1 \
  --probe-episodes 4 \
  --no-render
```

Final checkpoint:

```text
runs/autoresearch/communication_bits/promoted/PV1/8_bits_polished/checkpoints/model.pkl
```

## Four-Episode Probe Results

| Checkpoint | Mode | Delivered | Bit entropy | Distinct values | Major values | Mid bits |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| raw 8-bit | sampled | 6.25 | 0.926 | 214 | 2 | 8 |
| raw 8-bit | deterministic | 3.75 | 0.058 | 14 | 7 | 0 |
| consolidated | sampled | 9.00 | 0.392 | 23 | 1, 2, 11 | 4 |
| consolidated | deterministic | 2.50 | 0.202 | 5 | 1, 2 | 2 |
| polished | sampled | 13.25 | 0.247 | 15 | 1, 2 | 2 |
| polished | deterministic | 15.50 | 0.634 | 8 | 1, 2, 206 | 6 |

## Interpretation

The raw 8-bit checkpoint again shows the known failure mode: sampled behavior is
very diverse, but deterministic rollout has low forage delivery and almost no
bit entropy.

For this seed, consolidation alone did not repair deterministic forage behavior.
The polish stage was the decisive step: deterministic delivery rose to
`15.5 / 23` and deterministic write usage spread across three major symbols.

Compared with the earlier P0 result, PV1 has similar deterministic delivery but
weaker sampled delivery (`13.25 / 23` versus `17.0 / 23`). That makes the recipe
promising but not settled.

## Next Step

Run the same promoted validation recipe for PV2 and PV3. If deterministic
delivery stays near the PV1/P0/U2 range across seeds, keep the structure and
tune only the final polish length or sampled-policy quality.
