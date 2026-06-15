# Promoted Communication Validation Plan

## Hypothesis

The promoted notebook recipe should reproduce the U2 plus P0 behavior when run
end-to-end from the 25x25 forage checkpoint, not only when polish starts from
an autoresearch intermediate checkpoint.

## Change

Add a first-class autoresearch matrix entry for the full promoted sequence:

1. Train staged communication bits `2 -> 3 -> 5 -> 8`.
2. Run `8_bits_consolidated` for `5_000` updates.
3. Run `8_bits_polished` for `2_500` updates.
4. Probe the polished checkpoint over four episodes.

## Command

```bash
PYTHONPATH=src ant-byte autoresearch communication-run \
  --phase promoted_validation \
  --id PV1 \
  --probe-episodes 4 \
  --no-render
```

## Outputs

- run directory: `runs/autoresearch/communication_bits/promoted/PV1`
- final checkpoint:
  `runs/autoresearch/communication_bits/promoted/PV1/8_bits_polished/checkpoints/model.pkl`
- probe:
  `runs/autoresearch/communication_bits/promoted/PV1/probe_eval4/communication_probe.json`

## Stop Rule

If PV1 does not roughly match the promoted evidence, compare its intermediate
`8_bits`, `8_bits_consolidated`, and `8_bits_polished` probes against F/U/P
checkpoints before changing reward shaping again.
