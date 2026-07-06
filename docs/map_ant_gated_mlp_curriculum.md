# Gated Map-Ant MLP Curriculum

This historical experiment preserves the original JAX MAPPO curriculum that grew
both map size and ant count while using validation gates before advancing. It is
kept for reproducibility and interpretation, not as the strongest current result.

## Reproduction Surface

- Config: `experiments/map_ant_gated_mlp_curriculum.json`
- Notebook: `notebooks/historical/map_ant_gated_mlp_curriculum.ipynb`
- Workflow: `ant-byte train jax --config experiments/map_ant_gated_mlp_curriculum.json`
- Historical source: `research/direct-goal-repro-sweep`

## Design

- Stage plan: `4:1,6:1,8:2,10:2,12:3,16:4,20:5,25:6,32:8,40:10,50:10`.
- Actor vision radius stayed `1`.
- Food and hub stayed randomized.
- One writable bit stayed enabled and writes were allowed while moving.
- The critic architecture was the original MLP.
- Stages advanced only after deterministic and sampled validation gates.

## Historical Result

The strict gated run passed through `20x20_5_ants` and then plateaued around
`25x25_6_ants`. The best 25x25 attempts were close but placement-sensitive, so
the sampled gate exposed a real generalization issue. The run did not solve
`50x50`.

The later `autoresearch/map-ant-12x12-conv-critic` branch tested a new critic and
failed earlier. That branch remains evidence only; this document preserves the
older MLP curriculum as a separate historical experiment.
