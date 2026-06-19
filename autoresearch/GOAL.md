# Active Autoresearch Goal

Improve the JAX MAPPO forage learner beyond the current weak 25x25 curriculum result, then promote only candidates that survive the 16x16 to 25x25 scale cliff.

The previous autoresearch matrices and generated runs were removed. The active loop is now `autoresearch/loop.json`; every run writes a self-contained `experiment.md`, `plan.json`, and `summary.json` under `runs/autoresearch/forage_loop/<experiment-id>/`.

## Current Diagnosis

The current single-ant, radius-1, feed-forward setup learns small maps but degrades hard as the map grows. The strongest immediate hypotheses are:

- Exploration/capacity is too low for sparse larger maps.
- Local observation is too narrow once food and hub are far apart.
- Credit assignment is too short for delivery after pickup.
- Food distribution may be too diffuse for early robust skill acquisition.
- Autocurriculum may help only if the budget and reward signal are shaped around active-size progress.

## Promotion Gate

The first gate is 25x25, not 50x50. A candidate should beat the current 25x25-level behavior by a large margin before spending on 26x26 through 50x50.

Primary signal: final 25x25 `episode_return`, deliveries, pickups, and remaining food from the saved run summary.
