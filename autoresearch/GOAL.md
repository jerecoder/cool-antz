# Active Autoresearch Goal

Improve the JAX MAPPO forage learner beyond the current weak 25x25 curriculum result, then promote only candidates that survive held-out 25x25 evaluation with shuffled food and hub positions.

The active loop is `autoresearch/loop.json`. It compares reward shaping, ant count, actor vision, gamma/rollout length, food concentration, cookie-distance geometry, stage-size ladders, byte-trail memory shaping, and autocurriculum.

## Current Diagnosis

The current single-ant, radius-1, feed-forward setup learns small maps but degrades hard as the map grows. The main hypotheses are:

- Exploration throughput is too low for sparse larger maps.
- Radius-1 local observation is too narrow once source-hub routes are long.
- Delivery credit is too delayed for the current gamma and rollout horizon.
- Food and cookie placement may make successful delivery loops too rare.
- External byte memory exists but may need delivery-aligned shaping to become useful.
- Autocurriculum may help only if active-size advancement is directly rewarded and budgeted sanely.

## Promotion Gate

The first gate is 25x25. A candidate should beat the observed weak baseline by a large margin before spending on 26x26 through 50x50.

Primary signal: held-out `eval_mean_episode_return`, `eval_mean_delivered_food`, and delivered fraction from deterministic and sampled evaluation. Training-update return is supporting evidence only.
