# Forage Autoresearch Report

Generated: 2026-06-19

This report consolidates the local autoresearch run artifacts under
`runs/autoresearch/forage_loop` before cleanup. It covers the JAX MAPPO forage
experiments that replaced the earlier single-ant scale-up attempt.

## Executive Summary

The original single-ant, radius-1, feed-forward forage curriculum should not be
pushed to larger maps as-is. It learns small maps, then collapses as stage size
increases.

The useful 25x25 solution came from changing the problem from "one ant must
discover sparse long routes" to a denser-credit, multi-ant policy:

- `distance_bonus=0.02` was the first important unlock.
- Four ants plus distance shaping solved 25x25 when movement actions are sampled.
- Low-entropy sharpening helped greedy behavior, but did not fully turn the
  sampled solver into an argmax solver.
- The best 25x25 deployment policy is sampled movement with greedy writing,
  tuned by movement temperature.
- Larger 24-ant and 32-ant speed policies can deliver most food quickly on
  25x25, but they are less clean as a scientific claim because they change the
  embodied throughput substantially.
- 50x50 rare-source randomized maps remain unsolved. Explicit hub and nearest
  food vectors help, but random ant spawn and rare sources are still hard.

The loop produced 54 planned runs: 53 completed summaries and 1 incomplete
latest run.

## Baseline Failure

The notebook scale-up runs confirmed the user's read:

| Run | Evidence |
| --- | --- |
| Completed 4x4 to 50x50 notebook sweep | Return fell from `12.281` at 4x4 to `0.719` at 25x25 and `0.156` at 50x50. |
| Interrupted scaled-budget notebook run | Return fell through the 20s: `20x20=1.281`, `22x22=0.922`, `24x24=0.484`, and partial `25x25=0.797`. |

Verdict: the old setup spends compute preserving a behavior that stops scaling.
The first gate should be 25x25, not 50x50.

## Training Setup

The active loop is `autoresearch/loop.json`, called through:

```bash
PYTHONPATH=src ant-byte autoresearch loop-run --id <EXPERIMENT_ID> --wandb-mode online
```

The CLI path resolves a self-contained plan, then calls the shared JAX forage
curriculum helper. Each run writes:

- `experiment.md`
- `plan.json`
- `summary.json`
- optional `evaluation.json`
- checkpoints
- W&B local logs

The default 4x4 to 25x25 curriculum budget is:

| Stage group | Stages | Updates/stage | Rollout | Env steps/update | Env steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 4, 6, 8 | 600 | `8 envs * 128 steps` | 1,024 | 1,843,200 |
| Mid | 10, 12, 15 | 900 | `8 envs * 192 steps` | 1,536 | 4,147,200 |
| Large | 18, 20, 22, 24, 25 | 1,200 | `8 envs * 256 steps` | 2,048 | 12,288,000 |
| Total | 11 stages | 10,500 | mixed | mixed | 18,278,400 |

Later follow-ups used shorter budgets:

| Run type | Typical updates | Typical env steps | Purpose |
| --- | ---: | ---: | --- |
| 25x25 warm-start fine-tune | 500 to 1,600 | 1.02M to 3.28M | sharpen or select a 25x25 policy |
| Checkpoint evaluation grids | 0 | 0 | evaluate movement/write sampling policies |
| 24-ant speed policy | 500 to 1,000 | 1.02M to 2.05M | improve fast 25x25 delivery |
| 50x50 rare-map runs | 900 to 1,350 | 2.64M to 3.48M | test large randomized maps |

The current compute knob is `global_update_cap`: real env steps are roughly
`global_update_cap * num_envs * num_steps`, summed across stages.

## Best 25x25 Results

The strongest policies solve nearly all 23 food items on held-out shuffled
25x25 resets when movement is sampled.

| Run | Family | Best eval mode | Delivered | Fraction | Success | Updates |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `DISTANCE_CAP4` | distance plus four ants | sampled | 23.00 | 1.000 | 1.000 | 10,500 |
| `DISTANCE_CAP4_DISTILL` | warm-start distillation | sampled | 23.00 | 1.000 | 1.000 | 1,600 |
| `DISTANCE_CAP4_GREEDY_TUNE` | deterministic-policy fine-tune | sampled | 23.00 | 1.000 | 1.000 | 1,200 |
| `DISTANCE_CAP4_LONG_CREDIT_GENTLE_GREEDY` | mode alignment | sampled | 23.00 | 1.000 | 1.000 | 800 |
| `DISTANCE_CAP4_LONG_CREDIT_TUNE` | credit assignment | sampled | 23.00 | 1.000 | 1.000 | 900 |
| `DISTANCE_VISION2_CAP4` | radius-2 plus four ants | sampled | 23.00 | 1.000 | 1.000 | 10,500 |
| `DISTANCE_CAP4_SHARP_TEMP_FINE_POLICY` | deployment policy | sampled move temp 1.25, greedy write | 22.83 | 0.993 | 0.938 | 0 |
| `DISTANCE_CAP4_SHARP_TEMP_CONFIRM_GRID` | deployment policy | sampled move temp 1.40, greedy write | 22.80 | 0.991 | 0.885 | 0 |
| `DISTANCE_VISION2_CAP4_SHARP` | vision sharpening | sampled | 22.75 | 0.989 | 0.875 | 12,150 |
| `DISTANCE_CAP4_SHARP_T125_CONFIRM_POLICY` | deployment policy | sampled move temp 1.25, sampled write | 22.63 | 0.984 | 0.828 | 0 |

Interpretation:

- The learned route behavior exists.
- It is mostly represented as a stochastic movement policy, not a robust greedy
  argmax policy.
- Sampling movement while making writing greedy is the best deployment compromise.

## Best Fast 25x25 Results

The speed-oriented line changes the embodied throughput: 24 or 32 ants, shorter
episode horizons, checkpoint selection, and condensed food sources. These runs
are useful if the goal is high delivery rate in 430 to 550 steps, but they are
not directly comparable to the clean four-ant claim.

| Run | Final train return | Remaining food | Best delivered | Updates | Env steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `DISTANCE_CAP32_SPEED_SOURCES12_430` | 21.605 | 5.500 | 21.875 | 600 | 1,228,800 |
| `DISTANCE_CAP24_SPEED_SOURCES12_430` | 18.876 | 3.375 | 21.354 | 1,000 | 2,048,000 |
| `DISTANCE_CAP24_SPEED_HELDOUT_SELECT_430` | 18.801 | 3.500 | 21.875 | 500 | 1,024,000 |
| `DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430` | 18.256 | 4.125 | 21.984 | 700 | 1,433,600 |
| `DISTANCE_CAP24_SPEED_SOURCES16_430` | 17.981 | 3.750 | 21.948 | 750 | 1,536,000 |
| `DISTANCE_CAP24_SPEED_BEST_SELECT_SEED2_430` | 17.010 | 2.250 | 22.016 | 700 | 1,433,600 |

Interpretation:

- More ants are extremely effective for throughput.
- Checkpoint selection matters because later updates can drift.
- Strong final train return does not always equal best held-out checkpoint.

## 50x50 Rare-Map Results

The 50x50 experiments used larger randomized maps, 23 total food, rare food
sources, randomized hub/food, and in later runs random ant spawn. This remains
the hard frontier.

| Run | Intervention | Best eval | Delivered | Fraction | Success | Updates |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_HELDOUT_SELECT` | 8 ants, hub vector, nearest-food vector, held-out checkpoint selection | stage eval at update 460 | 18.313 | 0.796 | 0.250 | 500 |
| `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_RANDOM_SPAWN` | same vectors with random ant spawn | sampled move temp 0.90, greedy write | 15.203 | 0.661 | 0.078 | 900 |
| `DISTANCE_CAP8_BIGMAP_RARE_HUBVECTOR_RANDOM_SPAWN` | hub vector only with random ant spawn | sampled move temp 1.00, greedy write | 12.328 | 0.536 | 0.016 | 1,100 |
| `DISTANCE_CAP8_BIGMAP_RARE_RANDOM_SPAWN` | no explicit vectors, random spawn | sampled move temp 1.00, greedy write | 10.958 | 0.476 | 0.042 | 1,350 |
| `DISTANCE_CAP12_BIGMAP_RARE_VISION2_RANDOM_SPAWN` | 12 ants, radius-2 vision, random spawn | sampled move temp 1.00, greedy write | 9.500 | 0.413 | 0.021 | 900 |

Interpretation:

- Explicit nearest-food and hub vectors help on rare 50x50 maps.
- Random ant spawn is a big difficulty jump.
- Radius-2 vision plus 12 ants did not rescue the rare-map random-spawn setup.
- 50x50 is not solved; the best current result is about 18.3/23 delivered.

## Latest Incomplete Run

`DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_SPAWN_RADIUS_CURRICULUM` attempted a reset
distribution curriculum:

| Stage | Planned updates | Rollout | Result |
| --- | ---: | --- | --- |
| `50x50_spawn8` | 250 | `8 envs * 384 steps` | completed, train return `5.898` |
| `50x50_spawn16` | 250 | `8 envs * 384 steps` | completed, train return `7.209` |
| unrestricted `50x50` | 500 | `8 envs * 384 steps` | stopped at `130/500`, train return around `7.207` |

Planned budget was 1,000 updates or 3,072,000 env steps. It reached 630 updates
or about 1,935,360 env steps. It wrote checkpoints, including a local
`50x50_best.pkl`, but no final `summary.json`. GPU was idle after inspection,
so the run was no longer active.

## What Worked

1. Dense progress reward.
   `distance_bonus=0.02` turned sparse navigation into something MAPPO could
   actually learn.

2. Multi-ant exploration.
   Four ants were enough to make the 25x25 sampled policy solve the task.

3. Movement sampling with greedy writing.
   The policy's movement distribution carries useful route choices. Greedy
   writing avoids noisy byte-grid behavior while preserving the movement policy.

4. Checkpoint selection.
   Best-checkpoint selection by held-out eval often beat final checkpoint
   selection, especially in speed and 50x50 runs.

5. Explicit navigation vectors on 50x50.
   Nearest-food plus hub vectors were the best tested aid for rare-source 50x50
   random layouts.

## What Did Not Work

1. Continuing the original single-ant curriculum.
   It decays hard after the small maps.

2. Treating greedy argmax as solved.
   Many policies fully solve sampled eval while greedy eval remains weak.

3. Pure entropy reduction or direct greedy fine-tuning.
   These helped some metrics but did not reliably produce a fully greedy 25x25
   solver.

4. More vision alone.
   Radius-2 actor vision did not dominate the four-ant radius-1 recipe.

5. Rare 50x50 random ant spawn.
   The policy still struggles when ants, hub, and sparse food sources are all
   randomized on the large map.

## Recommended Claims

The cleanest supported claim is:

> A single local ant with sparse delivery reward does not scale, but a
> distance-shaped four-ant MAPPO policy can solve held-out shuffled 25x25 forage
> when movement actions are sampled; the remaining gap is mainly deployment-mode
> consolidation, not absence of route knowledge.

The strongest practical 25x25 policy is:

- `DISTANCE_CAP4_SHARP` checkpoint
- sampled movement at temperature roughly 1.25 to 1.40
- greedy write head

The strongest 50x50 claim is weaker:

> Explicit hub and nearest-food vectors plus held-out checkpoint selection
> improve rare-source 50x50 performance to about 18.3/23 delivered, but the
> randomized large-map task is not solved.

## Cleanup Note

The heavy local payloads from these experiments are checkpoints, W&B binary
logs, and media. The findings above are copied here so those generated payloads
can be deleted while retaining the decision trail.

Cleanup performed after consolidation:

- Removed generated `checkpoints/`, `wandb/`, and `media/` directories under
  `runs/autoresearch`.
- Kept lightweight `plan.json`, `summary.json`, `evaluation.json`, and
  `experiment.md` records.
- Reduced `runs/autoresearch` from about 1.1 GB to about 12 MB.
