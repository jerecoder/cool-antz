# Complete Experiment Chronology

Date: 2026-07-03

This document is the evidence ledger for the report and report website. It
connects the commit history, experiment configs, local run artifacts, report
figures, and current `report-writing-site` branch state into one reasoning
line.

It is intentionally more explicit than the paper narrative. The paper can be
shorter; the website can be more visual. This file is where the full contract
of each experiment family lives.

## Evidence Scope

- Git history inspected: 257 commits from `04b2937` through `a4ed53c`.
- Current branch: `report-writing-site`, tracking `origin/report-writing-site`.
- Current worktree status after index refresh: clean.
- Local commits beyond `origin/main`: 22 commits, from the 100x100 cleanup/bridge
  loop through the report checkpoint commit `a4ed53c`.
- Experiment configs inspected: every `experiments/*.json`.
- Notebook launch surfaces inspected: every notebook under `notebooks/**`.
- Local run evidence inspected: `runs/autoresearch/forage_loop/**`,
  `runs/notebooks/**`, `runs/training/**`, `runs/debug/**`,
  `runs/evaluation/**`, `runs/overnight_efficiency_sweep/**`, and
  `runs/bridge_100x100_sweep/**`.
- Report evidence inspected: `report/main.tex`, `report/data/*.csv`,
  `report/data/figure_sources.md`, and generated figures.

The main caution is that not every row below is a clean ablation. Many rows are
intervention bundles: checkpoint lineage, optimizer state, critic architecture,
source geometry, evaluation mode, ant count, and write semantics often move
together.

## Core Environment Contract

The task is a cooperative gridworld. A colony of ants must find food sources,
pick up food, return to the hub, and deliver it. The raw environment reward is
the delivered-food count unless explicit penalties or completion bonuses are
enabled.

### State Space

The environment state contains:

- Grid geometry: width, height, optional obstacles/maze walls, optional active
  window for curricula.
- Hub: one grid coordinate, fixed or randomly sampled.
- Food: an integer grid. `food_count` is total bites; `food_sources` is the
  number of depleted source cells that share those bites.
- Ants: per-ant position, facing direction, carrying flag, and an occupancy
  grid.
- External memory: a byte/value grid with values from `0` through
  `2^write_bits - 1`.
- Episode counters and diagnostics: delivered food, remaining food, step count,
  visited/viewed cells, writes, overwrites, byte occupancy, and curriculum
  stage fields for autocurriculum variants.

This is the first important semantic boundary: `food_count=23` with
`food_sources=6` means 23 deliverable bites concentrated across six depleting
sources, not 23 independent singleton food tiles.

### Action Space

For `N` ants and `B` write bits, the joint action is:

```text
MultiDiscrete([5, 2^B] * N)
```

Each ant chooses:

- movement: stay, up, right, down, or left;
- write value: an integer in `[0, 2^B - 1]`.

Writes are blocked on food and hub tiles. By default writes only apply on
`ACTION_STAY`; most JAX MAPPO forage configs set `write_while_moving=True`, so
movement and writing happen in the same step. Some runs use per-ant write
channels, where ant `i` can only change bit `i mod B`; later shared-write runs
remove that restriction.

### Actor Observation

The deployed actor is local. For each ant, it receives a facing-aware local
patch:

- local food amount;
- local ant occupancy;
- local byte bits;
- local hub indicator;
- local border/obstacle mask;
- optional one-hot agent identity;
- own carrying flag;
- own facing direction.

For actor radius `r`, patch size is `(2r + 1)^2`. Approximate actor feature
dimension is:

```text
patch_size * (4 + write_bits) + identity_features + 1 + 4
```

where `identity_features` is `0` for one ant, otherwise either `num_ants` or
the configured repeated identity type count.

### Critic Observation

Training uses CTDE: centralized training, decentralized execution. The critic
receives a centralized vector containing normalized ant positions, carrying
flags, facing, ant-count grid, food grid, byte grid, hub position, active grid
size, and auxiliary global features such as carrier fraction, byte occupancy,
mean hub distance, food centroid, and distance-curriculum stage fields.

Critic architecture is therefore a first-class causal boundary:

- `mlp`: flat dense critic over the full central observation.
- `structured_mlp`: split grid/entity/global features before dense fusion.
- `strided_cnn`: spatial CNN over grid planes plus entity/global features.
- `set_cnn`: spatial CNN plus set pooling over ants, useful for large ant
  counts.
- `resnet_cnn`: compact residual CNN critic.

Changing the critic changes PPO credit assignment, even when actor observations
and deployed execution remain local.

### Reward Channels

The raw JAX environment reward is:

- `+1` per delivered bite;
- optional per-ant step penalty;
- optional write penalty;
- optional completion bonus.

The trainer can add shaping:

- pickup bonus;
- normalized progress to food while empty and to hub while carrying;
- carrying-only hub progress;
- first-visit and first-view bonuses;
- border view/moat penalties;
- autocurriculum stage completion bonus;
- byte trail, byte following, carrying-write bonuses;
- write-bit penalties or entropy bonuses;
- no-byte-read and no-write ablations.

For reporting, always separate `env_return` and `delivery_events` from shaped
`episode_return`.

## Reasoning Line

The project story is not "add bytes, get communication." The cleaner reasoning
line is:

1. Define the task correctly: delivery-only reward with depleting multi-bite
   sources.
2. Make sparse long-horizon foraging learnable: distance shaping and enough
   ant coverage.
3. Notice deployment-mode reality: many policies work under sampled movement
   but collapse under greedy movement.
4. Treat byte memory as a hypothesis, not a conclusion: no-write and bit-count
   evidence does not yet prove causal communication.
5. For 50x50, separate source geometry from critic architecture: the
   `strided_cnn` critic is a major intervention.
6. For 60-ant 50x50, report the result as strong behavior engineering with
   saturated writes, not proof of compact emergent language.
7. For 250x250 and 100x100, keep raw deliveries, pickups, and
   pickup-to-delivery conversion above shaped return in the story.

## Chronology

### 2026-06-10: Base Environment And Contract Reset

Commit range: `04b2937` through `dd83222` created the repo, sprites, random
rollout tooling, and the first environment. Commit `11b581d` is the major
semantic boundary: colony delivery reward and depleting food sources.

Contract at this point:

| Field | Value / meaning |
| --- | --- |
| Environment | Gym gridworld, later mirrored by JAX |
| Action | per-ant movement plus write value |
| Food | integer grid; sources deplete |
| Reward target | delivery to hub, not pickup alone |
| Memory | writable byte grid, blocked on hub/food |

Reasoning consequence: all earlier pickup-heavy intuition is no longer
paper-safe. From here on, a result only matters if ants close the
source-to-hub loop.

### 2026-06-11 to 2026-06-12: JAX MAPPO, Communication Bits, Ant Count

Key commits:

- `451af73`: TorchRL MAPPO curriculum.
- `774d196`: JAX environment core.
- `ac72984`: JAX MAPPO trainer.
- `4a0b853`: communication curriculum notebook.
- `6c7f204`: transfer communication training from forage policy.
- `e65e440`: extend MAPPO curriculum to 25x25.
- `037dc52`: ant-count curriculum notebook.
- `4014816`: preserve JAX food count state.
- `0f2938c`: add ant occupancy grid to state.

Experiment contracts:

| Family | Contract | Result |
| --- | --- | --- |
| `communication_bits` | 25x25, 1 ant, 23 bites, 6 sources, 2 to 8 bits, MLP critic | Bit count alone did not unlock scale. |
| `ant_count_25x25_3_bits` | 25x25, 3 bits, ants swept 2,3,4,6,8 | Train return rose monotonically with ants. |

Observed notebook summaries:

| Run family | Setting | Return signal |
| --- | --- | ---: |
| communication bits, 15x15-ish | 2,3,5,8 bits | train `env_return` roughly `4.38`, `4.69`, `5.00`, `4.81` |
| communication bits from 25x25 anchor | 2,3,5,8 bits | train `env_return` roughly `0.44`, `0.38`, `0.44`, `0.31` |
| ant count from 25x25 3-bit checkpoint | 2,3,4,6,8 ants | train `env_return` `4.81`, `6.81`, `7.81`, `9.94`, `10.88` |

Reasoning consequence: coverage mattered more than the size of the byte
alphabet in the early evidence.

### 2026-06-13 to 2026-06-14: Embodiment And Direct Baselines

Key commits:

- `22e19de`: forward-facing ant vision.
- `4d7eea0`: body-relative movement.
- `5cf5faa`: direct-goal baseline experiment.
- `887fc70`: checkpoint evaluation helpers.

Reasoning consequence: the actor became more embodied. Facing direction,
body-relative local patches, and movement/action-mode evaluation matter when
interpreting greedy versus sampled deployment.

### 2026-06-15: Communication Autoresearch And Write Semantics

There are 50 commits on this date. They create the dense communication
autoresearch loop, probes, consolidation/polish sweeps, ranking, and two
important semantic fixes:

- `5978ddc`: separate movement and writing timesteps.
- `36d04ff`: shuffle evaluation layouts.

Representative conclusion from the preserved report: the best rendered
communication checkpoint was useful, but the evidence did not prove byte
communication causality.

Reasoning consequence: videos and nonzero bytes are not enough. Evaluation must
use held-out shuffled layouts and no-write/no-byte-read controls.

### 2026-06-16 to 2026-06-19: Autoresearch Forage Loop

Key commit themes:

- expose cookie reward and gamma controls;
- extend forage curriculum to 50x50;
- add W&B previews and diagnostics;
- add autocurriculum environments and memory reward probes;
- replace the earlier autoresearch with the forage improvement loop;
- run dense second-wave distance, ant-count, deployment-temperature, speed, and
  rare-source experiments.

The key 25x25 result is `DISTANCE_CAP4`:

| Contract | Value |
| --- | --- |
| Grid | staged 4x4 through 25x25 |
| Ants | 4 |
| Actor radius | 1 |
| Write bits | 1 |
| Critic | MLP |
| Reward | delivery env reward plus pickup bonus and `distance_bonus=0.02` |
| Deployment | sampled movement wins; greedy movement weak |
| Result | sampled held-out `23/23`, success `1.0`; deterministic `2.75/23` |

This was the first clean behavioral unlock. But it combines ant count and model
capacity: `DISTANCE_SHAPE` used one ant and hidden size 128, while
`DISTANCE_CAP4` used four ants and hidden size 192.

Representative autoresearch matrix:

| Run | Family | Ants | Bits | r | Critic | Hidden | Dist | Pickup | Best eval | Delivered | Fraction | Success |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `DISTANCE_SHAPE` | reward shaping | 1 | 1 | 1 | MLP | 128 | 0.02 | 0.25 | deterministic | 13.8 | 0.598 | 0 |
| `DISTANCE_CAP4` | reward+capacity | 4 | 1 | 1 | MLP | 192 | 0.02 | 0.25 | sampled | 23 | 1.0 | 1.0 |
| `DISTANCE_CAP4_SHARP` | policy sharpening | 4 | 1 | 1 | MLP | 192 | 0.02 | 0.25 | sampled | 20.5 | 0.891 | 0.5 |
| `DISTANCE_CAP4_LONG_CREDIT_TUNE` | credit assignment | 4 | 1 | 1 | MLP | 192 | 0.02 | 0.25 | sampled | 23 | 1.0 | 1.0 |
| `DISTANCE_CAP4_LONG_CREDIT_GENTLE_GREEDY` | mode alignment | 4 | 1 | 1 | MLP | 192 | 0.02 | 0.25 | sampled | 23 | 1.0 | 1.0 |
| `DISTANCE_CAP4_NO_WRITE` | action ablation | 4 | 1 | 1 | MLP | 192 | 0.02 | 0.25 | sampled | 22 | 0.957 | 0.5 |
| `DISTANCE_VISION2_CAP4` | radius 2 | 4 | 1 | 2 | MLP | 256 | 0.02 | 0.25 | sampled | 23 | 1.0 | 1.0 |
| `DISTANCE_CAP24_SPEED_SOURCES12_430` | speed | 24 | 1 | 1 | MLP | 192 | 0.02 | 0.25 | sampled t=0.90 | 21.4 | 0.928 | 0.417 |
| `DISTANCE_CAP32_SPEED_SOURCES12_430` | speed | 32 | 1 | 1 | MLP | 192 | 0.02 | 0.25 | sampled t=1.00 | 21.9 | 0.951 | 0.5 |
| `DISTANCE_CAP8_BIGMAP_RARE_RANDOM_SPAWN` | rare 50x50 | 8 | 2 | 1 | MLP | 192 | 0.02 | 0.25 | sampled t=1.00 | 11.0 | 0.476 | 0.042 |
| `DISTANCE_CAP8_BIGMAP_RARE_HUBVECTOR_RANDOM_SPAWN` | rare 50x50 + hub vector | 8 | 2 | 1 | MLP | 192 | 0.02 | 0.25 | sampled t=1.00 | 12.3 | 0.536 | 0.016 |
| `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_RANDOM_SPAWN` | rare 50x50 + hub/food vectors | 8 | 2 | 1 | MLP | 192 | 0.02 | 0.25 | sampled t=0.90 | 15.2 | 0.661 | 0.078 |
| `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_HELDOUT_SELECT` | held-out selection | 8 | 2 | 1 | MLP | 192 | 0.02 | 0.25 | sampled t=0.90 | 16.2 | 0.704 | 0.188 |

Reasoning consequence: the cleanest claim is that distance-shaped four-ant
MAPPO solves held-out 25x25 under sampled movement. Rare 50x50 remains
unsolved, though navigation vectors help discovery.

### 2026-06-23 to 2026-06-25: Notebook Consolidation And 50x50 Spatial Critic Branch

Key commits:

- `e5826fb`: simplify full-layout half-food notebook.
- `41e5eba`: checkpoint working research state.
- `4342d2e` through `424d965`: extract workflow, research, rollout, and
  checkpoint helpers.
- `a18a040`, `0037142`: organize notebook taxonomy.
- `b1c39dd`, `4201933`: continue full-layout from selected weights.
- `e6fa253`: add 64-env full-layout continuation.
- `67a6ade`: shared-writes full-layout notebook.
- `e59a0fb`, `26e5968`, `97ccf63`: shared-write bit-cost notebook and stricter
  penalty.

This is the key 50x50 causal boundary. The early efficient 50x50 line uses the
flat MLP critic. The proximity/full-layout line switches to `strided_cnn`.

Representative contracts:

| Config | Grid | Ants | Food | Sources | Bits | Actor r | Critic | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `exploration_to_forage_50x50.json` | 50x50 | 4 | 48 | 12 | 1 | 1 | MLP | efficient continuation |
| `exploration_to_forage_proximity_sources_50x50.json` | 50x50 outer / 30x30 inner | 4 | 250 | 2 | 4 | 2 | `strided_cnn` | scratch spatial critic branch |
| `exploration_to_forage_full_layout_8ants_half_food_50x50.json` | 50x50 | 8 | 125 | 2 | 4 | 2 | `strided_cnn` | selected continuation |
| `exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes.json` | 50x50 | 8 | 125 | 2 | 4 | 2 | `strided_cnn` | shared write values |
| `exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost.json` | 50x50 | 8 | 125 | 2 | 4 | 2 | `strided_cnn` | write cost |

Report figure data records:

| Result | Delivered / total | Success | Write nonzero |
| --- | ---: | ---: | ---: |
| efficient 50x50 MLP | `21.5/48` | 0.0 | 0.015 |
| proximity positive `strided_cnn` | `40.875/250` | 0.0 | 0.927 |
| 8 ants write cost `strided_cnn` | `69.375/125` | 0.125 | 0.943 |

Reasoning consequence: do not tell the story as "full layout, shared writes,
or more ants caused the 50x50 improvement" without naming the spatial critic
change. That critic changed the value-learning problem.

### 2026-06-28 to 2026-06-30: 250x250 Half-Scale Branch

This run family is mostly represented by local run artifacts and later configs.
It explores one rich source at 250x250 with many ants, long horizons, and
critic architecture changes.

Representative contracts:

| Run/config | Grid | Ants | Food | Sources | Bits | Actor r | Critic | Result signal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| one-food resnet actor warmstart | 250x250 | 500 | 5000 | 1 | 4 | 2 | `resnet_cnn` | essentially no useful delivery |
| one-food NPC teacher | 250x250 | 500 | 5000 | 1 | 4 | 2 | `strided_cnn` | weak delivery, high byte saturation |
| distance autocurriculum source teacher | 250x250 | 500 | 5000 | 1 | 4 | 2 | `set_cnn` | positive shaping with zero delivery in diagnosis |
| fixed8 reset-boundary 256 | 250x250 | 500 | 5000 | 1 | 4 | 2 | `set_cnn` | 654 final deliveries, 1003 best train deliveries |
| byte-decay follow-up | 250x250 | 500 | 5000 | 1 | 4 | 2 | `set_cnn` | 553 best deliveries, lower byte saturation |

Key metrics from local summaries:

| Run | Deliveries | Pickups | Episode return | Byte fraction | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| source-teacher diagnosis | 0 | 0 | 26.25 | 0.899 | shaping can rise while objective is zero |
| fixed8 reset-boundary final | 654 | 869 | 726.4 | 0.0022 | reset/distance contract creates real delivery locally |
| fixed8 reset-boundary best train | 1003 | 1279 | 1231.8 | 0.0010 | best checkpoint selection matters |
| byte decay | 553 | 741 | 578.0 | 0.0004 | lower byte occupancy, less delivery |

Reasoning consequence: 250x250 is a cautionary branch. Shaped return and byte
maps can look alive while raw delivery is zero. The reset-boundary intervention
is real progress, but it is not proof of general 250x250 foraging or byte
communication.

### 2026-07-01: 8-Bit Shared-Write Continuation

Commit `a6ba235` adds the 8-bit shared-write continuation. It keeps the
50x50, 8-ant, half-food, `strided_cnn` branch but changes write bits from 4 to
8 and rescales write penalty from `0.01` to `0.0002`.

Reasoning consequence: this is not an isolated "more bits" test. It changes the
write alphabet and the cost scale at the same time.

### 2026-07-02 to 2026-07-03: 100x100 Bridge, 60-Ant 50x50, Report Checkpoint

Local commits beyond `origin/main`:

| Commit | Date | Message | Chronology role |
| --- | --- | --- | --- |
| `80b902d` | 2026-07-02 | add memory-safe cleanup eval loop | start large-scale cleanup/bridge automation |
| `345b88a` | 2026-07-02 | add 100x100 bridge sweep launcher | 100x100 bridge candidate generation |
| `483a22c` | 2026-07-02 | add 100x100 continuation sweep | continuation from best bridge checkpoints |
| `f0d80f1` | 2026-07-02 | add 100x100 temperature eval grid | deployment-temperature selection |
| `9665577` to `aea6e52` | 2026-07-02 | progress video uploader fixes | video evidence pipeline |
| `c4851c7` | 2026-07-02 | allow continuation source override | bridge lineage control |
| `c1a0741`, `10b09d0`, `f776de5` | 2026-07-03 | 250x250 continuation configs | truncation and d8/d16 follow-ups |
| `a203c2d` to `5b27fd2` | 2026-07-03 | best-policy bigmap render and palette tooling | website/report video evidence |
| `a4ed53c` | 2026-07-03 | checkpoint overnight research state | report draft, figures, 60-ant notebook/configs, logs |

100x100 bridge summary:

| Run family | Best local result | Interpretation |
| --- | --- | --- |
| easy 125-food 2-source bridge | up to `124.2/125`, success `0.75` in sweep summaries | 60/120-ant bridge can solve easier 100x100 settings |
| mid 250-food 4-source bridge | up to `249.8/250`, success `0.75` in continuation summaries | continuation and temperature selection strongly help |
| hard 375-food 6-source bridge | up to `373/375`, success `0.75` in continuation summaries | impressive but still a continuation/selection branch |
| temperature grids | best mid temp around `0.475`, hard around `0.5` | behavior remains deployment-temperature sensitive |

60-ant 50x50 branch contract from the checkpoint commit:

| Config | Grid | Ants | Food | Sources | Bits | Actor r | Critic | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `...60ants...from_best.json` | 50x50 | 60 | 125 | 2 | 8 | 2 | `strided_cnn` | 8 repeating identity types, centered 24x24 hub window |
| `...stabilize_from_60best.json` | 50x50 | 60 | 125 | 2 | 8 | 2 | `strided_cnn` | fresh optimizer, low LR, entropy off, lower temperature |
| `...speed_shaping_from_60best.json` | 50x50 | 60 | 125 | 2 | 8 | 2 | `strided_cnn` | adds distance/time shaping for faster delivery |
| `...time_penalty_from_stabilized_best.json` | 50x50 | 60 | 125 | 2 | 8 | 2 | `strided_cnn` | ablates distance shaping after speed trial |

Report figure data records the current 60-ant headline:

| Result | Delivered / total | Success | Write nonzero |
| --- | ---: | ---: | ---: |
| 60 ants confirmed | `123.90625/125` | 0.90625 | 0.998 |

Reasoning consequence: this is a strong behavior result. It is also deeply
confounded: 60 ants, 8 bits, identity features, selected continuation, spatial
critic, centered hub window, temperature selection, and saturated writes all
coexist. Present it as frontier engineering progress, not as clean causal proof
of byte communication.

## Experiment Config Ledger

This table covers every `experiments/*.json` config currently tracked.

| Config | Grid | Ants | Food | Sources | Bits | Actor r | Critic | Max steps | Envs x rollout | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `autocurriculum.json` | 50x50 | 1 | 12 | 2 | 1 | 1 | MLP | 10000 | 16 x 80 | scratch |
| `communication_bits.json` | 25x25 | 1 | 23 | 6 | 2 | 1 | MLP | 2500 | 16 x 80 | 25x25 forage checkpoint |
| `direct_goal_baseline.json` | 50x50 | 10 | 48 | 25 | 5 | 1 | MLP | 10000 | 16 x 80 | scratch |
| `exploration_curriculum.json` | 50x50 | 1 | 48 | 12 | 1 | 1 | MLP | 6250 | 16 x 80 | scratch |
| `exploration_to_forage_50x50.json` | 50x50 | 4 | 48 | 12 | 1 | 1 | MLP | 6250 | 16 x 256 | efficient 50x50 checkpoint |
| `exploration_to_forage_padded_sources_50x50.json` | 50x50 | 4 | 18 | 2 | 1 | 1 | MLP | 1000 | 16 x 256 | 20x20 efficient checkpoint |
| `exploration_to_forage_scratch_smooth_sources_50x50.json` | 80x80 outer / 50x50 inner | 4 | 250 | 2 | 4 | 2 | MLP | 1000 | 16 x 256 | scratch |
| `exploration_to_forage_proximity_sources_50x50.json` | 50x50 outer / 30x30 inner | 4 | 250 | 2 | 4 | 2 | `strided_cnn` | 1000 | 16 x 256 | scratch |
| `exploration_to_forage_full_layout_8ants_half_food_50x50.json` | 50x50 | 8 | 125 | 2 | 4 | 2 | `strided_cnn` | 2000 | 16 x 256 | long3 best |
| `exploration_to_forage_full_layout_8ants_half_food_50x50_64env.json` | 50x50 | 8 | 125 | 2 | 4 | 2 | `strided_cnn` | 2000 | 64 x 256 | long4 8k |
| `exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes.json` | 50x50 | 8 | 125 | 2 | 4 | 2 | `strided_cnn` | 2000 | 64 x 256 | 64env best |
| `exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost.json` | 50x50 | 8 | 125 | 2 | 4 | 2 | `strided_cnn` | 2000 | 64 x 256 | shared-write best |
| `exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_8bits.json` | 50x50 | 8 | 125 | 2 | 8 | 2 | `strided_cnn` | 2000 | 64 x 256 | 4-bit write-cost best |
| `exploration_to_forage_full_layout_16ants_half_food_8types_50x50_from_shared_writes.json` | 50x50 | 16 | 125 | 2 | 8 | 2 | `strided_cnn` | 2000 | 64 x 256 | 8-ant shared-write best |
| `exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_from_best.json` | 50x50 | 60 | 125 | 2 | 8 | 2 | `strided_cnn` | 2000 | 16 x 256 | 8-bit best |
| `exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_stabilize_from_60best.json` | 50x50 | 60 | 125 | 2 | 8 | 2 | `strided_cnn` | 2000 | 16 x 256 | 60-ant best |
| `exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_speed_shaping_from_60best.json` | 50x50 | 60 | 125 | 2 | 8 | 2 | `strided_cnn` | 2000 | 16 x 256 | 60-ant best |
| `exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_time_penalty_from_stabilized_best.json` | 50x50 | 60 | 125 | 2 | 8 | 2 | `strided_cnn` | 2000 | 16 x 256 | stabilized 60-ant best |
| `forage_curriculum.json` | 50x50 | 1 | 48 | 12 | 1 | 1 | MLP | 10000 | 16 x 256 | scratch |
| `maze_exploration_curriculum.json` | 50x50 | 1 | 48 | 12 | 2 | 1 | MLP | 6250 | 16 x 80 | scratch |
| `half_scale_distance_autocurriculum_250x250_healthy_reset.json` | 250x250 | 500 | 5000 | 1 | 4 | 2 | `set_cnn` | 60000 | 1 x 64 | reset-boundary latest |
| `half_scale_distance_250x250_reasonable_truncation_from_frontier.json` | 250x250 | 500 | 5000 | 1 | 4 | 2 | `set_cnn` | 4096 | 1 x 256 | no-decay frontier |
| `half_scale_distance_250x250_d16_trunc4096_from_685best.json` | 250x250 | 500 | 5000 | 1 | 4 | 2 | `set_cnn` | 4096 | 1 x 256 | 685-delivery best |
| `half_scale_distance_250x250_soft_d8d16_trunc4096_from_685best.json` | 250x250 | 500 | 5000 | 1 | 4 | 2 | `set_cnn` | 4096 | 1 x 256 | 685-delivery best |
| `smoke.json` | 4x4 | 1 | 1 | 1 | default | default | MLP | 8 | 1 x 4 | scratch |

## Commit Coverage

All 257 commits were scanned. Date density:

| Date | Commits | Main role |
| --- | ---: | --- |
| 2026-06-10 | 18 | base env, sprites, random rollout, delivery/source contract |
| 2026-06-11 | 18 | JAX env/MAPPO, communication notebook, research workflow |
| 2026-06-12 | 7 | transfer fixes, 25x25 and ant-count notebooks |
| 2026-06-13 | 2 | forward-facing vision, body-relative movement |
| 2026-06-14 | 6 | direct baseline, rollout state, evaluation helpers |
| 2026-06-15 | 50 | communication sweeps, write semantics, shuffled eval |
| 2026-06-16 | 3 | cookie reward/gamma controls |
| 2026-06-17 | 13 | 50x50 curriculum, W&B, distance shaping, memory diagnostics |
| 2026-06-18 | 27 | autocurriculum, memory shaping, unified forage loop |
| 2026-06-19 | 53 | 25x25 distance/capacity/speed/rare-source matrix |
| 2026-06-23 | 1 | full-layout notebook simplification |
| 2026-06-24 | 30 | workflow refactors and full-layout continuations |
| 2026-06-25 | 6 | shared-write and write-cost continuations |
| 2026-07-01 | 1 | 8-bit shared-write continuation |
| 2026-07-02 | 13 | cleanup loop, 100x100 bridge sweeps, progress videos |
| 2026-07-03 | 9 | 250x250 continuations, bigmap render tooling, report checkpoint |

## Paper-Safe Claims

- Strong: the task contract is delivery-only foraging with depleting sources,
  and `food_sources` changes task geometry.
- Strong: distance shaping plus four ants unlocked 25x25 under sampled movement.
- Strong: sampled movement versus greedy movement is a real deployment axis.
- Strong: the 50x50 spatial-critic branch must be separated from earlier MLP
  critic experiments.
- Strong: 250x250 shaping return can be misleading without raw delivery metrics.
- Tentative: 60-ant 50x50 behavior is very strong but confounded.
- Not proven: byte communication is causally responsible for the best results.

## Website Storyline

The report website should show the same chronology visually:

1. Task contract: delivery, sources, bytes, actor/critic split.
2. Small maps and first failure: one ant does not scale.
3. 25x25 unlock: distance shaping plus ant coverage, with sampled-vs-greedy
   comparison.
4. Communication caution: bit sweeps and no-write evidence.
5. 50x50 transition: the critic-architecture boundary.
6. 60-ant frontier: strong behavior and saturated-write caveat.
7. 250x250/100x100 diagnostics: real deliveries versus shaped activity.
8. Transparency panel: what is proven, what is confounded, and which ablations
   are still needed.
