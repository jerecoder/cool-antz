# Experiment History Analysis

Date: 2026-07-01

This is a paper-prep timeline built from the current checkout plus local run
artifacts. Main evidence sources:

- `autoresearch/REPORT.md`
- `autoresearch/loop.json`
- `runs/autoresearch/forage_loop/*/{plan.json,summary.json,evaluation.json}`
- `runs/notebooks/**/{summary.json,metrics.jsonl,training_result.json}`
- `runs/notebooks/**/wandb/*/files/wandb-summary.json`
- `runs/evaluation/*/summary.json`
- `experiments/*.json`

Important caution: not every run uses the same denominator. Some results are
23-food 25x25 gates, some are 48-food or 125/250-food 50x50 variants, and the
speed line changes ant count and episode horizon. The most comparable metrics
are held-out delivered fraction, delivered food per ant-step, and success rate,
not raw episode return alone.

## Executive Summary

The largest hidden detail is the task-contract change, not a hyperparameter:
the environment moved to delivery-only reward with depleting multi-bite food
sources. Before that change, pickup was rewarded, delivery was worth much more,
there was a default per-step cost, and `food_count` behaved like many scattered
unit foods. After the change, `food_count` means total bites, `food_sources`
controls how concentrated those bites are, pickup has no raw reward, and each
delivered bite gives `+1`. That makes food-source density a first-order
experimental variable.

Within that corrected task, the largest positive change was not more
communication bits. It was making the foraging problem learnable: concentrated
multi-bite sources, dense distance progress reward, and enough ant coverage.
The original single-ant setup learned small maps and then collapsed. Adding
`distance_bonus=0.02` made the 25x25 task partially learnable, and adding four
ants plus a larger hidden state made the sampled 25x25 policy solve the
held-out shuffled gate.

The biggest later 50x50 detail is the critic architecture change. The early
autoresearch and efficient-50x50 lines used the default flat MLP critic. The
proximity/full-layout 50x50 line switched to `critic_architecture=strided_cnn`,
a centralized spatial value function over the 50x50 grid plus entity features.
That affects PPO credit assignment while leaving the deployed actor's local
observation contract mostly unchanged. Any 50x50 gain after that point is
confounded with the critic upgrade.

The second major lesson is that the learned behavior often lives in the action
distribution, not in greedy argmax deployment. Many policies solve or nearly
solve 25x25 with sampled movement but fail or weaken under greedy movement.
For the paper, sampled-movement/greedy-write is a legitimate deployment mode,
but the greedy gap should be reported honestly.

The third lesson is negative but important: byte memory is not yet causally
established. Communication-bit sweeps gave only modest train-return gains, and
the `DISTANCE_CAP4_NO_WRITE` ablation still delivered `22/23` sampled on the
25x25 gate. The current evidence supports "local memory/action channel exists"
and "write dynamics matter for some scaling runs", but not "emergent byte
communication caused the main 25x25 success."

The 50x50 story is not solved. Inside the `strided_cnn` critic branch,
full-layout 50x50 half-food runs improved a lot, especially after shared-write
and write-cost continuations, but remained below full completion. Rare-source
randomized 50x50 with random ant spawn stayed hard:
direct rare-source random-spawn eval reached about `10.96/23`, hub-vector
reached `12.33/23`, nearest-food plus hub vector reached `15.20/23` on the
final confirmation gate, and held-out selection pushed a selected checkpoint to
about `16.20/23` on a separate gate. That supports "vectors help discovery",
not "the pure local communication problem is solved."

The late 250x250 half-scale distance-autocurriculum line exposed a trap:
positive shaping return can coexist with zero raw deliveries. That line also
changed critic families (`resnet_cnn` / `strided_cnn` / `set_cnn`). A
reset-boundary fixed-distance intervention in the later `set_cnn` family fixed
delivery locally enough to produce hundreds of deliveries in windows, but the
broad 250x250 task and communication causality remain open.

## Subtle Differences That Change The Interpretation

Several experiment names read like single-variable ablations, but the actual
configs often change checkpoint lineage, optimizer settings, stage schedule,
hidden size, entropy, clipping, evaluation protocol, or source layout at the
same time. For paper figures, treat these as intervention bundles unless a
matched rerun exists.

Important examples:

| Comparison | Hidden difference | Paper-safe interpretation |
| --- | --- | --- |
| Pre-contract-reset env -> current forage env | Raw reward changed from pickup `+1`, delivery `+10`, default step penalty to delivery-only `+1` with no pickup reward; food changed from many unit positions to depleting multi-bite sources controlled by `food_sources`. | Do not compare early/pre-reset behavior directly to current JAX MAPPO forage results. This is a task definition change, not a training trick. |
| Fixed `food_count`, varied `food_sources` | `23` food over `6`, `12`, `16`, or `2` sources changes repeated source exploitation, reacquisition, and route diversity. | `food_sources` is a major task difficulty/control variable, not metadata. |
| Default MLP critic -> `strided_cnn` critic | The value function changes from a flat dense critic over flattened central observations to a spatial CNN critic over grid planes plus entity features. | This is a major training/credit-assignment intervention. Do not attribute 50x50 proximity/full-layout gains only to source layout, write semantics, or ant count. |
| `DISTANCE_SHAPE` -> `DISTANCE_CAP4` | Changed ant count `1 -> 4` and hidden size `128 -> 192`. | The big jump is best described as multi-ant coverage plus more model capacity under distance shaping, not a pure ant-count-only effect. |
| `DISTANCE_CAP4` -> `DISTANCE_CAP4_SHARP` | Lowered entropy, added PPO clipping, and increased high-stage update caps. | Greedy improvement cannot be attributed to entropy reduction alone. |
| `DISTANCE_CAP4_SHARP` -> `DISTANCE_CAP4_LONG_CREDIT_TUNE` | Warm-started from `SHARP`, collapsed to a 25x25-only continuation, changed `gamma`, `gae_lambda`, and learning rate. | This is late-stage consolidation, not a fresh curriculum comparison. |
| `DISTANCE_CAP4_LONG_CREDIT_TUNE` -> `DISTANCE_CAP4_LONG_CREDIT_GENTLE_GREEDY` | Added deterministic rollouts but also changed entropy, LR, clip, update epochs, and budget. | The deterministic gain is a mode-alignment bundle, not isolated proof that deterministic rollouts caused it. |
| `DISTANCE_CAP4_SHARP` -> `DISTANCE_CAP4_NO_WRITE` | Ablated writing while also changing continuation schedule, LR, clip, and epochs. | The sampled `22/23` result is strong evidence that writes are not necessary for this 25x25 policy, but it is not a perfectly controlled no-write retrain. |
| `DISTANCE_CAP4` -> `DISTANCE_VISION2_CAP4` | Increased vision radius and hidden size, with small entropy changes. | Radius-2 vision did not obviously help; do not call it a clean vision-only ablation. |
| `DISTANCE_CAP24_SPEED_*` | Changed ants, food sources, horizon, step penalty, LR/entropy, stage schedule, and checkpoint lineage. | These are throughput-engineering runs, not clean communication or ant-count ablations. |
| Rare-source `HUBVECTOR` / `NAVVECTOR` | Added vectors while also changing stage schedule, byte shaping, LR/entropy, penalties, and checkpoint selection. | Vectors are suggestive discovery aids, but the exact effect size needs matched reruns. |
| Full-layout shared-write/write-cost line | Each step is a continuation from selected `strided_cnn` checkpoints; `64env`, shared writes, write cost, bit count, and source lineage are entangled. | The line is promising engineering progress, not clean causal evidence for communication. |
| 8-bit shared-write continuation | Increased write bits `4 -> 8` while lowering write penalty `0.01 -> 0.0002`, a 50x cost rescale. | The incomplete 8-bit run cannot answer whether more bits help. |
| 250x250 `resnet_cnn`/`strided_cnn` -> `set_cnn` lines | Early one-food 250x250 runs used `resnet_cnn` or `strided_cnn` and ended at zero delivery; later distance/reset-boundary runs use `set_cnn`. | Treat the 250x250 line as a critic-architecture search plus reset/reward search, not just longer training. |
| 250x250 autocurriculum -> fixed8 reset-boundary | Changed the reset/distance contract inside the later `set_cnn` family rather than simply training longer. | The reset-boundary result shows the source-boundary contract mattered; it does not prove the broad autocurriculum is solved. |

Also separate metric identities carefully:

- `final` checkpoint metrics often understate the best behavior because several
  runs peak and then collapse.
- `best stage` and `held-out selected` metrics include checkpoint selection and
  should not be compared as if they were final-training metrics.
- W&B summaries, local `metrics.jsonl`, and evaluation summaries can refer to
  different checkpoints or gates.
- `sampled movement`, `greedy movement`, and `sampled write` are different
  deployment protocols, not small reporting details.
- The low-level env default writes only on `stay`, while the reported JAX MAPPO
  forage configs generally pass `--write-while-moving`. Write metrics and write
  costs must be interpreted under the actual timing mode used by the run.

Planned controls that appear in `autoresearch/loop.json` but do not have local
run directories in this checkout include `CAPACITY4`, `VISION2`, `NEAR_COOKIE`,
`DENSE8`, `GAMMA999`, `LADDER_FINE`, `VISION2_CAP4`, `BYTE_TRAIL`,
`AUTO_STAGE`, `DISTANCE_CAP4_SPEED_4A_550`, and
`DISTANCE_CAP8_SPEED_RAMP_8A`. Do not cite them as completed experiments unless
new evidence is recovered elsewhere.

## Timeline

### 0. Task contract reset: delivery-only reward and depleting sources

Representative artifacts:

- commit `11b581d` on 2026-06-10 (`fix: reward colony deliveries and deplete
  food sources`)
- `README.md` reward/action-space sections
- `src/ant_byte_env/env.py`
- `src/ant_byte_env/jax_env.py`
- `src/ant_byte_env/curricula/stages.py`

This is the major before/after boundary I was underweighting. The environment
contract changed from a pickup-heavy task to a delivery-only foraging task:

| Before reset | After reset / current contract |
| --- | --- |
| Pickup gave raw reward `+1`. | Pickup gives no raw reward. |
| Delivery gave raw reward `+10`. | Each delivered bite gives raw reward `+1`. |
| A default per-ant step cost was charged. | Default step penalty is `0`; later speed runs add explicit small penalties. |
| Food was effectively many unit food positions. | `food_count` is total bites split across `food_sources`; sources deplete. |

This changed the learning problem. The agent is no longer rewarded just for
finding food; it must complete pickup-to-hub loops. At the same time,
concentrated multi-bite sources make repeated successful trips possible once a
source is discovered.

The standard forage curriculum then made source concentration systematic:

| Stage | Food bites | Food sources | Meaning |
| --- | ---: | ---: | --- |
| 8x8 | 6 | 2 | roughly 3 bites/source |
| 12x12 | 10 | 3 | roughly 3.3 bites/source |
| 25x25 | 23 | 6 | roughly 3.8 bites/source |
| 50x50 | 48 | 12 | 4 bites/source |

Paper-safe conclusion: the 25x25 `23/23` result means clearing 23 delivered
bites from 6 depleting sources, not visiting 23 independent singleton foods.
Changing `food_sources` at fixed `food_count` is therefore a major task-geometry
intervention.

JAX-specific follow-up: the JAX env was introduced after the reward/source reset
on 2026-06-11, and commit `4014816` on 2026-06-12 fixed it to preserve the
actual summed food count as the termination target. That matters because
termination and delivered fraction depend on total bites, not just configured
nominal `food_count`.

### 1. Early JAX curriculum and communication notebooks

Representative artifacts:

- `results/curated/index.json`
- `runs/notebooks/communication_bits/*/summary.json`
- `runs/notebooks/communication_bits_25x25/*/summary.json`
- `runs/notebooks/ant_count_25x25_3_bits/*/summary.json`

Findings:

| Experiment | Result | Interpretation |
| --- | ---: | --- |
| 15x15 JAX curated checkpoint | curated as a representative success | The JAX MAPPO pipeline could learn small/mid maps. |
| Communication bits, 15x15-ish source | train return rose only modestly: 2 bits `6.40`, 3 bits `6.88`, 5 bits `7.36`, 8 bits `7.00` | More bits alone did not create a strong new capability. |
| Communication bits from 25x25 anchor | train returns stayed low: roughly `0.60` to `0.77` | Increasing byte alphabet from the 25x25 checkpoint did not solve scale. |
| Ant count from 25x25 3-bit checkpoint | return increased monotonically from 2 ants `7.79` to 8 ants `18.23` | Ant coverage/throughput was a real lever. |

Paper-safe conclusion: early evidence favored multi-ant coverage over raw byte
capacity, but all of this sits under the delivery-only/depleting-source task
contract.

### 2. Original single-ant scale-up failure

Representative artifacts:

- `autoresearch/REPORT.md`
- `experiments/forage_curriculum.json`
- `runs/notebooks/exploration_to_forage_50x50/summary.json`

The old single-ant, radius-1, feed-forward curriculum learned small maps but
failed as size grew. The consolidated autoresearch report records return
falling from `12.281` at 4x4 to `0.719` at 25x25 and `0.156` at 50x50. The
local `exploration_to_forage_50x50` summary is consistent with this: final
`delivery_events=4`, `episode_return=0.288`, and many carried/visited cells but
little delivery.

Cause: sparse long-horizon discovery/return was too hard for one local ant with
weak credit assignment. More precisely, under the corrected delivery-only
contract, one ant could often find/carry/visit but could not reliably convert
that into repeated source-to-hub delivery loops as maps grew.

### 3. First real unlock: distance shaping

Representative artifacts:

- `runs/autoresearch/forage_loop/DISTANCE_SHAPE/{plan.json,evaluation.json}`
- `autoresearch/REPORT.md`

`DISTANCE_SHAPE` introduced `distance_bonus=0.02`, long-horizon gamma, moving
writes, concentrated multi-bite food sources, and the 4x4 to 25x25 ladder. It
reached:

- deterministic held-out 25x25: `13.75/23`
- sampled held-out 25x25: `7.0/23`

This did not solve the task, but it was the first major jump from the weak
single-ant baseline. The likely causal mechanism was not just "more reward" in
the abstract: distance shaping supplied dense route feedback on top of a
delivery-only task where concentrated sources made repeated delivery loops
worth learning.

### 4. Biggest positive jump: four ants plus distance shaping

Representative artifacts:

- `runs/autoresearch/forage_loop/DISTANCE_CAP4/{plan.json,summary.json,evaluation.json}`

`DISTANCE_CAP4` kept `distance_bonus=0.02` and added four ants, hidden size 192,
moving writes, concentrated food, and the same staged 25x25 ladder. Relative to
`DISTANCE_SHAPE`, this was not only an ant-count change: hidden size also moved
from 128 to 192.

Result:

- deterministic: `2.75/23`, success `0.0`
- sampled: `23.0/23`, success `1.0`

This is the largest practical 25x25 jump in the repo, but the exact causal
label should be "multi-ant/capacity under distance shaping" rather than
"ant-count-only ablation." It also reveals the deployment gap: route knowledge
exists in the stochastic movement distribution, while greedy argmax can
collapse.

Paper-safe conclusion: the four-ant, hidden-192 distance-shaped MAPPO variant
solves held-out randomized 25x25 when movement actions are sampled.

### 5. Policy sharpening and long-credit follow-ups

Representative artifacts:

- `DISTANCE_CAP4_SHARP`
- `DISTANCE_CAP4_LONG_CREDIT_TUNE`
- `DISTANCE_CAP4_LONG_CREDIT_GENTLE_GREEDY`
- `DISTANCE_CAP4_GREEDY_TUNE`
- `DISTANCE_CAP4_NO_WRITE`
- `DISTANCE_VISION2_CAP4`

Key results:

| Run | Change bundle | Deterministic | Sampled | Takeaway |
| --- | --- | ---: | ---: | --- |
| `DISTANCE_CAP4_SHARP` | lower entropy, PPO clip, longer high-stage budget | `13.0/23` | `20.5/23` | Greedy improved, sampled weakened. |
| `DISTANCE_CAP4_LONG_CREDIT_TUNE` | warm-started 25x25-only long-credit continuation | `11.0/23` | `23.0/23` | Restored sampled solve, did not solve greedy. |
| `DISTANCE_CAP4_LONG_CREDIT_GENTLE_GREEDY` | deterministic rollout fraction plus lower LR/entropy/clip changes | `14.75/23` | `23.0/23` | Best greedy-ish improvement while preserving sampled solve. |
| `DISTANCE_CAP4_GREEDY_TUNE` | deterministic policy fine-tune | `7.75/23` | `23.0/23` | Did not convert to robust greedy deployment. |
| `DISTANCE_CAP4_NO_WRITE` | no-write continuation plus optimizer/schedule changes | `0.0/23` | `22.0/23` | Writes are not required for sampled 25x25 movement, but this is not a perfect one-variable retrain. |
| `DISTANCE_VISION2_CAP4` | radius-2 actor vision plus larger hidden size | `0.0/23` | `23.0/23` | More vision was not a clear improvement over four ants. |

What worked: late-stage consolidation and gentle mode alignment preserved the
sampled solution and improved deterministic behavior somewhat.

What did not work: the SHARP entropy/PPO-tightening bundle or greedy
fine-tuning did not make a fully robust greedy policy.

### 6. Deployment policy: sampled movement, greedy writing

Representative artifacts:

- `DISTANCE_CAP4_SHARP_TEMP_POLICY`
- `DISTANCE_CAP4_SHARP_TEMP_FINE_POLICY`
- `DISTANCE_CAP4_SHARP_TEMP_CONFIRM_GRID`
- `DISTANCE_CAP4_SHARP_T125_CONFIRM_POLICY`

Best held-out deployment results:

| Run | Protocol | Delivered | Success |
| --- | --- | ---: | ---: |
| `DISTANCE_CAP4_SHARP_TEMP_FINE_POLICY` | sampled movement temp `1.25`, greedy write | `22.83/23` | `0.938` |
| `DISTANCE_CAP4_SHARP_TEMP_CONFIRM_GRID` | sampled movement temp `1.40`, greedy write | `22.80/23` | `0.885` |
| `DISTANCE_CAP4_SHARP_T125_CONFIRM_POLICY` | sampled movement temp `1.25`, sampled write | `22.63/23` | `0.828` |

This is the strongest practical 25x25 deployment story. The policy does not
need sampled writing, but it benefits from sampled movement.

### 7. Speed/throughput line on 25x25

Representative artifacts:

- `DISTANCE_CAP24_SPEED_*`
- `DISTANCE_CAP32_SPEED_SOURCES12_430`

This line changed the embodied system: more ants, shorter horizons, step
penalties, often more food sources, and different optimizer/stage settings.
Because `food_sources` changes source density at fixed total bites, these runs
also change task geometry. They are useful engineering evidence, but less clean
scientifically because colony throughput, source concentration, and the task
budget all change.

Representative results:

| Run | Ants / sources / horizon | Delivered | Success |
| --- | --- | ---: | ---: |
| `DISTANCE_CAP24_SPEED_SOURCES12_POLISH_430` | 24 ants, 12 sources, 430 steps | `21.98/23` | `0.50` |
| `DISTANCE_CAP24_SPEED_HELDOUT_SELECT_430` | held-out checkpoint selection | `21.80/23` | `0.586` |
| `DISTANCE_CAP32_SPEED_SOURCES12_430` | 32 ants, 12 sources, 430 steps | `21.88/23` | `0.50` |

What worked: more ants massively improved throughput.

What did not work: this did not produce a cleaner communication claim, because
more bodies can solve much of the task by parallel search and transport.

### 8. Rare-source randomized 50x50

Representative artifacts:

- `DISTANCE_CAP8_BIGMAP_RARE_RANDOM_SPAWN`
- `DISTANCE_CAP12_BIGMAP_RARE_VISION2_RANDOM_SPAWN`
- `DISTANCE_CAP8_BIGMAP_RARE_HUBVECTOR_RANDOM_SPAWN`
- `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_RANDOM_SPAWN`
- `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_HELDOUT_SELECT`

The goal shifted toward a harder and more paper-relevant generalization test:
23 total food, rare sources, larger maps, random food, random colony, and random
ant spawn.

Results:

| Run | Change bundle | Final held-out result |
| --- | --- | ---: |
| `DISTANCE_CAP8_BIGMAP_RARE_RANDOM_SPAWN` | 8 ants, rare 25->50 curriculum, random ant spawn, byte-trail shaping | `10.96/23`, success `0.042` |
| `DISTANCE_CAP12_BIGMAP_RARE_VISION2_RANDOM_SPAWN` | 12 ants, radius-2 vision, stronger shaping, different 40/50-only schedule | `9.50/23`, success `0.021` |
| `DISTANCE_CAP8_BIGMAP_RARE_HUBVECTOR_RANDOM_SPAWN` | hub vector plus schedule/reward/optimizer changes | `12.33/23`, success `0.016` |
| `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_RANDOM_SPAWN` | hub + nearest-food vectors plus reward/optimizer changes | `15.20/23`, success `0.078` |
| `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_HELDOUT_SELECT` | held-out checkpoint selection plus final continuation tweaks | `16.20/23`, success `0.188` on final confirmation; training-time selected windows were higher |

What worked: explicit navigation/food vectors are the most plausible positive
difference, especially the nearest-food vector. Held-out checkpoint selection
also helped.

The hub-vector result is mixed: mean delivered food improved from `10.96` to
`12.33`, but success rate fell from `0.042` to `0.016`. The nearest-food-vector
line is the clearer positive signal, though still bundled with other changes.

What did not work: random ant spawn plus rare sources remained hard; simply
adding more ants and radius-2 vision from the weak rare-source checkpoint did
not rescue performance.

Paper-safe conclusion: 50x50 rare-source randomized foraging remains unsolved.
The best evidence says discovery/navigation is the bottleneck, but the vector
effect size should be rerun under matched schedules before being plotted as a
clean ablation.

### 9. Smooth/proximity/full-layout 50x50 line

Representative artifacts:

- `experiments/exploration_to_forage_scratch_smooth_sources_50x50.json`
- `experiments/exploration_to_forage_proximity_sources_50x50.json`
- `runs/notebooks/exploration_to_forage_50x50_efficient/**`
- `runs/notebooks/exploration_to_forage_proximity_sources_*`
- commit `41e5eba` on 2026-06-24, which introduced the named
  `critic_architecture` surface and the `strided_cnn` proximity-source config
  in the current history

This line explored padded arenas, positive-only rewards, smooth source-count
annealing, proximity/cluster curricula, 8-ant full-layout continuations, and
half-food settings. It also crosses a major critic boundary: the efficient
50x50 line has no `critic_architecture` field and therefore uses the default
flat `mlp` critic, while the proximity/full-layout configs explicitly use
`strided_cnn`. The exact settings differ, so compare cautiously.

Representative local W&B summaries:

| Run family | Final / best surviving metric | Interpretation |
| --- | ---: | --- |
| `exploration_to_forage_50x50_efficient` | later W&B summary: `21.5/48`, success `0.0` | Better than baseline but still not solved; default flat MLP critic. |
| `padded_sources_50x50` | `0.25/18` eval | Padded source workflow did not work in this form. |
| `proximity_sources_positive_only_50x50_outer_30x30_inner` | `40.875/250` eval | Positive-only/proximity plus `strided_cnn` critic made progress but not completion. |
| `full_layout_50x50_from_best` | `17.625/250` final eval; best stage eval `36.375/250` | `strided_cnn` full-layout continuation drifted; peak was better than final. |
| `8ants_half_food_2src_from_best` | best stage eval `28.56/125`, final `6.5/125` | `strided_cnn` half-food continuation; again peak-before-collapse. |
| `long3_from_long2_latest` | `43.875/125` final eval | Continued `strided_cnn` training improved the half-food line. |
| `shared_writes_from_64env_best` | `45.25/125` final eval | Shared write space helped modestly within the `strided_cnn` line. |
| `shared_writes_write_cost_from_shared_best` | `69.375/125`, success `0.125` | Strongest local half-food 50x50 full-layout result, still inside the `strided_cnn` line. |

The write-cost run is a real improvement in delivered fraction, but it does not
prove communication causality. The 4-bit write-cost summary still had high
write-action nonzero rate (`~0.94`), so the gain may come from a combination of
checkpoint lineage, `strided_cnn` critic credit assignment, shared-write
semantics, reward regularization, and training stability rather than sparse
meaningful messages alone.

There is another subtle lineage issue here: the 64-env run changed environment
parallelism and source checkpoint, the shared-write run continued from a
selected 64-env checkpoint, and the write-cost run continued from the
shared-write best. These should be plotted as a continuation trajectory, not as
independent randomized treatments.

Paper-safe conclusion: the 50x50 proximity/full-layout branch should be labeled
as a spatial-critic branch. It is not valid to compare it against earlier flat
MLP-critic runs as if only the source curriculum, ant count, or write mechanism
changed.

### 10. Write-cost multiplier and six-source sweeps

Representative artifacts:

- `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost*.json`
- `runs/notebooks/fl50_8ants_half_food_shared_writes_write_cost_x300_6src_from_latest/training_result.json`
- launch manifests for x5/x50/x100/x200/x500/x200_6src/x300_6src

Local completed evidence is strongest for `x300_6src`:

- best stage/eval at update 1: `68.375/125`
- final eval after 20,000 updates: `19.625/125`
- final write nonzero rate: `0.198`

This suggests severe write cost and/or six-source continuation reduced write
spam but did not preserve delivery. The peak at update 1 means the source
checkpoint already contained much of the useful behavior; subsequent training
degraded it.

The other multiplier runs have launch manifests and uploaded checkpoints but
not enough local metric history in this checkout to rank them safely.

### 11. 250x250 half-scale distance/autocurriculum

Representative artifacts:

- `runs/notebooks/half_scale_distance_autocurriculum_source_teacher_250x250/summary.json`
- `runs/training/half_scale_distance_autocurriculum_source_teacher_250x250/*`
- `runs/training/half_scale_distance_fixed8_source_reset_boundary_256_250x250/*`
- `runs/evaluation/fixed8-reset-boundary-champion-eval-20260630/summary.json`
- `runs/training/half_scale_one_food_resnet_actor_warmstart_250x250/*`
- `runs/training/half_scale_one_food_npc_teacher_from_source_250x250/*`

The first half-scale source-teacher/autocurriculum runs looked active by shaped
return but failed raw delivery at the end:

- notebook summary: `delivery_events=0`, `env_return=0`, `episode_return=14.64`,
  `mean_carrying_ants=447`
- diagnosis run: `delivery_events=0`, `pickup_events=0`, `shaping_return=26.25`,
  `mean_carrying_ants=257`

This exposed the pickup/carrying/shaping trap: positive progress rewards and
stage counters can hide the absence of actual delivery.

This branch also changed critic families. The early one-food 250x250 warmstart
used `resnet_cnn` and ended with zero pickups/deliveries; the NPC teacher run
used `strided_cnn` and also ended at zero. The later distance/autocurriculum and
reset-boundary runs in the local artifacts use `set_cnn`. So the 250x250 story
is not only a reward/reset story; it is also a critic-architecture search over
large sets of ants and sources.

The more subtle version is that some runs were not uniformly dead. The
diagnosis line had an early high-delivery window, then collapsed into zero
delivery while still accumulating shaping return and carrying many ants. The
failure mode is therefore "early usable behavior was not stabilized under the
curriculum," not just "the task never learned."

A reset-boundary fixed-distance intervention changed the local result:

- `fixed8-reset-boundary256` final summary: `delivery_events=654`,
  `pickup_events=869`, pickup-to-delivery `0.753`, `env_return=654`
- fresh-window eval, reset-boundary final: delivery median `215`, mean `269`,
  zero-delivery-window rate `0.0`
- fresh-window eval, reset-boundary best: delivery median `157.5`, mean `293`,
  pickup-to-delivery mean `0.852`

This is a meaningful late improvement, but it is a fixed-distance/source-boundary
result in the `set_cnn` critic family, with no broad curriculum completion
claim. It does not yet solve the 250x250 autocurriculum objective.

Other 250x250 continuations sharpen the same warning. The write-entropy/stage
window variant had nonzero delivery windows but ended with near-saturated write
activity and low delivery. The byte-decay continuation reduced write spam but
lost much of the reset-boundary delivery performance. The no-decay/frontier
variant started high and then declined. These are stability failures, not simple
"more communication regularization helps" results.

### 12. Current/incomplete 8-bit shared-write continuation

Representative artifacts:

- `runs/notebooks/fl50_8ants_half_food_shared_writes_write_cost_8bits_from_best/**`
- `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_8bits.json`

The July 1 run reached about `3200/20000` updates in the surviving output log,
wrote a 2,000-update checkpoint/video, and has no active training process now.
The output includes a CUDA OOM warning during startup and no final summary.

This run also changed two important things at once: write bits increased from 4
to 8, while write penalty dropped from `0.01` to `0.0002`. Treat it as
incomplete and non-causal: it is not evidence for or against 8-bit shared writes.

## Biggest Causal Deltas

Ranked by practical effect size and evidence quality:

1. Delivery-only reward plus depleting multi-bite food sources
   - Before: pickup and delivery rewards made "finding food" and "delivering
     food" less cleanly separated, and food was effectively many unit positions.
   - After the reset: pickup has no raw reward, delivery is `+1` per bite,
     sources deplete, and `food_sources` controls source concentration.
   - Interpretation: this defines the actual scientific task. It made
     pickup-to-delivery conversion the central metric and made source density a
     major causal variable.

2. Standard concentrated source geometry
   - Standard 25x25: `23` bites over `6` sources.
   - Standard 50x50 forage curriculum: `48` bites over `12` sources.
   - Rare-source 50x50: `23` bites, often down to `2` sources.
   - Interpretation: repeated exploitation of discovered sources is part of
     the solved 25x25 setting; rare-source discovery is part of why 50x50 stays
     hard.

3. `distance_bonus=0.02`
   - Before: single-ant curriculum collapsed around 25x25/50x50.
   - After `DISTANCE_SHAPE`: `13.75/23` deterministic and `7/23` sampled on
     held-out 25x25.
   - Interpretation: dense route-progress credit was the first training unlock
     after the task was framed as delivery-only depleting-source foraging.

4. Four ants/model capacity with distance shaping
   - `DISTANCE_CAP4` sampled result: `23/23`, success `1.0`.
   - Interpretation: exploration/coverage, plus the accompanying hidden-size
     increase, was the main missing ingredient for 25x25; bit capacity was not.

5. Action-mode selection
   - Greedy movement often failed while sampled movement solved or nearly
     solved.
   - Best practical deployment was sampled movement at temp `1.25` to `1.40`
     with greedy writing.
   - Interpretation: the policy learned useful route probabilities but not a
     robust argmax plan.

6. Spatial centralized critic for 50x50
   - Earlier 50x50-efficient line: default `mlp` critic, about `21.5/48` on a
     later W&B eval with success `0.0`.
   - Proximity/full-layout line: explicit `strided_cnn` critic; local results
     include `40.875/250` on proximity and later `69.375/125` in the write-cost
     half-food continuation.
   - Interpretation: this is a major credit-assignment/training architecture
     change. It does not change the actor's local deployment information in the
     same way, but it changes the value function used to train that actor.

7. More ants, shorter horizons, and source-count changes
   - 24/32-ant speed runs approached `22/23` within 430 steps.
   - Interpretation: very useful for performance, but it changes embodiment,
     horizon, and sometimes source density, weakening claims about communication
     or single-policy reasoning.

8. Explicit vector/selection bundles on rare 50x50
   - rare random-spawn baseline: `10.96/23`
   - hub vector: `12.33/23`
   - hub + nearest-food vector: `15.20/23`, held-out selection `16.20/23`
   - Interpretation: food discovery/navigation was the bottleneck, but matched
     reruns are needed for exact vector effect sizes.

9. Shared writes plus write cost in half-food 50x50
   - shared writes final eval: `45.25/125`
   - write-cost continuation final eval: `69.375/125`
   - Interpretation: promising practical improvement, but not yet a clean
     communication result, and it happens inside the `strided_cnn` critic line.

10. Reset-boundary fixed-distance 250x250
   - prior half-scale runs: zero deliveries despite positive shaped return
   - fixed8 reset-boundary final: 654 deliveries in training summary, fresh
     windows with zero-delivery-window rate `0.0`
   - Interpretation: fixing the reset/distance contract mattered more than
     generic longer training.

## What Worked

- Reframing the task around raw delivery from depleting multi-bite sources.
- Concentrating total food into fewer repeatable sources in the standard
  curriculum.
- Dense distance progress reward.
- Four-ant coverage for the 25x25 gate.
- Sampling movement while using greedy writing for deployment.
- Switching large-map training from a flat MLP critic to a spatial centralized
  critic (`strided_cnn`) for the 50x50 proximity/full-layout branch.
- Held-out shuffled evaluation and checkpoint selection.
- More ants for throughput-oriented 25x25 speed variants.
- Explicit hub/nearest-food vectors as diagnostic upper bounds on 50x50.
- Reset-boundary/fixed-distance diagnostics for the 250x250 source task.

## What Did Not Work

- Continuing the original single-ant curriculum.
- Increasing communication bits by itself.
- Treating greedy argmax as the main success metric.
- The entropy/PPO-tightening bundle or direct greedy fine-tuning as a full
  solution.
- The completed radius-2/vision bundles as a reliable fix.
- Random ant spawn plus rare 50x50 sources without explicit discovery aids.
- Severe write-cost/six-source continuation as a stable improvement after a
  good checkpoint.
- Trusting shaping return without raw delivery, pickup-to-delivery conversion,
  and carrying diagnostics.
- Treating `food_sources` as a minor display detail; it changes the task.
- Treating critic architecture as an implementation detail; for MAPPO it changes
  credit assignment and can move the learning curve without changing actor
  observations.

## Paper Claims Supported By Current Evidence

Strong:

- Current JAX MAPPO forage results are about delivery-only, depleting-source
  foraging; pre-reset pickup-reward behavior should not be mixed into the same
  result table.
- Food-source concentration is a major task variable. A `23/23` 25x25 result
  means clearing 23 bites from 6 depleting sources under the standard
  curriculum.
- A single local ant with sparse delivery reward does not scale reliably beyond
  small maps.
- Dense progress shaping plus multi-ant/model-capacity coverage can solve
  held-out randomized 25x25 foraging under sampled movement.
- Deployment mode matters: sampled movement captures learned route behavior much
  better than greedy argmax in many runs.
- The 50x50 proximity/full-layout branch is a `strided_cnn` centralized-critic
  branch. Its results should not be presented as only source-layout, write, or
  ant-count interventions.
- 50x50 rare-source randomized foraging remains open.

Moderate:

- Shared-write/write-cost continuations improve the 8-ant half-food 50x50 line.
- Spatial critic architecture is a plausible major contributor to the 50x50
  proximity/full-layout improvements, but exact effect size is not isolated
  without matched `mlp`/`strided_cnn` reruns.
- Explicit navigation/food-vector observations identify discovery as a 50x50
  bottleneck.
- Held-out checkpoint selection is important because final checkpoints often
  drift below earlier peaks.

Not yet supported:

- Byte-grid communication is the causal mechanism behind the main 25x25 solve.
- The 50x50 task is solved.
- More write bits improve behavior once ant count/reward shaping are controlled.
- The 250x250 half-scale autocurriculum is solved.

## Minimum Additional Work Before Paper Figures

1. Re-evaluate a small checkpoint set under one protocol:
   - `DISTANCE_SHAPE`
   - `DISTANCE_CAP4`
   - `DISTANCE_CAP4_LONG_CREDIT_GENTLE_GREEDY`
   - `DISTANCE_CAP4_NO_WRITE`
   - best 24-ant speed checkpoint
   - `DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_HELDOUT_SELECT`
   - best shared-write/write-cost 50x50 half-food checkpoint
   - fixed8 reset-boundary 250x250 checkpoint

   Keep `final`, `best`, and `held-out-selected` checkpoints in separate bars or
   panels. Do not mix them in one rank plot.

2. For each, report:
   - delivered fraction
   - success rate
   - actor observation contract and critic architecture
   - `food_count`, `food_sources`, and bites/source
   - delivery per 1000 ant-steps
   - first-pickup rate/time
   - pickup-to-delivery conversion
   - normal vs `no_byte_read` vs `no_write` where byte claims are made
   - sampled-movement/greedy-write plus greedy-movement/greedy-write

3. Normalize figures by task family:
   - 25x25 23-bite / 6-source gate
   - 25x25 speed 23-bite / source-count-specific / 430-step gate
   - 50x50 rare-source 23-bite / 2-source gate
   - 50x50 half-food 125-bite / source-count-specific gate
   - 250x250 fixed-distance/source task

4. Do not put raw episode return as the main cross-experiment plot. It mixes
   delivery, shaping, penalties, ant count, and horizon.

5. For causal figures, either rerun matched controls or label the x-axis with
   the full intervention bundle, e.g. "4 ants + hidden 192" or
   "hub vector + schedule/optimizer change" or "proximity curriculum +
   strided CNN critic."

## Suggested Paper Narrative

1. Define the corrected environment contract: delivery-only reward, depleting
   multi-bite sources, and the local observation/write channel.
2. Explain that `food_sources` is task geometry, not metadata.
3. Separate actor information from critic architecture; the critic changed from
   flat MLP to spatial CNN on the large-map branch.
4. Show the failure of sparse single-ant scale-up.
5. Show the 25x25 unlock: distance shaping, then four ants.
6. Explain the sampled-vs-greedy deployment gap.
7. Present ablations showing byte memory is not yet the cause of 25x25 success.
8. Present 50x50 as the frontier: strided-CNN-critic full-layout progress,
   rare-source failure, and vector diagnostics.
9. Present 250x250 as a diagnostic case study in reward-shaping failure:
   positive shaping without delivery, fixed by better reset-boundary metrics.

This story is honest and strong: it shows a real learning result, a clear
diagnostic methodology, and the boundaries of what is not solved yet.
