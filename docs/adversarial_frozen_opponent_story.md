# Adversarial Frozen-Opponent MAPPO Story

This document records the research path for the adversarial MAPPO branch. It is
not only an implementation index. The goal is to preserve why each experiment
was added, what changed in the environment, what failed, and why the current
checkpoint should be interpreted cautiously.

The short version: the adversarial lane became useful only after we treated it
as its own experiment family, kept the cooperative actor warm start alive with a
behavior anchor, selected checkpoints by evaluation instead of final rollout
noise, and made the environment diagnostics explicit enough to see when the
learner was exploiting a particular layout.

## Current Status

The branch implements a separate frozen-opponent adversarial lane under:

- `src/ant_byte_env/training/jax_mappo/adversarial/`
- `experiments/adversarial_frozen_opponent_probe.json`
- `experiments/adversarial_frozen_opponent_shared_writes_8ants.json`
- `notebooks/adversarial/frozen_opponent.ipynb`
- `notebooks/adversarial/capability_audit.ipynb`

The current selected checkpoint family is:

```text
runs/notebooks/adversarial_frozen_opponent/
  warmstart_shared_writes_8ants_team0_balance_refine_eval32_lr5e5_anchor0075/
```

The selected checkpoint is:

```text
runs/notebooks/adversarial_frozen_opponent/
  warmstart_shared_writes_8ants_team0_balance_refine_eval32_lr5e5_anchor0075/
  stage_01_target_125food_2src_dist20_36_mid16_eval32_lr5e5_anchor0075_team0_balance/
  checkpoints/best_model.pkl
```

The most meaningful render/eval action mode for this checkpoint family is:

```text
sampled_move_greedy_write
```

Deterministic renders were misleading for this line of work because the source
cooperative policy and the adversarial learner were trained and evaluated with
sampled movement but greedy writes.

## Initial Motivation

The original question was whether the AntByte setup could support a two-team
adversarial foraging game while preserving the small local actor. The important
research constraint was not to give the actor a much richer observation just
because the task became adversarial. The actor should still be cheap and local;
the centralized critic can be large, but the policy should remain close to the
cooperative communication experiment.

The first design therefore avoided live self-play. It used one trainable learner
team against one frozen opponent team loaded from the cooperative shared-writes
checkpoint. This made the opponent part of the learner's environment dynamics
and kept the first debugging target concrete.

The initial plan lives in:

```text
docs/planning/adversarial-marl-experiments.md
```

## V1 Implementation

The adversarial code was added as a separate lane rather than by rewriting the
cooperative environment. That separation was important because the cooperative
JAX runner and maintained experiment configs were already useful, and the
adversarial semantics were experimental.

The V1 environment contract:

- Two teams.
- `num_ants_per_team = N`.
- Total ant order is deterministic: all team 0 ants first, then all team 1 ants.
- `hub_pos` has shape `(2, 2)`.
- `delivered_food` has shape `(2,)`.
- Food, bytes, and obstacles are shared.
- A delivery counts only at the carrying ant team's own hub.
- Team rewards are exact opposites for one transition:
  `team0_delta - team1_delta` and `team1_delta - team0_delta`.

The V1 observation contract:

- Actor observations stay local.
- Own ants and own hub are encoded as positive local features.
- Opponent ants and opponent hub are encoded as negative local features.
- Off-screen opponents and off-screen hubs are not revealed.
- No new actor planes were appended, so the cooperative actor observation shape
  stayed compatible.

Warm start policy:

- The cooperative actor body, move head, and write head are copied.
- The frozen opponent gets the same actor by default.
- The adversarial critic is freshly initialized.
- The optimizer is freshly initialized.

The cooperative critic was not loaded because it predicts cooperative delivery
value, while the adversarial value target is learner advantage:

```text
own deliveries - opponent deliveries
```

This distinction mattered later. We did add a critic warmup option, but that is
not the same as loading the cooperative critic.

## First Probe And Early Debugging

The first real pilot was intentionally small and local-safe:

```text
experiments/adversarial_frozen_opponent_shared_writes_8ants.json
```

Important local-safe settings:

- `num_envs = 12`
- `num_steps = 96`
- `num_ants_per_team = 8`
- `food_count = 125`
- `food_sources = 2`
- `actor_vision_radius = 2`
- `write_bits = 4`
- `gamma = 0.997`
- `training_rollout_temperature = 0.75`
- opponent and eval action mode: `sampled_move_greedy_write`

The branch also gained a tiny probe config:

```text
experiments/adversarial_frozen_opponent_probe.json
```

That probe exists to validate mechanics quickly: reset, delivery counters, reward
signs, action composition, and a one-update dry run.

Early notebook work exposed two practical problems:

1. Evaluation and rendering could be slow enough to crash or stall the Jupyter
   kernel when embedded too heavily in training cells.
2. Rendering needed to match the adversarial action mode; deterministic videos
   made the frozen policy look weaker or stranger than the sampled-move policy
   actually was.

The response was to move reusable evaluation/rendering helpers into source files
and keep the notebook orchestration thinner.

## Capability Audit Before More Training

Before spending long compute, we audited whether the frozen cooperative actor
was capable in the adversarial layouts at all. This mattered because training
against a broken frozen opponent would teach the wrong lesson.

The audit checked:

- frozen checkpoint vs frozen checkpoint;
- learner/frozen side swaps;
- random vs frozen;
- full writes vs zero writes;
- sampled movement vs deterministic movement;
- fixed and randomized layout diagnostics.

One useful sanity check was the source-mode frozen-vs-frozen run:

```text
runs/notebooks/adversarial_frozen_opponent/
  frozen_vs_frozen_debug/frozen_vs_frozen_source_mode_8ep_metrics.json
```

It reported a small mean delivery difference of `2.875` over 8 episodes, with
win rate `0.5`. That was good enough to keep going: the frozen cooperative actor
was not random, and the two-team mechanics were not obviously one-sided.

## Why Early Training Looked Bad

The first training attempts produced many negative returns and weak videos. The
main failure mode was forgetting.

The cooperative checkpoint knew how to forage in a single-hub world. Once PPO
started updating the actor under the adversarial reward, the learner could drift
away from the cooperative foraging behavior faster than it learned the new
adversarial objective. Sparse food, distant hubs, and one-shot final checkpoints
made this worse. In practice the model could stop looking like the useful source
policy before it had learned a robust adversarial behavior.

We considered whether the cooperative critic should be warm-started, but the
critic target mismatch made that unsafe as a default. Instead the branch added
two safer stabilizers:

- optional critic warmup by freezing the actor and training only the adversarial
  critic for early updates;
- optional KL behavior anchor from the learner actor back to a fixed reference
  actor.

The critic warmup knob is a way to let the fresh adversarial critic adapt before
the actor moves. It is not cooperative critic transfer.

The KL behavior anchor became more important. It directly penalizes the learner
for drifting too far from the warm-start actor on the learner actor observations.
That lets us ask for adversarial improvement while keeping the low-cost policy
near the behavior that already knows how to forage.

## Curriculum Changes

The environment also changed across experiments. The first target setting was
too hard to learn from: sparse food, random hubs, and limited interaction meant
the learner could spend many updates seeing mostly unhelpful negative signal.

The curriculum first made food easier:

```text
warmstart_shared_writes_8ants_food_curriculum_cpu_fast
```

That used a simple food schedule:

- `500` food, `16` sources;
- `250` food, `8` sources;
- `125` food, `2` sources.

The next curriculum made this more gradual:

```text
warmstart_shared_writes_8ants_food_curriculum_10k_cpu_fast
```

Then the environment added hub placement controls. The important change was not
to hard-code one exact setup, but to sample from ranges:

- `hub_pair_distance_min`
- `hub_pair_distance_max`
- `food_midpoint_window_size`
- `layout_margin`
- `hub_center_window_size`

This let training start with easier contests and move toward the target layout:

- many food sources early;
- hubs closer early;
- food near the midpoint early;
- then fewer sources and wider hub distances;
- target stage: `125` food, `2` sources, hub distance `20-36`, midpoint window
  `16`, layout margin `6`.

The purpose was to create adversarial contact without overfitting to one exact
centered map.

## Eval-Gated Checkpointing

Another important change was best-checkpoint selection.

Final rollout metrics were noisy and sometimes lucky. The branch added
runner-level support for eval-gated checkpointing:

- `--save-best-model`
- `--best-model-metric`
- `--best-model-mode`
- `--best-model-selection`
- `--best-eval-episodes`
- `--best-eval-interval`

This made the notebook chunk orchestration safer. Instead of trusting the last
chunk, the runner could save the best checkpoint according to an evaluation
matrix metric.

Early eval-gated runs selected by:

```text
eval_learner_vs_frozen_mean_delivery_difference
```

Later runs selected by:

```text
eval_learner_vs_frozen_side_swap_adjusted_delivery_difference
```

The side-swap-adjusted metric was introduced because a checkpoint can look good
on one side while exposing a strong team/order bias. Side swaps are not cosmetic
here; they are a symmetry check on the experiment.

## KL Behavior Anchor

The KL anchor was added after the forgetting problem became clear.

The CLI flags are:

- `--behavior-anchor-coef`
- `--behavior-anchor-model`

When no explicit anchor model is provided, the anchor defaults to the
post-transfer learner actor. Checkpoints store the anchor params so resumed
chunks keep the same reference behavior.

The anchor is applied to the actor distribution on learner actor observations.
It is not reward shaping. This was intentional: we did not want the negative
opponent-delivery term to become a minor detail hidden under auxiliary rewards.

The final selected run used:

```text
behavior_anchor_coef = 0.0075
learning_rate = 5e-05
ent_coef = 0.00015
best_eval_episodes = 32
best_model_metric = eval_learner_vs_frozen_side_swap_adjusted_delivery_difference
```

## What The Selected Checkpoint Does

On randomized target layouts, the current selected checkpoint is better than
random and usually better than frozen, but the margin is not huge:

```text
runs/notebooks/adversarial_frozen_opponent/
  warmstart_shared_writes_8ants_team0_balance_refine_eval32_lr5e5_anchor0075/
  final_eval_32ep_2000step.json
```

Selected 32-episode randomized metrics:

- frozen vs frozen mean delivery difference: `3.875`
- learner vs frozen mean delivery difference: `13.4375`
- learner vs frozen side-swap-adjusted delivery difference: `9.5`
- learner vs frozen win rate: `0.625`
- learner vs random mean delivery difference: `30.40625`
- learner vs random win rate: `1.0`
- random vs frozen mean delivery difference: `-32.71875`
- random vs frozen win rate: `0.0`
- side-swapped score gap: `7.875`

On the fixed-center diagnostic scene, the same checkpoint looks much stronger:

```text
runs/notebooks/adversarial_frozen_opponent/
  warmstart_shared_writes_8ants_team0_balance_refine_eval32_lr5e5_anchor0075/
  fixed_center_eval_32ep_2000step.json
```

Fixed-scene setup:

- hubs: `[[21, 25], [29, 25]]`
- food sources: `[[25, 23], [25, 27]]`
- food count: `125`
- food sources: `2`

Selected fixed-scene metrics:

- learner vs frozen mean delivery difference: `56.46875`
- learner vs frozen side-swap-adjusted delivery difference: `40.9375`
- learner vs frozen win rate: `1.0`
- frozen vs learner mean delivery difference: `87.53125`
- learner vs random mean delivery difference: `18.40625`
- random vs frozen mean delivery difference: `-29.78125`

This is strong evidence that the learner found something useful in the fixed
contest, but it also warns us that the exploit is layout-sensitive.

## Interpreting The Learned Behavior

The fixed-center render suggested a plausible adversarial mechanism: the learner
uses writes near the opponent hub approach area to disrupt the frozen policy's
return path. The environment does not allow writing directly on hub tiles, so
"delete bits in the hub" should be read as "overwrite or clear bytes on tiles
close to the hub."

This matters because the frozen cooperative policy is exploitable. It learned to
use shared write bits in a cooperative single-hub world. In the adversarial
world, the learner can write in places that are useful to itself and harmful to
the opponent. The frozen policy can get lost or confused near the wrong hub or
near corrupted approach trails.

That interpretation is promising, but not proven as a general strategy yet. The
randomized target eval is only modestly positive, while the fixed-center eval is
very strong. The current best read is:

- the learner learned a real exploit against this frozen policy;
- the exploit is clearest when hubs and food create repeatable contact near the
  center;
- the policy still has a lot to learn if the goal is robust adversarial play
  across broad randomized layouts.

## Handicap Sweep

To test how much of the result depended on learner team size, we evaluated the
selected checkpoint on the same fixed test scene while masking learner slots to
simulate fewer active learner ants. The underlying checkpoint and env stayed
8-slot compatible; inactive learner slots were forced to `STAY/write=0`.

Artifact:

```text
runs/notebooks/adversarial_frozen_opponent/
  handicap_evals/fixed_center_learner_handicap_4v4_to_1v4_16ep_latest.json
```

16-episode sweep against 4 active frozen ants:

| Mode | Mean Delivery Difference | Learner Win Rate |
| --- | ---: | ---: |
| 4v4 | `35.8125` | `1.0` |
| 3v4 | `14.1875` | `0.625` |
| 2v4 | `2.5625` | `0.75` |
| 1v4 | `-48.1875` | `0.1875` |

The boundary was then confirmed with 64 episodes:

```text
runs/notebooks/adversarial_frozen_opponent/
  handicap_evals/fixed_center_learner_handicap_boundary_2v4_1v4_64ep_latest.json
```

64-episode boundary sweep:

| Mode | Mean Delivery Difference | Learner Win Rate |
| --- | ---: | ---: |
| 2v4 | `-4.828125` | `0.453125` |
| 1v4 | `-70.734375` | `0.03125` |

The practical conclusion is that the learner stays strong with 4 or 3 active
ants, the crossover is around 2 active ants, and 1 active ant is not enough.

## Rendering Choices

The adversarial renderer was adjusted to make role identity visible. The useful
semantic convention is:

- trained learner role: red;
- frozen opponent role: blue.

This is role-based, not hard-coded to team index. If the learner team is swapped,
the trained learner should still be red.

For videos, prefer MP4 over GIF. The notebook uses MP4 because long 2000-step
renders and fixed-scene diagnostics were too heavy and crash-prone in notebook
GIF workflows.

For fixed-scene diagnostics, the render intentionally uses the fixed hubs and
two food sources near the center:

```text
fixed_hub_positions = [[21, 25], [29, 25]]
fixed_food_positions = [[25, 23], [25, 27]]
```

## What Changed In The Environment Over Time

The environment did not change all at once. The sequence was roughly:

1. Two-team env with separate hubs and signed local observations.
2. Frozen opponent action composition in deterministic team order.
3. Evaluation matrix with side swaps and random baselines.
4. Source-mode action support, especially `sampled_move_greedy_write`.
5. Hub distance and food midpoint knobs.
6. Fixed-layout evaluation/render overrides.
7. Optional no-food-termination only for render diagnostics when needed.
8. Role-colored rendering so trained vs frozen identity stays obvious.
9. Handicap eval utilities that keep checkpoint-compatible 8 slots and mask
   inactive learner ants.

The most important environment knobs for current experiments are:

- `--hub-pair-distance-min`
- `--hub-pair-distance-max`
- `--food-midpoint-window-size`
- `--layout-margin`
- `--hub-center-window-size`
- `--learner-team`
- `--eval-action-mode sampled_move_greedy_write`
- `--opponent-action-mode sampled_move_greedy_write`

## Lessons Learned

1. Actor-only warm start is necessary but not sufficient.

The cooperative actor starts with useful foraging behavior, but PPO can quickly
erase it under the adversarial reward.

2. The cooperative critic should not be loaded by default.

The cooperative value target and adversarial advantage target are different
objects. A critic warmup phase is safer than pretending the cooperative critic is
already calibrated.

3. KL anchoring is the cleanest stabilizer so far.

It keeps the actor near the source behavior without changing the reward.

4. Eval-gated checkpointing matters.

Final training chunks can look good or bad for accidental reasons. Selection by
evaluation metrics, especially side-swap-adjusted metrics, made the run much
easier to reason about.

5. Fixed scenes are diagnostics, not proof of generalization.

The fixed-center scene is excellent for seeing mechanism, but randomized target
layouts are the better measure of robustness.

6. The frozen opponent is exploitable.

The frozen cooperative policy can be disrupted through shared writes near its
hub approach area. That is scientifically interesting, but it also means frozen
opponent results should not be overclaimed as general adversarial competence.

## Recommended Continuation

Before moving to live self-play, the next adversarial steps should be:

1. Confirm the write-disruption mechanism with targeted metrics around the
   opponent hub approach area.
2. Run no-write or protected-write diagnostics only as controlled tests, not as
   the main training objective.
3. Train against a small pool of frozen cooperative checkpoints or frozen
   snapshots so the learner cannot overfit to one brittle opponent.
4. Keep side-swap-adjusted eval selection.
5. Keep randomized target eval and fixed-scene render side by side.

Only after those pass should we move toward shared-policy self-play.

## Artifact Index

Planning:

- `docs/planning/adversarial-marl-experiments.md`

Core implementation:

- `src/ant_byte_env/training/jax_mappo/adversarial/env.py`
- `src/ant_byte_env/training/jax_mappo/adversarial/observations.py`
- `src/ant_byte_env/training/jax_mappo/adversarial/rollout.py`
- `src/ant_byte_env/training/jax_mappo/adversarial/runner.py`
- `src/ant_byte_env/training/jax_mappo/adversarial/evaluation.py`
- `src/ant_byte_env/training/jax_mappo/adversarial/rendering.py`
- `src/ant_byte_env/training/jax_mappo/adversarial/transfer.py`
- `src/ant_byte_env/training/jax_mappo/adversarial/checkpointing.py`

Configs and notebooks:

- `experiments/adversarial_frozen_opponent_probe.json`
- `experiments/adversarial_frozen_opponent_shared_writes_8ants.json`
- `notebooks/adversarial/frozen_opponent.ipynb`
- `notebooks/adversarial/capability_audit.ipynb`

Selected result artifacts:

- `runs/notebooks/adversarial_frozen_opponent/warmstart_shared_writes_8ants_team0_balance_refine_eval32_lr5e5_anchor0075/final_eval_32ep_2000step.json`
- `runs/notebooks/adversarial_frozen_opponent/warmstart_shared_writes_8ants_team0_balance_refine_eval32_lr5e5_anchor0075/fixed_center_eval_32ep_2000step.json`
- `runs/notebooks/adversarial_frozen_opponent/handicap_evals/fixed_center_learner_handicap_4v4_to_1v4_16ep_latest.json`
- `runs/notebooks/adversarial_frozen_opponent/handicap_evals/fixed_center_learner_handicap_boundary_2v4_1v4_64ep_latest.json`
