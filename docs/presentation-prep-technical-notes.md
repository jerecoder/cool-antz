# Presentation Prep: Shrink Vision, Lethal Cookies, and Wall Experiments

This is a technical defense sheet for the part of the project covering:

- shrink-vision curriculum learning,
- lethal cookies,
- random walls,
- near-nest walls,
- the later 60-ant transfer/stabilization probes.

Use it as a source of answers, not as slide text. The public webpage gives the
high-level narrative; this file keeps the implementation details, settings,
pipeline shape, and common traps in one place.

## Public Framing

The report site introduces the project as a question about emergent coordination
in a grid-world ant colony:

- Agents are ants in a foraging world.
- The actor is local and decentralized: each ant sees only a small oriented
  window around itself.
- Training uses MAPPO with a centralized critic: the critic can see the global
  state during training, but the actor policy is what matters at rollout time.
- Ants can write bytes into the map, so the project asks whether external memory
  becomes useful for coordination.
- The main public metric is food delivery to the nest/colony, often normalized
  as delivered food per ant step or delivered fraction.

The audience-facing story for this part:

1. Shrink vision asked whether giving the ants a very easy full-view task first
   would let us compress the policy down to local vision.
2. Lethal cookies asked whether ants can learn danger without an explicit danger
   label, using dead ants as an observable local signal.
3. Random walls and near-nest walls asked whether that behavior survives
   navigation obstacles instead of only working in open layouts.
4. The 60-ant transfer probe asked whether a strong non-lethal navigation policy
   could survive fine-tuning with lethal cookies and walls.

## Core MAPPO Pipeline

The training loop is:

1. Build an environment reset: hub, ants, food, lethal food if enabled, walls if
   enabled, byte grid.
2. Convert the raw environment observation into two network inputs:
   actor observations for each ant, and one centralized critic observation per
   environment.
3. Actor samples one `(move, write_value)` pair per ant.
4. Environment steps all ants, updates positions, food, bytes, deliveries,
   deaths, visited/viewed cells, and returns reward/info.
5. Runner stores rollout tensors for `num_steps * num_envs`.
6. PPO computes advantages/returns with the centralized critic.
7. Actor and critic update for `update_epochs` over `num_minibatches`.
8. Notebook workflow saves checkpoints, evaluates the best checkpoint, and
   renders preview or checkpoint videos.

Important deployment distinction:

- During training, the critic can use the centralized observation.
- During rollout/deployment, action selection only needs the actor input.
- Therefore the critic may know global state, but it cannot make ants magically
  see global information at execution time.

## Actor And Critic Inputs As Matrices

### Generic Local Actor Input

For the modern local actor, an ant with vision radius `r` receives an oriented
patch of side:

```text
side = 2 * r + 1
patch_size = side * side
```

With radius `2`, each ant sees a `5x5` local window. The local patch is rotated
into the ant's facing direction, so "forward" is consistent in actor input
space.

Conceptually, the actor input is a stack of local matrices plus a small vector
tail:

```text
Actor input for one ant

food patch              5x5
live-ant-count patch    5x5
dead-ant-count patch    5x5, only in lethal-cookie runs
byte bit 0 patch        5x5
byte bit 1 patch        5x5
...
byte bit k patch        5x5
hub patch               5x5
border/wall patch       5x5
identity features       vector
carrying flag           scalar
facing one-hot          4-vector
```

For the 60-ant sparse random-wall/lethal setup from checkpoint metadata:

```text
num_envs = 16
num_ants = 60
actor_vision_radius = 2
side = 5
write_bits = 8
agent_identity_types = 8

grid channels =
  food
  live ants
  dead ants
  8 byte-bit planes
  hub
  border/obstacle
= 13 channels

matrix part = 13 * 5 * 5 = 325
tail        = 8 identity + 1 carrying + 4 facing = 13
actor dim   = 338

actor batch shape = (16, 60, 338)
```

If someone asks "what does an actual input look like?", draw it like this:

```text
food:
0 0 0 0 0
0 0 1 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0

dead ants:
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0

wall/border:
0 0 0 0 0
0 0 0 0 0
0 0 0 1 1
0 0 0 1 1
0 0 0 0 0

tail:
[identity_type_0..7, carrying, facing_up, facing_right, facing_down, facing_left]
```

The key phrase: the actor does not receive a global map. It receives local
oriented matrices and a few per-ant features.

### Centralized Critic Input

For the same 60-ant setup, the critic checkpoint metadata had:

```text
central_obs_dim = 10424
central batch shape = (16, 10424)
```

That vector can be read as:

```text
ant table:
  60 ants * 7 features = 420
  features are normalized x/y position, carrying flag, and facing one-hot

global grid maps:
  live-ant-count map     50x50
  dead-ant-count map     50x50
  visible-food map       50x50, safe + lethal merged
  byte-value map         50x50
  total = 4 * 2500 = 10000

small global tail:
  normalized hub x/y
  normalized active grid width/height
  total = 4

central dim = 420 + 10000 + 4 = 10424
```

The `strided_cnn` critic consumes the spatial part as global maps and processes
them with stride-2 convolutions, then fuses the spatial embedding with the ant
table/global tail. In words: the critic sees the whole board during training,
but it predicts value only. It does not choose actions directly.

### Old Shrink-Vision Actor Input

The old shrink-vision MLP actor in `cool-antz-viejo` used a simpler formula:

```text
actor_dim = side^2 * (write_bits + 3) + 1
```

The local grid channels were:

- food,
- write-bit planes,
- hub,
- border,
- plus one carrying flag outside the patch.

With `write_bits = 5`:

```text
radius 25, side 51: actor_dim = 51^2 * 8 + 1 = 20809
radius 1,  side 3:  actor_dim = 3^2  * 8 + 1 = 73
```

That huge first input layer is one reason the "start with full vision" idea was
not automatically easier.

## Shrink-Vision Curriculum

### Goal

The hypothesis was:

> Start with large/full actor vision where the task should be easier, learn
> foraging, then progressively shrink the actor vision while warm-starting from
> the previous checkpoint.

The map stayed fixed at `50x50`. The curriculum changed the actor observation
radius, not the map size.

### Old Dense Curriculum

Main old files:

- `cool-antz-viejo/cool-antz/experiments/vision_shrink_curriculum.json`
- `cool-antz-viejo/cool-antz/notebooks/train_jax_vision_shrink_curriculum.ipynb`
- `cool-antz-viejo/cool-antz/src/ant_byte_env/training/jax_mappo/transfer.py`

Key settings:

| Setting | Value |
| --- | --- |
| Map | `50x50` |
| Backend | JAX MAPPO |
| Actor architecture | MLP |
| Vision radii | `[25, 20, 15, 10, 6, 3, 2, 1]` |
| Vision sides | `[51, 41, 31, 21, 13, 7, 5, 3]` |
| `num_ants` | `10` |
| `food_count` | `48` |
| `food_sources` | `25` |
| `cookie_distance` | `24` |
| `num_envs` | `64` |
| `num_steps` | `10000` |
| `num_minibatches` | `4` |
| `update_epochs` | `4` |
| `hidden_size` | `128` |
| `write_bits` | `5` |
| `pickup_bonus` | `0.25` |
| `distance_bonus` | `0.02` |
| `gamma` | default unless overridden by runner |
| Stage update cap | `10000` in metadata |
| First-layer freeze | `freeze_actor_first_layer_only_updates = 50` |

Dense transfer mechanism:

- The first actor dense layer depends on actor input size.
- When shrinking from a larger side to a smaller side, the transfer function
  center-crops each spatial channel of the old first-layer weights.
- The carrying row is copied.
- Later actor layers, move head, write head, critic/value head are copied.
- Adam optimizer state is reset.
- The first layer can be warmed up/frozen separately for a few updates.

What to say if asked why it struggled:

- Full vision increased input dimensionality massively.
- A large observation did not solve credit assignment or exploration.
- Without recurrent memory, the policy still had to learn pickup, return, and
  navigation from sparse signals.
- The first layer transfer preserved central local features, but it could not
  invent a stable foraging strategy if the large-vision policy was weak.

### Conv Vision-Shrink Curriculum

Main old files:

- `cool-antz-viejo/cool-antz/experiments/vision_shrink_conv_curriculum.json`
- `cool-antz-viejo/cool-antz/experiments/vision_shrink_conv_autoresearch.json`
- `cool-antz-viejo/cool-antz/notebooks/train_jax_conv_vision_shrink_curriculum.ipynb`
- `cool-antz-viejo/cool-antz/autoresearch/vision_shrink_conv_program.md`
- `cool-antz-viejo/cool-antz/src/ant_byte_env/training/jax_mappo/autoresearch.py`

Why try a conv actor:

- A dense actor has an input-size-specific first layer.
- A conv actor can reuse spatial filters over different patch sizes.
- The transfer no longer needs to crop the first layer; it reuses the conv
  filters, actor heads, critic, and optimizer state.

Conv actor shape:

```text
input patch side x side x channels
conv1 -> tanh
conv2 -> tanh
mean pool + max pool
concat carrying flag
dense actor body
move head and write head
```

Conv handoff config values:

| Setting | Value |
| --- | --- |
| Map | `50x50` |
| Actor architecture | `conv` |
| `conv_channels` | `32` |
| `hidden_size` | `128` |
| `learning_rate` | `0.001` |
| `anneal_lr` | `true` |
| `gamma` | `0.995` |
| `gae_lambda` | `0.95` |
| `clip_coef` | `0.3` |
| `ent_coef` | `0.01` |
| `vf_coef` | `0.5` |
| `max_grad_norm` | `1.0` |
| `num_envs` | `16` |
| `num_steps` | `4096` |
| `num_minibatches` | `8` |
| `update_epochs` | `8` |
| `max_steps` | `256` |
| `pickup_bonus` | `2.2` |
| `distance_bonus` | `0.24` |
| `write_bits` | `5` |
| Metadata stage cap | `240` |

Important artifact mismatch:

- The conv curriculum JSON metadata says the target is final `3x3` actor vision.
- The listed notebook handoff radii were `[25, 20, 15, 10, 6, 3]`, which stops
  at radius `3`, side `7x7`.
- The autoresearch trial curriculum did include `[6, 3, 2, 1]`, so the short
  search target was truly final radius `1`, side `3x3`.
- If asked, say this is a stale handoff metadata/config mismatch, not a change
  in the algorithm.

Notebook stage loop:

- Reads `experiments/vision_shrink_conv_curriculum.json`.
- Writes to `runs/notebooks/vision_shrink_conv_curriculum`.
- Each stage checkpoint is named
  `jax_mappo_conv_vision_stage_<side>x<side>.pkl`.
- If a stage checkpoint already exists and `SKIP_COMPLETED_STAGES = True`, the
  notebook skips training for that stage.
- `RERENDER_EXISTING_ROLLOUTS` controls whether old GIFs are regenerated.
- For each stage:

```text
stage_total_timesteps = stage_num_envs * stage_num_steps * GLOBAL_UPDATE_CAP
```

Rollout sizing used in the notebook:

```text
radius >= 20: 1 env, 512 steps
radius >= 10: 2 envs, 512 steps
smaller:      base num_envs/num_steps from config
```

Autoresearch pipeline:

1. Verify JAX CUDA backend.
2. Edit only `experiments/vision_shrink_conv_autoresearch.json`.
3. Run one GPU trial:

```bash
PYTHONPATH=src JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python -m ant_byte_env.training.jax_mappo.autoresearch \
  --config experiments/vision_shrink_conv_autoresearch.json \
  --score-metric trajectory_return \
  --trial-name <tag>
```

4. Read `runs/autoresearch/vision_shrink_conv_autoresearch/<trial>/summary.json`.
5. Append `autoresearch/vision_shrink_conv_results.tsv`.
6. Keep/discard based on `trajectory_return`.
7. On target success, write `autoresearch/vision_shrink_conv_success.json`.

Autoresearch search knobs:

- `learning_rate`
- `ent_coef`
- `pickup_bonus`
- `distance_bonus`
- `gamma`
- `gae_lambda`
- `clip_coef`
- `max_grad_norm`
- `num_minibatches`
- `update_epochs`
- `hidden_size`
- `conv_channels`
- `max_steps`
- short-trial metadata: `trial_update_cap`, `trial_num_envs`,
  `trial_num_steps`

Autoresearch success artifact:

```text
target_trajectory_return = 60
score = 88.8599853515625
trajectory_return = 88.8599853515625
env_return = 13.5
summary_path = runs/autoresearch/vision_shrink_conv_autoresearch/shape22-dist024-steps4096-env2/summary.json
description = target reached with 4096-step 2-env horizon, 16 minibatches,
              pickup_bonus=2.2, distance_bonus=0.24, aggressive PPO settings
```

What to say in one sentence:

> The shrink-vision curriculum was technically implemented and transfer worked
> mechanically, but the key lesson was that more vision was not a free
> curriculum: large inputs made optimization harder, and the base behavior was
> not reliable enough to compress cleanly.

## Lethal Cookies

### Mechanism

The lethal-cookie implementation is in historical/run code, especially commit
`58ea666`/`f75f7c8`. The current report branch source may not contain all lethal
JAX code, so use the historical commit and run artifacts for technical details.

Internal environment state separates:

```text
state.food         safe food
state.lethal_food lethal food
state.ants_alive  live/dead mask
```

But the actor/critic observation intentionally merges safe and lethal food:

```text
obs["food"] = state.food + state.lethal_food
```

That means there is no explicit "this is lethal" input channel. The new signal
is generated after an ant dies:

```text
obs["dead_ants_count"]
```

Pickup dynamics:

- If an alive, non-carrying ant steps onto safe food, it picks up food.
- If it steps onto lethal food and was not already carrying, it dies.
- Lethal pickup decrements `state.lethal_food`.
- The ant becomes inactive in `ants_alive`.
- Dead bodies are visible through `dead_ants_count`.
- Death subtracts `death_penalty` from reward.
- Delivery still gives `+1` per safe food returned to the hub.

Why this matters:

- Lethal cookies are hidden before contact.
- The policy can only learn danger by associating local corpses with future
  pickup risk.
- This is a social/environmental trace, not a privileged danger label.

### Open-Map Lethal Run

Main source:

- Historical config: `f75f7c8:experiments/exploration_to_forage_proximity_sources_50x50.json`
- Run directory:
  `runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_50x50_outer_30x30_inner`
- Best checkpoint:
  `checkpoints/best_proximity_sources_lethal_cookies2.pkl`

Key settings:

| Setting | Value |
| --- | --- |
| Source | continuation from positive-only proximity-sources checkpoint |
| Arena | `50x50` |
| Task window | interior `30x30` via `layout_margin = 10` |
| Hub window | centered `4x4` |
| Actor radius | `2` |
| Ants | `10` |
| Safe food | `50` units in `1` source |
| Lethal food | `50` units in `1` source |
| Safe/lethal distance | same Chebyshev ring, `cookie_distance = 9` |
| `max_steps` | `1000` |
| Reward mode | `forage` |
| Delivery reward | `+1` |
| Death penalty | `1.0` |
| Pickup bonus | `0.5` in the stronger run |
| Distance/carrying/view/visit rewards | `0` |
| Critic | `strided_cnn` |
| `write_bits` | `4` |
| PPO envs/steps | `16 envs`, `256 steps` |
| PPO minibatches/epochs | `4 minibatches`, `1 epoch` |
| `gamma` | `0.997` |
| Eval episodes | `8` |
| Eval action mode | `sampled_move_greedy_write` |
| Eval move temperature | `0.75` |
| Best metric | `eval_mean_delivered_food_per_1000_ant_steps` |

Important live-config note:

- The raw JSON contains a very large `total_timesteps`, but the notebook stage
  runner controlled the actual stage length.
- The best W&B summary for `best_proximity_sources_lethal_cookies2.pkl` shows
  `80000` updates and `327,680,000` global steps.
- As usual in this repo, `args` are behavior, while `metadata` can be a label or
  stale note. If they disagree, trust the executed run/config summary.

Observed result:

```text
eval_mean_delivered_food = 4.75
eval_mean_delivered_fraction = 0.095
eval_mean_episode_return = 4.375
eval_mean_delivered_food_per_1000_ant_steps = 0.475
```

Interpretation:

- With death penalty and hidden lethal food, the policy first tended to avoid
  pickup too much.
- Increasing pickup shaping recovered risk-taking.
- The best open-map policy learned visible local avoidance around corpse traces,
  but delivery was still weak compared with the non-lethal task.

## Random Walls

### Motivation

Open-map lethal behavior was not enough. When tested in maze-like layouts, the
policy could fail because:

- walls changed navigation,
- side rooms made local corpse/danger association harder,
- open-map motion habits did not transfer cleanly.

Random walls were added to train with obstacle variation instead of only testing
obstacles afterward.

### Initial Random-Wall Lethal Run

Main source:

- Notebook: `notebooks/source_layouts/proximity_sources_50x50_random_walls.ipynb`
- Run directory:
  `runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_random_walls_50x50_outer_30x30_inner`
- Initial checkpoint:
  `best_proximity_sources_lethal_cookies2.pkl`
- Best checkpoint:
  `checkpoints/best_proximity_sources_lethal_cookies_random_walls.pkl`

Executed W&B config highlights:

| Setting | Value |
| --- | --- |
| Arena | `50x50`, interior `30x30` |
| Hub window | `4x4` |
| Actor radius | `2` |
| Ants | `2` |
| Safe stage | `50` one-tile sources, stage name `50x50_clusters_50_r00_sources_050` |
| Lethal food | `50` lethal sources/units in the initial random-wall run |
| Death penalty | `1.0` |
| Pickup bonus | `0.5` |
| `max_steps` | `10000` |
| `write_bits` | `4` |
| Critic | `strided_cnn` |
| Random wall layouts | `256` |
| Wall seed | `11` |
| Wall count | `1..4` |
| Wall length | `6..18` |
| Wall width | `2` |
| L-turn probability | `0.6` |
| PPO | `16 envs`, `256 steps`, `4 minibatches`, `1 epoch` |
| Global update cap | `80000` |
| Checkpoint video interval | `5000` updates |
| Rollout temperature | `0.5` |

Observed result:

```text
eval_mean_delivered_food = 2.5
eval_mean_delivered_fraction = 0.05
eval_mean_episode_return = 0.5
eval_mean_delivered_food_per_1000_ant_steps = 6.6110278119606845
```

Interpretation:

- The policy showed local exclusion around corpse/danger regions.
- It did not become a robust wall-navigation forager.
- The result supported the idea that danger avoidance and navigation are
  separable difficulties.

## Near-Nest Walls

### Motivation

Near-nest walls changed the pressure:

- Reduce safe food to sparse single-bite sources.
- Keep lethal food near the nest so danger cannot be forgotten.
- Put more wall structure around the central/hub area.
- Force paths around walls, instead of letting the policy rely on open-map
  habits.

### Executed Near-Nest Run

Main source:

- Notebook: `notebooks/source_layouts/proximity_sources_50x50_random_walls.ipynb`
- Run directory:
  `runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_near_nest_walls_50x50_outer_30x30_inner`
- Initial checkpoint:
  `best_proximity_sources_lethal_cookies_random_walls.pkl`
- Best checkpoint:
  `checkpoints/best_proximity_sources_lethal_cookies_near_nest_walls.pkl`

Executed W&B config highlights:

| Setting | Value |
| --- | --- |
| Arena | `50x50`, interior `30x30` |
| Hub window | `4x4` |
| Actor radius | `2` |
| Ants | `2` |
| Safe food | `12` one-bite sources, stage `50x50_clusters_12_r00_sources_012` |
| Lethal food | `1` lethal source/unit in executed config |
| Lethal distance | `2..5` from hub in executed config |
| Death penalty | `1.0` |
| Pickup bonus | `0.5` |
| `max_steps` | `10000` |
| Critic | `strided_cnn` |
| `write_bits` | `4` |
| Wall layouts | `256` |
| Wall count | `5..9` |
| Wall length | `3..8` |
| Wall width | `2` |
| L-turn probability | `0.6` |
| Wall center window | `18` |
| PPO | `16 envs`, `256 steps`, `4 minibatches`, `1 epoch` |
| Global update cap | `80000` |
| Checkpoint video interval | `5000` updates |
| Rollout temperature | `0.5` |

Observed result:

```text
eval_mean_delivered_food = 12
eval_mean_delivered_fraction = 1.0
eval_mean_episode_return = 11
eval_mean_delivered_food_per_1000_ant_steps = 3.670155762287088
eval_success_rate = 1
```

Interpretation:

- This was the cleanest wall result: `12/12` deliveries.
- The policy actively routed around walls.
- The lethal source stayed mostly unexploited, but the setup was controlled and
  distribution-specific.
- The return being `11` while deliveries were `12` means the scalar return
  included a penalty term in that evaluation, so delivered fraction is the
  clearest success metric here.

## Later Sparse 60-Ant Random-Wall / Transfer Probe

This is a separate later probe from the 2-ant random-wall and near-nest proof.
It starts from a stabilized 60-ant non-lethal full-layout checkpoint.

Main artifacts:

- Static/historical config:
  `1975921:experiments/exploration_to_forage_proximity_sources_50x50_random_walls.json`
- Notebook:
  `notebooks/source_layouts/proximity_sources_50x50_random_walls.ipynb`
- Source checkpoint:
  `runs/notebooks/jerf_best_results/best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits_stabilized.pkl`
- Run directory:
  `runs/notebooks/fl50_60ants_sparse_far_food_near_lethal_random_walls_from_stabilized_best`
- Best checkpoint:
  `checkpoints/best_full_layout_60ants_sparse_far_food_near_lethal_random_walls.pkl`

Intended/static config highlights:

| Setting | Value |
| --- | --- |
| Arena | `50x50`, full layout |
| Hub center window | `24x24` |
| Actor radius | `2` |
| Ants | `60` |
| Identity types | `8`, assigned by ant index mod 8 |
| Safe food | `12` one-bite sources |
| Safe-food distance | Chebyshev distance `15` from hub |
| Lethal food | `1` source with `50` lethal units in the static config |
| Lethal distance | intended/current notebook summary `(2, 7)` |
| Random walls | enabled |
| Wall layouts | `256` |
| Wall count | `5..9` |
| Wall length | `3..8` |
| Wall width | `2` |
| L-turn probability | `0.6` |
| Wall center window | `18` |
| Critic | `strided_cnn` |
| `write_bits` | `8` shared write space |
| Write penalty | `0.0002`, decay `0.5` |
| Learning rate | `2.5e-05` |
| LR annealing | `true` |
| `clip_coef` | `0.05` |
| `ent_coef` | `0.0` |
| `max_grad_norm` | `0.25` |
| Training/eval temperature | `0.525` |
| PPO | `16 envs`, `256 steps`, `4 minibatches`, `1 epoch` |
| Eval episodes | `8` |
| Eval metric | `eval_mean_delivered_food_per_1000_ant_steps` |

Executed run caveat:

- The W&B runs for the `100` and `1000` update transfer probes used slightly
  different lethal settings than the later/static JSON.
- The `100` update run config used `lethal_food_count = 1`,
  `lethal_food_sources = 1`, distance `1..2`, and `death_penalty = 1`.
- The `1000` update run config used `lethal_food_count = 50`,
  `lethal_food_sources = 1`, distance `1..2`, and `death_penalty = 1`.
- The notebook/current-summary/static config later records the intended
  sparse-far-safe/near-lethal setup with distance `(2, 7)`.
- If asked, answer from the specific artifact being discussed: W&B run config
  for executed metrics, static JSON/notebook summary for intended handoff.

Observed transfer results:

| Run | Updates | Delivered | Fraction | Eval return | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Short transfer | `100` | `8.125` | `0.6771` | `7.125` | promising but not stable |
| Longer transfer | `1000` | `3.875` | `0.3229` | `-46.125` | degraded with fine-tuning |

Interpretation:

- The source policy had navigation competence.
- Lethal fine-tuning disrupted it instead of smoothly adding danger avoidance.
- More fine-tuning did not monotonically improve behavior.
- The likely future fix is a gentler schedule, for example gradually increasing
  `death_penalty` or lethal density rather than injecting the full hazard at
  once.

## Chebyshev Distance

Several configs say `cookie_distance` or lethal distance ranges. In this project
the source placement uses a grid-style distance band, effectively Chebyshev
distance:

```text
distance = max(abs(dx), abs(dy))
```

So distance `9` is a square ring around the hub, not a Euclidean circle. Distance
range `2..7` means a square annulus around the hub.

This matters because a source at `(hub_x + 7, hub_y)` and one at
`(hub_x + 7, hub_y + 7)` are both in the same Chebyshev band.

## Checkpoints, Videos, And Metrics

Common checkpoint flow:

- `load_model` or notebook `initial_checkpoint`: warm-start path.
- `save_best_model`: destination for the best eval-selected checkpoint.
- per-stage/latest checkpoint: saved at the end of a stage or at periodic update
  intervals.
- render-only checkpoint variables in notebooks may point to a best checkpoint
  without changing training.

Common evaluation settings:

- Best checkpoint selection was usually eval-based.
- Common best metric:
  `eval_mean_delivered_food_per_1000_ant_steps`.
- This metric normalizes for colony size and time, which is especially important
  when comparing `2` ants vs `60` ants.
- Delivered fraction is also useful when total food count is fixed.

Video cadence examples:

- Open lethal and early random/near-nest W&B notebooks often saved checkpoint
  videos every `5000` updates.
- The later static sparse 60-ant random-wall config records a denser intended
  checkpoint video interval of `500` updates.
- Final stage preview videos and periodic checkpoint videos are separate: a
  final preview shows the chosen rollout after a stage; checkpoint videos show
  training progression at intervals.

## Likely Questions And Short Answers

### What exactly was curriculum learning in shrink vision?

The map stayed `50x50`. The actor vision radius was reduced stage by stage. Each
smaller-vision stage warm-started from the previous larger-vision checkpoint.
Dense actors needed first-layer cropping; conv actors reused filters directly.

### Did the critic or actor see lethal cookies explicitly?

No explicit danger label was exposed. Internally the environment had
`food` and `lethal_food`, but observation merged them into `obs["food"]`. The
new visible clue was `dead_ants_count` after an ant died.

### Why add dead ants?

Without a danger label, the colony needs an observable consequence of bad
pickup. A dead body is a local social trace: future ants can learn that food near
that trace is risky.

### Did all ants have different networks?

No. They shared actor weights. In the 60-ant setup, the actor also received
identity-type features, so a shared network could condition behavior on a
repeating identity class. Shared weights do not imply identical actions, because
local observation, carrying state, facing, identity, and stochastic sampling can
differ.

### Why use random walls?

Open-map lethal behavior could be a local avoidance trick. Random walls tested
whether the policy could combine danger avoidance with navigation around
obstacles.

### Why did near-nest walls work better?

It constrained the distribution: fewer safe sources, stronger wall pressure
near the nest, and a checkpoint already exposed to random walls. That made the
navigation problem more learnable for that controlled setup.

### Why did the 60-ant transfer degrade?

The non-lethal source policy already knew a good navigation strategy, but lethal
fine-tuning changed the reward landscape abruptly. The policy drifted toward
avoidance/poor pickup behavior instead of adding a clean danger rule.

### Why was more vision not easier?

More vision increased the actor input dimension and did not remove sparse reward
or exploration difficulty. A full board in the input is not the same as a stable
strategy for pickup and return.

### Which result should I emphasize?

Emphasize the progression:

1. Shrink vision taught us that observation size alone was not the right
   curriculum.
2. Lethal cookies created a hidden-danger mechanism with corpse traces.
3. Random walls showed danger avoidance alone was not enough for robust
   navigation.
4. Near-nest walls produced a clean controlled success.
5. The 60-ant transfer showed that adding hazards to a strong non-lethal policy
   can destabilize it, suggesting gradual hazard curricula as future work.

## Artifact Map

Public report:

- `cool-antz/docs/index.html`

Old shrink-vision checkout:

- `/home/narf/Desktop/Facultad/RL/cool-antz-viejo/cool-antz/experiments/vision_shrink_curriculum.json`
- `/home/narf/Desktop/Facultad/RL/cool-antz-viejo/cool-antz/experiments/vision_shrink_conv_curriculum.json`
- `/home/narf/Desktop/Facultad/RL/cool-antz-viejo/cool-antz/experiments/vision_shrink_conv_autoresearch.json`
- `/home/narf/Desktop/Facultad/RL/cool-antz-viejo/cool-antz/notebooks/train_jax_vision_shrink_curriculum.ipynb`
- `/home/narf/Desktop/Facultad/RL/cool-antz-viejo/cool-antz/notebooks/train_jax_conv_vision_shrink_curriculum.ipynb`
- `/home/narf/Desktop/Facultad/RL/cool-antz-viejo/cool-antz/autoresearch/vision_shrink_conv_program.md`
- `/home/narf/Desktop/Facultad/RL/cool-antz-viejo/cool-antz/autoresearch/vision_shrink_conv_success.json`

Lethal/random-wall historical code and configs:

- `f75f7c8:experiments/exploration_to_forage_proximity_sources_50x50.json`
- `58ea666:src/ant_byte_env/jax_env.py`
- `58ea666:src/ant_byte_env/training/jax_mappo/core.py`
- `1975921:experiments/exploration_to_forage_proximity_sources_50x50_random_walls.json`

Current/local run artifacts:

- `cool-antz/runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_50x50_outer_30x30_inner`
- `cool-antz/runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_random_walls_50x50_outer_30x30_inner`
- `cool-antz/runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_near_nest_walls_50x50_outer_30x30_inner`
- `cool-antz/runs/notebooks/fl50_60ants_sparse_far_food_near_lethal_random_walls_from_stabilized_best`

Concrete W&B summary files used for headline metrics:

- `cool-antz/runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_50x50_outer_30x30_inner/wandb/run-20260628_015116-p2ovdyjk/files/wandb-summary.json`
- `cool-antz/runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_random_walls_50x50_outer_30x30_inner/wandb/run-20260701_223021-726mmgpr/files/wandb-summary.json`
- `cool-antz/runs/notebooks/exploration_to_forage_proximity_sources_lethal_cookies_near_nest_walls_50x50_outer_30x30_inner/wandb/run-20260702_102758-hn1g0xzv/files/wandb-summary.json`
- `cool-antz/runs/notebooks/fl50_60ants_sparse_far_food_near_lethal_random_walls_from_stabilized_best/wandb/run-20260704_122933-snraeot7/files/wandb-summary.json`
- `cool-antz/runs/notebooks/fl50_60ants_sparse_far_food_near_lethal_random_walls_from_stabilized_best/wandb/run-20260704_124604-uwxhx76s/files/wandb-summary.json`
