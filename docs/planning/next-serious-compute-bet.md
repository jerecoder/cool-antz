My main recommendation: **do not make the next serious compute bet another long scratch run with uniformly annealed source count.** Let the current run continue only if it is cheap, but evaluate its saved checkpoints now. For practical 50x50 success, the next run should preserve the final task structure earlier: **two macroscopic food sources from the beginning**, made easier by larger source footprint, shorter hub-source distances, or clustered food around two centers, then anneal those aids away.

The current curriculum probably teaches pickup and return mechanics, but it may not teach the final skill: **discover a rare remote source, establish/reuse a route, and keep exploiting it without a recurrent policy or explicit direction vector.**

A rough visibility calculation shows the issue. In a 50x50 active window, radius-2 vision covers at most 25 of 2500 cells, or about 1% of the window. If food sources are single positions, the chance that one ant’s local patch contains at least one source is approximately:

| Source count | One ant sees food in random patch | Four ants see food in random patches |
| -----------: | --------------------------------: | -----------------------------------: |
|          250 |                              ~92% |                                ~100% |
|          187 |                              ~85% |                                ~100% |
|           25 |                              ~22% |                                 ~63% |
|           10 |                              ~10% |                                 ~33% |
|            6 |                               ~6% |                                 ~21% |
|            2 |                               ~2% |                                  ~8% |

So the early stages are not just easier versions of the same problem. They are almost “food is everywhere” tasks. The final stage is a sparse discovery task. Curriculum learning can help sparse RL when intermediate tasks remain relevant and matched to the learner’s competence, but a curriculum can also fail when early tasks encourage policies that are not useful for the target task. That is the risk here. ([arXiv][1])

## Direct answers to your ten questions

### 1. Is smooth food-source annealing likely to solve the bridge?

**Possible, but I would not bet on it as currently defined.**

The curriculum is useful for teaching:

* movement,
* pickup,
* delivery,
* hub attraction while carrying,
* maybe basic trail writing.

But it weakly pressures the policy to learn:

* systematic outward search,
* scout dispersion,
* route retracing,
* source exploitation after rare discovery,
* causal byte use.

The most likely failure is a **phase transition** somewhere around 25, 16, 10, 8, or 6 sources, where pickup frequency collapses before the policy has learned a real exploration/trail strategy.

A better curriculum would keep the **number of semantic source locations fixed at two** and anneal something else:

* source footprint radius: large blob → small blob → single tile,
* hub-source distance: near → medium → full 50x50,
* clustered food around two centers: many food cells near two source centers → tighter clusters → two source tiles,
* optional food “halo” or local scent only during early stages, then remove.

That keeps the final problem’s structure intact: find one of two rare sources, then exploit it.

### 2. Keep training current run, or switch to staged gates?

**Keep it only as a diagnostic run. Switch the next real run to gated stages.**

Do not wait for the final source count before learning whether it failed. The current run should be treated as a probe that tells you **where the curriculum breaks**.

The next design should advance stages only after held-out competence is demonstrated. Automatic and competence-based curricula are commonly used in deep RL because fixed schedules can move on before the agent has acquired the necessary skill. ([arXiv][1])

A reasonable staged schedule:

| Stage | Task                                             | Gate before advancing                                            |
| ----- | ------------------------------------------------ | ---------------------------------------------------------------- |
| A     | 2 source clusters, large radius, medium distance | ≥80% episodes have pickup; ≥70% pickups become deliveries        |
| B     | 2 source clusters, smaller radius                | delivery per 1000 ant-steps does not collapse by more than ~30%  |
| C     | 2 source clusters, full 50x50 placement          | ≥60% episodes have at least one delivery                         |
| D     | 2 single-tile sources, high total food           | first-pickup rate and delivery-after-pickup both above baseline  |
| E     | final eval settings                              | normal policy beats no-byte-read and no-write if claiming memory |

The exact thresholds can move, but the gate should separately test **discovery**, **return**, and **repeated exploitation**. Mean reward alone will hide which subskill is broken.

### 3. Add a small decaying exploration or visit reward?

**Yes, but make it capped, team-level, non-carrying-only, and temporary.**

I would add an episodic first-visit bonus for active-window cells, but only while ants are **not carrying food**. Otherwise it competes with return-to-hub behavior.

A safe version:

```text
non-carrying team first-visit bonus:
    +0.0005 to +0.001 per newly visited active-window tile
    cap total novelty reward to 0.25–0.5 delivery-equivalents per episode
    decay toward zero during late sparse-source fine-tuning
    no novelty reward while carrying
    no novelty reward for repeatedly visiting the same tile
```

This keeps exploration useful without allowing “wander forever” to dominate delivery. Count-based and novelty-style exploration rewards are a standard way to attack sparse-reward exploration, but they need careful scaling because they can optimize coverage instead of task completion. ([NeurIPS Papers][2])

I would not add an uncapped visit reward. With no step penalty and positive-only rewards, uncapped novelty can easily produce policies that explore attractively but forage poorly.

### 4. Is `carrying_hub_distance_bonus=0.02` too weak?

**Probably yes for 50x50, especially without a hub vector or RNN.**

Earlier success with stronger `0.05`-style return shaping is important evidence. In the current setup, once a rare source is found, failing to return wastes the rarest event in the episode. The carrying return policy should be made extremely reliable before asking the agent to solve sparse discovery.

I would use `0.05` initially, but I would change the implementation if it currently rewards every closer step without penalizing farther steps. A positive-only “closer to hub” reward can be farmed by moving away and then moving closer again.

Better positive-only version:

```text
while carrying:
    reward only when the ant reaches a new best / minimum distance-to-hub
    since the current pickup
```

That gives a one-time reward for genuine progress and prevents oscillation farming.

The cleaner theoretical version is potential-based shaping, where reward depends on a signed potential difference rather than only positive progress. Potential-based shaping is specifically designed to accelerate learning while preserving the optimal policy under standard assumptions. ([People @ EECS Berkeley][3])

### 5. Is 250 total food too high for the final 2-source case?

**It is not necessarily too high for training, but it may be too high for judging success.**

With two sources and 250 total food, each source is effectively a rich mine. That is useful because once a source is discovered, repeated delivery gives strong learning signal. But it can also hide the discovery problem: a policy that finds one source occasionally and then exploits it heavily may look better than it really is.

I would evaluate at multiple total-food settings:

| Eval setting                   | What it tells you                                    |
| ------------------------------ | ---------------------------------------------------- |
| 2 sources, 250 total food      | Can the policy exploit a discovered rich source?     |
| 2 sources, 64 or 50 total food | Does behavior survive a less extreme source density? |
| 2 sources, 23 total food       | Does it solve the older rare-source target?          |

If success only appears at 250 food, the result is still useful, but the claim should be “rich-source 50x50 foraging,” not “general sparse rare-source foraging.”

### 6. Evaluate every saved source-count checkpoint?

**Yes. This is mandatory.**

You need a checkpoint-by-eval-source-count matrix. Do not only evaluate each checkpoint on the source count it trained on.

Minimum matrix:

```text
checkpoints:
    sources_250, 227, 206, 187, ...
    plus all later saved stages from the older 17-stage run

eval source counts:
    250, 100, 50, 25, 16, 10, 8, 6, 4, 3, 2
```

For each cell, record:

* mean delivered food per 1000 ant-steps,
* first-pickup rate,
* mean time to first pickup,
* pickup-to-delivery conversion rate,
* mean time from pickup to delivery,
* deliveries after first delivery,
* unique active-window coverage,
* byte occupancy/saturation,
* greedy movement result,
* sampled movement result.

The key diagnostic is whether the policy loses at **discovery**, **return**, or **exploitation**.

### 7. Warm start or scratch?

For **practical 50x50 success**, use a warm start.

Scratch is useful for a clean story, but practical success should not require scratch. Your own prior results already say shaped 25x25/50x50 policies learned useful behavior, while sparse 50x50 from scratch remained brittle. MAPPO is a strong cooperative MARL baseline, but on-policy methods can still be expensive in sparse long-horizon settings, so reusing a competent initialization is sensible. ([arXiv][4])

I would run:

```text
Run W: warm-started gated two-source curriculum
Run S: scratch gated two-source curriculum, same budget if affordable
```

If compute is limited, run W first and keep S as the scientific control later.

Warm-start caveat: if the best prior policy used hub/food vector observations, it may not transfer cleanly to the no-vector setting. In that case, either use a matching-observation checkpoint or treat the warm-start result as a practical engineering result, not as evidence that byte memory alone solved the task.

### 8. Mandatory ablations?

Yes. Minimum mandatory evals:

| Eval                            | Purpose                                                         |
| ------------------------------- | --------------------------------------------------------------- |
| greedy movement + greedy write  | deterministic deployment robustness                             |
| sampled movement + greedy write | practical deployment; avoids argmax collapse                    |
| movement temperature sweep      | find whether useful behavior exists in distribution             |
| normal bytes                    | main condition                                                  |
| no byte read                    | tests whether reading memory matters                            |
| no write                        | tests whether producing memory matters                          |
| held-out randomized layouts     | prevents memorized map claims                                   |
| source-count-specific eval      | finds transfer breakpoint                                       |
| total-food-specific eval        | separates rich-source exploitation from sparse-source discovery |

For sampled movement, I would sweep:

```text
temperature ∈ {0.5, 0.75, 1.0, 1.25}
write mode = greedy
```

Do not select the best checkpoint only using `greedy_move_greedy_write`. Your earlier evidence says greedy movement can collapse even when the action distribution contains useful routes. Select one checkpoint for practical deployment using sampled movement on validation, and separately report greedy.

For byte causality, require:

```text
normal > no_byte_read
normal > no_write
```

on held-out layouts, preferably with confidence intervals. If normal does not beat both, do not claim communication or memory.

### 9. Are per-ant byte channels the right bias?

**They are a good trail bias, but a weak communication bias.**

Per-ant channels make sense if the desired behavior is:

* each ant leaves its own breadcrumb trail,
* ants avoid or follow specific ants’ trails,
* byte overwrites are minimized.

But they are less natural if the desired behavior is colony-level communication such as:

* “food found here,”
* “path to hub,”
* “outbound trail,”
* “avoid explored region,”
* “source exhausted.”

A shared 4-bit symbol space is more expressive, but it creates harder coordination and overwrite problems. I would not spend major compute comparing byte semantics until you have a policy that forages at all. For now:

* keep per-ant channels for the practical run,
* evaluate no-byte-read and no-write,
* later compare against shared-symbol bytes once foraging works.

Also watch for **byte saturation**. Persistent non-decaying marks can turn the grid into a carpet of stale trails. If most visited tiles become permanently marked, local byte observations may stop carrying directional information.

### 10. What experiment next if compute is limited?

Run this:

## Next experiment: warm-started, gated, target-preserving two-source curriculum

Use the best available compatible checkpoint. If the best prior policy used extra vector observations, use it only if architecture compatibility is clean; otherwise warm-start from the closest non-vector shaped policy.

### Environment

Keep:

```text
active window: 50x50 inside 80x80 padding
ants: 4
vision radius: 2
feed-forward actor
centralized critic
write_while_moving: true
per_ant_write_channels: true
positive-only rewards
```

Change the curriculum from:

```text
uniform source count: 250 -> 2
```

to something like:

```text
two macro food sources throughout
```

with stages:

| Stage | Macro sources |               Food footprint | Hub-source distance |                    Total food |
| ----- | ------------: | ---------------------------: | ------------------: | ----------------------------: |
| 1     |             2 | radius 5 or clustered ~11x11 |        short/medium |                    128 or 250 |
| 2     |             2 |                     radius 3 |              medium |                    128 or 250 |
| 3     |             2 |                     radius 2 |          full 50x50 |                    128 or 250 |
| 4     |             2 |                     radius 1 |          full 50x50 |                    128 or 250 |
| 5     |             2 |                  single tile |          full 50x50 |                    128 or 250 |
| 6     |             2 |                  single tile |          full 50x50 | 23, 50, 64, and 250 eval only |

If the environment only supports source positions, approximate this by placing many food positions in two clusters around two centers, then shrink the cluster radius. The important part is that the policy always experiences the task as “there are two source regions,” not “food is uniformly everywhere.”

### Rewards

Use:

```text
delivery_reward = +1.0
pickup_bonus = +0.05
carrying_hub_progress_bonus = +0.05 initially
```

But make the hub progress reward non-repeatable:

```text
pay only for new best closeness to hub since pickup
```

Add exploration only for non-carrying ants:

```text
team first-visit active-window bonus = +0.0005 to +0.001
cap total novelty bonus at 0.25–0.5 per episode
decay during late stages
zero while carrying
```

This preserves your positive-only philosophy while reducing the chance of reward farming.

### Training progression

Do not advance on a fixed schedule. Advance when held-out validation passes:

```text
first-pickup rate >= 70–80%
pickup-to-delivery conversion >= 70%
mean delivered food per 1000 ant-steps stable or improving
sampled movement succeeds on held-out layouts
```

If greedy movement is a scientific requirement, add a separate greedy gate. If practical success is the priority, do not let greedy failure block all progress.

### Minimum evaluation for this run

For every stage-best checkpoint, evaluate:

```text
layouts: at least 64 held-out random layouts
movement:
    greedy_move_greedy_write
    sampled_move_greedy_write, temp in {0.5, 0.75, 1.0, 1.25}
byte modes:
    normal
    no_byte_read
    no_write
eval source settings:
    current stage
    final 2-source / 250-food
    final 2-source / 64-food
    final 2-source / 23-food
```

Metrics:

```text
delivered_food_per_1000_ant_steps
first_pickup_rate
time_to_first_pickup
pickup_to_delivery_rate
time_pickup_to_delivery
deliveries_after_first_delivery
unique_tiles_visited
byte_occupancy_fraction
normal - no_byte_read gap
normal - no_write gap
sampled - greedy gap
```

## Expected failure modes to look for

### Failure mode 1: discovery collapse

Symptoms:

```text
first_pickup_rate falls sharply at small source footprint or source_count <= 8
delivery conditional on pickup remains decent
```

Interpretation: return/exploitation exists, exploration does not.

Fixes:

* keep two macro sources but enlarge footprint longer,
* add capped non-carrying novelty,
* increase ants only as a later engineering option,
* use warm start,
* gate by first-pickup rate.

### Failure mode 2: return collapse

Symptoms:

```text
pickups occur
deliveries do not
pickup_to_delivery_rate low
ants wander while carrying
```

Interpretation: source discovery is not the bottleneck; hub relocalization is.

Fixes:

* stronger carrying progress bonus,
* non-repeatable hub-distance shaping,
* train explicit return stages,
* inspect whether bytes form usable home trails.

### Failure mode 3: exploitation collapse

Symptoms:

```text
first delivery occurs
few repeated deliveries
ants do not reuse source route
```

Interpretation: the policy can stumble into success but has not learned route memory or source exploitation.

Fixes:

* keep source dense after discovery,
* increase episode horizon if needed,
* make byte trails more stable,
* consider shared trail symbols later.

### Failure mode 4: byte non-causality

Symptoms:

```text
normal ≈ no_byte_read
normal ≈ no_write
```

Interpretation: bytes are decorative or ignored.

Fixes:

* do not claim communication,
* continue optimizing for foraging first,
* later test shared-symbol bytes or decaying pheromone-like channels.

### Failure mode 5: byte saturation

Symptoms:

```text
large fraction of active window marked
byte grid becomes nearly constant
ablation has little effect
```

Interpretation: persistent marks have become stale coverage memory, not trails.

Fixes:

* add decay if allowed,
* limit writes,
* reward byte sparsity only if needed,
* use event-triggered writes: near pickup, after pickup, near hub, or route turns.

### Failure mode 6: sampled-only success

Symptoms:

```text
sampled movement delivers
greedy movement fails
```

Interpretation: the policy distribution contains useful behavior, but argmax collapses to a bad mode.

Fixes:

* select practical checkpoint by sampled validation,
* report greedy separately,
* anneal training temperature only after sampled behavior is reliable,
* consider later distillation/fine-tuning for greedy deployment.

## Concrete decision rule

I would make the next decision in this order:

1. **Immediately evaluate current saved checkpoints across source counts.**
   This tells you where the present curriculum breaks.

2. **If current checkpoints already show nonzero transfer to 10, 6, or 4 sources**, continue the run and maybe add only evaluation tooling.

3. **If transfer collapses before 10 sources**, stop treating uniform source-count annealing as the main path.

4. **Launch the warm-started gated two-source-cluster curriculum.**

5. **Only after practical foraging works**, spend compute on the byte-representation question: per-ant channels vs shared symbols vs decay.

My strongest bet: the practical unlock is not “250 uniformly random sources smoothly down to 2.” It is **two-source semantics from the beginning**, made discoverable by footprint/distance/cluster aids, combined with **gated progression**, **stronger non-repeatable return shaping**, and **sampled-movement checkpoint selection**.

[1]: https://arxiv.org/abs/2003.04664?utm_source=chatgpt.com "Automatic Curriculum Learning For Deep RL: A Short Survey"
[2]: https://papers.neurips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning.pdf?utm_source=chatgpt.com "A Study of Count-Based Exploration for Deep ..."
[3]: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf?utm_source=chatgpt.com "Policy invariance under reward transformations"
[4]: https://arxiv.org/abs/2103.01955?utm_source=chatgpt.com "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"
