# Cool Antz Project Bible And History

This document is a review artifact for the report and site. It consolidates the
branch archaeology, experiment chronology, lost process notes, and claim
boundaries discovered during the integration cleanup.

It is intentionally more narrative than `docs/experiments.md` and more cautious
than the public report. Use it to decide what belongs in the final paper/site,
what should remain an appendix, and what needs more code or provenance cleanup
before being shown publicly.

Current reference point:

- Branch: `repo/research-integration-cleanup`
- Current integrated head during this review: `79e7f17`
- Important external branch heads:
  - `origin/report-writing-site@bd1f6ee`
  - `origin/main@a6ba235`
  - `origin/feat/multi-device-jax-mappo@83d54e7`
  - `origin/research/timed-release-roles@38cf3d8`
  - `origin/research/adversarial-marl-experiments@4aad30d`
  - `origin/lethal_cookies@58ea666`
  - `origin/vision_shrink_curriculum@b955a6f`
  - `autoresearch/map-ant-12x12-conv-critic@cc7c499`
  - `research/direct-goal-repro-sweep@5ccced7`

## Executive Story

The strongest story is not that byte communication was proven. The strongest
story is that a difficult sparse cooperative foraging task became understandable
only after the project learned to separate task semantics, metric semantics,
deployment modes, critic architecture, checkpoint transfer, ant coverage, and
communication hypotheses.

The project starts with a deceptively simple objective: local ants must collect
food and return it to a hub. In practice, the hard part is not only finding food.
It is repeated delivery, credit assignment over long loops, stochastic versus
greedy policy consolidation, and deciding whether written byte marks are useful
communication or just another correlated behavior.

The clean report narrative should be:

1. The environment contract changed from loose movement/collection behavior to
   delivery-centered foraging.
2. Sparse direct training failed as a final-task baseline.
3. Small curricula and map-ant workflows showed real competence but exposed
   generalization and consolidation limits.
4. A 25x25 distance-shaped line solved the task under sampled movement, while
   greedy deployment remained weak.
5. Byte memory became a serious hypothesis, but ablations and no-write results
   prevented a causal communication claim.
6. 50x50 progress depended on source geometry, checkpoint selection, deployment
   temperature, and especially spatial centralized critics.
7. The 60-ant 50x50 result is the strongest frontier behavior, but it is bundled
   with many interventions and saturated writes.
8. 100x100, 250x250, and 1000x1000 artifacts are best framed as continuation,
   diagnostics, and actor-only stress tests.
9. Timed-release, adversarial, lethal-cookie, maze, vision-shrink, and 12x12
   conv-critic lines are valuable side lanes, mostly probes or negative evidence.

The deeper lesson is that the failures are part of the contribution. They show
what the system did not prove.

## Evidence Rules

These rules should govern report and site wording.

- Use `env_return`, delivery counts, delivered fraction, and success rate for
  paper-facing task performance.
- Treat `episode_return` as diagnostic when shaping, write costs, pickup bonuses,
  distance bonuses, view/visit rewards, or byte rewards are enabled.
- Never mix final checkpoint, best checkpoint, selected checkpoint, and held-out
  selected checkpoint metrics without naming the distinction.
- Always state the denominator: `23/23`, `48`, `125`, `250`, `375`, or `5000`
  food are different tasks.
- Always state action mode when it matters. Several strong results depend on
  `sampled_move_greedy_write`; greedy movement can be much worse.
- Treat byte writes as a substrate or behavior until matched no-write,
  no-byte-read, write-cost, byte-shuffle, or protected-write controls establish
  causality.
- Treat `food_sources` as task geometry, not metadata. Changing it changes
  source density, discovery, repetition, and route structure.
- Treat multi-device training as infrastructure unless the rollout shape or
  batch size is itself the experimental treatment.
- Treat videos and browser sandbox demos as qualitative evidence unless paired
  with an evaluation manifest.

## Terminology

- `food_count`: total deliverable bites in the episode.
- `food_sources`: number of source locations. This changes the task.
- `env_return`: raw environment reward, usually delivery-facing.
- `episode_return`: rollout training reward after shaping.
- `pickup`: ant removes food from a source.
- `delivery`: carrying ant reaches the hub and increments delivered food.
- `write_bits`: number of bits or values available in the persistent byte grid.
- `shared writes`: ants write into the same public byte grid.
- `per_ant_write_channels`: ant `i` writes only a channel such as
  `i % write_bits`, while all ants can still observe the grid.
- `sampled_move_greedy_write`: sample movement from policy distribution, choose
  write action greedily. This became the most meaningful deployment mode for
  several learned policies.

## Branch Lineage Map

### Cleanup Base

The cleanup branch reorganized JAX MAPPO into smaller modules while preserving
experiment surfaces:

- `src/ant_byte_env/training/jax_mappo/`
- `experiments/*.json`
- `notebooks/*`
- `docs/experiments.md`
- `docs/experiment_rationale.md`

The integration principle was selective porting, not a wholesale merge from
`report-writing-site`, because that branch was based on `origin/main` and did
not contain the cleanup branch's modular JAX MAPPO split.

### Report Site And Large Scale

Source lineage:

- `origin/report-writing-site@bd1f6ee`
- Branch base around `554ee8d`

Important progression:

- `e59a0fb` to `97ccf63`: 4-bit shared-write write-cost continuation and strict
  write-bit penalty.
- `a6ba235`: 8-bit shared-write continuation.
- `80b902d` to `c4851c7`: cleanup loops, 100x100 sweeps, temperature grids, and
  progress-video machinery.
- `c1a0741`, `10b09d0`, `f776de5`: 250x250 truncation and continuation configs.
- `a203c2d` to `5b27fd2`: big-map actor-only renderer, replay state sidecars,
  layout overrides, and palette rerender helper.
- `a4ed53c`: report evidence bundle: `report/`, generated figures, logs, 60-ant
  configs and notebook surfaces.
- `034b752`: `docs/planning/complete-experiment-chronology.md`.
- `c82519e` to `c0d7c9a`: report website and actual 50x50 frontier policy
  sandbox export.
- `2634925`: Spanish site rewrite.
- `bd1f6ee`: missing side branches added to report/site narrative.

Important current gap:

- Raw branch logs under `logs/100x100_*.log` and `logs/hs250_*.log` were not
  imported into current HEAD. The polished site, report CSVs, and figures
  survived, but the intermediate provenance trail is thinner.

### Main And Multi-Device

Source lineages:

- `origin/main@a6ba235`
- `origin/feat/multi-device-jax-mappo@83d54e7`

Important progression:

- `e59a0fb`: 4-bit shared-write write-cost notebook/config.
- `26e5968`, `97ccf63`: stricter write penalty.
- `a6ba235`: 8-bit continuation and channel-grid renderer updates.
- `80a330b`, `7e29845`: JAX data-parallel helpers and `pmap` trainer mode.
- `bb6d481`: 16-ant 8-type 4-device config.
- `a634252`, `02c71f0`, `2c17ba9`: channel-grid checkpoint renderer, bit
  labels, centered no-food rollout mode.
- `d68e9c3` to `83d54e7`: write-cost multiplier sweep through x150.

Important nuance:

- The 8-bit continuation is not an isolated "more bits helped" experiment.
  It changed `write_bits` from 4 to 8 and also changed write-cost scale. Dense
  all-bit episode cost moved from about `300` under strict 4-bit cost to about
  `6.375` in the 8-bit continuation.
- The write-cost sweep is a continuation lineage, not a clean factorial
  communication ablation.
- Multi-device is infrastructure unless a report claim specifically discusses
  batch/rollout shape.

### Direct Goal And Map-Ant Autoresearch

Source lineages:

- `research/direct-goal-repro-sweep@5ccced7`
- `autoresearch/map-ant-12x12-conv-critic@cc7c499`

Important progression:

- `bc45f99`: notebook-first direct-goal reward-shaping workflow.
- `52a70e8` to `afc9835`: reproducible direct-goal sparse sweep, next-step
  docs, checkpoint evaluation, W&B tracking, and multiseed confirmation.
- `5ccced7`: JAX MAPPO eval/render controls, split-head action modes,
  best-checkpoint selection, rollout temperature, and reproducibility hooks.
- `acdcdcf`: reframed the handoff around final direct-goal baseline and map-ant
  curriculum; reinforced that writing is part of the task.
- `2b12bad`: 12x12 conv-critic autoresearch loop.
- `effbfb5`, `edb66e6`, `482c745`: shorter gate cadence and CPU-friendly
  throughput.
- `cc7c499`: mutation chain reached `M030`.

Important current gap:

- Current HEAD retains some docs and scripts, but the branch-only matrices and
  CLI subcommands for old direct-goal and map-ant autoresearch are not fully
  present.
- `scripts/run_direct_goal_sparse_sweep.sh` and `scripts/run_map_ant_12x12_sweep.sh`
  should be treated carefully until their dependencies are verified.

### Timed Release

Source lineage:

- `origin/research/timed-release-roles@38cf3d8`

Profiles:

| Commit | Purpose | Shape | Critic | Transfer | Temperature |
| --- | --- | --- | --- | --- | --- |
| `46aa332` | First V1 timed release | `16 x 128`, `40,960,000` timesteps | `strided_cnn` | full load | eval `0.75` |
| `8d7b582` | Local CPU speed probe | `12 x 96`, `576,000` timesteps | `mlp` | actor-only warm start | eval/render `0.52` |
| `38cf3d8` | L4 continuation | `128 x 256`, `16,384,000` base timesteps plus continuation | `strided_cnn` | full actor+critic+optimizer | train `0.75`, eval/render `0.52` |

Important current gap:

- Current config matches the L4 profile.
- `docs/research_integration_plan.md` still contains stale language about the
  local actor-only MLP profile being final.
- Branch notebooks contained exact low-delivery chunk/eval outputs, including
  values such as `eval_mean_delivered_food=11.25` and
  `eval_mean_delivered_fraction=0.09`, that are not preserved as a first-class
  evidence table.

### Adversarial Frozen Opponent

Source lineage:

- `origin/research/adversarial-marl-experiments@4aad30d`

Important progression:

- Planned adversarial MARL as a separate lane.
- Implemented frozen-opponent two-team MAPPO.
- Added curriculum resets, hub-center fade, critic warmup, KL behavior anchor,
  eval-gated training, guarded refinement notebooks, side-swap diagnostics, and
  role renders.

Important framing:

- This is frozen-opponent capability/refinement, not self-play.
- The adversarial critic is intentionally an MLP and is not the cooperative
  source critic, because the value target is adversarial learner advantage.
- Fixed-center diagnostics are much stronger than randomized target metrics, so
  layout sensitivity must be visible in the story.

### Lethal Cookies And Maze

Source lineage:

- `origin/lethal_cookies@58ea666`

Intent:

- Lethal food looks like normal food to the actor through the normal food
  channel.
- Lethal pickup kills the ant and creates death diagnostics.
- Same-distance safe/lethal source sampling tests whether the policy can handle
  deceptive but geometry-matched sources.
- Maze notebooks tested geometry, rendering, and checkpoint rollout surfaces.

Important current risk:

- Current `jax_env.py` tracks `dead_ants_count`.
- Current `build_actor_observations` does not appear to include a dead-ant local
  actor plane.
- Branch transfer code supported `transfer_source_args` and dead-ant actor/critic
  channel adaptation. Current modular `transfer.py` does not appear to support
  `transfer_source_args`.
- Therefore, the lethal-cookie docs/config notes may overstate actor-observation
  support in current HEAD. Treat this as a code/doc cleanup item before training
  from that config.

### Vision Shrink

Source lineage:

- `origin/vision_shrink_curriculum@b955a6f`

Original objective:

- Shrink actor vision from large local windows toward smaller windows while
  keeping the map fixed at 50x50.
- The branch had a Codex autoresearch program that allowed only JSON mutation
  and notebook execution for scoring.

Important branch-only evidence:

- Final branch state moved to one `21x21` stage:
  - `vision_radii=[10]`
  - `global_update_cap=5`
  - `return_progress_bonus=0.1`
  - `carrying_step_penalty=0.01`
- The executed run was weak/negative:
  - `21x21: 5/5 updates`
  - `ret=-289.670`
  - `ret_avg=-78.295`

Important current gap:

- Current HEAD preserves vision-shrink configs as exploratory/deferred, but not
  all branch-only trainer knobs or the exact negative notebook output.

## Core Experiment Chronology

### 1. Environment Contract Reset

The project became clearer after the task was defined as delivery-centered
foraging:

- ants must carry food back to the hub;
- food sources can have multiple bites;
- `food_count` is total food, not source count;
- `food_sources` changes geometry and discovery;
- actor observations are local, while critic observations are centralized.

This matters because many earlier "activity" signals were not evidence of
solving the task.

### 2. Sparse Direct-Goal Baseline

Historical direct-goal evidence used:

- `50x50`
- `10` ants
- `48` food
- `12` food sources
- `5` write bits
- sparse delivery reward

The sparse sweep found weak stochastic foraging but no deterministic policy:

- Best noted sparse candidate: `S4`
- Deterministic delivered food: `0.0 / 48`
- Sampled delivered food: `6.75 / 48`
- Success rate: `0.0`

Interpretation:

- Not "nothing learned."
- The policy distribution sometimes found food/delivery.
- Argmax behavior did not consolidate into a useful final-task policy.

Current integration warning:

- Current `experiments/direct_goal_baseline.json` has `food_sources=25`, not the
  historical `12` used by `5ccced7`.
- Report tables should separate "historical sparse sweep baseline" from the
  current maintained baseline config.

### 3. Single-Ant And Autocurriculum Failure

Old single-ant/autocurriculum evidence showed scaling collapse:

- `4x4`: return around `12.281`
- `25x25`: return around `0.719`
- `50x50`: return around `0.156`

Interpretation:

- Growing the board did not automatically produce reusable foraging.
- Sparse credit assignment and exploration dominated.

### 4. Historical MLP Map-Ant Curriculum

The older map-ant gated MLP workflow remains important historical evidence:

- Reached `20x20_5_ants`.
- Plateaued around `25x25_6_ants`.
- Did not solve `50x50`.

Frame it as:

- real mid-scale progress;
- strict gates and useful workflow evidence;
- generalization/placement brittleness.

Do not let the newer failed 12x12 conv-critic branch erase this older progress.

### 5. 12x12 Conv-Critic Autoresearch

The newer conv-critic/autoresearch branch tested whether a more structured JAX
MAPPO critic could master an honest randomized map-ant curriculum from `4x4`
through `12x12`.

Constraints:

- actor vision radius `1`;
- randomized food and hub;
- padded `50x50` critic observation;
- `write_bits=1`;
- `write_while_moving`;
- deterministic and sampled gates;
- no zero-write ablations for this communication experiment.

Important missing mutation ledger:

- `M021`: write-band anti-spam.
- `M022`: selective lower-entropy writing.
- `M023` and `M024`: greedy movement mixes.
- `M025` to `M027`: carried-write trail shaping.
- `M028`: multi-ant-only write floor.
- `M029`: urgency sharpening.
- `M030`: balanced consolidation from `M029` attempt 16.

Important negative findings:

- A pickup bug was ruled out. A deterministic render trace at `20x20_5_ants`
  had `47` visible-food ant-steps, `18` targeted food-cell steps, `18` pickups,
  and `0` cases of stepping onto food without pickup.
- The failure was policy action selection, not pickup mechanics.
- `attempt_091` showed the 25x25 plateau was seed-sensitive:
  - deterministic delivered `0.9647`;
  - deterministic success `0.8125`;
  - sampled delivered `0.9348`;
  - sampled success `0.7500`;
  - seed changes could move the same checkpoint from `23/23` to `0/23`.
- `B007` delivered `1.0` deterministic and sampled on 12x12, but write rates
  remained high.
- `A010` was a valid bounded-write 12x12 endpoint:
  - deterministic delivered `0.9813`;
  - sampled delivered `1.0`;
  - sampled write rate `0.4832`;
  - visual read still showed broad wedge/staircase writing.
- `F031` was a fresh anti-stair near-pass:
  - solved `9/16` sampled episodes;
  - delivered `0.8937`;
  - success `0.562`;
  - pickup-to-delivery `0.9499`;
  - length `462.8`.

Interpretation:

- The branch is not simply "failed."
- It is a disciplined negative-search ledger around communication, marker spam,
  silence, and deterministic consolidation.
- It belongs in the report/site as evidence of boundary-finding.

### 6. 25x25 Distance-Shaped Unlock

The first strong positive result was the 25x25 distance-shaped line:

- `DISTANCE_CAP4` under sampled movement: `23/23`, success `1.0`.
- Greedy movement for the same line: `2.75/23`, success `0.0`.
- `DISTANCE_CAP4_NO_WRITE` still reached sampled `22/23`.

Interpretation:

- 25x25 was solved under sampled movement, not greedy argmax.
- Distance shaping and ant coverage mattered.
- No-write performance blocks a simple causal communication claim.

Safe wording:

> The 25x25 delivery task was solved under sampled movement in the distance-shaped
> 4-ant setting, while greedy deployment and communication causality remained
> unresolved.

### 7. Communication Bits, Memory, And Write Cost

Evidence points:

- Early bit-count sweeps moved much less than ant-count/coverage changes.
- R8/R11 memory probes performed as well or better with no-byte/no-write
  variants.
- R9 created a gap but damaged foraging.
- R12 had no final probe.

Write-cost and 8-bit nuances:

- The 4-bit strict write-cost line made dense all-bit writing more expensive
  than the full delivery budget.
- The 8-bit continuation changed both alphabet size and write-cost scale.
- Therefore the 8-bit line is not clean evidence that more bits caused the
  improvement.

Safe wording:

> Policies trained with byte memory and shared writes can forage well, and some
> policies write heavily, but the current evidence does not prove that byte
> communication caused the solved behaviors.

### 8. Rare 50x50 And Source Geometry

Rare-source 50x50 exposed discovery as a bottleneck:

- baseline around `10.958/23`;
- hub vector around `12.328/23`;
- hub plus nearest-food vector around `15.203/23`;
- best held-out selected branch around `16.198/23`;
- another archival read reports around `18.313/23`, fraction `0.796`, success
  `0.250` for a hub plus nearest-food vector line.

Interpretation:

- Navigation aids helped discovery.
- Rare 50x50 was not solved.
- This line should not be merged into the 125-food full-layout frontier without
  naming the changed denominator.

### 9. 50x50 Spatial Critic Frontier

50x50 progress crossed an important inference boundary when the centralized
critic architecture changed.

Important numbers:

- MLP frontier row: `21.5/48`.
- 8-ant strided-CNN write-cost row: `69.375/125`, delivered fraction `0.555`,
  success `0.125`, write nonzero around `0.943`.
- 60-ant confirmed frontier: `123.90625/125`, delivered fraction `0.99125`,
  success `0.90625`, write nonzero around `0.998`.

Interpretation:

- The 60-ant result is strong.
- It is not a clean causal communication result.
- It bundles ant count, 8 bits, identity features, selected continuation,
  spatial critic, temperature selection, source geometry, and saturated writes.

Safe wording:

> The 60-ant 50x50 frontier nearly emptied the 125-food task in evaluation, but
> the mechanism remains bundled and should be framed as frontier engineering
> rather than proof of a byte language.

### 10. 100x100 Bridge

The 100x100 line is continuation and selection, not from-scratch training.

Branch logs show intermediate candidates:

- Easy 125-food, 2-source bridge candidates reached delivered fractions around
  `0.770` to `0.866` with success `0.438`.
- Mid 250-food, 4-source bridge candidates reached fractions around `0.875` and
  `0.902`, with success up to `0.625`.
- Continuation hard375 line reached site/report claims around `372/375` to
  `373/375`, success `0.625` to `0.75`, and rate around `0.803` deliveries per
  1000 ant-steps.

Important current gap:

- The current site/report has polished claims and media.
- Raw branch logs are not in HEAD.

Safe wording:

> The 100x100 bridge demonstrates selected continuation and deployment tuning,
> not a fresh-from-scratch solution.

### 11. 250x250 Diagnostics

250x250 is the clearest warning that shaped return and byte activity can mislead.

Key evidence:

- Source-teacher diagnostic:
  - `delivery_events=0`;
  - `pickup_events=0`;
  - shaped return around `26.25`;
  - byte fraction around `0.899`.
- Reset-boundary recovery:
  - final `654` deliveries;
  - `869` pickups;
  - best train window around `1003` deliveries.
- Later long d12 eval:
  - mean deliveries `0.392`;
  - median `0`;
  - pickup-to-delivery around `0.008`.

Interpretation:

- Reset-boundary recovered local delivery.
- General 250x250 autocurriculum was not solved.
- Delivery metrics must stay separate from shaping and bytes.

### 12. 1000x1000 Actor-Only Stress Test

The 1000x1000 media is actor-only deployment, not end-to-end training.

Site evidence:

- source policy: stabilized 50x50 frontier actor;
- render grid: 1000x1000;
- 500 ants and 5000 food in the visualization;
- first delivery at step `14574`;
- `147` deliveries by step `120000`.

Safe wording:

> The 1000x1000 artifact is a stress-test visualization of local actor behavior,
> not a trained 1000x1000 policy.

### 13. Timed-Release Roles

Research question:

- Does the 8-ant cooperative shared-write policy contain reusable role structure
  when ant ranks are released one at a time?

Mechanics:

- rank 0 active at step 0;
- one additional rank released every 150 steps;
- rank identity is the existing actor identity one-hot;
- inactive ants are hidden/masked and forced to no-op;
- actor loss masks inactive agents.

Final intended reportable profile:

- `num_envs=128`;
- `num_steps=256`;
- `total_timesteps=16,384,000`;
- `critic_architecture=strided_cnn`;
- `actor_only_warm_start=false`;
- training movement temperature `0.75`;
- eval/render movement temperature `0.52`.

Safe wording:

> Timed release is a role-specialization probe. It does not prove roles emerged
> until compared against all-active, randomized-release, zero-write, no-byte-read,
> and greedy controls.

### 14. Adversarial Frozen Opponent

Research question:

- Can a warm-started cooperative actor adapt to a two-team zero-sum delivery
  objective against a frozen cooperative opponent?

Important semantics:

- two teams;
- shared food and bytes;
- two hubs;
- team-specific delivery;
- rewards are delivery deltas;
- opponent is frozen;
- actor warm-start is copied;
- adversarial critic is fresh MLP because the value target differs.

Evidence:

- Randomized target eval:
  - learner-vs-frozen delivery diff `13.4375`;
  - side-swap-adjusted diff `9.5`;
  - win rate `0.625`.
- Fixed-center eval:
  - diff `56.46875`;
  - adjusted `40.9375`;
  - win rate `1.0`.

Interpretation:

- This is real frozen-opponent exploitation/capability evidence.
- It is layout-sensitive.
- It is not self-play or broad adversarial competence.

### 15. Lethal Cookies And Maze

Status:

- Pipeline and geometry evidence.
- No comparable final delivery/success metrics should be claimed.

Important risk:

- The lethal-cookie config notes say dead ants provide the actor a new clue.
- Current observation code should be checked before claiming that.
- Current Python renderer rejects lethal-food checkpoints.

Safe wording:

> Lethal-cookie and maze work show that the environment, geometry, reset, and
> metric surfaces exist; learned-performance claims require recovered or rerun
> comparable metrics.

### 16. Vision Shrink

Status:

- Exploratory/deferred.
- Branch had actual trainer/config changes and a negative short run.

Safe wording:

> Vision shrink remains an exploratory attempt to reduce actor visual range. The
> recovered evidence is weak/negative, not a successful curriculum.

## Strongest Quantitative Evidence

| Family | Strong number | What it supports | Caveat |
| --- | ---: | --- | --- |
| Direct sparse historical baseline | deterministic `0/48`, sampled `6.75/48` | sparse final-task training did not consolidate | historical config used 12 sources |
| Single-ant/autocurriculum | `12.281 -> 0.719 -> 0.156` | naive scale-up failed | not all curricula ruled out |
| 25x25 distance | sampled `23/23`, success `1.0` | sampled deployment solved 25x25 | greedy `2.75/23`; no-write still strong |
| 25x25 no-write | sampled `22/23` | byte causality not proven | matched details matter |
| Rare 50x50 | about `16.2/23` to `18.3/23` | navigation aids helped discovery | not solved, different denominator |
| 8-ant 50x50 write-cost | `69.375/125` | spatial critic/write-cost lineage improved | bundled variables |
| 60-ant 50x50 | `123.90625/125`, success `0.90625` | strongest frontier behavior | saturated writes, many confounds |
| 100x100 hard bridge | `372/375` to `373/375` | selected continuation scales | not from scratch |
| 250x250 reset-boundary | final `654`, best around `1003` | local recovery after reset change | later d12 eval collapsed |
| 1000x1000 actor-only | `147` by step `120000` | behavior transfers visually | not trained there |
| Adversarial fixed-center | win rate `1.0` | frozen-opponent exploit in fixed layout | randomized eval modest |
| Timed release chunk output | `11.25/125`, fraction `0.09` | mechanics/eval path existed | not role evidence |
| Vision shrink | `ret=-289.670` | weak/negative short run | not final proof |

## Lost Or Under-Documented Material

These are the highest-value pieces to preserve or summarize before final report
writing.

### Direct Goal

Branch-only or compressed material:

- `autoresearch/direct_goal_sweep.json`
- `autoresearch/ideas/021-direct-goal-next-steps.md`
- flat notebook versions:
  - `notebooks/train_jax_direct_goal_baseline.ipynb`
  - `notebooks/train_jax_direct_goal_reward_shaping.ipynb`
- CLI support for `direct-goal-plan`, `direct-goal-run`, `direct-goal-rank`

What is worth preserving:

- The sparse S0-S5 hypothesis matrix.
- S4 stochastic-only result.
- Multiseed promotion logic.
- Reward-shaping screen plan R0-R7.
- Behavior diagnostics: pickup-to-delivery, carrying steps, write rates,
  ant spread, movement/write histograms.

### Map-Ant 12x12

Branch-only or compressed material:

- `autoresearch/map_ant_12x12_sweep.json`
- `src/ant_byte_env/training/jax_mappo/map_ant_curriculum.py`
- `map-ant-plan`, `map-ant-run`, `map-ant-rank` CLI support
- M021-M030 mutation chain

What is worth preserving:

- The full mutation ledger.
- The pickup-bug ruled-out trace.
- `attempt_091` seed sensitivity.
- `B007`, `A010`, `F031` near-pass evidence.
- Manual review rule: a gate pass is not a scientific success if the video is
  marker spam, silence, or a broad staircase.

### Communication Autoresearch

Branch-only or compressed material:

- `autoresearch/communication_sweep.json`
- `autoresearch/communication_consolidation_sweep.json`
- `autoresearch/ideas/000-020.md`
- `autoresearch/protocol.md`

What is worth preserving:

- The research process:
  - observe;
  - hypothesize;
  - mutate one dominant variable;
  - measure with exact artifacts;
  - decide keep/mutate/drop/blocked.
- The fact that the project repeatedly tried to distinguish write diversity,
  write spam, real delivery, and deterministic consolidation.

### Report Site Logs

Branch-only or compressed material:

- `logs/100x100_bridge_sweep_*.log`
- `logs/100x100_continuation_sweep_*.log`
- `logs/100x100_temperature_grid_*.log`
- `logs/100x100_progress_videos.log`
- `logs/hs250_*.log`

What is worth preserving:

- Intermediate 100x100 candidate results.
- Temperature grid values.
- W&B run IDs and artifact references.
- 250x250 W&B summaries.

### Timed Release

Compressed material:

- exact notebook chunk outputs;
- per-rank role metrics;
- low-delivery early chunk evidence.

What is worth preserving:

- Profile table: initial, local CPU, L4.
- Exact statement that CPU MLP profile was a speed probe, not the final report
  config.

### Vision Shrink

Compressed material:

- `autoresearch/vision_range_program.md`
- negative `21x21` run output
- branch-only reward/architecture knobs

What is worth preserving:

- The exact negative result.
- Why it was deferred instead of reported as success.

## Current HEAD Mismatches And Cleanup Risks

These should be checked before final report/site publication.

| Area | Risk | Why it matters |
| --- | --- | --- |
| Direct-goal source count | historical evidence used `food_sources=12`; current config uses `25` | report tables can mix two baselines |
| Direct-goal scripts | `run_direct_goal_sparse_sweep.sh` may reference missing branch CLI/matrix | runnable docs may fail |
| Map-ant 12x12 scripts | `run_map_ant_12x12_sweep.sh` may reference missing branch CLI/matrix | runnable docs may fail |
| Map-ant 12x12 doc | current prose stops at `M021`; branch reached `M030` | negative-search history is incomplete |
| Timed-release plan doc | stale actor-only MLP language | conflicts with current L4 config |
| Lethal-cookie actor clue | config says dead-ant actor plane exists; current actor observations may omit it | training claims may be wrong |
| Lethal transfer | branch supported `transfer_source_args`; current modular transfer may not | old maze rollout notebook behavior may not reproduce |
| Vision shrink | branch trainer knobs and negative output not preserved | status looks vaguer than evidence |
| 100x100/250x250 logs | raw logs not imported | provenance weaker for site claims |
| README | older wording may imply fully observable env | conflicts with local actor story |

## Report-Safe Claims

Use these statements confidently.

- The final task is delivery-centered cooperative foraging, not simple food
  discovery.
- Sparse direct training did not produce a reliable deterministic final-task
  policy.
- 25x25 was solved under sampled movement in the distance-shaped line.
- Greedy argmax deployment was a major failure mode in some otherwise strong
  policies.
- Byte memory is available and policies write to it, but current evidence does
  not prove causal emergent communication.
- Spatial centralized critics were a major boundary in 50x50 progress.
- The 60-ant 50x50 frontier is strong but bundled.
- 250x250 shows why shaped return and byte activity must be separated from raw
  delivery.
- 100x100 is selected continuation and deployment tuning.
- 1000x1000 is actor-only stress testing.
- Adversarial work is frozen-opponent capability evidence, not self-play.
- Timed-release is a role-specialization probe, not role proof.
- Maze/lethal/vision work is pipeline, deferred, or negative evidence unless
  comparable metrics are recovered.

## Claims To Avoid

Avoid these unless new matched evidence is added.

- "The ants learned a causal byte language."
- "More bits solved the task."
- "50x50 is solved" without naming the 60-ant selected-continuation setup.
- "250x250 is solved."
- "1000x1000 was trained."
- "Adversarial competence emerged" as a broad claim.
- "Timed release produced roles."
- "Maze/lethal-cookie policies worked."
- "The conv critic fixed map-ant."

## Suggested Report/Site Structure

1. Problem and environment contract.
2. Metrics and evidence rules.
3. Sparse direct-goal baseline.
4. Historical curricula and map-ant progress.
5. 25x25 sampled-movement unlock.
6. Communication hypothesis and no-causal-proof evidence.
7. Rare 50x50 and source geometry.
8. Spatial critic and 50x50 frontier.
9. 60-ant result with confound box.
10. Large-scale diagnostics: 100x100, 250x250, 1000x1000.
11. Side lanes:
    - timed release;
    - adversarial frozen opponent;
    - map-ant 12x12 conv-critic negative search;
    - maze/lethal cookies;
    - vision shrink.
12. Limitations and next experiments.
13. Provenance appendix.

## Provenance Appendix Template

Every headline claim should eventually have a row like this.

| Claim | Config | Checkpoint | Eval artifact | Action mode | Temperature | Denominator | Status |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| 60-ant 50x50 frontier | `experiments/exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_stabilize_from_60best.json` | `runs/notebooks/fl50_60ants_half_food_sw_wc_8bits_stabilize_from_60best/checkpoints/best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits_stabilized.pkl` | confirmation eval / `report/data/figure_critic_50x50.csv` | `sampled_move_greedy_write` | around `0.5` to `0.525` depending render/eval | `125` | strong but bundled |

For media, include:

- video path;
- poster path if used;
- renderer script;
- checkpoint;
- seed offset;
- action mode;
- movement temperature;
- whether the media is evaluation, continuation render, or actor-only stress
  test.

## Next Documentation Tasks

1. Decide whether this bible should stay as an internal review doc or become a
   public appendix.
2. Fix direct-goal baseline wording so historical `12`-source evidence is not
   mixed with current `25`-source config.
3. Update `docs/map_ant_12x12_conv_critic_autoresearch.md` with M021-M030 and
   the strongest negative evidence.
4. Add a provenance manifest for site media and headline metrics.
5. Decide whether to import branch raw logs or summarize them into committed
   CSV/JSON ledgers.
6. Resolve or clearly document stranded old autoresearch scripts.
7. Resolve lethal-cookie dead-ant observation and transfer-source behavior before
   treating that config as runnable research evidence.
8. Preserve the vision-shrink negative output in a short appendix.
9. Add timed-release profile table to docs and remove stale final-profile wording.
10. Add fixed-center versus randomized adversarial metrics to the site/report
    side-lane summary.

## One-Sentence Thesis

Cool Antz produced strong cooperative foraging behavior and a careful evidence
stack, but its most important scientific result is the separation of useful
multi-agent behavior from unproven causal byte communication.
