# Map-Ant Curriculum Autoresearch Log

This file tracks resumable curriculum experiments that are too stateful to leave only
in chat. Runtime artifacts live under
`runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/`.

## Current State

- Last passed stage: `20x20_5_ants`
- Last passed checkpoint:
  `runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/checkpoints/20x20_5_ants/attempt_007.pkl`
- Current blocked stage: `25x25_6_ants`
- Best reliable parent for retries:
  `runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/checkpoints/25x25_6_ants/attempt_091.pkl`
- Current best gate score on `25x25_6_ants`: `0.875`
- Common near-pass failure:
  `sampled.eval_mean_delivered_fraction`

## Rendered Rollouts

- `runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/media/10x10_2_ants_attempt_011_latest_passed_rollout.mp4`
- `runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/media/12x12_3_ants_attempt_001_latest_passed_rollout.mp4`
- `runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/media/16x16_4_ants_attempt_099_latest_passed_rollout.mp4`
- `runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/media/20x20_5_ants_attempt_007_latest_passed_rollout.mp4`
- `runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F018_low_dense_9x9_3ants_from_F014/media/9x9_3_ants_attempt_001_low_dense_rollout.mp4`

## Fresh 12x12 Anti-Stair Mini-Curriculum

Goal: rebuild a small randomized curriculum from a scratch lineage ending at
`12x12_3_ants`, without increasing actor vision radius, fixing placements, or
making the environment easier. The behavioral goal is to break the huge
staircase-writing pattern while preserving actual pickup-to-delivery behavior.

Stable constraints used across the useful runs:

- Actor vision radius stayed `1`.
- Food and hub stayed randomized.
- Observation canvas stayed `50x50`.
- Writes were monitored explicitly; latest useful branches had applied write
  rate `0.0`, so they avoided the staircase but may also be underusing memory.

Current best fresh-lineage checkpoint:

`runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F018_low_dense_9x9_3ants_from_F014/checkpoints/9x9_3_ants/attempt_001.pkl`

Current blocker:

`10x10_3_ants` under the low-dense scout recipe. Attempt 1 reached deterministic
delivered fraction `0.5859`, success `0.562`, pickup-to-delivery `0.5938`,
episode length `266.8`, and write rate `0.0`. Attempt 2 regressed to delivered
fraction `0.5234`, success `0.500`, pickup-to-delivery `0.5312`, episode length
`277.4`, and write rate `0.0`.

| candidate | parent | result | deterministic delivered | deterministic success | deterministic p2d | deterministic length | write rate | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `F008-scratch-obs50-foundation` | scratch | failed `6x6_1_ants` after passing `4x4_1_ants` | `0.8906` best tail | `0.875` | `0.9688` sampled p2d tail | `95.2` sampled tail | `0.0` | Proved the anti-stair pressure could eliminate writes on small stages. |
| `F009-robust-from-F008-6x6` | F008 `6x6` attempt 010 | failed `6x6_1_ants` | `0.9375` | `0.875` | `0.9531` | `75.8` | `0.0` | Better length and returns, still missed delivery robustness. |
| `F010-sharpen-from-F009-6x6` | F009 `6x6` attempt 008 | failed strict `6x6_1_ants` | `0.9844` attempt 1 | `0.938` | `1.0000` attempt 1 | `76.3` | `0.0` | Best deterministic no-stair 6x6 parent; strict sampled gate stayed brittle. |
| `F011-step-pressure-from-F010-6x6` | F010 `6x6` attempt 003 | rejected | `0.8750` | `0.875` | `0.8750` | `83-85` | `0.0` | More step penalty hurt pickup-to-delivery instead of fixing it. |
| `F014-carry-shaping-8x8-from-F013` | F013 `8x8` attempt 002 | useful but did not strict-pass `8x8_2_ants` | `0.8646` | `0.812` | `0.8646` | `139.3` | `0.0` | Strong carry shaping improved 8x8 but reward-hacked larger stages. |
| `F017-scout-9x9-3ants-from-F014-8x8` | F014 `8x8` attempt 001 | rejected | `0.5179` | `0.312` | `0.5714` | `255.8` | `0.0` | High shaped return did not mean delivery; dense shaping swamped the sparse task. |
| `F018-low-dense-9x9-3ants-from-F014` | F014 `8x8` attempt 001 | passed scout `9x9_3_ants`, blocked at `10x10_3_ants` | `0.7946` on passed 9x9 | `0.750` | `0.8482` | `180.7` | `0.0` | Best current recipe: lower dense shaping plus 3 ants earlier. |

Useful recipe from F018:

```bash
--stage-plan 4:1,6:1,8:2,9:3,10:3,11:3,12:3
--food-counts-by-stage 2,4,6,7,8,9,10
--gamma 0.98 --ent-coef 0.0025 --learning-rate 3e-5
--pickup-bonus 0.16 --step-penalty 0.0015
--visible-food-approach-bonus 0.003 --visible-food-stall-penalty 0.0005
--carrying-hub-approach-bonus 0.012 --carrying-hub-stall-penalty 0.002
--write-bit-penalty 0.0001 --write-overwrite-penalty 0.0005
--gate-eval-modes deterministic
--gate-min-delivered-fraction 0.75 --gate-min-pickup-to-delivery 0.75
--gate-max-applied-write-rate 0.20 --gate-length-fraction 0.75
```

Interpretation:

The run did break the visible staircase pattern, but the best policies have
collapsed to near-zero writes. That is good for avoiding the old failure mode,
but it likely creates a new scale wall at `10x10`: the ants can solve a bridge
case without writing, then lose delivery reliability as random placement grows.
The next research direction should be controlled communication, not stronger
write suppression: allow modest write rate, avoid broad staircase writes with a
max-write gate, and look for a positive reason to write near food/hub paths if a
trainer-side signal is added later.

## Observed Failure Mode

The rendered rollouts show ants often have food visible in their actor window but
choose a route that loops beside or around the food instead of stepping onto the
food cell. Pickup itself is automatic and works when an ant enters a food cell.
The failure is therefore policy action selection before pickup, not an environment
pickup bug.

Numerical tracing of the `20x20_5_ants` deterministic render found:

- 47 ant-steps with visible food.
- 18 steps that targeted an actual food cell.
- 18 pickups.
- 0 cases where an ant stepped onto food and failed to pick it up.

## Candidate Summary

- `C029-a074-seed13-lr2e7` produced `attempt_091`, a stable near-pass:
  deterministic delivered fraction `0.9647`, deterministic success `0.8125`,
  sampled delivered fraction `0.9348`, sampled success `0.7500`.
- Micro-LR continuations from `attempt_091` (`C030`, `C031`) preserved the
  near-pass briefly, then drifted.
- Step penalty branches (`C032`, `C034`) did not move sampled delivered fraction
  above the plateau.
- Stronger pickup bonus (`C035`) improved training return but did not fix the
  visible-food bypass behavior.
- Visible-food shaping was added as an opt-in trainer-only signal:
  `--visible-food-approach-bonus` and `--visible-food-stall-penalty`.
- Gentle visible-food shaping (`C036`, `C037`) still plateaued around sampled
  delivered fraction `0.93-0.94`.
- Aggressive visible-food shaping from `attempt_091` (`C038`) produced
  `attempt_195.pkl`, but manual evaluation failed both deterministic and sampled
  gates. Do not use `attempt_195.pkl` as a parent.
- Restarting `25x25_6_ants` from the passed `20x20_5_ants` checkpoint with
  strong visible-food shaping (`C039-from20-visfood05-stall05-lr1e6`) failed.
  Best C039 attempt was `attempt_202` with gate score `0.750`, deterministic
  delivered fraction `0.9647`, deterministic success `0.8125`, sampled
  delivered fraction `0.8832`, sampled success `0.6250`. Do not use the C039
  tail as a parent.
- Conservative entropy sharpening from `attempt_091` (`C040-a091-ent005-lr2e8`)
  preserved the clean near-pass but did not improve it. Attempts `204-211`
  stayed at gate score `0.875`, deterministic delivered fraction `0.9647`,
  deterministic success `0.8125`, sampled success `0.7500`, sampled pickups
  `21.62`, and sampled delivered fraction `0.9266-0.9348`.
- Larger-batch updates from `attempt_091` (`C041-a091-env32-lr1e7`) were worse
  than the clean plateau. Attempts `212-217` reached sampled pickups up to
  `22.00`, but sampled success stayed `0.625-0.6875`, sampled delivered fraction
  stayed below `0.95`, and the best gate score was `0.750`.
- Higher long-horizon discount plus small entropy encouragement from
  `attempt_091` (`C042-a091-g995-ent001-lr5e8`) preserved deterministic behavior
  but did not raise sampled delivery enough. Attempts `218-224` failed only
  `sampled.eval_mean_delivered_fraction`; best sampled success was `0.7500`,
  best sampled delivered fraction was `0.9348`, and the best checkpoint recorded
  by state was `attempt_224`. Attempt `225` regressed to sampled success
  `0.6875`.
- Higher long-horizon discount plus entropy sharpening
  (`C043-a091-g995-entneg5e4-lr1e7`) was worse. Early attempts `226-228` failed
  only sampled delivered fraction, but later attempts damaged sampled success
  and then deterministic delivery. The best sampled delivered fraction reached
  `0.9429` at `attempt_233`, but deterministic delivered fraction had collapsed
  to `0.8342`, so this branch should not be used as a parent.
- Reusing the optimizer state from `attempt_091`
  (`C044-a091-keepopt-g985-entneg5e4-lr5e8`) changed the update loss but did not
  break the plateau. Best sampled delivered fraction was still `0.9348`; attempts
  `238-241` settled at sampled delivered fraction `0.9266` with sampled success
  `0.7500`. The final state best is `attempt_241`, but it is not better than the
  clean parent for deciding the next branch.
- Midpoint discount retry (`C045-a091-g990-entneg5e4-lr5e8`) also stayed flat.
  Attempt `242` matched the plateau with sampled delivered fraction `0.9348`
  and sampled success `0.7500`; attempts `245-248` settled at `0.9239`, and the
  final attempt `249` regressed sampled success to `0.6875`. Deterministic
  metrics remained strong throughout.

## Deterministic Scout

The strict gate is blocked by sampled generalization, not by deterministic task
competence. A deterministic-only branch is useful to learn whether later
curriculum stages are reachable, but it should not be treated as the final
solved policy. Keep the strict run directory intact and use a separate scout run
directory with `--gate-eval-modes deterministic`; sampled metrics are still
evaluated and stored in `gate_history.jsonl`, but they do not block scout
progress.

Suggested deterministic scout command:

```bash
PYTHONPATH=src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
/home/juan/miniconda3/envs/cool-antz/bin/python \
  -m ant_byte_env.training.jax_mappo.map_ant_curriculum \
  --run-dir runs/autoresearch/map_ant_curriculum/25x25_to_50x50_deterministic_scout \
  --stage-plan 25:6,32:8,40:10,50:10 \
  --food-sources-by-stage 1,1,2,2 \
  --start-stage 25x25_6_ants \
  --start-checkpoint runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/checkpoints/25x25_6_ants/attempt_248.pkl \
  --candidate-id S001-detgate-from-a248-g985-entneg5e4-lr5e8 \
  --seed 13 \
  --gamma 0.985 \
  --ent-coef -0.0005 \
  --learning-rate 5e-8 \
  --actor-vision-radius 1 \
  --pickup-bonus 0.05 \
  --step-penalty 0.0 \
  --gate-eval-modes deterministic \
  --gate-update-chunk-cap 1 \
  --gate-max-stage-attempts 8 \
  --gate-length-fraction 0.65 \
  --reset-opt-state-on-load
```

Scout result:

- `S001-detgate-from-a248-g985-entneg5e4-lr5e8` passed `25x25_6_ants`
  deterministically on attempt `001`, then failed `32x32_8_ants` after attempts
  `001-008`.
- The `25x25_6_ants` scout pass had deterministic delivered fraction `0.9647`,
  deterministic success `0.8125`, and sampled delivered fraction `0.9212`.
- The `32x32_8_ants` scout plateau failed only
  `deterministic.eval_mean_delivered_fraction`; attempts stayed around
  deterministic delivered fraction `0.9125` with deterministic success `0.8125`.
- Passed scout checkpoint:
  `runs/autoresearch/map_ant_curriculum/25x25_to_50x50_deterministic_scout/checkpoints/25x25_6_ants/attempt_001.pkl`.
- Current 32x32 scout checkpoint:
  `runs/autoresearch/map_ant_curriculum/25x25_to_50x50_deterministic_scout/checkpoints/32x32_8_ants/attempt_008.pkl`.

## 25x25 Seeded Renders

Rendered two `25x25_6_ants` videos from the deterministic scout pass using
different render reset seeds:

- Timeout seed: `seed_offset=444444`, reset seed `444457`, hub `(19, 14)`.
  Render path:
  `runs/autoresearch/map_ant_curriculum/25x25_to_50x50_deterministic_scout/media/25x25_6_ants_attempt_001_seed444444_hub19_14_rollout.mp4`.
  Render-matching rollout delivered `18/23` and timed out at `2500` steps.
- Success seed: `seed_offset=900001`, reset seed `900014`, hub `(18, 4)`.
  Render path:
  `runs/autoresearch/map_ant_curriculum/25x25_to_50x50_deterministic_scout/media/25x25_6_ants_attempt_001_seed900001_hub18_4_rollout.mp4`.
  Render-matching rollout delivered `23/23` and terminated at step `1810`.

Quick render-seed sweep with render-style pinned hub reset:

| seed offset | hub | delivered | length | outcome |
| --- | --- | ---: | ---: | --- |
| `100000` | `(0, 23)` | `23/23` | `2216` | pass |
| `12345` | `(23, 7)` | `23/23` | `1152` | pass |
| `222222` | `(21, 18)` | `23/23` | `1099` | pass |
| `333333` | `(3, 24)` | `15/23` | `2500` | timeout |
| `444444` | `(19, 14)` | `18/23` | `2500` | timeout |
| `555555` | `(1, 1)` | `0/23` | `2500` | timeout |
| `765432` | `(13, 2)` | `0/23` | `2500` | timeout |
| `900001` | `(18, 4)` | `23/23` | `1810` | pass |

Interpretation: the 25x25 policy is strongly placement-sensitive. Changing the
hub/food seed can move the same checkpoint from a clean solve to a total failure,
so the strict sampled gate is exposing a real generalization issue rather than
only being too harsh.

## 12x12 3-Ant Behavior Shaping Track

Goal: return to the simpler randomized `12x12_3_ants` environment and try to
shift behavior away from dense staircase-like byte writing while preserving
pickup/delivery competence. Hub and food remain randomized; actor vision radius
remains `1`.

Best starting checkpoint:

`runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/checkpoints/12x12_3_ants/attempt_001.pkl`

This checkpoint passed the earlier strict 12x12 gate with deterministic delivered
fraction `0.9625`, deterministic success `0.875`, deterministic pickup-to-delivery
rate `1.0`, sampled delivered fraction `1.0`, and sampled success `1.0`. Its
weakness is high applied nonzero write rate: deterministic `0.4523`, sampled
`0.5001`.

Implementation notes added for this track:

- `--carrying-hub-approach-bonus` and `--carrying-hub-stall-penalty` add
  trainer-side shaping for ants already carrying food to move toward the hub.
- `--obs-width` and `--obs-height` let focused single-stage runs load
  checkpoints trained with the larger curriculum observation canvas.
- `--write-overwrite-penalty` penalizes nonzero writes onto tiles that were
  already nonzero, targeting repeated/repainted trail behavior separately from
  global write sparsity.

Candidate results:

| candidate | parent | best score | best deterministic delivered | best deterministic write | best sampled delivered | best sampled write | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `B001-12x12-hubshape-writeband` | scratch | `0.1667` | `0.0000` | `0.3813` | `0.1563` | `0.4971` | Bad parent; deterministic learned no pickups. |
| `B002-12x12-lr-shaping-writepen` | B001 attempt 008 | `0.1667` | `0.0000` | `1.0000` | `0.1688` | `0.5760` | Continued from a bad checkpoint; write head collapsed toward all-write. |
| `B003-12x12-goodbase-write-trim` | passed 12x12 checkpoint | `0.7500` | `0.9438` | `0.4368` | `1.0000` | `0.4635` | Good competence, but global write penalty `0.003` did not trim enough. |
| `B004-12x12-goodbase-strong-write-trim` | passed 12x12 checkpoint | `0.7500` | `0.9063` | `0.4166` | `1.0000` | `0.4298` | Stronger global write penalty `0.01` lowered writes slightly but hurt deterministic delivery. |
| `B005-12x12-goodbase-overwrite-trim` | passed 12x12 checkpoint | `0.8333` | `0.9625` | `0.4297` | `1.0000` | `0.4418` | Passed all delivery/length checks on attempt 3; still failed only max write-rate checks. |
| `B006-12x12-goodbase-write40-shaping` | passed 12x12 checkpoint | `0.8333` | `1.0000` | `0.4508` | `1.0000` | `0.5267` | Stronger task shaping plus intermediate `0.40` write ceiling preserved perfect delivery but reinforced high write use. |
| `B007-12x12-goodbase-step005` | passed 12x12 checkpoint | `1.0000` | `1.0000` | `0.4585` | `1.0000` | `0.5132` | Higher step penalty `0.005` plus strict `0.35` length gate passed on attempt 2; delivery was faster, but writes stayed high. |

Current read: start from the passed 12x12 checkpoint, not the scratch B001/B002
branches. The overwrite penalty direction is more promising than stronger task
shaping alone: B005 preserved delivery and failed only max write checks, while
B006 reached perfect delivery by leaning back into the old high-write behavior.
B007 confirms that a larger step penalty can make the policy more urgent about
pickup/delivery, but by itself it also leans back into high write use.
B007 attempt 2 is the best speed/food-urgency checkpoint so far:

`runs/autoresearch/map_ant_curriculum/12x12_3ants_goodbase_B007_step005/checkpoints/12x12_3_ants/attempt_002.pkl`

It passed all delivery/length checks with deterministic delivered fraction
`1.0`, deterministic success `1.0`, sampled delivered fraction `1.0`, and
sampled success `1.0`. Mean episode length improved to deterministic `141.6875`
and sampled `189.875` under the strict `0.35` length gate. It still wrote too
often: deterministic `0.4585`, sampled `0.5132`.

B007 render:

`runs/autoresearch/map_ant_curriculum/12x12_3ants_goodbase_B007_step005/media/12x12_3_ants_attempt_002_step005_rollout.mp4`

Render trace on the exact video seed delivered all `10/10` food but took `464`
steps, so that sampled seed is slower than the average gate result. The specific
"visible food but loops away" issue improved sharply versus B005: B005 had `10`
move-away decisions across `39` visible-food non-carrying events, while B007 had
`1` move-away decision across `20` events, and that one happened while another
ant picked up the final remaining food on the same step.

Next useful experiments should use the passed 12x12 checkpoint as parent and
either:

- keep the stronger step penalty and reintroduce a write ceiling/overwrite
  penalty so urgency does not become dense writing, or
- try a midpoint step penalty such as `0.003` if the render behavior is better
  but the `0.005` policy is too brittle on individual sampled seeds.

### A010/A011 Anti-Stair Target Run

Goal: run the learned curriculum continuation into randomized `12x12_3_ants`
without making the environment easier, while adding an explicit upper write-rate
gate to reduce the dense staircase pattern. Actor vision radius stayed `1`; hub
and food stayed randomized.

Useful passed parent:

`runs/autoresearch/map_ant_curriculum/10x10_to_50x50_chain/checkpoints/10x10_2_ants/attempt_011.pkl`

Cold-start and 10x10 anti-stair retries failed to pass the strict deterministic
delivery gate. The useful route was to continue from the already-passed 10x10
curriculum checkpoint into the target 12x12 stage.

| candidate | parent | write ceiling | score | deterministic delivered | deterministic write | sampled delivered | sampled write | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `A010-target-12-antistair-from-passed10` | passed 10x10 attempt 011 | `0.50` | `1.0000` | `0.9813` | `0.4398` | `1.0000` | `0.4832` | Passed target on attempt 1. This is the current best 12x12 endpoint with a bounded write-rate gate. |
| `A011-target-12-sparsewrite-from-A010` | A010 attempt 001 | `0.35` | `0.7500` | `0.9438` best | `0.3922` best | `1.0000` | `0.3973` best | Lowered writes slightly but never reached `0.35`, and deterministic delivery fell below the gate. |

A010 checkpoint:

`runs/autoresearch/map_ant_curriculum/12x12_3ants_goodbase_A010_target_antistair/checkpoints/12x12_3_ants/attempt_001.pkl`

A010 render:

`runs/autoresearch/map_ant_curriculum/12x12_3ants_goodbase_A010_target_antistair/media/12x12_3_ants_attempt_001_antistair_rollout.mp4`

Visual read: A010 is a valid target pass and keeps sampled write rate below the
new `0.50` ceiling, but the rendered rollout still shows a broad wedge/staircase
of `1` writes. A sudden `0.35` ceiling plus strong write penalties was too harsh;
the next promising direction is gradual write-budget annealing, for example
`0.50 -> 0.45 -> 0.40`, while preserving the A010 task competence.

### F023-F026 Fresh Anti-Stair Mini Curriculum

Goal: restart the curriculum from a clean small-map foundation to see whether
the dense staircase behavior was inherited from earlier larger-map checkpoints.
The environment was not made easier: actor vision radius stayed `1`, food and
hub stayed randomized, and the run used a max applied write-rate gate to keep
the staircase from returning.

Fresh deterministic-gated branch:

| candidate | route | best passed stage | blocker | note |
| --- | --- | --- | --- | --- |
| `F023-seed7-foundation-ladder` | scratch `4x4 -> 12x12` | `6x6_1_ants` | deterministic `7x7_2_ants` delivery/success/p2d | Broke the staircase but collapsed to a near no-write policy. Sampled 7x7 behavior was strong. |
| `F024-det-sharpen-7x7-from-F023-6x6` | `6x6` pass -> `7x7` | none beyond parent | deterministic `7x7_2_ants` delivery/success/p2d | Lower entropy, lower LR, and higher step pressure did not improve deterministic 7x7. |

Sampled-gated branch:

| stage | attempt | pass | sampled delivered | sampled success | sampled p2d | sampled length | sampled write | note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `7x7_2_ants` | 1 | yes | `0.9250` | `0.812` | `0.9406` | `115.8` | `0.0002` | Started from F023 7x7 attempt 4. |
| `8x8_2_ants` | 2 | yes | `0.8854` | `0.750` | `0.9719` | `160.6` | `0.0005` | Attempt 1 missed only success. |
| `9x9_3_ants` | 2 | yes | `0.9643` | `0.875` | `0.9911` | `230.1` | `0.0004` | First 3-ant pass in this fresh lineage. |
| `10x10_3_ants` | 2 | yes | `0.9219` | `0.750` | `0.9540` | `259.0` | `0.0001` | Latest successful stage. |
| `11x11_3_ants` | 1 | no | `0.9236` | `0.688` | `0.9649` | `363.0` | `0.0002` | Best near-pass; missed only success. |
| `11x11_3_ants` | 2 | no | `0.8750` | `0.562` | `0.9566` | `402.4` | `0.0001` | Regressed; added length failure. |

Important checkpoints:

- F025 latest pass:
  `runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F025_sampled_gate_from_F023_7x7_best/checkpoints/10x10_3_ants/attempt_002.pkl`
- F025 best 11x11 near-pass:
  `runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F025_sampled_gate_from_F023_7x7_best/checkpoints/11x11_3_ants/attempt_001.pkl`
- F026 long-horizon retry from the 11x11 near-pass regressed:
  attempt 1 sampled delivered `0.8681`, success `0.625`, p2d `0.9343`,
  length `374.8`; it failed success and length. The `gamma=0.99`, lower LR,
  and stronger hub shaping direction did not solve the close-out.

Renders produced from F025:

- Sampled 10x10 latest pass:
  `runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F025_sampled_gate_from_F023_7x7_best/media/10x10_3_ants_attempt_002_sampled_rollout.mp4`
- Deterministic 10x10 contrast:
  `runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F025_sampled_gate_from_F023_7x7_best/media/10x10_3_ants_attempt_002_deterministic_rollout.mp4`
- Sampled 11x11 near-pass:
  `runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F025_sampled_gate_from_F023_7x7_best/media/11x11_3_ants_attempt_001_sampled_nearpass_rollout.mp4`

Current read: the fresh lineage did break the huge staircase, but mostly by
learning a no-write/random-walk-ish strategy rather than a clean food-to-hub
communication behavior. Deterministic gating blocks at 7x7; sampled gating can
carry the fresh policy through 10x10 but stalls at 11x11 because success rate
does not close enough episodes. The next useful run should start from F025
`11x11_3_ants` attempt 1, not F026, and tune for success without length
regression. Avoid the F026 `gamma=0.99` branch unless paired with a different
time-pressure setup.

## Fresh 12x12 Target Retries

Follow-up target-only retries kept the fresh lineage and did not make the
environment easier: randomized hub/food, 12x12, 3 ants, 10 food, one source,
actor vision radius 1, and sampled-only gate. Best checkpoint:

`runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F031_ent001_lr5e5_from_F029_a6/checkpoints/12x12_3_ants/attempt_001.pkl`

Best sampled gate metrics so far:

| Run | Source | Delivered | Success | P2D | Length | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| F029 a06 | F025 11x11 a01 then direct 12x12 | `0.8938` | `0.500` | `0.9308` | `455.4` | Strong harvesting, still misses full clears. |
| F031 a01 | F029 a06, lower entropy/lr | `0.8937` | `0.562` | `0.9499` | `462.8` | Current best success rate. |
| F032 a01 | F031 a01, stronger hub shaping | `0.8313` | `0.500` | `0.8963` | `471.6` | Regressed. |
| F033 a01 | F031 a01, stronger visible-food stall | `0.8500` | `0.375` | `0.9217` | `503.4` | Regressed. |
| F034 a01 | F029 a06, lower entropy again | `0.8500` | `0.438` | `0.9170` | `471.9` | Worse than F031. |
| F035 a02 | F031 a01, completion bonus 2.0 | `0.8438` | `0.500` | `0.9302` | `473.4` | New bonus did not improve this checkpoint. |

Render for the current best sampled near-pass:

`runs/autoresearch/map_ant_curriculum/12x12_3ants_fresh_F031_ent001_lr5e5_from_F029_a6/media/12x12_3_ants_attempt_001_sampled_nearpass_rollout.mp4`

Per-episode replay of the F031 sampled gate used the sampled gate seed offset.
It solved 9/16 episodes. Failures were close rather than total: several ended
with 7-9 delivered, and two had source food depleted but one carried food not
returned by timeout. This suggests the current bottleneck is robust full clears
under random hub-food separation, not pickup-to-delivery conversion or excessive
writing. The actor already observes facing direction as a one-hot, so adding
direction is not an open missing-observation item.

Added trainer-side `--completion-bonus` as a default-off reward-shaping flag.
It rewards crossing the full delivered-food count during training and does not
change evaluation, random placement, map size, food count, ant count, or vision.
Initial `2.0` retry did not beat F031 a01, but the flag is now available for
future sweeps.

## Validation

Latest focused validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
/home/juan/miniconda3/envs/cool-antz/bin/python -m pytest \
  tests/test_notebook_workflows.py \
  tests/test_notebooks.py \
  tests/test_train_mappo_jax.py \
  tests/test_map_ant_curriculum.py
```

Result: `87 passed, 1 warning`.

After adding deterministic-only gate mode:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
/home/juan/miniconda3/envs/cool-antz/bin/python -m pytest \
  tests/test_notebook_workflows.py \
  tests/test_notebooks.py \
  tests/test_train_mappo_jax.py \
  tests/test_map_ant_curriculum.py -q
```

Result: `90 passed, 1 warning`.

After adding carrying-to-hub shaping, obs-canvas overrides, and overwrite
write shaping:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
/home/juan/miniconda3/envs/cool-antz/bin/python -m pytest \
  tests/test_train_mappo_jax.py \
  tests/test_map_ant_curriculum.py \
  tests/test_notebook_workflows.py -q
```

Result: `94 passed, 1 warning`.
