# Adversarial MARL Experiment Plan

Goal: test whether AntByte can support a two-team adversarial foraging setup with
the smallest useful change from the current JAX MAPPO code. This branch starts
from `repo/research-integration-cleanup` and should treat adversarial work as an
experimental path, not as a replacement for the cooperative environment.

## First Experiment

Start with one trainable MAPPO team against one frozen opponent policy loaded
from the best compatible cooperative checkpoint.

This is simpler than two live MAPPO trainers because the opponent is part of the
environment dynamics from the learner's perspective. It tests the important new
pieces first:

- two hubs;
- equal ants per team;
- team-specific delivery counters;
- rewards with the opponent's deliveries subtracted;
- action composition for controlled and opponent ants;
- checkpoint warm-start from the cooperative policy;
- side swapping during evaluation.

Do not start with two simultaneously training policies. That adds non-stationary
self-play instability before the environment and reward signs are validated.

## Minimal Environment Contract

Add an experimental JAX adversarial environment or wrapper with these semantics:

```text
num_ants_per_team = N
total ants = 2N
team_count = 2
hub_pos shape = (2, 2)
delivered_food shape = (2,)
reward[team] = own_deliveries - opponent_deliveries
```

The first version can keep shared food, shared byte grid, shared obstacles, and
the existing movement/write action format. Ant order should be deterministic:
team 0 ants first, then team 1 ants. That keeps action concatenation and
checkpoint transfer easy to inspect.

Episode termination should initially stay simple:

- truncate at `max_steps`;
- terminate when all food is exhausted or a configurable delivery limit is hit;
- report both teams' delivered counts in info.

## Observation Strategy

Keep actor observations local. Ants should still see only their configured
actor-vision radius, not global vectors to the opponent hub or distant opponent
ants. The adversarial version should change the meaning of local features, not
give the actor a broader information surface.

For a requested learner team:

- local food, border, obstacle, carrying, facing, and write-bit features keep the
  current semantics;
- the byte grid stays shared, with the same values for both teams;
- local ant occupancy should distinguish own ants from opponent ants when they
  appear inside the local window;
- the local hub feature should distinguish own hub from opponent hub only when a
  hub appears inside the local window;
- no actor feature should reveal an off-screen hub or off-screen opponent ant.

The most checkpoint-compatible first encoding is to reuse existing local planes
with signed symbols: own hub/ants positive, opponent hub/ants negative, and
absence zero. If that proves too brittle, append local-only own/opponent planes
and initialize the new input weights near zero, but do not add global actor
features.

The centralized critic can still receive full team-labeled state, because MAPPO
already uses centralized training. Keep the actor information surface local and
use the critic for global value estimation.

Avoid feeding the same absolute state to one critic with opposite reward signs.
If both teams share one actor/critic later, every team trajectory must be
canonicalized into that team's perspective before computing values or updates.

## Training Path

Phase 1 should use a frozen opponent:

1. Load the cooperative checkpoint into the trainable learner.
2. Load the same checkpoint into the frozen opponent.
3. Build learner actor observations from learner perspective.
4. Build opponent actor observations from opponent perspective.
5. Sample learner actions from trainable params.
6. Sample opponent actions from frozen params.
7. Concatenate actions in environment ant order.
8. Step the adversarial env.
9. Update only the learner MAPPO params from learner-perspective rewards.

Keep one scalar value target for the learner in this phase. The runner should
look like the current JAX MAPPO runner with an extra frozen policy call during
rollout collection.

## Evaluation Matrix

The first useful evaluation is small and diagnostic:

| Matchup | Purpose |
| --- | --- |
| frozen checkpoint vs frozen checkpoint | establishes symmetric baseline |
| learner vs frozen checkpoint | measures adversarial improvement |
| frozen checkpoint vs learner | catches side-order and perspective bugs |
| random policy vs frozen checkpoint | confirms the frozen policy is meaningful |
| learner vs random policy | confirms the learner can exploit a weak opponent |

Report at least:

- own deliveries;
- opponent deliveries;
- delivery difference;
- win rate by delivery difference;
- pickup events by team;
- mean episode length;
- side-swapped score gap.

If side-swapped results differ materially, fix environment symmetry before
spending compute.

## Tests Before Compute

Add tests before launching real training:

1. Reset creates two distinct hubs and equal ants per team.
2. Delivering to team 0 hub increments only team 0 delivered count.
3. Delivering to team 1 hub increments only team 1 delivered count.
4. Team 0 reward is the negative of team 1 reward for the same transition.
5. Perspective transform swaps own/opponent hubs and ant ordering correctly.
6. Frozen-opponent rollout composes actions in the expected ant order.
7. A one-update adversarial dry run loads a checkpoint or initialized params and
   completes without changing frozen opponent params.

Keep these tests on tiny maps with deterministic food and hub positions. The
goal is to validate mechanics, not performance.

## Success Criteria For The Probe

The frozen-opponent probe is worth expanding only if:

- symmetric frozen-vs-frozen evaluation is close to zero delivery difference;
- side-swapped learner evaluations are consistent;
- learner beats random and does not lose trivially to the frozen policy;
- rewards and delivered counts agree with visual or scripted rollouts;
- the frozen opponent params are unchanged after training.

After this works, the next experiment should be shared self-play with
perspective-canonicalized trajectories. Two independent live MAPPO trainers
should come later, after the symmetric shared-policy version is stable.

## Non-Goals

- Do not rewrite the cooperative environment as adversarial-only.
- Do not change the current maintained experiment configs.
- Do not claim communication or strategic blocking until no-byte-read and
  no-write ablations show that bytes matter.
- Do not spend long compute on live self-play until the frozen-opponent probe
  passes the symmetry and side-swap checks.
