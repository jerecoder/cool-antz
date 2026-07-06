# Experiment Rationale

This memo explains why each experiment family exists and how to phrase its
evidence safely in the report.

## Baselines

`direct_goal_baseline` is the main negative baseline because it trains the
final 50x50, 10-ant, 5-bit target directly from sparse delivery reward. It
answers whether the target task is solved without curriculum or shaping.

`smoke` is not a research baseline. It exists to catch broken CLI, dependency,
or run-directory plumbing.

## Curriculum And Autocurriculum

The basic forage, exploration, and autocurriculum configs test whether easier
subtasks can produce reusable policy structure. Exploration-only runs are
diagnostic: they can show coverage competence, but they do not prove foraging.
Autocurriculum runs are useful when the question is whether one policy can
survive a growing task distribution without manual stage handoff.

## Source Layouts

The padded-source, proximity-source, and smooth-source experiments separate
task geometry from model capability. They test whether failure on rare 50x50
sources is caused by sparse discovery, unstable delivery conversion, or a
fundamental multi-agent coordination limit. Claims should name the geometry:
food-source count, source footprint, hub randomization, and hidden arena size.

## Communication And Write Cost

Shared-write, per-ant-channel, write-bit, and write-cost experiments ask
whether byte marks are useful, cheap, or overused. They do not by themselves
prove communication is causal. The safe claim is that a policy trained with a
given write interface achieved a measured result; causal claims need matched
ablations with comparable checkpoints, seeds, and task geometry.

## Scaling

The 8-ant, 16-ant, 60-ant, and multi-device configs test whether more actors,
identity types, or hardware throughput change delivery behavior. Multi-device
data parallelism is an execution strategy, not a scientific treatment: it
should be reported as infrastructure unless the batch-size or rollout-shape
change is itself part of the experiment.

## Timed Release

The timed-release roles experiment tests whether the cooperative 8-ant shared
write policy contains reusable role structure when ant ranks are revealed over
time. The final intended profile is a full L4 continuation: 128 envs, 256
steps, 16,384,000 base timesteps, strided CNN critic, training movement
temperature 0.75, and eval/render movement temperature 0.52. Inactive ants are
masked out of actor loss and observations until released.

This does not prove role causality by itself; it is a continuation/probe of a
specific trained policy.

## Adversarial Frozen Opponent

The adversarial frozen-opponent configs ask whether a warm-started cooperative
actor can operate under an opponent-shaped objective. The result should be
phrased as an adversarial capability audit, not as cooperative foraging
performance. Use the adversarial docs and metrics separately from delivery
claims.

## Map-Ant And Direct Goal

The map-ant gated MLP curriculum preserves historical evidence that the older
MLP critic made real progress under strict gates but did not solve the final
50x50 task. The 12x12 conv-critic/autoresearch documents are evidence ledgers
for a failed newer lineage, not a replacement mainline workflow.

Unsupported historical shaping fields are retained as config/workflow
metadata. The live workflow does not pass them to the JAX trainer.

## Maze And Lethal Cookies

Maze and lethal-cookie work is pipeline/geometry evidence unless matched
delivery metrics are recovered. Lethal food is visible through the normal food
channel and hidden as a private lethal channel; death events and remaining
lethal food are diagnostics. Do not compare lethal-cookie results directly to
positive-only runs without naming the changed task.

## Vision Shrink

Vision-shrink configs remain exploratory. They are useful for asking whether a
large-vision policy can be warm-started toward a local-vision policy, but no
strong final evidence was recovered during this integration pass.

## Report-Site Large-Scale Evidence

The report-site imports are evidence and presentation assets, not a wholesale
code merge. The 60-ant 50x50, 100x100 bridge/progress-video, and 250x250
distance/truncation diagnostics should be cited through
`docs/planning/complete-experiment-chronology.md`,
`docs/planning/experiment-history-analysis.md`, and `report/data/`.
