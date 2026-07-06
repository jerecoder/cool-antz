# Research Integration Plan

This branch should become the clean source of truth for report writing,
codespace review, and rerunning the important experiments. The guiding rule is
to improve organization and documentation without changing the behavior of
reproducible experiment configs unless a config is currently not executable.

## Baseline Experiments

Baseline experiments should be first-class in the docs because they explain why
the later curricula exist.

| Surface | Role | Integration decision |
| --- | --- | --- |
| `experiments/smoke.json` | Tiny Torch MAPPO plumbing check. | Keep as an engineering smoke test, not a research result. |
| `experiments/direct_goal_baseline.json` | Sparse final-task JAX MAPPO baseline: 50x50, 10 ants, 5 write bits, randomized food and hub, no shaping. | Keep as the negative-control baseline. Document that it asks MAPPO to solve the final target from scratch with only delivery reward. |
| `notebooks/baselines/direct_goal.ipynb` | Notebook-facing direct baseline. | Keep grouped under `notebooks/baselines/`. Add rationale text in docs rather than bloating the notebook. |
| `research/direct-goal-repro-sweep` | Sparse hyperparameter and reward-shaping branch. | Port only durable evidence/rationale. Do not overwrite the current baseline config with older branch settings. |

The final docs should say plainly: the direct sparse baseline is expected to be
hard. It is the anchor showing that the later exploration, forage, source
layout, communication, write-cost, timed-release, and adversarial experiments
are not arbitrary additions; they are responses to sparse long-horizon credit
assignment and coordination pressure.

## Markdown Structure

The current `docs/experiments.md` is useful but too table-driven. The polished
repo should have both an index and explanation docs.

Recommended structure:

- `docs/experiments.md`: concise reproduction index with every maintained config,
  notebook, expected artifact root, and one-line purpose.
- `docs/experiment_rationale.md`: narrative rationale by experiment family:
  baseline, forage curriculum, exploration curriculum, source layouts, scaling,
  communication/write bits, write-cost, timed-release roles, adversarial MARL,
  lethal cookies, and historical map-ant.
- `docs/integration_audit.md`: branch integration status, guardrails, and exact
  decisions about merge vs port vs archive.
- Existing focused docs such as `docs/map_ant_gated_mlp_curriculum.md`,
  `docs/timed_release_roles.md`, and
  `docs/adversarial_frozen_opponent_story.md` should link back to the index and
  state which claims depend on ignored `runs/` artifacts.

Required doc corrections already identified:

- `docs/integration_audit.md` should keep the historical fork point
  (`554ee8d`) separate from the current remote main used for merge readiness
  checks. Refresh its current-main hash and conflict list after integrating the
  newer main/write-cost work.
- `docs/experiments.md` currently lacks write-cost, multi-device, timed-release,
  adversarial, lethal-cookie, direct-goal sweep evidence, and 12x12 conv-critic
  evidence entries.
- Historical map-ant metadata points to
  `src/ant_byte_env/training/jax_mappo/map_ant_curriculum.py`, but the current
  runnable workflow lives in `src/ant_byte_env/workflows/map_ant.py`.
- Timed-release docs must pin the final branch-tip settings: local CPU probe,
  `actor_only_warm_start=true`, fresh MLP critic, and eval/render move
  temperature `0.52`.
- Adversarial docs must distinguish committed code/configs from ignored
  result paths under `runs/`.
- Lethal-cookie docs should be written fresh because that branch changes the
  identity of existing proximity-source files and has contradictory metadata.

## JAX MAPPO Organization Review

The current split is good and should be preserved:

- `cli.py`: training arguments and validation.
- `types.py`: parameter, optimizer, rollout, and metric containers.
- `models.py`: initialization and critic/actor forward passes.
- `observations.py`: actor and critic observation builders.
- `policy.py`: action distributions, sampling, log-prob, and entropy.
- `rewards.py`: trainer-side shaping and write penalties.
- `curriculum.py`: reset and random-layout sampling.
- `rollout.py`: collect PPO rollout data.
- `updates.py`: GAE, Adam, PPO loss, and update loops.
- `runner.py`: executable training loop, logging, checkpointing, and callbacks.
- `transfer*.py`: checkpoint adaptation.
- `evaluation.py`, `probe.py`, `layout_audit.py`: analysis/eval surfaces.
- `core.py`: compatibility facade only.

Cleanup targets:

1. Keep `core.py` as a compatibility facade, but new code should import from
   the owned modules directly.
2. Split the flat `cli.py` parser into small argument-section helpers before
   adding more branch flags. This preserves the CLI while making intent visible.
3. Factor `runner.py` metric assembly and best-checkpoint selection into small
   helpers before integrating adversarial/timed-release additions.
4. Reduce `Transition`/`Rollout` duplication or add helper constructors because
   branch integrations add fields such as `agent_masks` and
   `behavior_anchor_kl`.
5. Integrate timed-release `agent_masks` in shared PPO as an optional all-ones
   mask for normal experiments. This should not change existing rollouts.
6. Integrate adversarial `behavior_anchor_kl` and `freeze_actor` as default-off
   PPO update features.
7. Port multi-device support so `data_parallel.py` imports from `types.py` and
   `updates.py`, not from `core.py`.
8. Keep branch workflow packages (`timed_release/`, `adversarial/`) separate
   from the main cooperative runner.

Formatting and style notes:

- Some MAPPO files need normal blank-line polish after imports/classes.
- `tests/test_train_mappo_jax.py` is large but valuable. Do not split tests
  until the integration is stable; during integration, adding focused workflow
  test files is safer than reshuffling this file.
- Avoid behavior-only comments. Add comments only around non-obvious experiment
  compatibility decisions.

## Confirmed Reproducibility Bug

A tiny actual run of the historical map-ant workflow currently fails before
training. The workflow forwards obsolete or workflow-only flags to the JAX
trainer:

- `--wandb-project-name`
- `--write-overwrite-penalty`
- `--visible-food-approach-bonus`
- `--visible-food-stall-penalty`
- `--carrying-hub-approach-bonus`
- `--carrying-hub-stall-penalty`

The config dry-run succeeds because it validates workflow argument resolution,
not the nested trainer invocation. Since the committed historical config sets
these legacy shaping values to `0.0`, the behavior-preserving fix is to stop
forwarding zero-valued unsupported flags or map supported tracking flags to the
current JAX names. Add a tiny map-ant smoke test that executes one update, not
only `--dry-run`.

## Branch Integration Decisions

| Ref | Decision |
| --- | --- |
| `origin/main` | Integrate first. It adds 4-bit and 8-bit write-cost experiments plus render-style support. |
| `origin/feat/multi-device-jax-mappo` | Port after main. Keep write-cost sweeps, data-parallel helper, and hardware-gated config/docs. |
| `research/timed-release-roles` | Integrate branch-tip experiment and package. Keep actor-only warm start and `agent_masks`; polish notebook/docs. |
| `research/adversarial-marl-experiments` | Integrate package/config/tests; keep claims careful because selected artifacts live under ignored `runs/`. |
| `origin/lethal_cookies` | Port manually. Preserve positive-only proximity-source baseline and add lethal-cookie files under new names. |
| `research/direct-goal-repro-sweep` | Evidence only, except possible baseline rationale/reward-shaping notebook. |
| `autoresearch/map-ant-12x12-conv-critic` | Evidence only plus possible guardrails; do not port old monolithic JAX code. |
| `origin/vision_shrink_curriculum` | Defer. Its config/notebook story is internally inconsistent. |

## Validation Ladder

Run after each integration phase:

```bash
PYTHONPATH=src python3 -m compileall -q src scripts
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q tests/test_notebook_layout.py tests/test_workflow_experiments.py
```

Before declaring the branch polished:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q -o addopts=''
PYTHONPATH=src python3 -m ant_byte_env.cli train jax --config experiments/direct_goal_baseline.json --dry-run
PYTHONPATH=src python3 -m ant_byte_env.cli train jax --config experiments/map_ant_gated_mlp_curriculum.json --dry-run
```

Also run tiny actual smoke tests for workflow configs whose dry-run does not
exercise nested trainer invocations.
