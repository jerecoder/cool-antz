# Experiment Tree Mega Audit

Generated: 2026-07-04T19:36:39+00:00

Baseline branch: `report-writing-site`

Baseline HEAD: `21d3cdf` (`21d3cdfb6b9fbb0b01227845701a59be94129bd2`)

Baseline git status before audit files were generated:

```text
## report-writing-site...origin/report-writing-site
```

This directory is the exhaustive evidence index for the current `cool-antz`
checkout. It was generated because the report and GitHub Pages site now mention
many, but not all, experiment branches. The goal here is not a polished paper
story. The goal is a ledger: every reachable commit, every visible branch/ref,
every current experiment config, notebook, run evidence directory, local W&B run,
log, and experiment-relevant file surface gets a family assignment and a path
back to the evidence.

A dirty `report/main.tex` diff was observed at the beginning of the manual audit,
then landed in branch history as commit `21d3cdf` while the audit was running.
The baseline used for these generated indexes is the clean worktree at
`21d3cdf`. Local environment internals such as `.venv/**` are intentionally
excluded from `files.tsv`; they are not experiment evidence. Empty TSV cells are
written as `.` so the generated ledgers stay diff-clean.

## Index Files

| file | rows | meaning |
| --- | --- | --- |
| commits.tsv | 334 | Every reachable commit from all local refs, with family and changed-path summary. |
| commit-files.tsv | 1651 | Every path-level change from every reachable commit. |
| branches.tsv | 13 | Every local/remote/tag ref visible in this checkout, with relationship to HEAD. |
| local-changes-at-baseline.tsv | 1 | Worktree status before this audit directory was generated. |
| files.tsv | 7771 | Every non-.git, non-local-environment file visible in the checkout, excluding the audit output directory itself. |
| experiments.tsv | 25 | Every experiments/*.json config with core contract fields. |
| notebooks.tsv | 17 | Every notebook under notebooks/** with launch/config/W&B hints. |
| runs.tsv | 145 | Every run directory containing summary/evaluation/train/config/plan/metrics evidence. |
| wandb-local-runs.tsv | 230 | Every top-level local W&B run folder present under wandb/run-* paths. |
| logs.tsv | 9 | Every logs/*.log file with first/last non-empty line. |
| wandb-api-status.txt | 1 | Live W&B API status and authentication limitation. |

## Experiment Families

| family | commits | commit-file rows | files | configs | notebooks | run dirs | local W&B runs | logs | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foundation-task-contract | 8 | 16 | 11 | 0 | 0 | 0 | 0 | 0 | Environment birth, task semantics, sprites, random rollout, delivery/depleting-source reset. |
| jax-mappo-training-core | 21 | 44 | 155 | 1 | 0 | 0 | 13 | 0 | JAX environment, MAPPO trainer, checkpoint/eval/workflow plumbing, runtime resources. |
| autocurriculum-exploration-baselines | 29 | 167 | 20 | 4 | 4 | 0 | 0 | 0 | Forage/exploration/autocurriculum notebooks and configs, direct-goal baselines, and early launch surfaces before specialized branches. |
| communication-memory-bytes | 94 | 581 | 327 | 1 | 1 | 10 | 27 | 0 | Communication-bit, byte-memory, write-head, write-cost, no-write, and memory-use investigations. |
| ant-count-25x25-autoresearch | 75 | 187 | 256 | 0 | 1 | 52 | 0 | 0 | 25x25 forage unlock path: distance shaping, ant-count/capacity, greedy/sample deployment, autoresearch CAP/vision lines. |
| source-layout-50x50 | 4 | 25 | 444 | 4 | 4 | 2 | 3 | 0 | 50x50 source geometry: efficient 50x50, padded, proximity, scratch smooth, rare/random spawn, vectors, and layout-density previews. |
| critic-full-layout-50x50 | 29 | 95 | 127 | 6 | 4 | 3 | 0 | 0 | Full-layout 50x50 and spatial critic branch, including shared writes, write cost, 64env, 8-ant continuations. |
| sixty-ant-50x50-stabilization | 6 | 109 | 35 | 4 | 1 | 0 | 0 | 0 | 60-ant 50x50 shared-write 8-bit transfer, identity features, stabilization, speed/time penalty continuations. |
| hundred-bridge-maze-stress | 16 | 100 | 2457 | 1 | 1 | 16 | 173 | 7 | 100x100 bridge/continuation/progress-video work plus maze/labyrinth stress and report-site maze evidence. |
| twofifty-frontier-reset | 15 | 168 | 2755 | 4 | 1 | 25 | 11 | 2 | 250x250 half-scale, frontier, reset-boundary, distance autocurriculum, set/resnet critic, truncation continuations. |
| bigmap-deployment-rendering | 10 | 29 | 593 | 0 | 0 | 6 | 1 | 0 | Large-map actor-only deployment, 1000x1000/bigmap renders, palette/layout helpers, JS sandbox policy runner. |
| adversarial-roles-side-branches | 14 | 107 | 0 | 0 | 0 | 0 | 0 | 0 | Adversarial/frozen-opponent, timed-release cooperative roles, lethal-cookies side branch, branch-only probes. |
| report-site-paper-assets | 8 | 19 | 65 | 0 | 0 | 0 | 0 | 0 | Spanish report, GitHub Pages report site, plots, videos, tree posters, planning chronology, generated figures. |
| infra-tests-ci-cleanup | 3 | 4 | 522 | 0 | 0 | 31 | 2 | 0 | Tests, CI, packaging, cache/storage cleanup, docs scaffolding, non-experiment maintenance. |
| curated-results-and-vault | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | Curated result indexes, vault placeholders, retained artifacts that are evidence rather than active experiments. |
| unclassified-review-needed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Evidence exists but classifier could not confidently place it; inspect manually. |

## Branch And Ref Placement

| ref | kind | commit | family | ahead of HEAD | behind HEAD | merged into HEAD | subject |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feat/multi-device-jax-mappo | local | 6879fa4 | jax-mappo-training-core | 18 | 53 | no | perf: skip nearest-open scan for empty arenas |
| main | local | 5b27fd2 | bigmap-deployment-rendering | 0 | 31 | yes | feat: add bigmap palette rerender helper |
| report-writing-site | local | 21d3cdf | report-site-paper-assets | 0 | 0 | yes | docs: sync Spanish report with site story |
| train-half-scale-resnet | local | 3c0eac2 | twofifty-frontier-reset | 19 | 53 | no | feat: add half-scale resnet training launch |
| origin/HEAD | remote | a6ba235 | critic-full-layout-50x50 | 0 | 52 | yes | feat: add 8-bit shared-write continuation |
| origin/feat/multi-device-jax-mappo | remote | 83d54e7 | jax-mappo-training-core | 15 | 53 | no | feat: add x150 write cost experiment |
| origin/lethal_cookies | remote | 58ea666 | hundred-bridge-maze-stress | 2 | 56 | no | modified:   experiments/exploration_to_forage_proximity_sources_50x50.json  new file:   notebooks/source_layouts/proximity_sources_50x50 ... |
| origin/main | remote | a6ba235 | critic-full-layout-50x50 | 0 | 52 | yes | feat: add 8-bit shared-write continuation |
| origin/repo/research-integration-cleanup | remote | c241433 | jax-mappo-training-core | 7 | 56 | no | test: skip torch mappo tests without torchrl extras |
| origin/report-writing-site | remote | 21d3cdf | report-site-paper-assets | 0 | 0 | yes | docs: sync Spanish report with site story |
| origin/research/adversarial-marl-experiments | remote | 4aad30d | adversarial-roles-side-branches | 16 | 56 | no | Document adversarial frozen-opponent experiments |
| origin/research/timed-release-roles | remote | 61f6647 | adversarial-roles-side-branches | 11 | 56 | no | Park tuned timed-release continuation |
| origin/vision_shrink_curriculum | remote | b955a6f | ant-count-25x25-autoresearch | 3 | 182 | no | fix: update vision range parameters and execution counts in curriculum notebook |

## How To Read This

- `commits.tsv` is the commit-level chronology. It answers "where does this commit belong?"
- `commit-files.tsv` is the path-level change ledger. It answers "what did this commit touch?"
- `files.tsv` is the current artifact surface. It includes ignored/generated experiment files such as `runs/**` and local W&B folders, but excludes `.git`, local virtualenv internals, and this audit output directory.
- `experiments.tsv` is the experiment contract table. It is the fastest way to compare ants, grid size, food, bits, critic architecture, checkpoint lineage, and W&B naming.
- `runs.tsv` indexes local evidence directories that contain summaries, evaluations, configs, plans, markdown, train results, or metric streams.
- `wandb-local-runs.tsv` indexes local W&B run folders only. Live cloud listing is not complete because the SDK reported `relogin required`; see `wandb-api-status.txt`.

## What This Audit Proves

- The repo now has 334 reachable commits across visible refs, not the older 257 commits recorded in the stale chronology header.
- The current checkout has 25 experiment JSON configs, 17 notebooks, 145 local run evidence directories, 230 local W&B run folders, 9 log files, and 7771 indexed experiment-relevant files outside `.git`, `.venv`, and this audit output.
- Branch-only lines are real evidence surfaces: adversarial/frozen-opponent, timed-release roles, lethal-cookies maze/radius probes, multi-device/write-cost work, half-scale resnet launch, and report-site history all remain visible through refs even when they are not part of the main report narrative.
- W&B cloud exhaustiveness is currently blocked by authentication. The local W&B evidence is indexed, but a future pass should run after `wandb login`/`wandb relogin` and regenerate a cloud-run table.

## Interpretation Guardrails

Many rows are not clean ablations. Checkpoint lineage, optimizer state, critic
architecture, ant count, source geometry, write semantics, temperature, and
selection metric often move together. The `family` column is a placement in the
experiment tree, not causal proof.

Rows classified as `unclassified-review-needed` should be inspected manually
before using them in the paper or site. They are deliberately not hidden.
