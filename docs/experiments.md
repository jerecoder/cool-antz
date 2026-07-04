# Experiment Index

This is the source-of-truth index for committed experiment surfaces. Generated
checkpoints, videos, W&B payloads, and raw logs remain under ignored `runs/`.
Committed JSON configs, notebooks, report figures, and planning ledgers are the
auditable reproduction surface.

## How To Read Metrics

- `env_return` is the environment delivery reward. Use it, delivery counts, and
  `eval_mean_delivered_fraction` for paper-facing task performance.
- `episode_return` may include trainer-side shaping such as pickup, distance,
  novelty, border, write-cost, or terminal bonuses. It is useful for debugging,
  but it is not interchangeable with deliveries.
- `experiments/direct_goal_baseline.json` is the main negative research
  baseline. `experiments/smoke.json` is only an engineering plumbing check.
- Byte communication is not claimed causal unless a matched write/no-write or
  write-cost ablation supports that claim.

## Experiment Configs

| Config | Family | Status | Source | Notebook / Entry Point | Evidence / Caveat |
| --- | --- | --- | --- | --- | --- |
| `experiments/smoke.json` | Smoke | Engineering check | Cleanup base | `ant-byte train torch ... --dry-run` | Torch plumbing only; not a research baseline. |
| `experiments/direct_goal_baseline.json` | Baseline | Maintained negative baseline | Cleanup base plus direct-goal evidence | `notebooks/baselines/direct_goal.ipynb` | Sparse final target from scratch; compare deliveries/`env_return`, not shaped return. |
| `experiments/forage_curriculum.json` | Curriculum | Maintained | Cleanup base | `notebooks/curriculum/forage.ipynb` | Staged map-size curriculum. |
| `experiments/autocurriculum.json` | Autocurriculum | Maintained | Cleanup base | `notebooks/curriculum/autocurriculum.ipynb` | Active-grid growth after deliveries. |
| `experiments/exploration_curriculum.json` | Exploration | Maintained | Cleanup base | `notebooks/curriculum/exploration.ipynb` | Coverage objective, not delivery proof. |
| `experiments/maze_exploration_curriculum.json` | Maze | Maintained pipeline | Cleanup base | `notebooks/curriculum/maze_exploration.ipynb`; `notebooks/source_layouts/proximity_sources_50x50 maze.ipynb` | Geometry/obstacle pipeline evidence; no comparable final delivery metrics are claimed. |
| `experiments/communication_bits.json` | Communication | Maintained | Cleanup base | `notebooks/communication/bit_curriculum.ipynb` | Warm-started communication-bit curriculum. |
| `experiments/map_ant_gated_mlp_curriculum.json` | Map-ant | Historical, runnable workflow | `research/direct-goal-repro-sweep` | `metadata.workflow=map_ant_gated_curriculum` | Preserves old MLP gated progress; stage argv translates W&B/reset flags and does not forward unsupported shaping flags. |
| `experiments/exploration_to_forage_50x50.json` | Exploration-to-forage | Maintained | Cleanup base | `notebooks/exploration_to_forage/base_50x50.ipynb` | Warm-start bridge into delivery. |
| `experiments/exploration_to_forage_padded_sources_50x50.json` | Source layout | Maintained | Cleanup base | `notebooks/source_layouts/padded_sources_50x50.ipynb` | Padded hidden-arena source-count curriculum. |
| `experiments/exploration_to_forage_proximity_sources_50x50.json` | Source layout | Maintained positive-only run | Cleanup base | `notebooks/source_layouts/proximity_sources_50x50.ipynb` | Source proximity/footprint aid; no lethal cookies. |
| `experiments/exploration_to_forage_proximity_sources_lethal_cookies_50x50.json` | Lethal cookies | Pipeline/geometry evidence | `origin/lethal_cookies@58ea666` | `notebooks/source_layouts/proximity_sources_50x50 cookie radius.ipynb` | Lethal food is visible as food but hidden as a channel; comparable success metrics are not claimed. |
| `experiments/exploration_to_forage_scratch_smooth_sources_50x50.json` | Source layout | Maintained | Cleanup base | `notebooks/source_layouts/scratch_smooth_sources_50x50.ipynb` | Scratch smooth-source annealing in padded 80x80/50x50 task window. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50.json` | Scaling | Maintained continuation | Cleanup base | `notebooks/scaling/full_layout_8ants_half_food_50x50.ipynb` | Full-layout 8-ant half-food continuation. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_64env.json` | Scaling | Maintained continuation | Cleanup base | same family | 64-env continuation; compare wall-clock separately. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes.json` | Shared writes | Maintained continuation | Cleanup base | `notebooks/scaling/full_layout_8ants_half_food_shared_writes_50x50.ipynb` | Shared write values; not causal alone. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost.json` | Write cost | Imported evidence | `origin/report-writing-site@bd1f6ee` | `notebooks/scaling/full_layout_8ants_half_food_shared_writes_write_cost_50x50.ipynb` | Continuation with write cost. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_8bits.json` | Write bits/cost | Imported evidence | `origin/report-writing-site@bd1f6ee` | `notebooks/scaling/full_layout_8ants_half_food_shared_writes_write_cost_8bits_50x50.ipynb` | 8-bit shared-write/write-cost continuation. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x5.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Multiplier sweep; use as ablation evidence only with matched context. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x10.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Same sweep family. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x50.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Same sweep family. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x100.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Same sweep family. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x150.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Same sweep family. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x200.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Same sweep family. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x500.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Same sweep family. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x1000.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Same sweep family. |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes_write_cost_x10000.json` | Write-cost sweep | Imported evidence | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Same sweep family. |
| `experiments/exploration_to_forage_full_layout_16ants_half_food_8types_50x50_from_shared_writes.json` | Ant/type scaling | Maintained continuation | Cleanup base | scaling family | 16 ants with 8 per-ant write-channel types. |
| `experiments/exploration_to_forage_full_layout_16ants_half_food_8types_50x50_multidevice.json` | Multi-device scaling | Integrated, hardware dependent | `origin/feat/multi-device-jax-mappo@83d54e7` | JSON only | Requires requested local JAX devices; runner supports `--jax-parallelism=data`. |
| `experiments/exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_from_best.json` | 60-ant scaling | Imported report evidence | `origin/report-writing-site@bd1f6ee` | `notebooks/scaling/full_layout_60ants_half_food_shared_writes_write_cost_8bits_50x50.ipynb` | Large-scale continuation; cite report ledger before making claims. |
| `experiments/exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_stabilize_from_60best.json` | 60-ant scaling | Imported report evidence | `origin/report-writing-site@bd1f6ee` | same family | Stabilization continuation. |
| `experiments/exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_speed_shaping_from_60best.json` | 60-ant scaling | Imported report evidence | `origin/report-writing-site@bd1f6ee` | same family | Adds speed shaping; shaped return is not delivery. |
| `experiments/exploration_to_forage_full_layout_60ants_half_food_50x50_shared_writes_write_cost_8bits_time_penalty_from_stabilized_best.json` | 60-ant scaling | Imported report evidence | `origin/report-writing-site@bd1f6ee` | same family | Time-penalty continuation. |
| `experiments/half_scale_distance_autocurriculum_250x250_healthy_reset.json` | 250x250 distance | Imported report evidence | `origin/report-writing-site@bd1f6ee` | `notebooks/curriculum/distance_autocurriculum_250x250.ipynb` | Healthy reset baseline for half-scale diagnostics. |
| `experiments/half_scale_distance_250x250_reasonable_truncation_from_frontier.json` | 250x250 distance | Imported report evidence | `origin/report-writing-site@bd1f6ee` | same family | Truncation/continuation diagnostic. |
| `experiments/half_scale_distance_250x250_d16_trunc4096_from_685best.json` | 250x250 distance | Imported report evidence | `origin/report-writing-site@bd1f6ee` | same family | Distance-16 diagnostic from frontier checkpoint. |
| `experiments/half_scale_distance_250x250_soft_d8d16_trunc4096_from_685best.json` | 250x250 distance | Imported report evidence | `origin/report-writing-site@bd1f6ee` | same family | Soft d8/d16 diagnostic. |
| `experiments/timed_release_roles_8ants_shared_writes.json` | Timed release | Integrated L4 continuation | `origin/research/timed-release-roles@38cf3d8` | `notebooks/timed_release/roles.ipynb`; `metadata.workflow=timed_release_roles` | Final profile: 128 envs, 256 steps, 16,384,000 base timesteps, strided CNN critic, train temp 0.75, eval/render temp 0.52. |
| `experiments/adversarial_frozen_opponent_probe.json` | Adversarial | Maintained probe | `research/adversarial-marl-experiments` | `notebooks/adversarial/capability_audit.ipynb`; `metadata.workflow=adversarial_frozen_opponent` | Fast capability audit. |
| `experiments/adversarial_frozen_opponent_shared_writes_8ants.json` | Adversarial | Maintained workflow | `research/adversarial-marl-experiments` | `notebooks/adversarial/frozen_opponent.ipynb`; `metadata.workflow=adversarial_frozen_opponent` | Frozen-opponent actor warm start; evaluate as adversarial reward, not cooperative delivery. |
| `experiments/vision_range_curriculum.json` | Vision shrink | Exploratory/deferred | `origin/vision_shrink_curriculum@b955a6f` | `notebooks/vision/vision_range_curriculum.ipynb` | Config cleaned to live JAX CLI; no strong final evidence recovered. |
| `experiments/vision_range_curriculum_12ret.json` | Vision shrink | Exploratory/deferred | `origin/vision_shrink_curriculum@b955a6f` | same notebook | Alternate 12-source exploratory return profile. |

## Report Evidence

- `docs/planning/complete-experiment-chronology.md` and
  `docs/planning/experiment-history-analysis.md` are the imported evidence
  ledgers from `origin/report-writing-site@bd1f6ee`.
- `report/` contains the paper draft, figure data, figure generator, and
  rendered figures used by the report site.
- `docs/index.html` plus `docs/report-site/` are the static report/codespace
  presentation assets. They should reference committed assets only.

## Validation Commands

```bash
python3 -m compileall -q src
git diff --check
PYTHONPATH=src python3 -m ant_byte_env.cli train jax --config experiments/direct_goal_baseline.json --dry-run
PYTHONPATH=src python3 -m ant_byte_env.cli train jax --config experiments/timed_release_roles_8ants_shared_writes.json --dry-run
PYTHONPATH=src python3 -m ant_byte_env.cli train jax --config experiments/map_ant_gated_mlp_curriculum.json --dry-run
```
