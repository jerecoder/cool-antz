# Integration Audit

This branch (`repo/research-integration-cleanup`) was cut from `origin/main` at
`554ee8d`. The current merge-readiness audit fetched `origin/main` at `a6ba235`,
so the branch still needs a reconciliation pass before it can land on main. Its
purpose is to make the repository easier to understand, rerun, and audit while
preserving reproducible experiment behavior.

## Branch Inventory

| Ref | Role | Integration decision |
| --- | --- | --- |
| `origin/main` | Current best-organized base. In this audit it resolves to `a6ba235`, four commits beyond the branch fork point. It already contains grouped notebooks, experiment JSON files, workflow modules, runtime resource checks, curated results, and the archived forage autoresearch report. | Reconcile before merge; keep the newer write-cost continuation and render/test updates unless a conflict proves they duplicate this branch's newer surfaces. |
| `research/direct-goal-repro-sweep` | Older direct-goal/autoresearch line with flat notebook names, direct-goal sweep artifacts, and the original gated map-ant MLP curriculum. | Selectively port durable artifacts only. The gated map-ant MLP curriculum is preserved as a documented historical experiment; do not merge wholesale because this branch would remove newer workflow and notebook organization. |
| `autoresearch/map-ant-12x12-conv-critic` | Failed/new-critic map-ant autoresearch line. It includes useful evidence and some diagnostics, but did not produce a solved curriculum. | Preserve as branch evidence. Do not make its autoresearch loop a mainline workflow. Keep the maintained map-growth and ant-scaling experiments in `experiments/` and `notebooks/`; port only isolated, tested utilities if they improve those workflows without changing the environment or actor information surface. |
| `origin/vision_shrink_curriculum` | Vision-range curriculum experiment branch. | Ported as exploratory configs plus a grouped notebook. It remains deferred evidence; do not change default actor vision or baseline semantics. |
| local `main` | Old local main behind `origin/main`. | Ignore for integration. |

## Integration Head Ledger

These refs were fetched before the final integration pass on
`repo/research-integration-cleanup`.

| Ref | Head used | Imported surface |
| --- | --- | --- |
| `origin/report-writing-site` | `bd1f6ee` | Report evidence ledgers, `report/`, `docs/report-site/`, `docs/index.html`, 60-ant 50x50, 100x100/report assets, and 250x250 distance diagnostics. |
| `origin/research/timed-release-roles` | `38cf3d8` | Timed-release roles workflow, config, notebook, docs, and tests. |
| `origin/feat/multi-device-jax-mappo` | `83d54e7` | Data-parallel helper/configs and write-cost multiplier sweep configs. |
| `origin/vision_shrink_curriculum` | `b955a6f` | Exploratory vision-shrink configs and notebook. |
| `origin/lethal_cookies` | `58ea666` | Lethal-cookie geometry config/notebooks and JAX env support. |
| `research/adversarial-marl-experiments` | `4aad30d` | Frozen-opponent adversarial workflow, configs, notebooks, docs, and tests. |
| `research/direct-goal-repro-sweep` | `5ccced7` | Historical gated map-ant workflow evidence and direct-goal/map-ant docs. |
| `autoresearch/map-ant-12x12-conv-critic` | `cc7c499` | 12x12 conv-critic/autoresearch evidence docs and sweep scripts only. |

## Guardrails

- Do not simplify the environment to make experiments pass.
- Do not expand default actor observations, add food/hub location hints, or change actor vision radius as part of cleanup.
- Do not silently convert communication experiments into no-write or zero-write tasks.
- Do not delete local generated runs, old branches, videos, checkpoints, or W&B payloads unless explicitly requested.
- Keep generated payloads ignored; preserve configs, summaries, curated indexes, and docs.
- Make refactors behavior-preserving: public CLI commands and old MAPPO import paths must continue to work.

## Current Cleanup State

- Full test validation passes under Python 3.10 with
  `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q`.
- `.vscode/` is ignored so editor state does not pollute branch status.
- `autoresearch/REPORT.md` is the durable historical evidence for the forage autoresearch loop; long autoresearch matrices are archival, not the primary user-facing workflow.
- `experiments/*.json` and grouped notebooks are the canonical reproduction surfaces.
- Committed experiment JSON files and notebooks are the reproduction surface.
  Small scripts may remain for report assets, checkpoint transfer, or one-shot
  audits when they are cited by docs; raw generated payloads stay ignored.

## Current Merge Readiness

As of `origin/main@a6ba235` and branch head `8090fd8`, this branch is not a
clean merge yet. The integration pass must resolve conflicts in:

- `notebooks/README.md`
- `notebooks/scaling/full_layout_8ants_half_food_shared_writes_write_cost_8bits_50x50.ipynb`
- `scripts/render_jax_channel_grid_video.py`
- `src/ant_byte_env/env.py`
- `tests/test_render.py`

After resolving those conflicts, rerun the validation ladder before treating the
branch as merge-ready.

## Next Integration Decisions

1. Keep the gated map-ant MLP curriculum as a historical experiment, not as the
   failed new-critic autoresearch loop. Its final surface is a clear
   config/notebook plus a tested workflow entrypoint.
2. Preserve its claims accurately: it is evidence that the old MLP critic made
   real gated progress with growing maps and ant counts, but it is not a solved
   50x50 result.
3. Keep the failed `autoresearch/map-ant-12x12-conv-critic` branch as evidence
   unless one isolated utility is worth porting.
4. Keep `training/jax_mappo/core.py` as a compatibility facade while the owned
   modules remain the primary implementation surface.

## MAPPO Organization Finding

The JAX MAPPO implementation is not throwaway code: it has tests, checkpoint
compatibility, evaluation modes, and clean runner/rollout/evaluation boundaries.
The main maintenance problem is concentration of concerns:

- `training/jax_mappo/core.py` owns types, observations, model init/forward,
  action distributions, reward shaping, GAE, optimizer logic, and PPO update.
- `training/jax_mappo/transfer.py` owns many checkpoint adaptation cases in one
  large module.
- CLI flags encode years of experiment history and should remain stable, but
  should be grouped internally for readability.

The integration branch should split these concerns while keeping compatibility
facades so notebooks, tests, and old scripts do not break.
