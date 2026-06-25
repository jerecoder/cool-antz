# Integration Audit

This branch (`repo/research-integration-cleanup`) starts from `origin/main` at
`554ee8d`. Its purpose is to make the repository easier to understand, rerun,
and audit while preserving reproductive experiment behavior.

## Branch Inventory

| Ref | Role | Integration decision |
| --- | --- | --- |
| `origin/main` | Current best-organized base. It already contains grouped notebooks, experiment JSON files, workflow modules, runtime resource checks, curated results, and the archived forage autoresearch report. | Use as the base for integration. |
| `research/direct-goal-repro-sweep` | Older direct-goal/autoresearch line with flat notebook names and direct-goal sweep artifacts. | Selectively port durable direct-goal artifacts only if they are not represented by current notebooks/configs/docs. Do not merge wholesale because it would remove newer workflow and notebook organization. |
| `autoresearch/map-ant-12x12-conv-critic` | Failed/new-critic map-ant autoresearch line. It includes useful evidence and some diagnostics, but did not produce a solved curriculum. | Preserve as evidence. Do not make it a mainline workflow. Port only isolated, tested utilities if they improve maintained experiments without changing the environment or actor information surface. |
| `origin/vision_shrink_curriculum` | Vision-range curriculum experiment branch. | Treat as optional experiment lineage. Port only as a documented config/notebook if later desired; do not change default actor vision or baseline semantics. |
| local `main` | Old local main behind `origin/main`. | Ignore for integration. |

## Guardrails

- Do not simplify the environment to make experiments pass.
- Do not expand default actor observations, add food/hub location hints, or change actor vision radius as part of cleanup.
- Do not silently convert communication experiments into no-write or zero-write tasks.
- Do not delete local generated runs, old branches, videos, checkpoints, or W&B payloads unless explicitly requested.
- Keep generated payloads ignored; preserve configs, summaries, curated indexes, and docs.
- Make refactors behavior-preserving: public CLI commands and old MAPPO import paths must continue to work.

## Current Cleanup State

- Test collection passes under Python 3.10 after replacing `datetime.UTC` with `timezone.utc`.
- `.vscode/` is ignored so editor state does not pollute branch status.
- `autoresearch/REPORT.md` is the durable historical evidence for the forage autoresearch loop; long autoresearch matrices are archival, not the primary user-facing workflow.
- `experiments/*.json` and grouped notebooks are the canonical reproduction surfaces.
- `scripts/` contains transitional launchers for full-layout continuations. Keep them until each has an equivalent documented config command and dry-run validation.

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
