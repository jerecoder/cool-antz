# Integration Audit

This branch (`repo/research-integration-cleanup`) starts from `origin/main` at
`554ee8d`. Its purpose is to make the repository easier to understand, rerun,
and audit while preserving reproductive experiment behavior.

## Branch Inventory

| Ref | Role | Integration decision |
| --- | --- | --- |
| `origin/main` | Current best-organized base. It already contains grouped notebooks, experiment JSON files, workflow modules, runtime resource checks, curated results, and the archived forage autoresearch report. | Use as the base for integration. |
| `research/direct-goal-repro-sweep` | Older direct-goal/autoresearch line with flat notebook names, direct-goal sweep artifacts, and the original gated map-ant MLP curriculum. | Selectively port durable artifacts only. The gated map-ant MLP curriculum should become a documented historical experiment; do not merge wholesale because this branch would remove newer workflow and notebook organization. |
| `autoresearch/map-ant-12x12-conv-critic` | Failed/new-critic map-ant autoresearch line. It includes useful evidence and some diagnostics, but did not produce a solved curriculum. | Preserve as branch evidence. Do not make its autoresearch loop a mainline workflow. Keep the maintained map-growth and ant-scaling experiments in `experiments/` and `notebooks/`; port only isolated, tested utilities if they improve those workflows without changing the environment or actor information surface. |
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
- One-off continuation launchers were removed from `scripts/`; committed experiment JSON files and notebooks are the reproduction surface.

## Next Integration Decisions

1. Port the gated map-ant MLP curriculum as a historical experiment, not as the
   failed new-critic autoresearch loop. The final surface should be a clear
   config/notebook plus a small tested workflow entrypoint if needed.
2. Preserve its claims accurately: it is evidence that the old MLP critic made
   real gated progress with growing maps and ant counts, but it is not a solved
   50x50 result.
3. Keep the failed `autoresearch/map-ant-12x12-conv-critic` branch as evidence
   unless one isolated utility is worth porting.
4. Continue MAPPO cleanup by moving real ownership boundaries, then remove old
   import shells once notebooks/tests have migrated. Do not add compatibility
   files that exist only to hide obsolete organization.

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
