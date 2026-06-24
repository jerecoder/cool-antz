# Archived Forage Autoresearch

This folder preserves the autoresearch evidence trail. It is not the main
presentation surface for the MIT course version of this project.

The durable takeaway lives in `autoresearch/REPORT.md`: the original single-ant
curriculum did not scale cleanly, distance-shaped multi-ant policies solved the
25x25 gate under sampled movement, and rare-source 50x50 remains unsolved. Most
of the experiment machinery and generated payloads can be treated as archival
after those claims are copied into the presentation path.

For the course repo, prefer:

- thin notebooks that run one clear experiment or visualization;
- stable source modules under `src/ant_byte_env/`;
- curated results and short claims;
- `autoresearch/REPORT.md` as the historical summary.

Do not treat the live loop matrix as required course infrastructure unless new
experiments are explicitly needed.

## Historical Commands

Plan one experiment:

```bash
PYTHONPATH=src ant-byte autoresearch loop-plan --id DISTANCE_SHAPE --wandb-mode disabled
```

Run one experiment:

```bash
PYTHONPATH=src ant-byte autoresearch loop-run --id DISTANCE_SHAPE --wandb-mode online
```

Run pending experiments in priority order:

```bash
PYTHONPATH=src ant-byte autoresearch loop-auto --max-runs 2 --wandb-mode online
```

Rank completed runs:

```bash
PYTHONPATH=src ant-byte autoresearch loop-rank
```

## Artifacts To Keep

Each run writes under `runs/autoresearch/forage_loop/<experiment-id>/`:

- `experiment.md`: hypothesis, intervention, baseline, evaluation gate, and report notes
- `plan.json`: fully resolved executable settings
- `evaluation.json`: deterministic and sampled held-out checkpoint evaluation
- `summary.json`: training result, evaluation result, and artifact pointers

The autonomous controller also writes `ledger.json` and append-only
`ledger.jsonl` under the loop root. W&B receives the training metrics plus
research-plan artifacts; the sidecar ledger run logs the summary/evaluation
files and flattened evaluation metrics.

The generated directories under `runs/autoresearch/**/checkpoints`,
`runs/autoresearch/**/wandb`, and `runs/autoresearch/**/media` are disposable
once the small JSON/Markdown records and `REPORT.md` have been preserved.
