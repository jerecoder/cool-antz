# Forage Improvement Autoresearch

This folder contains the active autoresearch loop for improving JAX MAPPO forage performance at the 25x25 gate. Older communication, no-cheat 50x50, and one-off sweep matrices have been removed from the active interface so new runs use one comparable experiment ledger.

## Commands

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

## Artifacts

Each run writes under `runs/autoresearch/forage_loop/<experiment-id>/`:

- `experiment.md`: hypothesis, intervention, baseline, evaluation gate, and report notes
- `plan.json`: fully resolved executable settings
- `evaluation.json`: deterministic and sampled held-out checkpoint evaluation
- `summary.json`: training result, evaluation result, and artifact pointers

The autonomous controller also writes `ledger.json` and append-only `ledger.jsonl` under the loop root. W&B receives the training metrics plus research-plan artifacts; the sidecar ledger run logs the summary/evaluation files and flattened evaluation metrics.
