# Forage Improvement Autoresearch

This folder contains the active autoresearch loop. Older communication and no-cheat 50x50 sweep files were intentionally removed so new runs do not mix stale assumptions with the current goal.

## Commands

Plan an experiment:

```bash
PYTHONPATH=src ant-byte autoresearch loop-plan --id CAPACITY4 --wandb-mode disabled
```

Run an experiment:

```bash
PYTHONPATH=src ant-byte autoresearch loop-run --id CAPACITY4 --wandb-mode online
```

Rank completed runs:

```bash
PYTHONPATH=src ant-byte autoresearch loop-rank
```

Each run writes:

- `experiment.md`: conceptual hypothesis, intervention, baseline, and report notes
- `plan.json`: resolved executable settings
- `summary.json`: training result and artifact pointers

For forage-curriculum experiments, `experiment.md` and `plan.json` are also attached to the same W&B run as research-plan artifacts.
