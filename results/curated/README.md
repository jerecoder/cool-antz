# Curated Results

This directory tracks the small, human-readable index of results worth keeping.
Large generated files are not committed; they live under ignored `runs/`.

- `index.json` lists the curated entries and their preserved legacy paths.
- `runs/legacy/checkpoints/` contains pre-cleanup checkpoints and rollout media.
- `runs/legacy/vault/` contains pre-cleanup timestamped vault entries.

For a full metadata scan of legacy runs, use:

```bash
PYTHONPATH=src ant-byte results index runs/legacy results/curated/index.generated.json
```
