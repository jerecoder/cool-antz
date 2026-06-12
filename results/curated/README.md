# Curated Results

This directory tracks the small, human-readable index of results worth keeping.
Large generated files are not committed; they live under ignored `runs/`.

- `index.json` lists curated entries and the run paths they came from.
- `runs/<experiment>/<run_id>/` contains generated configs, metrics,
  checkpoints, and media.

For a metadata scan of generated runs, use:

```bash
PYTHONPATH=src ant-byte results index runs results/curated/index.generated.json
```
