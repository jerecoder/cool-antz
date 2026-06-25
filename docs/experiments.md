# Experiment Reproduction Map

This file is the repo-facing index for rerunning and interpreting maintained
experiments. Generated checkpoints, media, and W&B payloads live under ignored
`runs/`; committed files should be configs, notebooks, summaries, and curated
metadata.

## How To Run

Validate a config without training:

```bash
PYTHONPATH=src ant-byte train jax --config experiments/forage_curriculum.json --dry-run
PYTHONPATH=src ant-byte train torch --config experiments/smoke.json --dry-run
```

Run a maintained notebook from the repo root or a notebook subfolder. Each
notebook resolves the project root by walking up to `pyproject.toml`.

```bash
jupyter notebook notebooks/curriculum/forage.ipynb
```

## Maintained Configs

| Config | Notebook | Purpose | Artifacts |
| --- | --- | --- | --- |
| `experiments/smoke.json` | none | Tiny Torch MAPPO plumbing run. | `runs/smoke/...` |
| `experiments/direct_goal_baseline.json` | `notebooks/baselines/direct_goal.ipynb` | Sparse final-target JAX baseline: 50x50, 10 ants, 5 write bits, randomized food/hub. | `runs/notebooks/direct_goal_baseline/...` or explicit run root |
| `experiments/forage_curriculum.json` | `notebooks/curriculum/forage.ipynb` | Main staged forage curriculum from 4x4 to 50x50 with radius-1 actor vision and moving writes. | `runs/notebooks/forage_curriculum/...` |
| `experiments/autocurriculum.json` | `notebooks/curriculum/autocurriculum.ipynb` | Single-policy autocurriculum that grows the active grid after deliveries. | `runs/notebooks/autocurriculum/...` |
| `experiments/exploration_curriculum.json` | `notebooks/curriculum/exploration.ipynb` | Exploration-only curriculum with coverage rewards. | `runs/notebooks/exploration_curriculum/...` |
| `experiments/maze_exploration_curriculum.json` | `notebooks/curriculum/maze_exploration.ipynb` | Exploration curriculum with generated wide-corridor maze obstacles. | `runs/notebooks/maze_exploration_curriculum/...` |
| `experiments/communication_bits.json` | `notebooks/communication/bit_curriculum.ipynb` | Communication-bit curriculum warm-started from the forage checkpoint. | `runs/notebooks/communication_bits/...` |
| `experiments/exploration_to_forage_50x50.json` | `notebooks/exploration_to_forage/base_50x50.ipynb` | Warm-started exploration-to-forage curriculum into 50x50 delivery. | `runs/notebooks/exploration_to_forage_50x50...` |
| `experiments/exploration_to_forage_padded_sources_50x50.json` | `notebooks/source_layouts/padded_sources_50x50.ipynb` | 50x50 padded hidden arena with source-count curriculum inside a smaller task window. | `runs/notebooks/exploration_to_forage_padded_sources_50x50/...` |
| `experiments/exploration_to_forage_proximity_sources_50x50.json` | `notebooks/source_layouts/proximity_sources_50x50.ipynb` | Positive-only proximity/source-footprint curriculum in a 50x50 arena. | `runs/notebooks/exploration_to_forage_proximity_sources...` |
| `experiments/exploration_to_forage_scratch_smooth_sources_50x50.json` | `notebooks/source_layouts/scratch_smooth_sources_50x50.ipynb` | Scratch smooth-source annealing in an 80x80 padded arena with 50x50 task window. | `runs/notebooks/exploration_to_forage_scratch_smooth_sources...` |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50.json` | `notebooks/scaling/full_layout_8ants_half_food_50x50.ipynb` | Full-layout 8-ant half-food continuation on unrestricted 50x50 random layouts. | `runs/notebooks/exploration_to_forage_proximity_sources_full_layout...` |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_64env.json` | same family | 64-env continuation of the full-layout 8-ant half-food run. | `runs/notebooks/...64env...` |
| `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50_shared_writes.json` | `notebooks/scaling/full_layout_8ants_half_food_shared_writes_50x50.ipynb` | Full-layout continuation with unrestricted shared write values. | `runs/notebooks/...shared_writes...` |
| `experiments/exploration_to_forage_full_layout_16ants_half_food_8types_50x50_from_shared_writes.json` | scaling family | 16-ant continuation with 8 per-ant write-channel types from the shared-write checkpoint. | `runs/notebooks/...16ants_half_food_8types...` |

## Scripts

| Script | Status | Replacement path |
| --- | --- | --- |
| `scripts/run_full_layout_8ants_half_food_continuation.py` | Transitional launcher for the full-layout half-food config family. | Keep until dry-run parity is documented through `experiments/exploration_to_forage_full_layout_8ants_half_food_50x50.json`. |
| `scripts/run_full_layout_proximity_continuation.py` | Transitional launcher for a proximity-source continuation from a best checkpoint. | Keep until represented as a maintained experiment JSON or explicitly archived. |

## Results And Claims

- `autoresearch/REPORT.md` is the durable summary of the autoresearch search:
  the single-ant sparse scale-up did not solve large maps; distance-shaped
  multi-ant MAPPO solved the clean 25x25 sampled-movement gate; rare-source
  randomized 50x50 remains unsolved.
- `results/curated/index.json` stores small committed metadata for selected
  artifacts. Large source files remain under ignored `runs/`.
- To regenerate a metadata scan from local runs:

```bash
PYTHONPATH=src ant-byte results index runs results/curated/index.generated.json
```
