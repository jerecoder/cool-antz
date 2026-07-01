# Notebooks

Notebook filenames are short and grouped by workflow. Open notebooks from the
repo root or any notebook subfolder; each notebook resolves the project root by
walking up to `pyproject.toml`.

## Curriculum

- `curriculum/forage.ipynb` - main staged forage curriculum.
- `curriculum/exploration.ipynb` - exploration-only curriculum.
- `curriculum/maze_exploration.ipynb` - maze exploration curriculum.
- `curriculum/autocurriculum.ipynb` - single-policy autocurriculum.

## Exploration To Forage

- `exploration_to_forage/base_50x50.ipynb` - base 50x50 transfer workflow.

## Communication

- `communication/bit_curriculum.ipynb` - communication-bit curriculum.

## Source Layouts

- `source_layouts/padded_sources_50x50.ipynb` - padded source layout workflow.
- `source_layouts/proximity_sources_50x50.ipynb` - proximity source workflow.
- `source_layouts/scratch_smooth_sources_50x50.ipynb` - scratch smooth-source workflow.

## Scaling

- `scaling/ant_count_curriculum.ipynb` - ant-count scaling from a communication checkpoint.
- `scaling/full_layout_8ants_half_food_50x50.ipynb` - larger 8-ant full-layout experiment.
- `scaling/full_layout_8ants_half_food_shared_writes_50x50.ipynb` - 8-ant full-layout continuation with unrestricted write values for every ant.
- `scaling/full_layout_8ants_half_food_shared_writes_write_cost_50x50.ipynb` - shared-write continuation with a small trainer-side cost for set write bits.
- `scaling/full_layout_8ants_half_food_shared_writes_write_cost_8bits_50x50.ipynb` - 8-bit shared-write continuation from the best write-cost checkpoint.

## Baselines

- `baselines/direct_goal.ipynb` - direct-goal baseline.
