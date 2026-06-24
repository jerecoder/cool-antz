# MIT Course Mainline Plan

Goal: make the repository understandable and defensible for a course
presentation without carrying every autoresearch detour into the main story.

## Publish Strategy

Use the current `autoresearch` branch as the working cleanup branch. Before
publishing to `main`, keep only the parts that help explain or reproduce the
project:

- core environment and JAX MAPPO implementation;
- reusable workflow modules under `src/ant_byte_env/workflows/`;
- curriculum builders under `src/ant_byte_env/curricula/`;
- thin notebooks and experiment JSON files that demonstrate the final story;
- curated result summaries, especially `autoresearch/REPORT.md`.

Treat the active autoresearch loop as archival. The loop was useful for
exploration, but it is not the course-facing API. If this branch is merged or
pushed to `main`, the presentation should frame autoresearch as prior search
work, not as the project architecture.

## Course-Facing Story

The clean story is:

1. AntByte is a cooperative foraging environment with local observations,
   optional byte-grid communication, and increasingly sparse food layouts.
2. The original single-ant curriculum learns small maps but does not scale
   cleanly to large sparse layouts.
3. Multi-ant distance-shaped MAPPO solves the 25x25 gate when movement is
   sampled, showing route knowledge exists.
4. Deterministic deployment and rare-source 50x50 remain open problems.
5. The cleaned code separates environment/training logic from notebook
   orchestration so experiments can be explained without a giant helper file.

## What To Discard Or De-Emphasize

- Long autoresearch matrices as presentation material.
- Raw W&B/local run payloads, checkpoints, and generated media unless curated.
- One-off failed experiment branches that are already summarized in
  `autoresearch/REPORT.md`.
- The idea that 50x50 is solved; the current evidence supports 25x25 much more
  strongly.

## Next Cleanup Targets

1. Split `notebook_workflows.py` into orchestration modules:
   `forage`, `communication`, `rendering`, and `ant_count_transfer`.
2. Split `research_loop.py` into archived planning/ranking utilities or remove
   it from the course-facing path after the report is preserved.
3. Break `training/jax_mappo/core.py` by ownership: observations, model
   forward passes, rewards, losses, rollout collection, and transfer.
4. Split giant tests to match those modules so the code can be explained and
   modified without touching 4000-line test files.
