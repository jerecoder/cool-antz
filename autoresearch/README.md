# Autoresearch

Autoresearch is a small, legible loop for turning AntByte hunches into checked
experiments. The goal is not to build a giant agent framework. The goal is to
make it easy to ask a concrete research question, run the smallest useful
experiment, inspect the evidence, and decide what to try next.

The Karpathy-inspired part is the taste: plain files, simple commands, visible
state, and very little magic. Prefer one hackable script or notebook cell over a
deep abstraction. Prefer a tiny table of numbers over a dashboard nobody reads.

## Loop

1. Write the hypothesis before running anything.
2. Pick the smallest config change that could falsify it.
3. Run one bounded experiment.
4. Save the exact command, checkpoint path, metrics, and rollout artifact.
5. Decide: keep, revert, or mutate the idea.

## Folder Map

- `ideas/` contains research notes. Each note should name the problem, the
  current evidence, candidate interventions, and the next experiment.
- `templates/experiment.md` is the copy-paste template for a single run.
- `protocol.md` is the operating checklist for manual or agent-assisted runs.

Generated checkpoints, videos, metrics, and large logs should stay under
`runs/`, `results/`, or `vault/`, not in this folder.

## Current North Star

Make communication in AntByte measurable and useful. A communication method is
interesting only if it improves forage behavior, survives deterministic rollout,
and uses the available writable bits for a reason we can explain.
