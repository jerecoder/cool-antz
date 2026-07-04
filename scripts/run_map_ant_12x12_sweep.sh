#!/usr/bin/env bash
set -euo pipefail

PHASE="${PHASE:-screen}"
MATRIX="${MATRIX:-autoresearch/map_ant_12x12_sweep.json}"
RUN_ROOT="${RUN_ROOT:-runs/autoresearch/map_ant_12x12}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

RUN_ARGS=()
if [[ "${ALLOW_DIRTY:-0}" == "1" ]]; then
  RUN_ARGS+=(--allow-dirty)
fi
if [[ "${RERUN_COMPLETED:-0}" == "1" ]]; then
  RUN_ARGS+=(--rerun-completed)
fi
if [[ "${SKIP_RESOURCE_CHECK:-0}" == "1" ]]; then
  RUN_ARGS+=(--skip-resource-check)
fi
if [[ "${RENDER_ROLLOUTS:-0}" == "1" ]]; then
  RUN_ARGS+=(--render-rollouts)
fi

RENDER_ARGS=()
if [[ -n "${MAX_RENDER_FRAMES:-}" ]]; then
  RENDER_ARGS+=(--max-render-frames "${MAX_RENDER_FRAMES}")
fi
if [[ -n "${TILE_SIZE:-}" ]]; then
  RENDER_ARGS+=(--tile-size "${TILE_SIZE}")
fi

TRACK_ARGS=()
if [[ "${WANDB:-0}" == "1" || "${WANDB_TRACK:-0}" == "1" ]]; then
  TRACK_ARGS+=(--track)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  TRACK_ARGS+=(--wandb-project-name "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  TRACK_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_GROUP:-}" ]]; then
  TRACK_ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_MODE:-}" ]]; then
  TRACK_ARGS+=(--wandb-mode "${WANDB_MODE}")
fi

IDS=("$@")
if [[ "${#IDS[@]}" -eq 0 ]]; then
  case "${PHASE}" in
    screen)
      IDS=(
        M000-control
        M010-long-lowent
        M020-visible-carry
        M030-urgency-completion
        M040-write-band-overwrite
        M050-explore-temp
      )
      ;;
    promotion)
      echo "Promotion requires explicit IDs from the ranked top candidates." >&2
      echo "Example: PHASE=promotion $0 M020-visible-carry_seed2 M020-visible-carry_seed3" >&2
      exit 2
      ;;
    mutation)
      echo "Mutation phase is intentionally manual; pass explicit IDs after editing the matrix." >&2
      exit 2
      ;;
    *)
      echo "No default IDs configured for phase ${PHASE}; pass IDs explicitly." >&2
      exit 2
      ;;
  esac
fi

mkdir -p "${RUN_ROOT}/${PHASE}"

for id in "${IDS[@]}"; do
  plan_path="${RUN_ROOT}/${PHASE}/${id}_plan.json"
  echo "Planning ${PHASE}/${id} -> ${plan_path}"
  "${PYTHON_BIN}" -m ant_byte_env.cli autoresearch map-ant-plan \
    --matrix "${MATRIX}" \
    --phase "${PHASE}" \
    --id "${id}" \
    --run-root "${RUN_ROOT}" \
    "${RENDER_ARGS[@]}" \
    "${TRACK_ARGS[@]}" \
    > "${plan_path}"

  echo "Running ${PHASE}/${id}"
  "${PYTHON_BIN}" -m ant_byte_env.cli autoresearch map-ant-run \
    --matrix "${MATRIX}" \
    --phase "${PHASE}" \
    --id "${id}" \
    --run-root "${RUN_ROOT}" \
    "${RENDER_ARGS[@]}" \
    "${TRACK_ARGS[@]}" \
    "${RUN_ARGS[@]}"
done

ranking_path="${RUN_ROOT}/${PHASE}/ranking.json"
echo "Ranking ${PHASE} -> ${ranking_path}"
"${PYTHON_BIN}" -m ant_byte_env.cli autoresearch map-ant-rank \
  --matrix "${MATRIX}" \
  --phase "${PHASE}" \
  --run-root "${RUN_ROOT}" \
  > "${ranking_path}"

echo "Done. Inspect ${ranking_path} and render top/near-pass candidates before declaring success."
