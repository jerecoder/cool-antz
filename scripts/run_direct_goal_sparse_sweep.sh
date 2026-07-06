#!/usr/bin/env bash
set -euo pipefail

PHASE="${PHASE:-sparse_hyperparams}"
MATRIX="${MATRIX:-autoresearch/direct_goal_sweep.json}"
RUN_ROOT="${RUN_ROOT:-runs/autoresearch/direct_goal}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

EVAL_ARGS=()
if [[ -n "${EVAL_EPISODES:-}" ]]; then
  EVAL_ARGS=(--eval-episodes "${EVAL_EPISODES}")
fi

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
    sparse_hyperparams)
      IDS=(S0 S1 S2 S3 S4 S5)
      ;;
    sparse_multiseed)
      IDS=(
        S4_seed2 S4_seed3 S4_seed4
        S5_seed2 S5_seed3 S5_seed4
        S2_seed2 S2_seed3 S2_seed4
      )
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
  "${PYTHON_BIN}" -m ant_byte_env.cli autoresearch direct-goal-plan \
    --matrix "${MATRIX}" \
    --phase "${PHASE}" \
    --id "${id}" \
    --run-root "${RUN_ROOT}" \
    "${EVAL_ARGS[@]}" \
    "${TRACK_ARGS[@]}" \
    > "${plan_path}"

  echo "Running ${PHASE}/${id}"
  "${PYTHON_BIN}" -m ant_byte_env.cli autoresearch direct-goal-run \
    --matrix "${MATRIX}" \
    --phase "${PHASE}" \
    --id "${id}" \
    --run-root "${RUN_ROOT}" \
    "${EVAL_ARGS[@]}" \
    "${TRACK_ARGS[@]}" \
    "${RUN_ARGS[@]}"
done

ranking_path="${RUN_ROOT}/${PHASE}/ranking.json"
echo "Ranking ${PHASE} -> ${ranking_path}"
"${PYTHON_BIN}" -m ant_byte_env.cli autoresearch direct-goal-rank \
  --matrix "${MATRIX}" \
  --phase "${PHASE}" \
  --run-root "${RUN_ROOT}" \
  > "${ranking_path}"

echo "Done. Inspect ${ranking_path}"
