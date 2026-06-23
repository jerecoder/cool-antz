#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./autoresearch/run_codex_vision_range_tmux.sh [session] [target_ret] [max_trials]

Defaults:
  session:    ant_vision_range_autoresearch
  target_ret: 60.0
  max_trials: 0     # 0 means unlimited

The launcher starts Codex in tmux with the vision-range autoresearch prompt.
Codex is instructed to tune experiments/vision_range_curriculum.json, run the
vision-range notebook, and score the 51x51 stage return.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PROGRAM_PATH="${PROJECT_ROOT}/autoresearch/vision_range_program.md"

SESSION="${1:-ant_vision_range_autoresearch}"
TARGET_RET="${2:-60.0}"
MAX_TRIALS="${3:-0}"
SAFE_SESSION="${SESSION//[^A-Za-z0-9_.-]/_}"
LOG_DIR="${PROJECT_ROOT}/runs/autoresearch/vision_range_curriculum"
LOG_PATH="${LOG_DIR}/${SAFE_SESSION}.log"

if [[ ! -f "${PROGRAM_PATH}" ]]; then
  echo "Missing program file: ${PROGRAM_PATH}" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found on PATH." >&2
  exit 1
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex is required but was not found on PATH." >&2
  exit 1
fi

if [[ "${ANT_VISION_RANGE_AUTORESEARCH_CHILD:-}" == "1" ]]; then
  cd "${PROJECT_ROOT}"
  mkdir -p "${LOG_DIR}" runs/autoresearch/vision_range_curriculum/notebook_exec

  export VISION_RANGE_TARGET_RET="${TARGET_RET}"
  export VISION_RANGE_MAX_TRIALS="${MAX_TRIALS}"
  export VISION_RANGE_CONFIG="${VISION_RANGE_CONFIG:-experiments/vision_range_curriculum.json}"
  export VISION_RANGE_NOTEBOOK="${VISION_RANGE_NOTEBOOK:-notebooks/train_jax_vision_range_curriculum.ipynb}"
  export VISION_RANGE_RUN_DIR="${VISION_RANGE_RUN_DIR:-runs/notebooks/vision_range_curriculum}"
  export VISION_RANGE_EXEC_DIR="${VISION_RANGE_EXEC_DIR:-runs/autoresearch/vision_range_curriculum/notebook_exec}"
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

  PROMPT="$(< "${PROGRAM_PATH}")"
  PROMPT="${PROMPT}"$'\n\n'"Launcher settings:"
  PROMPT="${PROMPT}"$'\n'"- VISION_RANGE_TARGET_RET=${VISION_RANGE_TARGET_RET}"
  PROMPT="${PROMPT}"$'\n'"- VISION_RANGE_MAX_TRIALS=${VISION_RANGE_MAX_TRIALS}"
  PROMPT="${PROMPT}"$'\n'"- JAX_PLATFORMS=${JAX_PLATFORMS}"
  PROMPT="${PROMPT}"$'\n'"- CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  PROMPT="${PROMPT}"$'\n\n'"Start by inspecting ${VISION_RANGE_CONFIG} and any existing ${VISION_RANGE_RUN_DIR}/51x51/summary.json, then run the loop."

  exec codex \
    --no-alt-screen \
    --cd "${PROJECT_ROOT}" \
    --sandbox workspace-write \
    --ask-for-approval on-request \
    "${PROMPT}"
fi

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

tmux new-session -d \
  -s "${SESSION}" \
  -c "${PROJECT_ROOT}" \
  "ANT_VISION_RANGE_AUTORESEARCH_CHILD=1 bash \"${SCRIPT_PATH}\" \"${SESSION}\" \"${TARGET_RET}\" \"${MAX_TRIALS}\""

tmux pipe-pane -o -t "${SESSION}" "cat >> \"${LOG_PATH}\""

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
echo "Log file: ${LOG_PATH}"
echo "Target 51x51 ret: ${TARGET_RET}"
echo "Max trials: ${MAX_TRIALS} (0 means unlimited)"
