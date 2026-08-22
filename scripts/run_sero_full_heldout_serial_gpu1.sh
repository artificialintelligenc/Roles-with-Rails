#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-1}"
RUN_TAG="${2:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="$ROOT_DIR/results/evaluation"
QUEUE_LOG="$RESULTS_DIR/sero_full_heldout_gpu${GPU_ID}_${RUN_TAG}.queue.log"
BENCHMARKS=(trip calendar meeting olympiadbench)

mkdir -p "$RESULTS_DIR"
: > "$QUEUE_LOG"

log() {
    local msg="$1"
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$QUEUE_LOG"
}

run_benchmark() {
    local benchmark="$1"
    local benchmark_log="$RESULTS_DIR/${benchmark}_sero_full_heldout_gpu${GPU_ID}_${RUN_TAG}.log"
    local benchmark_results_dir="$RESULTS_DIR/sero_ckpt_${benchmark}_full_heldout_gpu${GPU_ID}_${RUN_TAG}"
    local attempt
    local max_attempts=3

    : > "$benchmark_log"

    for attempt in $(seq 1 "$max_attempts"); do
        log "Starting ${benchmark} attempt ${attempt}/${max_attempts} on GPU ${GPU_ID}"
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        PYTHONUNBUFFERED=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluate.py" \
            --system sero \
            --benchmark "$benchmark" \
            --tasks 20 \
            --eval_tasks 0 \
            --warmup_epochs 2 \
            --main_epochs 8 \
            --t_round 1 \
            --n_max 4 \
            --batch_size 8 \
            --loo_refresh 40 \
            --entropy_beta 0.05 \
            --lr 0.001 \
            --ema_mu 0.1 \
            --fast_credit_alpha 0.5 \
            --exploration_gamma 0.1 \
            --seed 42 \
            --eval_set heldout \
            --suffix "_full_heldout_gpu${GPU_ID}_${RUN_TAG}" \
            --results_dir "$benchmark_results_dir" \
            2>&1 | tee -a "$benchmark_log" "$QUEUE_LOG"
        local status=${PIPESTATUS[0]}

        if [[ $status -eq 0 ]]; then
            log "Finished ${benchmark} successfully"
            return 0
        fi

        log "${benchmark} attempt ${attempt} failed with status ${status}"
    done

    return 1
}

main() {
    local failures=()

    log "Queue start: GPU ${GPU_ID}, tag ${RUN_TAG}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$QUEUE_LOG" || true

    for benchmark in "${BENCHMARKS[@]}"; do
        if ! run_benchmark "$benchmark"; then
            failures+=("$benchmark")
            log "Continuing after ${benchmark} failure"
        fi
    done

    if (( ${#failures[@]} > 0 )); then
        log "Queue finished with failures: ${failures[*]}"
        return 1
    fi

    log "Queue finished successfully for all benchmarks"
}

main "$@"