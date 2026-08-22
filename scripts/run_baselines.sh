#!/usr/bin/env bash
# ============================================================================
# Baseline Evaluation Script — 正式实验用
#
# 支持除 sero 以外的所有 baseline:
#   cot, sc, static, static_dag, workflow, random_evo
#
# 评测集模式:
#   1. EVAL_SET=subset        → train_split.json 中固定的分层子集 (naturalplan=300, 其他=100)
#   2. EVAL_SET=heldout       → 排除固定 train keys 的 held-out 测试集 (naturalplan=30 train, tablebench=40 train, 其他=20)
#   3. EVAL_SET=natural_full  → 原始 benchmark 全量测试集（不走 split 过滤）
#   4. EVAL_SET=legacy        → 原始采样模式
#
# 评测样本量:
#   1. EVAL_TASKS=N      → 从所选评测集中取前 N 个样本
#   2. EVAL_TASKS=0      → 使用所选评测集的全部样本
#
# 样本规模:
#   heldout:      naturalplan=900, trip=1580, meeting=980, calendar=980, olympiadbench=846, tablebench=796
#   natural_full: naturalplan=3600, trip=1600, meeting=1000, calendar=1000, olympiadbench=897, tablebench=836
#
# 用法:
#   # held-out 全量测试集
#   EVAL_SET=heldout EVAL_TASKS=0 bash scripts/run_baselines.sh trip "cot sc"
#
#   # 100 样本分层子集
#   EVAL_SET=subset bash scripts/run_baselines.sh trip "cot sc static static_dag workflow"
#
#   # 原始 Natural Plan 全量测试集
#   EVAL_SET=natural_full EVAL_TASKS=0 bash scripts/run_baselines.sh trip "cot sc"
#
#   # 指定数量
#   EVAL_SET=heldout EVAL_TASKS=200 bash scripts/run_baselines.sh meeting cot
#
#   # 全部 benchmark × 全部 baseline
#   EVAL_SET=heldout EVAL_TASKS=0 bash scripts/run_baselines.sh
#
# 输出:
#   results/evaluation/{benchmark}_{system}{SUFFIX}.json
# ============================================================================
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    OPENROUTER_API_KEY="$("$PYTHON_BIN" - <<'PY'
from sero.config import OPENROUTER_API_KEY
print(OPENROUTER_API_KEY or "")
PY
)"
fi
export OPENROUTER_API_KEY
export HF_HOME="${HF_HOME:-}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
if [[ "${TRANSFORMERS_CACHE:-}" == "$HF_HOME" ]]; then
    unset TRANSFORMERS_CACHE
elif [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
    export TRANSFORMERS_CACHE
fi
# ── 可配置参数 (环境变量覆盖) ────────────────────────────────────────────────
EVAL_TASKS="${EVAL_TASKS:-0}"          # 0 = 所选评测集的全部样本
EVAL_SET="${EVAL_SET:-}"
EVAL_SUBSET="${EVAL_SUBSET:-false}"    # 兼容旧接口: true 等价于 EVAL_SET=subset
SC_K="${SC_K:-3}"                      # SC 投票采样数
SEED="${SEED:-42}"
SUFFIX="${SUFFIX:-_baseline}"

# random_evo 训练参数 (与 trial_run.sh 一致, 但 epoch 更大)
TRAIN_TASKS="${TRAIN_TASKS:-}"
WARMUP_EPOCHS="${RE_WARMUP:-1}"
MAIN_EPOCHS="${RE_MAIN:-2}"
BATCH_SIZE=8
T_ROUND=1
N_MAX=4
LOO_REFRESH=40
ENTROPY_BETA=0.05
LR=0.001
EMA_MU=0.1
FAST_CREDIT_ALPHA=0.5
EXPLORATION_GAMMA=0.1

# ── Benchmark / System ───────────────────────────────────────────────────────
ALL_BENCHMARKS=("naturalplan" "trip" "meeting" "calendar" "olympiadbench" "tablebench")
ALL_SYSTEMS=("cot" "sc" "static" "static_dag" "workflow" "random_evo")

if [[ $# -ge 1 ]]; then
    BENCHMARKS=("$1")
else
    BENCHMARKS=("${ALL_BENCHMARKS[@]}")
fi
if [[ $# -ge 2 ]]; then
    read -ra SYSTEMS <<< "$2"
else
    SYSTEMS=("${ALL_SYSTEMS[@]}")
fi

# ── 校验: 禁止 sero ─────────────────────────────────────────────────────────
for sys in "${SYSTEMS[@]}"; do
    if [[ "$sys" == "sero" ]]; then
        echo "ERROR: 本脚本不支持 sero, 请用 full_sero_run.sh" >&2
        exit 1
    fi
done

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set." >&2
    exit 1
fi

if [[ -z "$EVAL_SET" ]]; then
    if [[ "$EVAL_SUBSET" == "true" ]]; then
        EVAL_SET="subset"
    else
        EVAL_SET="heldout"
    fi
fi

case "$EVAL_SET" in
    subset|heldout|natural_full|legacy)
        ;;
    *)
        echo "ERROR: EVAL_SET must be one of: subset, heldout, natural_full, legacy" >&2
        exit 1
        ;;
esac

# ── 评测模式描述 ─────────────────────────────────────────────────────────────
case "$EVAL_SET" in
    subset)
        EVAL_MODE="fixed stratified subset (naturalplan=300, others=100)"
        ;;
    heldout)
        if [[ "$EVAL_TASKS" == "0" ]]; then
            EVAL_MODE="full held-out test set"
        else
            EVAL_MODE="${EVAL_TASKS} samples from held-out test set"
        fi
        ;;
    natural_full)
        if [[ "$EVAL_TASKS" == "0" ]]; then
            EVAL_MODE="full original Natural Plan test set"
        else
            EVAL_MODE="${EVAL_TASKS} samples from original Natural Plan test set"
        fi
        ;;
    legacy)
        if [[ "$EVAL_TASKS" == "0" ]]; then
            EVAL_MODE="legacy full benchmark sampling"
        else
            EVAL_MODE="${EVAL_TASKS} samples from legacy benchmark sampling"
        fi
        ;;
esac

echo "================================================================"
echo "  Baseline Evaluation (正式实验)"
echo "  Benchmarks: ${BENCHMARKS[*]}"
echo "  Systems:    ${SYSTEMS[*]}"
echo "  Eval set:   ${EVAL_SET}"
echo "  Eval mode:  ${EVAL_MODE}"
echo "  Train tasks:${TRAIN_TASKS:-auto (naturalplan=30, tablebench=40, others=20)}"
echo "  SC_K:       ${SC_K}"
echo "  Seed:       ${SEED}"
echo "  Suffix:     ${SUFFIX}"
echo "  Python:     ${PYTHON_BIN}"
echo "================================================================"
echo ""

PASSED=0
FAILED=0
SKIPPED=0
TOTAL_START=$(date +%s)

for bench in "${BENCHMARKS[@]}"; do
    for sys in "${SYSTEMS[@]}"; do
        EFFECTIVE_EVAL_SET="$EVAL_SET"
        EFFECTIVE_TRAIN_TASKS="$TRAIN_TASKS"
        if [[ -z "$EFFECTIVE_TRAIN_TASKS" ]]; then
            if [[ "$bench" == "naturalplan" ]]; then
                EFFECTIVE_TRAIN_TASKS=30
            elif [[ "$bench" == "tablebench" ]]; then
                EFFECTIVE_TRAIN_TASKS=40
            else
                EFFECTIVE_TRAIN_TASKS=20
            fi
        fi

        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "  ${bench} × ${sys}"
        echo "────────────────────────────────────────────────────────────"

        # ── 构建命令 ─────────────────────────────────────────────────────
        CMD=("$PYTHON_BIN" scripts/evaluate.py
            --system "$sys"
            --benchmark "$bench"
            --eval_tasks "$EVAL_TASKS"
            --seed "$SEED"
            --suffix "$SUFFIX"
            --eval_set "$EFFECTIVE_EVAL_SET"
        )

        if [[ "$sys" == "sc" ]]; then
            CMD+=(--sc_k "$SC_K")
        fi

        if [[ "$EVAL_SET" == "natural_full" && "$sys" == "random_evo" ]]; then
            echo "WARN: ${bench} × ${sys} 将在原始 Natural Plan 全量集上评测, 结果包含训练任务。"
        fi

        # random_evo 需要训练参数
        if [[ "$sys" == "random_evo" ]]; then
            CMD+=(
                --tasks "$EFFECTIVE_TRAIN_TASKS"
                --warmup_epochs "$WARMUP_EPOCHS"
                --main_epochs "$MAIN_EPOCHS"
                --t_round "$T_ROUND"
                --n_max "$N_MAX"
                --batch_size "$BATCH_SIZE"
                --loo_refresh "$LOO_REFRESH"
                --entropy_beta "$ENTROPY_BETA"
                --lr "$LR"
                --ema_mu "$EMA_MU"
                --fast_credit_alpha "$FAST_CREDIT_ALPHA"
                --exploration_gamma "$EXPLORATION_GAMMA"
            )
        fi

        echo ">> ${CMD[*]}"
        START_T=$(date +%s)
        if "${CMD[@]}"; then
            ELAPSED=$(( $(date +%s) - START_T ))
            echo "OK: ${bench} × ${sys} (${ELAPSED}s)"
            ((++PASSED))
        else
            echo "FAILED: ${bench} × ${sys}" >&2
            ((++FAILED))
        fi
    done
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
TOTAL_H=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_M=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "================================================================"
echo "  Baseline Evaluation 完成"
echo "  通过: ${PASSED}  失败: ${FAILED}  跳过: ${SKIPPED}"
echo "  总耗时: ${TOTAL_H}h ${TOTAL_M}m"
echo "  结果: results/evaluation/*${SUFFIX}.json"
echo "================================================================"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
