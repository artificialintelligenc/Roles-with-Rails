#!/usr/bin/env bash
# ============================================================================
# Full SERO Run
#
# 关键默认值只在下方超参区定义一次；实际运行时的配置与成本摘要以脚本打印为准。
#
# 成本估算口径：
#   phase_a_calls_per_pass = N_MAX × T_ROUND
#   train_calls_upper ~= total_eps × (2 × phase_a_calls_per_pass + executor_est)
#   loo_calls_upper   ~= loo_refreshes × loo_est_pool × loo_sample_tasks × 2 × phase_a_calls_per_pass
#   eval_calls_upper  ~= eval_tasks × phase_a_calls_per_pass
#
# 注意：由于 noop / invalid 动作现在会复用 before 结果，真实调用量通常低于这里打印的 upper bound。
#
# 用法:
#   bash scripts/full_sero_run.sh                # naturalplan, seed=42
#   bash scripts/full_sero_run.sh trip 42        # 指定 seed
#   bash scripts/full_sero_run.sh calendar 123   # 其他 benchmark
#   bash scripts/full_sero_run.sh naturalplan 42 # combined NaturalPlan mixed benchmark
#   bash scripts/full_sero_run.sh olympiadbench 42
#   bash scripts/full_sero_run.sh tablebench 42
#   EVAL_SET=subset bash scripts/full_sero_run.sh trip 42
#   EVAL_SET=heldout EVAL_TASKS=0 bash scripts/full_sero_run.sh trip 42
#   EVAL_SET=natural_full EVAL_TASKS=0 bash scripts/full_sero_run.sh trip 42
#
# 注意:
#   naturalplan 默认使用 EVAL_SET=heldout + EVAL_TASKS=0, 即 900 clean-first balanced main heldout。
#   EVAL_SET=natural_full 时, 训练仍使用固定 train_split 任务,
#   但评测会覆盖原始 Natural Plan 全量数据, 因而包含训练任务。
# ============================================================================
set -euo pipefail

BENCHMARK="${1:-naturalplan}"
SEED="${2:-42}"
RESULT_TAG="${RESULT_TAG:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "$RESULT_TAG" ]]; then
    SUFFIX="_full_${RESULT_TAG}_s${SEED}"
    DEFAULT_SERO_RESULTS_DIR="results/evaluation/sero_ckpt_${SEED}_${RESULT_TAG}_${BENCHMARK}"
else
    SUFFIX="_full_s${SEED}"
    DEFAULT_SERO_RESULTS_DIR="results/evaluation/sero_ckpt_${SEED}"
fi
SERO_RESULTS_DIR="${SERO_RESULTS_DIR:-$DEFAULT_SERO_RESULTS_DIR}"

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
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# ── 数据规模与评测集控制 ─────────────────────────────────────────────────────
TRAIN_TASKS="${TRAIN_TASKS:-}"
EVAL_TASKS="${EVAL_TASKS:-}"
EVAL_SET="${EVAL_SET:-}"
EVAL_SUBSET="${EVAL_SUBSET:-}"  # 兼容旧接口: true 等价于 EVAL_SET=subset

# ── 个性化 SERO 超参 (环境变量可覆盖；Olympiad/default 保持原有配置) ─────────────
case "$BENCHMARK" in
    naturalplan)
        HP_PROFILE="${HP_PROFILE:-default_olympiad_compatible}"
        : "${WARMUP_EPOCHS:=2}"
        : "${MAIN_EPOCHS:=8}"
        : "${BATCH_SIZE:=8}"
        : "${T_ROUND:=1}"
        : "${N_MAX:=4}"
        : "${P_MAX:=12}"
        : "${P_MIN:=4}"
        : "${LOO_REFRESH:=40}"
        : "${ENTROPY_BETA:=0.05}"
        : "${LR:=0.001}"
        : "${EMA_MU:=0.1}"
        : "${FAST_CREDIT_ALPHA:=0.5}"
        : "${EXPLORATION_GAMMA:=0.1}"
        : "${LOO_MIN_POOL_SIZE:=4}"
        : "${NEW_ROLE_INITIAL_N_UPDATES:=3}"
        : "${NOOP_COLLAPSE_THRESHOLD:=0.85}"
        ;;
    tablebench)
        HP_PROFILE="${HP_PROFILE:-default_olympiad_compatible}"
        : "${WARMUP_EPOCHS:=2}"
        : "${MAIN_EPOCHS:=8}"
        : "${BATCH_SIZE:=8}"
        : "${T_ROUND:=1}"
        : "${N_MAX:=4}"
        : "${P_MAX:=12}"
        : "${P_MIN:=4}"
        : "${LOO_REFRESH:=40}"
        : "${ENTROPY_BETA:=0.05}"
        : "${LR:=0.001}"
        : "${EMA_MU:=0.1}"
        : "${FAST_CREDIT_ALPHA:=0.5}"
        : "${EXPLORATION_GAMMA:=0.1}"
        : "${LOO_MIN_POOL_SIZE:=4}"
        : "${NEW_ROLE_INITIAL_N_UPDATES:=3}"
        : "${NOOP_COLLAPSE_THRESHOLD:=0.85}"
        ;;
    *)
        HP_PROFILE="${HP_PROFILE:-default_olympiad_compatible}"
        : "${WARMUP_EPOCHS:=2}"
        : "${MAIN_EPOCHS:=8}"
        : "${BATCH_SIZE:=8}"
        : "${T_ROUND:=1}"
        : "${N_MAX:=4}"
        : "${P_MAX:=12}"
        : "${P_MIN:=4}"
        : "${LOO_REFRESH:=40}"
        : "${ENTROPY_BETA:=0.05}"
        : "${LR:=0.001}"
        : "${EMA_MU:=0.1}"
        : "${FAST_CREDIT_ALPHA:=0.5}"
        : "${EXPLORATION_GAMMA:=0.1}"
        : "${LOO_MIN_POOL_SIZE:=4}"
        : "${NEW_ROLE_INITIAL_N_UPDATES:=3}"
        : "${NOOP_COLLAPSE_THRESHOLD:=0.85}"
        ;;
esac

STRICT_TUNED_PROFILE="gemini25_flash_lite_natplan_sero_tuned_v1"
AGENT_MODEL_NAME="${SERO_AGENT_MODEL:-}"
EXECUTOR_MODEL_NAME="${SERO_EXECUTOR_MODEL:-}"

EXTRA_SERO_ARGS=()
SPECIALIST_SLOTS_SUMMARY="${SPECIALIST_SLOTS:-}"
VALIDATOR_SLOTS_SUMMARY="${VALIDATOR_SLOTS:-}"
MIN_CAPABILITY_FAMILY_DIVERSITY_SUMMARY="${MIN_CAPABILITY_FAMILY_DIVERSITY:-}"
MAX_CAPABILITY_FAMILY_DOMINANCE_SUMMARY="${MAX_CAPABILITY_FAMILY_DOMINANCE:-}"
ORTHOGONAL_ROLE_BLACKLIST_TOP_K_SUMMARY="${ORTHOGONAL_ROLE_BLACKLIST_TOP_K:-}"

if [[ "$HP_PROFILE" == "$STRICT_TUNED_PROFILE" ]]; then
    if [[ "$BENCHMARK" != "naturalplan" ]]; then
        echo "ERROR: HP_PROFILE=$STRICT_TUNED_PROFILE only supports BENCHMARK=naturalplan." >&2
        exit 1
    fi
    if [[ "$AGENT_MODEL_NAME" != "gemini-2.5-flash-lite" || "$EXECUTOR_MODEL_NAME" != "gemini-2.5-flash-lite" ]]; then
        echo "ERROR: HP_PROFILE=$STRICT_TUNED_PROFILE requires SERO_AGENT_MODEL=gemini-2.5-flash-lite and SERO_EXECUTOR_MODEL=gemini-2.5-flash-lite." >&2
        exit 1
    fi

    WARMUP_EPOCHS=1
    MAIN_EPOCHS=9
    BATCH_SIZE=4
    P_MAX=10
    P_MIN=3
    LOO_REFRESH=20
    EMA_MU=0.05
    ENTROPY_BETA=0.08
    EXPLORATION_GAMMA=0.15

    SPECIALIST_SLOTS_SUMMARY=5
    VALIDATOR_SLOTS_SUMMARY=1
    MIN_CAPABILITY_FAMILY_DIVERSITY_SUMMARY=0.6
    MAX_CAPABILITY_FAMILY_DOMINANCE_SUMMARY=0.5
    ORTHOGONAL_ROLE_BLACKLIST_TOP_K_SUMMARY=3

fi

if [[ -n "$SPECIALIST_SLOTS_SUMMARY" ]]; then
    EXTRA_SERO_ARGS+=(--specialist_slots "$SPECIALIST_SLOTS_SUMMARY")
fi
if [[ -n "$VALIDATOR_SLOTS_SUMMARY" ]]; then
    EXTRA_SERO_ARGS+=(--validator_slots "$VALIDATOR_SLOTS_SUMMARY")
fi
if [[ -n "$MIN_CAPABILITY_FAMILY_DIVERSITY_SUMMARY" ]]; then
    EXTRA_SERO_ARGS+=(--min_capability_family_diversity "$MIN_CAPABILITY_FAMILY_DIVERSITY_SUMMARY")
fi
if [[ -n "$MAX_CAPABILITY_FAMILY_DOMINANCE_SUMMARY" ]]; then
    EXTRA_SERO_ARGS+=(--max_capability_family_dominance "$MAX_CAPABILITY_FAMILY_DOMINANCE_SUMMARY")
fi
if [[ -n "$ORTHOGONAL_ROLE_BLACKLIST_TOP_K_SUMMARY" ]]; then
    EXTRA_SERO_ARGS+=(--orthogonal_role_blacklist_top_k "$ORTHOGONAL_ROLE_BLACKLIST_TOP_K_SUMMARY")
fi

# ── 校验 & 摘要 ──────────────────────────────────────────────────────────────
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set." >&2
    exit 1
fi

if [[ -z "$EVAL_SET" ]]; then
    if [[ "$EVAL_SUBSET" == "true" ]]; then
        EVAL_SET="subset"
    elif [[ "$EVAL_SUBSET" == "false" ]]; then
        EVAL_SET="heldout"
    elif [[ "$BENCHMARK" == "naturalplan" ]]; then
        EVAL_SET="heldout"
    else
        EVAL_SET="subset"
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

case "$BENCHMARK" in
    naturalplan)
        HELDOUT_FULL=900
        NATURAL_FULL=3600
        ;;
    trip)
        HELDOUT_FULL=1580
        NATURAL_FULL=1600
        ;;
    meeting)
        HELDOUT_FULL=980
        NATURAL_FULL=1000
        ;;
    calendar)
        HELDOUT_FULL=980
        NATURAL_FULL=1000
        ;;
    olympiadbench)
        HELDOUT_FULL=846
        NATURAL_FULL=897
        ;;
    tablebench)
        HELDOUT_FULL=796
        NATURAL_FULL=836
        ;;
    *)
        echo "ERROR: full_sero_run.sh supports: naturalplan, trip, meeting, calendar, olympiadbench, tablebench" >&2
        exit 1
        ;;
esac

if [[ -z "$TRAIN_TASKS" ]]; then
    if [[ "$BENCHMARK" == "naturalplan" ]]; then
        TRAIN_TASKS=30
    elif [[ "$BENCHMARK" == "tablebench" ]]; then
        TRAIN_TASKS=40
    else
        TRAIN_TASKS=20
    fi
fi

if [[ -z "$EVAL_TASKS" ]]; then
    if [[ "$BENCHMARK" == "naturalplan" && "$EVAL_SET" == "heldout" ]]; then
        EVAL_TASKS=0
    else
        EVAL_TASKS=100
    fi
fi

if [[ "$EVAL_SET" == "subset" ]]; then
    if [[ "$BENCHMARK" == "naturalplan" ]]; then
        EFFECTIVE_EVAL_TASKS=300
    else
        EFFECTIVE_EVAL_TASKS=100
    fi
elif [[ "$EVAL_SET" == "legacy" ]]; then
    if [[ "$EVAL_TASKS" == "0" ]]; then
        EFFECTIVE_EVAL_TASKS="$NATURAL_FULL"
    else
        EFFECTIVE_EVAL_TASKS="$EVAL_TASKS"
    fi
elif [[ "$EVAL_TASKS" == "0" ]]; then
    if [[ "$EVAL_SET" == "heldout" ]]; then
        EFFECTIVE_EVAL_TASKS="$HELDOUT_FULL"
    else
        EFFECTIVE_EVAL_TASKS="$NATURAL_FULL"
    fi
else
    EFFECTIVE_EVAL_TASKS="$EVAL_TASKS"
fi

TOTAL_EPS=$(( (WARMUP_EPOCHS + MAIN_EPOCHS) * TRAIN_TASKS ))
WARMUP_EPS=$(( WARMUP_EPOCHS * TRAIN_TASKS ))
MAIN_EPS=$(( MAIN_EPOCHS * TRAIN_TASKS ))
GRAD_STEPS=$(( TOTAL_EPS / BATCH_SIZE ))
LOO_COUNT=$(( TOTAL_EPS / LOO_REFRESH ))
PHASEA_CALLS_PER_PASS=$(( N_MAX * T_ROUND ))
LOO_EST_POOL=7
LOO_SAMPLE_TASKS=3
EXECUTOR_CALL_EST_X100=33
EPISODE_CALLS_X100=$(( 2 * PHASEA_CALLS_PER_PASS * 100 + EXECUTOR_CALL_EST_X100 ))
LOO_CALLS_UPPER=$(( LOO_EST_POOL * LOO_SAMPLE_TASKS * 2 * PHASEA_CALLS_PER_PASS ))
EVAL_CALLS_UPPER=$(( EFFECTIVE_EVAL_TASKS * PHASEA_CALLS_PER_PASS ))
API_EST_X100=$(( TOTAL_EPS * EPISODE_CALLS_X100 + LOO_COUNT * LOO_CALLS_UPPER * 100 + EVAL_CALLS_UPPER * 100 ))
printf -v API_EST_FMT "%d.%02d" $(( API_EST_X100 / 100 )) $(( API_EST_X100 % 100 ))

echo "================================================================"
echo "  SERO Full Run"
echo "  Benchmark:      ${BENCHMARK}"
echo "  Seed:           ${SEED}"
echo "  HP profile:     ${HP_PROFILE}"
echo "  Eval set:       ${EVAL_SET}"
echo "  Train/Eval:     ${TRAIN_TASKS}/${EFFECTIVE_EVAL_TASKS}"
echo "  Epochs:         ${WARMUP_EPOCHS}w + ${MAIN_EPOCHS}m = $((WARMUP_EPOCHS+MAIN_EPOCHS))"
echo "  Episodes:       total=${TOTAL_EPS}  warmup=${WARMUP_EPS}  main=${MAIN_EPS}"
echo "  Gradient steps: ~${GRAD_STEPS}  (batch_size=${BATCH_SIZE})"
echo "  LOO refreshes:  ~${LOO_COUNT}   (interval=${LOO_REFRESH})"
echo "  T_ROUND=${T_ROUND}  N_MAX=${N_MAX}  P_MAX=${P_MAX}  P_MIN=${P_MIN}"
echo "  EMA_MU=${EMA_MU}  ALPHA=${FAST_CREDIT_ALPHA}  GAMMA=${EXPLORATION_GAMMA}"
echo "  LOO_MIN_POOL_SIZE=${LOO_MIN_POOL_SIZE}  NEW_ROLE_INITIAL_N_UPDATES=${NEW_ROLE_INITIAL_N_UPDATES}"
echo "  NOOP_COLLAPSE_THRESHOLD=${NOOP_COLLAPSE_THRESHOLD}"
if [[ -n "$SPECIALIST_SLOTS_SUMMARY" || -n "$VALIDATOR_SLOTS_SUMMARY" ]]; then
    echo "  Retrieval:      specialist_slots=${SPECIALIST_SLOTS_SUMMARY}  validator_slots=${VALIDATOR_SLOTS_SUMMARY}"
fi
if [[ -n "$MIN_CAPABILITY_FAMILY_DIVERSITY_SUMMARY" || -n "$MAX_CAPABILITY_FAMILY_DOMINANCE_SUMMARY" || -n "$ORTHOGONAL_ROLE_BLACKLIST_TOP_K_SUMMARY" ]]; then
    echo "  Diversity:      preserve_seed_family_coverage=config  min_div=${MIN_CAPABILITY_FAMILY_DIVERSITY_SUMMARY}  max_dom=${MAX_CAPABILITY_FAMILY_DOMINANCE_SUMMARY}  max_sim=config  orth_top_k=${ORTHOGONAL_ROLE_BLACKLIST_TOP_K_SUMMARY}"
fi
echo "  Python:         ${PYTHON_BIN}"
echo "  Checkpoints:    ${SERO_RESULTS_DIR}"
echo "  PhaseA/pass:    ${PHASEA_CALLS_PER_PASS}  executor_est≈0.${EXECUTOR_CALL_EST_X100}"
echo "  Est. API calls: ~${API_EST_FMT} (upper bound)"
echo "================================================================"
echo ""

if [[ "$EVAL_SET" == "natural_full" ]]; then
    echo "WARN: natural_full 模式会把训练任务也包含进评测集, 结果不是 held-out。"
    echo ""
fi

START_TIME=$(date +%s)

"$PYTHON_BIN" scripts/evaluate.py \
    --system sero \
    --benchmark "$BENCHMARK" \
    --tasks "$TRAIN_TASKS" \
    --eval_tasks "$EVAL_TASKS" \
    --seed "$SEED" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --main_epochs "$MAIN_EPOCHS" \
    --t_round "$T_ROUND" \
    --n_max "$N_MAX" \
    --p_max "$P_MAX" \
    --p_min "$P_MIN" \
    --batch_size "$BATCH_SIZE" \
    --loo_refresh "$LOO_REFRESH" \
    --entropy_beta "$ENTROPY_BETA" \
    --lr "$LR" \
    --ema_mu "$EMA_MU" \
    --fast_credit_alpha "$FAST_CREDIT_ALPHA" \
    --exploration_gamma "$EXPLORATION_GAMMA" \
    --loo_min_pool_size "$LOO_MIN_POOL_SIZE" \
    --new_role_initial_n_updates "$NEW_ROLE_INITIAL_N_UPDATES" \
    --noop_collapse_threshold "$NOOP_COLLAPSE_THRESHOLD" \
    --eval_set "$EVAL_SET" \
    --results_dir "$SERO_RESULTS_DIR" \
    --suffix "$SUFFIX" \
    "${EXTRA_SERO_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "================================================================"
echo "  完成 — 耗时: ${HOURS}h ${MINS}m"
echo "  结果: results/evaluation/${BENCHMARK}_sero${SUFFIX}.json"
echo "================================================================"
