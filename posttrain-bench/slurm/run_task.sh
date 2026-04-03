#!/bin/bash
# SLURM + Pyxis equivalent of PostTrainBench's run_task.sh
# Usage: sbatch slurm/submit.sbatch gsm8k hf_agent Qwen/Qwen3-1.7B-Base 10 claude-opus-4-6

set -euo pipefail

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
NUM_HOURS="$4"
AGENT_CONFIG="$5"

# Paths — adjust POSTTRAINBENCH_DIR and HF_CACHE to your cluster
POSTTRAINBENCH_DIR="${POSTTRAINBENCH_DIR:-/fsx/aksel/PostTrainBench}"
AGENT_REPO_DIR="${AGENT_REPO_DIR:-/fsx/aksel/hf_agent}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
DOCKER_IMAGE="${DOCKER_IMAGE:-posttrainbench:latest}"

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')
EXPERIMENT_NAME="${POST_TRAIN_BENCH_EXPERIMENT_NAME:-}"
RANDOM_UUID=$(uuidgen || cat /proc/sys/kernel/random/uuid)

EVAL_DIR="${POSTTRAINBENCH_DIR}/results/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${SLURM_JOB_ID:-local}"
mkdir -p "${EVAL_DIR}"

exec 1> >(tee "${EVAL_DIR}/output.log")
exec 2> >(tee "${EVAL_DIR}/error.log" >&2)

echo "Task: $EVALUATION_TASK | Agent: $AGENT | Model: $MODEL_TO_TRAIN | Hours: $NUM_HOURS | Config: $AGENT_CONFIG"
echo "Results: $EVAL_DIR"

# ── Prepare job directory ──
JOB_DIR="/tmp/posttrain_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"
mkdir -p "${JOB_DIR}/task"

cp "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${JOB_DIR}/task/"
if [ -d "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${JOB_DIR}/task/"
fi
cp -r "${POSTTRAINBENCH_DIR}/src/eval/templates" "${JOB_DIR}/task/"
if [ -d "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/task_context"/* "${JOB_DIR}/task/"
fi

# Generate prompt
BENCHMARK=$(cat "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")
PROMPT=$(python3 "${POSTTRAINBENCH_DIR}/src/eval/general/get_prompt.py" \
    --model-to-train "$MODEL_TO_TRAIN" \
    --benchmark-id "$EVALUATION_TASK" \
    --num-hours "$NUM_HOURS" \
    --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

# Generate timer script
bash "${POSTTRAINBENCH_DIR}/src/utils/create_timer.sh" "$NUM_HOURS" "${JOB_DIR}/task/timer.sh"

# Copy solve script
cp "${AGENT_REPO_DIR}/posttrain-bench/solve.sh" "${JOB_DIR}/agent_solve.sh"
chmod +x "${JOB_DIR}/agent_solve.sh"

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"
START_TIME=$(date +%s)

# Run inside Docker container via srun + pyxis
srun --overlap \
    --container-image="${DOCKER_IMAGE}" \
    --container-mounts="${JOB_DIR}:/home/ben,${HF_CACHE}:/root/.cache/huggingface" \
    --container-workdir="/home/ben/task" \
    --no-container-mount-home \
    bash -c "
        export PROMPT='$(echo "$PROMPT" | sed "s/'/'\\\\''/g")'
        export AGENT_CONFIG='${AGENT_CONFIG}'
        export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY}'
        export HF_TOKEN='${HF_TOKEN:-}'
        export HF_HOME='/root/.cache/huggingface'
        bash /home/ben/agent_solve.sh
    " > "${SOLVE_OUT}" 2>&1 || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf '%02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) > "${EVAL_DIR}/time_taken.txt"

echo "============================================"
echo "=== TASK COMPLETE ($(cat ${EVAL_DIR}/time_taken.txt)) ==="
echo "============================================"

# Copy final model out
if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
    echo "Final model saved to $EVAL_DIR/final_model"
else
    echo "WARNING: No final_model directory found"
fi

# Save task directory snapshot
cp -r "${JOB_DIR}/task" "$EVAL_DIR/task" 2>/dev/null || true

# ── Evaluation ──
echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

EVAL_COUNTER=0

run_eval() {
    local max_tokens_arg="$1"
    local eval_num="$2"
    # Kill any leftover GPU processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 5
    srun --overlap \
        --container-image="${DOCKER_IMAGE}" \
        --container-mounts="${EVAL_DIR}:/eval_dir,${HF_CACHE}:/root/.cache/huggingface,${POSTTRAINBENCH_DIR}:/ptb" \
        --container-workdir="/ptb/src/eval/tasks/${EVALUATION_TASK}" \
        --no-container-mount-home \
        bash -c "
            export HF_HOME='/root/.cache/huggingface'
            export VLLM_API_KEY='inspectai'
            python evaluate.py \
                --model-path /eval_dir/final_model \
                --templates-dir /ptb/src/eval/templates \
                --limit -1 \
                ${max_tokens_arg} \
                --json-output-file /eval_dir/metrics.json
        " > "$EVAL_DIR/final_eval_${eval_num}.txt" 2>&1 || true
}

run_eval_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        if [ -f "${EVAL_DIR}/metrics.json" ]; then return 0; fi
        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        echo "Evaluation attempt $EVAL_COUNTER"
        timeout 28800 bash -c "$(declare -f run_eval); run_eval '$max_tokens_arg' '$EVAL_COUNTER'" || true
    done
}

# Try default, then reduced tokens
run_eval_retry 4 ""

case "${EVALUATION_TASK}" in
    aime2025)     MAX_T="--max-tokens 12000" ;;
    bfcl)         MAX_T="--max-tokens 12000" ;;
    gpqamain)     MAX_T="--max-tokens 12000" ;;
    gsm8k)        MAX_T="--max-tokens 3000" ;;
    humaneval)    MAX_T="--max-tokens 3000" ;;
    arenahardwriting) MAX_T="--max-new-tokens 12288" ;;
    healthbench)  MAX_T="--max-new-tokens 12288" ;;
    *)            MAX_T="" ;;
esac
run_eval_retry 3 "$MAX_T"

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"

if [ -f "${EVAL_DIR}/metrics.json" ]; then
    cat "${EVAL_DIR}/metrics.json"
else
    echo "WARNING: No metrics.json produced"
fi
