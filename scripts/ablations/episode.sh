#!/bin/bash
# usage: episode.sh MODEL GPU ENV TASK DIFFICULTY SEED BUDGET_L MAX_TIME NAME OUTDIR [CONFIG]
#
# One closed-loop ablation episode (docs/ablations.md). It is the sweep's episode.sh with the
# experiment's three extra arguments - the tier's generated configuration, the tier's level and
# the inference budget L - and without the local-vLLM and RoboDojo model kinds: experiment 1
# runs the vision-LLM primitive agent on a hosted API or a local vLLM server.
#
# The delay charged per model query is L + ABL_INTERFACE_DELAY; `roboclock eval` does that sum
# itself and records L, delta and D separately, so nothing here adds them up.
source "$(dirname "$0")/../xpolicylab/sweep/common.sh"
M=$1; G=$2; E=$3; T=$4; D=$5; SEED=$6; L=$7; MAXT=$8; NAME=$9; OUT=${10}; CFG=${11:-}
# "oracle" is not a model: it is the feasibility diagnostic (docs/ablations.md). It runs the
# same tiers, configurations and seeds as the matrix, queries nothing and is charged no delay,
# and its results are kept in their own directory, never mixed into the matrix.
if [ "$M" = oracle ]; then
  KIND=oracle; POLICY=oracle; POLICY_ARGS="--oracle"; WALL_PER_SIM=20
else
  model_cfg "$M" || exit 1
  case $KIND in
    vllm|api) ;;
    *) echo "ablations: model '$M' is not a vision-LLM model (kind=$KIND)" >&2; exit 1 ;;
  esac
fi

export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 CUDA_VISIBLE_DEVICES=$G
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=$REPO/src
mkdir -p "$(dirname "$OUT")" "$ABL_QLOG" "$ABL_LOGS" 2>/dev/null
if [ "$KIND" = oracle ]; then
  :
else
export VISION_LLM_LOG=$ABL_QLOG/$NAME.jsonl VISION_LLM_EPISODE=$NAME
export VISION_LLM_THINKING=1 VISION_LLM_THINKING_BUDGET=1024
case $T in C*) export VISION_LLM_MAX_QUERIES=$VLM_MAX_QUERIES_C ;;
           *) export VISION_LLM_MAX_QUERIES=$VLM_MAX_QUERIES ;; esac
unset VISION_LLM_SAM3_URL VISION_LLM_COOKIE
if [ "$KIND" = vllm ]; then
  export VISION_LLM_URL=http://127.0.0.1:${ABL_VLLM_PORT:-$PORT_BASE}/v1 VISION_LLM_MODEL=$SERVED
  export VISION_LLM_DIALECT=vllm VISION_LLM_HISTORY=${VISION_LLM_HISTORY:-4} VISION_LLM_TIMEOUT=1800
  unset VISION_LLM_API_KEY
  until vllm_ready "${ABL_VLLM_PORT:-$PORT_BASE}"; do sleep 10; done
else
  export VISION_LLM_MAX_TOKENS=${VISION_LLM_MAX_TOKENS:-4096}
  export VISION_LLM_URL=$NV_API_URL VISION_LLM_MODEL=$API_MODEL VISION_LLM_DIALECT=openai
  export VISION_LLM_TIMEOUT=${VLM_API_TIMEOUT:-180} VISION_LLM_RETRIES=${VLM_API_RETRIES:-6}
  export VISION_LLM_RETRY_TIMEOUTS=1
  VISION_LLM_API_KEY=$(api_key "${SWEEP_SLOT:-0}"); export VISION_LLM_API_KEY
fi
fi
export ROBOCLOCK_INDUSTRY_LOCAL_ROOT=${ABL_ROOT}/industry_local
[ "$SHARE" != yes ] && [ "$SHARE" != no ] && [ -n "$SHARE" ] && export "$SHARE=1"

CFG_ARGS=(); [ -n "$CFG" ] && CFG_ARGS=(--config "$CFG")
timeout_s=$(python3 -c "print(int(900 + float('$MAXT') * $WALL_PER_SIM))")
cd "$REPO"
for attempt in 1 2 3; do
  rm -rf "$OUT" "$ABL_QLOG/$NAME.jsonl"
  flock "$START_LOCK" bash -c "sleep $START_STAGGER; while [ \$(awk '{print int(\$1)}' /proc/loadavg) -gt $LOAD_MAX ]; do sleep 15; done"
  start=$(date +%s)
  timeout "$timeout_s" "$PY" -m roboclock eval --env "$E" --task "$T" --difficulty "$D" \
    --seeds 1 --seed-start "$SEED" --policy "$POLICY" $POLICY_ARGS \
    --inference-latency zero --time-model block \
    --inference-budget "$L" --interface-delay "$([ "$KIND" = oracle ] && echo 0 || echo "${ABL_INTERFACE_DELAY:-0.02}")" \
    "${CFG_ARGS[@]}" --output "$OUT" < /dev/null > "$ABL_LOGS/$NAME.log" 2>&1
  rc=$?
  grep -q "Vulkan cannot enumerate a GPU" "$ABL_LOGS/$NAME.log" || break
  echo "$NAME attempt $attempt $(date +%T) vulkan" >> "$ABL_ROOT/vulkan_failures.txt"
  sleep 60
done
echo "$NAME rc=$rc wall=$(( $(date +%s) - start ))s gpu=$G attempt=$attempt timeout=${timeout_s}s" >> "$ABL_ROOT/done.txt"
exit $rc
