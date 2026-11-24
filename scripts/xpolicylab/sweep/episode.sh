#!/bin/bash
# usage: episode.sh MODEL GPU PORT ENV TASK SEED MAX_TIME   one episode at $LATENCY inference latency
# (PORT - for an API model)
source "$(dirname "$0")/common.sh"
M=$1; G=$2; P=$3; E=$4; T=$5; SEED=$6; MAXT=$7
model_cfg "$M" || exit 1
[ -n "$EPISODE_ENV" ] && export $EPISODE_ENV
W=$(model_dir "$M"); NAME=${E}_${T}_s${SEED}
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 CUDA_VISIBLE_DEVICES=$G OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=$REPO/src:$ROBODOJO
case $KIND in
  xpolicylab) export XPOLICYLAB_URL=ws://127.0.0.1:$P XPOLICYLAB_LOG=$W/qlog/$NAME.jsonl ;;
  vllm|api)
    # The primitive agent: native tool calling, thinking on (a 1024-token budget on vLLM), no SAM3.
    export VISION_LLM_LOG=$W/qlog/$NAME.jsonl VISION_LLM_THINKING=1 VISION_LLM_THINKING_BUDGET=1024
    case $T in C*) export VISION_LLM_MAX_QUERIES=$VLM_MAX_QUERIES_C ;; *) export VISION_LLM_MAX_QUERIES=$VLM_MAX_QUERIES ;; esac
    unset VISION_LLM_SAM3_URL VISION_LLM_COOKIE
    if [ "$KIND" = vllm ]; then
      export VISION_LLM_URL=http://127.0.0.1:$P/v1 VISION_LLM_MODEL=$SERVED VISION_LLM_DIALECT=vllm
      # serve_vlm.sh gives 16384 tokens of context: the tools and three views take about 9k of them
      # (Qwen3.8), each past turn about 0.7k. The agent drops old turns if a request still overflows.
      export VISION_LLM_HISTORY=${VISION_LLM_HISTORY:-4}
      export VISION_LLM_TIMEOUT=1800
      unset VISION_LLM_API_KEY
      # The server may still be loading (or restarting); wait before taking an Isaac start.
      until vllm_ready "$P"; do sleep 10; done
    else
      # Standard OpenAI fields only. The agent sends reasoning_effort (VISION_LLM_REASONING_EFFORT, default
      # medium). The provider's reasoning is not capped at the thinking budget, so the answer gets 4096
      # tokens, reasoning included (at 2048 Gemini 4 Argon ran out on 6% of its turns, 2026-11-09).
      export VISION_LLM_MAX_TOKENS=${VISION_LLM_MAX_TOKENS:-4096}
      export VISION_LLM_URL=$NV_API_URL VISION_LLM_MODEL=$API_MODEL VISION_LLM_DIALECT=openai
      # A turn takes 3 to 30 s. A request that hangs is a stalled connection: give up on it after
      # VLM_API_TIMEOUT and send it again, like any other transient failure.
      export VISION_LLM_TIMEOUT=${VLM_API_TIMEOUT:-180} VISION_LLM_RETRIES=${VLM_API_RETRIES:-6} VISION_LLM_RETRY_TIMEOUTS=1
      VISION_LLM_API_KEY=$(api_key "${SWEEP_SLOT:-0}"); export VISION_LLM_API_KEY
    fi ;;
esac
export ROBOCLOCK_INDUSTRY_LOCAL_ROOT=$W/industry_local
if [ "${SWEEP_FRAMES:-0}" = 1 ] && [ -n "$FRAMEVAR" ]; then export "$FRAMEVAR=$W/frames/$NAME"; fi
if [ "$SHARE" != yes ] && [ "$SHARE" != no ] && [ "$SLOTS_PER_GPU" -gt "$PER_GPU" ]; then export "$SHARE=1"; fi
SPEED_ARGS=()
if [ "$LATENCY" = ideal ] && [ -n "$MODEL_SPEEDS" ]; then SPEED_ARGS=(--model-speeds "$MODEL_SPEEDS"); fi
timeout_s=$(python3 -c "print(int(900 + float('$MAXT') * $WALL_PER_SIM))")
cd "$REPO"
for attempt in 1 2 3 4 5; do
  rm -rf "$W/out/$NAME" "$W/qlog/$NAME.jsonl" "$W/frames/$NAME"
  # Hold the start lock through the stagger and until the load is below LOAD_MAX. The check is then fresh
  # when Isaac starts; checked before queueing for the lock, it goes stale and the load overshoots.
  # Above 2/3 of LOAD_MAX the start also waits START_SETTLE more: an Isaac episode takes 1-2 min to reach
  # its full CPU use and the 1-min load lags that, so starts every START_STAGGER overshoot LOAD_MAX.
  flock "$START_LOCK" bash -c "sleep $START_STAGGER
    if [ \$(awk '{print int(\$1)}' /proc/loadavg) -gt $((LOAD_MAX * 2 / 3)) ]; then sleep $START_SETTLE; fi
    while [ \$(awk '{print int(\$1)}' /proc/loadavg) -gt $LOAD_MAX ]; do sleep 15; done"
  start=$(date +%s)
  timeout "$timeout_s" "$PY" -m roboclock eval --env "$E" --task "$T" --difficulty "$DIFFICULTY" \
    --seeds 1 --seed-start "$SEED" --policy "$POLICY" $POLICY_ARGS --inference-latency "$LATENCY" --control-delay "$CONTROL_DELAY" \
    "${SPEED_ARGS[@]}" --output "$W/out/$NAME" < /dev/null > "$W/logs/$NAME.log" 2>&1
  rc=$?
  # Isaac sometimes cannot see the GPU at start-up. Nothing ran, so start again.
  grep -q "Vulkan cannot enumerate a GPU" "$W/logs/$NAME.log" || break
  echo "$NAME attempt $attempt $(date +%T) vulkan" >> "$W/vulkan_failures.txt"
  sleep 60
done
echo "$NAME rc=$rc wall=$(( $(date +%s) - start ))s gpu=$G port=$P attempt=$attempt timeout=${timeout_s}s" >> "$W/done.txt"
