#!/bin/bash
# Run by start.sh, detached: build the queues, start the servers, wait for them, start the slots.
source "$(dirname "$0")/common.sh"
export REPO SIM_PYTHON=$PY ENVS TASKS FIRST_TASKS DIFFICULTY SEEDS SEED_START JOBS_FILE
cd "$REPO" || exit 1
echo "== $(date '+%F %T') launch $*"
for spec in "$@"; do
  M=${spec%%:*}; model_cfg "$M" || exit 1
  IFS=, read -ra G <<< "${spec#*:}"
  W=$(model_dir "$M")
  n=0  # servers
  case $KIND in
    xpolicylab) n=$(( ${#G[@]} * PER_GPU )) ;;
    vllm)
      if [ $(( ${#G[@]} % TP )) -ne 0 ]; then echo "$M: $TP GPUs per server, so list a multiple of $TP GPUs"; exit 1; fi
      n=$(( ${#G[@]} / TP )) ;;
    api)
      IFS=, read -ra lines <<< "$NV_API_KEY_INDEX"
      for k in $(seq 0 $(( ${#lines[@]} - 1 ))); do
        if [ -z "$(api_key "$k")" ]; then echo "$M: no API key (NVIDIA_API_KEY, or line ${lines[$k]} of $NV_API_KEY_FILE)"; exit 1; fi
      done ;;
  esac
  # A server times each query from before its own queue (XPolicyLab's latency_ms) or the client times the
  # round trip (vision-LLM agent), so under measured latency an episode sharing a server is charged the other
  # episodes' inference too: one episode per server. A hosted API's round trip includes the network.
  if [ "$LATENCY" = measured ] && [ "$KIND" != api ] && [ $(( ${#G[@]} * SLOTS_PER_GPU )) -gt "$n" ]; then
    echo "$M: measured latency needs one episode per server (lower SWEEP_SLOTS_PER_GPU)"; exit 1
  fi
  if [ "$SHARE" = no ] && [ "$SLOTS_PER_GPU" -gt "$PER_GPU" ]; then
    echo "$M: $SLOTS_PER_GPU slots per GPU need as many servers ($PER_GPU); fix model_cfg"; exit 1
  fi
  mkdir -p "$W/servers" "$W/out" "$W/logs" "$W/qlog"
  rm -f "$W/STOP"
  # Ideal latency needs this model's speed in the speeds file; check it before any server starts.
  if [ "$LATENCY" = ideal ] && [ "$KIND" != xpolicylab ]; then
    name=${SERVED:-$API_MODEL}
    if ! PYTHONPATH="$REPO/src" "$PY" -c "import sys; from roboclock.evaluation.evaluator import load_model_speeds as l
s = l(sys.argv[1] or None); sys.exit(0 if sys.argv[2] in s else 1)" "$MODEL_SPEEDS" "$name" 2>/dev/null; then
      echo "$M: LATENCY=ideal but the speeds file (${MODEL_SPEEDS:-the default model_speeds.yaml}) has no speed for $name"; exit 1
    fi
  fi
  q=$(python3 "$SWEEP_DIR/queue.py" "$W/out" "$W/queue.txt" "$M" 2>&1) || { echo "$M: queue.py failed: $q"; exit 1; }
  echo "$M: $q"
  {
    echo "started $(date '+%F %T') on $(hostname)"
    dirty=$(git -C "$REPO" status --porcelain -- src scripts 2>/dev/null | head -n1)
    echo "roboclock $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)${dirty:+ +local changes}"
    case $KIND in
      xpolicylab)
        echo "xpolicylab $(git -C "$ROBODOJO/XPolicyLab" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "layout $spec, $PER_GPU servers and $SLOTS_PER_GPU episodes per GPU" ;;
      vllm)
        echo "model $SERVED from $MODELS_DIR/$VLM_DIR on vLLM ($("$VLLM_PYTHON" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)), TP $TP"
        echo "agent $POLICY $POLICY_ARGS, native tool calling ($TOOL_PARSER parser), thinking budget 1024, ${VISION_LLM_HISTORY:-4} past turns in view, at most $VLM_MAX_QUERIES queries per episode ($VLM_MAX_QUERIES_C for C tasks)"
        echo "layout $spec, one server per $TP GPU(s), $SLOTS_PER_GPU episodes per GPU" ;;
      api)
        if [ -n "${NVIDIA_API_KEY:-}" ]; then key="NVIDIA_API_KEY"; else key="line(s) $NV_API_KEY_INDEX of $NV_API_KEY_FILE, one per slot in turn"; fi
        echo "model $API_MODEL at $NV_API_URL (key from $key)"
        echo "agent $POLICY $POLICY_ARGS, native tool calling, openai dialect, reasoning_effort ${VISION_LLM_REASONING_EFFORT:-medium}, max_tokens ${VISION_LLM_MAX_TOKENS:-4096}, ${VISION_LLM_HISTORY:-6} past turns in view, at most $VLM_MAX_QUERIES queries per episode ($VLM_MAX_QUERIES_C for C tasks)"
        echo "layout $spec, $SLOTS_PER_GPU episodes per GPU" ;;
    esac
    echo "episodes $DIFFICULTY, seeds $SEED_START..$((SEED_START + SEEDS - 1)), envs $ENVS${TASKS:+, tasks $TASKS}, $LATENCY inference latency, control delay ${CONTROL_DELAY} s"
    if [ "$LATENCY" = ideal ]; then
      f=${MODEL_SPEEDS:-$REPO/src/roboclock/evaluation/model_speeds.yaml}
      echo "model speeds $f (sha256 $(sha256sum "$f" 2>/dev/null | cut -c1-16))"
    fi
    if [ -n "$JOBS_FILE" ]; then echo "jobs $JOBS_FILE (sha256 $(sha256sum "$JOBS_FILE" 2>/dev/null | cut -c1-16))"; fi
  } >> "$W/run_info.txt"
  for P in $(seq $PORT_BASE $((PORT_BASE + n - 1))); do
    if port_open $P; then echo "$M: port $P is already in use; stop whatever holds it"; exit 1; fi
  done
  : > "$W/servers.txt"
  i=0
  case $KIND in
    xpolicylab)
      for g in "${G[@]}"; do
        for _ in $(seq "$PER_GPU"); do
          P=$((PORT_BASE + i)); i=$((i + 1))
          # Started straight from this shell, so $! is the server itself and leads its own process group.
          env $SERVE_ENV setsid nohup bash "scripts/xpolicylab/$SERVE" "$ROBODOJO" $P "$g" $SERVE_ARGS \
            > "$W/servers/$P.log" 2>&1 < /dev/null &
          echo "$P $g $!" >> "$W/servers.txt"
          echo "$M: server port $P on GPU $g"
        done
      done ;;
    vllm)
      for i in $(seq 0 $((n - 1))); do
        P=$((PORT_BASE + i)); gpus=$(IFS=,; echo "${G[*]:$((i * TP)):$TP}")
        setsid nohup bash "$SWEEP_DIR/serve_vlm.sh" "$M" $P "$gpus" > "$W/servers/$P.log" 2>&1 < /dev/null &
        echo "$P $gpus $!" >> "$W/servers.txt"
        echo "$M: server port $P on GPUs $gpus"
      done ;;
  esac
done
for spec in "$@"; do
  M=${spec%%:*}; model_cfg "$M"
  IFS=, read -ra G <<< "${spec#*:}"
  W=$(model_dir "$M"); ok=1
  while read -r P g pid; do
    ready() { if [ "$KIND" = vllm ]; then vllm_ready "$1"; else port_open "$1"; fi; }
    for _ in $(seq 180); do ready $P && break; alive "$pid" || break; sleep 10; done
    ready $P || { echo "$M: server $P did not come up; end of its log:"; tail -n 15 "$W/servers/$P.log"; ok=0; }
  done < "$W/servers.txt"
  [ $ok = 1 ] || { echo "$M: no slots started"; continue; }
  # A fresh policy server is slow on its first query (pi0.5 compiles; InternW0 takes ~1.2 s more);
  # take it here, not in an episode: under LATENCY=measured an episode is charged the server's time.
  if [ "$KIND" = xpolicylab ] && [ "${SWEEP_WARMUP:-1}" = 1 ]; then
    warm=()  # not a bare wait: the servers are this shell's background jobs too
    while read -r P g _; do
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$REPO/src:$ROBODOJO" timeout 600 "$PY" "$SWEEP_DIR/warmup.py" "$P" "$POLICY" 2>&1 | tail -n 1 &
      warm+=($!)
    done < "$W/servers.txt"
    wait "${warm[@]}"
  fi
  : > "$W/slots.txt"
  slot=0  # numbers the model's slots: an API model's slots take its keys in turn
  for g in "${G[@]}"; do
    if [ "$KIND" = api ]; then
      ports=(-)
    else  # the servers whose GPUs include g
      ports=($(awk -v g="$g" '{n = split($2, a, ","); for (i = 1; i <= n; i++) if (a[i] == g) print $1}' "$W/servers.txt"))
    fi
    for j in $(seq 0 $((SLOTS_PER_GPU - 1))); do
      P=${ports[$((j % ${#ports[@]}))]}
      setsid nohup bash "$SWEEP_DIR/slot.sh" "$M" "$g" $P $slot > "$W/slot_g${g}_$j.log" 2>&1 < /dev/null &
      echo "$! $g $P" >> "$W/slots.txt"
      slot=$((slot + 1))
    done
  done
  echo "$M: $(wc -l < "$W/slots.txt") slots started"
done
echo "== $(date '+%F %T') launch done"
