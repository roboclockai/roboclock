#!/bin/bash
# usage: stop.sh [MODEL...]   kill the slots, running episodes and servers of MODEL (default: every model
# under SWEEP_ROOT). Killed episodes have no result, so start.sh queues them again on resume.
# To let running episodes finish instead: touch $SWEEP_ROOT/<model>/STOP, wait, then run stop.sh.
# Kills only PIDs this sweep recorded, plus Python processes whose command line names the model's
# output directory; never a pattern match on the whole process table.
source "$(dirname "$0")/common.sh"
models=("$@")
[ ${#models[@]} -gt 0 ] || for W in "$ROOT"/*/; do [ -f "$W/run_info.txt" ] && models+=("$(basename "$W")"); done
tree() { local c; for c in $(ps -o pid= --ppid "$1"); do echo "$c"; tree "$c"; done; }
for M in "${models[@]}"; do
  W=$(model_dir "$M")
  [ -d "$W" ] || { echo "$M: no $W"; continue; }
  touch "$W/STOP"
  pids=""
  while read -r pid _; do alive "$pid" && pids="$pids $pid $(tree "$pid")"; done < <(cat "$W/slots.txt" 2>/dev/null)
  pids="$pids $(ps -eo pid=,comm=,args= | awk -v w="$W/out/" '$2 ~ /^python/ && index($0, w) {print $1}')"
  n=$(echo $pids | wc -w)
  [ "$n" -gt 0 ] && kill -9 $pids 2>/dev/null
  echo "$M: killed $n slot and episode processes"
  while read -r P g pid; do
    alive "$pid" || continue
    if [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pid" ]; then
      kill -- -"$pid" 2>/dev/null; sleep 5; kill -9 -- -"$pid" 2>/dev/null
    else
      kill -9 "$pid" $(tree "$pid") 2>/dev/null
    fi
    echo "$M: stopped server port $P (GPU $g, pid $pid)"
  done < <(cat "$W/servers.txt" 2>/dev/null)
done
