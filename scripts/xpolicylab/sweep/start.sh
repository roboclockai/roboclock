#!/bin/bash
# usage: start.sh ROLE | MODEL:GPU[,GPU...] ...   (ROLE: A, B, fast or vlm; docs/robodojo-sweep.md, docs/vision-llm-sweep.md)
# Starts each model's policy servers and episode slots, all detached from this shell, and returns.
# Watch with status.sh, stop with stop.sh. Running start.sh again after a stop resumes the sweep:
# episodes that already have a result are not queued again.
set -eu
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { sed -n 2,5p "$0"; exit 1; }
specs=$(role_specs "$@")
for spec in $specs; do
  M=${spec%%:*}
  [ "$M" != "$spec" ] || { echo "bad spec '$spec' (want MODEL:GPU,GPU,...)"; exit 1; }
  model_cfg "$M"
  while read -r pid _; do
    if alive "$pid"; then echo "$M is still running (slot $pid); stop.sh $M first"; exit 1; fi
  done < <(cat "$(model_dir "$M")/slots.txt" 2>/dev/null)
  while read -r P _ pid; do
    if alive "$pid"; then echo "$M still has a server on port $P (pid $pid); stop.sh $M first"; exit 1; fi
  done < <(cat "$(model_dir "$M")/servers.txt" 2>/dev/null)
done
mkdir -p "$ROOT"
setsid nohup bash "$SWEEP_DIR/launch.sh" $specs >> "$ROOT/launch.log" 2>&1 < /dev/null &
echo "launcher pid $!, log $ROOT/launch.log"
echo "servers take 1-6 min to load; then one Isaac starts every ${START_STAGGER} s"
