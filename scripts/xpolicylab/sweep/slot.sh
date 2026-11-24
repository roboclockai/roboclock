#!/bin/bash
# usage: slot.sh MODEL GPU PORT [SLOT]   (PORT - for an API model) run MODEL's queued episodes one at a time until the queue is empty
# SLOT is the slot's number among the model's slots (default 0); an API model's slots take its keys in turn.
# A file named STOP in the model's directory makes the slot exit after its current episode.
source "$(dirname "$0")/common.sh"
M=$1; G=$2; P=$3; export SWEEP_SLOT=${4:-0}; W=$(model_dir "$M")
while true; do
  while [ "$(awk '{print int($1)}' /proc/loadavg)" -gt "$LOAD_MAX" ]; do sleep 30; done
  [ -e "$W/STOP" ] && break
  job=$(flock "$W/queue.lock" bash -c "head -n1 '$W/queue.txt'; sed -i 1d '$W/queue.txt'")
  [ -z "$job" ] && break
  echo "$(date +%T) start $job gpu=$G port=$P"
  bash "$SWEEP_DIR/episode.sh" "$M" "$G" "$P" $job
done
echo "$(date +%T) slot done"
