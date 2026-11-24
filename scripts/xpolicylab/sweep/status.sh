#!/bin/bash
# usage: status.sh   one line per model under SWEEP_ROOT: episodes done / running / queued, slots, servers
source "$(dirname "$0")/common.sh"
echo "$(date '+%F %T')  load $(cut -d' ' -f1-3 /proc/loadavg)  (slots hold new episodes above $LOAD_MAX)"
for W in "$ROOT"/*/; do
  W=${W%/}; M=$(basename "$W")
  [ -f "$W/run_info.txt" ] || continue
  done_n=$(cat "$W/done.txt" 2>/dev/null | wc -l)
  results=$(find "$W/out" -name episodes.jsonl -size +0 2>/dev/null | wc -l)
  # episodes that ran (done.txt may list one several times after a resume) but left no result
  failed=$(awk '{print $1}' "$W/done.txt" 2>/dev/null | sort -u | while read -r n; do [ -s "$W/out/$n/episodes.jsonl" ] || echo "$n"; done | wc -l)
  queued=$(cat "$W/queue.txt" 2>/dev/null | wc -l)
  running=$(ps -eo comm=,args= | awk -v w="$W/out/" '$1 ~ /^python/ && index($0, w)' | wc -l)
  slots=0; slots_all=0
  while read -r pid _; do slots_all=$((slots_all + 1)); alive "$pid" && slots=$((slots + 1)); done < <(cat "$W/slots.txt" 2>/dev/null)
  srv=0; srv_all=0
  while read -r _ _ pid; do srv_all=$((srv_all + 1)); alive "$pid" && srv=$((srv + 1)); done < <(cat "$W/servers.txt" 2>/dev/null)
  eta=""
  if [ "$done_n" -gt 0 ] && [ "$slots" -gt 0 ]; then
    mean=$(awk -F'wall=' '{split($2, a, "s"); s += a[1]} END {print s / NR}' "$W/done.txt")
    eta=$(python3 -c "print(f'~{($queued + $running / 2) * $mean / $slots / 3600:.1f} h left')")
  fi
  printf '%-9s results %3d  no result %d  running %2d  queued %3d  slots %d/%d  servers %d/%d  %s\n' \
    "$M" "$results" "$failed" "$running" "$queued" "$slots" "$slots_all" "$srv" "$srv_all" "$eta"
done
