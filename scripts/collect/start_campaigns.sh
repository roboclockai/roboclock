#!/bin/bash
# (Re)start the collection campaign(s) detached. Running episodes survive a campaign restart:
# each runs in its own session, and the new campaign skips seeds whose `.running` pid is alive.
#   bash scripts/collect/start_campaigns.sh [extra campaign.py args]
set -u
cd "$(dirname "$0")/../.."
ROOT=${ROOT:-$DEMOS_DIR/roboclock_oracle_v1}
ENVS=${ENVS:-lab,hospital,household,sports}
mkdir -p "$ROOT"
setsid nohup env PYTHONPATH=src python -u scripts/collect/campaign.py \
  --root "$ROOT" --envs "$ENVS" --gpus "${GPUS:-2,3,4,5,6,7}" --per-gpu "${PER_GPU:-1}" --stagger 30 "$@" \
  >> "$ROOT/campaign.log" 2>&1 < /dev/null &
echo "campaign pid $! -> $ROOT/campaign.log"
