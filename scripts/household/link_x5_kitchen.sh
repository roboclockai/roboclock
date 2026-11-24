#!/usr/bin/env bash
# Provide the Lightwheel kitchen (CC BY-NC 4.0, ~1.4 GB, not stored in git) at the path the
# Dual-X5 scenes reference. Default source: the copy scripts/household/fetch_lightwheel_kitchen.py
# unpacks; pass another verified copy to link that instead.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
src="${1:-$repo/assets/household/lightwheel_kitchen/Collected_KitchenRoom}"
dest="$repo/src/roboclock/envs/household/assets/x5/lightwheel_kitchen"
test -f "$src/KitchenRoom_A1.usdc" || { echo "Kitchen not found at $src" >&2; exit 1; }
mkdir -p "$dest"
if [ "${COPY:-0}" = "1" ]; then cp -a "$src" "$dest/"; else ln -sfn "$src" "$dest/Collected_KitchenRoom"; fi
ls -la "$dest"
