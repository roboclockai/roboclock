#!/usr/bin/env bash
# Serve the Wall-X RoboDojo checkpoint (x-square-robot/wall-oss-1.0-lite-robodojo) over
# XPolicyLab's websocket protocol.
#
#   bash scripts/xpolicylab/serve_wallx.sh [robodojo_dir] [port] [gpu] [checkpoint_dir]
#
# Needs XPolicyLab/policy/WallX (XPolicyLab main, 2026-11-04 or later) in the RoboDojo
# checkout, its uv venv at policy/WallX/.venv and the wall-x serving checkout (WALLX_REPO,
# default <robodojo_dir>/../wall-x). See docs/wallx.md. Equivalent to the adapter's
# setup_eval_policy_server.sh without its conda requirement. One server per concurrently
# running episode: the server's latency_ms includes the wait for its model lock.
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$ROBODOJO_DIR}")
PORT=${2:-6010}
GPU=${3:-0}
CHECKPOINT=$(realpath -m "${4:-$MODELS_DIR/wall-oss-1.0-lite-robodojo/checkpoint}")
XPL="${ROBODOJO_DIR}/XPolicyLab"
WALLX="${XPL}/policy/WallX"
WALLX_REPO=$(realpath -m "${WALLX_REPO:-${ROBODOJO_DIR}/../wall-x}")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for path in "${WALLX}/model.py" "${WALLX}/.venv/bin/python" "${CHECKPOINT}/model.safetensors" \
            "${WALLX_REPO}/wall_x/serving/robodojo/policy.py"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}; see docs/wallx.md (Setup)." >&2
        exit 1
    fi
done

# The pod-wide CUDA 13 ptxas override breaks Triton kernels of other torch builds.
unset TRITON_PTXAS_PATH
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROBODOJO_DIR}:${XPL}:${WALLX_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[serve] WallX ${CHECKPOINT} on ws://0.0.0.0:${PORT} (GPU ${GPU})"
exec "${WALLX}/.venv/bin/python" "${HERE}/wallx_server.py" "${XPL}/setup_policy_server.py" \
    --config_path "${WALLX}/deploy.yml" \
    --overrides \
        host=0.0.0.0 \
        port="${PORT}" \
        policy_name=WallX \
        bench_name=RoboDojo \
        task_name=roboclock \
        ckpt_name=wall-oss-1.0-lite-robodojo \
        env_cfg_type=arx_x5 \
        action_type=ee \
        seed=0 \
        checkpoint_path="${CHECKPOINT}"
