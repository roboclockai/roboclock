#!/usr/bin/env bash
# Serve the Awomo-0.5 RoboDojo checkpoint (Auwomo/Awomo-0.5-Robodojo @ 5a3e043) over
# XPolicyLab's websocket protocol.
#
#   bash scripts/xpolicylab/serve_awomo.sh [robodojo_dir] [port] [gpu] [weights_dir]
#
# Needs XPolicyLab/policy/Awomo05 in the RoboDojo checkout, its uv venv at
# policy/Awomo05/.venv, and the weights laid out as the adapter's README expects
# (weights_dir/model.pt, dataset_stats.json, flux2-klein-base-4b/, flux2-dev/ae.safetensors,
# Qwen3-VL-4B-Instruct/). See docs/awomo.md. Equivalent to the adapter's
# setup_eval_policy_server.sh without its conda requirement; the checkpoint options are the
# README's (ckpt=model.pt, input_color_order=bgr, replan_steps=8).
#
# The server keeps each episode's 25 Hz frame history, and reset clears every environment:
# run one server per concurrently running episode.
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$ROBODOJO_DIR}")
PORT=${2:-6023}
GPU=${3:-0}
XPL="${ROBODOJO_DIR}/XPolicyLab"
AWOMO="${XPL}/policy/Awomo05"
WEIGHTS=$(realpath -m "${4:-${AWOMO}/weights}")

for path in "${AWOMO}/model.py" "${AWOMO}/.venv/bin/python" "${WEIGHTS}/model.pt" \
            "${WEIGHTS}/dataset_stats.json" \
            "${WEIGHTS}/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors" \
            "${WEIGHTS}/flux2-dev/ae.safetensors" "${WEIGHTS}/Qwen3-VL-4B-Instruct/config.json"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}; see docs/awomo.md (Setup)." >&2
        exit 1
    fi
done

# The pod-wide CUDA 13 ptxas override breaks the Triton of other torch builds.
unset TRITON_PTXAS_PATH
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROBODOJO_DIR}:${XPL}${PYTHONPATH:+:${PYTHONPATH}}"
export AWOMO05_WEIGHTS="${WEIGHTS}"
export AWOMO05_FLUX2_BASE="${WEIGHTS}/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors"
export AWOMO05_FLUX2_AE="${WEIGHTS}/flux2-dev/ae.safetensors"
export AWOMO05_QWEN_VL="${WEIGHTS}/Qwen3-VL-4B-Instruct"

echo "[serve] Awomo05 ${WEIGHTS}/model.pt on ws://0.0.0.0:${PORT} (GPU ${GPU})"
# ckpt_local_copy=false: the weights already sit on local disk; staging would copy 15 GB.
exec "${AWOMO}/.venv/bin/python" "${XPL}/setup_policy_server.py" \
    --config_path "${AWOMO}/deploy.yml" \
    --overrides \
        host=0.0.0.0 \
        port="${PORT}" \
        policy_name=Awomo05 \
        bench_name=RoboDojo \
        task_name=roboclock \
        ckpt_name=model \
        env_cfg_type=arx_x5 \
        action_type=joint \
        seed=0 \
        ckpt=model.pt \
        weights_dir="${WEIGHTS}" \
        input_color_order="${AWOMO05_INPUT_COLOR_ORDER:-bgr}" \
        replan_steps=8 \
        ckpt_local_copy=false
