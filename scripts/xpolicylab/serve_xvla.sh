#!/usr/bin/env bash
# Serve the X-VLA RoboDojo checkpoint (RoboDojo-Benchmark/RoboDojo dataset,
# ckpt/RoboDojo/X_VLA/RoboDojo-sim-arx_x5-ee-0/ckpt-100000) over XPolicyLab's websocket protocol.
#
#   bash scripts/xpolicylab/serve_xvla.sh [robodojo_dir] [port] [gpu] [ckpt_name]
#
# Equivalent to the adapter's setup_eval_policy_server.sh (bench RoboDojo, ckpt "sim",
# arx_x5, ee, seed 0) without its conda requirement: the adapter's uv venv is
# policy/X_VLA/.venv. The checkpoint resolves through policy/X_VLA/checkpoints/
# RoboDojo-<ckpt_name>-arx_x5-ee-0 (a symlink to the dataset download) and the processor
# through deploy.yml's processor_path, checkpoints/shared/X-VLA-Pt. See docs/xvla.md.
#
# The model keeps no per-episode state between `infer` messages (update_obs + get_action run
# under one model lock), so several episodes may share one server.
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$ROBODOJO_DIR}")
PORT=${2:-6021}
GPU=${3:-7}
CKPT_NAME=${4:-sim}
XPL="${ROBODOJO_DIR}/XPolicyLab"
XVLA="${XPL}/policy/X_VLA"
RUN_DIR="${XVLA}/checkpoints/RoboDojo-${CKPT_NAME}-arx_x5-ee-0"

for path in "${XVLA}/model.py" "${XVLA}/.venv/bin/python" "${RUN_DIR}/ckpt-100000/model.safetensors" \
            "${XVLA}/checkpoints/shared/X-VLA-Pt/preprocessor_config.json"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}; see docs/xvla.md (Setup)." >&2
        exit 1
    fi
done

# The pod-wide CUDA 13 ptxas override breaks Triton kernels of other torch builds.
unset TRITON_PTXAS_PATH
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROBODOJO_DIR}:${XPL}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[serve] X_VLA ${RUN_DIR} on ws://0.0.0.0:${PORT} (GPU ${GPU})"
cd "${XPL}"
exec "${XVLA}/.venv/bin/python" "${XPL}/setup_policy_server.py" \
    --config_path "${XVLA}/deploy.yml" \
    --overrides \
        host=0.0.0.0 \
        port="${PORT}" \
        policy_name=X_VLA \
        bench_name=RoboDojo \
        task_name=roboclock \
        ckpt_name="${CKPT_NAME}" \
        env_cfg_type=arx_x5 \
        action_type=ee \
        seed=0 \
        action_dim=14
