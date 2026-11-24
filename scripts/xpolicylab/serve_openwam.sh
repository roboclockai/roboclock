#!/usr/bin/env bash
# Serve the OpenWAM-alpha RoboDojo checkpoint (OpenWAM/OpenWAM-Alpha-Sim-RoboDojo) over
# XPolicyLab's websocket protocol.
#
#   bash scripts/xpolicylab/serve_openwam.sh [robodojo_dir] [port] [gpu] [checkpoint_dir]
#
# Needs XPolicyLab/policy/OpenWAM (with its vendored OpenWAM/ tree) in the RoboDojo checkout
# and its uv venv at policy/OpenWAM/.venv. See docs/openwam.md. Equivalent to the adapter's
# setup_eval_policy_server.sh without its conda requirement: same setup_policy_server.py,
# deploy.yml and overrides. The checkpoint is self-contained (Wan2.2-TI2V-5B DiT + VAE,
# UMT5-XXL text encoder and tokenizer, ActionDiT, normalisation stats).
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$ROBODOJO_DIR}")
PORT=${2:-6025}
GPU=${3:-5}
CHECKPOINT=$(realpath -m "${4:-$MODELS_DIR/OpenWAM-Alpha-Sim-RoboDojo}")
XPL="${ROBODOJO_DIR}/XPolicyLab"
POLICY="${XPL}/policy/OpenWAM"
OPENWAM_ROOT="${POLICY}/OpenWAM"

for path in "${POLICY}/model.py" "${POLICY}/.venv/bin/python" "${CHECKPOINT}/config.yaml" \
            "${CHECKPOINT}/normalization_stats.npy" "${OPENWAM_ROOT}/configs/deploy.yaml"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}; see docs/openwam.md (Setup)." >&2
        exit 1
    fi
done
ACTION_DIM=$(bash "${XPL}/utils/get_action_dim.sh" "${ROBODOJO_DIR}" arx_x5)

# The pod-wide CUDA 13 ptxas override breaks Triton kernels of other torch builds.
unset TRITON_PTXAS_PATH
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROBODOJO_DIR}:${OPENWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[serve] OpenWAM ${CHECKPOINT} on ws://0.0.0.0:${PORT} (GPU ${GPU})"
exec "${POLICY}/.venv/bin/python" -u "${XPL}/setup_policy_server.py" \
    --config_path "${POLICY}/deploy.yml" \
    --overrides \
        port="${PORT}" \
        host=0.0.0.0 \
        bench_name=RoboDojo \
        task_name=roboclock \
        ckpt_name=OpenWAM-Alpha-Sim-RoboDojo \
        env_cfg_type=arx_x5 \
        seed=0 \
        policy_name=OpenWAM \
        action_type=ee \
        action_dim="${ACTION_DIM}" \
        openwam_root="${OPENWAM_ROOT}" \
        ckpt_dir="${CHECKPOINT}" \
        openwam_deploy_config="${OPENWAM_DEPLOY_CONFIG:-${OPENWAM_ROOT}/configs/deploy.yaml}" \
        allow_dummy_policy=false
