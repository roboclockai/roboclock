#!/usr/bin/env bash
# Serve VPP2 (Video Prediction Policy 2, joint Video + Action2B 100k bundle, absolute EE16)
# over XPolicyLab's websocket protocol.
#
#   bash scripts/xpolicylab/serve_vpp2.sh [robodojo_dir] [port] [gpu]
#
# Needs XPolicyLab/policy/VPP2 with its uv venv at policy/VPP2/.venv, the upstream code in
# policy/VPP2/upstream (install.sh's pinned clone) and the eval bundle plus Wan2.1 encoders in
# policy/VPP2/checkpoints (download_checkpoints.sh, eval stage). See docs/vpp2.md.
#
# Equivalent to the adapter's setup_eval_policy_server.sh without its conda requirement: it
# runs the adapter's launch_policy.py, which validates the bundle and then execs XPolicyLab's
# setup_policy_server.py with deploy.yml and the released inference settings (10 Euler steps,
# sigma shift 1, replan 24, seed 1). The server keeps the observation history per client id,
# so several episodes may share it, but its latency_ms includes the wait for its model lock.
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$ROBODOJO_DIR}")
PORT=${2:-6022}
GPU=${3:-2}
VPP2="${ROBODOJO_DIR}/XPolicyLab/policy/VPP2"

for path in "${VPP2}/launch_policy.py" "${VPP2}/.venv/bin/python" "${VPP2}/upstream/src/vpp2/policy.py" \
            "${VPP2}/checkpoints/joint2b_s100000/manifest.json" \
            "${VPP2}/checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}; see docs/vpp2.md (Setup)." >&2
        exit 1
    fi
done

# The pod-wide CUDA 13 ptxas override breaks Triton kernels of other torch builds.
unset TRITON_PTXAS_PATH
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning TOKENIZERS_PARALLELISM=false

echo "[serve] VPP2 joint2b_s100000 on ws://0.0.0.0:${PORT} (GPU ${GPU})"
exec "${VPP2}/.venv/bin/python" -u "${VPP2}/launch_policy.py" \
    --bench RoboDojo --task roboclock --ckpt joint2b_s100000 \
    --env-cfg arx_x5 --action-type ee --seed 1 \
    --gpu "${GPU}" --port "${PORT}" --host 0.0.0.0 \
    --steps "${VPP2_NUM_INFERENCE_STEPS:-10}" --shift "${VPP2_SIGMA_SHIFT:-1}" \
    --replan "${VPP2_REPLAN_STEPS:-24}"
