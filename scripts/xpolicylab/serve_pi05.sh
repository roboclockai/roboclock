#!/usr/bin/env bash
# Serve the RoboDojo pi0.5 checkpoint over XPolicyLab's websocket protocol.
#
#   bash scripts/xpolicylab/serve_pi05.sh [robodojo_dir] [port] [gpu] [seed]
#
# Binds 0.0.0.0 so the simulator machine can connect by pod IP or through a
# code-server /proxy/<port>/ route. Equivalent to XPolicyLab's
# setup_eval_policy_server.sh without its conda requirement. The first request
# triggers JAX compilation and takes noticeably longer than later ones.
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$HOME/RoboDojo}")
PORT=${2:-6000}
GPU=${3:-0}
SEED=${4:-0}
XPL="${ROBODOJO_DIR}/XPolicyLab"
PI05="${XPL}/policy/Pi_05"
MODEL="${ROBODOJO_DIR}/hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-${SEED}"

if [[ ! -d "${MODEL}/59999/params" ]]; then
    echo "Checkpoint not found at ${MODEL}; run scripts/xpolicylab/setup_pi05_server.sh first." >&2
    exit 1
fi
source "${PI05}/openpi/.venv/bin/activate"

# XPolicyLab uses 0.3 on 80 GB GPUs; pi0.5 inference needs about 14 GB.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROBODOJO_DIR}:${PI05}/openpi/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[serve] Pi_05 ${MODEL##*/}/59999 on ws://0.0.0.0:${PORT} (GPU ${GPU})"
exec python "${XPL}/setup_policy_server.py" \
    --config_path "${PI05}/deploy.yml" \
    --overrides \
        host=0.0.0.0 \
        port="${PORT}" \
        policy_name=Pi_05 \
        bench_name=RoboDojo \
        task_name=roboclock_lab \
        ckpt_name="RoboDojo-sim-arx_x5-joint-${SEED}" \
        model_path="${MODEL}" \
        env_cfg_type=arx_x5 \
        action_type=joint \
        action_dim=14 \
        seed="${SEED}"
