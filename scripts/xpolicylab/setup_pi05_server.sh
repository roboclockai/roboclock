#!/usr/bin/env bash
# One-time setup of the RoboDojo pi0.5 (XPolicyLab Pi_05) policy server on the policy machine.
#
#   bash scripts/xpolicylab/setup_pi05_server.sh [robodojo_dir] [seed]
#
# Clones RoboDojo with its XPolicyLab submodule (the IsaacLab/curobo submodules are not
# needed: roboclock is the simulator), installs the uv-managed OpenPI environment, and
# downloads the official checkpoint RoboDojo-sim-arx_x5-joint-<seed>/59999 (params and
# norm stats only, 12.4 GB; the 32 GB train_state is skipped).
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$HOME/RoboDojo}")
SEED=${2:-0}
CKPT=ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-${SEED}/59999

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required (OpenPI is uv-managed). Install it, then re-run:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ ! -d "${ROBODOJO_DIR}/.git" ]]; then
    git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git "${ROBODOJO_DIR}"
fi
git -C "${ROBODOJO_DIR}" submodule update --init XPolicyLab
echo "[setup] RoboDojo $(git -C "${ROBODOJO_DIR}" rev-parse --short HEAD)," \
    "XPolicyLab $(git -C "${ROBODOJO_DIR}/XPolicyLab" rev-parse --short HEAD)"

bash "${ROBODOJO_DIR}/XPolicyLab/policy/Pi_05/install.sh"

uvx --from 'huggingface_hub[hf_xet]' hf download RoboDojo-Benchmark/RoboDojo \
    --repo-type dataset \
    --include "${CKPT}/params/**" --include "${CKPT}/assets/**" --include "${CKPT}/_CHECKPOINT_METADATA" \
    --local-dir "${ROBODOJO_DIR}/hf"

test -f "${ROBODOJO_DIR}/hf/${CKPT}/assets/arx_x5_sim/norm_stats.json"
du -sh "${ROBODOJO_DIR}/hf/${CKPT}"
echo "[setup] done. Start the server with:"
echo "  bash scripts/xpolicylab/serve_pi05.sh ${ROBODOJO_DIR} <port> <gpu> ${SEED}"
