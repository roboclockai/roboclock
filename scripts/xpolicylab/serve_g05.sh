#!/usr/bin/env bash
# Serve the Galaxea G0.5 RoboDojo sim checkpoint (OpenGalaxea/g05-robodojo,
# g05_robodojo_fm_only_checkpoint.tar) over XPolicyLab's websocket protocol.
#
#   bash scripts/xpolicylab/serve_g05.sh [robodojo_dir] [port] [gpu] [checkpoint_file]
#
# Runs the adapter's own policy/G05/setup_eval_policy_server.sh (setup_policy_server.py with
# policy/G05/deploy.yml), with the uv venv at policy/G05/.venv as G05_PYTHON and FM action
# decoding, as the adapter README prescribes. See docs/g05.md.
#
# The server is stateless across requests (num_obs_steps 1, no server-side history), so
# several episodes may share one server under --inference-latency zero.
#
# A train45 role-E fine-tune (docs/five-domain-training.md) is served the same way: pass its
# checkpoint as checkpoint_file, e.g. $TRAIN45_DIR/E_g05/run1/checkpoints/step_7150.pt.
# The adapter finds run1/.hydra/config.yaml, run1/dataset_stats.json (the widened stats the run
# trained with) and run1/action_tokenizer.pt next to it. That config names classes from
# scripts/train45/E_g05/g05e_ext.py (the 256-token task cap), so that directory is put on
# PYTHONPATH here; the release checkpoint ignores it. The client must send the time-suffixed
# prompt the run was trained on: XPOLICYLAB_PROMPT_TIME=1.
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$ROBODOJO_DIR}")
PORT=${2:-6020}
GPU=${3:-4}
CHECKPOINT=$(realpath -m "${4:-$MODELS_DIR/g05-robodojo/hf_g05_robodojo_fm_only_checkpoint/checkpoints/checkpoint}")
G05="${ROBODOJO_DIR}/XPolicyLab/policy/G05"
# The checkpoint's .hydra/config.yaml reads model.model_arch.hf_processor_path from
# G05_HF_PROCESSOR_PATH with no usable default, and the sim archive ships no hf_processor/.
# The Qwen3.5 tokenizer and image-processor files come from the same repo's
# g05_robodojo_train_base_assets.tar (pretrained/hf_processor).
HF_PROCESSOR=$(realpath -m "${G05_HF_PROCESSOR_PATH:-$MODELS_DIR/g05-robodojo/g05_robodojo_train_base_assets/pretrained/hf_processor}")

for path in "${G05}/model.py" "${G05}/setup_eval_policy_server.sh" "${G05}/.venv/bin/python" \
            "${CHECKPOINT}" "$(dirname "${CHECKPOINT}")/../.hydra/config.yaml" \
            "${HF_PROCESSOR}/tokenizer.json"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}; see docs/g05.md (Setup)." >&2
        exit 1
    fi
done

# The pod-wide CUDA 13 ptxas override breaks the Triton kernels of torch 2.7.1 (fla).
unset TRITON_PTXAS_PATH
export TOKENIZERS_PARALLELISM=false
export G05_PYTHON="${G05}/.venv/bin/python"
export G05_CKPT_PATH="${CHECKPOINT}"
export G05_HF_PROCESSOR_PATH="${HF_PROCESSOR}"
# The saved config also names the training dataset root through an env var without a default.
# Serving never reads the dataset (normalisation comes from the run's dataset_stats.json), so
# any value resolves the config.
export ROBODOJO_LEROBOT_V30_ROOT="${ROBODOJO_LEROBOT_V30_ROOT:-/nonexistent/robodojo_lerobot_v30_not_needed_for_serving}"
export ROBODOJO_G05_ACTION_SOURCE=fm
# Role E's g05e_ext module (see the header); G05_EXTRA_PYTHONPATH overrides the directory.
E_G05_EXT="${G05_EXTRA_PYTHONPATH:-$(dirname "${BASH_SOURCE[0]}")/../train45/E_g05}"
if [[ -d "${E_G05_EXT}" ]]; then
    export PYTHONPATH="$(cd "${E_G05_EXT}" && pwd)${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "[serve] G05 ${CHECKPOINT} on ws://0.0.0.0:${PORT} (GPU ${GPU})"
# Arguments: bench task ckpt_name env_cfg action_type seed gpu policy_env port host.
exec bash "${G05}/setup_eval_policy_server.sh" \
    RoboDojo roboclock g05_robodojo_fm_only arx_x5 joint 0 "${GPU}" "${G05}/.venv" "${PORT}" 0.0.0.0
