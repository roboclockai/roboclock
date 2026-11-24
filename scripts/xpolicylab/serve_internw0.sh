#!/usr/bin/env bash
# Serve the InternW0-Delta RoboDojo checkpoint (InternRobotics/InternW0-Delta-RoboDojo,
# robodojo.pt) over XPolicyLab's websocket protocol.
#
#   bash scripts/xpolicylab/serve_internw0.sh [robodojo_dir] [port] [gpu]
#
# Equivalent to the adapter's setup_eval_policy_server.sh for the released weights
# (eval.sh RoboDojo <task> robodojo arx_x5 joint 0 ...), with the adapter's uv venv at
# policy/InternW0_delta/.venv instead of its conda env. Weights: checkpoints/robodojo.pt and
# assets/{Wan-AI/Wan2.2-TI2V-5B,Alibaba-DAMO-Academy/RynnBrain1.1-2B} inside the adapter
# (download_checkpoint.sh, download_assets.sh). See docs/internw0.md.
#
# The server is stateful: one session per env_idx (memory frames, pending actions, step
# counter), and a reset clears every session. Serve one server per concurrently running
# episode. Its latency_ms includes the wait for its model lock.
#
# A fine-tune (docs/five-domain-training.md, role D) is served by three variables, each defaulting
# to the released file:
#   WAM_CHECKPOINT_PATH      weights, e.g. <run>/checkpoints/weights/slot_0.pt
#   WAM_DATASET_STATS_PATH   the run's normalization stats, <run>/dataset_stats.json
#   WAM_TRAIN_CONFIG_PATH    the eval model config, e.g. eval_model_len256.yaml (256-token text context)
set -euo pipefail

ROBODOJO_DIR=$(realpath -m "${1:-$ROBODOJO_DIR}")
PORT=${2:-6024}
GPU=${3:-3}
XPL="${ROBODOJO_DIR}/XPolicyLab"
POLICY="${XPL}/policy/InternW0_delta"

checkpoint="${WAM_CHECKPOINT_PATH:-${POLICY}/checkpoints/robodojo.pt}"
stats="${WAM_DATASET_STATS_PATH:-${POLICY}/config/dataset_stats.json}"
model_config="${WAM_TRAIN_CONFIG_PATH:-${POLICY}/config/eval_model.yaml}"
for path in "${POLICY}/model.py" "${POLICY}/.venv/bin/python" "${checkpoint}" "${stats}" "${model_config}" \
            "${POLICY}/assets/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
            "${POLICY}/assets/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
            "${POLICY}/assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B/model.safetensors"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}; see docs/internw0.md (Setup)." >&2
        exit 1
    fi
done

# The pod-wide CUDA 13 ptxas override breaks the Triton of torch cu128 (FLA kernels).
unset TRITON_PTXAS_PATH
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${XPL}:${POLICY}/wam_runtime/src${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_MODEL_BASE_PATH="${POLICY}/assets"

echo "[serve] InternW0_delta ${checkpoint} (stats ${stats}, config ${model_config}) on ws://0.0.0.0:${PORT} (GPU ${GPU})"
cd "${POLICY}"
exec "${POLICY}/.venv/bin/python" -u "${XPL}/setup_policy_server.py" \
    --config_path "${POLICY}/deploy.yml" \
    --overrides \
        host=0.0.0.0 \
        port="${PORT}" \
        bench_name=RoboDojo \
        task_name=roboclock \
        ckpt_name=robodojo \
        env_cfg_type=arx_x5 \
        seed=0 \
        policy_name=InternW0_delta \
        action_type=joint \
        checkpoint_path="${checkpoint}" \
        dataset_stats_path="${stats}" \
        train_config_path="${model_config}" \
        base_model_dir="${POLICY}/assets/Wan-AI/Wan2.2-TI2V-5B" \
        vlm_model_path="${POLICY}/assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B" \
        wam_root="${POLICY}/wam_runtime" \
        allow_dummy_policy=false
