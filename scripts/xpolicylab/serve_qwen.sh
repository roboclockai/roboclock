#!/usr/bin/env bash
# Serve Qwen3.8-27B through vLLM's OpenAI-compatible API for roboclock.policies.vision_llm.
#
#   bash scripts/xpolicylab/serve_qwen.sh [model_dir] [port] [gpu]
#
# The weights take about 55 GB of GPU memory. Set VLLM_BIND=0.0.0.0 to reach the server from
# another machine; vLLM's API has no authentication unless --api-key is added. VLLM_GPU_UTIL
# (default 0.92) is the fraction of GPU memory vLLM may take. VLLM_MAX_NUM_SEQS (default 64) caps
# concurrent sequences: the model's linear-attention layers need one cache block per sequence,
# and vLLM's default of 256 does not fit at lower memory fractions. VLLM_MAX_MODEL_LEN (default
# 32768) can be lowered when memory is short; a vision_llm request uses about 3.5k tokens.
set -euo pipefail

MODEL=$(realpath -m "${1:-$MODELS_DIR/Qwen3.8-27B}")
PORT=${2:-8000}
GPU=${3:-0}
VLLM=${VLLM_PYTHON:-python}

if [[ ! -f "${MODEL}/config.json" ]]; then
    echo "Model not found at ${MODEL}; download it with:" >&2
    echo "  uvx --from 'huggingface_hub[hf_xet]' hf download Qwen/Qwen3.8-27B --local-dir ${MODEL}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
echo "[serve] Qwen3.8-27B on http://${VLLM_BIND:-127.0.0.1}:${PORT}/v1 (GPU ${GPU})"
exec "${VLLM}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
    --served-model-name Qwen/Qwen3.8-27B \
    --host "${VLLM_BIND:-127.0.0.1}" --port "${PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.92}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-64}" \
    --limit-mm-per-prompt '{"image": 4, "video": 0}' \
    --reasoning-parser qwen3
