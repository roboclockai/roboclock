#!/bin/bash
# usage: serve_vlm.sh MODEL PORT GPUS   one vLLM OpenAI server for a vision-LLM model of common.sh (KIND vllm)
# GPUS is a comma list of TP GPUs, e.g. 0,1. Started by launch.sh; can also be run by hand.
source "$(dirname "$0")/common.sh"
M=$1; P=$2; GPUS=$3
model_cfg "$M" || exit 1
[ "$KIND" = vllm ] || { echo "$M is not a vLLM model" >&2; exit 1; }
[ -f "$MODELS_DIR/$VLM_DIR/config.json" ] || { echo "no weights at $MODELS_DIR/$VLM_DIR" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPUS
# The pod-wide CUDA 13 ptxas breaks Triton; the NGC image's OMP_NUM_THREADS=95 makes every TP worker spin.
unset TRITON_PTXAS_PATH OMP_NUM_THREADS MKL_NUM_THREADS
exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODELS_DIR/$VLM_DIR" \
  --served-model-name "$SERVED" --host 127.0.0.1 --port "$P" \
  --tensor-parallel-size "$(echo "$GPUS" | tr ',' '\n' | wc -l)" --gpu-memory-utilization "$UTIL" \
  --max-model-len 16384 --max-num-seqs "${VLLM_MAX_NUM_SEQS:-32}" \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --reasoning-parser "$PARSER" --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
  ${REASONING_CONFIG:+--reasoning-config "$REASONING_CONFIG"}
