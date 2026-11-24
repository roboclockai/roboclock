#!/usr/bin/env bash
# Role D text cache: UMT5-XXL embeddings of every distinct prompt of the 45 converted repos, at
# context length 256 (the relabelled instructions reach 231 tokens with the "A video recorded ..."
# prefix; the release's 128 would truncate 25,374 of the 60,220 prompts).
# Output: $TEXT_CACHE_DIR/<sha256>.t5_len256.wan22ti2v5b.pt, ~2 MiB each, ~118 GiB in all.
# Then runs check_cache.py, which must print OK before training.
#
#   GPUS=0,1,2,3,4,5,6,7 bash text_cache.sh      # needs GPUs; the encoder is loaded once per rank
#   DRY_RUN=1 bash text_cache.sh                 # CPU: print the command and the composed data config
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    dirs=$(dataset_dirs_override) || { dirs="data.train.dataset_dirs=[$LEROBOT/lab_A1]"; echo "(dry run: repos incomplete, composing with $dirs)"; }
else
    dirs=$(dataset_dirs_override)
fi
args=(tools/text_cache.py task=robodojo
      "model.model_id=$WAN" "model.tokenizer_model_id=$WAN" model.redirect_common_files=false
      "model.tokenizer_max_len=$CONTEXT_LEN" "data.train.context_len=$CONTEXT_LEN" "$dirs")

cd "$UP"
unset TRITON_PTXAS_PATH  # the pod-wide CUDA 13 ptxas breaks Triton of torch cu128
export WAM_CACHE_ROOT=$CACHE_ROOT DIFFSYNTH_SKIP_DOWNLOAD=true HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES=$GPUS $D/venv-train/bin/torchrun --nnodes=1 --master_addr=127.0.0.1 --master_port=${MASTER_PORT:-29531} --nproc_per_node=$NGPU ${args[*]:0:7} <45 dataset_dirs>"
    "$PY" "${args[@]}" --cfg job --resolve | grep -E "context_len|text_embedding_cache_dir|tokenizer_max_len|model_id" | sort -u
    exit 0
fi
mkdir -p "$TEXT_CACHE_DIR" "$D/logs"
CUDA_VISIBLE_DEVICES=$GPUS nice -n 10 "$D/venv-train/bin/torchrun" --nnodes=1 --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29531}" --nproc_per_node="$NGPU" "${args[@]}" \
    2>&1 | tee -a "$D/logs/text_cache.log"
"$PY" "$HERE/check_cache.py" --repos "$LEROBOT" --cache "$TEXT_CACHE_DIR" --context-len "$CONTEXT_LEN" --tokens --deep 20 \
    2>&1 | tee -a "$D/logs/text_cache.log"
