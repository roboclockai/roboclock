#!/usr/bin/env bash
# Role D (InternW0-Delta) training, docs/five-domain-training.md section 7.
#
#   bash run_train.sh            full run: 1550 steps x 2048 (8/GPU x accum 32 x 8 GPUs) = 3,174,400 samples,
#                                weights every 310 steps (+ final) -> $D/run1
#   bash run_train.sh smoke      MAX_STEPS (default 5) steps, no saves, no eval, log every step -> $D/run_smoke
#   RESUME=1 bash run_train.sh   continue the full run from run1/checkpoints/state/latest
#   DRY_RUN=1 bash run_train.sh [smoke]   CPU only: preflight + print the accelerate command
#
# Release recipe kept (upstream configs/task/robodojo.yaml @90801baa): lr 5e-5 cosine to 1%, 5% warm-up,
# AdamW (0.9, 0.95), wd 0.01, grad clip 1, bf16, DeepSpeed ZeRO-1, 8 per GPU, colour jitter.
# Changed: init robodojo.pt (weights only, step 0), accum 32, max_steps, save_every, the 45 repos,
# context length 256, widened norm stats, +infra=online.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODE=${1:-full}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
case "$MODE" in
    full)  OUT=${OUT:-$D/run1};      MAX_STEPS=${MAX_STEPS:-1550} ;;
    smoke) OUT=${OUT:-$D/run_smoke}; MAX_STEPS=${MAX_STEPS:-5} ;;
    *) echo "usage: run_train.sh [full|smoke]" >&2; exit 2 ;;
esac
ACCUM=32
DRY=${DRY_RUN:-0}

# --- preflight -------------------------------------------------------------------------
fail() { if [[ "$DRY" == 1 ]]; then echo "PREFLIGHT (would fail): $*" >&2; else echo "PREFLIGHT FAILED: $*" >&2; exit 1; fi; }
[[ -x "$PY" ]] || fail "no training venv at $PY"
[[ -f "$INIT" ]] || fail "no init checkpoint $INIT"
[[ -f "$STATS" ]] || fail "no stats $STATS (run stats.py)"
[[ -f "$WAN/Wan2.2_VAE.pth" && -f "$VLM/model.safetensors" ]] || fail "missing Wan2.2 VAE or RynnBrain"
(( NGPU * 8 * ACCUM == 2048 )) || fail "global batch would be $((NGPU * 8 * ACCUM)), not 2048 (GPUS=$GPUS)"
if dirs=$(dataset_dirs_override); then
    if [[ "${SKIP_CACHE_CHECK:-0}" != 1 ]]; then
        "$PY" "$HERE/check_cache.py" --repos "$LEROBOT" --cache "$TEXT_CACHE_DIR" --context-len "$CONTEXT_LEN" \
            || fail "text cache incomplete: missing prompts would be silently replaced by random samples"
    fi
else
    fail "converted repos incomplete (run convert.py)"
    dirs="data.train.dataset_dirs=[<45 repos>]"
fi
if [[ "$DRY" != 1 ]] && command -v nvidia-smi >/dev/null; then
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', ' '$2 > 1000 {print $1}' | xargs)
    [[ -z "$busy" || "${FORCE:-0}" == 1 ]] || fail "GPUs in use: $busy (FORCE=1 to start anyway)"
fi
if [[ "$MODE" == full && "${RESUME:-0}" != 1 && -d "$OUT/checkpoints/weights" ]]; then
    fail "$OUT already has weights; RESUME=1 to continue, or OUT=... for a new run"
fi

# --- launch ----------------------------------------------------------------------------
overrides=(
    "output_dir=$OUT" seed=42  # upstream train.yaml default; set_global_seed asserts 0 < seed
    "model.model_id=$WAN" "model.tokenizer_model_id=$WAN" model.redirect_common_files=false
    "model.tokenizer_max_len=$CONTEXT_LEN" "data.train.context_len=$CONTEXT_LEN"
    "$dirs"
    batch_size=8 "gradient_accumulation_steps=$ACCUM" "max_steps=$MAX_STEPS"
    +infra=online
)
if [[ "$MODE" == smoke ]]; then
    overrides+=(save_every=1000000 save_on_train_end=false save_training_state=false eval_every=0 log_every=1)
else
    overrides+=(save_every=310)
fi
[[ "${RESUME:-0}" == 1 ]] && overrides+=(resume=latest)

export ROBODOJO_NORM_STATS=$STATS WAM_CACHE_ROOT=$CACHE_ROOT WAM_PRETRAIN_CHECKPOINT=$INIT WAM_VLM_PATH=$VLM \
       DIFFSYNTH_SKIP_DOWNLOAD=true WANDB_MODE=offline HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 \
       CUDA_VISIBLE_DEVICES=$GPUS NPROC_PER_NODE=$NGPU RUN_ID=${RUN_ID:-train45_$MODE} MASTER_PORT=${MASTER_PORT:-29517}
export PATH="$D/venv-train/bin:$PATH"   # run.sh calls `accelerate` and `python`
unset TRITON_PTXAS_PATH                 # the pod-wide CUDA 13 ptxas breaks Triton of torch cu128 (FLA kernels)
cd "$UP"
if [[ "$DRY" == 1 ]]; then
    DRY_RUN=1 bash run.sh robodojo "${overrides[@]}"
    exit 0
fi
mkdir -p "$OUT"
echo "[run_train] $MODE: $NGPU GPUs ($GPUS), 8 x $ACCUM x $NGPU = 2048, max_steps=$MAX_STEPS, out=$OUT"
bash run.sh robodojo "${overrides[@]}" 2>&1 | tee -a "$OUT/train.log"
