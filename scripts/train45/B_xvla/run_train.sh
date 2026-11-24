#!/usr/bin/env bash
# X-VLA five-domain fine-tune (docs/five-domain-training.md, role B).
#
#   bash scripts/train45/B_xvla/run_train.sh            # full run: 12,400 steps -> run1/ckpt-{2500,...,12400}
#   ITERS=50 bash scripts/train45/B_xvla/run_train.sh   # smoke: 50 steps, no weights saved, -> run_smoke/
#
# Env: ITERS (default 12400), OUT (default run1, smoke run_smoke), GPUS (default 0-7),
# WORK (default $TRAIN45_DIR/B_xvla, which must hold init_ckpt100k/ and roboclock45.json).
# The loader writes error_log.txt (skipped episodes) into WORK; check it after the run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
X=$ROBODOJO_DIR/XPolicyLab/policy/X_VLA
WORK=${WORK:-$TRAIN45_DIR/B_xvla}
ITERS=${ITERS:-12400}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=$(awk -F, '{print NF}' <<<"${GPUS}")
MULTI=(); (( NPROC > 1 )) && MULTI=(--multi_gpu)

if [[ "${ITERS}" == 12400 ]]; then
    OUT=${OUT:-run1}
    SAVE_INTERVAL=2500
else
    OUT=${OUT:-run_smoke}
    SAVE_INTERVAL=1000000000
    export XVLA_SKIP_SAVE=1   # train.py also saves at the last step; this writes no weights
fi

for path in "${WORK}/init_ckpt100k/model.safetensors" "${WORK}/init_ckpt100k/preprocessor_config.json" \
            "${WORK}/roboclock45.json" "${X}/.venv/bin/accelerate"; do
    [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 1; }
done
if [[ -e "${WORK}/${OUT}" && "${ITERS}" == 12400 ]]; then
    echo "${WORK}/${OUT} exists; train.py has no resume and would overwrite its checkpoints" >&2
    exit 1
fi

cd "${WORK}"
unset TRITON_PTXAS_PATH
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPUS}"
echo "[B_xvla] iters=${ITERS} save_interval=${SAVE_INTERVAL} out=${WORK}/${OUT} gpus=${GPUS} (${NPROC} procs)"

exec "${X}/.venv/bin/accelerate" launch "${MULTI[@]}" --num_processes "${NPROC}" --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no \
    "${HERE}/train_roboclock.py" \
    --models "${WORK}/init_ckpt100k" \
    --train_metas_path "${WORK}/roboclock45.json" \
    --output_dir "${WORK}/${OUT}" \
    --learning_rate 1e-4 --learning_coef 0.1 \
    --iters "${ITERS}" --freeze_steps 1000 --warmup_steps 2000 \
    --batch_size 32 --save_interval "${SAVE_INTERVAL}" --seed 0
