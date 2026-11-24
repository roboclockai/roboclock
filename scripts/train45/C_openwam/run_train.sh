#!/usr/bin/env bash
# Role C (OpenWAM-alpha) training launcher: docs/five-domain-training.md, section 6.
#
#   bash run_train.sh                  full run: 99,200 steps x 32, save every 20,000, keep 5
#   MAX_STEPS=50 bash run_train.sh     smoke test (section 8): saving off, OUT=.../smoke
#   DRY_RUN=1 bash run_train.sh        print the checks and the command, launch nothing
#
# Extra arguments are appended as Hydra overrides, e.g. training.save_full_states_for_resume=true.
# Optional env: OUT, GPUS (default 0-7), OPENWAM_ROOT, TRAIN45_PROMPTS, FARM.
#
# Before launching it checks that <farm>/meta/robodojo_normalization_stats.npy is the widened
# file make_stats.py wrote (sha256 in its report): without it the reader would silently rebuild
# stats from our data alone (rule 5).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENWAM_ROOT="${OPENWAM_ROOT:-$ROBODOJO_DIR/XPolicyLab/policy/OpenWAM/OpenWAM}"
VENV="$(cd "${OPENWAM_ROOT}/.." && pwd)/.venv"
FARM="${FARM:-$TRAIN45_DIR/data/RoboDojo}"
INIT=$MODELS_DIR/OpenWAM-Alpha-Sim-RoboDojo
WORK=$TRAIN45_DIR/C_openwam
REPORT="${WORK}/normalization_widening_report.json"
STATS="${FARM}/meta/robodojo_normalization_stats.npy"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC=$(awk -F, '{print NF}' <<< "${GPUS}")
export OPENWAM_ROOT TRAIN45_PROMPTS="${TRAIN45_PROMPTS:-$TRAIN45_DIR/prompts/instructions_main_4f541f2.json}"

if [[ -n "${MAX_STEPS:-}" ]]; then
    OUT="${OUT:-${WORK}/smoke}"
    SCHEDULE=(training.max_steps="${MAX_STEPS}" training.save_steps=0)
else
    OUT="${OUT:-${WORK}/run1}"
    SCHEDULE=(training.max_steps=99200 training.save_steps=20000 training.keep_last_k_ckpts=5)
fi

for path in "${VENV}/bin/torchrun" "${OPENWAM_ROOT}/scripts/train.py" "${INIT}/config.yaml" "${TRAIN45_PROMPTS}" "${REPORT}"; do
    [[ -e "${path}" ]] || { echo "[run_train] missing ${path}" >&2; exit 1; }
done
[[ -f "${STATS}" ]] || { echo "[run_train] missing ${STATS}; run make_stats.py first" >&2; exit 1; }
want=$("${VENV}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "${REPORT}")
have=$(sha256sum "${STATS}" | cut -d' ' -f1)
[[ "${want}" == "${have}" ]] || { echo "[run_train] ${STATS} is not the widened file of ${REPORT}; rerun make_stats.py --force" >&2; exit 1; }
(( NPROC == 8 )) || echo "[run_train] WARNING: ${NPROC} GPUs; global batch is ${NPROC} x 4, not 32 (rescale max_steps)" >&2

# Pod-wide CUDA 13 ptxas breaks Triton kernels of this torch build; OMP 95 oversubscribes 8 ranks x 8 workers.
unset TRITON_PTXAS_PATH
export OMP_NUM_THREADS=8 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES="${GPUS}" PATH="${VENV}/bin:${PATH}"

CMD=(
    "${VENV}/bin/torchrun" --nnodes 1 --master_addr 127.0.0.1 --master_port "${MASTER_PORT:-29521}" --nproc_per_node "${NPROC}" "${HERE}/train_ptime.py"
    dataloader=robodojo dataloader.dataset_dir="${FARM}"
    training.finetune_ckpt_path="${INIT}"
    training.output_path="${OUT}" "hydra.run.dir=${OUT}/hydra/\${now:%Y-%m-%d_%H-%M-%S}"
    training.batch_size=4 training.num_epochs=null training.use_gradient_checkpointing=false
    "${SCHEDULE[@]}"
    project.seed=42
    "$@"
)

echo "[run_train] OUT=${OUT} GPUs=${GPUS} stats=${STATS} (sha256 ${have:0:12})"
echo "[run_train] $(printf '%q ' "${CMD[@]}")"
if [[ -n "${DRY_RUN:-}" ]]; then
    exit 0
fi
mkdir -p "${OUT}/logs"
LOG="${OUT}/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "[run_train] log: ${LOG}"
cd "${OPENWAM_ROOT}"
"${CMD[@]}" 2>&1 | tee -a "${LOG}"
