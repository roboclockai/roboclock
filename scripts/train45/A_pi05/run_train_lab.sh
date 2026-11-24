#!/bin/bash
# pi0.5 lab-only fine-tune on 8 x A100 (configs in train_pi05_roboclock.py).
#   ./run_train_lab.sh <exp> [config]   config: pi05_roboclock_lab_time (default, prompt + env time) | pi05_roboclock_lab
# Checkpoints: $PI05_TRAIN_DIR/checkpoints/<config>/<exp>/<step>/{params,assets/arx_x5_sim}
EXP=${1:-lab_v2_time_5k}
CFG=${2:-pi05_roboclock_lab_time}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=$PI05_TRAIN_DIR/lerobot
export HF_DATASETS_CACHE=$PI05_TRAIN_DIR/cache/hf
export JAX_COMPILATION_CACHE_DIR=$PI05_TRAIN_DIR/cache/jax
export PYTHONPATH=$ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/src
mkdir -p "$HF_DATASETS_CACHE" "$JAX_COMPILATION_CACHE_DIR"
cd $ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi
exec .venv/bin/python $PI05_TRAIN_DIR/train_pi05_roboclock.py "$CFG" --exp-name "$EXP" --overwrite
