#!/bin/bash
# Role A data (roboclock docs/five-domain-training.md): 45 training tasks, prompts = roboclock main 4f541f2 gen_instruction().
export PYTHONPATH=$ROBODOJO_DIR:$ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/src
export HF_LEROBOT_HOME=$PI05_TRAIN_DIR/lerobot JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd $PI05_TRAIN_DIR
exec nice -n 10 $ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python convert_roboclock_to_lerobot.py \
  --src $TRAIN45_DIR/data/RoboDojo --repo-id roboclock_train45_v1 --shards 12 \
  --instructions $TRAIN45_DIR/prompts/instructions_main_4f541f2.json
