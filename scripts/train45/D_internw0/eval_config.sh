#!/usr/bin/env bash
# Writes the role-D eval model config outside the RoboDojo submodule: the adapter's
# config/eval_model.yaml with model.tokenizer_max_len 128 -> 256, the context length the
# fine-tune is trained at. Nothing else changes (training keeps the release architecture).
# The adapter reads it through deploy key train_config_path (setup_eval_policy_server.sh:
# WAM_EVAL_CONFIG_PATH); scripts/xpolicylab/serve_internw0.sh hard-codes the adapter's file and
# needs a pass-through, like the stats (WAM_DATASET_STATS_PATH -> run1/dataset_stats.json).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SRC=$ROBODOJO_DIR/XPolicyLab/policy/InternW0_delta/config/eval_model.yaml
DST=$D/eval_model_len256.yaml
{
    echo "# role D (train45): $SRC with tokenizer_max_len $CONTEXT_LEN (written by scripts/train45/D_internw0/eval_config.sh)"
    sed "s/^  tokenizer_max_len: 128$/  tokenizer_max_len: $CONTEXT_LEN/" "$SRC"
} > "$DST"
changed=$(diff <(grep -v '^# role D' "$DST") "$SRC" | grep -c '^[<>]' || true)
grep -q "^  tokenizer_max_len: $CONTEXT_LEN$" "$DST" && [[ "$changed" == 2 ]] || { echo "unexpected diff ($changed lines)" >&2; exit 1; }
echo "wrote $DST"; diff "$DST" "$SRC" || true
