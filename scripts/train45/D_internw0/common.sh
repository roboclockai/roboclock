# Shared paths for role D (InternW0-Delta) of docs/five-domain-training.md. Source, don't run.
D=$TRAIN45_DIR/D_internw0
UP=$D/upstream                              # InternW0-Delta @ 90801baa3bdc7829c1e4989edfe3d0fb2410ea6a
PY=$D/venv-train/bin/python                 # training / text-cache env
PY_CONVERT=$D/venv-convert/bin/python       # lerobot 0.3.3 (v2.1 writer) env
FARM=$TRAIN45_DIR/data/RoboDojo       # the 45 training tasks
LEROBOT=$D/lerobot                          # convert.py output: one v2.1 repo per task
STATS=$D/stats_train45.json                 # stats.py output
CACHE_ROOT=$D/cache                         # text cache: $CACHE_ROOT/text_embed/robodojo
TEXT_CACHE_DIR=$CACHE_ROOT/text_embed/robodojo
CONTEXT_LEN=256                             # UMT5 tokens; the new instructions reach 231 with the prefix
WAN=$MODELS_DIR/Wan2.2-TI2V-5B        # VAE + UMT5 + tokenizer (dir name must stay Wan2.2-TI2V-5B)
VLM=$MODELS_DIR/RynnBrain1.1-2B
INIT=$MODELS_DIR/InternW0-Delta-RoboDojo/robodojo.pt
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# The farm's task names: entries with arx_x5/data (role C keeps a meta/ dir next to them).
farm_tasks() {
    local p
    for p in "$FARM"/*/arx_x5/data; do basename "$(dirname "$(dirname "$p")")"; done
}

# Hydra list of the converted repos, one per farm task, in farm order. Fails unless every
# task has a complete repo (convert.py writes $LEROBOT/_status/<task>.json last).
dataset_dirs_override() {
    local root=${1:-$LEROBOT} dirs=() task
    for task in $(farm_tasks); do
        if [[ ! -f "$root/_status/$task.json" || ! -f "$root/$task/meta/info.json" ]]; then
            echo "missing converted repo for $task under $root" >&2
            return 1
        fi
        dirs+=("$root/$task")
    done
    local IFS=,
    echo "data.train.dataset_dirs=[${dirs[*]}]"
}
