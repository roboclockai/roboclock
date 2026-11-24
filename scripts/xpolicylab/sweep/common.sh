# Settings shared by the model sweep scripts. Sourced, not run. RoboDojo models: docs/robodojo-sweep.md;
# vision-LLM models (local vLLM servers or a hosted API such as NVIDIA's): docs/vision-llm-sweep.md.
SWEEP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SWEEP_DIR/../../.." && pwd)
ROOT=${SWEEP_ROOT:-$WORK_DIR/robodojo_sweep}
ROBODOJO=${ROBODOJO:-$ROBODOJO_DIR}
PY=${SIM_PYTHON:-python}
DIFFICULTY=${DIFFICULTY:-easy}
SEEDS=${SEEDS:-3}
SEED_START=${SEED_START:-0}
ENVS=${ENVS:-"lab hospital household sports industry"}
TASKS=${TASKS:-}                                  # optional filter, e.g. "household:B2 lab:A5"
FIRST_TASKS=${FIRST_TASKS:-}                      # optional, queued first, e.g. "hospital:A1"
# zero: the simulation pauses during inference; measured: each query's server time passes in simulation
# (needs one episode per server, see launch.sh); ideal: see roboclock eval --help.
LATENCY=${LATENCY:-zero}
# Control-interface delay delta (s) charged per model query on top of its inference latency (roboclock eval
# --control-delay). The paper's Latency-Aligned mode uses one 50 Hz control step, 0.02 s; the
# inference-paused (zero) runs use none. Set CONTROL_DELAY to override either.
if [ "$LATENCY" = zero ]; then CONTROL_DELAY=${CONTROL_DELAY:-0}; else CONTROL_DELAY=${CONTROL_DELAY:-0.02}; fi
# Isaac starts wait while the 1-min load is above LOAD_MAX. At zero latency the load overstates CPU use
# (Isaac idles while it waits for the model): 17 episodes gave load 115 with 51% of the CPU idle.
LOAD_MAX=${LOAD_MAX:-$(( $(nproc) * 3 / 2 ))}
START_STAGGER=${START_STAGGER:-20}                # seconds between Isaac start-ups on the node
START_SETTLE=${START_SETTLE:-40}                  # extra seconds between start-ups above 2/3 LOAD_MAX (episode.sh)
START_LOCK=${START_LOCK:-${TMPDIR:-/tmp}/.isaac-start.lock}
# roboclock checks `vulkaninfo --summary` before Isaac starts. On a busy node it takes 5-35 s, so the 15 s
# default fails healthy start-ups as "Vulkan cannot enumerate a GPU".
export ROBOCLOCK_VULKAN_PROBE_TIMEOUT=${ROBOCLOCK_VULKAN_PROBE_TIMEOUT:-120}

# Isaac Sim's graphics libraries (vulkan, GLU, Xt), where the image does not have them in ldconfig: the
# directory they were extracted to is put on the library path of everything the sweep starts.
ISAAC_GRAPHICS_LIBS=${ISAAC_GRAPHICS_LIBS:-/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu}
if [ -d "$ISAAC_GRAPHICS_LIBS" ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":$ISAAC_GRAPHICS_LIBS:"* ]]; then
  export LD_LIBRARY_PATH=$ISAAC_GRAPHICS_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

# Vision-LLM models (KIND vllm and api) run the primitive agent (roboclock.policies.vision_llm_primitives).
VLLM_PYTHON=${VLLM_PYTHON:-python}
MODELS_DIR=${MODELS_DIR:-$MODELS_DIR}
VLM_MAX_QUERIES=${VLM_MAX_QUERIES:-300}        # model turns per episode, then the arms hold (0: no cap)
VLM_MAX_QUERIES_C=${VLM_MAX_QUERIES_C:-100}    # the same for C-class (throughput) tasks (user, 2026-11-09)
# LATENCY=ideal charges each turn ttft_s + output_tokens / max_tokens_per_s from this file (roboclock eval
# --model-speeds); empty: roboclock's surveyed src/roboclock/evaluation/model_speeds.yaml. The open-VLM LTSR
# runs use model_speeds_8xa100.yaml (each model on 8 A100s with its fastest drafter, docs/vlm-ltsr.md).
MODEL_SPEEDS=${MODEL_SPEEDS:-}
# Optional job list: lines "MODEL ENV TASK SEED" (# comments). Only the listed episodes of a model are
# queued; used to split one run over several nodes (docs/vision-llm-sweep.md, Several nodes).
JOBS_FILE=${JOBS_FILE:-}
# Hosted OpenAI-compatible API for nv/<model id> models (e.g. nv/gcp/google/gemini-4-argon). The key is
# read from NVIDIA_API_KEY, else from NV_API_KEY_FILE ("1. sk-..." or "sk-..." per line; by default the
# first of $API_KEY_FILE and $API_KEY_FILE that exists). NV_API_KEY_INDEX names the
# line, or several ("1,2"): the slots then take the keys in turn, so one sweep spreads over several keys.
NV_API_URL=${NV_API_URL:-https://inference-api.example.com/v1}
if [ -z "${NV_API_KEY_FILE:-}" ]; then
  NV_API_KEY_FILE=$API_KEY_FILE
  [ -f "$NV_API_KEY_FILE" ] || { [ -f $API_KEY_FILE ] && NV_API_KEY_FILE=$API_KEY_FILE; }
fi
NV_API_KEY_INDEX=${NV_API_KEY_INDEX:-1}

# model_cfg MODEL sets one model's settings:
#   KIND                   xpolicylab: RoboDojo model behind an XPolicyLab server; vllm: vision-LLM agent
#                          on a local vLLM server (TP GPUs per server); api: vision-LLM agent on a hosted
#                          API (no server; the GPUs run only Isaac)
#   POLICY / POLICY_ARGS   roboclock eval --policy factory and extra arguments
#   SERVE / SERVE_ARGS     server script in scripts/xpolicylab/, called as SERVE ROBODOJO PORT GPU SERVE_ARGS
#   SERVE_ENV              environment for the server process
#   EPISODE_ENV            environment for each episode's roboclock client (e.g. XPOLICYLAB_PROMPT_TIME=1)
#   PER_GPU                servers per GPU
#   SLOTS_PER_GPU          episodes running at once per GPU, spread over that GPU's servers
#   SHARE                  yes: a server may serve several episodes at once; no: one episode per server;
#                          any other value names the client variable that enables sharing (set to 1)
#   PORT_BASE              port of the model's first server; the others follow
#   WALL_PER_SIM           episode timeout = 900 s + budget x WALL_PER_SIM (about twice the measured rate)
#   FRAMEVAR               the client's head-frame dump variable, set only with SWEEP_FRAMES=1
# GPU memory per server (A100 80 GB, with Isaac on the same GPU at about 5 GB each):
#   wallx 14 GB, internw0 29.5 GB, vpp2 48.5 GB, awomo 22 GB, xvla 4 GB, g05 12 GB, openwam 24 GB, pi05 20 GB.
model_cfg() {
  KIND=xpolicylab; POLICY_ARGS="--policy-hz 25"; SERVE_ARGS=""; SERVE_ENV=""; FRAMEVAR=""; EPISODE_ENV=""
  TP=1; REASONING_CONFIG=""; API_MODEL=""; SERVED=""; VLM_DIR=""; PARSER=""; UTIL=""; TOOL_PARSER=""
  case $1 in
    wallx)    POLICY=roboclock.envs.lab.policies.wallx:make;    SERVE=serve_wallx.sh;    PER_GPU=2; SLOTS_PER_GPU=2; SHARE=no
              PORT_BASE=6100; WALL_PER_SIM=40; FRAMEVAR=WALLX_FRAME_DIR ;;
    internw0) POLICY=roboclock.envs.lab.policies.internw0:make; SERVE=serve_internw0.sh; PER_GPU=2; SLOTS_PER_GPU=2; SHARE=no
              PORT_BASE=6120; WALL_PER_SIM=45; FRAMEVAR=INTERNW0_FRAME_DIR ;;
    # An InternW0 fine-tune (docs/five-domain-training.md, role D): weights INTERNW0FT_CKPT (e.g.
    # <run>/checkpoints/weights/slot_0.pt, see weights_manifest.json), the run's stats and its 256-token eval
    # config; the client sends the env time in the prompt as in training (INTERNW0FT_PROMPT_TIME=0: not).
    # Unset, INTERNW0FT_CKPT makes the server exit; it never falls back to the released weights.
    internw0ft) POLICY=roboclock.envs.lab.policies.internw0:make; SERVE=serve_internw0.sh; PER_GPU=2; SLOTS_PER_GPU=2; SHARE=no
              SERVE_ENV="WAM_CHECKPOINT_PATH=${INTERNW0FT_CKPT:-/nonexistent/set-INTERNW0FT_CKPT} WAM_DATASET_STATS_PATH=${INTERNW0FT_STATS:-$TRAIN45_DIR/D_internw0/run1/dataset_stats.json} WAM_TRAIN_CONFIG_PATH=${INTERNW0FT_CONFIG:-$TRAIN45_DIR/D_internw0/eval_model_len256.yaml}"
              EPISODE_ENV="XPOLICYLAB_PROMPT_TIME=${INTERNW0FT_PROMPT_TIME:-1}"
              PORT_BASE=${INTERNW0FT_PORT_BASE:-6300}; WALL_PER_SIM=45; FRAMEVAR=INTERNW0_FRAME_DIR ;;
    vpp2)     POLICY=roboclock.envs.lab.policies.vpp2:make;     SERVE=serve_vpp2.sh;     PER_GPU=1; SLOTS_PER_GPU=2; SHARE=yes
              PORT_BASE=6140; WALL_PER_SIM=42; FRAMEVAR=VPP2_FRAME_DIR ;;
    awomo)    POLICY=roboclock.envs.lab.policies.awomo:make;    SERVE=serve_awomo.sh;    PER_GPU=2; SLOTS_PER_GPU=2; SHARE=AWOMO_SHARED_SERVER
              PORT_BASE=6160; WALL_PER_SIM=75; FRAMEVAR=AWOMO_FRAME_DIR ;;
    xvla)     POLICY=roboclock.envs.lab.policies.xvla:make;     SERVE=serve_xvla.sh;     PER_GPU=1; SLOTS_PER_GPU=3; SHARE=yes
              PORT_BASE=6200; WALL_PER_SIM=10; FRAMEVAR=XVLA_FRAME_DIR ;;
    g05)      POLICY=roboclock.envs.lab.policies.g05:make;      SERVE=serve_g05.sh;      PER_GPU=2; SLOTS_PER_GPU=3; SHARE=yes
              PORT_BASE=6220; WALL_PER_SIM=10; FRAMEVAR=G05_FRAME_DIR ;;
    openwam)  POLICY=roboclock.envs.lab.policies.openwam:make;  SERVE=serve_openwam.sh;  PER_GPU=2; SLOTS_PER_GPU=3; SHARE=yes
              PORT_BASE=6240; WALL_PER_SIM=10; FRAMEVAR=OPENWAM_FRAME_DIR ;;
    pi05)     POLICY=roboclock.envs.lab.policies.xpolicylab:make; POLICY_ARGS=""; SERVE=serve_pi05.sh; SERVE_ARGS=0
              SERVE_ENV=XLA_PYTHON_CLIENT_MEM_FRACTION=0.25;  PER_GPU=1; SLOTS_PER_GPU=3; SHARE=yes
              PORT_BASE=6260; WALL_PER_SIM=4 ;;
    # Open VLMs on vLLM (weights in MODELS_DIR, see docs/vision-llm-sweep.md): DIR SERVED PARSER UTIL
    # TOOL_PARSER. PARSER is vLLM's reasoning parser, TOOL_PARSER its tool-call parser: the agent calls
    # its primitives as tools, and vLLM turns the model's own tool-call format into tool_calls. Checked
    # in vLLM 0.33 for qwen38 (2026-11-10); the others are the parsers vLLM registers for those models.
    qwen38)    vlm_cfg Qwen3.8-27B Qwen/Qwen3.8-27B qwen3 0.62 qwen3_xml; TP=2; PORT_BASE=8100 ;;
    muse)      vlm_cfg Muse-Glimmer-30B meta-models/Muse-Glimmer-30B muse_glimmer 0.62 muse_glimmer; TP=2; PORT_BASE=8110
               # Its parser has no reasoning markers, so vLLM would ignore thinking_token_budget.
               REASONING_CONFIG='{"reasoning_start_str": " to=self<|message|>", "reasoning_end_str": "<|eom|><|start|>assistant to=user<|message|>"}' ;;
    gemma4)    vlm_cfg gemma-4-26B-A4B-it google/gemma-4-26B-A4B-it gemma4 0.62 gemma4; TP=2; PORT_BASE=8120 ;;
    qwen35_9b) vlm_cfg Qwen3.5-9B Qwen/Qwen3.5-9B qwen3 0.50 qwen3_xml; PORT_BASE=8130 ;;
    # The FP8 checkpoint, which fits one 46 GB L40S without tensor parallelism (the bf16 weights
    # are 54 GB and need two). Quantization changes the model, so its results are its own and do
    # not line up with a bf16 Qwen3.8-27B run.
    qwen38_fp8) vlm_cfg Qwen3.8-27B-FP8 Qwen/Qwen3.8-27B-FP8 qwen3 0.90 qwen3_xml; PORT_BASE=8160 ;;
    qwen35_4b) vlm_cfg Qwen3.5-4B Qwen/Qwen3.5-4B qwen3 0.40 qwen3_xml; PORT_BASE=8140 ;;
    # Qwen3.8-Flash-Next (180B MoE) needs all 8 GPUs: start.sh flashnext:0,1,2,3,4,5,6,7.
    flashnext) vlm_cfg Qwen3.8-Flash-Next Qwen/Qwen3.8-Flash-Next qwen3 0.68 qwen3_xml; TP=8; PORT_BASE=8150 ;;
    # Any model of the hosted API, by its id: nv/<model id>.
    nv/?*)     KIND=api; API_MODEL=${1#nv/}; vlm_policy; PER_GPU=0; SLOTS_PER_GPU=3; SHARE=yes; PORT_BASE=0 ;;
    *) echo "unknown model '$1' (wallx internw0 internw0ft vpp2 awomo xvla g05 openwam pi05," \
            "qwen38 muse gemma4 qwen35_9b qwen35_4b flashnext, nv/<model id>)" >&2; return 1 ;;
  esac
  PER_GPU=${SWEEP_PER_GPU:-$PER_GPU}; SLOTS_PER_GPU=${SWEEP_SLOTS_PER_GPU:-$SLOTS_PER_GPU}
  WALL_PER_SIM=${SWEEP_WALL_PER_SIM:-$WALL_PER_SIM}
}

# The vision-LLM agent's settings shared by local and hosted models.
# VLM_AGENT picks the agent: primitives (default; pixel targets, so the head depth image) or
# move_eef (RoboDojo's single Cartesian tool, RGB only).
vlm_policy() {
  case ${VLM_AGENT:-primitives} in
    primitives) POLICY=roboclock.policies.vision_llm_primitives:make; POLICY_ARGS="--depth" ;;
    # RoboDojo's budget of model calls (VLM_EEF_MAX_QUERIES), not the primitive agent's.
    move_eef)   POLICY=roboclock.policies.vision_llm_move_eef:make; POLICY_ARGS=""; VLM_MAX_QUERIES=${VLM_EEF_MAX_QUERIES:-100} ;;
    *) echo "unknown VLM_AGENT '$VLM_AGENT' (primitives, move_eef)" >&2; return 1 ;;
  esac
  WALL_PER_SIM=150
}
# vlm_cfg DIR SERVED PARSER UTIL TOOL_PARSER: one open VLM served by vLLM from MODELS_DIR/DIR.
vlm_cfg() {
  KIND=vllm; VLM_DIR=$1; SERVED=$2; PARSER=$3; UTIL=$4; TOOL_PARSER=$5; vlm_policy
  PER_GPU=1; SLOTS_PER_GPU=3; SHARE=yes
}

# model_dir MODEL: the model's directory under ROOT ("/" in an API model id becomes "_").
model_dir() { echo "$ROOT/${1//\//_}"; }

# api_key_line [SLOT]: the line of NV_API_KEY_FILE that slot SLOT (default 0) uses.
api_key_line() {
  local lines; IFS=, read -ra lines <<< "$NV_API_KEY_INDEX"
  echo "${lines[$(( ${1:-0} % ${#lines[@]} ))]}"
}
# api_key [SLOT]: the hosted API's key, for the episode's environment only (never on a command line or in a log).
api_key() {
  if [ -n "${NVIDIA_API_KEY:-}" ]; then echo "$NVIDIA_API_KEY"; return; fi
  sed -n "$(api_key_line "${1:-0}")p" "$NV_API_KEY_FILE" 2>/dev/null | sed -E 's/^[[:space:]]*[0-9]+\.[[:space:]]+//; s/[[:space:]]+$//'
}

# role_specs ROLE|SPEC...  ->  MODEL:GPU,GPU,... specs. Roles assume 8 free 80 GB GPUs.
role_specs() {
  case $1 in
    A)    echo "awomo:0,1,2,3,4,5,6,7" ;;
    B)    echo "wallx:0,1 internw0:2,3,4 vpp2:5,6,7" ;;
    fast) echo "pi05:0,1 xvla:2,3 g05:4,5 openwam:6,7" ;;
    vlm)  echo "qwen38:0,1 muse:2,3 gemma4:4,5 qwen35_9b:6 qwen35_4b:7" ;;
    *)    echo "$@" ;;
  esac
}

# alive PID: running and not a zombie (setsid children reparent to PID 1, which may never reap them)
alive() { local st; st=$(awk '{print $3}' /proc/$1/stat 2>/dev/null); [ -n "$st" ] && [ "$st" != Z ]; }
port_open() { python3 -c "import socket,sys; socket.create_connection(('127.0.0.1', int(sys.argv[1])), 2)" "$1" 2>/dev/null; }
# vllm_ready PORT: the vLLM server answers (it binds its port before the model has loaded)
vllm_ready() { curl -sf --max-time 5 "http://127.0.0.1:$1/v1/models" > /dev/null 2>&1; }
