# Scripts

Asset preparation, demo collection, fine-tuning inputs, and policy servers. Run them from the
repository root with the Python environment that has Isaac Sim installed.

No script assumes a fixed directory layout. Locations outside the repository are read from
environment variables, and a script that needs one fails on the unexpanded name if it is unset:

| Variable | Points to |
| :-- | :-- |
| `ROBOCLOCK_DIR` | A checkout of this repository, when a script runs from another directory. |
| `ROBODOJO_DIR` | A RoboDojo checkout with XPolicyLab and its policy environments. |
| `MODELS_DIR` | Downloaded model checkpoints, one folder per model. |
| `DEMOS_DIR` | Recorded oracle demonstrations (the output of `scripts/collect/`). |
| `TRAIN45_DIR` | Working directory of the 45-task fine-tuning runs (`scripts/train45/`). |
| `PI05_TRAIN_DIR` | Working directory of the π0.5 fine-tuning run. |
| `WALLX_DIR` | A WALL-OSS checkout. |
| `ISAAC_VENV`, `VLLM_VENV` | Python environments for Isaac Sim and for vLLM. |
| `API_KEY_FILE`, `HF_TOKEN_FILE` | Files holding a hosted-model API key and a Hugging Face token. |
| `WORK_DIR` | Scratch space for sweeps, ablations, and other run outputs. |
