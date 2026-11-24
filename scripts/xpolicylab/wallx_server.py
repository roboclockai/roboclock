"""XPolicyLab policy server for the Wall-X checkpoint, with a vision-attention fallback.

wall-x's InferConfig pins the vision tower to ``flash_attention_2``. Without the flash-attn
package (no wheel for torch 2.10 / CUDA 13 on this pod), this launcher switches the vision
tower to PyTorch SDPA, which computes the same attention (bf16 rounding aside). The language
model already runs SDPA in wall-x serving. Everything else is XPolicyLab's
``setup_policy_server.py``: the same arguments, deploy.yml and overrides.

Run through scripts/xpolicylab/serve_wallx.sh.
"""

import importlib.util
import runpy
import sys


def patch_vision_attention():
    if importlib.util.find_spec("flash_attn") is not None:
        return
    import wall_x.serving.robodojo.policy as policy

    original = policy.build_infer_config

    def build_infer_config(config):
        infer_config = original(config)
        vision = infer_config.model_config.vision_config
        if vision._attn_implementation == "flash_attention_2":
            vision._attn_implementation = "sdpa"
            print("[wallx_server] flash-attn is not installed: vision tower uses sdpa")
        return infer_config

    policy.build_infer_config = build_infer_config


if __name__ == "__main__":
    server, *arguments = sys.argv[1:]
    patch_vision_attention()
    sys.argv = [server, *arguments]
    runpy.run_path(server, run_name="__main__")
