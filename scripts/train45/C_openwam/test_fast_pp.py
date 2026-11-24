"""fast_pp.py must return bit-identical tensors to upstream OpenWAM preprocessing (needs one GPU).

    CUDA_VISIBLE_DEVICES=0 <OpenWAM venv>/bin/python scripts/train45/C_openwam/test_fast_pp.py

Checks both patched entry points (the native Wan path used by OpenWAM-Alpha-Sim-RoboDojo and the
external-encoder path) on random frames, every uint8 value 0..255, and constant 0 / 255 frames,
in bf16 (the training dtype) and fp32; then times the native path on a batch of 4 x 9 frames.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ptime  # noqa: E402  (for OPENWAM_ROOT, the tree training imports openwam from)

sys.path.insert(0, str(train_ptime.OPENWAM_ROOT))
import fast_pp  # noqa: E402
from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder  # noqa: E402
from openwam.model.video_backbone.wan import encode as wan_encode  # noqa: E402


def main():
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    cases = {
        "random 9x384x320": [Image.fromarray(rng.integers(0, 256, (384, 320, 3), dtype=np.uint8)) for _ in range(9)],
        "all 256 values": [Image.fromarray(np.tile(np.arange(256, dtype=np.uint8)[:, None, None], (1, 320, 3)))] * 2,
        "0 and 255": [Image.fromarray(np.full((384, 320, 3), v, np.uint8)) for v in (0, 255)],
    }
    vae = torch.nn.Linear(2, 2)
    encoder = WanVideoVAEEncoder.__new__(WanVideoVAEEncoder)
    torch.nn.Module.__init__(encoder)

    def native(frames, dtype):
        return wan_encode.preprocess_video(frames, encoder=None, dtype=dtype, device=device)

    def external(frames, dtype):
        encoder._m = vae.to(device, dtype)
        return encoder.preprocess_video(frames)

    dtypes = (torch.bfloat16, torch.float32)
    want = {(p.__name__, n, d): p(f, d) for p in (native, external) for n, f in cases.items() for d in dtypes}
    upstream_native = wan_encode._preprocess_video_native
    fast_pp.install()
    assert wan_encode._preprocess_video_native is not upstream_native, "native path not patched"
    for (path, name, dtype), a in want.items():
        b = {"native": native, "external": external}[path](cases[name], dtype)
        same = (a.shape == b.shape and a.dtype == b.dtype and a.is_contiguous() and b.is_contiguous()
                and torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
        print(f"{path:8s} {str(dtype):15s} {name:17s} {tuple(a.shape)} bit-identical={same}")
        assert same, f"{path} {dtype} {name} differs"

    frames = cases["random 9x384x320"]
    for label, fn in (("upstream", upstream_native), ("fast_pp", wan_encode._preprocess_video_native)):
        run = lambda: [fn(frames, dtype=torch.bfloat16, device=device) for _ in range(4)]  # noqa: E731
        run()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(4):
            run()
        torch.cuda.synchronize()
        print(f"{label}: {(time.perf_counter() - t0) / 4 * 1e3:.1f} ms per batch of 4 x 9 frames")
    print("PASS")


if __name__ == "__main__":
    main()
