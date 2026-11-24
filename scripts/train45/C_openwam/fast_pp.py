"""Faster, bit-identical replacement for OpenWAM's per-frame video preprocessing (Wan backbone).

Upstream, per PIL frame (wan/preprocess.py preprocess_image):
    torch.Tensor(np.array(img, float32)).to(bf16, cuda) * (2/255) + (-1)   -> repeat to (1, C, H, W)
then torch.stack along T -> (1, C, T, H, W). That is one small H2D copy plus a CPU-side dtype
conversion per frame (36 per step at batch 4), ~2.1 s of a 3.4 s step under training load.

Here: stack the uint8 frames once, copy once, cast on the GPU, then apply the same two
elementwise ops in the same dtype. uint8 values 0..255 are exact in bf16/fp16/fp32, so the
result is bit-identical. Falls back to upstream for anything that is not a list of same-shape
uint8 HxWx3 frames, or for a non-default pattern.

Two entry points exist upstream and both are patched:
  - wan/encode.py `_preprocess_video_native` (used when the backbone has no external video
    encoder; this is the OpenWAM-Alpha-Sim-RoboDojo case);
  - encoder/wan22_vae.py `WanVideoVAEEncoder.preprocess_video` (external-encoder case).

Measured 2026-11-14 on 8 x A100 80 GB, batch 4/GPU, 30 steps (median of steps 5-30, timers with
cuda syncs): preprocess 2.04 -> 0.01 s, step 3.35 -> 1.29 s; losses equal the unpatched run's
within run-to-run noise (step 1 identical). Bit identity: test_fast_pp.py (one GPU, seconds).
"""

import numpy as np
import torch


def _fast(frames, dtype, device, min_value, max_value):
    """(1, C, T, H, W) tensor, or None if the frames are not plain same-shape uint8 HxWx3."""
    arrays = [np.asarray(img) for img in frames]
    if not arrays or any(a.dtype != np.uint8 or a.ndim != 3 or a.shape != arrays[0].shape for a in arrays):
        return None
    x = torch.from_numpy(np.stack(arrays)).to(device=device).to(dtype=dtype)  # T H W C, exact
    x = x * ((max_value - min_value) / 255) + min_value
    return x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def install():
    from openwam.model.video_backbone.encoder import wan22_vae
    from openwam.model.video_backbone.wan import encode as wan_encode

    if getattr(wan_encode, "_fast_pp", False):
        return
    native = wan_encode._preprocess_video_native

    def preprocess_video_native(video, *, dtype, device, pattern="B C T H W", min_value=-1, max_value=1):
        out = _fast(video, dtype, device, min_value, max_value) if pattern == "B C T H W" else None
        if out is None:
            return native(video, dtype=dtype, device=device, pattern=pattern, min_value=min_value, max_value=max_value)
        return out

    preprocess_video_native.__wrapped__ = native
    wan_encode._preprocess_video_native = preprocess_video_native

    cls = wan22_vae.WanVideoVAEEncoder
    upstream = cls.preprocess_video

    def preprocess_video(self, frames):
        param = next(self._m.parameters())
        out = _fast(frames, param.dtype, param.device, -1.0, 1.0)
        return upstream(self, frames) if out is None else out

    preprocess_video.__wrapped__ = upstream
    cls.preprocess_video = preprocess_video
    wan_encode._fast_pp = True
