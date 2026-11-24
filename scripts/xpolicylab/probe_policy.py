"""Send synthetic lab observations to an XPolicyLab server and report latency and actions.

Checks connectivity and the observation/action contract without starting Isaac Sim:
  XPOLICYLAB_URL=ws://<policy host>:6000 python scripts/xpolicylab/probe_policy.py --calls 5
"""

import argparse
import time

import numpy as np

from roboclock.envs.lab.policies import xpolicylab
from roboclock.envs.lab.robot import HOME


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=3, help="Inference requests to time")
    parser.add_argument("--instruction", default="stack the bowls")
    parser.add_argument("--image", help="Optional head-camera image file (default: gray frame)")
    args = parser.parse_args()

    if args.image:
        from PIL import Image

        head = np.asarray(Image.open(args.image).convert("RGB").resize((960, 720)))
    else:
        head = np.full((720, 960, 3), 128, dtype=np.uint8)
    observation = {
        "rgb": head,
        "joint_positions": np.tile(HOME, 2).astype(np.float32),
        "instruction": args.instruction,
        "sim_time": np.array([0.0], dtype=np.float32),
    }

    start = time.perf_counter()
    policy = xpolicylab.make()
    print(
        f"connected in {time.perf_counter() - start:.2f} s; "
        f"server {policy.connection.server_instance}"
    )
    policy.reset(observation)
    for call in range(args.calls):
        start = time.perf_counter()
        reply = policy.connection.request(
            "infer", {"observation": policy.observation(observation)}, trial_id=policy.trial_id
        )
        round_trip = (time.perf_counter() - start) * 1000.0
        chunk = np.stack([xpolicylab.robodojo_vector(action) for action in reply["actions"]])
        server = reply.get("latency_ms", float("nan"))
        print(
            f"call {call}: round trip {round_trip:.0f} ms, server {server:.0f} ms, "
            f"chunk {chunk.shape}, finite={bool(np.isfinite(chunk).all())}"
        )
    targets = xpolicylab.lab_action(chunk)
    outside = (targets < xpolicylab.LOW) | (targets > xpolicylab.HIGH)
    print("observed joints:", np.round(observation["joint_positions"], 3).tolist())
    print("first target:   ", np.round(targets[0], 3).tolist())
    print("last target:    ", np.round(targets[-1], 3).tolist())
    print("arm targets outside lab limits:", int(outside[:, xpolicylab.ARM_DIMS].sum()))


if __name__ == "__main__":
    main()
