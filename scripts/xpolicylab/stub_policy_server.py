"""XPolicyLab websocket server with a stand-in for Pi_05, for wiring and network checks.

It decodes and validates each observation where Pi_05 would (three RGB cameras, dual_x5
joint state) and answers with a chunk that holds the observed state. --probe adds a slow
joint-1 ramp (+ on the left arm, - on the right) to confirm arm order and direction in
the simulator. No GPU or checkpoint is needed.

Run it with a Python that has XPolicyLab's websocket dependencies (e.g. the Pi_05 venv):
  python scripts/xpolicylab/stub_policy_server.py --xpolicylab ~/RoboDojo/XPolicyLab --port 6000
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

DUAL_X5 = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class StubPi05:
    def __init__(self, chunk, probe, pack, unpack):
        self.chunk, self.probe = chunk, probe
        self.pack, self.unpack = pack, unpack
        self.state = None
        self.calls = 0

    def reset(self):
        self.state = None
        print("[stub] reset", flush=True)

    def update_obs(self, obs):
        shapes = {}
        for camera in CAMERAS:
            image = np.asarray(obs["vision"][camera]["color"])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{camera} is not an HxWx3 uint8 RGB image: {image.shape}")
            shapes[camera] = image.shape[:2]
        self.state = self.pack(obs, "joint", DUAL_X5, source_type="obs").astype(np.float32)
        self.calls += 1
        if self.calls <= 3 or self.calls % 20 == 0:
            head = np.asarray(obs["vision"]["cam_head"]["color"]).reshape(-1, 3).mean(axis=0)
            print(
                f"[stub] obs {self.calls}: {obs.get('instruction', '')[:50]!r} cameras={shapes} "
                f"head_mean_rgb={np.round(head).astype(int).tolist()} "
                f"state={np.round(self.state, 3).tolist()}",
                flush=True,
            )

    def get_action(self):
        if self.state is None:
            raise AssertionError("update_obs first")
        chunk = np.repeat(self.state[None], self.chunk, axis=0)
        ramp = np.linspace(0.0, self.probe, self.chunk, dtype=np.float32)
        chunk[:, 0] += ramp
        chunk[:, 7] -= ramp
        return self.unpack(chunk, "joint", DUAL_X5, source_type="obs")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--xpolicylab", type=Path, required=True, help="XPolicyLab checkout directory"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--chunk", type=int, default=50, help="Actions per chunk (Pi_05: 50)")
    parser.add_argument(
        "--probe", type=float, default=0.0, help="Joint-1 ramp per chunk in radians"
    )
    args = parser.parse_args()

    root = args.xpolicylab.resolve()
    if root.name != "XPolicyLab":
        parser.error("--xpolicylab must point at a directory named XPolicyLab")
    sys.path[:0] = [str(root.parent), str(root)]
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

    model = StubPi05(args.chunk, args.probe, pack_robot_state, unpack_robot_state)
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port))
    print(
        f"[stub] serving ws://{args.host}:{args.port} (chunk={args.chunk}, probe={args.probe})",
        flush=True,
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
