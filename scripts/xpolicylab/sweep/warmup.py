"""Warm an XPolicyLab policy server up before episodes use it.

  python warmup.py PORT POLICY      (POLICY = module:factory of the sweep's client, e.g.
                                     roboclock.envs.lab.policies.internw0:make)

A fresh server is slow on its first query: pi0.5 compiles the model (about 20 s on an A100), and
InternW0-Delta's first query takes about 2.3 s against about 1.1 s later (D zero sweep, 2026-11-15).
Under LATENCY=measured the server's latency_ms is charged to the episode, so the first episode on each
server would stand still that much longer. A dummy episode of black images and zero joints takes two
queries here instead, through the client's own reset() and act() at 25 Hz; the episode's reset()
then clears the server's session. Prints each query's server time.
"""

import importlib
import os
import sys

import numpy as np

os.environ["XPOLICYLAB_URL"] = f"ws://127.0.0.1:{int(sys.argv[1])}"
os.environ.pop("XPOLICYLAB_LOG", None)
from roboclock.envs.lab.policies.xpolicylab import CAMERAS  # noqa: E402

module, factory = sys.argv[2].split(":")
policy = getattr(importlib.import_module(module), factory)()
observation = {key: np.zeros((240, 320, 3), np.uint8) for _, key in CAMERAS}
observation.update(
    joint_positions=np.zeros(16, np.float32),
    instruction="Warm-up query.",
    sim_time=np.zeros(1, np.float32),
)
policy.reset(observation)
times = []
for step in range(200):
    observation["sim_time"] = np.array([step / 25.0], np.float32)
    policy.act(observation)
    if policy.queried:
        times.append(policy.last_inference_latency)
        if len(times) == 2:
            break
print(f"port {sys.argv[1]}: warm-up server times {', '.join(f'{t:.2f} s' for t in times)}")
