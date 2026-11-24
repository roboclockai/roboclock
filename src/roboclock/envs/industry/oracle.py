"""Recorded privileged diagnostic controllers as a 16-joint position oracle.

Each Industry task was recorded with a native controller that reads privileged
scene state (product poses, contact forces, events) and writes the two arms'
joint-position targets (``DualARX.commands``). The environment creates that
controller with the scene at time 0, as the recorded runner did. ``act()`` runs it
on the latest native observation and returns its targets as the action.

Two details keep the closed loop identical to the recording:

* Commands stay float64 inside the controller between steps (its IK seeds and rate
  limits integrate them); the environment receives them as the float32 action.
  PhysX drive targets are float32 either way, so the applied targets are the same.
* Controller writes to ``scene.events`` go to a private overlay. The controller
  still reads every scene/monitor event through it, but the scene's own predicates
  and the environment's scoring never see controller-written events.

Process-side work that recordings did inside the controllers (Task 43's adhesive
dispenser, Task 46's machine outfeed and gate) is part of the scene and runs in the
environment for every policy; Task 50's case-grip finger gains are a task rule in
``scoring.PaletteMonitor``.
"""

from collections import ChainMap

import numpy as np

SIDES = ("left", "right")


class IndustryOracle:
    def __init__(self, env):
        self.env = env
        self.controller = None

    def reset(self, observation):
        env = self.env
        if env.native_controller is None:
            raise RuntimeError("The Industry oracle needs an environment created with oracle=True")
        self.controller = env.native_controller
        self.overlay = {}
        self.commands = {side: env.scene.dual.commands[side].copy() for side in SIDES}

    def act(self, observation):
        env = self.env
        scene, dual = env.scene, env.scene.dual
        for side in SIDES:
            dual.commands[side][:] = self.commands[side]
        events = scene.events
        scene.events = ChainMap(self.overlay, events)
        try:
            self.controller.step(env.native_observation)
        finally:
            scene.events = events
        self.commands = {side: dual.commands[side].copy() for side in SIDES}
        action = np.concatenate([self.commands[side][env.joint_index[side]] for side in SIDES])
        return action.astype(np.float32)

    @property
    def phase(self):
        return getattr(self.controller, "phase", None)

    @property
    def failure(self):
        return getattr(self.controller, "failure", None)
