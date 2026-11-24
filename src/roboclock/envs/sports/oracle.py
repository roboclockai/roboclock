"""Privileged diagnostic oracle: each task's recorded scripted controller.

The controllers are the ones the recording drivers run; they read
privileged simulator state through the environment's active adapter and return
full native X5 commands, converted here to the shared 16-dim action. Not a
learned or visual policy.
"""

import numpy as np


class SportsOracle:
    def __init__(self, env):
        self.env = env
        self.controller = None
        self.clipped_steps = 0

    def reset(self, observation):
        self.controller = self.env.adapter.make_controller()
        self.clipped_steps = 0

    @property
    def phase(self):
        return getattr(self.controller, "phase", None)

    def act(self, observation):
        if self.controller is None:
            self.reset(observation)
        adapter = self.env.adapter
        action = adapter.joints.to_unified(self.controller.act())
        low, high = self.env.action_space.low, self.env.action_space.high
        bounded = np.clip(action, low, high)
        # Recorded controllers stay inside the bounds; count any exception.
        self.clipped_steps += int(np.any(bounded != action))
        return bounded.astype(np.float32)
