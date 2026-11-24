import numpy as np

from roboclock.envs.hospital.mdp.events import SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, WindowTask


class InstrumentHandoff(WindowTask):
    """A live procedure: the surgeon calls for one instrument at a time and it must reach the tray fast.

    Step k is called at t_call (the start of its window), when the wall board swaps its "standing by"
    panel for the instrument's name. The instrument must be in the Mayo tray within the response
    window that follows. Which instrument is wanted is not known before the call.
    """

    task_id = "A3"
    name = "instrument_handoff"
    discipline = "surgery"

    def __init__(self, spec):
        super().__init__(spec)
        self.called = {}  # step -> time of the call

    def gen_instruction(self):
        return (
            "You are the scrub nurse in a live operation; the surgeons at the table (not shown) call out what "
            "they need. When a call comes (the wall board describes the situation, not the instrument, the "
            "emergency light flashes and the step's dial starts running), decide which "
            "instrument that calls for and move it from the back table to the Mayo tray before the step's light "
            "turns red. Do not put any other instrument in the tray."
        )

    def tilt_ok(self, obj) -> bool:
        """A tool must lie flat, but it may rest on either face: roll about its long axis is free."""
        q = obj.orientation
        return abs(2 * (q[1] * q[3] - q[0] * q[2])) <= np.sin(self.spec.orientation_tolerance)

    def slots(self):
        return [
            (f"instr_{k}", "tray", f"ind_step_{k}", window)
            for k, window in enumerate(self.spec.windows)
        ]

    def process(self, dt):
        commands = super().process(dt)
        open_call = any(
            k in self.called and f"instr_{k}" not in self.action_times and self.time <= end
            for k, (_, end) in enumerate(self.spec.windows)
        )
        commands += self.flash("alert", open_call, (1.0, 0.08, 0.05), (0.22, 0.03, 0.03))
        for k, (start, _) in enumerate(self.spec.windows):
            if k not in self.called and self.time + EPS >= start:
                self.called[k] = self.time
                self.event_log.append(
                    {
                        "time": self.time,
                        "type": "call",
                        "step": k,
                        "tool": self.layout.steps[k],
                        "situation": self.layout.info["situations"][k],
                    }
                )
                commands += [
                    SceneCommand("show", f"call_{k}", True),
                    SceneCommand("show", f"wait_{k}", False),
                ]
        return commands

    def metrics(self):
        result = super().metrics()
        result["call_times"] = dict(self.called)
        result["response_times"] = {
            name: self.action_times[name] - self.called[k]
            for k, (name, _, _, _) in enumerate(self.slots())
            if name in self.action_times and k in self.called
        }
        return result
