import math
import numpy as np

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, ReactiveTask, any_arm_pressing


def _seconds(limit: float) -> str:
    """A limit in seconds for the instruction, rounded down to 0.1 s so it never overstates."""
    return f"{math.floor(limit * 10 + 1e-9) / 10:g}"


ORANGE = (1.0, 0.5, 0.0)
# Seconds from both pads on the chest to SHOCK ready (rhythm analysis and charge). 8 s since 2026-11-16
# (was 2 s): Philips HeartStart FRx "typically in 8 seconds"; ZOLL AED Plus at most 12 s; the
# response limits in configs/*.yaml moved out by the same 6 s.
ANALYSIS = 8.0


class CardiacArrest(ReactiveTask):
    """A patient arrests: free the pads, put them on the chest, and deliver the shock in time.

    The pad cartridge is locked until its green tab is pulled away. Pad 1 and pad 2 go on their marks on the
    chest. The AED then analyzes for 8 s (ANALYSIS) and its SHOCK light comes on; pressing the SHOCK button before
    the pads are on is a physical failure, and pressing it before the light is ignored. t_shock - t_arrest < limit, strict.
    """

    task_id = "B2"
    name = "cardiac_arrest_aed"
    consumables = ("tab_0",)
    discipline = "emergency"
    tab, pads = "tab_0", ("pad_0", "pad_1")

    def __init__(self, spec):
        super().__init__(spec)
        self.tab_time = None
        self.pad_times = {}
        self.shock_ready_time = None
        self.shock_time = None
        self.premature_press = False
        self._pads_released = False

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        self.button = np.array(layout.info["aed"]["button"])

    @property
    def press_ready(self):
        return self.shock_ready

    @property
    def press_done(self):
        return self.shock_time is not None

    @property
    def shock_ready(self):
        return self.shock_ready_time is not None and self.time >= self.shock_ready_time

    def gen_instruction(self):
        return (
            "A patient has collapsed. When the alarm sounds, pull the AED's green tab to free the pads, put pad 1 "
            "and pad 2 on their marks on the patient's chest, and press the orange SHOCK button when the unit is "
            f"ready. Deliver the shock within {_seconds(self.spec.response_limit)} seconds of the alarm."
        )

    def process(self, dt):
        commands = []
        self.trigger_event()
        limit = self.spec.response_limit
        if self.event_time is not None and self.shock_time is None:
            if self.time - self.event_time + EPS >= limit:
                self.violation("late_shock", self.time - self.event_time - limit)
        if self.event_time is None:
            light = BLUE
        elif self.shock_time is None:
            light = RED
            commands += self.timer("indicator", (self.event_time + limit - self.time) / limit)
        else:
            light = GREEN
        commands.append(SceneCommand("indicator", "indicator", light))
        code_blue = self.event_time is not None and self.shock_time is None
        commands += self.flash("alert", code_blue, (0.10, 0.35, 1.0), (0.03, 0.05, 0.20))
        if self.shock_time is not None:
            shock_light = GREEN
        else:
            shock_light = ORANGE if self.shock_ready else BLUE
        commands.append(SceneCommand("indicator", "shock", shock_light))
        if self.tab_time is not None and not self._pads_released:
            self._pads_released = True
            commands += [SceneCommand("release", pad, [0.0, 0.0, 0.0]) for pad in self.pads]
        return commands

    # -- scoring ----------------------------------------------------------------------------

    def _dwell(self, key, condition, dt):
        self.dwell[key] = self.dwell.get(key, 0.0) + dt if condition else 0.0
        return self.dwell[key] + EPS >= self.placement_dwell

    def _at_rest(self, name, state):
        obj = state.objects[name]
        return not obj.grasped and float(np.linalg.norm(obj.velocity)) < 0.08

    def evaluate(self, state, dt):
        tab, home = state.objects[self.tab], self.layout.objects[self.tab].position
        moved = float(np.linalg.norm(tab.position[:2] - home[:2]))
        if self.event_time is None:
            if any(state.objects[n].grasped for n in (self.tab, *self.pads)):
                self.violation("pre_event_action")
            return
        if self.tab_time is None and self._dwell(
            "tab", moved > 0.03 and self._at_rest(self.tab, state), dt
        ):
            self.tab_time = self.time
            self.correct.add(self.tab)
            self.event_log.append({"time": self.time, "type": "tab_pulled"})
        for k, name in enumerate(self.pads):
            if k in self.pad_times:
                continue
            obj = state.objects[name]
            on_mark = {
                j: float(np.linalg.norm(obj.position[:2] - self.layout.zones[f"mark_{j}"].center))
                < 0.04
                for j in range(2)
            }
            height = self.layout.zones[f"mark_{k}"].support_z + self.layout.shapes[name].half[2]
            settled = (
                self._at_rest(name, state)
                and name in self.moved
                and abs(obj.position[2] - height) < 0.012
                and self.tilt_ok(obj)
            )
            if self._dwell(("pad", k), on_mark[k] and settled, dt):
                self.pad_times[k] = self.time
                self.correct.add(name)
                self.event_log.append({"time": self.time, "type": f"pad_{k + 1}_on"})
            elif self._dwell(("wrong_pad", k), on_mark[1 - k] and settled, dt):
                self.physical_failure("wrong_pad_placement")
        if len(self.pad_times) == 2 and self.shock_ready_time is None:
            self.shock_ready_time = max(self.pad_times.values()) + ANALYSIS
            self.event_log.append({"time": self.time, "type": "analyzing"})
        pressing = any_arm_pressing(state, self.button)
        if self._dwell("press", pressing, dt) and self.shock_time is None:
            if len(self.pad_times) < 2:
                self.physical_failure("shock_without_pads")
            elif not self.shock_ready:
                if not self.premature_press:
                    self.premature_press = True
                    self.event_log.append({"time": self.time, "type": "premature_press"})
            else:
                self.shock_time = self.time
                self.record_action(self.time)
                self.attempted.add(self.tab)
                if self.time - self.event_time + EPS >= self.spec.response_limit:
                    self.violation(
                        "late_shock", self.time - self.event_time - self.spec.response_limit
                    )
                self.event_log.append({"time": self.time, "type": "shock_delivered"})
                self.goal_achieved = True

    def metrics(self):
        result = super().metrics()
        result.update(
            tab_time=self.tab_time,
            pad_times={f"pad_{k}": t for k, t in self.pad_times.items()},
            shock_ready_time=self.shock_ready_time,
            shock_time=self.shock_time,
            response_time=None if self.shock_time is None else self.shock_time - self.event_time,
        )
        return result
