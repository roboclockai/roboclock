import math
import numpy as np

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, ReactiveTask, any_arm_pressing


def _seconds(limit: float) -> str:
    """A limit in seconds for the instruction, rounded down to 0.1 s so it never overstates."""
    return f"{math.floor(limit * 10 + 1e-9) / 10:g}"


SHOW_DELAY = 1.5  # seconds the scene keeps running after the fire is covered
ALARM_SHARE = 0.4  # the alarm must be raised within this share of the response limit


class FireBlanket(ReactiveTask):
    """A dish catches fire: raise the alarm, pull both black tabs of the blanket holder, then smother the fire.

    Fire safety teaches RACE (rescue, alarm, contain, extinguish): the alarm button is pressed first, within 40 % of
    the limit. The folded blanket is locked until both tabs have been pulled away. Covering counts once the alarm is
    raised and the blanket rests, released and flat, over the dish; t_cover - t_fire < limit, strict.
    """

    task_id = "B3"
    name = "fire_blanket"
    discipline = "emergency"
    tabs, blanket = ("tab_0", "tab_1"), "blanket"
    consumables = tabs

    def __init__(self, spec):
        super().__init__(spec)
        self.tab_times = {}
        self.alarm_time = None
        self.cover_time = None
        self._blanket_released = False
        self._shown = False

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        fire = layout.info["fire"]
        self.fire_center = np.array(fire["center"])
        self.dish_top = float(fire["dish_top"])
        self.cover_radius = float(fire["cover_radius"])
        self.button = np.array(fire["alarm"])

    @property
    def press_ready(self):
        return self.event_time is not None  # the alarm is pressed once the fire has started

    @property
    def press_done(self):
        return self.alarm_time is not None

    @property
    def blanket_free(self):
        return len(self.tab_times) == len(self.tabs)

    def gen_instruction(self):
        return (
            "A dish on the bench catches fire. First press the red FIRE ALARM button, then pull both black tabs "
            "of the fire-blanket holder to free the blanket, and lay it over the burning dish. Raise the alarm "
            f"within {_seconds(ALARM_SHARE * self.spec.response_limit)} seconds and smother the fire within "
            f"{_seconds(self.spec.response_limit)} seconds of it starting."
        )

    def process(self, dt):
        commands = []
        before = self.event_time
        self.trigger_event()
        limit = self.spec.response_limit
        if self.event_time is not None and self.cover_time is None:
            if self.time - self.event_time + EPS >= limit:
                self.violation("late_cover", self.time - self.event_time - limit)
        if self.event_time is not None and self.alarm_time is None:
            if self.time - self.event_time + EPS >= ALARM_SHARE * limit:
                self.violation("late_alarm", self.time - self.event_time - ALARM_SHARE * limit)
        if before is None and self.event_time is not None:
            commands.append(SceneCommand("show", "flames", True))
        if self.event_time is None:
            light = BLUE
        elif self.cover_time is None:
            light = RED
            commands += self.timer("indicator", (self.event_time + limit - self.time) / limit)
        else:
            light = GREEN
        commands.append(SceneCommand("indicator", "indicator", light))
        strobe = self.alarm_time is not None and self.cover_time is None
        commands += self.flash("alert", strobe, (1.0, 0.08, 0.05), (0.22, 0.03, 0.03))
        alarm_light = (
            BLUE if self.event_time is None else GREEN if self.alarm_time is not None else RED
        )
        commands.append(SceneCommand("indicator", "alarm_ok", alarm_light))
        if self.blanket_free and not self._blanket_released:
            self._blanket_released = True
            commands.append(SceneCommand("release", self.blanket, [0.0, 0.0, 0.0]))
        if self.cover_time is not None and not self._shown:
            self._shown = True
            commands += [
                SceneCommand("show", "flames", False),
                SceneCommand("show", "blanket_cover", True),
            ]
        return commands

    # -- scoring ----------------------------------------------------------------------------

    def _dwell(self, key, condition, dt):
        self.dwell[key] = self.dwell.get(key, 0.0) + dt if condition else 0.0
        return self.dwell[key] + EPS >= self.placement_dwell

    def _at_rest(self, name, state):
        obj = state.objects[name]
        return not obj.grasped and float(np.linalg.norm(obj.velocity)) < 0.08

    def _pressing(self, state):
        return any_arm_pressing(state, self.button)

    def evaluate(self, state, dt):
        if self.event_time is None:
            if any(state.objects[n].grasped for n in (*self.tabs, self.blanket)):
                self.violation("pre_event_action")
            if self._dwell("press_early", self._pressing(state), dt):
                self.violation("pre_event_action")  # the alarm is raised before there is a fire
            return
        if self.alarm_time is None and self._dwell("alarm", self._pressing(state), dt):
            self.alarm_time = self.time
            self.event_log.append({"time": self.time, "type": "alarm_raised"})
        for name in self.tabs:
            if name in self.tab_times:
                continue
            home = self.layout.objects[name].position
            moved = float(np.linalg.norm(state.objects[name].position[:2] - home[:2]))
            if self._dwell(("tab", name), moved > 0.05 and self._at_rest(name, state), dt):
                self.tab_times[name] = self.time
                self.correct.add(name)
                self.event_log.append({"time": self.time, "type": "tab_pulled", "object": name})
        if self.cover_time is not None:
            # Keep the episode running briefly so the fire going out and the cloth over the dish are on screen.
            if self.time - self.cover_time >= SHOW_DELAY:
                self.goal_achieved = True
            return
        if not self.blanket_free or self.alarm_time is None:
            return  # the blanket only counts once it is free and the alarm has been raised
        obj = state.objects[self.blanket]
        height = self.dish_top + self.layout.shapes[self.blanket].half[2]
        covering = (
            float(np.linalg.norm(obj.position[:2] - self.fire_center)) < self.cover_radius
            and abs(obj.position[2] - height) < 0.02
            and self._at_rest(self.blanket, state)
            and self.blanket in self.moved
            and self.tilt_ok(obj)
        )
        if self._dwell("cover", covering, dt):
            self.cover_time = self.time
            self.correct.add(self.blanket)
            self.attempted.add(self.blanket)
            self.record_action(self.time)
            if self.time - self.event_time + EPS >= self.spec.response_limit:
                self.violation("late_cover", self.time - self.event_time - self.spec.response_limit)
            self.event_log.append({"time": self.time, "type": "fire_covered"})

    def metrics(self):
        result = super().metrics()
        result.update(
            tab_times=dict(self.tab_times),
            alarm_time=self.alarm_time,
            cover_time=self.cover_time,
            response_time=None if self.cover_time is None else self.cover_time - self.event_time,
        )
        return result
