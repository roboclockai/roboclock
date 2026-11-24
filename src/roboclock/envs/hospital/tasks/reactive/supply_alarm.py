import numpy as np

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, ReactiveTask, any_arm_pressing


class SupplyAlarm(ReactiveTask):
    """ICU bedside pumps on critical short half-life infusions (vasopressors) alarm when the infusion is
    interrupted: press the SILENCE button of the alarming pump within its limit.

    Easy has one alarm. Medium and hard have several pumps alarming one after another, and the last one is a
    critical alarm with a shorter limit. t_press - t_alarm < limit, strict, per alarm. Pressing the button of a pump
    that is not alarming is a physical failure; pressing before its alarm is a timing violation.
    """

    task_id = "B1"
    name = "pump_alarm"
    discipline = "nursing"

    def __init__(self, spec):
        super().__init__(spec)
        self.delivery_times = {}  # pump -> time its alarm was silenced
        self._pressing_for = {}

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        self.buttons = [np.array(p["button"]) for p in layout.info["pumps"]]

    def alarms(self):
        """(pump, time, limit) for each alarm, in order."""
        times = self.spec.alarms or (self.spec.event_time,)
        limits = self.spec.alarm_limits or (self.spec.response_limit,) * len(times)
        stations = self.layout.alarm_stations
        return [(stations[i], times[i], limits[i]) for i in range(len(times))]

    @property
    def station(self):
        return self.layout.alarm_station

    @property
    def delivery_time(self):
        return self.delivery_times.get(self.station)

    def alarm_for(self, station):
        return next((a for a in self.alarms() if a[0] == station), None)

    # -- what the oracle needs: the pump to press now ---------------------------------------------

    def target(self):
        """The alarming, unsilenced pump with the earliest deadline; None while no pump alarms.

        A pump that has not alarmed yet is never a target: which pump alarms next, and when, cannot
        be seen, so the oracle may not move toward it (it used to hover over the next pump).
        """
        active = [
            a for a in self.alarms() if a[0] not in self.delivery_times and a[1] <= self.time
        ]
        return min(active, key=lambda a: a[1] + a[2]) if active else None

    @property
    def cue_observed(self):
        """A pump's light has turned red (its alarm time has passed)."""
        return any(time <= self.time for _, time, _ in self.alarms())

    @property
    def button(self):
        target = self.target()
        return None if target is None else self.buttons[target[0]]

    @property
    def press_ready(self):
        return self.target() is not None

    @property
    def press_done(self):
        return self.goal_achieved

    def gen_instruction(self):
        alarms = self.alarms()
        if len(alarms) == 1:
            return (
                "ICU bedside: each infusion pump runs a critical short half-life drug (a vasopressor) and alarms "
                "when its infusion is interrupted. When a pump's light turns red, press the SILENCE "
                f"button on that pump within {alarms[0][2]:.2f} seconds of the alarm."
            )
        return (
            "ICU bedside: the infusion pumps run critical short half-life drugs (vasopressors) and alarm when an "
            "infusion is interrupted. When a pump's light turns red, press the SILENCE button on "
            "that pump within the response limit. Several pumps can alarm close together, and the last one is a "
            f"critical alarm with a shorter limit ({alarms[-1][2]:.1f} s, against {alarms[0][2]:.1f} s for the "
            "others): silence the earliest deadline first."
        )

    def process(self, dt):
        self.trigger_event()
        alarms = self.alarms()
        for station, time, limit in alarms:
            if station not in self.delivery_times and self.time + EPS >= time + limit:
                self.violation("late_delivery", self.time - time - limit)
        commands = []
        for j in range(len(self.buttons)):
            alarm = self.alarm_for(j)
            if alarm is None or self.time < alarm[1]:
                color = BLUE
            else:
                color = GREEN if j in self.delivery_times else RED
            commands.append(SceneCommand("indicator", f"ind_bin_{j}", color))
            if alarm is not None and self.time >= alarm[1]:
                if j in self.delivery_times:
                    commands += self.timer(f"ind_bin_{j}", 1.0, "green")
                elif self.time < alarm[1] + alarm[2]:
                    commands += self.timer(
                        f"ind_bin_{j}", (alarm[1] + alarm[2] - self.time) / alarm[2]
                    )
        return commands

    def _pressing(self, state, j):
        return any_arm_pressing(state, self.buttons[j])

    def _dwell(self, key, condition, dt):
        self.dwell[key] = self.dwell.get(key, 0.0) + dt if condition else 0.0
        return self.dwell[key] + EPS >= self.placement_dwell

    def evaluate(self, state, dt):
        for j in range(len(self.buttons)):
            if j in self.delivery_times or not self._dwell(
                ("press", j), self._pressing(state, j), dt
            ):
                continue
            alarm = self.alarm_for(j)
            if alarm is None:
                self.physical_failure("wrong_pump")
            elif self.time < alarm[1]:
                self.violation("pre_event_action")  # silencing a pump before it alarms
            else:
                self.delivery_times[j] = self.time
                self.correct.add(f"pump_{j}")
                self.attempted.add(f"pump_{j}")
                self.record_action(self.time)
                self.event_log.append(
                    {"time": self.time, "type": "alarm_silenced", "object": f"pump_{j}"}
                )
                if self.time - alarm[1] + EPS >= alarm[2]:
                    self.violation("late_delivery", self.time - alarm[1] - alarm[2])
        self.goal_achieved = len(self.delivery_times) == len(self.alarms())

    def metrics(self):
        result = super().metrics()
        result["delivery_times"] = dict(self.delivery_times)
        result["response_times"] = {
            station: self.delivery_times[station] - time
            for station, time, _ in self.alarms()
            if station in self.delivery_times
        }
        return result
