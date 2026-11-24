import numpy as np

from roboclock.envs.hospital.mdp.events import GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask

FILL_TIME = 2.5  # seconds a syringe has to sit on the dock to fill


class SyringeDose(HospitalTask):
    """Each prepared dose goes into its own bed's tray strictly before its own beyond-use time.

    The doses expire at different times and the robot carries one at a time, so the order in which
    they are delivered decides whether all of them make it. Times count from the episode start.

    With dose preparation (layout `dose_prep`) every syringe starts empty: it is set on the filling dock, fills over
    FILL_TIME seconds while it rests there (the liquid appears, the dock dial runs), and only a filled syringe may go
    to its bed's tray.
    """

    task_id = "A4"
    name = "syringe_dose"
    category = "temporal_window"
    discipline = "pharmacy"

    def __init__(self, spec):
        super().__init__(spec)
        self.placement_times = {}
        self.drawn = {}  # syringe -> time its dose finished filling
        self._announced = set()
        self._fill_fraction = (
            None  # share of the fill done on the dock right now, if a syringe is filling
        )
        self._dock_done = False

    @property
    def dose_prep(self):
        return "dose_prep" in self.layout.info

    def docked(self, name, state):
        """The syringe is resting on the filling dock (not in the gripper, not moving)."""
        prep = self.layout.info["dose_prep"]
        obj = state.objects[name]
        rest = prep["support_z"] + self.layout.shapes[name].half[2]
        return (
            not obj.grasped
            and float(np.linalg.norm(obj.position[:2] - np.array(prep["dock"]))) < 0.05
            and abs(obj.position[2] - rest) < 0.015
            and float(np.linalg.norm(obj.velocity)) < 0.05
        )

    @property
    def expiries(self):
        return self.spec.expiries or (self.spec.expiry_time,)

    def slots(self):
        """(syringe, tray, bed light, T_exp) for each dose."""
        result = []
        for k, expiry in enumerate(self.expiries):
            name = f"syringe_{k}"
            tray = self.layout.assignments[name]
            result.append((name, tray, "ind_" + tray.split("_")[-1], expiry))
        return result

    def gen_instruction(self):
        times = ", ".join(f"{e:.0f}" for e in self.expiries)
        return (
            "Each prepared syringe dose is labeled with its bed and its beyond-use time. Place every "
            "dose in its own bed's administration tray before that time (the doses expire at "
            f"{times} seconds). A bed's dial runs down to its dose's expiry."
        ) + (
            " The syringes start empty: set each one on the DOSE DRAW dock and leave it for "
            f"{FILL_TIME:.1f} seconds until the liquid has filled it, then take it to its bed."
            if self.dose_prep
            else ""
        )

    def process(self, dt):
        commands = []
        for name, _, indicator, expiry in self.slots():
            expired = self.time + EPS >= expiry
            if expired and name not in self.placement_times:
                self.violation("expired_dose", self.time - expiry)
            commands.append(SceneCommand("indicator", indicator, RED if expired else GREEN))
            if name in self.placement_times:
                commands += self.timer(indicator, 1.0, "green")
            elif not expired:
                commands += self.timer(indicator, (expiry - self.time) / expiry)
        if self.dose_prep:
            for name in self.drawn:
                if name not in self._announced:
                    self._announced.add(name)  # the liquid appears in the syringe
                    commands.append(SceneCommand("show", f"liquid_{name.split('_')[-1]}", True))
            if self._fill_fraction is not None:
                commands += self.timer("dock", 1.0 - self._fill_fraction, "amber")
            elif self.drawn and not self._dock_done:
                self._dock_done = True
                commands += self.timer("dock", 1.0, "green")
        return commands

    def evaluate(self, state, dt):
        slots = self.slots()
        if self.dose_prep:
            self._fill(state, dt)
        own = {name: tray for name, tray, _, _ in slots}
        for name, container in self.entries:
            if own.get(name) != container:
                self.physical_failure("wrong_container")
        for name, tray, _, expiry in slots:
            entered = self.entries.get((name, tray))
            if entered is None:
                continue
            if self.dose_prep and name not in self.drawn:
                self.physical_failure(
                    "undrawn_dose"
                )  # an empty syringe was put in a patient's tray
            if name not in self.placement_times:
                self.placement_times[name] = entered
                self.record_action(entered)
                if entered + EPS >= expiry:
                    self.violation("expired_dose", entered - expiry)
            if self.settled(name, tray, state, dt, upright=False):
                self.correct.add(name)
                self.attempted.add(name)
        self.goal_achieved = len(self.correct) == len(slots)

    def _fill(self, state, dt):
        """Syringes resting on the dock fill; one at a time fills (the dock holds one)."""
        self._fill_fraction = None
        for name in (n for n, *_ in self.slots()):
            if name in self.drawn:
                continue
            key = ("fill", name)
            if self.docked(name, state):
                self.dwell[key] = self.dwell.get(key, 0.0) + dt
                self._fill_fraction = min(self.dwell[key] / FILL_TIME, 1.0)
                if self.dwell[key] + EPS >= FILL_TIME:
                    self.drawn[name] = self.time
                    self._fill_fraction = None
                    self.event_log.append({"time": self.time, "type": "dose_drawn", "object": name})
            else:
                self.dwell[key] = 0.0

    @property
    def temporal_success(self):
        return len(self.placement_times) == len(self.slots()) and not self.violations

    def metrics(self):
        result = super().metrics()
        result["placement_times"] = dict(self.placement_times)
        result["expiries"] = list(self.expiries)
        result["drawn_times"] = dict(self.drawn)
        return result
