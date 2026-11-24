"""C4: blood-tube color-cap triage into analyzer racks (score_version 2, docs/c-class-timing.md).

Jobs: Q = 12 tubes. A tube released, still and upright in a well of the rack matching its stopper
color for the placement dwell is one job, done at that moment. A tube first racked in a wrong well
is an accuracy error on the record, but it may still be moved to a right well and earn its job.

  physical  <=>  all 12 racked, each still released and upright in its well at the end, nothing
                 dropped, no tube left in a well of another rack (final misplacements <= 0).
  temporal  <=>  t_Q (the 12th tube) <= T_tier = m_tier * t_o, judged at every tier from one run.

Accuracy (tubes whose first resting well was right / every tube placed or dropped) is reported.
"""

import numpy as np

from roboclock.envs.hospital.bedside.ward_c4 import in_well, rate_steps
from roboclock.envs.hospital.mdp.events import GREEN, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, WardThroughputTask


class BloodTubeTriage(WardThroughputTask):
    task_id = "C4"
    name = "blood_tube_triage"
    discipline = "laboratory"
    bedside_task = "ward_c4"
    bedside_module = "ward_c4"
    instruction = (
        "Mass-casualty surge at the ward specimen station: take each stoppered blood tube from the "
        "arrival trays and stand it upright in a well of the analyzer rack matching its stopper "
        "color (purple = HEMATOLOGY, blue = COAG, red = CHEMISTRY)."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.placed = {}  # tube -> well where it first came to rest (for accuracy)
        self.placement_times = {}  # tube -> time it first came to rest in a well
        self.resting = {}  # tube -> well it rests in now, once past the placement dwell
        self._shown = {}

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        info = layout.info["ward_c4"]
        self.tube_class = dict(info["tube_class"])
        self.well_class = dict(info["well_class"])

    def accepts(self, name, container):
        return self.well_class.get(container) == self.tube_class.get(name)

    def goal_text(self):
        return f"Rack all {self.jobs_required} tubes as quickly as possible."

    # --- display ----------------------------------------------------------------------------

    def _show(self, group, index, count):
        if self._shown.get(group) == index:
            return []
        previous = self._shown.get(group, 0)
        self._shown[group] = index
        if self.layout is None or index >= count:
            return []
        return [
            SceneCommand("show", f"c4_{group}_{previous}", False),
            SceneCommand("show", f"c4_{group}_{index}", True),
        ]

    def live_rate(self):
        return 60.0 * len(self.correct) / max(self.time, EPS)

    def process(self, dt):
        horizon = self.spec.max_time
        left = (horizon - self.time) / horizon
        commands = [SceneCommand("indicator", "indicator", GREEN)]
        commands += self.timer("indicator", left)
        commands += self.timer("c4_shift_bar", left)
        n = len(self.layout.objects) if self.layout else self.spec.num_items
        commands += self._show("count", len(self.correct), n + 1)
        steps = rate_steps()
        rate = min(self.live_rate(), steps[-1]) if self.time > 1.0 else 0.0
        commands += self._show("rate", int(round(rate / 0.5)), len(steps))
        return commands

    # --- scoring ----------------------------------------------------------------------------

    def resting_in(self, name, obj, dt):
        """The well a released, still, upright tube has rested in for the placement dwell."""
        for well, target in self.layout.targets.items():
            ok = (
                name in self.moved
                and not obj.grasped
                and np.linalg.norm(obj.velocity) < 0.08
                and in_well(self.layout.shapes[name], obj, target)
            )
            key = (name, well)
            self.dwell[key] = self.dwell.get(key, 0.0) + dt if ok else 0.0
            if self.dwell[key] + EPS >= self.placement_dwell:
                return well
        return None

    def evaluate(self, state, dt):
        for name, obj in state.objects.items():
            if name in self.correct or name in self.dropped:
                continue
            well = self.resting_in(name, obj, dt)
            if well is not None:
                if self.resting.get(name) == well:
                    continue  # still resting where it was recorded
                self.resting[name] = well
                right = self.accepts(name, well)
                if name not in self.placed:
                    self.placed[name] = well
                    self.placement_times[name] = self.time
                self.attempted.add(name)
                self.event_log.append(
                    {
                        "time": self.time,
                        "type": "tube_racked",
                        "object": name,
                        "well": well,
                        "correct": right,
                    }
                )
                if right:
                    self.correct.add(name)
                    self.counted[name] = well
                    self.record_action(self.time)
                continue
            self.resting.pop(name, None)
            # A released tube resting away from the tray and outside every well is a failed placement.
            away = np.linalg.norm(obj.position - self.layout.objects[name].position) > 0.05
            loose = (
                name in self.moved
                and not obj.grasped
                and away
                and np.linalg.norm(obj.velocity) < 0.08
            )
            key = (name, "loose")
            self.dwell[key] = self.dwell.get(key, 0.0) + dt if loose else 0.0
            if self.dwell[key] + EPS >= self.placement_dwell:
                self.attempted.add(name)
        self.goal_achieved = len(self.correct) == len(self.layout.objects)

    @property
    def accuracy(self):
        """Tubes whose first resting well was right, over every tube placed or dropped."""
        first_right = sum(self.accepts(name, well) for name, well in self.placed.items())
        return first_right / max(1, len(self.attempted | self.dropped))

    def inside(self, name, obj, container):
        """Upright on the floor of the well (lenient envelope, as measured at the end)."""
        return in_well(self.layout.shapes[name], obj, self.layout.targets[container], lenient=True)

    def metrics(self):
        result = super().metrics()
        result.update(
            {
                "accuracy": self.accuracy,
                "required_accuracy": self.spec.required_accuracy,
                "placements": {
                    k: {"well": w, "time": self.placement_times[k], "correct": self.accepts(k, w)}
                    for k, w in self.placed.items()
                },
                "last_correct_time": self.last_job_time,
                "physical_model": "rigid_tube_upright_in_open_well",
            }
        )
        return result
