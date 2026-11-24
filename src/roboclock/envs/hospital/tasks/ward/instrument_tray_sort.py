"""C2: sort loose surgical instruments from a sterilization basin into labeled Mayo-tray slots.

Jobs (score_version 2, docs/c-class-timing.md): Q = 6 instruments. Every scalpel, hemostat and
tissue forceps lifted out of the basin and laid flat, released and at rest in a silicone slot of
its own type is one job, done at that moment.

  physical  <=>  all 6 sorted, each still released in its slot at the end, nothing dropped and no
                 instrument left in a slot of another type (final misplacements <= 0).
  temporal  <=>  t_Q (the 6th sort) <= T_tier = m_tier * t_o, judged at every tier from one run.

Accuracy (correct / placements) is reported: placements count every settled slot placement
(right or wrong), every instrument released at rest outside both the basin and the slots, and
every instrument dropped to the floor. An instrument that is nudged or dropped back into the
basin is not a placement and can be picked again.
"""

import numpy as np

from roboclock.envs.hospital.mdp.events import SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, WardThroughputTask


class InstrumentTraySort(WardThroughputTask):
    task_id = "C2"
    name = "instrument_tray_sort"
    discipline = "sterile_processing"
    bedside_module = "ward_c2"
    instruction = (
        "OR turnover for two back-to-back cases: take the loose stainless instruments out of the "
        "sterilization basin and lay each one flat in a silicone slot of its own type (SCALPEL, "
        "HEMOSTAT or FORCEPS, matching the colored tape) on the CASE 1 or CASE 2 Mayo tray."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.resting = {}  # item -> slot (or "stray") it last settled in since it was last lifted
        self.wrong = []  # (item, slot, time) settled in a slot of another type
        self.strays = []  # (item, time) released at rest outside the basin and the slots
        self._shown = {}

    # --- geometry helpers ------------------------------------------------------------------

    @property
    def c2(self):
        return self.layout.info["c2"]

    def accepts(self, name, container):
        info = self.c2
        return info["slot_types"].get(container) == info["item_types"].get(name)

    def in_basin(self, obj):
        basin = self.c2["basin"]
        offset = np.abs(np.asarray(obj.position[:2]) - np.asarray(basin["center"]))
        return bool(
            np.all(offset <= np.asarray(basin["half"]) + 0.01)
            and obj.position[2] <= basin["rim_z"] + 0.03
        )

    # --- scoring -----------------------------------------------------------------------------

    @property
    def placements(self):
        return len(self.correct) + len(self.wrong) + len(self.strays) + len(self.dropped)

    @property
    def accuracy(self):
        return len(self.correct) / max(1, self.placements)

    def evaluate(self, state, dt):
        for name, obj in state.objects.items():
            if name in self.correct or name in self.dropped:
                continue
            if obj.grasped:
                self.resting.pop(name, None)
                self.dwell[(name, "stray")] = 0.0
                continue
            slot = None
            for container in self.layout.targets:
                if self.settled(name, container, state, dt, upright=False):
                    slot = container
                    break
            if slot is not None:
                if self.resting.get(name) != slot:
                    self.resting[name] = slot
                    self.attempted.add(name)
                    if self.accepts(name, slot):
                        self.correct.add(name)
                        self.counted[name] = slot
                        self.record_action(self.entries.get((name, slot), self.time))
                        self.event_log.append(
                            {"time": self.time, "type": "sorted", "object": name, "slot": slot}
                        )
                    else:
                        self.wrong.append((name, slot, self.time))
                        self.event_log.append(
                            {"time": self.time, "type": "wrong_slot", "object": name, "slot": slot}
                        )
                continue
            stray = (
                name in self.moved
                and np.linalg.norm(obj.velocity) < 0.08
                and not self.in_basin(obj)
                and not any(
                    np.all(
                        np.abs(obj.position[:2] - t.position[:2]) <= t.half_extents[:2] + 0.006
                    )
                    for t in self.layout.targets.values()
                )
            )
            key = (name, "stray")
            self.dwell[key] = self.dwell.get(key, 0.0) + dt if stray else 0.0
            if self.dwell[key] + EPS >= self.placement_dwell and self.resting.get(name) != "stray":
                self.resting[name] = "stray"
                self.attempted.add(name)
                self.strays.append((name, self.time))
                self.event_log.append({"time": self.time, "type": "stray", "object": name})
            elif self.in_basin(obj) and self.resting.get(name) == "stray":
                self.resting.pop(name, None)  # it slid back into the basin; it may be picked again
        self.goal_achieved = len(self.correct) == len(self.layout.objects)

    # --- visible readout ----------------------------------------------------------------------

    def process(self, dt):
        commands = super().process(dt)
        if "readouts" not in self.c2:
            return commands
        count = len(self.correct)
        for k in range(len(self.layout.objects) + 1):
            name, visible = f"c2_count_{k}", k == count
            if self._shown.get(name) != visible:
                self._shown[name] = visible
                commands.append(SceneCommand("show", name, visible))
        return commands

    def goal_text(self):
        return f"Sort all {self.jobs_required} instruments as quickly as possible."

    def metrics(self):
        result = super().metrics()
        result.update(
            correct_operations=len(self.correct),
            attempted_operations=self.placements,
            accuracy=self.accuracy,
            wrong_slot_placements=[list(w[:2]) for w in self.wrong],
            stray_releases=[s[0] for s in self.strays],
            sorted_items={name: slot for name, slot in self.counted.items()},
            sort_times=[e["time"] for e in self.event_log if e["type"] == "sorted"],
        )
        return result
