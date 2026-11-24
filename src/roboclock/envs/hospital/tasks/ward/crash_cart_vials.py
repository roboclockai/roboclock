"""C3: restock a crash cart's medication drawer with color-coded vials after a Code Blue.

Jobs (score_version 2, docs/c-class-timing.md): Q = 12 vials. A vial resting released, upright
(spec.orientation_tolerance, fixed at 0.2 rad) and still for the placement dwell on the floor of
a drawer pocket whose lane carries its drug is one job, done at that moment.

  physical  <=>  all 12 stocked, each still released and upright inside its pocket at the end,
                 nothing dropped, no vial left in a pocket of another drug (misplacements <= 0).
  temporal  <=>  t_Q (the 12th vial) <= T_tier = m_tier * t_o, judged at every tier from one run.
"""

import numpy as np

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, WardThroughputTask

COUNT_PREFIX = "c3_count_"  # matches bedside/ward_c3.py


class CrashCartVials(WardThroughputTask):
    task_id = "C3"
    name = "crash_cart_vials"
    discipline = "emergency_logistics"
    bedside_module = "ward_c3"
    instruction = (
        "A Code Blue just ended and the crash cart's medication drawer is depleted. Using both "
        "arms, move each color-capped vial (EPI red, ATRO green, AMIO blue) from the supply trays "
        "on the cart top and stand it upright in a pocket of the drawer-insert lane labeled with "
        "the same drug."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self._shown_count = 0
        self.placement_times = {}

    def goal_text(self):
        return f"Stock all {self.jobs_required} vials as quickly as possible."

    # --- scoring ------------------------------------------------------------------------------

    def accepts(self, name, container):
        info = self.layout.info["c3"]
        return info["item_drug"].get(name) == info["pocket_drug"].get(container)

    def evaluate(self, state, dt):
        for name, obj in state.objects.items():
            if name in self.correct or name in self.dropped:
                continue
            for container in self.layout.targets:
                if self.settled(name, container, state, dt, upright=True):
                    self.attempted.add(name)
                    if self.accepts(name, container):
                        self.correct.add(name)
                        self.counted[name] = container
                        self.placement_times[name] = self.time
                        self.record_action(self.entries.get((name, container), self.time))
                        self.event_log.append(
                            {"time": self.time, "type": "vial_stocked", "object": name, "pocket": container}
                        )
                    elif name not in self.placement_times:
                        self.placement_times[name] = self.time
                        self.event_log.append(
                            {"time": self.time, "type": "wrong_pocket", "object": name, "pocket": container}
                        )
                    break
            else:
                # A released vial at rest away from its supply place (tipped, wrong spot) is an attempt.
                away = np.linalg.norm(obj.position - self.layout.objects[name].position) > 0.05
                wrong = name in self.moved and not obj.grasped and away and np.linalg.norm(obj.velocity) < 0.08
                key = (name, "attempt")
                self.dwell[key] = self.dwell.get(key, 0.0) + dt if wrong else 0.0
                if self.dwell[key] + EPS >= self.placement_dwell:
                    self.attempted.add(name)
        self.goal_achieved = len(self.correct) == len(self.layout.objects)

    @property
    def accuracy(self) -> float:
        return len(self.correct) / max(1, len(self.attempted | self.dropped))

    def job_valid(self, name, container, obj):
        """Still released inside its pocket and upright at the end."""
        return super().job_valid(name, container, obj) and self.tilt_ok(obj)

    # --- on-screen timing ---------------------------------------------------------------------

    def process(self, dt):
        horizon = self.spec.max_time
        count = len(self.correct)
        light = GREEN if count >= self.jobs_required else BLUE
        commands = [SceneCommand("indicator", "indicator", light)]
        commands += self.timer("indicator", (horizon - self.time) / horizon)
        if count != self._shown_count:
            commands.append(SceneCommand("show", f"{COUNT_PREFIX}{self._shown_count}", False))
            commands.append(SceneCommand("show", f"{COUNT_PREFIX}{count}", True))
            self._shown_count = count
        return commands

    def metrics(self):
        result = super().metrics()
        result["placement_times"] = dict(self.placement_times)
        result["last_placement_time"] = max(self.placement_times.values(), default=None)
        return result
