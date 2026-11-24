"""Contact-verified sorting into one rack per tube color (a batch C task, see ThroughputTask)."""

import numpy as np

from roboclock.envs.lab.mdp.geometry import bottom_tip, fits_xy, in_slot
from roboclock.envs.lab.tasks.base import ThroughputTask


class RackOrganization(ThroughputTask):
    task_id = "C1"
    name = "rack_organization"
    discipline = "biology"
    asset = "test_tube"
    rack_organization = True
    rack_colors = ("red", "blue", "green")
    instruction = (
        "Sort the mixed red, blue and green test tubes from the gray MIXED racks into the "
        "color racks: one color per rack. Any empty slot in the matching-color rack "
        "is valid. Keep tubes upright and release each tube fully into its slot."
    )

    def compatible_targets(self, name):
        color = int(name.split("_")[-1]) % 3
        return [f"target_{color * 12 + slot}" for slot in range(12)]

    def observe(self, state, dt):
        # Bind scoring to the actually entered same-color hole, independently of
        # the diagnostic planner's chosen destination. Changing holes requires
        # a fresh downward crossing; a previous insertion cannot be reused.
        for name in self.active:
            obj = state.objects[name]
            rim = self.layout.targets[self.layout.assignments[name]].rim_z
            # Source and color racks are separate, so only a tube that has left
            # its source hole can enter a color-rack hole.
            if (
                np.linalg.norm(obj.position[:2] - self.layout.objects[name].position[:2]) < 0.05
                or bottom_tip(self, name, obj)[2] > rim + 0.018
            ):
                continue
            actual = next(
                (
                    key
                    for key in self.compatible_targets(name)
                    if fits_xy(self, name, obj, self.layout.targets[key])
                ),
                None,
            )
            if actual is not None and actual != self.layout.assignments[name]:
                self.layout.assignments[name] = actual
                self.insertion_times.pop(name, None)
                self.entry_times.pop(name, None)
                self.dwell.pop((name, "correct"), None)
        super().observe(state, dt)

    def placed(self, name, state, dt, precise=False):
        target_name = self.layout.assignments[name]
        target = self.layout.targets[target_name]
        if any(
            other != name
            and target_name in state.objects[other].fixture_contacts
            and self.in_place(other, state.objects[other], target)
            for other in self.correct
        ):
            self.dwell[(name, "correct")] = 0.0
            return False
        return super().placed(name, state, dt, precise=True)

    def in_place(self, name, obj, target):
        # A credited tube stays valid while it stands in its hole, also when it leans
        # against the hole wall (``in_slot``); credit itself still needs ``seated``.
        return in_slot(self, name, obj, target)

    def free_targets(self, name, state):
        return [
            key
            for key in self.compatible_targets(name)
            if not any(
                other != name
                and np.linalg.norm(obj.position[:2] - self.layout.targets[key].position[:2]) < 0.035
                and obj.position[2] < self.layout.targets[key].rim_z + 0.09
                for other, obj in state.objects.items()
            )
        ]

    def metrics(self):
        result = super().metrics()
        result["sorting_rule"] = "one_color_per_rack_any_free_slot"
        result["rack_colors"] = list(self.rack_colors)
        result["correct_by_color"] = {
            color: sum(int(name.split("_")[-1]) % 3 == i for name in self.correct)
            for i, color in enumerate(self.rack_colors)
        }
        return result
