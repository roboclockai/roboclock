from roboclock.envs.hospital.mdp.events import AMBER, BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask
from roboclock.envs.lab.mdp.state import upright_error

HOOK = "fixture:hook_peg"
TEMP_STEPS = 7  # readouts 4.0 .. 10.0 C


class BloodBagTransfer(HospitalTask):
    """The cold-chain clock runs from lifting the bag out of the fridge drawer until it hangs on the hook.

    t_remove: the first tick the bag is lifted clear of its cradle. t_place: the first tick the bag's
    eyelet rests on the peg while the robot has let go. Success needs t_place - t_remove < D and the
    bag to stay hanging and still.
    """

    task_id = "A2"
    name = "blood_bag_transfer"
    category = "temporal_window"
    discipline = "transfusion"
    bag = "bag_0"
    hang_dwell = 0.5

    def __init__(self, spec):
        super().__init__(spec)
        self.removal_time = None
        self.placement_time = None
        self._temp_level = 0

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        self.rest_z = float(layout.objects[self.bag].position[2])
        self.hook_x = layout.info["hook"]["x"]
        self.peg_z = layout.info["hook"]["peg_z"]

    def gen_instruction(self):
        return (
            "Pull open the blood bank drawer, take the blood bag out, and hang it by its eyelet on "
            f"the IV stand's hook within {self.spec.transit_limit:.1f} seconds of lifting it out. "
            "The timer light is green, turns amber after 60% of the time, and red at the limit."
        )

    def process(self, dt):
        limit = self.spec.transit_limit
        if self.removal_time is None:
            color = BLUE
        else:
            elapsed = self.time - self.removal_time
            if self.placement_time is None and elapsed + EPS >= limit:
                self.violation("late_transit", elapsed - limit)
            if self.placement_time is not None:
                color = RED if "late_transit" in self.violations else GREEN
            elif elapsed + EPS >= limit:
                color = RED
            elif elapsed >= 0.6 * limit:
                color = AMBER
            else:
                color = GREEN
        commands = [SceneCommand("indicator", "indicator", color)]
        if self.removal_time is not None and self.placement_time is None:
            commands += self.timer("indicator", (limit - (self.time - self.removal_time)) / limit)
            # The fridge display shows the bag warming from 4 to 10 degrees by the time the limit runs out.
            level = min(TEMP_STEPS - 1, int(TEMP_STEPS * (self.time - self.removal_time) / limit))
            if level != self._temp_level:
                commands += [
                    SceneCommand("show", f"temp_{self._temp_level}", False),
                    SceneCommand("show", f"temp_{level}", True),
                ]
                self._temp_level = level
        return commands

    def hanging(self, obj) -> bool:
        return (
            HOOK in obj.fixture_contacts
            and not obj.grasped
            and obj.position[2] < self.peg_z - 0.03
            and abs(obj.position[0] - self.hook_x) < 0.06
        )

    def evaluate(self, state, dt):
        obj = state.objects[self.bag]
        if self.removal_time is None and obj.position[2] > self.rest_z + 0.02:
            self.removal_time = self.event_time = self.time
            self.event_log.append({"time": self.time, "type": "removed_from_fridge"})
        # The lift only counts if the robot holds the bag. A brief gap in the grasp reading while it
        # is carried is tolerated; a bag that stays loose in the air for 0.3 s was not lifted by the robot.
        loose = (
            self.removal_time is not None
            and obj.position[2] > self.rest_z + 0.02
            and not obj.grasped
            and not self.hanging(obj)
        )
        self.dwell["loose"] = self.dwell.get("loose", 0.0) + dt if loose else 0.0
        if self.dwell["loose"] + EPS >= 0.3:
            self.physical_failure("removed_without_grasp")
        if self.hanging(obj) and self.placement_time is None:
            self.placement_time = self.time
            self.record_action(self.time)
            if self.removal_time is None:
                self.removal_time = self.time
            transit = self.time - self.removal_time
            if transit + EPS >= self.spec.transit_limit:
                self.violation("late_transit", transit - self.spec.transit_limit)
        steady = (
            self.hanging(obj)
            and float(abs(obj.velocity).max()) < 0.15
            and upright_error(obj.orientation) < 0.5
        )
        self.dwell["hang"] = self.dwell.get("hang", 0.0) + dt if steady else 0.0
        if self.dwell["hang"] + EPS >= self.hang_dwell:
            self.correct.add(self.bag)
            self.attempted.add(self.bag)
            self.goal_achieved = True

    def metrics(self):
        result = super().metrics()
        result["removal_time"] = self.removal_time
        result["placement_time"] = self.placement_time
        result["transit_time"] = (
            None
            if self.removal_time is None or self.placement_time is None
            else self.placement_time - self.removal_time
        )
        return result
