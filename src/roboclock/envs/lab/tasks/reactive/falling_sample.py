from roboclock.envs.lab.mdp.events import SceneCommand
from roboclock.envs.lab.mdp.geometry import object_extents
from roboclock.envs.lab.tasks.base import ReactiveTask


class FallingSample(ReactiveTask):
    task_id = "B3"
    name = "falling_sample"
    discipline = "physics"
    asset = "calibration"
    instruction = (
        "Catch the calibration sample after its unexpected release, before it hits the table."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.hit_table = False

    def process(self, dt):
        before = self.event_time
        commands = super().process(dt)
        if before is None and self.event_time is not None:
            commands.append(SceneCommand("release", "sample_0", [0.0, 0.0, 0.0]))
        return commands

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        if self.event_time is None:
            if obj.grasped:
                self.record_reaction()
            return
        bottom = obj.position[2] - object_extents(self, "sample_0", obj)[2]
        if obj.table_contact or obj.fixture_contacts or bottom <= self.layout.table_height + 0.0005:
            self.hit_table = True
            self.violation("sample_hit_table")
            if "surface_impact" not in self.physical_failures:
                self.physical_failures.append("surface_impact")
        if obj.grasped and self.action_time is None:
            self.record_reaction()
        retained = obj.grasped and obj.relative_tool_speed <= 0.20 and not self.hit_table
        self.dwell["catch"] = self.dwell.get("catch", 0) + dt if retained else 0
        if self.dwell["catch"] >= self.catch_dwell and not self.hit_table:
            self.correct.add("sample_0")
            self.attempted.add("sample_0")
            self.goal_achieved = True

    @property
    def terminated(self):
        return super().terminated or self.hit_table
