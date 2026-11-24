from roboclock.envs.lab.tasks.base import WindowTask


class IncubationTransfer(WindowTask):
    task_id = "A2"
    name = "incubation_transfer"
    discipline = "biology"
    instruction = "Insert the sample upright into the rack slot during the green incubation stage."

    def process(self, dt):
        if self.time > self.spec.valid_end and self.action_time is None:
            self.violation("missed_window", self.time - self.spec.valid_end)
        return super().process(dt)
