from roboclock.envs.lab.mdp.events import SceneCommand
from roboclock.envs.lab.tasks.base import ThroughputTask


class SampleSorting(ThroughputTask):
    task_id = "C1"
    name = "sample_sorting"
    discipline = "biology"
    instruction = "Sort arriving samples by color: red to the left bin, blue to the middle bin, green to the right bin."
    continuous_arrivals = True
    bin_sorting = True
    arrival_interval = 2.0

    def process(self, dt):
        commands = []
        for i, name in enumerate(self.layout.objects):
            if name not in self.active and self.time >= i * self.arrival_interval:
                self.active.add(name)
                commands.append(SceneCommand("activate", name, None))
        return commands
