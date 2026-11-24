from roboclock.envs.hospital.mdp.events import SceneCommand
from roboclock.envs.hospital.tasks.base import WindowTask


class PillDelivery(WindowTask):
    task_id = "A1"
    name = "pill_delivery"
    discipline = "nursing"
    instruction = (
        "You are the pharmacy runner. A bed's request arrives when its tray light turns green; fill "
        "that bed's tray before the light turns red. Each prescription vial has a label strip in the "
        "patient's color and a bed number. Requests overlap, and a STAT request is more urgent: "
        "serve the earliest deadline first. Leave vials that belong to no bed on the pad, and never "
        "use another patient's tray."
    )

    def slots(self):
        return [
            (f"cup_{i}", f"tray_{i}", f"ind_tray_{i}", window)
            for i, window in enumerate(self.spec.windows)
        ]

    def __init__(self, spec):
        super().__init__(spec)
        self._announced = set()

    def process(self, dt):
        commands = super().process(dt)
        for k, (start, end) in enumerate(self.spec.windows):
            if k not in self._announced and self.time >= start:
                self._announced.add(k)  # the request appears on the wall screen
                commands.append(SceneCommand("show", f"request_{k}", True))
        return commands
