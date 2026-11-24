from roboclock.envs.lab.tasks.base import ThroughputTask


class RackLoading(ThroughputTask):
    task_id = "C2"
    name = "rack_loading"
    discipline = "chemistry"
    instruction = "Load each sample into the rack slot with its matching color, keeping every container upright."
