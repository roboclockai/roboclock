from roboclock.envs.lab.tasks.base import ThroughputTask


class CalibrationPlacement(ThroughputTask):
    task_id = "C3"
    name = "calibration_placement"
    discipline = "physics"
    asset = "calibration"
    instruction = "Place each calibration sample on its matching color marker, aligned with the marker's white axis."
