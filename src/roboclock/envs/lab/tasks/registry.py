"""Explicit task registry, following RoboDojo's task/config separation."""

from roboclock.envs.lab.config import load_config, sample_episode, throughput_episode
from roboclock.envs.lab.tasks.reactive.alarm_transfer import AlarmTransfer
from roboclock.envs.lab.tasks.reactive.equipment_faults import (
    MotorStall,
    ThermalShortCircuit,
    UnstableTestTube,
)
from roboclock.envs.lab.tasks.reactive.safety_interlocks import (
    FreezerDoor,
    VacuumVent,
    WasteClamp,
)
from roboclock.envs.lab.tasks.temporal_window.bath_transfer import FilmDevelopment, HeatShockTransfer
from roboclock.envs.lab.tasks.temporal_window.liquid_processes import (
    Chromatography,
    PhaseSeparation,
    Titration,
    WaterCollection,
)
from roboclock.envs.lab.tasks.throughput.calibration_placement import CalibrationPlacement
from roboclock.envs.lab.tasks.throughput.rack_loading import RackLoading
from roboclock.envs.lab.tasks.throughput.rack_organization import RackOrganization

TASKS = {
    cls.task_id: cls
    for cls in (
        WaterCollection,
        PhaseSeparation,
        Chromatography,
        Titration,
        FilmDevelopment,
        HeatShockTransfer,
        AlarmTransfer,
        ThermalShortCircuit,
        MotorStall,
        UnstableTestTube,
        FreezerDoor,
        VacuumVent,
        WasteClamp,
        RackOrganization,
        RackLoading,
        CalibrationPlacement,
    )
}


def make_task(task_id: str, difficulty="easy", seed=0, config=None):
    config = config or load_config()
    task_id = "B1" if task_id == "alarm_transfer" else task_id
    matches = [cls for key, cls in TASKS.items() if task_id in (key, cls.name)]
    if not matches:
        raise ValueError(f"Unknown task {task_id!r}; choose one of {', '.join(TASKS)}")
    cls = matches[0]
    if cls.category == "throughput":
        # One episode for every difficulty, run to the horizon (docs/c-class-timing.md).
        spec = throughput_episode(cls.task_id, difficulty, seed, config)
    else:
        spec = sample_episode(cls.task_id, difficulty, seed, config)
    if getattr(cls, "liquid_process", False):
        return cls(spec, parameters=config.get("liquid_tasks", {}).get(cls.task_id, {}))
    if getattr(cls, "bath_task", False):
        return cls(spec, parameters=config.get("bath_tasks", {}).get(cls.task_id, {}))
    return cls(spec)
