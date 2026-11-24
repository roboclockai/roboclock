from dataclasses import replace

import numpy as np

from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState


def snapshot(task, **changes):
    objects = {
        name: ObjectState(obj.position.copy(), obj.orientation.copy())
        for name, obj in task.layout.objects.items()
    }
    objects.update(changes)
    return PhysicalState(objects, np.array([0.4, 0.0, 1.0]), robot_moving=True)


def at_target(task, name="sample_0", **kwargs):
    target = task.layout.targets[task.layout.assignments[name]]
    kwargs.setdefault("fixture_contacts", frozenset([task.layout.assignments[name]]))
    kwargs.setdefault("relative_tool_speed", 0.0)
    return ObjectState(target.position.copy(), target.orientation.copy(), **kwargs)


def observe_for(task, state, duration=0.25, dt=0.01):
    for _ in range(round(duration / dt)):
        task.time += dt
        task.observe(state, dt)


def approach_targets(task, state):
    """Measured frame just above each lip, preceding the downward insertion frame."""
    previous = dict(state.objects)
    for name, obj in state.objects.items():
        target = task.layout.targets[task.layout.assignments[name]]
        if np.linalg.norm(obj.position[:2] - target.position[:2]) < 0.10:
            position = obj.position.copy()
            position[2] = target.rim_z + task.layout.dimensions[name][1] / 2 + 0.020
            previous[name] = replace(obj, position=position, fixture_contacts=frozenset())
    task.previous = replace(state, objects=previous)
