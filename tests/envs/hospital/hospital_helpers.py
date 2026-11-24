from dataclasses import replace

import numpy as np

from roboclock.envs.hospital.mdp.geometry import object_extents
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState


def snapshot(task, **changes):
    objects = {
        name: ObjectState(obj.position.copy(), obj.orientation.copy())
        for name, obj in task.layout.objects.items()
    }
    objects.update(changes)
    return PhysicalState(objects, np.array([0.4, 0.0, 1.0]), robot_moving=True)


def in_container(task, name, container, **kwargs):
    """Resting on the container floor, released and still."""
    layout = task.layout
    target = layout.targets[container]
    orientation = kwargs.pop("orientation", layout.objects[name].orientation)
    extent = object_extents(layout.shapes[name], orientation)
    position = np.array([*target.position[:2], target.support_z + extent[2] + 0.001])
    kwargs.setdefault("fixture_contacts", frozenset([container]))
    return ObjectState(position, orientation.copy(), **kwargs)


def above_container(task, name, container):
    """Frame just above the lip, preceding the downward entry frame."""
    layout = task.layout
    target = layout.targets[container]
    obj = layout.objects[name]
    extent = object_extents(layout.shapes[name], obj.orientation)
    position = np.array([*target.position[:2], target.rim_z + extent[2] + 0.02])
    return ObjectState(position, obj.orientation.copy(), grasped=True)


def observe_for(task, state, duration=0.25, dt=0.01):
    for _ in range(round(duration / dt)):
        task.time += dt
        task.observe(state, dt)


def place(task, name, container, at=None, dwell=0.25, **kwargs):
    """Lower an item into a container at time `at` and let it settle."""
    if at is not None:
        task.time = at
    approach = replace(
        snapshot(task),
        objects={**snapshot(task).objects, name: above_container(task, name, container)},
    )
    task.observe(approach, 0.01)
    final = snapshot(task, **{name: in_container(task, name, container, **kwargs)})
    task.observe(final, 0.01)
    observe_for(task, final, dwell)


# --- ward C1-C4 (score_version 2): one correct final state per job ---------------------------


def ward_solution(task):
    """{item: ObjectState} that does every job of a ward C task, in job order (C1: by order)."""
    import math

    layout, tid = task.layout, task.task_id
    solution = {}
    if tid == "C1":
        used = set()
        for bin_name, order in layout.orders.items():
            for kind, count in order.items():
                for _ in range(count):
                    vial = next(
                        v for v, k in layout.item_types.items() if k == kind and v not in used
                    )
                    used.add(vial)
                    solution[vial] = in_container(task, vial, bin_name)
    elif tid == "C2":
        info, used = layout.info["c2"], set()
        q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])  # long axis along y
        for name in layout.objects:
            slot = next(
                s
                for s, k in info["slot_types"].items()
                if k == info["item_types"][name] and s not in used
            )
            used.add(slot)
            target = layout.targets[slot]
            extent = object_extents(layout.shapes[name], q)
            position = np.array([*target.position[:2], target.support_z + extent[2] + 0.0005])
            solution[name] = ObjectState(position, q, fixture_contacts=frozenset([slot]))
    elif tid == "C3":
        info, used = layout.info["c3"], set()
        for name in layout.objects:
            pocket = next(
                p
                for p, d in info["pocket_drug"].items()
                if d == info["item_drug"][name] and p not in used
            )
            used.add(pocket)
            target = layout.targets[pocket]
            position = np.array(
                [*target.position[:2], target.support_z + layout.shapes[name].height / 2]
            )
            solution[name] = ObjectState(
                position, np.array([1.0, 0.0, 0.0, 0.0]), fixture_contacts=frozenset([pocket])
            )
    elif tid == "C4":
        used = set()
        for name in task.tube_class:
            well = next(
                w
                for w, k in task.well_class.items()
                if k == task.tube_class[name] and w not in used
            )
            used.add(well)
            target = layout.targets[well]
            bottom = np.array([*target.position[:2], target.support_z + 0.001])
            solution[name] = ObjectState(
                bottom + np.array([0.0, 0.0, layout.shapes[name].height / 2]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
    else:
        raise ValueError(tid)
    return solution


def do_jobs(task, times, solution=None, dt=0.01, dwell=0.3):
    """Do the solution's jobs one by one, the k-th starting at times[k]; earlier ones stay put."""
    solution = solution or ward_solution(task)
    names = list(solution)
    for k, start in enumerate(times):
        task.time = start
        state = snapshot(task, **{n: solution[n] for n in names[: k + 1]})
        observe_for(task, state, dwell, dt)
    return snapshot(task, **{n: solution[n] for n in names[: len(times)]})
