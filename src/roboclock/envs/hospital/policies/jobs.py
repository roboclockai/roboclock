"""Which item to move next, where it goes, and when grasping/releasing is allowed.

Privileged diagnostic logic: it reads task state and the layout. It is not part of the agent
observation contract and not a validated expert.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Job:
    pick: np.ndarray  # tool-center target at the grasp
    destination: np.ndarray  # object center when released
    yaw: float  # object yaw wanted at release
    hover: np.ndarray | None = None  # where to wait before the grasp is allowed
    grasp_ok: bool = True
    transfer_ok: bool = True
    release_ok: bool = True
    fast: bool = False  # carry quickly after the grasp
    rush: bool = False  # also approach and grasp quickly (deadline runs from an event)
    path: list = field(default_factory=list)  # object-center waypoints for a threading insertion


def object_yaw(q) -> float:
    """Yaw modulo pi: the gripper is symmetric, so pick the smaller rotation."""
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    return (yaw + np.pi / 2) % np.pi - np.pi / 2


def grid_offsets(columns, rows, pitch_x, pitch_y):
    return [
        ((c - (columns - 1) / 2) * pitch_x, (r - (rows - 1) / 2) * pitch_y)
        for r in range(rows)
        for c in range(columns)
    ]


# Slot grids inside each container (x, y offsets from its center), sized to the item footprints.
PHARMACY_SLOTS = grid_offsets(3, 4, 0.05, 0.05)
CART_SLOTS = grid_offsets(3, 4, 0.066, 0.066)


class HospitalJobs:
    def __init__(self):
        self.reset()

    def reset(self):
        self.assigned = {}  # item -> (container, slot index)
        self.used = {}  # container -> slots handed out

    # -- helpers ----------------------------------------------------------------------------

    def release_height(self, task, name, container):
        target = task.layout.targets[container]
        rest = task.layout.shapes[name].rest_height
        if task.layout.objects[name].asset == "pill_vial":
            # Tall and narrow: lower it into the tray and set it down instead of dropping it.
            return target.support_z + rest + 0.010
        return target.rim_z + rest + 0.010

    def pick_point(self, task, name, state):
        shape = task.layout.shapes[name]
        obj = state.objects[name]
        lift = (
            (0.3 if shape.height > 0.085 else 0.2) * shape.height
            if shape.kind == "cylinder"
            else 0.008
        )
        if shape.kind == "box" and 2 * shape.half[2] > 0.05:
            lift = 0.015
        if task.layout.objects[name].asset == "pill_vial":
            lift = 0.0  # grasp at mid-height, where a tall vial is most stable
        if task.layout.objects[name].asset in ("aed_pad", "aed_tab", "blanket_tab"):
            lift = 0.004  # thin parts: the fingertips reach the sides without touching the surface
        if task.layout.objects[name].asset == "blood_bag":
            lift = task.layout.info["bag_grasp_dz"]  # pinch the top bar of the hanging loop
        return obj.position + np.array([0.0, 0.0, lift])

    def slot(self, task, name, container, slots):
        if name not in self.assigned:
            index = self.used.get(container, 0)
            self.used[container] = index + 1
            self.assigned[name] = (container, index)
        container, index = self.assigned[name]
        return slots[index % len(slots)] if slots else (0.0, 0.0)

    def container_job(self, task, state, name, container, offset=(0.0, 0.0), **gates):
        target = task.layout.targets[container]
        destination = np.array(
            [
                target.position[0] + offset[0],
                target.position[1] + offset[1],
                self.release_height(task, name, container),
            ]
        )
        return Job(self.pick_point(task, name, state), destination, 0.0, **gates)

    # -- selection --------------------------------------------------------------------------

    def candidates(self, task, state, exclude=()):
        return [
            n
            for n in task.layout.objects
            if n not in task.correct and n not in task.dropped and n not in exclude
        ]

    def nearest(self, state, names):
        return (
            min(
                names,
                key=lambda n: (np.linalg.norm(state.objects[n].position - state.ee_position), n),
            )
            if names
            else None
        )

    def most_exposed(self, state, names, radius=0.10):
        """Pick the item with the fewest neighbors, so the hand has room to reach it."""
        if not names:
            return None

        def key(name):
            here = state.objects[name].position[:2]
            crowd = sum(
                np.linalg.norm(state.objects[other].position[:2] - here) < radius
                for other in names
                if other != name
            )
            return (crowd, np.linalg.norm(state.objects[name].position - state.ee_position), name)

        return min(names, key=key)

    def choose(self, task, state, exclude=()):
        tid = task.task_id
        remaining = self.candidates(task, state, exclude)
        if tid in ("A1", "A3"):
            prefix = "cup" if tid == "A1" else "instr"
            windows = {f"{prefix}_{i}": w for i, w in enumerate(task.spec.windows)}
            remaining = [n for n in remaining if n in windows]  # decoys are never moved
            # Among the requests that have arrived, the earliest deadline first; otherwise wait for
            # the next arrival.
            arrived = [n for n in remaining if windows[n][0] <= task.time]
            if arrived:
                return min(arrived, key=lambda n: windows[n][1])
            return min(remaining, key=lambda n: windows[n][0]) if remaining else None
        if tid == "A4":
            if task.dose_prep:
                # A syringe on the dock is left alone until it has filled; the dock holds one at a time.
                waiting = [
                    n
                    for n in task.layout.objects
                    if n.startswith("syringe_") and n not in task.drawn and task.docked(n, state)
                ]
                if waiting:
                    return waiting[0] if waiting[0] in remaining else None
            # Earliest beyond-use time first: syringe k is the k-th to expire.
            return min(remaining, key=lambda n: int(n.split("_")[-1])) if remaining else None
        if tid == "A2":
            return remaining[0] if remaining else None
        if tid == "B1":
            return None  # nothing to carry: the oracle presses the alarming pump's button
        if tid == "B2":
            # Free the pads first, then pad 1 and pad 2; the shock button is pressed after that. The pads are
            # welded into the unit until the tab registers as pulled, so nothing else is touched before that.
            if task.tab_time is None:
                return task.tab if task.tab in remaining else None
            for name in (task.tab, *task.pads):
                if name in remaining:
                    return name
            return None
        if getattr(task, "name", "") == "hazardous_drug_spill":
            from roboclock.envs.hospital import spill_scene

            return spill_scene.choose(self, task, state, remaining)
        if tid == "B3":
            # Free the blanket by pulling both tabs, then smother the fire with it.
            if task.alarm_time is None:
                return None  # raise the alarm first (the oracle presses the button once the fire has started)
            for name in (*task.tabs, task.blanket):
                if name in remaining and (name != task.blanket or task.blanket_free):
                    return name
            return None
        if tid == "C1":
            usable = [n for n in remaining if self.container_for(task, n) is not None]
            return self.nearest(state, usable)
        if tid == "C2":
            return self.most_exposed(state, remaining)
        if tid == "C3":
            return self.nearest(state, remaining)
        return None

    def container_for(self, task, name):
        tid, layout = task.task_id, task.layout
        if name in self.assigned:
            return self.assigned[name][0]
        if tid == "C2":
            return next(
                c for c, kind in layout.bin_types.items() if kind == layout.item_types[name]
            )
        if tid == "C3":
            return "basket"
        if tid == "C1":
            kind = layout.item_types[name]
            pending = {}
            for item, (container, _) in self.assigned.items():
                if item not in task.correct:
                    key = (container, layout.item_types[item])
                    pending[key] = pending.get(key, 0) + 1
            open_bins = [
                (task.remaining[c].get(kind, 0) - pending.get((c, kind), 0), c)
                for c in layout.targets
            ]
            best = max(open_bins)
            return best[1] if best[0] > 0 else None
        return layout.assignments.get(name)

    # -- description ------------------------------------------------------------------------

    def describe(self, task, state, name) -> Job:
        tid, layout, now = task.task_id, task.layout, task.time
        if tid in ("A1", "A3"):
            index = int(name.split("_")[-1])
            start = task.spec.windows[index][0]
            container = layout.assignments[name]
            # Instruments sit side by side in the Mayo tray, one lane per step.
            offset = (0.0, (index - 1) * 0.055) if tid == "A3" else (0.0, 0.0)
            return self.container_job(
                task,
                state,
                name,
                container,
                offset,
                grasp_ok=now >= start,  # the request (or the surgeon's call) has to arrive first
                transfer_ok=now >= start - 0.12,
                release_ok=now >= start - 0.10,
            )
        if tid == "A2":
            # Thread the bag's eyelet onto the peg from its free end: line up with the peg first,
            # then slide the bag along it. The peg passes through the middle of the eyelet's hole.
            hook = layout.info["hook"]
            height = hook["peg_z"] - 0.092
            start = np.array([hook["x"], hook["tip_y"] - 0.08, height])
            end = np.array([hook["x"], hook["tip_y"] + 0.07, height])
            return Job(self.pick_point(task, name, state), end, 0.0, path=[start, end])
        if tid == "A4":
            if task.dose_prep and name not in task.drawn:
                # Empty syringe: set it on the filling dock (and, once there, wait for the fill before touching it).
                prep = layout.info["dose_prep"]
                z = prep["support_z"] + layout.shapes[name].rest_height + 0.004
                job = Job(
                    self.pick_point(task, name, state),
                    np.array([*prep["dock"], z]),
                    0.0,
                    grasp_ok=not task.docked(name, state),
                    fast=True,
                )
                job.hover = self.pick_point(task, name, state) + [0.0, 0.0, 0.12]
                return job
            return self.container_job(task, state, name, layout.assignments[name], fast=True)
        if tid == "B1":
            job = self.container_job(
                task,
                state,
                name,
                layout.assignments[name],
                grasp_ok=task.alarm_for(int(name.split("_")[-1]))[1] <= now,
                fast=True,
                rush=True,
            )
            job.hover = self.pick_point(task, name, state) + [0.0, 0.0, 0.12]
            return job
        if tid == "B2":
            event = task.event_time is not None
            shape = layout.shapes[name]
            if name == task.tab:
                # Pull the tab off the AED toward the robot and set it on the bench.
                home = layout.objects[name].position
                pull = np.array(layout.info["aed"]["tab_pull"])
                support = layout.info["aed"].get("support_z", layout.table_height)  # the AED's table
                z = support + shape.rest_height + 0.006
                destination = np.array([*(home[:2] + pull), z])
            else:
                k = task.pads.index(name)
                mark = layout.zones[f"mark_{k}"]
                z = mark.support_z + shape.rest_height + 0.008
                destination = np.array([*mark.center, z])
            job = Job(
                self.pick_point(task, name, state),
                destination,
                0.0,
                grasp_ok=event and (name == task.tab or task.tab_time is not None),
                fast=True,
                rush=True,
            )
            job.hover = self.pick_point(task, name, state) + [0.0, 0.0, 0.10]
            # Before the arrest the planner holds the start pose (HospitalTrajectory.awaiting_cue): it
            # no longer moves to a waiting pose near the AED (layout.info["aed"]["ready"] is unused).
            return job
        if getattr(task, "name", "") == "hazardous_drug_spill":
            from roboclock.envs.hospital import spill_scene

            return spill_scene.describe(self, task, state, name)
        if tid == "B3":
            shape = layout.shapes[name]
            event = task.event_time is not None
            if name == task.blanket:
                z = task.dish_top + shape.rest_height + 0.012
                destination = np.array([*task.fire_center, z])
            else:
                home = layout.objects[name].position
                pull = np.array(layout.info["fire"]["tab_pull"])
                z = layout.table_height + shape.rest_height + 0.006
                destination = np.array([*(home[:2] + pull), z])
            job = Job(
                self.pick_point(task, name, state),
                destination,
                0.0,
                grasp_ok=event,
                fast=True,
                rush=True,
            )
            job.hover = self.pick_point(task, name, state) + [0.0, 0.0, 0.10]
            return job
        if tid in ("C1", "C2"):
            container = self.container_for(task, name)
            slots = PHARMACY_SLOTS if tid == "C1" else CART_SLOTS
            offset = self.slot(task, name, container, slots)
            return self.container_job(task, state, name, container, offset)
        if tid == "C3":
            container, index = self.assigned.get(name, (None, None))
            if container is None:
                self.slot(task, name, "basket", [])
                container, index = self.assigned[name]
            row, layer = index % 8, index // 8
            job = self.container_job(task, state, name, container, ((0.0), (row - 3.5) * 0.028))
            job.destination[2] += 0.022 * layer
            return job
        raise ValueError(f"Unknown task {tid}")
