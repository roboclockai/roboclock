"""Run the preserved legacy hospital planner against a kinematic stand-in for the robot and physics.

The stand-in grasps anything between the closed fingers, carries it, and drops it to the first
support below. It checks the planner's selection, gating and timing logic and the scoring chain
end to end. It does not test PhysX contact behavior, grasp stability or reachability.
"""

from dataclasses import replace

import numpy as np
import pytest

from roboclock.envs.hospital.config import load_legacy_config as load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.mdp.geometry import fits_xy, object_extents
from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory
from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState

PHYSICS_DT, DECIMATION = 1 / 120, 4


class KinematicWorld:
    def __init__(self, task):
        self.task, self.layout = task, task.layout
        self.pos = {n: o.position.copy() for n, o in self.layout.objects.items()}
        self.quat = {n: o.orientation.copy() for n, o in self.layout.objects.items()}
        self.floating = set(self.layout.held)
        self.held = None
        self.offset = np.zeros(3)
        self.ee = np.array([0.35, 0.0, 1.05])
        self.yaw = 0.0
        self.contact = {}
        self.shift = 0.0  # how far the drawer is pulled open (A2)
        self.dragging = False
        self.home = {n: o.position.copy() for n, o in self.layout.objects.items()}

    def bag_support(self, name):
        """A2: the bag rests on the drawer floor over its footprint, else on the bench."""
        home = self.home[name]
        drawer_y = home[1] + self.shift
        if abs(self.pos[name][0] - home[0]) < 0.14 and abs(self.pos[name][1] - drawer_y) < 0.14:
            return home[2], None
        return self.layout.table_height + 0.071, None

    def try_hang(self, name):
        """A2: released with the eyelet around the peg, the bag hangs from it."""
        info = self.layout.info.get("hook")
        if info is None or name != "bag_0":
            return False
        x, y, z = self.pos[name]
        if (
            abs(x - info["x"]) < 0.03
            and abs(z - (info["peg_z"] - 0.092)) < 0.03
            and info["tip_y"] - 0.02 < y < info["base_y"]
        ):
            self.pos[name] = np.array(
                [
                    info["x"],
                    min(max(y, info["tip_y"] + 0.03), info["base_y"] - 0.03),
                    info["peg_z"] - 0.108,
                ]
            )
            self.contact[name] = "fixture:hook_peg"
            return True
        return False

    def aed_surface(self, name):
        """B2/B3: pads, tabs and the blanket rest on the chest, the AED or the dish if they are over it, else on the bench."""
        x, y, z = self.pos[name]
        top = self.layout.table_height
        for fixture in self.layout.fixtures:
            if fixture.name in ("chest", "aed", "fire_dish", "dose_dock"):
                half = fixture.size / 2
                surface = fixture.position[2] + half[2]
                if (
                    abs(x - fixture.position[0]) < half[0]
                    and abs(y - fixture.position[1]) < half[1]
                ):
                    if surface <= z + 0.02:
                        top = max(top, surface)
        return top + self.layout.shapes[name].half[2] + 0.001, None

    def support(self, name):
        if self.layout.task_id in ("B2", "B3"):
            return self.aed_surface(name)
        if name == "bag_0" and "hook" in self.layout.info:
            return self.bag_support(name)
        shape, position = self.layout.shapes[name], self.pos[name]
        extent = object_extents(shape, self.quat[name])[2]
        obj = ObjectState(position, self.quat[name])
        for container, target in self.layout.targets.items():
            if fits_xy(shape, obj, target):
                return target.support_z + extent + 0.001, container
        return self.layout.table_height + extent + 0.001, None

    def state(self):
        objects = {}
        for name in self.pos:
            grasped = name == self.held
            contacts = frozenset([self.contact[name]]) if self.contact.get(name) else frozenset()
            objects[name] = ObjectState(
                self.pos[name].copy(),
                self.quat[name].copy(),
                grasped=grasped,
                fixture_contacts=contacts,
                relative_tool_speed=0.0,
            )
        q = np.array([0.0, -np.sin(self.yaw / 2), np.cos(self.yaw / 2), 0.0])
        return PhysicalState(objects, self.ee.copy(), self.held is None, True, q)

    def apply(self, command):
        self.ee, self.yaw = command.position.copy(), command.yaw
        handle = self.layout.info.get("handle")
        if handle is not None:
            at_handle = np.linalg.norm(self.ee - (handle + [0.0, self.shift, 0.0])) < 0.015
            self.dragging = bool(command.close and (self.dragging or at_handle))
            if self.dragging:
                limit = self.layout.info["drawer_travel"]
                self.shift = float(min(max(self.ee[1] - handle[1], 0.0), limit))
                if self.held is None and "bag_0" in self.pos and "bag_0" not in self.floating:
                    self.pos["bag_0"][1] = self.home["bag_0"][1] + self.shift
        if command.close and self.held is None and not self.dragging:
            for name, position in self.pos.items():
                if name in self.floating:
                    continue
                reach = 0.04 + (
                    self.layout.info.get("bag_grasp_dz", 0.0) if name == "bag_0" else 0.0
                )
                if (
                    np.linalg.norm(position[:2] - self.ee[:2]) < 0.015
                    and 0 <= self.ee[2] - position[2] < reach
                ):
                    self.held, self.offset = name, self.ee - position
                    self.contact[name] = None
                    break
        elif not command.close and self.held is not None:
            self.floating.discard(self.held)
            released, self.held = self.held, None
            if self.try_hang(released):
                self.floating.add(released)  # stays put: it hangs
        if self.held:
            self.pos[self.held] = self.ee - self.offset
        for name in self.pos:
            if name != self.held and name not in self.floating:
                z, container = self.support(name)
                if self.pos[name][2] > z:
                    self.pos[name][2] = z  # falls to the support below in one control tick
                self.contact[name] = container if abs(self.pos[name][2] - z) < 0.002 else None

    def command_scene(self, command):
        if command.kind == "release":
            self.floating.discard(command.name)


def run(task_id, difficulty="easy", seed=0, limit=None, dose_prep=False, **overrides):
    config = load_config()
    config["tasks"]["A4"]["dose_prep"] = dose_prep
    task = make_task(task_id, difficulty, seed, config)
    task.spec = replace(task.spec, **overrides)
    task.bind(make_layout(task, config))
    world = KinematicWorld(task)
    task.previous = world.state()
    planner = HospitalTrajectory(config["control"], PHYSICS_DT * DECIMATION)
    limit = limit or task.spec.max_time
    start, held_start = world.ee.copy(), []
    while task.time < limit and not task.terminated:
        command = planner.command(task, world.state())
        if command is None:  # B1-B3 before the cue: the oracle holds the start pose
            held_start.append(task.time)
            assert np.allclose(world.ee, start)
        else:
            world.apply(command)
        for _ in range(DECIMATION):
            for scene_command in task.advance(PHYSICS_DT):
                world.command_scene(scene_command)
            task.observe(world.state(), PHYSICS_DT)
    task.held_start = held_start  # times at which the planner asked to hold the start pose
    task.final_ee = world.ee.copy()
    return task


@pytest.mark.parametrize("task_id", ["A1", "A2", "A3", "A4", "B1", "B2", "B3"])
@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_planner_completes_single_shot_tasks(task_id, difficulty):
    task = run(task_id, difficulty, seed=0)
    assert task.physical_success, (task.physical_failures, task.event_log[-5:])
    assert task.temporal_success, (task.violations, task.event_log[-5:])


@pytest.mark.parametrize("task_id", ["C1", "C2", "C3"])
def test_planner_keeps_loading_items(task_id):
    task = run(task_id, "easy", seed=0)
    assert len(task.correct) >= 3, (len(task.correct), task.event_log[-5:])
    assert task.metrics()["accuracy"] == 1.0


def test_pharmacy_orders_are_fulfilled_exactly():
    task = run("C1", "easy", seed=2)
    assert task.metrics()["orders_fulfilled"] >= 1
    assert task.metrics()["accuracy"] == 1.0
    assert task.metrics()["drops"] == 0


@pytest.mark.parametrize("task_id", ["B1", "B2", "B3"])
def test_planner_waits_without_acting_before_event_for_reactive_tasks(task_id):
    task = run(task_id, "easy", seed=1, limit=3.0, event_time=9.0)
    assert task.event_time is None
    assert "pre_event_action" not in task.violations
    assert not task.correct
    # The hand has not left the start pose: every control step so far asked to hold it.
    assert len(task.held_start) >= round(3.0 / (PHYSICS_DT * DECIMATION))
    assert np.allclose(task.final_ee, [0.35, 0.0, 1.05])


@pytest.mark.parametrize("task_id", ["B1", "B2", "B3"])
def test_planner_leaves_the_start_pose_only_once_the_cue_is_on(task_id):
    task = run(task_id, "easy", seed=1, limit=6.0, event_time=4.0)
    assert task.event_time is not None
    # Held on every control step up to the event, and on none after it.
    assert task.held_start and max(task.held_start) < task.event_time
    assert len(task.held_start) >= round(4.0 / (PHYSICS_DT * DECIMATION)) - 1
    assert not np.allclose(task.final_ee, [0.35, 0.0, 1.05])


def test_b1_planner_does_not_go_to_a_pump_that_has_not_alarmed():
    from roboclock.envs.hospital.layout import make_layout as layout_for

    config = load_config()
    task = make_task("B1", "easy", 0, config)
    task.spec = replace(
        task.spec, event_time=1.0, alarms=(1.0, 9.0), alarm_limits=(14.0, 14.0), response_limit=14.0
    )
    task.bind(layout_for(task, config))
    world = KinematicWorld(task)
    task.previous = world.state()
    planner = HospitalTrajectory(config["control"], PHYSICS_DT * DECIMATION)
    first, second = (task.buttons[s] for s, _, _ in task.alarms())
    while task.time < 8.9:
        command = planner.command(task, world.state())
        if command is not None:
            world.apply(command)
        for _ in range(DECIMATION):
            task.advance(PHYSICS_DT)
            task.observe(world.state(), PHYSICS_DT)
    assert len(task.delivery_times) == 1 and task.button is None
    # Between the alarms the hand stays lifted over the key it pressed, not over the next pump.
    assert np.allclose(world.ee[:2], first[:2], atol=0.03)
    assert np.linalg.norm(world.ee[:2] - second[:2]) > 0.15
    assert world.ee[2] == pytest.approx(first[2] + 0.14, abs=0.01)


@pytest.mark.parametrize("task_id", ["C1", "C2"])
def test_every_slot_fits_every_item_type_in_its_bin(task_id):
    from roboclock.envs.hospital.mdp.geometry import fits_xy
    from roboclock.envs.hospital.policies.jobs import CART_SLOTS, PHARMACY_SLOTS

    config = load_config()
    slots = PHARMACY_SLOTS if task_id == "C1" else CART_SLOTS
    for seed in range(10):
        task = make_task(task_id, "easy", seed, config)
        layout = make_layout(task, config)
        for name, obj in layout.objects.items():
            for container, target in layout.targets.items():
                for dx, dy in slots:
                    moved = ObjectState(
                        np.array([target.position[0] + dx, target.position[1] + dy, 0.8]),
                        obj.orientation,
                    )
                    assert fits_xy(layout.shapes[name], moved, target), (name, container, dx, dy)


def test_bins_hold_the_largest_per_bin_demand():
    config = load_config()
    for task_id, capacity in (("C1", 12), ("C2", 12)):
        for seed in range(30):
            task = make_task(task_id, "easy", seed, config)
            layout = make_layout(task, config)
            if task_id == "C1":
                assert max(sum(o.values()) for o in layout.orders.values()) <= capacity
            else:
                counts = {}
                for kind in layout.item_types.values():
                    counts[kind] = counts.get(kind, 0) + 1
                assert max(counts.values()) <= capacity


def test_finger_ramp_closes_gently_and_opens_quickly():
    from roboclock.envs.hospital.policies.trajectories import (
        CLOSE_SPEED,
        OPEN_SPEED,
        ramp_finger,
    )

    dt = 1 / 30
    width = 0.04
    for _ in range(3):
        width = ramp_finger(width, 0.0, dt)
    assert width == pytest.approx(0.04 - 3 * CLOSE_SPEED * dt)
    assert ramp_finger(0.01, 0.04, dt) == pytest.approx(0.01 + OPEN_SPEED * dt)
    assert ramp_finger(0.02, 0.0199, dt) == pytest.approx(0.0199)  # never overshoots the target


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_planner_fills_every_dose_on_the_dock_before_delivering_it(difficulty):
    task = run("A4", difficulty, seed=0, dose_prep=True)
    assert task.physical_success, (task.physical_failures, task.event_log[-6:])
    assert task.temporal_success, (task.violations, task.placement_times, task.expiries)
    assert set(task.drawn) == {"syringe_0", "syringe_1", "syringe_2"}
    for name, placed in task.placement_times.items():
        assert (
            task.drawn[name] + 0.5 < placed
        )  # every dose was filled first, then carried to its bed
