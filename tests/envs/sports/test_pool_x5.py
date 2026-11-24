"""Regression checks for native pool planning; physical evidence is separate."""
from types import SimpleNamespace

import numpy as np

from roboclock.envs.rates import PHYSICS_HZ
from roboclock.envs.sports._shared.precision.pool_x5 import X5ClearanceOracle


class PathKinematics:
    def __init__(self, unreachable_midpoint=False):
        self.last = np.zeros(6)
        self.target = None
        self.unreachable_midpoint = unreachable_midpoint

    def solve(self, position, rotation, seed=None):
        self.target = np.asarray(position).copy()
        self.last = np.ones(6)
        return self.last, 0.

    def fk(self, q):
        position = self.target.copy()
        if self.unreachable_midpoint and .025 < position[0] < .05:
            position[2] += .02
        return position, np.eye(3)


def backend(kin=None):
    kin = kin or PathKinematics()
    return SimpleNamespace(time=1., active_side="right", config={},
        rig=SimpleNamespace(kin={"right": kin}, arm_ids={"right": list(range(6))}, targets={}),
        robot=SimpleNamespace(get_joint_positions=lambda: np.zeros(6)))


def test_stroke_checks_interior_and_preserves_ik_solver_state():
    kin = PathKinematics(unreachable_midpoint=True)
    oracle = X5ClearanceOracle(backend(kin), SimpleNamespace())
    assert not oracle.path_feasible(np.zeros(3), np.array([.075, 0, 0]), np.eye(3))
    assert len(oracle.path_checks[0]["errors"]) == 7
    np.testing.assert_array_equal(kin.last, np.zeros(6))


def test_continuous_reachable_stroke_is_accepted():
    oracle = X5ClearanceOracle(backend(), SimpleNamespace())
    assert oracle.path_feasible(np.zeros(3), np.array([.075, 0, 0]), np.eye(3))


def test_stroke_timeout_does_not_fabricate_contact():
    b = backend()
    b.time = 2.1
    b.config["sim"] = {"physics_dt": 1 / PHYSICS_HZ, "control_decimation": 5}
    b.last_state = {"objects": {"cue": {"forces": [1., 1.]}}}
    b.move = lambda *a, **kw: 0.
    task = SimpleNamespace(shots=[])
    oracle = X5ClearanceOracle(b, task)
    oracle.phase = "stroke"
    oracle.start = 0.
    oracle.stroke_origin = oracle.stroke_tip_origin = oracle.previous_stroke_tip = np.zeros(3)
    oracle.stroke_end = np.array([.075, 0, 0])
    oracle.direction = np.array([1., 0, 0])
    oracle.measured_tool = lambda: (np.zeros(3), np.eye(3), np.zeros(3), np.eye(3))
    oracle.act()
    assert oracle.phase == "retract"
    assert task.shots == []


def test_native_command_ramp_remains_bounded_without_resetting_to_actual():
    from roboclock.envs.sports._shared.precision.pool_x5 import move_cue_arm

    commands = np.full(6, .1)
    rig = SimpleNamespace(
        pose=lambda side: (np.zeros(3), np.eye(3)), targets={},
        robot=SimpleNamespace(get_joint_positions=lambda: np.zeros(6)),
        arm_ids={"right": list(range(6))}, command=commands,
        kin={"right": SimpleNamespace(solve=lambda *a, **kw: (np.full(6, .5), 0.))},
        grip=lambda *args: None,
    )
    move_cue_arm(rig, "right", np.array([.1, 0, 0]), np.eye(3), 0., .01, joint_speed=2.)
    # 0.1 + 2 rad/s * 0.01 s, not actual(0) + 0.02.
    np.testing.assert_allclose(rig.command, .12)


def test_planner_prefers_clear_lane_over_nearest_blocked_combination():
    b = backend()
    b.pool_targets = ["target1", "target2", "target3"]
    b.pockets = [np.array([x, y]) for x in [1.25, .15] for y in [-.12, .22]]
    b.ball_radius = .023
    b.last_state = {"hands": {"main": {"position": [.34, -.28, .95],
        "orientation": [0, 0, 1, 0]}}, "objects": {
        "cue": {"position": [.34, -.28, .905], "orientation": [1, 0, 0, 0]},
        "cue_ball": {"position": [.56, -.12, .773]},
        "target1": {"position": [.78, -.12, .773]},
        "target2": {"position": [1.025, -.12, .773]},
        "target3": {"position": [1.065, .10, .773]},
    }}
    oracle = X5ClearanceOracle(b, SimpleNamespace(pots={}, shots=[]))
    oracle.path_feasible = lambda *args: True
    assert oracle.plan()
    assert oracle.object == "target3"
    # Once the blocking ball is potted, the nearer straight route is preferred.
    oracle.task.pots["target2"] = 1.
    assert oracle.plan()
    assert oracle.object == "target1"


def test_elevated_ready_pose_clears_cushion_that_blocks_flat_cue():
    from scipy.spatial.transform import Rotation

    from roboclock.envs.sports._shared.precision.pool_x5 import cue_clears_jaws

    def geometry(pitch):
        r = Rotation.from_euler("z", 20.45, degrees=True) * Rotation.from_euler("y", pitch, degrees=True)
        direction = r.apply([1, 0, 0])
        tip = np.array([.56, -.12, .773])
        tip[:2] -= .103 * direction[:2] / np.linalg.norm(direction[:2])
        tip[2] -= .08 * direction[2] / np.linalg.norm(direction[:2])
        return tip, direction

    assert not cue_clears_jaws(*geometry(5))
    assert cue_clears_jaws(*geometry(10))


def test_tilted_stroke_reaches_ball_instead_of_stopping_at_planar_gap():
    from scipy.spatial.transform import Rotation

    from roboclock.envs.sports._shared.precision.pool_x5 import cue_stroke_travel

    radius, gap = .023, .040
    for pitch in (10., 26., 45., 55.):
        direction = Rotation.from_euler('y', pitch, degrees=True).apply([1., 0., 0.])
        travel = cue_stroke_travel(radius, gap, direction)
        # The ball must cross the flat front-cap contact plane before stroke end.
        separation = (radius + gap) / np.linalg.norm(direction[:2])
        assert separation - travel < radius
        assert separation - travel > 0.
        assert np.isclose((separation - travel) - radius, -.015)
