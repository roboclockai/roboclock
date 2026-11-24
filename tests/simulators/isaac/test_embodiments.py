"""Native replacement contracts; actual physics checks live in recorded probes."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roboclock.simulators.isaac.embodiments.config import ASSIGNMENTS, configure
from roboclock.simulators.isaac.embodiments.rig import RobotRig


def test_mobile_tasks_are_excluded_before_mutation():
    for task in ("B4",):
        config = {"keep": 1}
        with pytest.raises(ValueError, match="fixed-freeze"):
            configure(config, task)
        assert config == {"keep": 1}
    assert set(ASSIGNMENTS) == {"A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1", "C2", "C3"}


def test_tool_pose_rotates_grasp_frame_without_moving_tcp():
    rig = object.__new__(RobotRig)
    rig.tools = {
        "right": SimpleNamespace(
            get_world_pose=lambda: (np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
    }
    rig.offsets = {"right": np.array([0, 0, 0.145])}
    rig.tcp_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    p, q = rig.pose("right")
    np.testing.assert_allclose(p, [1, 2, 3.145])
    np.testing.assert_allclose(
        Rotation.from_quat(np.roll(q, -1)).as_matrix(), rig.tcp_rotation, atol=1e-12
    )


def test_fk_uses_canonical_tcp_frame_and_ik_recovers_target():
    from roboclock.simulators.isaac.embodiments.kinematics import USDKinematics

    kin = object.__new__(USDKinematics)
    kin.base = np.eye(4)
    kin.frames = [(np.eye(4), np.eye(4))]
    kin.axes = [np.array([0.0, 0.0, 1.0])]
    kin.tail = np.eye(4)
    kin.offset = np.array([0.4, 0, 0.1])
    kin.tcp_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    kin.low, kin.high, kin.last = np.array([-3.0]), np.array([3.0]), np.array([0.0])
    p, r = kin.fk([np.pi / 2])
    np.testing.assert_allclose(p, [0, 0.4, 0.1], atol=1e-12)
    np.testing.assert_allclose(
        r, Rotation.from_euler("z", 180, degrees=True).as_matrix(), atol=1e-12
    )
    q, error = kin.solve(p, r)
    assert error < 1e-5
    np.testing.assert_allclose(q, [np.pi / 2], atol=1e-5)


def test_analytic_ik_jacobian_matches_multiaxis_finite_differences():
    from roboclock.simulators.isaac.embodiments.kinematics import USDKinematics

    kin = object.__new__(USDKinematics)
    kin.base, kin.tail = np.eye(4), np.eye(4)
    kin.offset = np.array([0.1, 0.03, 0.05])
    kin.tcp_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    kin.axes, kin.frames = np.eye(3), []
    for _ in range(3):
        a = np.eye(4)
        a[:3, 3] = [0.1, 0.02, 0.2]
        kin.frames.append((a, np.eye(4)))
    rng = np.random.default_rng(28)
    for _ in range(12):
        q, position = rng.uniform(-1, 1, 3), rng.normal(size=3)
        rotation = Rotation.random(random_state=rng).as_matrix()
        error, jac = kin.error_and_jacobian(q, position, rotation)
        finite = np.column_stack(
            [
                (kin.error_and_jacobian(q + np.eye(3)[i] * 1e-7, position, rotation)[0] - error)
                / 1e-7
                for i in range(3)
            ]
        )
        np.testing.assert_allclose(jac, finite, atol=2e-6)


def test_active_arm_update_holds_parked_arm_and_grippers():
    rig = object.__new__(RobotRig)
    rig.command = np.arange(8, dtype=float)
    rig.arm_ids = {"left": [0, 2, 4], "right": [1, 3, 5]}
    rig.set_arm_command("right", np.full(8, -0.2))
    np.testing.assert_allclose(rig.command, [0, -0.2, 2, -0.2, 4, -0.2, 6, 7])


def test_floating_base_gravity_wrench_is_not_applied_to_arm_joints():
    from roboclock.simulators.isaac.embodiments.rig import joint_gravity_forces

    torques = np.arange(26, dtype=float)
    generalized = np.r_[0, 0, 3500, -50, 20, 0, torques]
    np.testing.assert_array_equal(joint_gravity_forces(generalized, 26), torques)
    np.testing.assert_array_equal(joint_gravity_forces(torques, 26), torques)
    with pytest.raises(ValueError):
        joint_gravity_forces(np.zeros(25), 26)


def test_isaac6_does_not_disable_reconstruction_for_legacy_fxaa(monkeypatch):
    import sys

    from roboclock.simulators.isaac import rendering, version

    class Settings(dict):
        def set(self, key, value):
            self[key] = value

    settings = Settings()
    monkeypatch.setitem(
        sys.modules,
        "carb",
        SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)),
    )
    monkeypatch.setattr(version, "installed_version", lambda: "6.2.0.0")
    rendering.configure_renderer({"visuals": {"anti_aliasing": 2}})
    assert settings["/rtx/post/aa/op"] == 4
    assert settings["/rtx-transient/dldenoiser/enabled"] is True
    monkeypatch.setattr(version, "installed_version", lambda: "5.1.0")
    rendering.configure_renderer({"visuals": {"anti_aliasing": 2}})
    assert settings["/rtx/post/aa/op"] == 2
