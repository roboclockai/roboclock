"""Native X5 driver contract; real PhysX checks are recorded separately."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.simulators.isaac.embodiments.config import ASSIGNMENTS, configure
from roboclock.simulators.isaac.embodiments.rig import RobotRig


def test_x5_fingers_respect_native_limits_and_inactive_joints():
    rig = object.__new__(RobotRig)
    rig.model = "dual_arx_x5"
    rig.command = np.arange(6, dtype=float)
    rig.grip_ids = {"right": [1, 4]}
    rig.robot = SimpleNamespace(get_joint_positions=lambda: rig.command)
    rig.grip("right", 0.02)
    np.testing.assert_allclose(rig.command, [0, 0.02, 2, 3, 0.02, 5])
    assert np.isclose(rig.width("right"), 0.04)
    rig.grip("right", -1)
    assert np.isclose(rig.width("right"), 0.0)
    rig.grip("right", 1)
    assert np.isclose(rig.width("right"), 0.088)


def test_default_native_tasks_use_identical_x5_platform():
    for task in ASSIGNMENTS:
        c = {"visuals": {}, "control": {}, "v2": {}, "workflow": {}}
        if task == "B4":
            c["workflow"]["fixed_freeze"] = True
        if task == "C3":
            c["workflow"]["fixed_service"] = True
        configure(c, task)
        assert c["embodiment"]["model"] == "dual_arx_x5"
        assert c["embodiment"]["fixed_base"]
    for other in ("mobile_fr3_duo_fixed", "galaxea_r1_fixed", "fr3_single_fixed"):
        c = {"visuals": {}, "control": {}, "v2": {}, "workflow": {}}
        with pytest.raises(ValueError, match="only dual_arx_x5"):
            configure(c, "A2", other)


def test_x5_glove_contacts_rotate_with_the_actual_mounted_tool():
    from scipy.spatial.transform import Rotation

    from roboclock.simulators.isaac.embodiments.tools import glove_contact_mask

    rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    native = np.array([[0.174, 0, 0], [0.235, 0.01, 0], [0, 0, 0.174], [0.4, 0, 0]])
    assert glove_contact_mask(native, rotation).tolist() == [True, True, False, False]
    canonical = native @ rotation
    np.testing.assert_array_equal(
        glove_contact_mask(canonical), glove_contact_mask(native, rotation)
    )


def test_presentation_lighting_does_not_accumulate_per_frame():
    import pytest

    pytest.importorskip("pxr.UsdLux")
    from pxr import Usd, UsdLux

    from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_lighting

    stage = Usd.Stage.CreateInMemory()
    light = UsdLux.RectLight.Define(stage, "/Fill")
    light.CreateIntensityAttr(100.0)
    backend = SimpleNamespace(stage=stage)
    for _ in range(4):
        apply_presentation_lighting(backend, 1.3)
    assert np.isclose(light.GetIntensityAttr().Get(), 130)
    apply_presentation_lighting(backend, 1.5)
    assert np.isclose(light.GetIntensityAttr().Get(), 150)
    # A backend reset creates a fresh stage; its lights also need the profile.
    backend.stage = Usd.Stage.CreateInMemory()
    reset_light = UsdLux.RectLight.Define(backend.stage, "/Fill")
    reset_light.CreateIntensityAttr(200.0)
    apply_presentation_lighting(backend, 1.5)
    assert np.isclose(reset_light.GetIntensityAttr().Get(), 300)
