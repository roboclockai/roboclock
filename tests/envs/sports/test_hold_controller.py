"""Regression for independently preloaded jaws during red-light holding."""

from types import SimpleNamespace

import numpy as np

from roboclock.envs.sports.freeze.oracle import Oracle


def test_red_hold_preserves_each_jaw_target_and_does_not_reclose_symmetrically():
    b = SimpleNamespace(
        config={"freeze": {}},
        command=np.zeros(8),
        rig=SimpleNamespace(
            arm_ids={"left": [0, 1], "right": [2, 3]}, grip_ids={"left": [4, 5], "right": [6, 7]}
        ),
        last_state={"joints": np.array([0.1, 0.2, 0.3, 0.4, 0.026, 0.011, 0.024, 0.013])},
    )
    t = SimpleNamespace(red=lambda: 0, done=False, round=0)
    o = Oracle(b, t)
    for side, targets in (("left", [0.0245, 0.0095]), ("right", [0.0225, 0.0115])):
        o.arms[side].update(phase="park", grip_targets=np.array(targets))
    o.act()
    assert np.allclose(b.command, [0.1, 0.2, 0.3, 0.4, 0.0245, 0.0095, 0.0225, 0.0115])
    # Measured motion must not make the hold controller chase a drifting object.
    b.last_state["joints"] += 0.001
    o.act()
    assert np.allclose(b.command, [0.1, 0.2, 0.3, 0.4, 0.0245, 0.0095, 0.0225, 0.0115])
