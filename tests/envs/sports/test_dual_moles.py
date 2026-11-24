"""Preserve event semantics when routing two physical mallets."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.sports.arcade.dual_moles import DualMoleOracle, _MoleTaskView
from roboclock.envs.sports.arcade.tasks import MalletMoles, configuration
from roboclock.simulators.isaac.embodiments.config import configure


def test_two_mallets_preserve_schedule_deadlines_and_hole_geometry():
    for seed in (0, 1, 7):
        original = configuration("B1")
        dual = configure(configuration("B1"), "B1", "dual_arx_x5")
        a, b = MalletMoles("easy", seed, original), MalletMoles("easy", seed, dual)
        assert a.events == b.events
        assert a.limits == b.limits
        assert set(b.layout.bodies) == {"mallet", "mallet_left"}
        for name in a.layout.controls:
            assert a.layout.controls[name] == b.layout.controls[name]
        assert dual["embodiment"]["active_arms"] == ["left", "right"]


@pytest.mark.parametrize(
    "index,visible,valid,expected",
    [
        (None, False, True, []),
        (0, False, True, []),
        (0, True, False, []),
        (0, True, True, [0]),
    ],
)
def test_router_waits_for_actual_rise_and_ignores_decoys(index, visible, valid, expected):
    task = SimpleNamespace(
        current_index=index, mole_up_times={0: 12.2} if visible else {}, events=[dict(valid=valid)]
    )
    router = object.__new__(DualMoleOracle)
    router.env = SimpleNamespace(
        task=task, backend=SimpleNamespace(rig=SimpleNamespace(command=np.zeros(16)))
    )
    router.assignments = {}
    routed = []
    router.route = routed.append
    router.arms = {
        side: SimpleNamespace(act=lambda obs: None, phase="wait", diagnostics={})
        for side in ("left", "right")
    }
    router.act({})
    assert routed == expected


def test_only_assigned_hand_sees_a_current_target():
    router = SimpleNamespace(
        env=SimpleNamespace(task=SimpleNamespace(current_index=4)), assignments={4: "left"}
    )
    assert _MoleTaskView(router, "left").current_index == 4
    assert _MoleTaskView(router, "right").current_index is None


@pytest.mark.parametrize(
    "active,phase,elapsed,expired",
    [
        (False, "aim", 0.1, True),
        (False, "strike", 0.1, True),
        (True, "aim", 1.7, True),
        (True, "aim", 0.5, False),
        (True, "strike", 1.7, False),
    ],
)
def test_unreachable_or_expired_target_cannot_stall_later_rounds(active, phase, elapsed, expired):
    from roboclock.envs.sports.arcade.oracle import mole_target_expired

    task = SimpleNamespace(time=20.0 + elapsed, mole_active=lambda index: active)
    assert mole_target_expired(task, 4, phase, 20.0, 1.6) is expired


@pytest.mark.parametrize(
    "force,held,exposed,travel,expected",
    [
        (2.0, True, True, 0.01, True),
        (2.0, False, True, 0.01, False),
        (2.0, True, False, 0.01, False),
        (0.0, True, True, 0.01, False),
        (0.05, True, True, 0.01, False),
        (2.0, True, True, 0.0019, False),
        (2.0, True, True, 0.002, True),
        (2.0, True, True, 0.035, True),
        (2.0, True, True, 0.0351, False),
    ],
)
def test_both_mallets_require_held_head_contact_and_valid_plunger_travel(
    force, held, exposed, travel, expected
):
    from roboclock.envs.sports.arcade.contacts import mole_head_activated

    assert mole_head_activated(force, held, exposed, travel) is expected
