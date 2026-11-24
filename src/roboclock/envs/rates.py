"""Time base shared by every domain; no simulator or domain imports.

Physics runs at exactly PHYSICS_HZ in every task (docs/robodojo-rig-spec.md; multiples
are not used), and environments step at CONTROL_HZ, a whole number of physics ticks. The
oracles that produce the ground-truth demonstrations act at CONTROL_HZ; a policy may
act at CONTROL_HZ or POLICY_HZ (``roboclock eval --policy-hz``, see the evaluator).
Recordings run at a divisor of CONTROL_HZ.
"""

import math

PHYSICS_HZ = 250
CONTROL_HZ = 50
POLICY_HZ = 25
VIDEO_FPS = 25

CONTROL_DT = 1 / CONTROL_HZ


def check_rates(physics_dt, control_decimation, *, where="sim"):
    """Reject a physics step or decimation that leaves the shared time base."""
    physics_hz = 1 / physics_dt
    if not math.isclose(physics_hz, PHYSICS_HZ):
        raise ValueError(f"{where}: physics must run at exactly {PHYSICS_HZ} Hz (physics_dt 1/{PHYSICS_HZ} s), got {physics_hz:g} Hz")
    if not math.isclose(physics_hz / control_decimation, CONTROL_HZ):
        raise ValueError(f"{where}: control must run at {CONTROL_HZ} Hz")
