"""Explicitly privileged V11 feedback baseline factory."""
from .lift import LiftOracle
from .pool import ClearanceOracle


def Oracle(b, t):
    if b.config["workflow"].get("bench_press"):
        from roboclock.envs.sports.bench_press.oracle import BenchOracle
        return BenchOracle(b, t)
    if b.kind == "C1" and b.config.get("pool", {}).get("spot_clearance"):
        from .pool_spot import SpotOracle
        return SpotOracle(b, t)
    if b.kind == "C1" and b.config.get("embodiment", {}).get("model") == "dual_arx_x5":
        from .pool_x5 import X5ClearanceOracle
        return X5ClearanceOracle(b, t)
    return (LiftOracle if b.kind == "B3" else ClearanceOracle)(b, t)
