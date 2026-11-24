"""V11 evaluator factory."""
from .lift import Lift
from .pool import Clearance


def Task(b):
    if b.config["workflow"].get("bench_press"):
        from roboclock.envs.sports.bench_press.task import BenchPress
        return BenchPress(b)
    if b.kind == "C1" and b.config.get("pool", {}).get("spot_clearance"):
        from .pool_spot import SpotClearance
        return SpotClearance(b)
    return (Lift if b.kind == "B3" else Clearance)(b)
