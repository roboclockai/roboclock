"""Per-sub-package runtime adapters of the unified sports environment (lazy imports)."""

ADAPTERS = {
    "A1": "roboclock.envs.sports.adapters.chess:ChessAdapter",
    "B5": "roboclock.envs.sports.adapters.arcade:ArcadeAdapter",
    "B1": "roboclock.envs.sports.adapters.arcade:ArcadeAdapter",
    "A2": "roboclock.envs.sports.adapters.workflow:ArenaAdapter",
    "B2": "roboclock.envs.sports.adapters.workflow:ArenaAdapter",
    "C2": "roboclock.envs.sports.adapters.workflow:ArenaAdapter",
    "B3": "roboclock.envs.sports.adapters.workflow:BenchAdapter",
    "B4": "roboclock.envs.sports.adapters.workflow:FreezeAdapter",
    "C1": "roboclock.envs.sports.adapters.workflow:RackAdapter",
    "C3": "roboclock.envs.sports.adapters.workflow:ServiceAdapter",
}


def adapter_class(task_id):
    from roboclock.envs.registry import import_symbol

    return import_symbol(ADAPTERS[task_id])
