"""Optional RoboDojo start pose: both X5 arms with every arm joint at 0 rad.

RoboDojo episodes start with all six joints of both arms at 0 and the grippers open.
roboclock tasks start from their own postures instead: lab and hospital HOME, household
READY_Q, and per-task pre-positions (IK ready poses over a key, parked arms).
A policy trained on RoboDojo spends its first chunk pulling the arms back toward 0.

`robot.start_pose: zero` in a domain config, or `start_pose="zero"` for an env
(CLI `--start-pose zero`), replaces those postures at reset. Joints 1-6 of both arms
become 0, and so do their PD targets. Each gripper keeps its task's start width (RoboDojo
starts open; some tasks start closed on purpose, e.g. to press a button). The default
`task` keeps every task's own posture. Arms the environment drives itself, such as the
sports A1 opponent, keep theirs.

Oracles, recorded demonstrations and deadlines were authored for the task postures. With
`zero` they start from the folded pose, with the tool horizontal about 0.24 m ahead of and
0.16 m above each base. Some oracles then miss their deadlines or IK checks. So `zero` is
an evaluation setting for learned policies, not a change to the tasks.
"""

START_POSES = ("task", "zero")
ARM_DOF = 6


def start_pose(config) -> str:
    """The configured start pose, `task` unless `robot.start_pose` says otherwise."""
    value = ((config or {}).get("robot") or {}).get("start_pose", "task")
    if value not in START_POSES:
        raise ValueError(f"robot.start_pose must be one of {', '.join(START_POSES)}, got {value!r}")
    return value


def zero_start(config) -> bool:
    return start_pose(config) == "zero"


def set_start_pose(config, value):
    """Record an env's `start_pose` argument in its config (None keeps the config's value).

    The value lives in config["robot"], so run manifests that store the config record it.
    """
    if value is None:
        return config
    if value not in START_POSES:
        raise ValueError(f"start_pose must be one of {', '.join(START_POSES)}, got {value!r}")
    config.setdefault("robot", {})["start_pose"] = value
    return config
