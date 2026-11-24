"""Visible score feedback from observed events, without future target disclosure."""


def round_status(task):
    if not task.terminated:
        return "PLAYING"
    return "COMPLETE" if task.metrics()["joint_success"] else "CHECK RESULT"


def mole_score(task):
    # Show only earned hits and observed mistakes, not future target counts,
    # future holes, or the identity of the currently rising mole.
    failures = task.metrics().get("physical_failures", task.physical_failures)
    mistakes = len(task.violations) + len(failures)
    return round_status(task), f"HITS {len(task.correct):02d} / ERRORS {mistakes:02d}"


def taiko_instruction(task):
    if task.terminated:
        return f"{round_status(task)} {len(task.correct)}/{len(task.events)}"
    if task.time < 10:
        return "PICK UP BOTH STICKS"
    upcoming = [
        e
        for i, e in enumerate(task.events)
        if i not in task.correct and e["time"] + task.limits["note_window"] >= task.time
    ]
    if not upcoming:
        return "FINISHING"
    next_time = upcoming[0]["time"]
    lanes = {e["control"] for e in upcoming if abs(e["time"] - next_time) < 1e-8}
    return "BOTH" if len(lanes) == 2 else "ROBOT LEFT" if "pad_0" in lanes else "ROBOT RIGHT"
