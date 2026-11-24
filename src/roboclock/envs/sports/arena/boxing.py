"""Seeded boxing coach: one visible cue per strike, mixed combinations and holds."""

import numpy as np


def schedule(config, seed):
    rng = np.random.default_rng(seed)
    first = "right" if rng.integers(2) else "left"
    other = "left" if first == "right" else "right"
    rest = [
        ("DOUBLE JAB", [other, other]),
        ("THREE PUNCH", [first, other, first]),
        ("SINGLE", [other]),
        ("FEINT / HOLD", []),
        ("ONE TWO", [other, first]),
    ]
    rng.shuffle(rest)
    now = config["boxing_start"]
    commands = []
    for name, hands in [("SINGLE", [first]), *rest]:
        cues = [now]
        for _ in hands[1:]:
            cues.append(cues[-1] + float(rng.uniform(2.8, 3.5)))
        end = cues[-1] + (2.4 if not hands else 2.6)
        commands.append(dict(name=name, hands=hands, time=now, cues=cues if hands else [], end=end))
        now = end + float(rng.uniform(0.6, 1.5))
    return commands


def active_command(commands, time):
    return next((i for i, c in enumerate(commands) if c["time"] <= time < c["end"]), None)


def next_strike(task, time):
    index = active_command(task.commands, time)
    if index is None:
        return None
    command = task.commands[index]
    count = sum(h["command"] == index for h in task.hits)
    if count >= len(command["hands"]) or time < command["cues"][count]:
        return None
    return index, count, command["hands"][count], command["cues"][count]
