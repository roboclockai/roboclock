"""Capture task and shared rendering code used to produce an episode."""

import hashlib
from pathlib import Path


def source_hashes():
    paths = list(Path("src/roboclock/envs/sports/arena").rglob("*.py"))
    paths += list(Path("src/roboclock/simulators/isaac").rglob("*.py"))
    paths += list(Path("src/roboclock/simulators/isaac/embodiments").glob("*.json"))
    paths += list(Path("src/roboclock/envs/sports/arcade").rglob("*.py"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
