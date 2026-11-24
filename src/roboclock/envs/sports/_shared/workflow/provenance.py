"""Hash the task implementation and imported simulator helpers."""

import hashlib
from pathlib import Path

from roboclock.envs.sports.arena.provenance import source_hashes as previous_hashes


def source_hashes():
    result = previous_hashes()
    paths = list(Path("src/roboclock/envs/sports/_shared/workflow").rglob("*.py"))
    paths += list(Path("src/roboclock/envs/sports/common").glob("*.py"))
    paths += list(Path("src/roboclock/physics").glob("*.py"))
    paths += [Path("src/roboclock/envs/lab/appearance.py")]
    result.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return result
