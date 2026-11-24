"""Hash V11 implementation together with its shared dependencies."""
import hashlib
from pathlib import Path
from roboclock.envs.sports._shared.workflow.provenance import source_hashes as previous_hashes


def source_hashes():
    result = previous_hashes()
    result.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path("src/roboclock/envs/sports/_shared/precision").glob("*.py")})
    result.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path("src/roboclock/envs/sports/bench_press").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts})
    return result
