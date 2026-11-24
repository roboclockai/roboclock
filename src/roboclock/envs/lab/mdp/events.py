"""Symbolic laboratory process updates and commands for physical dynamics."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SceneCommand:
    kind: str
    name: str
    value: Any


def window_error(time: float, start: float, end: float) -> float:
    return max(start - time, time - end, 0.0)


BLUE = (0.1, 0.3, 1.0)
GREEN = (0.1, 1.0, 0.2)
RED = (1.0, 0.1, 0.1)
