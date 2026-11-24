"""Indicator colors and window arithmetic; scene commands are shared with the lab domain."""

from roboclock.envs.lab.mdp.events import BLUE, GREEN, RED, SceneCommand, window_error  # noqa: F401

AMBER = (1.0, 0.6, 0.05)


def timer_color(fraction: float) -> str:
    """Countdown bar color for the fraction of the time that is left."""
    return "green" if fraction > 0.4 else "amber" if fraction > 0.2 else "red"


def window_color(time: float, start: float, end: float):
    """Blue before the window, green inside (inclusive), red after."""
    return BLUE if time < start else GREEN if time <= end else RED
