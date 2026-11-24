"""Judging must not depend on which arm of the rig acts.

`PhysicalState.ee_position` is the first arm's tool point (the rig's right arm). Until 2026-11-10
the button presses of B1, B2, B3 and C2 read only that point, so a timely press made with the
other arm went unjudged (GPT 6 Astra, hospital B1 seed 0).
"""

from dataclasses import replace

import numpy as np
import pytest
from hospital_helpers import snapshot

from roboclock.envs.hospital.tasks.base import FINGERTIP_BELOW_TOOL, any_arm_pressing, pressing_button
from roboclock.envs.lab.mdp.state import PhysicalState

FAR = np.array([0.0, 0.0, 2.0])  # a tool point nowhere near a button


def on_button(button):
    """A tool point whose fingertips rest on the button's top."""
    return np.array([button[0], button[1], button[2] + FINGERTIP_BELOW_TOOL + 0.004])


def test_a_state_lists_both_tool_points():
    one = PhysicalState({}, np.array([0.1, 0.2, 0.3]))
    assert [p.tolist() for p in one.tool_points()] == [[0.1, 0.2, 0.3]]
    two = replace(one, secondary_ee_position=np.array([0.4, 0.5, 0.6]))
    assert [p.tolist() for p in two.tool_points()] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.parametrize("presser", ["first", "second"])
def test_either_arm_presses_a_button(presser):
    button = np.array([0.4, -0.1, 0.9])
    here = on_button(button)
    assert pressing_button(here, button)
    state = PhysicalState({}, here if presser == "first" else FAR,
                          secondary_ee_position=FAR if presser == "first" else here)
    assert any_arm_pressing(state, button)
    assert not any_arm_pressing(PhysicalState({}, FAR, secondary_ee_position=FAR), button)


def test_the_supply_alarm_is_silenced_by_the_other_arm(task_factory):
    task = task_factory("B1", response_limit=8.0, event_time=5.0)
    while task.event_time is None:
        task.advance(0.01)
    j = task.layout.alarm_station
    state = replace(task.previous or snapshot(task), ee_position=FAR,
                    secondary_ee_position=on_button(task.buttons[j]))
    for _ in range(40):
        task.time += 0.01
        task.observe(state, 0.01)
    assert task.physical_success and task.temporal_success


def test_the_other_arm_pressing_the_wrong_pump_is_still_a_failure(task_factory):
    task = task_factory("B1", response_limit=8.0, event_time=5.0)
    while task.event_time is None:
        task.advance(0.01)
    wrong = (task.layout.alarm_station + 1) % 3
    state = replace(task.previous or snapshot(task), ee_position=FAR,
                    secondary_ee_position=on_button(task.buttons[wrong]))
    for _ in range(40):
        task.time += 0.01
        task.observe(state, 0.01)
    assert "wrong_pump" in task.physical_failures
