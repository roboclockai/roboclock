"""The shared cue must rest physically before either hand releases or switches."""
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.sports._shared.precision.pool_exchange import act


def setup_exchange(supported, phase='exchange_lower'):
    home = np.array([.45, .10, .794])
    cue = {'position': home.copy(), 'support_contacts': {'CueRest': .4 if supported else 0.}}
    b = SimpleNamespace(time=1., cue_exchange_home=home,
                        last_state={'objects': {'cue': cue}}, move=lambda *a, **kw: 0.)
    o = SimpleNamespace(b=b, phase=phase, start=0., exchange_high=np.array([.45, .1, 1.06]),
                        exchange_goal=home + [0, 0, .045], exchange_rotation=np.eye(3),
                        rotation=np.eye(3), diagnostics=[])
    o.transition = lambda phase: setattr(o, 'phase', phase)
    return o


@pytest.mark.parametrize('phase', ['exchange_lower', 'exchange_release'])
def test_pose_on_rest_without_support_does_not_advance(phase):
    o = setup_exchange(False, phase)
    act(o)
    assert o.phase == phase


def test_supported_placement_allows_release():
    o = setup_exchange(True)
    act(o)
    assert o.phase == 'exchange_release'


def test_arm_switch_changes_observation_mapping_not_object_pose():
    o = setup_exchange(True, 'exchange_retreat')
    o.b.rig = SimpleNamespace(fingers={'left': ['l0', 'l1'], 'right': ['r0', 'r1']},
                             pose=lambda side: (np.array([.3, .3, 1.1]), np.array([1., 0, 0, 0])))
    o.b.filter_paths = ['r0', 'r1', 'l0', 'l1', 'CueRest']
    o.b.hand_targets = {}
    original = o.b.last_state['objects']['cue']['position'].copy()
    act(o)
    assert o.b.active_side == 'left' and o.phase == 'approach'
    assert o.b.filter_paths == ['l0', 'l1', 'r0', 'r1', 'CueRest']
    assert o.b.hand_targets['main'][1].shape == (3, 3)
    np.testing.assert_array_equal(o.b.last_state['objects']['cue']['position'], original)
