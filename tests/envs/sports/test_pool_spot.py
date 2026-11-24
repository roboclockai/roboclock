"""Adversarial rules for the opt-in physical white-ball reset drill."""
from types import SimpleNamespace

import numpy as np

from roboclock.envs.sports._shared.precision.config import config
from roboclock.envs.sports._shared.precision.pool_spot import SpotClearance


def task():
    c = config('C1')
    c['pool'] = dict(spot_clearance=True, placement_zones=[[.62, -.112, .773]])
    b = SimpleNamespace(config=c, pool_targets=['target1', 'target2', 'target3'])
    return SpotClearance(b)


def test_only_resetting_left_hand_may_touch_white():
    t = task()
    s = {'objects': {'cue_ball': {'forces': np.array([0., 0., 1., 1.])},
                     'target1': {'forces': np.array([0., 0., 1., 1.])}}}
    assert not t.hand_contact_allowed('cue_ball', s)
    t.reset_open = True
    assert t.hand_contact_allowed('cue_ball', s)
    assert not t.hand_contact_allowed('target1', s)
    s['objects']['cue_ball']['forces'][0] = 1.
    assert not t.hand_contact_allowed('cue_ball', s)


def test_one_pot_quota_does_not_pass_three_ball_clearance():
    t = task()
    t.done = True
    t.shots = [dict(valid=True, pots=['target1'])]
    t.pots = {'target1': 2.}
    r = t.result()
    assert r['temporal_success']  # Legacy minimum rate alone is insufficient.
    assert not r['success'] and not r['physical_success'] and not r['complete_clearance']


def test_three_valid_pots_require_no_fouls_and_full_budget():
    t = task()
    t.shots = [dict(valid=True, pots=list(t.b.pool_targets))]
    t.pots = dict.fromkeys(t.b.pool_targets, 2.)
    assert not t.result()['success']
    t.done = True
    assert t.result()['success']
    t.fail('hand_ball_contact', 3.)
    assert not t.result()['success']


def test_single_target_reset_probe_never_claims_clearance():
    t = task()
    t.b.pool_targets = ['target1']
    t.done = True
    t.shots = [dict(valid=True, pots=['target1'])]
    t.pots = {'target1': 2.}
    r = t.result()
    assert r['diagnostic_probe'] and r['diagnostic_targets_potted']
    assert not r['success'] and not r['complete_clearance']


def reset_task_and_state():
    t = task()
    t.b.fingers = ['r0', 'r1', 'l0', 'l1']
    names = ['cue', 'cue_ball', *t.b.pool_targets]
    t.b.filter_paths = t.b.fingers + ['/World/Objects/' + n for n in names]
    t.b.pockets = []
    t.reset_open = True
    def body(p):
        return dict(position=np.array(p, float), orientation=[1., 0., 0., 0.],
                    forces=np.zeros(len(t.b.filter_paths)), velocity=np.zeros(3),
                    angular_velocity=np.zeros(3), support_contacts={})
    s = dict(time=0., objects={n: body([.7, .1, .773]) for n in names},
             hands={'left': {'position': np.array([.62, -.112, .785])}})
    s['objects']['cue']['position'] = np.array([.3, -.1, 1.])
    s['objects']['cue']['forces'][:2] = 1.
    s['objects']['cue_ball']['position'] = np.array([.62, -.112, .773])
    return t, s


def test_reset_requires_grasp_lift_support_release_and_hand_withdrawal():
    t, s = reset_task_and_state()
    ball = s['objects']['cue_ball']
    ball['forces'][2:4] = 1.
    t.update(s)
    s['time'] = .2
    t.update(s)
    assert t.reset_grasped and not t.resets
    s['time'] = .3
    ball['position'][2] = .9
    s['hands']['left']['position'][2] = .912
    t.update(s)
    assert t.reset_lifted
    s['time'] = .4
    ball['forces'][:] = 0.
    ball['position'][2] = .773
    ball['support_contacts']['PoolBed'] = .5
    s['hands']['left']['position'][2] = .785
    t.update(s)
    s['time'] = 1.
    t.update(s)
    assert not t.resets  # Released but fingers have not cleared the ball.
    s['hands']['left']['position'][2] = 1.
    s['time'] = 1.1
    t.update(s)
    s['time'] = 1.4
    t.update(s)
    assert len(t.resets) == 1 and not t.reset_open
    assert t.valid_count == 0  # A reset cannot score a pot.


def test_unsupported_white_placement_does_not_count_even_after_withdrawal():
    t, s = reset_task_and_state()
    t.reset_grasped = t.reset_lifted = True
    s['hands']['left']['position'][2] = 1.
    for now in [0., .3, 1.]:
        s['time'] = now
        t.update(s)
    assert not t.resets


def test_reset_after_budget_cannot_be_accepted():
    t, s = reset_task_and_state()
    t.reset_grasped = t.reset_lifted = True
    s['hands']['left']['position'][2] = 1.
    s['objects']['cue_ball']['support_contacts']['PoolBed'] = .5
    s['time'] = 59.9
    t.update(s)
    s['time'] = 60.3
    t.update(s)
    assert t.done and not t.resets
