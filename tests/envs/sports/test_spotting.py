"""Adversarial spotting traces; actual contact validation is run in Isaac."""
from types import SimpleNamespace
import numpy as np
import pytest
from roboclock.envs.rates import PHYSICS_HZ
from roboclock.envs.sports._shared.precision.config import config
from roboclock.envs.sports.bench_press.task import BenchPress
from roboclock.envs.sports.bench_press.process import HumanEffort


def setup(level="easy"):
    b=SimpleNamespace(config=config("B3",level),time=0.,human_effort=SimpleNamespace(released=False))
    return b,BenchPress(b)


def step(b,task,t,z=.98,vz=0.,forces=(0,0,0,0),gaps=(.076,.076),support=None,x=None,depart=False):
    c=b.config["workflow"]
    p=np.array([c['bench_x'] if x is None else x,0.,z])
    b.time=t
    hand_center=p if task.event_time is not None or depart else np.asarray(c['standby_position'])
    state=dict(time=t,objects={"bar":dict(position=p,velocity=[0,0,vz],forces=np.array(forces),support_contacts=support or {})},
               finger_gaps=np.array(gaps),arm_contact_peak=0.,
               hands={side:dict(position=hand_center+[0,sign*c['bench_grip_y'],0]) for side,sign in [('left',1),('right',-1)]})
    task.update(state)
    return state


def onset(b,t):
    step(b,t,3.9,z=.86)
    step(b,t,4.,vz=.1)
    step(b,t,4.1)
    step(b,t,4.25)
    step(b,t,4.3,vz=-.04)
    assert t.event_time==4.3


def test_normal_lowering_after_lockout_is_not_exhaustion():
    b,t=setup()
    step(b,t,3.9,z=.86)
    step(b,t,4.,vz=.1)
    step(b,t,5.,z=1.12)
    step(b,t,5.5,z=1.02,vz=-.1)
    step(b,t,6.,z=.86)
    assert t.event_time is None and not t.failures


def test_small_rebound_during_normal_lowering_does_not_arm_a_failed_press():
    b,t=setup()
    step(b,t,4.,z=1.12)
    step(b,t,4.5,z=1.04,vz=-.1)
    step(b,t,4.6,z=1.04,vz=.03)
    step(b,t,4.7,z=1.04)
    step(b,t,4.9,z=1.04)
    step(b,t,5.,z=1.04,vz=-.04)
    assert t.event_time is None


def test_incidental_hook_contacts_do_not_establish_weight_support():
    b,t=setup();onset(b,t)
    for now in (4.4,4.48):step(b,t,now,forces=(1,1,1,1),gaps=(.028,.028))
    c=b.config['workflow']
    for now in (5.,5.6):step(b,t,now,z=c['rack_height'],x=c['rack_x'],support={'RackLeft':.3,'RackRight':.3})
    assert not t.done and not t.result()['physical_success']


def test_stall_alone_does_not_start_clock():
    b,t=setup()
    step(b,t,3.9,z=.86)
    step(b,t,4.,vz=.1)
    step(b,t,4.1)
    step(b,t,4.25)
    assert t.stalled and t.event_time is None
    step(b,t,4.3,vz=-.04)
    assert t.event_time==4.3


@pytest.mark.parametrize('forces',[(0,0,0,0),(1,1,0,0),(1,0,1,0)])
def test_closed_jaws_or_single_hand_are_not_a_catch(forces):
    b,t=setup();onset(b,t)
    for now in (4.4,4.5,4.6):step(b,t,now,forces=forces,gaps=(.028,.028))
    assert t.catch_time is None and not b.human_effort.released


def test_bilateral_grip_must_arrest_the_fall():
    b,t=setup();onset(b,t)
    for now in (4.4,4.5):step(b,t,now,vz=-.10,forces=(1,1,1,1),gaps=(.028,.028))
    assert t.catch_time is None
    for now in (4.6,4.68):step(b,t,now,forces=(1,1,1,1),gaps=(.028,.028))
    assert t.catch_time==4.68 and b.human_effort.released


def test_rack_pose_requires_two_supports_release_and_dwell():
    b,t=setup();onset(b,t)
    for now in (4.4,4.48):step(b,t,now,forces=(1,1,1,1),gaps=(.028,.028))
    c=b.config['workflow']
    for now in (5.,5.6):step(b,t,now,z=c['rack_height'],x=c['rack_x'],support={'RackLeft':2},forces=(1,1,1,1),gaps=(.028,.028))
    assert not t.done
    for now in (6.,6.6):step(b,t,now,z=c['rack_height'],x=c['rack_x'],support={'RackLeft':6,'RackRight':6},forces=(1,1,1,1),gaps=(.028,.028))
    assert not t.done
    for now in (7.,7.6):step(b,t,now,z=c['rack_height'],x=c['rack_x'],support={'RackLeft':6,'RackRight':6})
    assert t.done and t.result()['success']


def test_late_but_physical_rescue_keeps_separate_scores():
    b,t=setup('hard');onset(b,t)
    c=b.config['workflow']
    late=t.event_time+c['catch_deadline']+.02  # grasp first closes after the deadline
    for now in (late,late+.08):step(b,t,now,forces=(1,1,1,1),gaps=(.028,.028))
    for now in (6.,6.6):step(b,t,now,z=c['rack_height'],x=c['rack_x'],support={'RackLeft':6,'RackRight':6})
    assert t.result()['physical_success'] and not t.result()['temporal_success']
    assert 'late_catch' in t.failures


def test_preclosing_before_failure_is_penalized():
    b,t=setup()
    step(b,t,3.,gaps=(.028,.028))
    step(b,t,3.1,gaps=(.028,.028))
    assert 'premature_grasp' in t.failures


def test_torso_contact_fails_even_if_bar_has_not_crossed_height_limit():
    b,t=setup()
    step(b,t,3.,support={'HumanTorso':1.})
    assert t.done and 'human_safety_contact' in t.failures


def test_seed_reproduces_fatigue_and_difficulty_only_changes_deadline():
    cases=[config('B3',level,7)['workflow'] for level in ('easy','medium','hard')]
    keys=('fatigue_rep','fatigue_progress','fatigue_stall','bench_mass','bench_top','bench_bottom')
    for k in keys:assert len({c[k] for c in cases})==1
    assert [c['catch_deadline'] for c in cases]==[2.05,1.43,1.0]
    assert config('B3','easy',7)==config('B3','easy',7)


def test_fatigued_human_applies_bounded_force_without_pose_edits():
    recorded=[]
    c=config('B3')
    body=dict(position=np.array([.32,0,.98]),velocity=np.zeros(3),orientation=[1,0,0,0],angular_velocity=np.zeros(3))
    b=SimpleNamespace(config=c,last_state={'objects':{'bar':body}},time=0.,contact_rows={'bar':0},
                      contacts=SimpleNamespace(apply_forces_and_torques_at_pos=lambda **kw:recorded.append(kw)))
    human=HumanEffort(b)
    b.time=human.fatigue_at+c['workflow']['fatigue_stall']+1.
    original=body['position'].copy()
    human.apply()
    assert 0. < human.force[2] < c['workflow']['bench_mass']*9.81
    np.testing.assert_array_equal(body['position'],original)
    human.released=True;human.apply()
    assert not np.any(recorded[-1]['forces']) and not np.any(recorded[-1]['torques'])


def test_capture_holds_interception_position_instead_of_chasing_a_fall():
    from roboclock.envs.sports.bench_press.oracle import BenchOracle
    commands=[]
    bar=dict(position=np.array([.32,0.,1.0]),velocity=np.array([0.,0.,-.04]),orientation=[1,0,0,0])
    b=SimpleNamespace(config=config('B3'),time=5.,last_state={'objects':{'bar':bar},'finger_gaps':[.076,.076],'hands':{side:dict(position=bar['position']+[0,sign*.36,0]) for side,sign in [('left',1),('right',-1)]}},
                      move=lambda side,target,*args,**kw:commands.append((side,target.copy(),kw)))
    task=SimpleNamespace(event_time=5.,catch_time=None)
    oracle=BenchOracle(b,task)
    oracle.act()
    first=commands[-2:]
    bar['position'][2]-=.04
    b.time+=.05
    oracle.act()
    for a,z in zip(first,commands[-2:]):
        np.testing.assert_allclose(a[1],z[1])
        assert z[2]['grip']==0.


def test_robot_waits_at_fixed_pose_through_normal_reps_and_stall():
    from roboclock.envs.sports.bench_press.oracle import BenchOracle
    commands=[]
    bar=dict(position=np.array([.32,0.,1.12]),velocity=np.array([0.,0.,-.1]),orientation=[1,0,0,0])
    b=SimpleNamespace(config=config('B3'),time=3.,last_state={'objects':{'bar':bar}},
                      move=lambda side,target,*args,**kw:commands.append((side,target.copy(),kw)))
    task=SimpleNamespace(event_time=None,catch_time=None,stalled=False)
    oracle=BenchOracle(b,task)
    for height in [1.12,.86,1.02]:
        bar['position'][2]=height
        task.stalled=height==1.02
        oracle.act()
    for side,target,kw in commands:
        np.testing.assert_allclose(target,np.array(b.config['workflow']['standby_position'])+[0,.36 if side=='left' else -.36,0])
        assert kw['grip']==.038
    assert oracle.phase=='wait'


def test_leaving_standby_before_exhaustion_is_allowed():
    b,t=setup()
    step(b,t,3.,depart=True)
    step(b,t,3.1,depart=True)
    assert 'premature_departure' not in t.failures and t.result()['standby_success']
    assert t.result()['standby_peak_error'] > b.config['workflow']['standby_tolerance']


def test_pre_event_bar_contact_is_a_failure_with_open_jaws():
    b,t=setup()
    step(b,t,3.,forces=(1,0,0,0))
    step(b,t,3.04,forces=(1,0,0,0))
    assert 'premature_bar_contact' in t.failures


@pytest.mark.parametrize('position,phase', [
    ([0.23, 0.0, 1.171], 'seat'),
    ([0.28, 0.0, 1.171], 'transfer'),
    ([0.23, 0.0, 1.09], 'transfer'),
])
def test_x5_reracking_requires_alignment_and_clearance(position, phase):
    """A retained bar above the nominal TCP must not deadlock safe transport."""
    from roboclock.envs.sports.bench_press.oracle import BenchOracle

    c = config('B3')
    c['embodiment'] = dict(model='dual_arx_x5')
    p = np.asarray(position)
    commands = {}
    def move(side, target, *args, **kw):
        commands[side] = np.asarray(target)
    b = SimpleNamespace(time=1.0, config=c, move=move,
                        last_state=dict(objects=dict(bar=dict(position=p, velocity=np.zeros(3),
                                                              orientation=[1, 0, 0, 0])),
                                        hands={side: dict(position=p + [0, sign*0.36, -0.016])
                                               for side, sign in [('left', 1), ('right', -1)]}))
    oracle = BenchOracle(b, SimpleNamespace())
    oracle.phase = 'transfer'
    oracle.capture_rotation = np.eye(3)
    oracle.act()
    assert oracle.phase == phase
    np.testing.assert_array_equal(b.last_state['objects']['bar']['position'], position)
    if phase == 'seat':
        # Compensate the observed 16 mm offset rather than hovering over hooks.
        assert commands['left'][2] == pytest.approx(c['workflow']['rack_height'] - 0.012 - 0.016)
        assert commands['right'][2] == pytest.approx(commands['left'][2])


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_supplemental_interrupted_bilateral_contact_resets_catch_dwell(level):
    b, t = setup(level)
    onset(b, t)
    dwell = t.c["catch_dwell"]
    start = 4.31
    step(b, t, start, forces=(1, 1, 1, 1), gaps=(.028, .028))
    step(b, t, start + dwell * .4, forces=(1, 1, 0, 0), gaps=(.028, .028))
    resumed = start + dwell * .5
    step(b, t, resumed, forces=(1, 1, 1, 1), gaps=(.028, .028))
    step(b, t, start + dwell * 1.1, forces=(1, 1, 1, 1), gaps=(.028, .028))
    assert t.catch_time is None and not b.human_effort.released
    step(b, t, resumed + dwell + 1 / PHYSICS_HZ, forces=(1, 1, 1, 1), gaps=(.028, .028))
    assert t.catch_time is not None and b.human_effort.released
