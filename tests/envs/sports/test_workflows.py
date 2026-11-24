"""Adversarial timing inputs; physical validation is recorded separately."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roboclock.envs.sports._shared.precision.config import config
from roboclock.envs.sports._shared.precision.lift import Lift
from roboclock.envs.sports._shared.precision.pool import Clearance


def test_unreachable_far_side_pot_plans_physical_cushion_reposition():
    from roboclock.envs.sports._shared.precision.pool import ClearanceOracle
    b = SimpleNamespace(config=config("C1"), pool_targets=["target1"], ball_radius=.023,
                        pockets=[np.array([x, y]) for x in [1.25, .15] for y in [-.12, .22]],
                        last_state={"objects": {"cue_ball": {"position": [.698, .260, .773]},
                                                  "target1": {"position": [1.073, -.156, .773]}},
                                    "hands": {"main": {"position": [.7, 0., 1.]}}})
    task = SimpleNamespace(pots={}, shots=[])
    oracle = ClearanceOracle(b, task)
    assert oracle.plan() and oracle.mode == "reposition"
    assert oracle.object == "cue_ball" and oracle.direction[1] > .9
    assert b.last_state["objects"]["cue_ball"]["position"] == [.698, .260, .773]
    task.pots = {"target1": 1.}
    assert not oracle.plan()


def body(p, f, support=None, v=(0, 0, 0)):
    return dict(position=np.array(p, float), orientation=[1, 0, 0, 0], forces=np.array(f, float),
                velocity=np.array(v, float), angular_velocity=np.zeros(3), support_contacts=support or {})


def test_nonconverging_aim_recovers_without_moving_task_objects():
    from roboclock.envs.sports._shared.precision.pool import ClearanceOracle
    cue = body([.5, 0., .8], [1., 1.])
    ball = body([.7, .2, .773], [0., 0.])
    hand = {"position": np.array([.5, 0., 1.]), "orientation": [1., 0., 0., 0.]}
    b = SimpleNamespace(time=6., hand_targets={}, config=config("C1"),
                        last_state={"objects": {"cue": cue, "cue_ball": ball}, "hands": {"main": hand}},
                        move=lambda *args, **kwargs: 1.)
    t = SimpleNamespace(pots={}, ready=False)
    oracle = ClearanceOracle(b, t)
    oracle.phase, oracle.object = "aim", "target1"
    oracle.act()
    assert oracle.phase == "raise" and oracle.excluded_target == "target1"
    assert oracle.replans == 1
    np.testing.assert_array_equal(cue["position"], [.5, 0., .8])
    np.testing.assert_array_equal(ball["position"], [.7, .2, .773])


def backend():
    return SimpleNamespace(config=config("C1"), time=0., pool_targets=["target1"],
                           pockets=[np.array([1.24, -0.12])],
                           filter_paths=["/left", "/right", "/World/Objects/cue", "/World/Objects/cue_ball"])


def state(now, hit=False, collision=False, pot=False, support=True):
    return dict(time=now, objects={
        "cue": body([0.36, -0.12, 0.774], [1, 1, 0, 0]),
        "cue_ball": body([0.56, -0.12, 0.774], [0, 0, float(hit), 0]),
        "target1": body([1.24, -0.12, 0.623] if pot else [0.8, -0.12, 0.774],
                        [0, 0, 0, float(collision)], {"Pocket0": 0.5} if support and pot else {})})


def prepared():
    b = backend()
    t = Clearance(b)
    for now in [0., 0.3, 0.5]:
        t.update(state(now))
    return t


def test_duplicate_supported_pot_counts_once_and_requires_full_budget():
    t = prepared()
    t.update(state(1., hit=True))
    t.update(state(1.1, collision=True))
    for now in [2., 2.3, 3., 40.]:
        t.update(state(now, pot=True))
    assert t.result()["valid_pots"] == 1
    assert not t.result()["temporal_success"]
    t.update(state(60., pot=True))
    assert t.result()["temporal_success"] and t.result()["pots_per_minute"] == 1


def test_pot_after_budget_does_not_count():
    t = prepared()
    t.update(state(59.7, hit=True))
    t.update(state(59.8, collision=True))
    t.update(state(59.9, pot=True))
    t.update(state(60.2, pot=True))
    assert t.done and t.result()["valid_pots"] == 0


@pytest.mark.parametrize("support,collision", [(False, True), (True, False)])
def test_pot_requires_support_and_causal_ball_collision(support, collision):
    t = prepared()
    t.update(state(1., hit=True))
    t.update(state(1.1, collision=collision))
    for now in [2., 2.5]:
        t.update(state(now, pot=True, support=support))
    assert not t.pots


def test_second_touch_before_settle_invalidates_the_shot():
    t = prepared()
    t.update(state(1., hit=True))
    s = state(1.05)
    s["objects"]["cue_ball"]["position"][0] += .08
    s["objects"]["cue_ball"]["velocity"][0] = 0.2
    t.update(s)
    t.update(state(1.1, hit=True))
    assert "strike_before_settled" in t.failures
    assert not any(s["valid"] for s in t.shots)


def test_zero_force_sample_without_tip_separation_is_one_stroke():
    t = prepared()
    t.update(state(1., hit=True))
    t.update(state(1.004))
    t.update(state(1.008, hit=True))
    assert len(t.shots) == 1
    assert "strike_before_settled" not in t.failures
    t.update(state(1.07, hit=True))
    assert "push_shot" in t.failures


def test_scratch_invalidates_an_earlier_pot_on_same_stroke():
    t = prepared()
    t.update(state(1., hit=True))
    t.update(state(1.1, collision=True))
    t.update(state(2., pot=True))
    t.update(state(2.3, pot=True))
    s = state(2.4, pot=True)
    s["objects"]["cue_ball"]["position"][2] = 0.6
    t.update(s)
    assert t.result()["valid_pots"] == 0
    assert not t.result()["physical_success"]


def test_combination_pot_requires_a_connected_contact_chain():
    b = backend()
    b.pool_targets.append("target2")
    b.filter_paths += ["/World/Objects/target1", "/World/Objects/target2"]
    t = Clearance(b)
    for now in [0., .3, .5, 1., 1.1, 2., 2.3]:
        s = state(now, hit=now == 1., collision=now == 1.1)
        for obj in s["objects"].values():
            obj["forces"] = np.pad(obj["forces"], (0, 2))
        s["objects"]["target2"] = body([1.24, -.12, .623] if now >= 2 else [1., -.12, .774],
                                          [0, 0, 0, 0, float(now == 1.1), 0],
                                          {"Pocket0": .5} if now >= 2 else {})
        t.update(s)
    assert t.result()["valid_pots"] == 1
    assert t.pots == {"target2": 2.3}


def lift_state(now, z=1.04, vz=0):
    return dict(time=now, objects={"bar": body([0.36, 0, z], [1, 1, 1, 1], v=(0, 0, vz))}, arm_contact_peak=0.)


def test_hold_reminder_is_not_permission_to_lower():
    b = SimpleNamespace(config=config("B3"), time=0., bar_home=np.array([0.36, 0, .84]))
    t = Lift(b)
    for now in np.arange(0, 2.2, .05):
        b.time = now
        t.update(lift_state(now))
    assert t.reminder_shown and t.down is None
    t.update(lift_state(2.21, z=.99, vz=-.1))
    assert "early_lowering" in t.failures


def test_lift_rounds_have_distinct_height_and_duration():
    b = SimpleNamespace(config=config("B3"), time=0., bar_home=np.array([.36, 0, .84]))
    t = Lift(b)
    pairs = []
    for _ in range(3):
        pairs.append((t.target_height, t.hold_duration))
        t.rounds.append({})
    assert len(set(pairs)) == 3


def test_lowering_is_measured_from_actual_height_at_down():
    b = SimpleNamespace(config=config("B3"), time=0., bar_home=np.array([.36, 0, .84]))
    t = Lift(b)
    t.phase, t.down, t.down_height = "DOWN", 1., 1.029
    t.update(lift_state(1.05, z=1.025, vz=-0.1))
    assert t.lower is None
    t.update(lift_state(1.1, z=1.016, vz=-0.1))
    assert t.lower == 1.1


def test_actual_rotation_interrupts_hold_even_if_velocity_channel_is_zero():
    b = SimpleNamespace(config=config("B3"), time=0., bar_home=np.array([.36, 0, .84]))
    t = Lift(b)
    for now in np.arange(0, 5, .02):
        s = lift_state(now)
        s["objects"]["bar"]["orientation"] = np.roll(Rotation.from_rotvec([0, 0.2 * now, 0]).as_quat(), 1)
        t.update(s)
    assert t.down is None


def test_stationary_measured_pose_tolerates_solver_bias_velocity():
    b = SimpleNamespace(config=config("B3"), time=0., bar_home=np.array([.36, 0, .84]))
    t = Lift(b)
    for now in np.arange(0, 5, .02):
        s = lift_state(now)
        s["objects"]["bar"]["angular_velocity"] = np.array([0, .22, 0])
        t.update(s)
    assert t.down is not None


def test_temporal_difficulty_keeps_layout_and_round_program_constant():
    for name in ["B3", "C1"]:
        easy, hard = config(name), config(name, "hard")
        assert easy["scene"] == hard["scene"]
        assert easy["workflow"]["lift_heights"] == hard["workflow"]["lift_heights"]
        assert easy["workflow"]["pool_horizon"] == hard["workflow"]["pool_horizon"]
        assert easy["primary_temporal_category"] == hard["primary_temporal_category"]


@pytest.mark.parametrize('height_error,expected_phase', [(0.06, 'aim'), (0.0, 'stroke')])
def test_native_pool_does_not_stroke_above_the_ball(height_error, expected_phase):
    """The X5 preview reached the horizontal aim line while its tip was still high."""
    from roboclock.envs.sports._shared.precision.pool import ClearanceOracle

    tip_goal = np.array([0.50, 0.0, 0.783])
    cue = body(tip_goal + [-0.177, 0, height_error], [1, 1])
    hand = dict(position=np.array([0.35, 0, 0.90]), orientation=[1, 0, 0, 0])
    b = SimpleNamespace(
        time=1.0, config=config('C1'), rig=object(),
        last_state=dict(objects=dict(cue=cue, cue_ball=body([0.55, 0, 0.773], [0, 0]),
                                     target1=body([0.8, 0, 0.773], [0, 0])),
                        hands=dict(main=hand)),
        move=lambda *args, **kwargs: 0.0,
    )
    t = SimpleNamespace(pots={}, ready=True)
    oracle = ClearanceOracle(b, t)
    oracle.phase, oracle.mode, oracle.object = 'aim', 'pot', 'target1'
    oracle.desired_cue_rotation = np.eye(3)
    oracle.tip_goal = tip_goal
    oracle.goal = oracle.nominal_goal = hand['position'].copy()
    before = cue['position'].copy()
    oracle.act()
    assert oracle.phase == expected_phase
    np.testing.assert_array_equal(cue['position'], before)


@pytest.mark.parametrize('cue_height,phase', [(0.941, 'wait'), (0.80, 'raise')])
def test_x5_transport_uses_complete_cue_clearance(cue_height, phase):
    """A clear cue may leave an unreachable waypoint; a low cue must keep lifting."""
    from roboclock.envs.sports._shared.precision.pool import ClearanceOracle

    c = config('C1')
    c['embodiment'] = dict(model='dual_arx_x5')
    cue = body([0.399, -0.069, cue_height], [1, 1])
    cue['orientation'] = np.roll(Rotation.from_euler('y', -12, degrees=True).as_quat(), 1)
    b = SimpleNamespace(
        config=c, time=1.0, ball_radius=0.023, pool_targets=['target1'],
        last_state=dict(objects=dict(cue=cue, cue_ball=body([0.8, 0, 0.773], [0, 0]),
                                     target1=body([1.0, 0, 0.773], [0, 0]))),
        move=lambda *args, **kwargs: 0.011,
    )
    oracle = ClearanceOracle(b, SimpleNamespace(pots={}))
    oracle.phase = 'raise'
    before = cue['position'].copy()
    oracle.act()
    assert oracle.phase == phase
    np.testing.assert_array_equal(cue['position'], before)
