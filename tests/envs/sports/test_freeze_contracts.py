"""Adversarial evaluation and independent randomization contracts."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.rates import PHYSICS_HZ
from roboclock.envs.sports.freeze.config import config
from roboclock.envs.sports.freeze.task import Task, deadline_met, inside_goal


def backend():
    c = config()
    c["freeze"]["red_windows"] = [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]
    return SimpleNamespace(
        config=c,
        time=0,
        filter_paths=[
            "f0",
            "f1",
            "f2",
            "f3",
            "/World/Rink",
            "/World/Table",
            "/World/Objects/puck",
            "/World/Objects/left_pusher",
            "/World/Objects/right_pusher",
        ],
    )


def state(b, now, velocity=0.0):
    b.time = now

    dt = now - getattr(b, "test_previous_time", now)
    b.test_displacement = getattr(b, "test_displacement", 0.0) + velocity * dt
    b.test_previous_time = now

    def obj(p):
        p = np.array(p, float)
        p[0] += b.test_displacement
        return dict(
            position=np.array(p, float),
            orientation=np.array([1.0, 0, 0, 0]),
            velocity=np.array([velocity, 0, 0]),
            angular_velocity=np.zeros(3),
            forces=np.zeros(9),
            support_contacts={"Rink": 0.4},
        )

    return dict(
        time=now,
        joints=np.zeros(16),
        arm_contact_peak=0.0,
        objects={
            "puck": obj(b.config["freeze"]["puck_start"]),
            "left_pusher": obj([0.44, 0.4, 0.85]),
            "right_pusher": obj([0.44, -0.4, 0.85]),
        },
        hands={s: obj([0.44, y, 1.0]) for s, y in [("left", 0.4), ("right", -0.4)]},
    )


def test_seed_reproducibility_and_timing_only_difficulty():
    a, b = config("easy", 7), config("hard", 7)
    assert a == config("easy", 7)
    assert a["freeze"]["red_windows"] != config("easy", 8)["freeze"]["red_windows"]
    assert a["freeze"]["puck_start"] != config("easy", 8)["freeze"]["puck_start"]
    ca, cb = copy.deepcopy(a["freeze"]), copy.deepcopy(b["freeze"])
    # A red lasts the response limit plus the seed's hold: only the reds (and with them the
    # onsets and the horizon) stretch with the tier; holds and greens are the seed's.
    timing = ("response_limit", "red_windows", "horizon")
    ra, rb = ([x.pop(k) for k in timing] for x in (ca, cb))
    assert ca == cb
    assert [z - a0 - ra[0] for a0, z in ra[1]] == pytest.approx([z - a0 - rb[0] for a0, z in rb[1]])
    assert ra[2] - sum(z - a0 for a0, z in ra[1]) == pytest.approx(rb[2] - sum(z - a0 for a0, z in rb[1]))
    assert a["embodiment"]["fixed_base"]


@pytest.mark.parametrize("level,limit", [("easy", 8.0), ("medium", 4.0), ("hard", 2.0)])
@pytest.mark.parametrize("seed", range(10))
def test_reds_fit_the_response_limit_with_a_hold_and_three_reds_are_guaranteed(level, limit, seed):
    f = config(level, seed)["freeze"]
    assert f["response_limit"] == limit >= 1.0
    windows = f["red_windows"]
    assert len(windows) >= 3
    for a0, z in windows:
        # Even a stop at the deadline, confirmed after the stable dwell, leaves a hold.
        assert z - a0 - limit - f["stable_dwell"] >= 2.0
    starts = [a0 for a0, _ in windows] + [f["horizon"]]
    green = starts[0] + sum(s - z for (_, z), s in zip(windows, starts[1:]))
    assert green == pytest.approx(48.0)
    assert windows[-1][1] < f["horizon"]
    # The third red starts within the first 30 s of green.
    assert windows[2][0] - sum(z - a0 for a0, z in windows[:2]) <= 30.0


def test_strict_deadline_boundary():
    assert deadline_met(0.39999, 0.4)
    assert not deadline_met(0.4, 0.4)
    assert not deadline_met(0.40001, 0.4)


def test_goal_requires_whole_puck():
    c = config()["freeze"]
    g = c["goals"][0]
    assert inside_goal(g, g, c)
    p = np.array(g)
    p[0] += c["goal_half_size"][0] - 0.005
    assert not inside_goal(p, g, c)


def test_stopping_requires_stable_dwell_and_reports_onset():
    b = backend()
    t = Task(b)
    for now in np.arange(0.9, 1.6, 1 / PHYSICS_HZ):
        t.update(state(b, float(now), velocity=0.1 if now < 1.2 else 0))
    r = t.reactions[0]
    assert 0.18 < r["latency"] < 0.25
    assert r["confirmation_time"] - r["stop_time"] >= 0.15
    assert r["within_deadline"]


def test_late_stop_not_repaired_by_eventual_rest():
    b = backend()
    b.config["freeze"]["response_limit"] = 0.2
    t = Task(b)
    for now in np.arange(0.9, 2.0, 1 / PHYSICS_HZ):
        t.update(state(b, float(now), velocity=0.1 if now < 1.4 else 0))
    assert "late_stop" in t.failures
    assert not t.reactions[0]["within_deadline"]


def test_motion_after_confirmed_stop_fails_hold():
    b = backend()
    t = Task(b)
    for now in np.arange(0.9, 1.6, 1 / PHYSICS_HZ):
        t.update(state(b, float(now), velocity=0.05 if now > 1.4 else 0))
    assert "red_motion" in t.failures
    assert not t.reactions[0]["hold_ok"]


def test_idle_never_completes_route():
    b = backend()
    t = Task(b)
    for now in np.arange(0, b.config["freeze"]["horizon"] + 0.1, 0.05):
        t.update(state(b, float(now)))
    assert t.done and not t.result()["success"]
    assert "incomplete_route" in t.failures


@pytest.mark.parametrize("unsupported,raised", [(True, False), (False, True)])
def test_goal_pose_alone_does_not_count(unsupported, raised):
    b = backend()
    t = Task(b)
    t.checkpoint = True
    t.last_contacts["right"] = 0
    for now in np.arange(0.01, 0.7, 0.01):
        s = state(b, float(now))
        s["objects"]["puck"]["position"] = np.array(b.config["freeze"]["goals"][0])
        if unsupported:
            s["objects"]["puck"]["support_contacts"] = {}
        if raised:
            s["objects"]["puck"]["position"][2] += 0.1
        t.update(s)
    if unsupported:
        assert t.round == 0
    if raised:
        assert not t.result()["physical_success"]


def test_legacy_mobile_configuration_is_rejected_without_mutation():
    from roboclock.simulators.isaac.embodiments.config import configure

    c = {"keep": 1}
    with pytest.raises(ValueError, match="fixed-freeze"):
        configure(c, "B4")
    assert c == {"keep": 1}


def test_pose_path_does_not_cancel_fast_back_and_forth_motion():
    b = backend()
    t = Task(b)
    for i in range(12):
        s = state(b, i / PHYSICS_HZ)
        s["objects"]["puck"]["position"][0] += 0.001 if i % 2 else -0.001
        stopped = t.stopped(s)
    assert not stopped
    assert t.current_speeds["puck"]["linear"] > 0.4


def test_constraint_velocity_spikes_do_not_replace_measured_motion():
    b = backend()
    t = Task(b)
    for i in range(12):
        s = state(b, i / PHYSICS_HZ)
        s["objects"]["left_pusher"]["velocity"] = np.array([1.0, 0, 0])
        stopped = t.stopped(s)
    assert stopped
    assert t.current_speeds["left_pusher"]["linear"] == 0



@pytest.mark.parametrize("body", ["puck", "left_pusher", "right_pusher", "hand_left", "hand_right"])
def test_supplemental_each_body_must_hold_after_confirmed_stop(body):
    b = backend()
    t = Task(b)
    for i in range(180):
        now = 0.9 + i / PHYSICS_HZ
        s = state(b, now)
        if now > 1.4:
            if body.startswith("hand_"):
                o = s["hands"][body[5:]]
            else:
                o = s["objects"][body]
            o["position"][0] += (now - 1.4) * 0.10
            o["velocity"][:] = 0  # Pose history must detect motion despite this.
        t.update(s)
    assert t.reactions[0]["within_deadline"]
    assert not t.reactions[0]["hold_ok"]
    assert "red_motion" in t.failures
