"""Independent post-hoc check of a Dual-X5 episode from its measured trajectory (no controller phases).

    python scripts/household/validate_x5_episode.py RUN_DIR

Recomputes the physical and temporal predicates from the 25 Hz trajectory samples (object poses,
contact forces, gripper poses) and the ground-truth hidden process parameters in result.json
(``metrics.episode_spec``), independently of the online task model. Recorded runs (RoboDojo rig spec) hold
the overview RUN_DIR/<TASK>.mp4 (1280x960) and the policy videos head.mp4, left_wrist.mp4, right_wrist.mp4
(640x480); all are probed and fully decoded, must run at 25 fps with equal frame counts (frame-synchronized,
= frames.json), and the overview must show a new image every frame. A legacy RUN_DIR/demo.mp4 is checked
alone. The result's rig check (robodojo_rig_check) must have passed. Writes RUN_DIR/validation.json; exits 1
when any check fails.
"""
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

H = .8584


def pinch(row, index, cols, side, prev, dt):
    """Bilateral jaw contact with low object motion relative to the tool frame."""
    f = np.linalg.norm(np.array(row['contact_force_vectors'])[index][cols], axis=-1)
    tcp = row['tcp'][side]
    rel = R.from_quat(tcp['quaternion_xyzw']).inv().apply(np.array(row['tool_positions'][index])-tcp['position'])
    speed = 0. if side not in prev or dt == 0 else float(np.linalg.norm(rel-prev[side])/dt)
    prev[side] = rel
    return bool(f.min() > .04 and speed < .10)


def check_a1(rows, spec, setup):
    """Sugar pour (2026-11-19). The grams are re-integrated from the measured jar tilt and stream landing point with
    the hidden jar fill and flow coefficient (the flow law of task.py written out again, at the 25 Hz samples), the
    set-down, the stir turns and the spoon's release from poses and contact-force norms."""
    cup, tea_z = np.array([.2080456, .1400678]), H+.064
    jar_g, glass_g, spill_g, air_g, air_s = spec['jar_sugar_g'], 0., 0., 0., 0.
    turns = max_turns = 0.
    last, prev, prev_t = None, {}, None
    down = spoon = 0.
    set_down = sugar_at_set_down = None
    poured = stir_before = False
    violations = []
    for row in rows:
        t = row['t']
        dt = 0. if prev_t is None else t-prev_t
        prev_t = t
        f = np.array(row['contact_forces_N'])
        pos, vel = np.array(row['tool_positions']), np.array(row['tool_velocities'])
        tilt = row['jar_tilt_deg']
        lip = np.array(row['jar_lip'])
        released = f[0, :4].max() < .03
        upright_down = released and f[0, 4] > .05 and tilt < 10. and np.linalg.norm(vel[0, :3]) < .025 and pos[0, 2] < H+.005
        if dt > 0:
            onset = 30.+80.*(1.-jar_g/94.)
            excess = tilt-onset
            rate = 0.
            if excess > 0 and jar_g > 0 and not upright_down:
                rate = spec['max_flow_g_s']*spec['flow_coefficient']*min(1., excess/spec['flow_span_deg'])**1.5
            out = min(jar_g, rate*dt)
            jar_g -= out
            poured |= out > 0
            if np.linalg.norm(lip[:2]-cup) < .0385-.004 and lip[2] > tea_z:
                air_g += out
            else:
                air_s += out
            land = 1-math.exp(-dt/spec['tail_tau_s'])
            glass_g, spill_g = glass_g+air_g*land, spill_g+air_s*land
            air_g, air_s = air_g*(1-land), air_s*(1-land)
        down = down+dt if (poured and upright_down) else 0.
        if set_down is None and down >= spec['release_dwell']:
            set_down, sugar_at_set_down = t, glass_g
        head = np.array(row['spoon_head_center'])
        zmin, zmax = row['head_z_interval']
        in_cup = zmin > H+.0075 and zmax < tea_z and row['head_max_cup_radius'] < .0385
        held = bool(f[1, 2:4].min() > .04 or f[1, 0:2].min() > .04)
        if in_cup and set_down is None:
            stir_before = True
        if in_cup and held:
            angle = math.atan2(head[1]-cup[1], head[0]-cup[0])
            if last is not None:
                turns = max(0., turns-math.atan2(math.sin(angle-last), math.cos(angle-last))/(2*math.pi))
            last = angle
            max_turns = max(max_turns, turns)
        else:
            last = None
        lo, hi = np.array(row['head_xy_bounds'])  # basket centre (0.34, 0.10) on the 0.60 m rig
        spoon_ok = (f[1, 6] > .04 or f[1, 4] > .04) and f[1, :4].max() < .03 and np.linalg.norm(vel[1, :3]) < .025 \
            and lo[0] > .279 and hi[0] < .401 and lo[1] > .060 and hi[1] < .140 and zmin > H+.002 and zmax < H+.046 and max_turns >= 2.
        spoon = spoon+dt if spoon_ok else 0.
        prev['spoon'] = max(prev.get('spoon', 0.), spoon)
        if pos[:, 2].min() < .80:
            violations.append('tool_below_worktop')
    if stir_before:
        violations.append('spoon_in_glass_before_jar_set_down')
    if spill_g >= spec['spill_limit_g']:
        violations.append('sugar_spilled')
    band = spec['band']
    temporal = sugar_at_set_down is not None and band[0] <= sugar_at_set_down <= band[1]
    physical = (set_down is not None and max_turns >= spec['required_turns'] and prev.get('spoon', 0.) >= spec['release_dwell']
                and not violations)
    return dict(physical_success=bool(physical), temporal_success=bool(temporal), set_down_s=set_down,
                sugar_g_gt=sugar_at_set_down, spilled_g_gt=spill_g, target_band_g=band, measured_clockwise_turns=max_turns,
                spoon_released_dwell_s=prev.get('spoon', 0.), violations=sorted(set(violations)))


def check_a2(rows, spec, setup):
    """Ice dispenser (half a glass). Press/release from the measured paddle angle (switch 5 deg, release 3 deg); the
    deliveries the dispenser owed for that press are recomputed from the hidden spec; the cubes in the glass and the
    ice level (top of the highest cube's oriented box over the glass's inner height) come from the measured cube and
    glass poses at the end, with the glass upright, released and at rest on the counter."""
    ts = spec['time_scale']
    rng = np.random.default_rng([int(spec['seed']), 7302]) if spec['seed'] else None
    jit = rng.uniform(-spec['jitter'], spec['jitter'], setup['pool']) if rng is not None else np.zeros(setup['pool'])
    due = np.cumsum([spec['spin_up_s']]+[spec['interval_s']*(1+j) for j in jit[1:]])
    r_in, floor, rim, top = setup['glass_inner_radius_m'], setup['glass_floor_m'], setup['glass_rim_m'], setup['counter_top_z']
    hx, hy, hz = setup['cube_half_extents_m']
    corners = np.array([[sx*hx, sy*hy, sz*hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    (x0, y0), (x1, y1) = setup['island_top_xy']
    press = release = None
    held, prev_t, dwell, best, max_tilt = 0., None, 0., 0., 0.
    violations, final, hist = [], None, []
    for row in rows:
        t = row['t']
        dt = 0. if prev_t is None else t-prev_t
        prev_t = t
        a = row['paddle_deg']
        if press is None and a >= setup['press_deg']:
            press = t
        if press is not None and release is None:
            held = held+dt if a > setup['release_deg'] else held
            if a <= setup['release_deg'] and held >= spec['press_dwell_s']:
                release = t
        gp, gr = np.array(row['glass_m']), R.from_quat(row['glass_quaternion_xyzw'])
        tilt = math.degrees(math.acos(np.clip(gr.apply([0, 0, 1])[2], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        if gp[2] < top-.05:
            violations.append('glass_dropped')
        cp = np.array(row['cube_positions'])
        hist.append((t, cp))
        while len(hist) > 2 and hist[1][0] <= t-.2:  # speed from displacement over 0.2 s (solver velocities jitter)
            hist.pop(0)
        cv = np.linalg.norm(cp-hist[0][1], axis=-1)/max(t-hist[0][0], 1e-3)
        rel = gr.inv().apply(cp-gp)
        inside = (np.hypot(rel[:, 0], rel[:, 1]) < r_in+.002) & (rel[:, 2] > floor-.006) & (rel[:, 2] < rim+.004)
        waiting = cp[:, 2] > setup['bin_z']  # still in the bin
        down = (abs(gp[2]-top) < .006 and tilt < 8. and max(row['jaw_N']) < .3 and x0 < gp[0] < x1 and y0 < gp[1] < y1
                and not np.any(~waiting & (cv > .02)))
        dwell = dwell+dt if down and release is not None else 0.
        if dwell > best:
            best = dwell
            tops = [rel[i, 2]+(gr.inv()*R.from_quat(np.roll(row['cube_quaternions_wxyz'][i], -1))).apply(corners)[:, 2].max()
                    for i in np.flatnonzero(inside & ~waiting)]
            level = (max(tops)-floor)/(rim-floor) if tops else 0.
            final = dict(t=t, in_glass=int((inside & ~waiting).sum()), outside=int((~inside & ~waiting).sum()), ice_level=float(level))
    owed = None
    if press is not None and release is not None:
        run = (release-press)*ts+spec['coast_s']
        owed = int(np.sum(due <= run))
    if max_tilt > spec['max_tilt_deg']:
        violations.append('glass_tipped')
    if final is not None and final['outside']:
        violations.append('cube_outside_glass')
    if final is not None and owed is not None and final['in_glass'] != owed:
        violations.append('count_differs_from_deliveries_owed')
    lo, hi = spec['level_band']
    physical = press is not None and release is not None and best >= spec['settle_s'] and not violations
    temporal = final is not None and lo <= final['ice_level'] <= hi
    return dict(physical_success=bool(physical), temporal_success=bool(temporal), press_s=press, release_s=release,
                deliveries_if_held_s=None if press is None else (press+due[:10]/ts).round(3).tolist(), deliveries_owed=owed,
                final=final, level_band=[lo, hi], settled_on_counter_s=best, max_glass_tilt_deg=max_tilt,
                violations=sorted(set(violations)))


def check_b1(rows, spec, setup):
    """Overflow re-integrated from the measured lever tilt; closure = lever shut with no hand contact for 1 s."""
    ts = spec['time_scale']
    headroom = spec['headroom_l']
    stored = overflow = 0.
    early = False
    prev_t, touched, closed_for, closure, reopened = None, False, 0., None, False
    for row in rows:
        t = row['t']
        dt = 0. if prev_t is None else t-prev_t
        prev_t = t
        u = min(1., max(0., row['lever_deg']/15.))
        surge_time = spec.get('surge_time', 0.)
        rate = spec['inflow_l_s'] if t >= surge_time else spec['normal_flow_l_s']
        q_in = rate*ts*u
        stored += q_in*dt
        if stored > headroom:
            overflow += stored-headroom
            stored = headroom
        contact = max(row['lever_contact_N']) > .3
        touched |= contact
        if t < surge_time and (contact or u < .9):
            early = True
        closed = t >= surge_time and q_in < spec['closed_flow_l_s']*ts and not contact
        closed_for = closed_for+dt if closed else 0.
        if closure is None and touched and closed_for >= spec['closure_dwell']:
            closure = t-spec['closure_dwell']
        if closure is not None and q_in > spec['closed_flow_l_s']*ts:
            reopened = True
    onset = spec.get('surge_time', 0.)
    normal = spec.get('normal_flow_l_s', spec['inflow_l_s'])*ts
    time_to_overflow = (headroom/normal if normal > 0 and headroom <= normal*onset
                        else onset + (headroom-normal*onset)/(spec['inflow_l_s']*ts))
    violations = (['pre_event_action'] if early else [])+(['faucet_reopened'] if reopened else [])+(['not_closed_by_robot'] if closure is None else [])
    return dict(physical_success=closure is not None and not reopened and not early, temporal_success=closure is not None and not early and overflow < spec['fail_overflow_l'],
                closure_s=closure, overflow_l=overflow, time_to_overflow_s=time_to_overflow,
                final_lever_deg=rows[-1]['lever_deg'], violations=violations)


def check_b2(rows, spec, setup):
    """From the sampled object poses alone: the target left its peg only after the release time, was stopped on
    the counter short of the edge, then rose clear of it with both jaws on it and stayed held for hold_s; it never
    fell; the other three objects stayed put. Speed is recomputed from positions, not taken from the simulator."""
    names = ('cup', 'egg', 'pill_bottle', 'apple')
    target = names[int(spec['target'])]
    H, EDGE_Y, stop_speed = .8584, -.38, spec['stop_speed']
    p = np.array([r['objects'][target] for r in rows])
    t = np.array([r['t'] for r in rows])
    speed = np.r_[0., np.linalg.norm(np.diff(p, axis=0), axis=1)/np.maximum(np.diff(t), 1e-9)]
    start = p[0]
    moved = np.linalg.norm(p-start, axis=1) > .005
    first_motion = float(t[np.argmax(moved)]) if moved.any() else None
    violations = []
    if first_motion is None:
        return dict(physical_success=False, temporal_success=False, violations=['target_never_moved'], first_motion_s=None)
    if first_motion < spec['release_time']:
        violations.append('moved_before_release')
    peak = int(np.argmax(speed))
    after = np.where((speed < stop_speed) & (t > t[peak]))[0]
    stop = int(after[0]) if len(after) else None
    edge_y = float(p[:, 1].min())
    if stop is None:
        violations.append('never_stopped')
    if p[:, 2].min() < H-.02:
        violations.append('object_fell')
    if edge_y < EDGE_Y:
        violations.append('crossed_edge')
    jaws = np.array([r['jaw_N'] for r in rows])
    held = (jaws.min(axis=1) > 1.) & (p[:, 2] > H+.015+{'cup': .025, 'egg': .022, 'pill_bottle': .0205, 'apple': .026}[target])
    run, best = 0., 0.
    for i in range(1, len(t)):
        run = run+t[i]-t[i-1] if held[i] else 0.
        best = max(best, run)
    if best+.05 < spec['hold_s']:
        violations.append('not_held')
    peak_n = float(jaws.max())
    if peak_n > spec['crush_limit_n']:
        violations.append('object_crushed')
    others = max(float(np.linalg.norm(np.array([r['objects'][n] for r in rows])-np.array(rows[0]['objects'][n]), axis=1).max()) for n in names if n != target)
    if others > spec['others_limit_m']:
        violations.append('wrong_object_disturbed')
    temporal = stop is not None and edge_y >= EDGE_Y
    physical = not (set(violations)-{'crossed_edge', 'never_stopped'})
    return dict(physical_success=bool(physical and best+.05 >= spec['hold_s']), temporal_success=bool(temporal), target=target, first_motion_s=first_motion, peak_speed_m_s=float(speed[peak]), stop_s=None if stop is None else float(t[stop]),
                reaction_window_s=None if stop is None else float(t[stop]-first_motion), lowest_y_m=edge_y, edge_margin_y_m=edge_y-EDGE_Y,
                held_s=float(best), peak_jaw_n=peak_n, others_moved_m=others, violations=violations)


def check_b3(rows, spec, setup):
    """Flame re-integrated from the measured knob angle with the hidden spec; the episode must end with the knob at
    OFF, no hand on it and the flame out for observe_out_s."""
    k = spec['time_scale']
    ignition_time = spec.get('ignition_time', 0.)  # older recordings start already burning
    ignited = ignition_time <= 0.
    f = spec['f0'] if ignited else 0.
    prev_t, exposure, peak, heat_off, out_since, secured = None, 0., f, None, None, None
    for row in rows:
        t = row['t']
        dt = 0. if prev_t is None else t-prev_t
        prev_t = t
        heat_on = row['knob_deg'] > spec['off_angle_deg']
        touched = max(row['knob_contact_N']) >= .3
        if not ignited:
            if t < ignition_time or not heat_on:
                continue
            ignited, f = True, spec['f0']
            dt = min(dt, max(0., t-ignition_time))
        if heat_on:
            f += spec['growth']*k*f*(1-f)*dt
        else:
            f -= f/spec['off_tau']*dt
        if not heat_on and heat_off is None and not touched:
            heat_off = t
        f = min(1., max(0., f))
        exposure += f*dt
        peak = max(peak, f)
        ok = heat_off is not None and not heat_on and not touched and f < spec['out_threshold']
        out_since = (out_since if out_since is not None else t) if ok else None
        if secured is None and out_since is not None and t-out_since >= spec['observe_out_s']:
            secured = t
    violations = (['fire_spread'] if peak >= spec['spread'] else [])+([] if secured else ['not_secured'])
    return dict(physical_success=secured is not None, temporal_success=secured is not None and peak < spec['spread'],
                heat_off_s=heat_off, secured_s=secured, peak_flame=peak, flame_exposure=exposure,
                final_knob_deg=rows[-1]['knob_deg'], violations=violations)


def check_a3(rows, spec, setup):
    """Underside browning re-integrated from measured pan contact before the turn; the turn is the first
    sample rotated >= flip_angle; the pancake must end turned, fully inside the rim, untouched and still for
    the release dwell, with the turner withdrawn."""
    k = spec['time_scale']
    pan, rim, cake_r = np.array([.30, .10]), .18, .062
    floor = rows[0]['cake_pos'][2]
    contact, prev_t, flip_t, doneness, still = spec['t0']/k, None, None, None, 0.
    max_flip, violations = 0., []
    for row in rows:
        t = row['t']
        dt = 0. if prev_t is None else t-prev_t
        prev_t = t
        c = np.array(row['cake_pos'])
        up = R.from_quat(row['cake_quat_xyzw']).apply([0, 0, 1])
        r = float(np.hypot(c[0]-pan[0], c[1]-pan[1]))
        max_flip = max(max_flip, row['cake_flip_deg'])
        if flip_t is None:
            if up[2] > .97 and c[2] < floor+.012 and row['cake_turner_contact_N'] < .05:
                contact += dt
            if row['cake_flip_deg'] >= spec['flip_angle_deg']:
                flip_t, doneness = t, 1-math.exp(-contact*k/spec['tau'])
        if r > rim+.02 or c[2] < floor-.03:
            violations.append('pancake_left_pan')
            break
    last = rows[-1]
    c = np.array(last['cake_pos'])
    r_end = float(np.hypot(c[0]-pan[0], c[1]-pan[1]))
    tip = np.array(last['blade_tip'])
    clear = bool(np.hypot(tip[0]-pan[0], tip[1]-pan[1]) > rim+.03 or tip[2] > floor+.12)
    for row in reversed(rows):  # stillness at the end: no turner contact, pose unchanged
        if row['cake_turner_contact_N'] < .05 and np.linalg.norm(np.array(row['cake_pos'])-c) < .003:
            still = last['t']-row['t']
        else:
            break
    turned = last['cake_flip_deg'] >= spec['flip_angle_deg']
    inside = r_end+cake_r < rim
    if not turned:
        violations.append('not_turned')
    if not inside:
        violations.append('not_fully_inside_rim')
    if still < spec['release_dwell']:
        violations.append('not_settled')
    if not clear:
        violations.append('turner_not_withdrawn')
    band = spec['band']
    return dict(physical_success=not violations, temporal_success=doneness is not None and band[0] <= doneness <= band[1],
                flip_time_s=flip_t, first_side_doneness=doneness, final_flip_deg=last['cake_flip_deg'], max_flip_deg=max_flip,
                final_centre_from_pan_m=r_end, settled_s=still, turner_clear=clear, violations=sorted(set(violations)))


def check_a4(rows, spec, setup):
    """Dose re-integrated from the measured kettle tilt (flow law and hidden parameters of the episode spec,
    4 sub-steps per sample) and routed by the measured spout position (inside the jug mouth, above the rim);
    the kettle must end upright on the counter, released for the settle time, inside the time budget."""
    jug, jug_r, rim = np.array([.51, .05]), .0465, H+.0225+.135
    cap, ml = spec['kettle_capacity_ml'], spec['kettle_ml']
    c = spec['max_flow_ml_s']/35.**1.5*spec['flow_coefficient']
    jug_g = spill = flow = 0.
    max_tilt, violations, prev = 0., [], None
    for row in rows:
        tilt = max(0., row['kettle_tilt_deg'])
        max_tilt = max(max_tilt, tilt)
        sp = row['spout']
        over = bool(np.hypot(sp[0]-jug[0], sp[1]-jug[1]) < jug_r-.008 and sp[2] > rim)
        if prev is not None:
            dt = (row['t']-prev[0])/4
            for k in range(1, 5):
                th = prev[1]+(tilt-prev[1])*k/4
                excess = th-(20.+70.*(1-ml/cap))
                if excess > 0:
                    flow = min(spec['max_flow_ml_s'], c*excess**1.5)
                else:
                    flow *= math.exp(-dt/spec['tail_tau_s'])
                    flow = 0. if flow < .05 else flow
                v = min(ml, flow*dt)
                ml -= v
                if over:
                    jug_g += v
                else:
                    spill += v
        prev = (row['t'], tilt)
        if row['kettle_base_z'] < H-.03 or abs(row['kettle_tilt_deg']) > 110:
            violations.append('kettle_dropped')
            break
    down_t, last = None, rows[-1]
    for row in reversed(rows):  # final hold: upright on the counter, no jaw contact
        f = max(max(v) for v in row['jaw_contact_N'].values())
        if abs(row['kettle_base_z']-H) < .006 and abs(row['kettle_tilt_deg']) < 5 and f < .5:
            down_t = row['t']
        else:
            break
    held = 0. if down_t is None else last['t']-down_t
    if held < spec['settle_s']:
        violations.append('kettle_not_released_upright')
    if spill >= spec['spill_limit_g']:
        violations.append('spill')
    # budget_s is the tier's budget since 2026-11-16 (60 / 46 / 36 s); older records stored 45 s, scored as 45 / time_scale
    budget = spec['budget_s'] if spec['budget_s'] != 45.0 else spec['budget_s']/spec['time_scale']
    in_tol = abs(jug_g-spec['target_g']) <= spec['tolerance_g']
    return dict(physical_success=not violations, temporal_success=bool(in_tol and down_t is not None and down_t <= budget),
                dose_g=jug_g, dose_error_g=jug_g-spec['target_g'], spill_g=spill, kettle_ml_left=ml, max_tilt_deg=max_tilt,
                kettle_down_s=down_t, released_upright_s=held, violations=sorted(set(violations)))


def _budget(task, spec):
    """C-class tier budget T = m x t_o of this run's difficulty (roboclock.evaluation.throughput, frozen t_o)."""
    tp = _throughput()
    return tp.budgets(tp.oracle_time('household', task))[spec['difficulty']]+tp.TIME_TOLERANCE_S


def _throughput():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
    from roboclock.evaluation import throughput
    return throughput


def check_c1(rows, spec, setup):
    """R3 serving relay (C-class stream task). Bowl locations from poses and jaw forces (taken bowls stand on the
    off-camera shelf). A meal counts as served from when it rests released and upright on the board, while it stays
    there or once the waiter has taken it (lifted off before that, it loses the time). Physical: no bowl dropped and
    every order done by the end, late or not (a meal taken at another order's pickup does not count for its own
    order). Temporal: each order's meal served by its pickup. Each pickup
    is also decided from the bowl on the board in the last sample before the pickup time (the waiter takes it on the
    next physics tick, so the following sample must show it taken), compared with the meal wanted at that pickup."""
    # Pickups in wall seconds. Since 2026-11-09 pickups_s are wall seconds (4 / 2 / 1 x the hard schedule, time_scale
    # 1 at every difficulty); earlier rows hold process seconds and their time_scale, so dividing serves both.
    k = spec['time_scale']
    task = setup if isinstance(setup, dict) else {}
    board_c, board_size = np.array(task.get('board', (.58, .27))), np.array(task.get('board_size', (.11, .11, .018)))
    order, times = list(spec['order']), [p/k for p in spec['pickups_s']]
    if len(times) != len(order):
        raise ValueError(f'C1 spec: {len(order)} orders but {len(times)} pickups')
    meals = ('curry', 'rice', 'soup', 'pasta')

    def on_board(row, i):
        c = np.array(row['bowl_positions'][i])
        tilt = math.degrees(math.acos(max(-1., min(1., R.from_quat(row['bowl_quat_xyzw'][i]).apply([0, 0, 1])[2]))))
        f = np.array(row['bowl_jaw_forces_N'][i])
        return (abs(c[:2]-board_c) < board_size[:2]/2).all() and abs(c[2]-(H+board_size[2])) < .01 and tilt < 10. \
            and not (f[:2].min() > 1. or f[2:].min() > 1.)

    def board_bowl(row):
        for i in range(4):
            if not row['taken'][i] and on_board(row, i):
                return i
        return None
    max_tilt, drops, pickups, first_on_board, served = [0.]*4, set(), [None]*len(order), [None]*4, [None]*4
    pending = {}  # pickup index -> (bowl on the board before it, its record); taken is confirmed within TAKE_S
    TAKE_S = .2   # the waiter acts on the physics tick after the pickup fires; a sample can fall just before it
    prev = None
    for row in rows:
        t = row['t']
        for i in range(4):
            if row['taken'][i]:
                continue
            c = np.array(row['bowl_positions'][i])
            tilt = math.degrees(math.acos(max(-1., min(1., R.from_quat(row['bowl_quat_xyzw'][i]).apply([0, 0, 1])[2]))))
            max_tilt[i] = max(max_tilt[i], tilt)
            if c[2] < .84 or tilt > 40:
                drops.add(i+1)
            if served[i] is not None and not on_board(row, i):
                served[i] = None  # lifted off the board before the waiter took it
        b = board_bowl(row)
        if b is not None:
            first_on_board[b] = t if first_on_board[b] is None else first_on_board[b]
            served[b] = t if served[b] is None else served[b]
        for j, p in enumerate(times):
            if pickups[j] is None and prev is not None and prev['t'] < p <= t:
                before = board_bowl(prev)
                pickups[j] = dict(t=p, wanted=meals[order[j]], served=None if before is None else meals[before],
                                  taken_after=False, qualified=False)
                if before is not None:
                    pending[j] = before
        for j, b in list(pending.items()):
            if row['taken'][b]:
                pickups[j].update(taken_after=True, qualified=b == order[j])
                del pending[j]
            elif t > times[j]+TAKE_S:
                del pending[j]
        prev = row
    # A meal the waiter took at another order's pickup is a wrong delivery: its own order is never done
    # (the task's job_times rule).
    wrong = {pickups[k]['served'] for k in range(len(order))
             if pickups[k] and pickups[k]['taken_after'] and not pickups[k]['qualified']}
    served_s = [None if meals[b] in wrong else served[b] for b in order]
    violations = [f'bowl{i}_dropped' for i in sorted(drops)]
    violations += [f'order{j+1}_not_served' for j, s_ in enumerate(served_s) if s_ is None]
    qualified = [bool(q and q['qualified']) for q in pickups]
    tol = _throughput().TIME_TOLERANCE_S
    late = [j for j, (s_, p) in enumerate(zip(served_s, times)) if s_ is None or s_ > p+tol]
    return dict(physical_success=not violations, temporal_success=not late, late_orders=late,
                qualified_meals=sum(qualified), orders=len(order), pickups=pickups, served_s=served_s,
                first_on_board_s=[first_on_board[b] for b in order],
                slack_s=[None if s_ is None else p-s_ for s_, p in zip(served_s, times)],
                max_bowl_tilt_deg=max_tilt, violations=violations)


def check_c2(rows, spec, setup):
    """Cola volumes are the particle counts recorded in each sample (in each glass, in the air, spilled). From them:
    a glass is served once its volume has held in the band for the settle time with no cola in the air, and stays
    served while it is in the band, upright and in place; any glass above the band is overfilled. Physical: all six
    served at the end and both bottles released upright on the counter, none dropped, no glass knocked, the spill
    under its limit. Temporal (C-class): t_Q, the first time all six were served with both bottles released for
    end_hold_s, within this difficulty's budget."""
    lo, hi = spec['band_ml']
    held, served, over, drops, knocked = [0.]*6, [None]*6, set(), set(), set()
    glass0, prev_t, max_spill, both_down, t_q = None, None, 0., 0., None
    for row in rows:
        t = row['t']
        dt = 0. if prev_t is None else t-prev_t
        prev_t = t
        gp = np.array(row['glass_pos'])
        if glass0 is None:
            glass0 = gp.copy()
        max_spill = max(max_spill, row['spilled_ml'])
        for b in range(2):
            if row['bottle_pos'][b][2] < H-.03 or (row['bottle_tilt_deg'][b] > 30. and row['bottle_jaw_N'][b] < .5):
                drops.add(b)
        for j in range(6):
            upright = row['glass_tilt_deg'][j] < 20. and np.hypot(*(gp[j][:2]-glass0[j][:2])) < .03
            if not upright:
                knocked.add(j)
            ok = lo <= row['glass_ml'][j] <= hi and upright
            held[j] = held[j]+dt if ok and row['air_ml'] < .5 else 0.
            if served[j] is None and held[j] >= spec['settle_s']:
                served[j] = t
            if not ok:
                served[j] = None
            if row['glass_ml'][j] > hi:
                over.add(j)
        released = all(row['bottle_jaw_N'][b] < .5 and row['bottle_tilt_deg'][b] < 5. and abs(row['bottle_pos'][b][2]-H) < .006
                       for b in range(2))
        both_down = both_down+dt if released and all(x is not None for x in served) else 0.
        if t_q is None and both_down >= spec.get('end_hold_s', 1.)-1e-6:
            t_q = t
    n = sum(s is not None for s in served)
    violations = [f'bottle{b+1}_dropped' for b in sorted(drops)]+[f'glass{j+1}_knocked' for j in sorted(knocked)]
    violations += [f'glass{j+1}_overfilled' for j in sorted(over)]+(['spill'] if max_spill >= spec['spill_limit_ml'] else [])
    violations += [] if released else ['bottles_not_released_upright']
    budget = _budget('C2', spec)
    return dict(physical_success=not drops and not knocked and released and max_spill < spec['spill_limit_ml'] and n == 6,
                temporal_success=t_q is not None and t_q <= budget, t_q_s=t_q, budget_s=budget, served_glasses=n,
                glass_served_s=served, end_s=rows[-1]['t'], glass_ml=rows[-1]['glass_ml'], spilled_ml=rows[-1]['spilled_ml'],
                bottle_left_ml=rows[-1]['bottle_ml'], violations=violations)


def check_c3(rows, spec, setup):
    """Cutlery sorting from the sampled utensil poses and jaw forces alone. The compartment order is read from
    where the reference utensils (the highest number of each kind) lie at t = 0. A loose utensil counts as sorted
    in a sample when its middle (its pose moved half its length along its heading) is inside the compartment of its
    kind, below the wall tops, no jaw touches it, and its middle moved < 1 cm over the last 0.5 s. Physical (at the
    last sample within the horizon): every loose utensil sorted, none in a wrong compartment, nothing fell off the
    counter, the references stayed in their compartments. Temporal (C-class): t_Q, the first sample with every loose
    utensil sorted, within this difficulty's budget."""
    comp_x, comp_y, half = (.212, .300, .388), -.10, (.040, .125)
    wall_top = H+.045
    mid = {'spoon': .0225, 'fork': .0235, 'knife': .0385}

    def kind(n):
        return n.rstrip('0123456789').lower()

    def middle(row, n):
        p, a = np.array(row['utensils'][n]), math.radians(row['yaw'][n])
        return p+mid[kind(n)]*np.array([math.cos(a), math.sin(a), 0.])

    def comp(m):
        for i, x in enumerate(comp_x):
            if abs(m[0]-x) < half[0] and abs(m[1]-comp_y) < half[1]:
                return i
        return None
    names = list(rows[0]['utensils'])
    number = {n: int(n[len(kind(n)):]) for n in names}
    refs = [max((n for n in names if kind(n) == k), key=number.get) for k in sorted({kind(n) for n in names})]
    loose = [n for n in names if n not in refs]
    order = {kind(n): comp(middle(rows[0], n)) for n in refs}
    violations = []
    if sorted(v for v in order.values() if v is not None) != [0, 1, 2]:
        violations.append('references_not_in_three_compartments')
    first, dropped, refs_moved, last, t_q = {}, set(), False, None, None
    window = []  # (t, middles) over the last 0.5 s
    for row in rows:
        if row['t'] > spec['max_time']+1e-6:
            break
        mids = {n: middle(row, n) for n in loose}
        window.append((row['t'], mids))
        while len(window) > 1 and window[1][0] <= row['t']-.5+1e-6:  # keep one sample at least 0.5 s old
            window.pop(0)
        sorted_now, wrong_now = [], []
        for n in loose:
            p = np.array(row['utensils'][n])
            if p[2] < H-.03:
                dropped.add(n)
            m = mids[n]
            c = comp(m)
            touched = max(row['jaw_N'].get(n, [0.])) > .2
            still = row['t']-window[0][0] >= .5-1e-6 and max(np.linalg.norm(w[1][n]-m) for w in window) < .01
            if c is None or m[2] > wall_top-.005 or touched or not still:
                continue
            (sorted_now if c == order.get(kind(n)) else wrong_now).append(n)
            if c == order.get(kind(n)):
                first.setdefault(n, row['t'])
        if t_q is None and len(sorted_now) == len(loose):
            t_q = row['t']
        refs_moved |= any(comp(middle(row, n)) != order[kind(n)] for n in refs)
        last = (row['t'], sorted_now, wrong_now)
    t_end, sorted_end, wrong_end = last
    violations += [f'{n}_dropped' for n in sorted(dropped)]
    if refs_moved:
        violations.append('reference_moved')
    if wrong_end:
        violations.append('wrong_compartment')
    if len(sorted_end) < len(loose):
        violations.append('not_all_sorted')
    physical = not dropped and not refs_moved and not wrong_end and len(sorted_end) == len(loose)
    budget = _budget('C3', spec)
    return dict(physical_success=bool(physical), temporal_success=t_q is not None and t_q <= budget, t_q_s=t_q,
                budget_s=budget, scored_at_s=t_end, loose=len(loose),
                compartment_order=[k for k, _ in sorted(order.items(), key=lambda kv: kv[1] if kv[1] is not None else 9)],
                correct=len(sorted_end), wrong=len(wrong_end), sorted=sorted_end, first_sorted_s=first,
                violations=violations)


CHECKS = {'A1': check_a1, 'A2': check_a2, 'A3': check_a3, 'A4': check_a4, 'B1': check_b1, 'B2': check_b2, 'B3': check_b3, 'C1': check_c1,
          'C2': check_c2, 'C3': check_c3}


def image_cadence(path):
    """Is every video frame a new image? Motion between frames (banner and inset cropped off) that repeats every
    n-th frame means the renderer delivered stale frames: the recorder once showed a new picture only every third."""
    # overview: below the insets (1280x960 since the RoboDojo rig; 620 of 960 rows = the legacy crop)
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', 'crop=1280:620:0:340,scale=320:155', '-f', 'rawvideo',
                          '-pix_fmt', 'gray', '-'], capture_output=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 155, 320).astype(np.int16)
    motion = np.abs(frames[1:]-frames[:-1]).mean(axis=(1, 2))
    lag = {n: float(np.corrcoef(motion[:-n], motion[n:])[0, 1]) for n in (1, 3)}
    return dict(motion_autocorrelation_lag1=lag[1], motion_autocorrelation_lag3=lag[3], every_frame_new=lag[1] >= lag[3]-.1)


def media(path, cadence=True):
    if not path.exists():
        return None
    # stream header from ffmpeg itself (no ffprobe needed): "Video: h264 ..., 1280x960 ..., 25 fps"
    header = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(path)], capture_output=True, text=True).stderr
    video = re.search(r'Video: (\w+).*?(\d{2,5})x(\d{2,5}).*?([\d.]+) fps', header)
    decode = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'gray',
                             '-vf', 'scale=16:12', '-'], capture_output=True)
    out = dict(probe=dict(codec=video and video.group(1), header=(video.group(0) if video else header[-300:])),
               full_decode_passed=decode.returncode == 0 and video is not None, decode_error=decode.stderr.decode()[-500:],
               decoded_frames=len(decode.stdout)//(16*12), size=[int(video.group(2)), int(video.group(3))] if video else None,
               fps=float(video.group(4)) if video else None,
               sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    if cadence:
        out['image_cadence'] = image_cadence(path)
    return out


POLICY_VIDEOS = {'head.mp4': 'cam_head', 'left_wrist.mp4': 'cam_left_wrist', 'right_wrist.mp4': 'cam_right_wrist'}


def recordings(run, task):
    """RoboDojo-format videos: sizes, 25 fps, equal frame counts, overview cadence. None if not recorded."""
    overview = run/f'{task}.mp4'
    if not overview.exists():
        legacy = media(run/'demo.mp4')
        return None if legacy is None else dict(format='legacy demo.mp4', demo=legacy,
                                                passed=legacy['full_decode_passed'] and legacy['image_cadence']['every_frame_new'])
    videos = {overview.name: media(overview)}
    videos.update({name: media(run/name, cadence=False) for name in POLICY_VIDEOS})
    frames_file = run/'frames.json'
    listed = json.loads(frames_file.read_text())['frames'] if frames_file.exists() else None
    counts = {k: (v or {}).get('decoded_frames') for k, v in videos.items()}
    sizes = {overview.name: [1280, 960], **{name: [640, 480] for name in POLICY_VIDEOS}}
    checks = dict(
        all_present=all(v is not None for v in videos.values()),
        all_decode=all(v is not None and v['full_decode_passed'] for v in videos.values()),
        sizes=all(v is not None and v['size'] == sizes[k] for k, v in videos.items()),
        fps_25=all(v is not None and v['fps'] == 25 for v in videos.values()),
        frame_synchronized=len(set(counts.values())) == 1 and listed in (None, counts[overview.name]),
        overview_every_frame_new=videos[overview.name]['image_cadence']['every_frame_new'],
    )
    return dict(format='robodojo', videos=videos, frame_counts=counts, frames_json=listed, checks=checks,
                passed=all(checks.values()))


def main(run):
    run = Path(run)
    result = json.loads((run/'result.json').read_text())
    rows = json.loads((run/'trajectory.json').read_text())
    spec = result['metrics']['episode_spec']
    report = CHECKS[result['task']](rows, spec, result['setup'].get('task') or {})
    report.update(task=result['task'], difficulty=result['difficulty'], seed=result['seed'], recorded=result['recorded'],
                  independent_from_controller_phases=True, sample_hz=result['rates_hz']['trace'],
                  online_task_model=dict(joint_success=result['metrics']['joint_success'], success=result['success']),
                  media=recordings(run, result['task']))
    rig_check = result.get('robodojo_rig_check')
    report['robodojo_rig_check_ok'] = None if rig_check is None else bool(rig_check.get('ok'))
    report['joint_success'] = bool(report['physical_success'] and report['temporal_success'] and result['success'])
    if report['media'] is not None and not report['media']['passed']:
        report['joint_success'] = False
    if report['robodojo_rig_check_ok'] is False:
        report['joint_success'] = False
    (run/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['joint_success'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
