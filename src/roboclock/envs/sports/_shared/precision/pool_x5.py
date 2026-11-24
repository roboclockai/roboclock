"""Measured-tip trajectories for the fixed native X5 pool baseline.

Privileged scene feedback is used only by this demonstration controller.
Objects are moved exclusively by native articulation drives and contact.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .pool import ClearanceOracle
from .pool_layout import PoolLayout


def pose_rotation(pose):
    return Rotation.from_quat(np.roll(pose["orientation"], -1)).as_matrix()


def segment_distance(point, start, end):
    delta = end - start
    fraction = np.clip(np.dot(point - start, delta) / max(np.dot(delta, delta), 1e-12), 0., 1.)
    return np.linalg.norm(point - (start + fraction * delta))


def cue_stroke_travel(radius, planar_gap, direction):
    """Convert the planar backswing to axial travel, including tilted strokes."""
    planar_fraction = np.linalg.norm(np.asarray(direction)[:2])
    return (radius + planar_gap) / planar_fraction - radius + .015


def cue_clears_jaws(tip, direction, travel=.095, layout=None, cue_rotation=None, grip_x=0.):
    """Conservative shaft-envelope sweep against the authored end cushions."""
    layout = layout or PoolLayout()
    padding = .011  # 9 mm shaft radius plus 2 mm approach clearance.
    for advance in np.linspace(0, travel, 9):
        end = tip + advance * direction
        start = end - .307 * direction
        delta = end - start
        for lower, upper in layout.jaw_boxes():
            if cue_rotation is not None:
                corners = np.array([[x, y, z] for x in (grip_x - .0375, grip_x + .0375)
                                    for y in (-.040, .040) for z in (-.012, .073)])
                corners = corners @ cue_rotation.T + end - .177 * direction
                if np.all(corners.max(axis=0) + .002 >= lower) and np.all(corners.min(axis=0) - .002 <= upper):
                    return False
            lower, upper = lower - padding, upper + padding
            t0, t1 = 0., 1.
            for axis in range(3):
                if abs(delta[axis]) < 1e-10:
                    if not lower[axis] <= start[axis] <= upper[axis]:
                        t0, t1 = 1., 0.
                        break
                else:
                    a, z = sorted(((lower[axis] - start[axis]) / delta[axis],
                                   (upper[axis] - start[axis]) / delta[axis]))
                    t0, t1 = max(t0, a), min(t1, z)
            if t0 <= t1:
                return False
    return True


def move_cue_arm(rig, side, position, rotation, grip, dt, speed=.25, joint_speed=2.5):
    """Rate-limit consecutive commands, preserving the native drive dynamics."""
    p, _ = rig.pose(side)
    previous = rig.targets.get(side, p)
    delta = np.asarray(position) - previous
    desired = previous + delta * min(1., speed * dt / max(1e-9, np.linalg.norm(delta)))
    rig.targets[side] = desired.copy()
    actual = rig.robot.get_joint_positions()
    ids = rig.arm_ids[side]
    q, error = rig.kin[side].solve(desired, rotation, seed=actual[ids])
    delta = q - rig.command[ids]
    delta *= min(1., joint_speed * dt / max(1e-9, np.max(np.abs(delta))))
    rig.command[ids] += delta
    rig.grip(side, grip)
    return float(np.linalg.norm(np.asarray(position) - p)), error


class X5ClearanceOracle(ClearanceOracle):
    """Rebase each stroke at the measured wrist, with whole-path IK checks."""

    def __init__(self, b, task):
        super().__init__(b, task)
        self.stable_since = None
        self.diagnostics = []
        self.stroke_origin = None
        self.path_checks = []
        self.aim_bias = np.zeros(3)
        self.gap = b.config.get("pool", {}).get("backswing", .080)
        self.travel = self.gap + .015

    def measured_tool(self):
        s = self.b.last_state
        hand, cue = s["hands"]["main"], s["objects"]["cue"]
        hr, cr = pose_rotation(hand), pose_rotation(cue)
        hp, cp = np.asarray(hand["position"]), np.asarray(cue["position"])
        return hp, hr, cp + cr @ np.array([.177, 0, 0]), cr

    def transition(self, phase):
        super().transition(phase)
        self.b.cue_stroking = phase == "stroke"
        self.stable_since = None
        if phase == "aim":
            self.aim_bias = np.zeros(3)
        self.diagnostics.append(dict(time=self.b.time, phase=phase))

    def path_feasible(self, start, end, rotation):
        """Test all stroke waypoints on one continuous native IK branch."""
        rig = self.b.rig
        kin = rig.kin[self.b.active_side]
        seed = self.b.robot.get_joint_positions()[rig.arm_ids[self.b.active_side]]
        saved = kin.last.copy()
        errors = []
        try:
            for fraction in np.linspace(0, 1, 7):
                target = start + fraction * (end - start)
                seed, _ = kin.solve(target, rotation, seed=seed)
                p, r = kin.fk(seed)
                errors.append((float(np.linalg.norm(p - target)),
                               float(Rotation.from_matrix(r @ rotation.T).magnitude())))
        finally:
            kin.last = saved
        valid = all(p < .003 and a < .025 for p, a in errors)
        self.path_checks.append(dict(time=self.b.time, valid=valid,
                                    start=start.tolist(), end=end.tolist(), errors=errors))
        return valid

    def plan(self):
        b, t = self.b, self.task
        layout = getattr(b, "pool_layout", PoolLayout())
        hp, hr, _, cr = self.measured_tool()
        cue = np.asarray(b.last_state["objects"]["cue"]["position"])
        ball = np.asarray(b.last_state["objects"]["cue_ball"]["position"])
        self.offset = hr.T @ (cue - hp)
        self.tool_rotation = hr.T @ cr
        if self.excluded_at is not None and np.linalg.norm(ball - self.excluded_at) > .04:
            self.excluded_target = None
        if ball[2] < .74:
            self.diagnostics.append(dict(time=b.time, phase="white_ball_unplayable"))
            return False
        choices = []
        for name in b.pool_targets:
            if b.config.get("pool", {}).get("spot_clearance"):
                remaining = [n for n in b.pool_targets if n not in t.pots]
                if remaining and name != remaining[0]:
                    continue
            if name in t.pots or name == self.excluded_target:
                continue
            obj = np.asarray(b.last_state["objects"][name]["position"])
            if obj[2] < .74:
                continue
            for pocket_index, pocket_center in enumerate(b.pockets):
                assigned = b.config.get("pool", {}).get("assigned_pockets", {}).get(name)
                if assigned is not None and assigned != pocket_index:
                    continue
                # Aim through the cushion throat, not the back of the pocket.
                # The bed ends at x=1.17 / .23; oblique rays aimed at x=1.25
                # can strike the near jaw despite pointing at the pocket center.
                pocket = np.asarray(pocket_center)
                if layout.scale != 1. and not layout.ball_path_clear(obj[:2], pocket, b.ball_radius):
                    continue
                outgoing = pocket - obj[:2]
                pot_distance = np.linalg.norm(outgoing)
                outgoing /= max(pot_distance, 1e-9)
                ghost = obj[:2] - 2 * b.ball_radius * outgoing
                direction = ghost - ball[:2]
                distance = np.linalg.norm(direction)
                direction /= max(distance, 1e-9)
                cosine = np.dot(direction, outgoing)
                if cosine > .3 and (layout.scale == 1. or layout.ball_path_clear(ball[:2], ghost, b.ball_radius)):
                    speed = np.clip((.32 + .65 * pot_distance + .2 * distance) / max(cosine, .6), .35, .85)
                    blockers = 0
                    for other in b.pool_targets:
                        if other == name or other in t.pots:
                            continue
                        other_pos = np.asarray(b.last_state["objects"][other]["position"])
                        if other_pos[2] < .74:
                            continue
                        if (segment_distance(other_pos[:2], ball[:2], ghost) < 2 * b.ball_radius + .003
                                or segment_distance(other_pos[:2], obj[:2], pocket) < 2 * b.ball_radius + .003):
                            blockers += 1
                    choices.append((cosine - .15 * distance - 2 * blockers, name, direction, speed, "pot"))
        if len(t.pots) == len(b.pool_targets):
            return False
        # A physical cushion stroke can recover a difficult white-ball position.
        # Try closed cushion segments, avoiding the four pocket mouths.
        for point in ([] if (b.config.get("pool", {}).get("spot_probe", False) or b.config.get("pool", {}).get("spot_clearance", False)) else
                      ([1.18, .05], [1.18, -.30], [.70, .355], [.70, -.355])):
            direction = layout.point(point) - ball[:2]
            direction /= max(np.linalg.norm(direction), 1e-9)
            choices.append((-2., "cue_ball", direction, .70, "reposition"))
        pitches = b.config.get("pool", {}).get("cue_pitch_candidates",
                  [b.config.get("pool", {}).get("cue_pitch_degrees", 10.)])
        rolls = b.config.get("pool", {}).get("cue_roll_candidates", [0.])
        candidates = [(choice, pitch, roll) for choice in sorted(choices, key=lambda c: c[0], reverse=True)
                      for pitch in pitches for roll in rolls]
        for (_, name, heading, speed, mode), pitch, roll in candidates:
            trim = b.config.get("pool", {}).get("heading_trim_degrees", 0.)
            yaw = np.arctan2(heading[1], heading[0]) + np.deg2rad(trim)
            cue_r = Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("y", pitch, degrees=True).as_matrix() @ Rotation.from_euler("x", roll, degrees=True).as_matrix()
            direction = cue_r[:, 0]
            rotation = cue_r @ self.tool_rotation.T
            h = direction[:2] / np.linalg.norm(direction[:2])
            tip = ball.copy()
            tip[:2] -= h * (b.ball_radius + self.gap)
            # Center-height impact after the 80 mm acceleration gap.
            tip[2] -= (b.ball_radius + self.gap) * direction[2] / np.linalg.norm(direction[:2])
            tip[2] += b.config.get("pool", {}).get("hit_height_offset", 0.)
            travel = cue_stroke_travel(b.ball_radius, self.gap, direction)
            if not cue_clears_jaws(tip, direction, travel=travel, layout=layout,
                                   cue_rotation=cue_r if layout.scale != 1. else None,
                                   grip_x=getattr(b, "cue_grip_x", 0.)):
                continue
            goal = tip - .177 * direction - rotation @ self.offset
            if not self.path_feasible(goal, goal + direction * travel, rotation):
                continue
            high = goal.copy()
            high[2] = .99
            if not self.path_feasible(high, goal, rotation):
                continue
            turn = hp.copy()
            turn[2] = .99
            if not self.path_feasible(turn, high, rotation):
                continue
            self.travel = travel
            self.object, self.mode, self.stroke_speed = name, mode, b.config.get("pool", {}).get("probe_stroke_speed", min(1.0, speed * 1.8))
            self.direction, self.rotation = direction, rotation
            self.desired_cue_rotation = cue_r
            self.tip_goal = tip
            self.goal = self.nominal_goal = goal
            self.fixed_aim_rotation = rotation.copy()
            self.planned_aim_rotation = rotation.copy()
            self.turn_goal = hp.copy()
            self.turn_goal[2] = .99
            self.shot_count = len(t.shots)
            self.stable_since = None
            self.diagnostics.append(dict(time=b.time, phase="planned_stroke", target=name,
                                         mode=mode, goal=goal.tolist(), direction=direction.tolist()))
            return True
        self.diagnostics.append(dict(time=b.time, phase="no_reachable_stroke"))
        return False

    def act(self):
        if (self.b.config.get("pool", {}).get("dual_cue_exchange", False)
                and not self.b.config.get("pool", {}).get("spot_clearance", False)):
            from . import pool_exchange
            if hasattr(self, "right_park"):
                self.b.move("right", self.right_park,
                            Rotation.from_euler("y", 180, degrees=True).as_matrix(), grip=.04, speed=.15)
            if self.phase.startswith("exchange_"):
                return pool_exchange.act(self)
            if (self.phase == "finished" and len(self.task.pots) < len(self.b.pool_targets)
                    and not getattr(self, "exchange_attempted", False)
                    and self.b.time < self.b.config["workflow"]["pool_horizon"] - 12.):
                pool_exchange.begin(self)
                return
        if (self.b.config.get("pool", {}).get("spot_probe", False)
                and self.phase == "wait" and len(self.task.shots) > 0):
            self.transition("finished")
        if self.phase == "finished":
            # The last native position command remains active during the full
            # evaluation budget; do not repeatedly solve an unchanged IK pose.
            return
        b, t = self.b, self.task
        if self.phase == "separate":
            b.move("main", self.contact_hold, self.rotation, grip=0., speed=.10)
            hp, _, tip, _ = self.measured_tool()
            gap = np.linalg.norm(np.asarray(b.last_state["objects"]["cue_ball"]["position"]) - tip)
            elapsed = b.time - self.start
            if (elapsed > .025 and gap > b.ball_radius + .006) or elapsed > .10:
                self.goal = hp.copy()
                self.transition("retract")
            return
        if self.phase == "wait" and t.ready:
            moving = [b.last_state["objects"][n] for n in ["cue_ball", *b.pool_targets]
                      if n not in t.pots]
            if max(np.linalg.norm(o["velocity"][:2]) for o in moving) > .0015:
                hold = self.goal.copy()
                hold[2] = .99
                b.move("main", hold, self.rotation, grip=0., speed=.20)
                if min(b.last_state["objects"]["cue"]["forces"][:2]) > .025:
                    self.last_grip = b.time
                return
        if self.phase not in ("aim", "stroke"):
            return super().act()
        hp, hr, tip, cr = self.measured_tool()
        elapsed = b.time - self.start
        held = min(b.last_state["objects"]["cue"]["forces"][:2]) > .025
        if held:
            self.last_grip = b.time
        elif b.time - self.last_grip > .15:
            self.goal = hp.copy()
            self.transition("raise")
            return
        if self.phase == "aim":
            # Position feedback is re-anchored to the measured TCP every tick.
            # Start from the transported orientation. Near the ready pose,
            # permit only a bounded measured-tool correction (no unbounded
            # integral chase of grasp compliance toward a joint limit).
            error = self.tip_goal - tip
            if np.linalg.norm(error) < b.config.get("pool", {}).get("aim_orientation_capture", .004):
                corrected = self.desired_cue_rotation @ cr.T @ hr
                correction = Rotation.from_matrix(corrected @ self.planned_aim_rotation.T).as_rotvec()
                correction *= min(1., .35 / max(np.linalg.norm(correction), 1e-9))
                corrected = Rotation.from_rotvec(correction).as_matrix() @ self.planned_aim_rotation
                step = Rotation.from_matrix(corrected @ self.fixed_aim_rotation.T).as_rotvec()
                dt = b.config["sim"]["physics_dt"] * b.config["sim"]["control_decimation"]
                step *= min(1., .24 * dt / max(np.linalg.norm(step), 1e-9))
                self.fixed_aim_rotation = Rotation.from_rotvec(step).as_matrix() @ self.fixed_aim_rotation
            self.rotation = self.fixed_aim_rotation.copy()
            if np.linalg.norm(error) < .004:
                dt = b.config["sim"]["physics_dt"] * b.config["sim"]["control_decimation"]
                self.aim_bias = np.clip(self.aim_bias + 2.0 * dt * error, -.003, .003)
            self.goal = hp + np.clip(error * .6, -.012, .012) + self.aim_bias
            b.move("main", self.goal, self.rotation, grip=0., speed=.10)
            axis = cr[:, 0]
            aligned = np.linalg.norm(error) < .0005 and np.dot(axis, self.direction) > .99998
            self.stable_since = (b.time if self.stable_since is None else self.stable_since) if aligned else None
            if self.stable_since is not None and b.time - self.stable_since > .20 and t.ready:
                # Start from the measured pose, discarding pending aim targets.
                # Hold the actual orientation to avoid a sudden wrist twist.
                travel = self.travel
                end = hp + self.direction * travel
                if self.path_feasible(hp, end, hr):
                    self.rotation = hr.copy()
                    self.stroke_origin = hp.copy()
                    self.stroke_tip_origin = tip.copy()
                    self.stroke_end = end
                    self.goal = hp.copy()
                    b.rig.targets[b.active_side] = hp.copy()
                    self.shot_count = len(t.shots)
                    self.transition("stroke")
                else:
                    self.goal = hp.copy()
                    self.excluded_target = self.object
                    self.excluded_at = np.asarray(b.last_state["objects"]["cue_ball"]["position"]).copy()
                    self.replans += 1
                    self.transition("raise")
            elif elapsed > b.config.get("pool", {}).get("aim_timeout", 5.):
                self.goal = hp.copy()
                self.excluded_target = self.object
                self.excluded_at = np.asarray(b.last_state["objects"]["cue_ball"]["position"]).copy()
                self.replans += 1
                self.transition("raise")
        else:
            # Smooth acceleration, followed by a bounded straight stroke.
            speed = min(self.stroke_speed, .12 + 4.0 * elapsed)
            along = np.dot(tip - self.stroke_tip_origin, self.direction)
            lateral = tip - (self.stroke_tip_origin + along * self.direction)
            advance = min(b.config.get("pool", {}).get("stroke_lookahead", .030),
                          max(0., self.travel - along))
            target = hp + self.direction * advance - np.clip(lateral * 2.0, -.006, .006)
            self.goal = target.copy()
            b.move("main", target, self.rotation, grip=0., speed=speed)
            if len(t.shots) > self.shot_count:
                self.contact_hold = hp.copy()
                self.goal = hp.copy()
                self.rotation = hr.copy()
                b.rig.targets[b.active_side] = hp.copy()
                self.transition("separate")
            elif along > self.travel - .001 or elapsed > 1.2:
                self.goal = hp.copy()
                self.transition("retract")
