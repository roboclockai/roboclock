"""Identity-counted throughput and measured cue strokes in one continuous scene."""

import numpy as np
from scipy.spatial.transform import Rotation


class Clearance:
    def __init__(self, b):
        self.b, self.c = b, b.config["workflow"]
        self.done = False
        self.phase = "GRASP CUE / CLOCK RUNNING"
        self.events, self.failures, self.shots = [], [], []
        self.pots, self.candidates = {}, {}
        self.grip_start = self.settle_start = None
        self.retained = False
        self.last_hit = False
        self.strike_armed = True
        self.contact_start = None
        self.active = None
        self.ready = False

    def contact(self, obj, name):
        return obj["forces"][self.b.filter_paths.index("/World/Objects/" + name)] > 0.03

    def fail(self, reason, now):
        if reason not in self.failures:
            self.failures.append(reason)
            self.events.append(dict(event=reason, time=now))
        if self.active is not None:
            self.active["valid"] = False

    def hand_contact_allowed(self, name, s):
        return False

    @property
    def valid_count(self):
        return sum(len(s["pots"]) for s in self.shots if s["valid"])

    def update(self, s):
        if self.done:
            return
        now = s["time"]
        # No event after the fixed budget may change the score.
        if now > self.c["pool_horizon"] + 1e-8:
            self.done = True
            return
        objects = s["objects"]
        cue, ball = objects["cue"], objects["cue_ball"]
        grasp = min(cue["forces"][:2]) > 0.025
        if grasp:
            self.grip_start = now if self.grip_start is None else self.grip_start
            if now - self.grip_start >= 0.2:
                self.retained = True
        else:
            self.grip_start = None
        names = [n for n in self.b.pool_targets if n not in self.pots]
        settled = all(np.linalg.norm(objects[n]["velocity"][:2]) < 0.004
                      and abs(objects[n]["velocity"][2]) < 0.03
                      and np.linalg.norm(objects[n]["angular_velocity"]) < 0.15
                      for n in ["cue_ball", *names])
        if settled:
            self.settle_start = now if self.settle_start is None else self.settle_start
        else:
            self.settle_start = None
        self.ready = self.retained and self.settle_start is not None and now - self.settle_start >= self.c["settle_duration"]
        hit = self.contact(ball, "cue")
        cue_rotation = Rotation.from_quat(np.roll(cue["orientation"], -1))
        relative = cue_rotation.inv().apply(np.asarray(ball["position"]) - cue["position"])
        # Distance to the finite cylindrical tip, including its flat cap.
        # A brief zero-force solver sample without geometric separation is
        # still the same contact episode, not a second stroke.
        axial_gap = max(.171 - relative[0], 0., relative[0] - .177)
        radial_gap = max(np.linalg.norm(relative[1:]) - .009, 0.)
        separated = np.hypot(axial_gap, radial_gap) > getattr(self.b, "ball_radius", .023) + .003
        if self.b.config.get("pool", {}).get("rounded_tip", False):
            separated = np.linalg.norm(relative - [.171, 0., 0.]) > .006 + self.b.ball_radius + .003
        if not hit and separated:
            self.strike_armed = True
        if hit and self.strike_armed:
            self.strike_armed = False
            tip = np.asarray(cue["position"]) + Rotation.from_quat(np.roll(cue["orientation"], -1)).apply([0.177, 0, 0])
            valid = grasp and np.linalg.norm(tip - ball["position"]) < 0.042
            previous = self.active
            # Contact can begin on the tick that motion destroys the settled predicate.
            was_ready = getattr(self, "previous_ready", False)
            self.active = dict(time=now, valid=bool(valid and was_ready), contacts=[], pots=[])
            self.shots.append(self.active)
            if not was_ready:
                self.fail("strike_before_settled", now)
                if previous is not None:
                    previous["valid"] = False
            if not valid:
                self.fail("invalid_cue_contact", now)
            self.events.append(dict(event="strike", time=now, shot=len(self.shots),
                                    cue_ball_position=np.asarray(ball["position"]).tolist()))
            self.contact_start = now
            self.settle_start = None
        if hit and self.contact_start is not None and now - self.contact_start > 0.065:
            self.fail("push_shot", now)
        self.last_hit = hit
        if any(max(objects[n]["forces"][:len(getattr(self.b, "fingers", [0, 1]))]) > 0.03
               and not self.hand_contact_allowed(n, s) for n in ["cue_ball", *self.b.pool_targets]):
            self.fail("hand_ball_contact", now)
        if self.active is not None:
            # Propagate physical contact ancestry for legal combination shots.
            reached = set(self.active["contacts"]) | {"cue_ball"}
            for _ in self.b.pool_targets:
                for name in self.b.pool_targets:
                    if name not in reached and any(self.contact(objects[name], other) for other in reached):
                        reached.add(name)
                        self.active["contacts"].append(name)
                        self.events.append(dict(event="ball_collision", ball=name, time=now))
        for name in self.b.pool_targets:
            obj = objects[name]
            if self.contact(obj, "cue"):
                self.fail("direct_target_contact", now)
            if self.active is not None and self.contact(obj, "cue_ball") and name not in self.active["contacts"]:
                self.active["contacts"].append(name)
                self.events.append(dict(event="ball_collision", ball=name, time=now))
            p = np.asarray(obj["position"])
            supported = any(obj.get("support_contacts", {}).get(f"Pocket{i}", 0) > 0.05 for i in range(len(self.b.pockets)))
            half = (self.b.pool_layout.capture_half_extents(self.b.ball_radius)
                    if hasattr(self.b, "pool_layout") and self.b.pool_layout.scale != 1.
                    else np.array([.077, .062]))
            inside = any(np.all(np.abs(p[:2] - pocket) < half) and .61 < p[2] < .70
                         for pocket in self.b.pockets)
            captured = inside and supported and np.linalg.norm(obj["velocity"]) < 0.08
            if name in self.pots:
                if not inside:
                    self.fail("potted_ball_escaped", now)
                    for shot in self.shots:
                        if name in shot["pots"]:
                            shot["valid"] = False
                continue
            if captured:
                self.candidates.setdefault(name, now)
                if now - self.candidates[name] >= 0.25:
                    if self.active is not None and self.active["valid"] and name in self.active["contacts"]:
                        self.pots[name] = now
                        self.active["pots"].append(name)
                        self.events.append(dict(event="valid_pot", time=now, ball=name, total=len(self.pots)))
            else:
                self.candidates.pop(name, None)
            if p[2] < 0.58:
                self.fail("target_off_table", now)
        if ball["position"][2] < 0.71:
            self.fail("scratch_or_off_table", now)
        if cue["position"][2] < 0.70:
            self.fail("lost_cue", now)
        self.previous_ready = self.ready and not hit
        self.phase = "TABLE CLEAR" if len(self.pots) == len(self.b.pool_targets) else "AIM / BALLS SETTLED" if self.ready else "WAIT / BALLS MOVING"
        if now >= self.c["pool_horizon"] - 1e-8:
            self.done = True

    def display(self):
        remaining = max(0, self.c["pool_horizon"] - self.b.time)
        activity = "CLEAR" if len(self.pots) == len(self.b.pool_targets) else "AIM" if self.ready else "WAIT"
        return (f"{self.valid_count}/{len(self.b.pool_targets)} POTS / {remaining:04.1f}s LEFT",
                f"GOAL {self.c['minimum_pots_per_minute']:.1f}/MIN / {activity}")

    def result(self):
        valid = self.valid_count
        rate = 60 * valid / self.c["pool_horizon"]
        physical = valid > 0 and not self.failures
        temporal = self.done and rate + 1e-9 >= self.c["minimum_pots_per_minute"]
        failures = list(self.failures)
        if self.done and not temporal:
            failures.append("throughput_below_target")
        return dict(task="C1", primary_temporal_category="throughput",
                    success=physical and temporal, physical_success=physical, temporal_success=temporal,
                    valid_pots=valid, total_targets=len(self.b.pool_targets), pots_per_minute=rate,
                    required_pots_per_minute=self.c["minimum_pots_per_minute"], budget=self.c["pool_horizon"],
                    complete_clearance=valid == len(self.b.pool_targets), failures=failures,
                    events=self.events, shots=self.shots,
                    foul_shots=sum(not s["valid"] for s in self.shots),
                    foul_rate=sum(not s["valid"] for s in self.shots) / max(1, len(self.shots)),
                    controller="privileged scripted feedback baseline")


class ClearanceOracle:
    def __init__(self, b, t):
        self.b, self.task = b, t
        self.phase, self.start = "approach", 0.0
        grasp_yaw = b.config.get("pool", {}).get("grasp_yaw_by_side", {}).get(getattr(b, "active_side", "right"), 0.)
        self.rotation = Rotation.from_euler("z", grasp_yaw, degrees=True).as_matrix() @ Rotation.from_euler("y", 180, degrees=True).as_matrix()
        self.offset = np.zeros(3)
        self.tool_rotation = np.eye(3)
        self.goal = np.array([0.31, -0.12, 0.80])
        self.direction = np.array([1., 0., 0.])
        self.shot_count = 0
        self.stroke_speed = 0.6
        self.last_grip = 0.0
        self.excluded_target = None
        self.excluded_at = None
        self.replans = 0

    def transition(self, phase):
        self.phase, self.start = phase, self.b.time

    def plan(self):
        b, t = self.b, self.task
        ball = np.asarray(b.last_state["objects"]["cue_ball"]["position"])
        if self.excluded_at is not None and np.linalg.norm(ball - self.excluded_at) > .04:
            self.excluded_target = None
        choices = []
        for name in b.pool_targets:
            if name in t.pots or name == self.excluded_target:
                continue
            obj = np.asarray(b.last_state["objects"][name]["position"])
            if obj[2] < 0.74:
                # A ball awaiting pocket dwell confirmation is no longer a
                # playable table target, even before its count is committed.
                continue
            for pocket in b.pockets:
                outgoing = pocket - obj[:2]
                pot_distance = np.linalg.norm(outgoing)
                outgoing /= pot_distance
                ghost = obj[:2] - 2 * b.ball_radius * outgoing
                direction = ghost - ball[:2]
                distance = np.linalg.norm(direction)
                direction /= max(distance, 1e-9)
                cosine = np.dot(direction, outgoing)
                # Prefer a nearly straight collision and nearby target.
                hand_xy = ball[:2] - direction * (0.235 - getattr(b, "cue_grip_x", 0.))
                reach = np.linalg.norm(hand_xy - np.asarray(b.config["robot_mount"][:2]))
                reachable = 0.18 < reach < 0.76
                if reachable and cosine > 0.1:
                    velocity = np.clip((0.32 + 0.65 * pot_distance + 0.2 * distance) / max(cosine, 0.6), 0.35, 0.85)
                    choices.append((cosine - 0.15 * distance, name, direction, velocity))
        if not choices:
            if len(t.pots) == len(b.pool_targets):
                return False
            # A direct pot can be unreachable from the far cushion. Strike the
            # white ball into that cushion from the reachable near side, then
            # replan from its measured rebound position. No pose/velocity edits.
            direction = np.array([[.12, -.12, .25][self.replans % 3], 1. if ball[1] > 0 else -1.])
            direction /= np.linalg.norm(direction)
            self.object, self.stroke_speed = "cue_ball", 0.70
            self.mode = "reposition"
        else:
            _, self.object, direction, self.stroke_speed = max(choices, key=lambda x: x[0])
            self.mode = "pot"
        rough_center = ball[:2] - direction * 0.22
        pitch_degrees = 16. if abs(rough_center[1]) > 0.31 or rough_center[0] > 1.14 else 5.
        pitch = np.deg2rad(pitch_degrees)
        self.direction = np.array([*(direction * np.cos(pitch)), -np.sin(pitch)])
        yaw = np.arctan2(direction[1], direction[0])
        cue_rotation = Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("y", pitch_degrees, degrees=True).as_matrix()
        self.desired_cue_rotation = cue_rotation
        hand_rotation = cue_rotation @ self.tool_rotation.T
        self.rotation = hand_rotation
        tip = ball + np.array([-direction[0] * (b.ball_radius + 0.022),
                               -direction[1] * (b.ball_radius + 0.022), 0.010])
        self.tip_goal = tip.copy()
        center = tip - self.direction * 0.177
        self.goal = center - hand_rotation @ self.offset
        self.nominal_goal = self.goal.copy()
        self.shot_count = len(t.shots)
        self.turn_goal = np.asarray(b.last_state["hands"]["main"]["position"]).copy()
        self.turn_goal[2] = .99
        return True

    def act(self):
        b, t = self.b, self.task
        grip, speed = 0., 0.20
        deadline = {"turn": 10., "position": 5., "aim": 5.}.get(self.phase)
        if deadline is not None and b.time - self.start > deadline:
            # Abandon a nonconverging pose instead of holding it for the rest
            # of the episode. Reset only the command reference, never a body.
            hand = b.last_state["hands"]["main"]
            self.excluded_target = self.object if self.object != "cue_ball" else None
            self.excluded_at = np.asarray(b.last_state["objects"]["cue_ball"]["position"]).copy()
            self.goal = np.asarray(hand["position"]).copy()
            self.rotation = Rotation.from_quat(np.roll(hand["orientation"], -1)).as_matrix()
            b.hand_targets["main"] = (self.goal.copy(), self.rotation.copy())
            self.replans += 1
            self.transition("raise")
        if self.phase in ("turn", "position", "aim") and (
            self.object in t.pots or b.last_state["objects"][self.object]["position"][2] < 0.74
        ):
            self.transition("raise")
        held = min(b.last_state["objects"]["cue"]["forces"][:2]) > 0.025
        if held:
            self.last_grip = b.time
        if self.phase not in ("approach", "descend", "grasp", "finished") and b.time - self.last_grip > 0.4:
            obj = b.last_state["objects"]["cue"]
            axis = Rotation.from_quat(np.roll(obj["orientation"], -1)).apply([1, 0, 0])
            grasp_yaw = b.config.get("pool", {}).get("grasp_yaw_by_side", {}).get(getattr(b, "active_side", "right"), 0.)
            self.rotation = Rotation.from_euler("z", np.arctan2(axis[1], axis[0]) + np.deg2rad(grasp_yaw)).as_matrix() @ Rotation.from_euler("y", 180, degrees=True).as_matrix()
            self.transition("approach")
        if self.phase == "aim":
            cue = b.last_state["objects"]["cue"]
            hand = b.last_state["hands"]["main"]
            cue_r = Rotation.from_quat(np.roll(cue["orientation"], -1)).as_matrix()
            hand_r = Rotation.from_quat(np.roll(hand["orientation"], -1)).as_matrix()
            tip = np.asarray(cue["position"]) + cue_r @ [0.177, 0, 0]
            self.rotation = self.desired_cue_rotation @ cue_r.T @ hand_r
            # The torque-limited six-axis X5 needs a slow outer correction loop;
            # avoid integrating past the moving tip.
            gain, step = 0.02, 0.0005
            self.goal = np.clip(self.goal + np.clip((self.tip_goal - tip) * gain, -step, step),
                                self.nominal_goal - 0.08, self.nominal_goal + 0.08)
        target = self.goal.copy()
        obj = b.last_state["objects"]["cue"]
        handle = np.asarray(obj["position"]) + Rotation.from_quat(np.roll(obj["orientation"], -1)).apply([getattr(b, "cue_grip_x", 0.), 0, 0.055])
        if self.phase == "approach":
            target, grip = handle + [0, 0, 0.14], 0.04
        elif self.phase in ("descend", "grasp"):
            target = handle + [0, 0, b.config.get("pool", {}).get("grasp_height_offset", -.010)]
            grip = 0.04 if self.phase == "descend" else 0.
        elif self.phase == "turn":
            target = self.turn_goal.copy()
        elif self.phase in ("raise", "position", "wait", "finished"):
            target[2] = .99
        elif self.phase == "stroke":
            target += self.direction * 0.07
            speed = self.stroke_speed
        elif self.phase == "retract":
            target -= self.direction * 0.04
            target[2] += 0.06
            speed = 0.8
        error = b.move("main", target, self.rotation, grip=grip, speed=speed)
        elapsed = b.time - self.start
        cue = b.last_state["objects"]["cue"]
        axis = Rotation.from_quat(np.roll(cue["orientation"], -1)).apply([1, 0, 0])
        tip = np.asarray(cue["position"]) + 0.177 * axis
        to_ball = np.asarray(b.last_state["objects"]["cue_ball"]["position"]) - tip
        heading = self.direction[:2] / np.linalg.norm(self.direction[:2])
        transverse = to_ball[:2] - heading * np.dot(to_ball[:2], heading)
        aligned = (0.033 < np.dot(to_ball[:2], heading) < 0.065
                   and np.linalg.norm(transverse) < 0.002
                   and abs(tip[2] - self.tip_goal[2]) < 0.004
                   and np.dot(axis, self.direction) > 0.98)
        if getattr(self, "mode", "pot") == "reposition":
            aligned = (0.033 < np.dot(to_ball[:2], heading) < 0.065
                       and np.linalg.norm(transverse) < .0045
                       and abs(tip[2] - self.tip_goal[2]) < .004
                       and np.dot(axis, self.direction) > .998)
        raised = error < 0.008
        if (self.phase == "raise"
                and b.config.get("embodiment", {}).get("model") == "dual_arx_x5"):
            # A native IK residual need not block a physically clear transport.
            # Conservatively enclose shaft, tip, saddle and handle, not only TCP.
            bounds = np.array([[x, y, z] for x in (-0.139, 0.186)
                               for y in (-0.040, 0.040) for z in (-0.012, 0.073)])
            cue_r = Rotation.from_quat(np.roll(cue["orientation"], -1))
            lowest = np.min(cue_r.apply(bounds)[:, 2] + cue["position"][2])
            obstacles = ["cue_ball", *[n for n in b.pool_targets if n not in t.pots]]
            highest = max(b.last_state["objects"][n]["position"][2] + b.ball_radius
                          for n in obstacles)
            raised = raised or lowest > highest + 0.050
        if self.phase == "approach" and error < 0.007:
            self.transition("descend")
        elif self.phase == "descend" and error < 0.005:
            self.transition("grasp")
        elif self.phase == "grasp" and t.grip_start is not None and b.time - t.grip_start > 0.35:
            hand = b.last_state["hands"]["main"]
            rot = Rotation.from_quat(np.roll(hand["orientation"], -1)).as_matrix()
            self.offset = rot.T @ (np.asarray(b.last_state["objects"]["cue"]["position"]) - hand["position"])
            self.tool_rotation = rot.T @ Rotation.from_quat(np.roll(b.last_state["objects"]["cue"]["orientation"], -1)).as_matrix()
            self.goal = np.asarray(hand["position"]).copy()
            self.transition("raise")
        elif self.phase == "raise" and raised:
            self.transition("wait")
        elif self.phase == "wait" and t.ready:
            self.transition("turn" if self.plan() else "finished")
        elif self.phase == "turn" and error < 0.008 and elapsed > 0.7:
            self.transition("position")
        elif self.phase == "position" and error < 0.006 and elapsed > 0.5:
            self.transition("aim")
        elif self.phase == "aim" and aligned and elapsed > 0.7 and t.ready:
            self.transition("stroke")
        elif self.phase == "stroke" and (len(t.shots) > self.shot_count or elapsed > 0.7):
            self.transition("retract")
        elif self.phase == "retract" and elapsed > 0.6:
            self.transition("raise")
