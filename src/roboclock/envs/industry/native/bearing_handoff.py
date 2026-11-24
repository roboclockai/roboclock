"""Native two-arm arrest/recovery oracle; no scored-body state writes.

Right arm continuously brakes after measured entry. Left approaches from
upstream along the measured bore projection, grips the original outer race,
and receives the load before the right arm withdraws vertically.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from .arx_cell_task import ArmMotion

# Release gate of the oracle's supported lowering: the held bearing's lowest point may stand
# up to 3 mm above the tray floor (was 1 mm; the scored band, tray_valid, is -1..+2 mm and
# unchanged). The tray's rim-cradle rails are 1.5 mm high: held on them, a bearing rested
# 1.5 mm above the floor with 44 N of tray support and never entered the 1 mm gate (seed 1
# with layout randomization, 2026-11-20, timed out in "lower"). Released, it settles inside
# the scored band.
RELEASE_FLOOR_BAND_M = 0.003


class BearingARXController:
    def __init__(self, scene, mode):
        self.s, self.mode = scene, mode
        self.motion = {side: ArmMotion(scene.dual, side) for side in ["left", "right"]}
        self.rotation = {}
        for side, motion in self.motion.items():
            motion.position_gain = 1.5
            motion.orientation_feedback = True
            self.rotation[side] = scene.dual.tool_world(side)[1].copy()
        self.capture_R = (
            Rotation.from_euler(
                "z", scene.resolved.get("brake_yaw_deg", 0), degrees=True
            ).as_matrix()
            @ Rotation.from_euler("y", 90, degrees=True).as_matrix()
        )
        self.output_R = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        self.capture = np.array([1.04, scene.resolved["lane_y_m"], scene.surface_z(1.04) + 0.077])
        self.phase = "standby"
        self.entered = 0.0
        self.complete = False
        self.failure = None
        self.travel = dict(right=0.020, left=0.040)
        self.since = self.lost_since = None
        self.arrest_window, self.lift_window, self.place_window, self.retention_window = (
            [],
            [],
            [],
            [],
        )
        self.lower_bias = 0.0

    def transition(self, phase):
        self.phase, self.entered = phase, float(self.s.world.current_time)
        self.s.events[phase + "_s"] = self.entered
        print("PHASE", phase, self.entered, flush=True)

    def move(self, side, goal, target_R, speed):
        current = self.rotation[side]
        actual = self.s.dual.tool_world(side)[1]
        delta = Rotation.from_matrix(target_R @ current.T).as_rotvec()
        if Rotation.from_matrix(current @ actual.T).magnitude() < 0.05:
            current = (
                Rotation.from_rotvec(
                    delta
                    * min(
                        1,
                        (
                            0.15
                            if self.phase
                            in [
                                "lift",
                                "lift_hold",
                                "clear",
                                "turn",
                                "transfer_side",
                                "transfer",
                                "lower",
                            ]
                            else 0.30
                        )
                        / CONTROL_HZ
                        / max(np.linalg.norm(delta), 1e-12),
                    )
                ).as_matrix()
                @ current
            )
            self.rotation[side] = current
        return self.motion[side].move(goal, speed, current) and np.linalg.norm(delta) < 0.005

    def window(self, rows, o, dwell, angle_deg=0.2):
        t = o["time_s"]
        rows.append((t, np.array(o["bearing_position"]), np.array(o["bearing_rotation"])))
        while len(rows) > 1 and rows[1][0] <= t - dwell:
            rows.pop(0)
        if len(rows) < 2 or t - rows[0][0] < dwell:
            return None
        span = float(np.linalg.norm(np.ptp([r[1] for r in rows], axis=0)))
        angle = float(
            Rotation.from_matrix(np.array([r[2] for r in rows]) @ rows[0][2].T).magnitude().max()
        )
        if span > 0.0002 or angle > np.deg2rad(angle_deg):
            return None
        return dict(
            time_s=t,
            samples=len(rows),
            duration_s=t - rows[0][0],
            position_span_m=span,
            orientation_from_first_rad=angle,
            raw_velocity_gate=False,
        )

    def inside(self, o):
        return (
            o["bounds_min_m"][0] >= self.s.resolved["zone_x_m"][0]
            and o["bounds_max_m"][0] <= self.s.resolved["zone_x_m"][1]
            and o["bounds_min_m"][1] >= 0.80
            and o["bounds_max_m"][1] <= 1.02
        )

    def step(self, o):
        s, t = self.s, o["time_s"]
        if self.complete:
            eligible = s.tray_valid(o)
            if not eligible:
                self.retention_window.clear()
            gate = s.resolved["tray_retention_stability"]
            stable = (
                self.window(
                    self.retention_window, o, gate["dwell_s"], gate["orientation_from_first_deg"]
                )
                if eligible
                else None
            )
            s.tray_retention_valid = stable is not None
            return
        if self.failure:
            return
        try:
            if self.mode == "passive":
                return
            if o["unexpected_contacts"]:
                raise RuntimeError("Unexpected native robot contact")
            entry = s.entry_at
            if s.first_hand_at is not None and (entry is None or s.first_hand_at < entry):
                raise RuntimeError("Hand contacted bearing before measured entry")
            if entry is not None and "arrest_confirmed_s" not in s.events:
                if t - entry > 3:
                    raise RuntimeError("3s arrest deadline expired")
                if o["bounds_max_m"][0] > s.resolved["zone_x_m"][1]:
                    raise RuntimeError("Bearing escaped before arrest")
            if self.phase in [
                "handoff_approach",
                "handoff_close",
                "brake_withdraw",
                "brake_park",
            ] and not self.inside(o):
                raise RuntimeError("Bearing escaped safety zone during handoff")
            if self.phase == "standby":
                # Hold the reset pose (no command is written) until the bearing's measured zone
                # entry, which also turns the QC lamp red. The scene resets the right hand at
                # the capture point; the oracle used to servo it there before the entry.
                if (
                    entry is not None
                    and self.mode != "absent"
                    and (self.mode != "late" or t - entry > 3.3)
                ):
                    self.transition("capture")
                    # The held hand hangs below the capture pose by the drives' gravity offset
                    # (about 5 mm). Command that offset at once (what the integrators held when
                    # the oracle servoed here before the entry): the bearing arrives in ~0.4 s.
                    motion = self.motion["right"]
                    limit = motion.position_correction_limit_m
                    actual, actual_R = s.dual.tool_world("right")
                    motion.command = self.capture.copy()
                    motion.correction = np.clip(self.capture - actual, -limit, limit)
                    motion.orientation_correction = np.clip(
                        Rotation.from_matrix(self.capture_R @ actual_R.T).as_rotvec(), -0.05, 0.05
                    )
                    self.rotation["right"] = self.capture_R.copy()
                    self.move("right", self.capture, self.capture_R, 0.04)
            elif self.phase in ["capture", "handoff_approach", "handoff_close"]:
                self.move("right", self.capture, self.capture_R, 0.04)
            if self.phase == "capture":
                if min(o["finger_forces_N"]["right"]) < 10:
                    self.travel["right"] = max(
                        s.resolved.get("capture_minimum_command_m", 0.002),
                        self.travel["right"] - 0.028 / CONTROL_HZ,
                    )
                supported = (
                    self.inside(o)
                    and o["lane_support_N"] > 0.2
                    # STOP is a supported mechanical brake, not a grasp.
                    # One original fingertip may arrest the outer race against
                    # the lane. The later left-hand pickup still requires two
                    # independently measured >10N contacts and a free lift.
                    and max(o["finger_forces_N"]["right"]) > 0.2
                )
                if not supported:
                    self.arrest_window.clear()
                stable = self.window(self.arrest_window, o, 0.15) if supported else None
                if stable:
                    s.events.update(
                        arrest_native_window=stable, arrest_confirmed_s=t, response_s=t - entry
                    )
                    s.cue_state = "ARREST VERIFIED / RECOVER"
                    s.display.set("arrested")
                    p, R = np.array(o["bearing_position"]), np.array(o["bearing_rotation"])
                    normal = R[:, 1].copy()
                    normal *= 1 if normal[1] >= 0 else -1
                    horizontal = np.r_[normal[:2] / np.linalg.norm(normal[:2]), 0]
                    # The rolling bearing arrests yawed to either side of the lane, so the
                    # hand follows the measured bore axis, as far as the left elbow clears
                    # the right arm holding the brake.
                    approach = s.resolved["handoff"].get("approach_from_bore_deg")
                    approach_yaw = None
                    if approach is not None:
                        approach_yaw = min(
                            np.rad2deg(np.arctan2(normal[1], normal[0])) + approach,
                            s.resolved["handoff"]["approach_yaw_max_deg"],
                        )
                        horizontal = np.array(
                            [
                                np.cos(np.deg2rad(approach_yaw)),
                                np.sin(np.deg2rad(approach_yaw)),
                                0.0,
                            ]
                        )
                    self.pickup_R = (
                        Rotation.from_euler(
                            "z", np.arctan2(horizontal[1], horizontal[0])
                        ).as_matrix()
                        @ Rotation.from_euler(
                            "y", s.resolved["handoff"]["pickup_pitch_deg"], degrees=True
                        ).as_matrix()
                    )
                    if s.resolved.get("recovery_keep_pickup_orientation", False):
                        self.output_R = self.pickup_R.copy()
                    elif "recovery_output_pitch_deg" in s.resolved:
                        self.output_R = (
                            Rotation.from_euler("z", approach_yaw, degrees=True).as_matrix()
                            @ Rotation.from_euler(
                                "y", s.resolved["recovery_output_pitch_deg"], degrees=True
                            ).as_matrix()
                        )
                    self.grip = (
                        p
                        - horizontal * s.resolved["handoff"]["bore_horizontal_offset_m"]
                        + [0, 0, s.resolved["handoff"]["pickup_above_center_m"]]
                    )
                    s.events["handoff_geometry"] = dict(
                        time_s=t,
                        measured_bearing_m=p.tolist(),
                        measured_bore=normal.tolist(),
                        left_tool_goal_m=self.grip.tolist(),
                        left_tool_rotation=self.pickup_R.tolist(),
                    )
                    self.transition("handoff_approach")
            elif self.phase == "handoff_approach":
                arrived = self.move("left", self.grip, self.pickup_R, 0.035)
                tool = s.dual.tool_world("left")[0]
                if arrived and np.linalg.norm(tool - self.grip) < 0.001:
                    self.since = None
                    self.transition("handoff_close")
            elif self.phase == "handoff_close":
                self.move("left", self.grip, self.pickup_R, 0.02)
                strong = min(o["finger_forces_N"]["left"]) > 10
                if not strong:
                    self.travel["left"] = max(0.002, self.travel["left"] - 0.028 / CONTROL_HZ)
                self.since = (t if self.since is None else self.since) if strong else None
                if self.since is not None and t - self.since >= 0.10:
                    tool, R = s.dual.tool_world("left")
                    self.offset = R.T @ (np.array(o["bearing_position"]) - tool)
                    self.lift_start = tool.copy()
                    self.brake_clear = s.dual.tool_world("right")[0] + [0, 0, 0.107]
                    s.events["strong_bilateral_pickup_s"] = t
                    s.events["pickup_arm"] = "left"
                    self.transition("brake_withdraw")
            elif self.phase == "brake_withdraw":
                self.move("left", self.grip, self.pickup_R, 0.02)
                # Withdraw the narrow brake vertically before opening. Opening
                # in place intersects the original outer-race pickup fingers.
                arrived = self.move("right", self.brake_clear, self.capture_R, 0.05)
                if arrived and max(o["finger_forces_N"]["right"]) < 0.01:
                    s.events["brake_vertical_clear_s"] = t
                    self.transition("brake_park")
            elif self.phase == "brake_park":
                self.move("left", self.grip, self.pickup_R, 0.02)
                park_R = (
                    Rotation.from_euler(
                        "z", s.resolved["handoff"]["right_park_yaw_deg"], degrees=True
                    ).as_matrix()
                    @ self.capture_R
                )
                arrived = self.move("right", [1.22, 0.93, 1.10], park_R, 0.05)
                if arrived and max(o["finger_forces_N"]["right"]) < 0.01:
                    s.active_side = "left"
                    s.events["right_clear_handoff_s"] = t
                    self.lift_start = s.dual.tool_world("left")[0].copy()
                    self.transition("lift")
            elif self.phase in [
                "lift",
                "lift_hold",
                "clear",
                "turn",
                "transfer_side",
                "transfer",
                "lower",
            ]:
                self.travel["left"] = max(0.002, self.travel["left"] - 0.004 / CONTROL_HZ)
                tool = s.dual.tool_world("left")[0]
                held = (
                    min(o["finger_forces_N"]["left"]) > 0.2
                    and np.linalg.norm(np.array(o["bearing_position"]) - tool) < 0.050
                )
                self.lost_since = (
                    None if held else (t if self.lost_since is None else self.lost_since)
                )
                if self.lost_since is not None and t - self.lost_since > 0.2:
                    raise RuntimeError("Native bearing grasp lost during recovery")
                if self.phase in ["lift", "lift_hold"]:
                    arrived = self.move("left", self.lift_start + [0, 0, 0.10], self.pickup_R, 0.05)
                    if self.phase == "lift" and arrived:
                        self.transition("lift_hold")
                    if self.phase == "lift_hold":
                        retained = (
                            arrived
                            and o["lane_support_N"] < 0.01
                            and min(o["finger_forces_N"]["left"]) > 10
                        )
                        if not retained:
                            self.lift_window.clear()
                        stable = self.window(self.lift_window, o, 0.2) if retained else None
                        if stable:
                            s.events["lift_native_window"] = stable
                            if self.mode == "grasp":
                                self.complete = s.manipulation_success = True
                                self.transition("grasp_complete")
                            else:
                                self.transition("clear")
                else:
                    p = np.array(o["bearing_position"])
                    body = dict(
                        clear=[0.95, 0.75, 1.10],
                        turn=[0.95, 0.75, 1.10],
                        transfer_side=[0.92, 0.72, 1.10],
                        transfer=[*s.resolved["recovery_target_xy_m"], 1.10],
                        lower=[
                            *s.resolved["recovery_target_xy_m"],
                            s.tray_floor + p[2] - o["bounds_min_m"][2] + 0.001,
                        ],
                    )[self.phase]
                    if self.phase == "lower":
                        body[2] += self.lower_bias
                        bounds = s.resolved["recovery_tray_inner_bounds_xy_m"]
                        supported = (
                            np.all(np.array(o["bounds_min_m"][:2]) >= bounds[0])
                            and np.all(np.array(o["bounds_max_m"][:2]) <= bounds[1])
                            and s.tray_floor - 0.001 <= o["bounds_min_m"][2] <= s.tray_floor + RELEASE_FLOOR_BAND_M
                            and (o["tray_support_N"] > 0.2 or o["tray_touching"])
                        )
                        if not supported:
                            self.place_window.clear()
                        stable = self.window(self.place_window, o, 0.15) if supported else None
                        if stable:
                            # Stop on actual supported placement, not a fixed
                            # tool offset that fights a compliant grasp/floor.
                            # Only drive targets change; product stays free.
                            self.release_goal, self.output_R = s.dual.tool_world("left")
                            self.rotation["left"] = self.output_R.copy()
                            self.motion["left"].command = self.release_goal.copy()
                            self.motion["left"].correction[:] = 0
                            self.motion["left"].orientation_correction[:] = 0
                            s.events["supported_tray_before_release"] = stable
                            self.transition("release")
                            return
                    target_R = self.pickup_R if self.phase == "clear" else self.output_R
                    # Keep the held root centered as the bounded tool rotates.
                    goal = np.array(body) - self.rotation["left"] @ self.offset
                    if self.move("left", goal, target_R, 0.06 if self.phase == "lower" else 0.10):
                        if self.phase == "lower":
                            # A nominal endpoint above the floor is not a
                            # placement. Descend at1mm/s until genuine support.
                            self.lower_bias = max(-0.003, self.lower_bias - 0.001 / CONTROL_HZ)
                        else:
                            self.transition(
                                dict(
                                    clear="turn",
                                    turn="transfer_side",
                                    transfer_side="transfer",
                                    transfer="lower",
                                )[self.phase]
                            )
            elif self.phase == "release":
                self.move("left", self.release_goal, self.output_R, 0.02)
                self.travel["left"] = min(0.040, self.travel["left"] + 0.010 / CONTROL_HZ)
                if self.travel["left"] >= 0.040:
                    self.since = None
                    self.transition("withdraw")
            elif self.phase == "withdraw":
                arrived = self.move("left", self.release_goal + [0, 0, 0.08], self.output_R, 0.08)
                valid = arrived and s.tray_valid(o)
                if not valid:
                    self.retention_window.clear()
                gate = s.resolved["tray_retention_stability"]
                stable = (
                    self.window(
                        self.retention_window,
                        o,
                        gate["dwell_s"],
                        gate["orientation_from_first_deg"],
                    )
                    if valid
                    else None
                )
                s.tray_retention_valid = stable is not None
                if stable:
                    s.events["released_tray_retention_window"] = stable
                    s.events["released_tray_retention_s"] = t
                    s.physical_completed = s.manipulation_success = True
                    s.success = s.final_valid(o)
                    s.cue_state = "RECOVERED / REWORK"
                    s.display.set("complete")
                    self.complete = True
                    self.transition("complete")
            for side in ["left", "right"]:
                s.dual.jaw_travel(side, self.travel[side])
            if self.phase != "standby" and t - self.entered > 18:
                raise RuntimeError("Bearing handoff timeout " + self.phase)
        except RuntimeError as e:
            self.failure = str(e)
            print("FAILURE", self.failure, flush=True)
