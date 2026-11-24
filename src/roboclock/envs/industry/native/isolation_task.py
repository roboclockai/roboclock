"""Task44: full ARX contact-driven closure of the original passive valve.

Leak motion is a process illustration, not fluid dynamics. The valve itself
has no actuator, post-reset joint writes or robot/product attachment.
"""

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from . import rig
from .arx_cell_task import (
    ArmMotion,
    ContactBook,
    LeakVisual,
    StatusDisplay,
    add_mounts,
    collide,
    remove_preview,
    set_signal,
)

# Only the right arm works (recorded right [0, 0.14, 1.13], left [-0.42, 0.05, 1.13],
# staggered 9 cm). The right base moves 0.15 m right and 0.06 m toward the valve: the
# head camera then sees the valve, the downstream flange/leak and the spill tray's far
# half, and the whole quarter turn (grasp radius 0.15-0.17 m) stays inside the original
# URDF reach. Centering the rig on the valve (right base at x=0.30) put the closed
# handle out of reach ("Unreachable bounded arm target" during the turn).
MOUNTS = rig.mounts_keeping("right", [0.15, 0.20, 1.13])
HANDLE = "/World/Valve/handle"
# Reset wrist roll of the working (right) arm. The valve grasp needs joint6 near 1.54 rad
# (1.51-1.57 over the grasp radii); from the common reset value (-0.8) that roll alone took
# 3.3 s at the arm's joint rate, far longer than the rest of the reach.
# Only the roll is pre-set: joints 1-5, and with them the hand's position beside the valve,
# stay at the common reset posture, for every policy.
RESET_JOINT6_RAD = 1.54
# Nominal valve pivot and the seeded floor-plane offset of the valve line (layout stream).
VALVE_PIVOT_M = (0.0, 0.50, 1.04)
VALVE_XY_OFFSET_M = 0.02


def resolve_isolation(seed=0, *, layout=False, timing=False, physical=False):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    r = dict(
        task_id=44,
        seed=seed,
        embodiment="dual_arx_x5",
        grasp_radius_m=float(streams[0].uniform(0.15, 0.17)) if layout else 0.16,
        # The whole valve line (skid, piping, valve, leak nipple) moves in the floor plane;
        # the valve pivot is VALVE_PIVOT_M plus this offset.
        valve_xy_offset_m=streams[0].uniform(-VALVE_XY_OFFSET_M, VALVE_XY_OFFSET_M, 2).tolist()
        if layout
        else [0.0, 0.0],
        leak_onset_s=float(streams[1].uniform(20, 24)) if timing else 22.0,
        joint_friction_coefficient=float(streams[2].uniform(0.01, 0.02)) if physical else 0.015,
        response_deadline_s=5.0,
        closed_threshold_deg=89.0,
        confirmation_dwell_s=0.15,
        estimated_leak_flow_l_s=0.14,  # 1/8 in hole at 60 psi; 14 s to the 2.0 L limit
        estimated_spill_limit_l=2.0,
        spill_model="Idealized constant-flow volume integrated over measured leak duration; geometry-fitted water visualization, not fluid dynamics. The response deadline remains the score; at this flow the spill limit is reached only after about 14 s.",
        randomization=dict(layout=layout, timing=timing, physical=physical),
        bounds=dict(
            grasp_radius_m=[0.15, 0.17],
            valve_xy_offset_m=[-VALVE_XY_OFFSET_M, VALVE_XY_OFFSET_M],
            leak_onset_s=[20, 24],
            joint_friction_coefficient=[0.01, 0.02],
        ),
        independence="Separate SeedSequence layout/temporal-event/physical streams",
        scoring="Native valve angle at least89deg (90deg mechanical limit), confirmed0.15s within5s of observable leak onset, with post-event native finger contact and retained closed release",
        assumptions="Original complete valve geometry and unactuated quarter-turn joint; dimensionless native joint friction is a declared demonstration parameter, not calibrated seal torque. Leak cue is symbolic and angle-driven.",
        controller="Privileged native-state bounded joint-drive oracle; not a trained policy",
    )
    r["sha256"] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
    return r


class IsolationScene:
    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.api.materials import PhysicsMaterial
        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .utility_scene import build_utility_environment
        from .v2.valve_fixture import build_passive_valve

        offset = np.asarray(resolved.get("valve_xy_offset_m", [0.0, 0.0]), float)
        self.base = build_utility_environment(output, resolution=(1280, 800), line_offset_xy=offset)
        self.world, self.stage = self.base.world, self.base.stage
        c = self.base.cell
        stage = self.stage
        self.resolved, self.events = resolved, {}
        self.pivot = np.array(VALVE_PIVOT_M) + np.r_[offset, 0.0]
        for name in ["housing", "handle"]:
            remove_preview(c, "/World/Valve/" + name)
        self.valve, self.valve_metadata = build_passive_valve(self.world, self.pivot)
        joint = stage.GetPrimAtPath("/World/Valve/LeverJoint")
        assert not UsdPhysics.DriveAPI(joint, "angular")
        PhysxSchema.PhysxJointAPI.Apply(joint).CreateJointFrictionAttr(
            resolved["joint_friction_coefficient"]
        )
        for name in ["housing", "handle"]:
            PhysxSchema.PhysxContactReportAPI.Apply(
                stage.GetPrimAtPath("/World/Valve/" + name)
            ).CreateThresholdAttr(0)
        PhysxSchema.PhysxArticulationAPI.Apply(
            stage.GetPrimAtPath("/World/Valve/Anchor")
        ).CreateSolverPositionIterationCountAttr(64)
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS, gripper_velocity_cap=False)
        add_mounts(c, MOUNTS)
        material = PhysicsMaterial(
            "/World/ValveContactMaterial", static_friction=0.9, dynamic_friction=0.75, restitution=0
        )
        self.static_colliders = collide(
            stage,
            ["/World/Piping/", "/World/Skid/", "/World/Containment/", "/World/ARXMounts/"],
            material,
        )
        for root in ["/World/Valve/handle", "/World/ARX/right"]:
            for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(
                        material.material, materialPurpose="physics"
                    )
                    a = PhysxSchema.PhysxCollisionAPI.Apply(p)
                    a.CreateContactOffsetAttr(0.0003)
                    a.CreateRestOffsetAttr(0)
        self.contacts = ContactBook(self.world, [HANDLE])
        self.leak = LeakVisual(c, offset)
        self.cue = c.appearance.material("isolation_cue", color=[0.03, 0.65, 0.2], emissive=True)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Control/Beacon")).Bind(
            self.cue.material
        )
        from .spill_visual import TraySpillVisual

        self.spill = TraySpillVisual(c, resolved.get("estimated_spill_limit_l", 2.0))
        self.spill_amount_l = 0.0
        self.spill_last_time = 0.0
        self.spill_was_leaking = False
        self.valve_metadata["spill_visual"] = dict(
            geometry_sha256=self.spill.geometry_sha256,
            nominal_flow_l_s=resolved.get("estimated_leak_flow_l_s", 0.14),
            warning_volume_l=self.spill.limit_l,
            warning_world_z_m=self.spill.warning_z,
            scope="Estimated fill visualization only; no fluid dynamics or change to the response-deadline score",
        )
        self.display = StatusDisplay(
            c,
            "Warning",
            {
                "NORMAL": "LINE NORMAL / MONITORING",
                "LEAK / CLOSE VALVE": "LEAK / CLOSE VALVE <=5 s",
                "ISOLATED": "VALVE CLOSED / ISOLATED",
            },
        )
        self.display.set("NORMAL")
        self.cue_state = None
        self.closed_since = None
        self.physical_completed = False
        self.success = False
        self.leaking = False
        self.world.reset()
        self.dual.initialize()
        self.valve.initialize()
        self.valve.set_joint_positions(np.zeros(1))
        self.valve.set_joint_velocities(np.zeros(1))
        self.dual.jaw_travel("right", 0.027)
        right = self.dual.commands["right"]
        right[self.dual.names["right"].index("joint6")] = RESET_JOINT6_RAD
        self.dual.robots["right"].set_joints_default_state(positions=right, velocities=np.zeros(8))
        self.dual.robots["right"].set_joint_positions(right)
        left = self.dual.commands["left"]
        left[self.dual.names["left"].index("joint1")] = 1.8
        self.dual.robots["left"].set_joint_positions(left)
        for mapping in [
            self.contacts.interval,
            self.contacts.pairs,
            self.contacts.minimum,
            self.contacts.active,
            self.contacts.unexpected,
        ]:
            mapping.clear()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=44)

    def grasp(self, angle):
        R = Rotation.from_euler("z", angle).as_matrix()
        return self.pivot + R @ np.array(
            [self.resolved["grasp_radius_m"], 0, 0.042]
        ), R @ Rotation.from_euler("y", np.pi / 2).as_matrix()

    def observe(self):

        t = float(self.world.current_time)
        angle = float(self.valve.get_joint_positions()[0])
        omega = float(self.valve.get_joint_velocities()[0])
        if t >= self.resolved["leak_onset_s"] and "leak_onset_s" not in self.events:
            self.events["leak_onset_s"] = t
        started = "leak_onset_s" in self.events
        forces = [self.contacts.force(HANDLE, f"/link{i}") for i in [7, 8]]
        closed = (
            np.deg2rad(self.resolved["closed_threshold_deg"]) <= angle <= np.deg2rad(91)
            and abs(omega) < 0.2
        )
        if started and max(forces) > 0.15:
            self.events.setdefault("post_event_contact_s", t)
        # Measured partial closure (stage score): the handle turned past half the closing
        # angle after the leak, under the gripper's post-event contact.
        if (
            "post_event_contact_s" in self.events
            and angle >= np.deg2rad(self.resolved["closed_threshold_deg"]) / 2
        ):
            self.events.setdefault("valve_half_closed_s", t)
        if started and closed and "post_event_contact_s" in self.events:
            self.closed_since = self.closed_since if self.closed_since is not None else t
            if (
                t - self.closed_since >= self.resolved["confirmation_dwell_s"]
                and not self.physical_completed
            ):
                self.physical_completed = True
                self.events.update(
                    closure_started_s=self.closed_since,
                    closure_confirmed_s=t,
                    response_s=t - self.events["leak_onset_s"],
                )
                self.success = bool(
                    t - self.events["leak_onset_s"] <= self.resolved["response_deadline_s"]
                    and "pre_event_motion_s" not in self.events
                )
                self.events["temporal_success"] = self.success
        else:
            self.closed_since = None
        if not started and abs(angle) > np.deg2rad(2):
            self.events.setdefault("pre_event_motion_s", t)
        self.leaking = bool(started and angle < np.deg2rad(self.resolved["closed_threshold_deg"]))
        # Process-only moving points: emitted by leak state, stopped by measured
        # native closure. They never serve as collision/flow validation.
        if self.spill_was_leaking:
            self.spill_amount_l += (t - self.spill_last_time) * self.resolved.get("estimated_leak_flow_l_s", 0.14)
        self.spill_last_time = t
        self.spill_was_leaking = self.leaking
        spill_level = self.spill.update(self.spill_amount_l)
        self.leak.update(self.leaking, t - self.events.get("leak_onset_s", t))
        state = (
            "ISOLATED" if self.physical_completed
            else "LEAK / CLOSE VALVE" if started else "NORMAL"
        )
        if state != self.cue_state:
            set_signal(
                self.cue,
                [1.0, 0.03, 0.02] if started and not self.physical_completed else [0.03, 0.9, 0.15],
            )
            self.display.set(state)
            self.cue_state = state
        robot = self.dual.robots["right"]
        jaws = robot.get_joint_positions()[
            [self.dual.names["right"].index(f"joint{i}") for i in [7, 8]]
        ]
        return dict(
            time_s=t,
            angle_rad=angle,
            angle_deg=float(np.rad2deg(angle)),
            angular_velocity_rad_s=omega,
            tool_position=self.dual.tool_world("right")[0].tolist(),
            finger_forces_N=forces,
            jaws_m=jaws.tolist(),
            leaking=self.leaking,
            estimated_spill_l=self.spill_amount_l,
            estimated_spill_level_world_z_m=spill_level,
            estimated_spill_limit_l=self.spill.limit_l,
            closed=bool(closed),
            cue_state=state,
            unexpected_contacts=self.contacts.unexpected.copy(),
        )


class IsolationController:
    def __init__(self, scene, mode):
        self.s, self.mode = scene, mode
        self.motion = ArmMotion(scene.dual, "right")
        self.phase = "settle"
        self.entered = 0.0
        self.failure = None
        self.complete = False
        self.travel = 0.027
        self.grip_since = None
        self.angle_target = 0.0
        self.grasped = False
        self.reach_joints = self.reach_tool = None

    def transition(self, phase):
        self.phase = phase
        self.entered = float(self.s.world.current_time)
        self.s.events["phase_" + phase + "_s"] = self.entered
        print("PHASE", phase, self.entered, flush=True)

    def step(self, o):
        s = self.s
        t = o["time_s"]
        angle = o["angle_rad"]
        if self.failure or self.mode in ["passive", "absent"]:
            return
        try:
            if o["unexpected_contacts"]:
                raise RuntimeError("Unexpected native robot contact")
            if "pre_event_motion_s" in s.events:
                raise RuntimeError("Valve moved before leak")
            p, R = s.grasp(0)
            if self.phase == "settle" and t > 1:
                self.transition("standby")
            if self.phase == "standby":
                # Hold the reset pose (no command is written) until the leak is in the scene:
                # ``leak_onset_s`` is set in the same observation that starts the leak stream,
                # turns the beacon red and switches the HMI text, so it is the first instant a
                # camera could show the cue. The oracle used to wait 2.5 cm above the handle.
                onset = s.events.get("leak_onset_s")
                if onset is not None and t >= onset + (5.5 if self.mode == "late" else 0):
                    self.transition("reach")
            if self.phase == "reach":
                # Joint-space move from the reset pose to 2.5 cm above the handle, at
                # ArmMotion's joint command rate. The gripper roll (joint6, about 2.3 rad from
                # the reset pose; the mirrored grasp would leave joint6's range during the
                # quarter turn) sets its duration.
                d, ids = s.dual, self.motion.ids
                above = p + [0, 0, 0.025]
                if self.reach_joints is None:
                    bp, bR = d.body_world("right", "base_link")
                    profile = d.manifest["profile"]
                    self.reach_joints, error = self.motion.kin.solve(
                        bR.T @ (above - bp),
                        bR.T @ R,
                        dict(zip(d.names["right"], d.commands["right"])),
                        profile["arm_lower_rad"],
                        profile["arm_upper_rad"],
                    )
                    if error > 0.003:
                        raise RuntimeError("Valve handle unreachable")
                rate = self.motion.command_speed_rad_s / CONTROL_HZ
                d.commands["right"][ids] += np.clip(
                    self.reach_joints - d.commands["right"][ids], -rate, rate
                )
                actual, actual_R = d.tool_world("right")
                # At rest (under 0.2 mm per control tick): the offset handed over below is
                # then the drives' gravity offset and not the lag of a moving arm.
                moving = self.reach_tool is None or np.linalg.norm(actual - self.reach_tool) > 0.0002
                self.reach_tool = actual.copy()
                if (
                    not moving
                    and np.abs(self.reach_joints - d.commands["right"][ids]).max() < 1e-9
                    and np.linalg.norm(actual - above) < 0.008
                    and Rotation.from_matrix(R @ actual_R.T).magnitude() < 0.05
                ):
                    # Hand over to the tool-space motion with the offset the drives hold the
                    # tool at (what its integrator would have reached hovering here).
                    limit = self.motion.position_correction_limit_m
                    self.motion.command = above.copy()
                    self.motion.correction = np.clip(above - actual, -limit, limit)
                    self.transition("approach")
            elif self.phase == "approach":
                if self.motion.move(p, 0.075, R):
                    self.transition("close")
            elif self.phase == "close":
                self.motion.move(p, 0.075, R)
                self.travel = max(0, self.travel - 0.04 / CONTROL_HZ)
                both = min(o["finger_forces_N"]) > 0.2
                self.grip_since = (
                    (self.grip_since if self.grip_since is not None else t) if both else None
                )
                if self.grip_since is not None and t - self.grip_since > 0.10:
                    self.grasped = True
                    s.events["bilateral_grasp_s"] = t
                    self.transition("turn")
            elif self.phase == "turn":
                self.travel = max(0, self.travel - 0.008 / CONTROL_HZ)
                self.angle_target = min(
                    np.pi / 2, max(0, angle) + 0.08, self.angle_target + 0.65 / CONTROL_HZ
                )
                p, R = s.grasp(self.angle_target)
                self.motion.move(p, 0.15, R)
                if s.physical_completed:
                    self.transition("release")
                if self.mode == "grasp" and self.grasped:
                    self.complete = True
            elif self.phase == "release":
                p, R = s.grasp(np.pi / 2)
                self.motion.move(p, 0.10, R)
                self.travel = min(0.044, self.travel + 0.04 / CONTROL_HZ)
                if min(o["jaws_m"]) > 0.041 and max(o["finger_forces_N"]) < 0.05:
                    self.transition("withdraw")
            elif self.phase == "withdraw":
                p, R = s.grasp(np.pi / 2)
                if self.motion.move(p + [0, -0.03, 0.05], 0.06, R):
                    self.complete = True
                    self.transition("complete")
            if s.physical_completed and not o["closed"]:
                raise RuntimeError("Valve reopened after isolation")
            if t - self.entered > 25 and self.phase not in ["standby", "complete"]:
                raise RuntimeError("Phase timeout: " + self.phase)
            s.dual.jaw_travel("right", self.travel)
        except RuntimeError as exc:
            self.failure = str(exc)
            print("FAILURE", self.failure, flush=True)
