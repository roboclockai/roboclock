"""Shared measurement base of the native X5 ``ReplacementBackend``: contacts, state and frames."""

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.sports.common.backend import SportsBackend
from roboclock.envs.sports.joints import joint_step_speed
from roboclock.simulators.isaac.rig_view import RigView

# Each task's joint speed limit: (section, key, default, control rate it was tuned at).
# C1 rate-limits the command itself (``move_cue_arm``), which needs no restating.
JOINT_SPEED_LIMITS = {
    "A3": ("basketball", "joint_speed", 1.7, 30),
    "B2": ("service", "joint_speed", 1.7, 30),
    "B3": ("workflow", "bench_joint_speed", 1.7, 60),
    "B4": ("freeze", "joint_speed", 1.7, 30),
    "C2": ("service", "joint_speed", 1.7, 30),
    "C3": ("service", "joint_speed", 1.7, 30),
}


def at_control_rate(config):
    """The task configuration with its joint speed limit restated for the control rate."""
    if config["task_id"] not in JOINT_SPEED_LIMITS:
        return config
    section, key, default, tuned_hz = JOINT_SPEED_LIMITS[config["task_id"]]
    dt = config["sim"]["physics_dt"] * config["sim"]["control_decimation"]
    values = config.get(section, {})
    speed = joint_step_speed(values.get(key, default), tuned_hz, dt)
    return {**config, section: {**values, key: speed}}


class Backend(RigView):
    collider = SportsBackend.collider

    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, at_control_rate(config)
        self.world = self.camera = None
        self.kind = config["task_id"]
        self.single_arm = self.kind != "B3"
        self.mobile = self.kind in ("B4", "C3")
        self.objects = {}
        self.pads = {}
        self.kin = {}
        self.tool_links = {}
        self.mount = 0.45
        self.tcp_offset = 0.17
        self.time = 0.0
        self.events = []
        self.hand_targets = {}

    def measure(self):
        force = self.contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        positions, orientations = self.contacts.get_world_poses()
        velocities = self.contacts.get_velocities()
        out = {
            "time": self.time,
            "objects": {},
            "hands": {},
            "joints": self.robot.get_joint_positions().copy(),
        }
        for name in [*self.objects, *self.pads]:
            row = self.contact_rows[name]
            p, q = positions[row], orientations[row]
            v, w = velocities[row, :3], velocities[row, 3:]
            f = np.linalg.norm(force[row], axis=-1)
            out["objects"][name] = dict(
                position=p.copy(),
                orientation=q.copy(),
                velocity=v.copy(),
                angular_velocity=w.copy(),
                forces=f.copy(),
                support_force_z=float(max(0.0, np.sum(force[row, len(self.fingers) :, 2])))
                if self.kind != "B2"
                else 0.0,
            )
            if True:
                out["objects"][name]["support_contacts"] = {
                    path.rsplit("/", 1)[-1]: float(max(0.0, force[row, j, 2]))
                    for j, path in enumerate(self.filter_paths)
                    if j >= len(self.fingers)
                }
        arm_force = self.arm_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        out["arm_contact_peak"] = float(np.max(np.linalg.norm(arm_force, axis=-1)))
        for side in ["main"] if self.single_arm else ["left", "right"]:
            p, q = self.hand_pose(side)
            out["hands"][side] = dict(position=p.copy(), orientation=q.copy())
        if self.kind == "B2":
            normal, points, _, _, counts, starts = self.contacts.get_contact_force_data(
                dt=self.config["sim"]["physics_dt"]
            )
            for name in self.pads:
                row = self.contact_rows[name]
                valid = []
                for j, side in enumerate(["left", "right"]):
                    p, q = self.tool_links[side].get_world_pose()
                    first, count = int(starts[row, j]), int(counts[row, j])
                    local = (np.asarray(points[first : first + count]) - p) @ Rotation.from_quat(
                        np.roll(q, -1)
                    ).as_matrix()
                    from roboclock.simulators.isaac.embodiments.tools import glove_contact_mask

                    inside = glove_contact_mask(local, getattr(self, "glove_rotation", None))
                    valid.append(
                        float(np.sum(np.abs(normal[first : first + count].reshape(-1)[inside])))
                    )
                out["objects"][name]["glove_forces"] = valid
        numeric = [out["joints"], force, out["arm_contact_peak"]]
        for body in out["objects"].values():
            numeric.extend(
                body[k] for k in ("position", "orientation", "velocity", "angular_velocity")
            )
        if not all(np.isfinite(value).all() for value in numeric):
            raise FloatingPointError("Non-finite PhysX state: invalid simulation episode")
        return out

    def frame(self):
        from roboclock.simulators.isaac.rendering import capture_camera

        before = float(self.world.current_time)
        for _ in range(3):
            self.world.render()
        # Settle current lighting and changed labels without advancing physics.
        capture_camera(subframes=8)
        rgb = np.asarray(self.camera.get_rgba()[:, :, :3], dtype=np.uint8).copy()
        if self.config["visuals"].get("render_scale", 1) != 1:
            from PIL import Image

            rgb = np.asarray(
                Image.fromarray(rgb).resize(
                    tuple(self.config["sim"]["camera_resolution"]), Image.Resampling.LANCZOS
                )
            ).copy()
        assert before == float(self.world.current_time)
        return rgb
