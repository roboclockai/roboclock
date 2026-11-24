"""Synchronized robot-view recordings from measured native X5 poses."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def look_pose(eye, target, up):
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    return np.asarray(eye), gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())


class RobotViewEncoder:
    """Keep the overview and two physical camera mounts on identical timestamps.

    The ego camera is a benchmark virtual camera rigidly mounted above the base.
    The left camera follows a fixed optical transform on the measured wrist. It
    never tracks privileged object locations or changes the robot's joint state.
    """

    def __init__(self, encoder, backend, output, resolution):
        from isaacsim.sensors.camera import Camera

        from roboclock.simulators.isaac.video import VideoEncoder

        self.encoder, self.backend = encoder, backend
        self.output = Path(output)
        self.resolution = list(resolution)
        self.cameras, self.encoders, self.timestamps = {}, {}, []
        views = ("ego", "left_wrist") if backend.config["visuals"].get("record_robot_views", True) else ()
        for name in views:
            path = (
                "/World/RobotViews/ego"
                if name == "ego"
                else backend.rig.tools["left"].prim_path + "/BenchmarkCamera"
            )
            cam = Camera(path, resolution=tuple(resolution))
            cam.initialize()
            cam.set_focal_length(0.015 if name == "ego" else 0.018)
            cam.set_horizontal_aperture(0.036)
            cam.set_clipping_range(0.01, 100)
            self.cameras[name] = cam
            self.encoders[name] = VideoEncoder(self.output / (name + ".mp4"), resolution, 15)
        self.update_poses()

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def update_poses(self):
        if not self.cameras:
            return
        c = self.backend.config["embodiment"]
        base = np.asarray(c["position"])
        r = Rotation.from_euler("z", c.get("yaw", 0))
        eye = base + r.apply([-0.20, 0, 0.58])
        target = base + r.apply([0.45, 0.10, 0.12])
        self.cameras["ego"].set_world_pose(*look_pose(eye, target, [0, 0, 1]), camera_axes="usd")
        # Parenting the sensor to the rigid wrist makes USD/PhysX update its
        # optical pose before every render, without a one-frame delayed target.
        eye = np.array([0.088, 0, 0.055])
        forward, up = np.array([0.940, 0, -0.342]), np.array([0.342, 0, 0.940])
        self.cameras["left_wrist"].set_local_pose(
            *look_pose(eye, eye + forward, up), camera_axes="usd"
        )

    def write(self, rgb):
        from roboclock.simulators.isaac.rendering import capture_camera

        before = float(self.backend.world.current_time)
        if not self.timestamps and self.cameras:
            capture_camera(subframes=8)
        assert float(self.backend.world.current_time) == before
        self.encoder.write(rgb)
        task_time = getattr(self.backend, "time", None)
        if task_time is None:
            task_time = self.backend.task.time
        self.timestamps.append(float(task_time))
        for name, camera in self.cameras.items():
            rgba = camera.get_rgba()
            if rgba is None or np.asarray(rgba).ndim != 3:
                raise RuntimeError(
                    f"Missing RGB buffer for {name}; inspect the renderer/GPU-memory log. "
                    "No completed video or evaluation is exported for this run."
                )
            frame = np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()
            self.encoders[name].write(frame)
            if len(self.timestamps) == 1:
                Image.fromarray(frame).save(self.output / (name + "_initial.png"))
            self.last_frames = getattr(self, "last_frames", {})
            self.last_frames[name] = frame

    def close(self):
        self.encoder.close()
        for name, encoder in self.encoders.items():
            encoder.close()
            Image.fromarray(self.last_frames[name]).save(self.output / (name + "_final.png"))
        (self.output / "views.json").write_text(
            json.dumps(
                dict(
                    frames=len(self.timestamps),
                    simulation_timestamps=self.timestamps,
                    cameras={name: description for name, description in {
                        "overview": "Task presentation camera",
                        "ego": "Fixed benchmark mast, virtual sensor",
                        "left_wrist": "Fixed extrinsics on measured left native wrist",
                    }.items() if name == "overview" or name in self.cameras},
                    resolution=self.resolution,
                    fps=15,
                    synchronized=True,
                    physics_advanced_during_capture=False,
                ),
                indent=2,
            )
        )


def attach_robot_views(encoder, backend, output):
    if encoder is None or backend.config.get("embodiment", {}).get("model") != "dual_arx_x5":
        return encoder
    return RobotViewEncoder(encoder, backend, output, backend.config["sim"]["camera_resolution"])


def observation_pose(position, yaw=0, side="left"):
    """Collision-clear auxiliary-hand observation pose for the fixed X5 mounting frame."""
    sign = 1 if side == "left" else -1
    base_rotation = Rotation.from_euler("z", yaw)
    target = np.asarray(position) + base_rotation.apply([0.25, sign * 0.36, 0.12])
    rotation = (
        base_rotation.as_matrix()
        @ Rotation.from_euler("z", -50 * sign, degrees=True).as_matrix()
        @ Rotation.from_euler("y", 125, degrees=True).as_matrix()
    )
    return target, rotation, 0.015
