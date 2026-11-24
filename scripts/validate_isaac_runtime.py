"""Actual RTX camera + PhysX gravity/support smoke test, without task assets."""

import argparse
import faulthandler
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from roboclock.simulators.isaac.runtime import IsaacRuntime
from roboclock.simulators.isaac.version import simulator_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    faulthandler.dump_traceback_later(120, repeat=True)
    with IsaacRuntime():
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdLux

        from roboclock.simulators.isaac.rendering import capture_camera

        world = World(stage_units_in_meters=1, physics_dt=1 / 240)
        world.scene.add_default_ground_plane()
        cube = world.scene.add(
            DynamicCuboid(
                "/World/Cube",
                name="cube",
                position=np.array([0.0, 0.0, 1.0]),
                scale=np.array([0.1, 0.1, 0.1]),
                color=np.array([0.8, 0.15, 0.05]),
                mass=1,
            )
        )
        import omni.usd

        light = UsdLux.DomeLight.Define(omni.usd.get_context().get_stage(), "/World/Light")
        light.CreateIntensityAttr(1000)
        position = np.array([2.0, -2.0, 1.6])
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(0, 0, 0.4), Gf.Vec3d(0, 0, 1))
        orientation = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
        camera = Camera("/World/Camera", resolution=(640, 480))
        world.reset()
        camera.initialize()
        camera.set_world_pose(position, orientation, camera_axes="usd")

        def frame(name):
            time = world.current_time
            for _ in range(3):
                world.render()
            capture_camera()
            assert time == world.current_time, "Rendering advanced the physics clock"
            rgb = np.asarray(camera.get_rgba()[:, :, :3], np.uint8).copy()
            assert rgb.shape == (480, 640, 3) and rgb.std() > 1
            Image.fromarray(rgb).save(args.output / name)
            return rgb

        initial = cube.get_world_pose()[0].copy()
        first = frame("initial.png")
        # Allow impact, transient rocking and settling before checking support.
        for _ in range(720):
            world.step(render=False)
        final = cube.get_world_pose()[0].copy()
        velocity = cube.get_linear_velocity().copy()
        last = frame("final.png")
        difference = float(np.abs(first.astype(float) - last.astype(float)).mean())
        result = dict(
            simulator=simulator_label(),
            python_environment=sys.executable,
            physics_steps=720,
            physics_hz=240,
            initial_position=initial.tolist(),
            final_position=final.tolist(),
            final_velocity=velocity.tolist(),
            mean_frame_difference=difference,
            frozen_time_capture=True,
            gravity_and_support_pass=bool(
                initial[2] - final[2] > 0.8
                and 0.045 < final[2] < 0.065
                and np.linalg.norm(velocity) < 0.1
            ),
            camera_pass=difference > 0.05,
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT", json.dumps(result), flush=True)
        assert result["gravity_and_support_pass"], result
        assert result["camera_pass"], "Camera buffer did not reflect the simulated fall"
        faulthandler.cancel_dump_traceback_later()
        world.stop()
        World.clear_instance()


if __name__ == "__main__":
    main()
