"""Task 41 video capture: fresh path-traced frames of the frozen native state.

The October 29 revision for stable reflections and clear moving edges. Each frame
gets a new render product at the same camera pose, fixed exposure and spatial OptiX
denoising, read directly from an RGB annotator: a persistent product carries
radiance and denoiser history from the previous arm pose. Renderer-only; every
capture checks that physics, the part and the camera did not change.
"""

import json
import time
from pathlib import Path

import numpy as np

SHAPE = (800, 1280)
RENDER_SETTINGS = {
    "/rtx/rendermode": "PathTracing",
    "/rtx/post/aa/op": 0,
    "/rtx/post/dlss/execMode": 2,
    "/rtx/post/histogram/enabled": False,
    "/rtx/pathtracing/spp": 4,
    "/rtx/pathtracing/totalSpp": 32,
    "/rtx/pathtracing/clampSpp": 32,
    "/rtx/pathtracing/maxBounces": 3,
    "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 3,
    "/rtx/pathtracing/maxVolumeBounces": 2,
    "/rtx/pathtracing/cached/enabled": False,
    "/rtx/pathtracing/optixDenoiser/enabled": True,
    "/rtx/pathtracing/optixDenoiser/blendFactor": 0.0,
    "/rtx/pathtracing/optixDenoiser/temporalMode/enabled": False,
    "/rtx/resetPtAccumOnAnimTimeChange": True,
    "/omni/replicator/pathTracedMotionBlurSubSamples": 0,
}


class FreshCapture:
    def __init__(self, scene, output, subframes, base_settings):
        import carb

        self.scene, self.output, self.subframes = scene, Path(output), subframes
        self.settings = carb.settings.get_settings()
        self.render_settings = {**base_settings, **RENDER_SETTINGS}
        self.audit = []
        self.annotator = None
        self._apply_settings()
        self._sync_signal_lamps()

    def _apply_settings(self):
        for key, value in self.render_settings.items():
            self.settings.set(key, value)

    def _sync_signal_lamps(self):
        """Under single-tick rendering only Fabric reaches the image; the lamps are graphics."""
        from isaacsim.core.prims import XFormPrim
        from pxr import UsdPhysics

        from .arx_cell_task import set_visual_target

        scene = self.scene
        views = {}
        for name, sphere in scene.signal_lights.items():
            prim = sphere.GetPrim()
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            view = scene.world.scene.add(
                XFormPrim(str(prim.GetPath()), name="process_signal_" + name, reset_xform_properties=True)
            )
            view.initialize()
            views[name] = view
        update_signal = scene.update_signal
        synced = [None]

        def synced_update(observation):
            update_signal(observation)
            if scene.signal_state != synced[0]:
                for name, view in views.items():
                    set_visual_target(view, np.asarray(scene.signal_positions[name][0].Get()))
                synced[0] = scene.signal_state

        scene.update_signal = synced_update

    def _physics(self):
        scene = self.scene
        return dict(
            time=scene.world.current_time,
            tick=scene.world.current_time_step_index,
            part=[np.array(v, copy=True) for v in scene.part.get_world_pose()],
            joints={s: r.get_joint_positions().copy() for s, r in scene.dual.robots.items()},
        )

    def _unchanged(self, before):
        now = self._physics()
        return (
            now["time"] == before["time"]
            and now["tick"] == before["tick"]
            and all(np.array_equal(a, b) for a, b in zip(before["part"], now["part"], strict=True))
            and all(np.array_equal(before["joints"][s], now["joints"][s]) for s in before["joints"])
        )

    def warm_up(self):
        """Attach the RGB annotator and warm it at the frozen initial state."""
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd

        from roboclock.simulators.isaac.rendering import capture_camera

        scene = self.scene
        self.renderer = omni.usd.get_context()
        self.timeline = omni.timeline.get_timeline_interface()
        # Camera's scheduled-frame cache is unsuitable for repeated zero-delta-time captures.
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([scene.camera.get_render_product_path()])
        before, process_start = self._physics(), scene.process_start
        scene.update_signal({"elapsed_s": 0.0})
        valid = 0
        for _ in range(4):
            scene.world.render()
            capture_camera(1)
            data = np.asarray(self.annotator.get_data())
            valid += data.shape[:2] == SHAPE and data.std() > 5
        unchanged = self._unchanged(before) and scene.process_start == process_start
        (self.output / "warmup-audit.json").write_text(
            json.dumps(
                dict(
                    physics_unchanged=unchanged,
                    valid_warm_frames=int(valid),
                    render_product=scene.camera.get_render_product_path(),
                    scope="Persistent direct RGB annotator warmed at frozen initial physics state; "
                    "no recipe or clock advancement.",
                )
            )
            + "\n"
        )
        if not unchanged or valid < 2:
            raise RuntimeError("Direct RGB warm-up failed")

    def rgb(self):
        return np.asarray(self.annotator.get_data())[:, :, :3].astype(np.uint8)

    def frame(self):
        """One video frame of the current state; the physics clock stays frozen."""
        from PIL import Image

        from roboclock.simulators.isaac.rendering import capture_camera

        scene = self.scene
        started = time.monotonic()
        camera = [np.array(v, copy=True) for v in scene.camera.get_world_pose()]
        intrinsics = np.asarray(scene.camera.get_intrinsics_matrix()).copy()
        before = self._physics()
        # Renderer/MDL animation time follows the reached simulation time; not a physics step.
        self.timeline.set_current_time(float(before["time"]))
        # A fresh product resets history and the static-scene sampling budget.
        self.annotator.detach([scene.camera.get_render_product_path()])
        scene.camera.destroy()
        scene.camera.initialize()
        self.annotator.attach([scene.camera.get_render_product_path()])
        self._apply_settings()
        self.renderer.reset_renderer_accumulation()
        # A transient empty or black GPU readback is not a frame: retry the same state.
        for attempt in range(6):
            scene.world.render()
            capture_camera(self.subframes)
            data = np.asarray(self.annotator.get_data())
            shaped = data.ndim == 3 and data.shape[:2] == SHAPE and data.shape[2] >= 3
            image = data[:, :, :3].astype(np.uint8) if shaped else np.empty((0, 0, 3), dtype=np.uint8)
            if shaped and image.std() > 5:
                break
            with (self.output / "invalid-readbacks.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        dict(
                            time_s=float(before["time"]),
                            attempt=attempt + 1,
                            shape=list(data.shape),
                            std=float(data.std()) if data.size else None,
                        )
                    )
                    + "\n"
                )
            if shaped:
                name = f"invalid_readback_{len(self.audit):04d}_{attempt}.png"
                Image.fromarray(image).save(self.output / name)
        else:
            raise RuntimeError("No valid actual camera readback after six frozen-state attempts")
        if not self._unchanged(before):
            raise RuntimeError("Capture changed measured physics state")
        if not (
            all(np.array_equal(a, b) for a, b in zip(camera, scene.camera.get_world_pose(), strict=True))
            and np.array_equal(intrinsics, scene.camera.get_intrinsics_matrix())
        ):
            raise RuntimeError("Fixed camera pose/intrinsics changed")
        with (self.output / "capture-diagnostics.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    dict(
                        time_s=float(before["time"]),
                        capture_wall_s=time.monotonic() - started,
                        camera_unchanged=True,
                        renderer_timeline_s=float(self.timeline.get_current_time()),
                        readback_attempts=attempt + 1,
                        actual_aa=self.settings.get("/rtx/post/aa/op"),
                        actual_spp=self.settings.get("/rtx/pathtracing/spp"),
                        actual_total_spp=self.settings.get("/rtx/pathtracing/totalSpp"),
                        pt_motion_subsamples=self.settings.get(
                            "/omni/replicator/pathTracedMotionBlurSubSamples"
                        ),
                        static_floor_mean=float(image[650:780, 5:220].mean()),
                        static_floor_std=float(image[650:780, 5:220].std()),
                    )
                )
                + "\n"
            )
        self.audit.append(
            dict(
                camera_unchanged=True,
                time_s=float(before["time"]),
                physics_tick=int(before["tick"]),
                physics_state_unchanged=True,
                fresh_render_product=True,
            )
        )
        return Image.fromarray(image)

    def profile(self):
        """Recorded capture settings, as read back from Kit."""
        return dict(
            subframes=self.subframes,
            history="fresh",
            single_tick_rendering=True,
            existing_nonphysical_lamps_synced_to_fabric=True,
            direct_rgb_annotator=True,
            persistent_render_product=False,
            fresh_render_product=True,
            fixed_exposure=True,
            settings={key: self.settings.get(key) for key in self.render_settings},
        )

    def write_audit(self):
        (self.output / "capture-audit.json").write_text(json.dumps(self.audit, indent=2) + "\n")
