"""Pinned single-GPU headless runtime for the October19 host.

Use NVIDIA EGL fallback from the existing diagnostics. Avoid global renderer
waitIdle startup barriers; captures explicitly synchronize at zero physics dt.
Shared system graphics configuration and historical runtime are unchanged.
"""

import os
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version

from .runtime_profiles import get_runtime_profile


class IndustryV2Runtime:
    def __init__(self, renderer=None, *, single_tick_rendering=False):
        self.profile = get_runtime_profile()
        self.renderer = self.profile.resolve_renderer(renderer)
        try:
            installed = version("isaacsim")
        except PackageNotFoundError as exc:
            raise RuntimeError("Isaac Sim is absent; use the selected profile's GPU environment") from exc
        self.profile.validate_environment(sys.version_info, installed)
        from roboclock.simulators.isaac.diagnostics import select_headless_driver

        self.driver_path = select_headless_driver()
        from isaacsim import SimulationApp

        launch = self.profile.launch_config(self.renderer)
        if single_tick_rendering:
            if not self.profile.migration_candidate:
                raise ValueError("Single-tick rendering override requires the explicit6.x profile")
            # Official6.x compatibility switch: explicit frozen-clock captures
            # must refresh even within one scheduled sensor tick.
            launch["extra_args"].append("--/rtx/hydra/supportMultiTickRate=false")
        self.app = SimulationApp(launch)
        if self.profile.migration_candidate:
            import omni.replicator.core as rep

            # Production capture is explicit and must not advance the task clock.
            # This policy needs native6.x task/readback validation before acceptance.
            rep.orchestrator.set_capture_on_play(False)
        import carb

        settings=carb.settings.get_settings()
        settings.set("/rtx/pathtracing/spp",8)
        settings.set("/rtx/pathtracing/totalSpp",64)
        settings.set("/rtx/pathtracing/maxBounces",6)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled",True)
        settings.set("/rtx/resetPtAccumOnAnimTimeChange",True)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is not None:
            traceback.print_exception(kind, value, tb)
        if self.driver_path:
            if os.environ.get("VK_ICD_FILENAMES") == self.driver_path:
                os.environ.pop("VK_ICD_FILENAMES")
            os.unlink(self.driver_path)
        import omni.kit.app

        omni.kit.app.get_app().post_quit(0 if kind is None else 1)
        if self.profile.migration_candidate:
            self.app.close(wait_for_replicator=False)
        else:
            self.app.close()
