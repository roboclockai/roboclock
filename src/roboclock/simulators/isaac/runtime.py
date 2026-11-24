"""Own one SimulationApp per process; initialize before any Omniverse imports."""

import os
import sys
import traceback
from pathlib import Path

from roboclock.simulators.isaac.version import installed_version, required_python


def simulator_version():
    return installed_version() or "unavailable"


class IsaacRuntime:
    def __init__(self, headless=True, renderer="RaytracedLighting"):
        self.driver_path = None
        self.exit_code = 0
        self.version = installed_version()
        expected_python = required_python(self.version)
        if self.version is None:
            raise RuntimeError("Isaac Sim is unavailable. Install requirements.txt; see README.md.")
        if expected_python is None:
            raise RuntimeError(f"Unsupported Isaac Sim version: {self.version}")
        if sys.version_info[:2] != expected_python:
            raise RuntimeError(
                f"Isaac Sim {self.version} requires Python {'.'.join(map(str, expected_python))}. "
                "Activate the matching environment; see README.md."
            )
        if sys.platform.startswith("linux"):
            from roboclock.simulators.isaac.diagnostics import graphics_status, select_headless_driver

            if headless:
                self.driver_path = select_headless_driver()

            graphics = graphics_status()
            missing = [name for name, available in graphics["libraries"].items() if not available]
            if missing:
                raise RuntimeError(
                    f"Missing Isaac Sim graphics libraries: {', '.join(missing)}. See README.md."
                )
            if graphics["vulkan_gpu_enumeration"] is False:
                raise RuntimeError(
                    "Vulkan cannot enumerate a GPU. Check the NVIDIA graphics driver/container graphics capability; see README.md."
                )
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Sim could not be imported. Check the active environment and requirements.txt."
            ) from exc
        portable_root = os.environ.get("ROBOCLOCK_ISAAC_PORTABLE_ROOT")
        portable_args = ["--portable-root", portable_root] if portable_root else []
        app_options = {}
        if self.version.startswith("6.0."):
            import isaacsim

            extra_args = [
                "--ext-folder",
                str(Path(isaacsim.__file__).parent / "extsDeprecated"),
                # Kit 110 can omit inactive RTX modes at startup. Selecting a
                # disabled mode later produces black camera color buffers.
                "--/persistent/rtx/modes/pt/enabled=true",
                "--/persistent/rtx/modes/rt/enabled=true",
                "--/persistent/rtx/modes/rt2/enabled=true",
            ]
            for extension in (
                "isaacsim.core.api",
                "isaacsim.core.prims",
                "isaacsim.core.utils",
                "isaacsim.robot.manipulators",
                "isaacsim.robot_motion.motion_generation",
                "isaacsim.sensors.camera",
            ):
                extra_args += ["--enable", extension]
            extra_args += portable_args + [
                "--/app/renderer/waitIdle=true",
                "--/app/hydraEngine/waitIdle=true",
            ]
        else:
            # Isaac Sim 5.1 and 6.2 (sports / chess recordings).
            app_options["anti_aliasing"] = 4
            extra_args = portable_args + [
                "--/app/updateOrder/checkForHydraRenderComplete=1000",
                "--/app/renderer/waitIdle=true",
                "--/app/hydraEngine/waitIdle=true",
            ]
            if self.version.startswith("6."):
                # NVIDIA still ships this compatibility extension but no longer enables
                # it in the base Python experience; keep the extension set the sports
                # and chess recordings ran with.
                extra_args += [
                    "--enable",
                    "isaacsim.robot.manipulators.examples",
                    "--/persistent/app/viewport/Viewport/Viewport0/scene/cameras/visible=false",
                ]
        self.app = SimulationApp(
            {
                "headless": headless,
                "create_new_stage": not headless,
                "disable_viewport_updates": headless,
                "limit_cpu_threads": int(os.environ.get("ROBOCLOCK_CPU_THREADS", "32")),
                "width": 1280,
                "height": 720,
                "fast_shutdown": True,
                "renderer": renderer,
                "samples_per_pixel_per_frame": 8,
                "extra_args": extra_args,
                **app_options,
            }
        )

    def close(self):
        # Kit's fast shutdown exits the process. Release our temporary driver file
        # first, and preserve the caller's error status through Kit's quit request.
        if self.driver_path:
            if os.environ.get("VK_ICD_FILENAMES") == self.driver_path:
                os.environ.pop("VK_ICD_FILENAMES", None)
            os.unlink(self.driver_path)
            self.driver_path = None
        import omni.kit.app

        omni.kit.app.get_app().post_quit(self.exit_code)
        self.app.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, tb):
        if exc_type is not None:
            self.exit_code = 130 if issubclass(exc_type, KeyboardInterrupt) else 1
            traceback.print_exception(exc_type, value, tb)
        self.close()
