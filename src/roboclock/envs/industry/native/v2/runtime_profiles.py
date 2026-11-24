"""Explicit simulator launch candidates; no robot, physics or scoring changes.

The default is the Isaac Sim 6.0.1 RT2 profile used by the recordings; Task 41 records with
the 6.0.1 path-tracing profile.6.x profiles reproduce the isolated RGB probes only: native
robot/contact/readback acceptance is pending.
Importing this module does not import Isaac Sim or start a graphics application.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    python_version: tuple[int, int]
    package_versions: tuple[str, ...]
    default_renderer: str
    allowed_renderers: tuple[str, ...]
    migration_candidate: bool = False

    def validate_environment(self, python_version, package_version):
        if tuple(python_version[:2]) != self.python_version:
            required = ".".join(map(str, self.python_version))
            raise RuntimeError(f"{self.name} requires Python{required}; use its isolated environment")
        if package_version not in self.package_versions:
            raise RuntimeError(
                f"{self.name} requires isaacsim=={self.package_versions[0]}; "
                f"found {package_version}. Simulator versions cannot be silently substituted"
            )

    def resolve_renderer(self, renderer=None):
        selected = renderer or os.environ.get("ROBOCLOCK_V2_RENDERER", self.default_renderer)
        if selected not in self.allowed_renderers:
            raise ValueError(f"Renderer {selected!r} is not declared for profile {self.name}")
        return selected

    def launch_config(self, renderer=None):
        config = dict(
            headless=True, renderer=self.resolve_renderer(renderer), width=1280, height=720,
            multi_gpu=False, max_gpu_count=1, active_gpu=0,
            extra_args=[
                "--/rtx/post/dlss/enabled=false",
                "--/rtx/post/aa/op=0",
                "--/app/asyncRendering=false",
            ],
        )
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            # activeGpu indexes Kit's Vulkan devices, which ignore CUDA_VISIBLE_DEVICES;
            # pinning 0 there selects a GPU with no visible CUDA twin and RTX fails.
            # Unpinned, Kit renders on the CUDA-visible GPU (physics stays on cuda:0).
            config.pop("active_gpu")
        if self.migration_candidate:
            config.update(limit_cpu_threads=4, sync_loads=False)
            config["extra_args"].append("--/app/hydraEngine/waitIdle=false")
        return config

    def metadata(self, renderer=None):
        return dict(
            name=self.name, required_python=list(self.python_version),
            allowed_isaacsim_versions=list(self.package_versions),
            launch_config=self.launch_config(renderer),
            capture_on_play=False if self.migration_candidate else "existing SDK default",
            close_wait_for_replicator=False if self.migration_candidate else True,
            acceptance_scope=(
                "Experimental6.x RGB launch candidate; native task, tensor/contact freshness, "
                "material completeness and continuous recording validation pending"
                if self.migration_candidate else "Historical5.1 runtime; consult per-case evidence"
            ),
        )


PROFILES = {
    "isaac601-pt-stability": RuntimeProfile(
        "isaac601-pt-stability", (3, 12), ("6.0.1.0", "6.0.1"),
        "PathTracing", ("PathTracing",), True,
    ),
    "isaac51": RuntimeProfile(
        "isaac51", (3, 11), ("5.1.0", "5.1.0.0"),
        "PathTracing", ("PathTracing", "RaytracedLighting"),
    ),
    "isaac601-rt2": RuntimeProfile(
        "isaac601-rt2", (3, 12), ("6.0.1.0", "6.0.1"),
        "RealTimePathTracing", ("RealTimePathTracing",), True,
    ),
    # robodojo_rig spec renderer (RaytracedLighting + DLAA + Isaac Lab quality preset),
    # applied after launch by native/rig.apply_renderer.
    "isaac601-robodojo": RuntimeProfile(
        "isaac601-robodojo", (3, 12), ("6.0.1.0", "6.0.1"),
        "RaytracedLighting", ("RaytracedLighting",), True,
    ),
    "isaac610-rt2": RuntimeProfile(
        "isaac610-rt2", (3, 12), ("6.1.0.0", "6.1.0"),
        "RealTimePathTracing", ("RealTimePathTracing",), True,
    ),
}


def get_runtime_profile(name=None):
    selected = name if name is not None else os.environ.get("ROBOCLOCK_V2_RUNTIME_PROFILE", "isaac601-robodojo")
    if selected not in PROFILES:
        raise ValueError(f"Unknown runtime profile {selected!r}; choose {', '.join(PROFILES)}")
    return PROFILES[selected]
