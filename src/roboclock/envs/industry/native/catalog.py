"""Per-task native configuration: which scene/controller each task uses and how
its recorded runner stepped it.

Before the merge each task ran its own frozen source copy and one of four runner
scripts. The loop differences between those runners are kept here as explicit
per-task settings, so the standalone runner (``runner.py``) and the benchmark
environment (``envs/industry/env.py``) step every task the way it was recorded.
"""

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from roboclock.envs.rates import CONTROL_HZ
from roboclock.envs.robodojo_rig import PHYSICS_HZ, RENDER_MODE  # 250 Hz physics, spec renderer

from .paths import portable_json_loads, robot_manifest, scene_assets

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class NativeTask:
    number: int
    module: str
    scene: str
    controller: str
    resolve: str
    # plating (41), rotary (42), cell (43/44) or arx (45-50): recorded runner family.
    family: str
    physics_hz: int
    # Recorded runners called scene.observe() on every physics step for these tasks
    # (observe() also advances event timers), otherwise only on control ticks.
    observe_every_step: bool
    # Recorded runners stopped this long after the controller reported completion;
    # None means the episode always ran its full budget (throughput tasks).
    post_complete_s: float | None
    # 47-50 observe once more after the loop before writing the final result.
    final_observe: bool = False
    # 47/48 cached link transforms within one physics step (read-only, asserted).
    readback_cache: bool = False
    # Kit 6 single-tick capture switch; off for the spec renderer (the environment's
    # shared runtime never used it, and recordings advance the clock between frames).
    single_tick_rendering: bool = False
    # Kit launch profile and renderer: the robodojo_rig spec renderer for every task.
    runtime_profile: str = "isaac601-robodojo"
    renderer: str = RENDER_MODE
    # Kwargs the recorded runner passed to the scene beyond (manifest, output, resolved).
    scene_kwargs: tuple = ()

    @property
    def legacy(self) -> int:
        return self.number

    @property
    def control_stride(self) -> int:
        return self.physics_hz // CONTROL_HZ

    def load(self):
        module = importlib.import_module(f"{__package__}.{self.module}")
        return (
            getattr(module, self.scene),
            getattr(module, self.controller),
            getattr(module, self.resolve),
        )

    def make_scene(self, output, resolved, mode="nominal", *, manifest=None):
        scene_class, _, _ = self.load()
        manifest = Path(manifest or robot_manifest())
        if self.family in ("plating", "rotary"):
            return scene_class(scene_assets(self.number), manifest, output, resolved, **dict(self.scene_kwargs))
        return scene_class(manifest, output, resolved, mode)

    def interval(self, scene):
        """Per-physics-step contact accumulator that the runners cleared before each step."""
        return scene.interval if self.family in ("plating", "rotary") else scene.contacts.interval

    def reviewed(self) -> dict:
        return portable_json_loads((CONFIGS / f"task{self.number}/reviewed.json").read_text())

    def resolved(self, seed=None, *, layout=False, timing=False, physical=False) -> dict:
        """Episode specification; the reviewed replay for its own seed, else a fresh resolve."""
        reviewed = self.reviewed()
        randomized = layout or timing or physical
        if not randomized and (seed is None or int(seed) == int(reviewed["seed"])):
            spec = reviewed
        else:
            _, _, resolve = self.load()
            spec = portable_json_loads(
                json.dumps(resolve(int(seed or 0), layout=layout, timing=timing, physical=physical))
            )
        verify(spec)
        return spec


def _digest(resolved: dict) -> str:
    body = {k: v for k, v in resolved.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def verify(resolved: dict) -> dict:
    if _digest(resolved) != resolved.get("sha256"):
        raise ValueError(f"Resolved task {resolved.get('task_id')} hash mismatch")
    return resolved


def seal(resolved: dict) -> dict:
    """Recompute the hash after a deliberate edit (difficulty levels set their temporal keys)."""
    resolved["sha256"] = _digest(resolved)
    return resolved


TASKS = {
    task.number: task
    for task in (
        NativeTask(41, "plating_task", "PlatingScene", "PlatingController", "resolve_plating",
                   "plating", PHYSICS_HZ, True, 3.0),
        NativeTask(42, "rotary_task", "RotaryScene", "RotaryController", "resolve_rotary",
                   "rotary", PHYSICS_HZ, True, 3.0, single_tick_rendering=False,
                   scene_kwargs=(("grip_offset", (0.038, 0, 0.042)),)),
        NativeTask(43, "bonding_task", "BondingScene", "BondingController", "resolve_bonding",
                   "cell", PHYSICS_HZ, True, 2.5),
        NativeTask(44, "isolation_task", "IsolationScene", "IsolationController",
                   "resolve_isolation", "cell", PHYSICS_HZ, True, 2.5),
        NativeTask(45, "motor_arx_task", "MotorARXScene", "MotorARXController", "resolve_motor",
                   "arx", PHYSICS_HZ, True, 2.5),
        NativeTask(46, "bearing_arx_task", "BearingARXScene", "BearingARXController",
                   "resolve_bearing", "arx", PHYSICS_HZ, True, 2.5),
        NativeTask(47, "kitting_batch", "KittingARXScene", "KittingBatchController",
                   "resolve_kitting_batch", "arx", PHYSICS_HZ, False, None, final_observe=True,
                   readback_cache=True),
        # 48 on the common rig (putaway_rig.py); the recorded facing-arm cell is
        # putaway_throughput.PutawayThroughputScene.
        NativeTask(48, "putaway_rig", "RigPutawayScene", "RigPutawayController",
                   "resolve_putaway_rig", "arx", PHYSICS_HZ, False,
                   None, final_observe=True, readback_cache=True),
        NativeTask(49, "weigh_sort_task", "WeighSortScene", "WeighSortController",
                   "resolve_weigh_sort", "arx", PHYSICS_HZ, False, None, final_observe=True),
        NativeTask(50, "palette_arx_task", "PaletteARXScene", "PaletteARXController",
                   "resolve_palette", "arx", PHYSICS_HZ, False, None, final_observe=True),
    )
}
