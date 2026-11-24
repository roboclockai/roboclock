"""Tasks on the native-X5 workflow backend: A2 (basketball, arena kind A3), B2, C2 (arena), B3, B4, C1 and C3.

Mirrors the recording drivers (``arena/demo.py``, ``_shared/precision/demo.py``,
``freeze/demo.py``, ``pool_rack/demo.py``, ``ball_service/demo.py``): one
``ReplacementBackend``, the sub-package's ``Task`` evaluator updated on every
physics tick, and its ``Oracle`` acting once per control period by writing the
backend command. ``ReplacementBackend.tick`` re-applies the command with gravity
feed-forward on every physics step, which is the robot's drive model.
"""

import numpy as np

from roboclock.envs.sports.adapters.base import Adapter, episode_rates


class WorkflowAdapter(Adapter):
    family = "workflow"

    def native_task(self, b):
        raise NotImplementedError

    def native_oracle(self, b, t):
        raise NotImplementedError

    def update_cues(self, b, t):
        raise NotImplementedError

    def build(self):
        from roboclock.simulators.isaac.embodiments.backend import ReplacementBackend
        from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_camera

        b = ReplacementBackend(self.runtime, self.c)
        self.backend = b
        b.reset()
        apply_presentation_camera(b)
        self.t = self.native_task(b)
        self.update_cues(b, self.t)
        self.after_build()

    @property
    def time(self):
        return float(self.backend.time)

    @property
    def done(self):
        return bool(self.t.done)

    def command(self):
        return np.array(self.backend.command, copy=True)

    def apply(self, native):
        # In place: the rig and backend share this array, and native controllers
        # write into it between control periods.
        self.backend.command[:] = native

    def tick(self):
        state = self.backend.tick()
        self.t.update(state)
        self.ticks += 1

    def after_control(self):
        self.update_cues(self.backend, self.t)

    def events(self):
        return self.t.events

    def result(self):
        b, t = self.backend, self.t
        r = t.result()
        r.update(
            full_episode=bool(t.done),
            difficulty=self.spec.difficulty,
            seed=self.spec.seed,
            embodiment=self.c["embodiment"],
            native_joint_names=list(b.rig.names),
            actuated_joint_names=list(b.rig.actuated_names),
            evaluation_end=float(b.time),
            recorded_sim_time=float(b.time),
            initial_ik=dict(b.rig.initial_ik),
            **episode_rates(self),
        )
        for key in ("actual_drive_limits", "actual_drive_gains"):
            if hasattr(b, key):
                r[key] = getattr(b, key)
        transitions = getattr(getattr(self.controller, "oracle", None), "transitions", None)
        if transitions is not None:
            r["controller_transitions"] = transitions
        return r

    def joint_success(self):
        return bool(self.t.result()["success"])

    def make_controller(self):
        self.controller = WorkflowController(self, self.native_oracle(self.backend, self.t))
        return self.controller


class WorkflowController:
    """The sub-package Oracle; it writes ``backend.command`` like in the recordings."""

    def __init__(self, adapter, oracle):
        self.adapter, self.oracle = adapter, oracle

    @property
    def phase(self):
        return getattr(self.oracle, "phase", None)

    def act(self):
        self.oracle.act()
        return self.adapter.command()


class ArenaAdapter(WorkflowAdapter):
    """A3 basketball, B2 boxing, C2 speed cups (``arena/``)."""

    def native_task(self, b):
        from roboclock.envs.sports.arena.tasks import Task

        return Task(b)

    def native_oracle(self, b, t):
        from roboclock.envs.sports.arena.oracle import Oracle

        return Oracle(b, t)

    def update_cues(self, b, t):
        from roboclock.envs.sports.arena.cues import update_cues

        update_cues(b, t)


class BenchAdapter(WorkflowAdapter):
    """B3 bench-press assistance (``bench_press/`` on ``_shared/precision/``).

    The lifter is a bounded human wrench applied to the bar inside the backend
    tick (``bench_press/process.py``); the evaluator releases it on a measured
    catch. The skinned athlete is visual and follows the bar before each frame.
    """

    def native_task(self, b):
        from roboclock.envs.sports._shared.precision.tasks import Task

        return Task(b)

    def native_oracle(self, b, t):
        from roboclock.envs.sports._shared.precision.oracle import Oracle

        return Oracle(b, t)

    def update_cues(self, b, t):
        from roboclock.envs.sports._shared.workflow.scenes import update_cues

        update_cues(b, t)

    def before_capture(self):
        b = self.backend
        b.athlete.update(
            b.last_state["objects"]["bar"]["position"],
            b.human_effort is not None and b.human_effort.released,
        )


class FreezeAdapter(WorkflowAdapter):
    """B4 stop/go puck pushing (``freeze/``)."""

    def native_task(self, b):
        from roboclock.envs.sports.freeze.task import Task

        return Task(b)

    def native_oracle(self, b, t):
        from roboclock.envs.sports.freeze.oracle import Oracle

        return Oracle(b, t)

    def update_cues(self, b, t):
        from roboclock.envs.sports.freeze.scene import update_cues

        update_cues(b, t)


class RackAdapter(WorkflowAdapter):
    """C1 fifteen-ball racking (``pool_rack/``)."""

    def native_task(self, b):
        from roboclock.envs.sports.pool_rack.task import Task

        return Task(b)

    def native_oracle(self, b, t):
        from roboclock.envs.sports.pool_rack.oracle import Oracle

        return Oracle(b, t)

    def update_cues(self, b, t):
        from roboclock.envs.sports._shared.workflow.scenes import update_cues

        update_cues(b, t)


class ServiceAdapter(WorkflowAdapter):
    """C3 ball service (``ball_service/``); the task fires the hopper gates on schedule."""

    def native_task(self, b):
        from roboclock.envs.sports.ball_service.task import Task

        return Task(b)

    def native_oracle(self, b, t):
        from roboclock.envs.sports.ball_service.oracle import Oracle

        return Oracle(b, t)

    def update_cues(self, b, t):
        from roboclock.envs.sports._shared.workflow.scenes import update_cues

        update_cues(b, t)
