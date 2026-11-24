"""Task lifecycle and independent physical/temporal scoring for the hospital domain.

All timestamps use elapsed physics time. Temporal violations stay latched while the episode
continues, so eventual manipulation success is measured separately from timing compliance.
Entry into a container is the first downward crossing of its rim, recorded before settling.
"""

from dataclasses import asdict

import numpy as np

from roboclock.envs.hospital.config import EpisodeSpec
from roboclock.envs.hospital.mdp.events import (
    GREEN,
    RED,
    SceneCommand,
    timer_color,
    window_color,
    window_error,
)
from roboclock.envs.hospital.mdp.geometry import bottom_z, entry_crossing, fits_xy, seated
from roboclock.envs.lab.mdp.state import PhysicalState, upright_error
from roboclock.evaluation import throughput

EPS = 1e-9
# The dual X5's fingertips project this far below its tool point (`ee_position`, the IK grasp
# center). A press is the closed fingertips on the button top, so press checks measure from them.
FINGERTIP_BELOW_TOOL = 0.01257


def pressing_button(ee, button):
    """Closed X5 fingertips on a button top (`button` = top-center point): within 16 mm in xy,
    fingertips between 20 mm under and 12 mm over the top surface."""
    tip = float(ee[2]) - FINGERTIP_BELOW_TOOL
    return (
        float(np.linalg.norm(np.asarray(ee[:2]) - np.asarray(button[:2]))) < 0.016
        and button[2] - 0.02 < tip < button[2] + 0.012
    )


def any_arm_pressing(state, button):
    """`pressing_button` for either arm's tool point. No instruction says which arm must press,
    so a press by the rig's other arm counts as well (until 2026-11-10 only the first arm's
    tool point was read, and a timely press by the other arm went unjudged)."""
    return any(pressing_button(point, button) for point in state.tool_points())


class HospitalTask:
    task_id = ""
    consumables: tuple = ()  # objects that may fall off the bench without failing the task (spent pull-tabs)
    name = ""
    category = ""
    discipline = ""
    instruction = ""

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        self.layout = None
        self.time = 0.0
        self.steps = 0
        self.event_time = None
        self.action_time = None
        self.completion_time = None
        self.first_motion_time = None
        self.correct, self.attempted, self.dropped = set(), set(), set()
        self.dropped_consumables = set()
        self.violations, self.physical_failures, self.event_log = [], [], []
        self.entries = {}  # (object, container) -> first downward entry time
        self.moved = set()
        self.dwell = {}
        self.temporal_error = 0.0
        self.collision_count = 0
        self.action_count = 0
        self.goal_achieved = False
        self.previous = None
        self.placement_dwell = 0.2
        self.recovery_dwell = 0.30
        self._timers = {}
        self._flash = {}

    def flash(self, light, active, bright, dim, period=0.8):
        """Flash an emergency beacon between two colors while `active`; dark otherwise. Only changes are sent."""
        on = active and int(self.time / (period / 2)) % 2 == 0
        color = bright if on else dim
        if self._flash.get(light) == color:
            return []
        self._flash[light] = color
        return [SceneCommand("indicator", light, color)]

    def timer(self, light: str, fraction: float, color: str | None = None) -> list[SceneCommand]:
        """Update the countdown bar beside a light, only when it has visibly changed."""
        fraction = min(max(fraction, 0.0), 1.0)
        color = color or timer_color(fraction)
        state = (round(fraction * 100), color)
        if self._timers.get(light) == state:
            return []
        self._timers[light] = state
        return [SceneCommand("timer", light, (fraction, color))]

    def bind(self, layout, placement_dwell: float = 0.2):
        self.layout = layout
        self.placement_dwell = placement_dwell

    def gen_instruction(self):
        return self.instruction

    # --- scoring helpers -------------------------------------------------------------------

    def violation(self, kind: str, error: float = 0.0):
        if kind not in self.violations:
            self.violations.append(kind)
            self.event_log.append({"time": self.time, "type": kind})
        self.temporal_error = max(self.temporal_error, error)

    def physical_failure(self, kind: str):
        if kind not in self.physical_failures:
            self.physical_failures.append(kind)
            self.event_log.append({"time": self.time, "type": kind})

    def record_action(self, timestamp: float | None = None):
        if self.action_time is None:
            self.action_time = self.time if timestamp is None else timestamp
            self.event_log.append({"time": self.action_time, "type": "task_action"})

    def tilt_ok(self, obj) -> bool:
        return upright_error(obj.orientation) <= self.spec.orientation_tolerance

    def settled(self, name: str, container: str, state: PhysicalState, dt: float, *, upright=True):
        """Released, at rest and seated in the container for the placement dwell."""
        obj = state.objects[name]
        target = self.layout.targets[container]
        inside = seated(self.layout.shapes[name], obj, target)
        # Items stacked in a basket rest on one another, so floor contact is not required there.
        inside = inside and (target.kind in ("basket", "bin") or container in obj.fixture_contacts)
        ok = (
            inside
            and not obj.grasped
            and np.linalg.norm(obj.velocity) < 0.08
            and name in self.moved
            and (not upright or self.tilt_ok(obj))
        )
        key = (name, container)
        self.dwell[key] = self.dwell.get(key, 0.0) + dt if ok else 0.0
        return self.dwell[key] + EPS >= self.placement_dwell

    # --- simulation loop ---------------------------------------------------------------------

    def advance(self, dt: float) -> list[SceneCommand]:
        """Called before each physics tick; physical results are measured afterwards."""
        if dt <= 0:
            raise ValueError("Physics dt must be positive")
        self.time += dt
        self.steps += 1
        return self.process(dt)

    def process(self, dt: float) -> list[SceneCommand]:
        return []

    def observe(self, state: PhysicalState, dt: float):
        for name, obj in state.objects.items():
            initial = self.layout.objects[name].position
            if obj.grasped or np.linalg.norm(obj.position - initial) > 0.04:
                self.moved.add(name)
            if obj.position[2] < self.layout.table_height - 0.12 and name not in self.dropped:
                if name in self.consumables:  # a spent pull-tab on the floor is not a failure
                    self.dropped_consumables.add(name)
                    continue
                self.dropped.add(name)
                self.event_log.append({"time": self.time, "type": "drop", "object": name})
        if self.event_time is not None and state.robot_moving and self.first_motion_time is None:
            self.first_motion_time = self.time
        for name, obj in state.objects.items():
            if name not in self.moved:
                continue
            previous = self.previous.objects.get(name) if self.previous else None
            for container, target in self.layout.targets.items():
                if entry_crossing(self.layout.shapes[name], obj, previous, target):
                    self.entries.setdefault((name, container), self.time)
        self.evaluate(state, dt)
        self.previous = state
        if self.goal_achieved and self.physical_success and self.completion_time is None:
            self.completion_time = self.time

    def evaluate(self, state: PhysicalState, dt: float):
        raise NotImplementedError

    # --- outcomes ----------------------------------------------------------------------------

    @property
    def physical_success(self) -> bool:
        return self.goal_achieved and not self.dropped and not self.physical_failures

    @property
    def temporal_success(self) -> bool:
        return self.action_time is not None and not self.violations

    @property
    def joint_success(self) -> bool:
        return self.physical_success and self.temporal_success

    @property
    def terminated(self) -> bool:
        return self.physical_success or bool(self.dropped)

    def metrics(self) -> dict:
        count = max(1, len(self.layout.objects))
        return {
            "task_id": self.task_id,
            "task_name": self.name,
            "domain": "hospital",
            "discipline": self.discipline,
            "category": self.category,
            "difficulty": self.spec.difficulty,
            "seed": self.spec.seed,
            "task_success": self.physical_success,
            "manipulation_success": self.physical_success,
            "temporal_success": self.temporal_success,
            "joint_success": self.joint_success,
            "completion_time": self.completion_time,
            "elapsed_time": self.time,
            "action_time": self.action_time,
            "event_time": self.event_time,
            "reaction_latency": None
            if self.action_time is None or self.event_time is None
            else self.action_time - self.event_time,
            "first_motion_latency": None
            if self.first_motion_time is None or self.event_time is None
            else self.first_motion_time - self.event_time,
            "temporal_error": self.temporal_error,
            "temporal_violations": list(self.violations),
            "deadline_missed": any(
                v.startswith("late_") or v == "missed_window" for v in self.violations
            ),
            "correct_operations": len(self.correct),
            "attempted_operations": len(self.attempted),
            "accuracy": len(self.correct) / max(1, len(self.attempted)),
            "operations_per_minute": 60 * len(self.correct) / max(self.time, EPS),
            "physical_failures": list(self.physical_failures),
            "drops": len(self.dropped),
            "drop_rate": len(self.dropped) / count,
            "collision_count": self.collision_count,
            "number_of_actions": self.action_count,
            "physics_steps": self.steps,
            "episode_spec": asdict(self.spec),
        }


class WindowTask(HospitalTask):
    """Items placed into their own container only while that container's window is open (A1, A3)."""

    category = "temporal_window"

    def slots(self):
        """(object, container, indicator, (t_o, t_c)) per item, defined by the subclass."""
        raise NotImplementedError

    def __init__(self, spec):
        super().__init__(spec)
        self.action_times = {}

    def process(self, dt):
        commands = []
        for name, _, indicator, (start, end) in self.slots():
            commands.append(
                SceneCommand("indicator", indicator, window_color(self.time, start, end))
            )
            if name in self.action_times:
                commands += self.timer(
                    indicator, 1.0, "green"
                )  # served in time: the dial rests full and green
            elif start <= self.time <= end:
                commands += self.timer(indicator, (end - self.time) / (end - start))
            if self.time > end and name not in self.action_times:
                self.violation("missed_window", self.time - end)
        return commands

    def evaluate(self, state, dt):
        slots = self.slots()
        for name, container in self.entries:
            assigned = self.layout.assignments.get(name)
            if assigned is None:
                # A decoy or a distractor item was put into a destination.
                self.violation("wrong_item")
                self.physical_failure("wrong_item")
            elif container != assigned:
                self.violation("wrong_container")
                self.physical_failure("wrong_container")
        for name, container, _, (start, end) in slots:
            entered = self.entries.get((name, container))
            if entered is None:
                continue
            if name not in self.action_times:
                self.action_times[name] = entered
                self.record_action(entered)
                error = window_error(entered, start, end)
                if error > EPS:
                    self.violation("early_action" if entered < start else "late_action", error)
            obj = state.objects[name]
            if self.settled(name, container, state, dt) and self.tilt_ok(obj):
                self.correct.add(name)
                self.attempted.add(name)
        self.goal_achieved = len(self.correct) == len(slots)

    @property
    def temporal_success(self):
        return len(self.action_times) == len(self.slots()) and not self.violations

    def metrics(self):
        result = super().metrics()
        result["window_success"] = self.joint_success
        result["placement_times"] = dict(self.action_times)
        return result


class ReactiveTask(HospitalTask):
    category = "event_reactive"

    def trigger_event(self):
        if self.event_time is None and self.time >= self.spec.event_time:
            self.event_time = self.time
            self.event_log.append({"time": self.time, "type": "event_triggered"})
            return True
        return False

    def elapsed(self):
        return None if self.event_time is None else self.time - self.event_time

    def check_response(self, timestamp: float, kind="late_reaction"):
        """Strict bound: a response at exactly delta t is late."""
        elapsed = timestamp - self.event_time
        if elapsed + EPS >= self.spec.response_limit:
            self.violation(kind, elapsed - self.spec.response_limit)

    def expire(self, responded: bool, kind="late_reaction"):
        if self.event_time is not None and not responded:
            elapsed = self.time - self.event_time
            if elapsed + EPS >= self.spec.response_limit:
                self.violation(kind, elapsed - self.spec.response_limit)

    def metrics(self):
        result = super().metrics()
        result["deadline_missed"] = bool(self.violations)
        return result


LIFT = 0.03  # m an item must rise above its starting pose to count as lifted


class ThroughputTask(HospitalTask):
    """Unique correct placements over the fixed time budget; early completion never ends the run.

    The archived tabletop C1-C3 (legacy profile) keep this rate rule. The ward C1-C4 are judged
    by WardThroughputTask below (score_version 2, docs/c-class-timing.md).
    """

    category = "throughput"

    def __init__(self, spec):
        super().__init__(spec)
        self.counted = {}  # item -> container where it earned credit
        self.lift_times = {}  # item -> first time it rose LIFT above its starting pose

    def observe(self, state, dt):
        for name, obj in state.objects.items():
            rise = obj.position[2] - self.layout.objects[name].position[2]
            if name not in self.lift_times and rise > LIFT:
                self.lift_times[name] = self.time
        super().observe(state, dt)

    def accepts(self, name: str, container: str) -> bool:
        raise NotImplementedError

    def evaluate(self, state, dt):
        for name, obj in state.objects.items():
            if name in self.correct or name in self.dropped:
                continue
            for container in self.layout.targets:
                if self.settled(name, container, state, dt, upright=False):
                    self.attempted.add(name)
                    if self.accepts(name, container):
                        self.correct.add(name)
                        self.counted[name] = container
                        self.record_action(self.entries.get((name, container), self.time))
                    break
            else:
                # A released item resting away from its source counts once as an attempt.
                away = np.linalg.norm(obj.position - self.layout.objects[name].position) > 0.05
                wrong = (
                    name in self.moved
                    and not obj.grasped
                    and away
                    and np.linalg.norm(obj.velocity) < 0.08
                )
                key = (name, "attempt")
                self.dwell[key] = self.dwell.get(key, 0.0) + dt if wrong else 0.0
                if self.dwell[key] + EPS >= self.placement_dwell:
                    self.attempted.add(name)
        self.goal_achieved = len(self.correct) == len(self.layout.objects)

    @property
    def physical_success(self):
        """Every placement that earned credit is still inside its container and released.

        The sheet sets a rate, not a requirement to finish every item, so unplaced items and
        drops are reported as metrics instead of failing the physical outcome.
        """
        if not self.correct or self.previous is None:
            return False
        for name, container in self.counted.items():
            obj = self.previous.objects[name]
            target = self.layout.targets[container]
            shape = self.layout.shapes[name]
            # Still inside the container, below its rim and released. Items thrown into a bin may
            # lean on a wall or on each other, which is physically in the bin.
            if (
                obj.grasped
                or not fits_xy(shape, obj, target, clearance=-0.01)
                or bottom_z(shape, obj) >= target.rim_z
            ):
                return False
        return True

    def process(self, dt):
        """A light and a countdown bar for the time budget, where the layout has one."""
        if not any(light.name == "indicator" for light in self.layout.indicators):
            return []
        budget = self.spec.max_time
        color = GREEN if self.time + EPS < budget else RED
        return [
            SceneCommand("indicator", "indicator", color),
            *self.timer("indicator", (budget - self.time) / budget),
        ]

    @property
    def required_rate(self):
        return self.spec.required_rate_per_min / 60

    @property
    def temporal_success(self):
        accuracy_ok = self.spec.required_accuracy is None or (
            len(self.correct) / max(1, len(self.attempted | self.dropped))
            >= self.spec.required_accuracy
        )
        return (
            bool(self.correct)
            and len(self.correct) / max(self.time, EPS) >= self.required_rate
            and accuracy_ok
        )

    @property
    def terminated(self):
        # Always run the full budget so early stopping cannot inflate N/T.
        return False

    def budget_text(self):
        return f"{self.spec.max_time:.1f} seconds"

    def gen_instruction(self):
        text = (
            f"{self.instruction} Process all {self.spec.num_items} items within "
            f"{self.budget_text()}, at least {self.spec.required_rate_per_min:.2f} correct items "
            "per minute"
        )
        if self.spec.required_accuracy is not None:
            text += f" and {100 * self.spec.required_accuracy:.1f}% order accuracy"
        return text + "."

    def metrics(self):
        result = super().metrics()
        result["attempted_operations"] = len(self.attempted | self.dropped)
        result["accuracy"] = len(self.correct) / max(1, result["attempted_operations"])
        # Measured milestones for the stage score: an item first lifted off its starting
        # pose, and first lowered over a container lip into it (``entries``).
        result["first_lift_time"] = min(self.lift_times.values(), default=None)
        result["first_entry_time"] = min(self.entries.values(), default=None)
        result["required_operations_per_minute"] = self.spec.required_rate_per_min
        if self.time >= self.spec.max_time - EPS and not self.temporal_success:
            result["temporal_violations"] = ["throughput_or_accuracy"]
        return result


class WardThroughputTask(ThroughputTask):
    """Ward C1-C4: Q jobs on the table at t = 0, judged as in docs/c-class-timing.md.

    - A job is one correct placement (C1: one ordered vial; C2-C4: one item). Its completion time
      is the time it earned credit; t_Q is the time of the Q-th (spec.num_items) credit.
    - physical: all Q jobs done, every credited item still valid at the last frame (released in
      its container), nothing dropped, no physical failure, and final misplacements (items
      resting in a container that did not credit them) within
      throughput.allowed_misplacements(required_accuracy, Q). Time plays no part.
    - temporal: from throughput.judge only, t_Q <= T_tier = m_tier * t_o, scored at every tier
      from one run; the difficulty only picks the tier reported as temporal/joint success.
    - The episode runs to H = throughput.horizon(t_o) (spec.max_time) and ends early when all Q
      jobs stayed physically valid for throughput.SETTLE_S ("all_done"), on the stall stop
      ("stall") or on a drop / physical failure ("failure").
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.job_times = {}  # item -> time it earned credit (never revoked)
        self.valid_since = None  # all Q jobs physically valid since this time
        self.end_reason = None  # all_done, stall or failure, latched

    # --- jobs ----------------------------------------------------------------------------------

    @property
    def jobs_required(self) -> int:
        """Q: every job of the task (spec.num_items; C1's spare vials are not jobs)."""
        return int(self.spec.num_items)

    @property
    def oracle_time(self) -> float:
        return throughput.oracle_time("hospital", self.task_id)

    def job_times_list(self) -> list:
        """One completion time per job (None if not done), Q entries."""
        times = sorted(self.job_times.values())[: self.jobs_required]
        return times + [None] * (self.jobs_required - len(times))

    @property
    def last_job_time(self):
        return max(self.job_times.values(), default=None)

    @property
    def all_jobs_done(self) -> bool:
        return len(self.correct) >= self.jobs_required

    # --- physical outcome ----------------------------------------------------------------------

    def inside(self, name, obj, container) -> bool:
        """The item lies inside the container: within its walls (1 cm lean allowed), below its rim."""
        target, shape = self.layout.targets[container], self.layout.shapes[name]
        return fits_xy(shape, obj, target, clearance=-0.01) and bottom_z(shape, obj) < target.rim_z

    def job_valid(self, name, container, obj) -> bool:
        """A credited item still holds at the end: released inside the container that credited it."""
        return not obj.grasped and self.inside(name, obj, container)

    def misplaced(self) -> int:
        """Items resting, released, in a container that did not credit them (last frame). An item still held
        or moving (being carried over a bin) is not a misplacement yet."""
        if self.previous is None:
            return 0
        count = 0
        for name, obj in self.previous.objects.items():
            if name in self.dropped or obj.grasped or np.linalg.norm(obj.velocity) >= 0.08:
                continue
            for container in self.layout.targets:
                if self.counted.get(name) != container and self.inside(name, obj, container):
                    count += 1
                    break
        return count

    @property
    def allowed_misplacements(self) -> int:
        return throughput.allowed_misplacements(self.spec.required_accuracy, self.jobs_required)

    @property
    def failed(self) -> bool:
        """A drop or a physical failure: physical success is no longer possible."""
        return bool(self.dropped) or bool(self.physical_failures)

    @property
    def physical_success(self) -> bool:
        if self.previous is None or self.failed or not self.all_jobs_done:
            return False
        objects = self.previous.objects
        if not all(self.job_valid(n, c, objects[n]) for n, c in self.counted.items()):
            return False
        return self.misplaced() <= self.allowed_misplacements

    # --- temporal outcome ----------------------------------------------------------------------

    def judged(self, physical=False, misplaced=None) -> dict:
        """throughput.judge on this episode's job times (temporal fields ignore `physical`)."""
        return throughput.judge(
            self.job_times_list(),
            self.jobs_required,
            physical,
            self.oracle_time,
            self.spec.difficulty,
            misplaced=misplaced,
        )

    @property
    def temporal_success(self) -> bool:
        return self.judged()["temporal_success"]

    # --- episode end ---------------------------------------------------------------------------

    def observe(self, state, dt):
        super().observe(state, dt)
        for name in self.correct:
            self.job_times.setdefault(name, self.time)
        if self.end_reason is not None:
            return
        if self.failed:
            reason = "failure"
        else:
            reason = None
            if self.physical_success:
                if self.valid_since is None:
                    self.valid_since = self.time
                if self.time - self.valid_since + EPS >= throughput.SETTLE_S:
                    reason = "all_done"
            else:
                self.valid_since = None
            # No stall while a complete, valid scene is settling: it ends as all_done SETTLE_S later.
            if (
                reason is None
                and self.valid_since is None
                and throughput.stalled(
                    self.time, self.last_job_time, self.oracle_time, steps=self.jobs_required
                )
            ):
                reason = "stall"
        if reason is not None:
            self.end_reason = reason
            self.event_log.append({"time": self.time, "type": f"episode_end_{reason}"})

    @property
    def terminated(self) -> bool:
        return self.end_reason is not None

    @property
    def termination_reason(self):
        if self.end_reason is not None:
            return self.end_reason
        return "horizon" if self.time >= self.spec.max_time - EPS else None

    # --- instruction and display ---------------------------------------------------------------

    def gen_instruction(self):
        return f"{self.instruction} {self.goal_text()}"

    def goal_text(self) -> str:
        return f"Finish all {self.jobs_required} items as quickly as possible."

    # --- metrics -------------------------------------------------------------------------------

    def metrics(self):
        result = super().metrics()
        result.pop("required_operations_per_minute", None)
        verdict = self.judged(self.physical_success, self.misplaced())
        ended = self.termination_reason is not None
        result.update(
            {
                "job_times_by_item": dict(self.job_times),
                "allowed_misplacements": self.allowed_misplacements,
                "termination_reason": self.termination_reason,
                "temporal_violations": ["late_completion"]
                if ended and not verdict["temporal_success"]
                else [],
            }
        )
        result.update(verdict)  # task/manipulation/temporal/joint success and completion_time
        result["deadline_missed"] = bool(result["temporal_violations"])
        return result
