"""Task lifecycle and independent physical/temporal scoring.

All timestamps use elapsed physics time. Temporal failure remains latched while
the episode continues, allowing eventual manipulation success to be measured.
"""

from dataclasses import asdict

import numpy as np

from roboclock.envs.lab.config import EpisodeSpec
from roboclock.envs.lab.mdp.events import BLUE, GREEN, RED, SceneCommand, window_error
from roboclock.envs.lab.mdp.geometry import insertion_crossing, seated
from roboclock.envs.lab.mdp.state import PhysicalState, orientation_error, upright_error
from roboclock.evaluation import throughput


class LabTask:
    task_id = ""
    name = ""
    category = ""
    discipline = ""
    asset = "container"
    instruction = ""

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        self.layout = None
        self.time = 0.0
        self.steps = 0
        self.event_time = None
        self.action_time = None
        self.completion_time = None
        self.correct = set()
        self.attempted = set()
        self.dropped = set()
        self.violations = []
        self.physical_failures = []
        self.insertion_times = {}
        self.insertion_offsets = {}
        self.temporal_error = 0.0
        self.dwell = {}
        self.entry_times = {}
        self.moved = set()
        self.active = set()
        self.first_motion_time = None
        self.collision_count = 0
        self.action_count = 0
        self.goal_achieved = False
        self.temperature = 25.0
        self.previous = None
        self.fixture_offset = 0.0
        self.alarm = False
        self.event_log = []

    def bind(self, layout, placement_dwell: float = 0.2):
        self.layout = layout
        self.placement_dwell = placement_dwell
        self.recovery_dwell = 0.30
        self.catch_dwell = 0.25
        self.active = (
            {"sample_0"} if getattr(self, "continuous_arrivals", False) else set(layout.objects)
        )

    def gen_instruction(self):
        text = self.instruction
        if self.category == "event_reactive":
            text += f" Respond within {self.spec.max_latency:.2f} seconds of the visible event."
            if self.task_id == "B1":
                text += f" Complete the safety transfer within {self.spec.transfer_deadline:.2f} seconds of the alarm."
        elif self.category == "throughput":
            text += f" Process all {self.spec.num_samples} samples as quickly as possible."
        return text

    def violation(self, kind: str, error: float = 0.0):
        if kind not in self.violations:
            self.violations.append(kind)
            self.event_log.append({"time": self.time, "type": kind})
        self.temporal_error = max(self.temporal_error, error)

    def trigger_event(self):
        if self.event_time is None and self.time >= self.spec.event_time:
            self.event_time = self.time
            self.event_log.append({"time": self.time, "type": "event_triggered"})
            return True
        return False

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
            if name not in self.active:
                continue
            if (
                obj.grasped
                or np.linalg.norm(obj.position - self.layout.objects[name].position) > 0.04
            ):
                self.moved.add(name)
            if obj.position[2] < self.layout.table_height - 0.12:
                if name not in self.dropped:
                    self.event_log.append({"time": self.time, "type": "drop", "object": name})
                self.dropped.add(name)
        if self.event_time is not None and state.robot_moving and self.first_motion_time is None:
            self.first_motion_time = self.time
        for name, obj in state.objects.items():
            target = self.layout.targets[self.layout.assignments[name]]
            previous = self.previous.objects.get(name) if self.previous else None
            offset = self.fixture_offset if self.task_id == "A4" else 0.0
            if name in self.moved and insertion_crossing(self, name, obj, previous, target, offset):
                self.insertion_times.setdefault(name, self.time)
                self.insertion_offsets.setdefault(name, self.fixture_offset)
        self.evaluate(state, dt)
        self.previous = state
        if self.physical_success and self.completion_time is None:
            self.completion_time = self.time

    def evaluate(self, state: PhysicalState, dt: float):
        raise NotImplementedError

    def placed(self, name: str, state: PhysicalState, dt: float, precise=False) -> bool:
        obj = state.objects[name]
        target = self.layout.targets[self.layout.assignments[name]]
        pos = target.position.copy()
        if self.task_id == "A4":
            pos[0] += self.fixture_offset
        inside = seated(
            self, name, obj, target, self.fixture_offset if self.task_id == "A4" else 0.0
        )
        inside = inside and self.layout.assignments[name] in obj.fixture_contacts
        if precise:
            inside = inside and (
                np.linalg.norm(obj.position - pos) <= self.spec.position_tolerance
                and self.rotation_error(obj.orientation, target.orientation)
                <= self.spec.orientation_tolerance
            )
        released = not obj.grasped
        key = (name, "correct")
        if inside and released and name in self.moved:
            self.entry_times.setdefault(name, self.time)
        else:
            self.entry_times.pop(name, None)
        settled = inside and released and np.linalg.norm(obj.velocity) < 0.08 and name in self.moved
        self.dwell[key] = self.dwell.get(key, 0.0) + dt if settled else 0.0
        return self.dwell[key] + 1e-9 >= self.placement_dwell

    def rotation_error(self, orientation, target_orientation):
        if self.asset in ("container", "bottle", "probe", "beaker", "test_tube"):
            return upright_error(orientation)
        return orientation_error(orientation, target_orientation)

    def record_action(self, timestamp: float | None = None):
        if self.action_time is None:
            self.action_time = self.time if timestamp is None else timestamp
            self.event_log.append({"time": self.action_time, "type": "task_action"})

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
        return {
            "task_id": self.task_id,
            "task_name": self.name,
            "domain": "laboratory",
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
            if self.first_motion_time is None
            else self.first_motion_time - self.event_time,
            "temporal_error": self.temporal_error,
            "temporal_violations": list(self.violations),
            "correct_operations": len(self.correct),
            "attempted_operations": len(self.attempted),
            "accuracy": len(self.correct) / max(1, len(self.attempted)),
            "operations_per_minute": 60 * len(self.correct) / max(self.time, 1e-9),
            "physical_failures": list(self.physical_failures),
            "drops": len(self.dropped),
            "drop_rate": len(self.dropped) / len(self.layout.objects),
            "collision_count": self.collision_count,
            "number_of_actions": self.action_count,
            "physics_steps": self.steps,
            "episode_spec": asdict(self.spec),
        }


class WindowTask(LabTask):
    category = "temporal_window"

    def process(self, dt):
        color = (
            BLUE
            if self.time < self.spec.valid_start
            else GREEN
            if self.time <= self.spec.valid_end
            else RED
        )
        return [SceneCommand("indicator", "indicator", color)]

    def check_window(self, timestamp: float):
        error = window_error(timestamp, self.spec.valid_start, self.spec.valid_end)
        if error > 1e-9:
            self.violation(
                "early_action" if timestamp < self.spec.valid_start else "late_action", error
            )

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        target = self.layout.targets[self.layout.assignments["sample_0"]]
        inserted = (
            "sample_0" in self.insertion_times
            and self.rotation_error(obj.orientation, target.orientation)
            <= self.spec.orientation_tolerance
        )
        if inserted and self.action_time is None:
            self.record_action(self.insertion_times["sample_0"])
            self.check_window(self.action_time)
        if inserted and self.placed("sample_0", state, dt, precise=True):
            self.correct.add("sample_0")
            self.attempted.add("sample_0")
            self.goal_achieved = True

    def metrics(self):
        result = super().metrics()
        result["window_success"] = self.joint_success
        return result


class ReactiveTask(LabTask):
    category = "event_reactive"

    def __init__(self, spec: EpisodeSpec):
        super().__init__(spec)
        self.start_tool_points = None
        self.pre_event_tool_drift = 0.0

    def observe(self, state: PhysicalState, dt: float):
        # How far either tool point left its start pose before the cue. A policy (or the
        # oracle) that waits in the start pose keeps this near zero; it is a metric, not a
        # violation, because some instructions allow motion before the event.
        if self.event_time is None:
            points = state.tool_points()
            if self.start_tool_points is None:
                self.start_tool_points = [point.copy() for point in points]
            self.pre_event_tool_drift = max(
                self.pre_event_tool_drift,
                *(
                    float(np.linalg.norm(point - start))
                    for point, start in zip(points, self.start_tool_points)
                ),
            )
        super().observe(state, dt)

    def process(self, dt):
        self.trigger_event()
        if self.event_time is not None and self.action_time is None:
            elapsed = self.time - self.event_time
            if elapsed >= self.spec.max_latency:
                self.violation("late_reaction", elapsed - self.spec.max_latency)
        return [
            SceneCommand("indicator", "indicator", RED if self.event_time is not None else BLUE)
        ]

    def record_reaction(self):
        self.record_action()
        if self.event_time is None:
            self.violation("pre_event_action")
        elif self.action_time - self.event_time >= self.spec.max_latency:
            self.violation(
                "late_reaction", self.action_time - self.event_time - self.spec.max_latency
            )

    def metrics(self):
        result = super().metrics()
        result["deadline_missed"] = (
            "late_reaction" in self.violations or "transfer_deadline" in self.violations
        )
        result["pre_event_tool_drift_m"] = self.pre_event_tool_drift
        return result


class ThroughputTask(LabTask):
    """C1-C3: Q = num_samples jobs, all on the table at t = 0 (batch C task, score_version 2).

    docs/c-class-timing.md states the rule, roboclock.evaluation.throughput its arithmetic. A job
    is done when its sample is credited correct (inserted, released and settled for the
    placement dwell); ``job_times`` keeps when.

    - physical: at the last frame all Q samples are credited and still valid (released, seated
      in and touching their own fixture, within the fixed tolerances), none dropped, and final
      misplacements within allowed_misplacements(required_accuracy, Q). No time terms.
    - temporal, per tier: t_Q <= T_tier = m_tier t_o. t_Q is ``completion_time``, the first
      frame with all Q placed and valid at once (what the frozen oracle time t_o measured).
    - The episode runs to the horizon spec.max_time = H = 5 t_o(Q) unless all Q stay placed
      and valid for SETTLE_S ("all_done") or the stall stop fires ("stall"). Drops do not end
      it.
    """

    category = "throughput"

    def __init__(self, spec: EpisodeSpec):
        super().__init__(spec)
        self.job_times = {}  # sample -> time it was credited correct
        self.valid_since = None  # start of the current stretch with all Q placed and valid
        self.end_reason = None  # "all_done" or "stall" once the task ends the episode

    @property
    def oracle_time(self) -> float:
        """Frozen oracle time t_o for this episode's Q (roboclock.evaluation.throughput)."""
        return throughput.oracle_time("lab", self.task_id, self.spec.num_samples)

    @property
    def allowed_misplacements(self) -> int:
        return throughput.allowed_misplacements(self.spec.required_accuracy, self.spec.num_samples)

    def evaluate(self, state, dt):
        for name in self.active:
            if name in self.correct or name in self.dropped:
                continue
            insertion_valid = (
                self.layout.targets[self.layout.assignments[name]].kind != "slot"
                or name in self.insertion_times
            )
            if insertion_valid and self.placed(
                name, state, dt, precise=not getattr(self, "bin_sorting", False)
            ):
                self.correct.add(name)
                self.attempted.add(name)
                self.job_times[name] = self.time
                self.record_action(self.entry_times[name])
            else:
                obj = state.objects[name]
                # Count a misplaced released object once, after settling outside its source.
                wrong = (
                    name in self.moved
                    and not obj.grasped
                    and np.linalg.norm(obj.velocity) < 0.08
                    and np.linalg.norm(obj.position - self.layout.objects[name].position) > 0.05
                )
                key = (name, "attempt")
                self.dwell[key] = self.dwell.get(key, 0.0) + dt if wrong else 0.0
                if self.dwell[key] >= self.placement_dwell:
                    self.attempted.add(name)
        self.goal_achieved = len(self.correct) == len(self.layout.objects)

    def observe(self, state, dt):
        super().observe(state, dt)
        if self.end_reason is not None:
            return
        if not self.physical_success:
            self.valid_since = None
        elif self.valid_since is None:
            self.valid_since = self.time
        if (
            self.valid_since is not None
            and self.time - self.valid_since + 1e-9 >= throughput.SETTLE_S
        ):
            self.end_reason = "all_done"
            self.event_log.append({"time": self.time, "type": "all_done"})
        elif self.valid_since is None and throughput.stalled(
            self.time, self.last_job_time, self.oracle_time, steps=self.spec.num_samples
        ):  # no stall while a complete, valid scene is settling
            self.end_reason = "stall"
            self.event_log.append({"time": self.time, "type": "stall"})

    @property
    def last_job_time(self) -> float | None:
        """Time of the latest progress: a newly credited sample, or t_Q itself."""
        times = list(self.job_times.values())
        if self.completion_time is not None:
            times.append(self.completion_time)
        return max(times, default=None)

    def in_place(self, name: str, obj, target) -> bool:
        """Whether a credited sample still sits in its target (geometry only), for ``still_valid``.

        Credit (``placed``) always uses the stricter ``seated``, so a sample that passes it passes
        this; tasks whose samples may settle into a pose ``seated`` rejects override it.
        """
        return seated(self, name, obj, target)

    def still_valid(self, name: str, obj) -> bool:
        """A credited sample is still released, in and touching its fixture, aligned."""
        target = self.layout.targets[self.layout.assignments[name]]
        if obj.grasped:
            return False
        if (
            not self.in_place(name, obj, target)
            or self.layout.assignments[name] not in obj.fixture_contacts
        ):
            return False
        return getattr(self, "bin_sorting", False) or (
            np.linalg.norm(obj.position - target.position) <= self.spec.position_tolerance
            and self.rotation_error(obj.orientation, target.orientation)
            <= self.spec.orientation_tolerance
        )

    def invalid_at_end(self) -> list[dict]:
        """Diagnostics: each credited sample that fails ``still_valid`` on the last frame, and why."""
        if self.previous is None:
            return []
        rows = []
        for name in sorted(self.correct):
            obj = self.previous.objects[name]
            if self.still_valid(name, obj):
                continue
            target = self.layout.targets[self.layout.assignments[name]]
            rows.append(
                {
                    "name": name,
                    "target": self.layout.assignments[name],
                    "grasped": bool(obj.grasped),
                    "in_place": bool(self.in_place(name, obj, target)),
                    "touching_target": self.layout.assignments[name] in obj.fixture_contacts,
                    "position_error_m": float(np.linalg.norm(obj.position - target.position)),
                    "orientation_error_rad": float(
                        self.rotation_error(obj.orientation, target.orientation)
                    ),
                }
            )
        return rows

    def misplaced(self) -> int:
        """Samples resting, released, in a target other than their own at the last frame."""
        if self.previous is None:
            return 0
        count = 0
        for name, obj in self.previous.objects.items():
            if name not in self.active or obj.grasped or np.linalg.norm(obj.velocity) >= 0.08:
                continue
            if name in self.correct and self.still_valid(name, obj):
                continue
            own = self.layout.assignments[name]
            if any(key != own and key in self.layout.targets for key in obj.fixture_contacts):
                count += 1
        return count

    @property
    def physical_success(self):
        if self.previous is None or not self.goal_achieved:
            return False
        if self.dropped or self.physical_failures:
            return False
        if not all(self.still_valid(name, self.previous.objects[name]) for name in self.correct):
            return False
        return self.misplaced() <= self.allowed_misplacements

    def job_time_list(self) -> list[float | None]:
        """Each sample's completion time in layout order, None if never credited."""
        return [self.job_times.get(name) for name in self.layout.objects]

    def judgement(self, physical: bool, misplaced: int | None = None) -> dict:
        return throughput.judge(
            self.job_time_list(),
            self.spec.num_samples,
            physical,
            self.oracle_time,
            self.spec.difficulty,
            t_q=self.completion_time,
            misplaced=misplaced,
        )

    @property
    def temporal_success(self):
        # t_Q within the requested tier's budget; no physical, rate or accuracy terms.
        return self.judgement(False)["temporal_success"]

    @property
    def terminated(self):
        return self.end_reason is not None

    @property
    def termination_reason(self) -> str | None:
        """all_done, stall or horizon once the episode is over (lab C has no failure stop)."""
        if self.end_reason is not None:
            return self.end_reason
        if self.time >= self.spec.max_time - 1e-9:
            return "horizon"
        return None

    def metrics(self):
        result = super().metrics()
        result["attempted_operations"] = len(self.attempted | self.dropped)
        result["accuracy"] = len(self.correct) / max(1, result["attempted_operations"])
        judged = self.judgement(self.physical_success, self.misplaced())
        result.update(judged)
        result["allowed_misplacements"] = self.allowed_misplacements
        result["termination_reason"] = self.termination_reason
        result["temporal_violations"] = (
            []
            if judged["temporal_success"]
            else ["not_all_done" if judged["t_q_s"] is None else "over_budget"]
        )
        errors = []
        if self.previous:
            for name in self.correct:
                obj = self.previous.objects[name]
                target = self.layout.targets[self.layout.assignments[name]]
                errors.append(
                    (
                        float(np.linalg.norm(obj.position - target.position)),
                        self.rotation_error(obj.orientation, target.orientation),
                    )
                )
        result["mean_position_error"] = float(np.mean([e[0] for e in errors])) if errors else None
        result["mean_orientation_error"] = (
            float(np.mean([e[1] for e in errors])) if errors else None
        )
        result["invalid_at_end"] = self.invalid_at_end()
        return result
