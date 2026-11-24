"""Household timing with released contact, geometric containment and stable support.

Clocks are abstract process models, not thermodynamics. Object motion is physical.
"""

from dataclasses import asdict

import numpy as np

from roboclock.envs.lab.mdp.events import BLUE, GREEN, RED, SceneCommand, window_error
from roboclock.physics.geometry import rotation_matrix


class HouseholdTask:
    discipline = "household"
    category = task_id = asset = instruction = ""
    response_mode = "entry"

    def __init__(self, spec):
        self.spec, self.time, self.steps = spec, 0.0, 0
        self.layout = self.previous = None
        self.correct, self.attempted, self.dropped, self.active = set(), set(), set(), set()
        self.violations, self.physical_failures, self.event_log = [], [], []
        self.event_time = self.action_time = self.completion_time = self.first_motion_time = None
        self.temporal_error = 0.0
        self.collision_count = self.action_count = 0
        self.entry_times, self.insertion_times, self.insertion_offsets = {}, {}, {}
        self.dwell, self.grasp_history = {}, set()
        self.failed_attempts, self.released = set(), set()
        self.goal_achieved = self.alarm = False
        self.fixture_offset, self.temperature = 0.0, 25.0
        self.placement_dwell, self.recovery_dwell, self.catch_dwell = 0.2, 0.3, 0.25

    def bind(self, layout, placement_dwell=0.2):
        self.layout, self.placement_dwell = layout, placement_dwell
        self.active = set(layout.objects)

    def gen_instruction(self):
        text = self.instruction
        if self.category == "temporal_window":
            text += f" Green interval: {self.spec.valid_start:.2f} to {self.spec.valid_end:.2f} simulation seconds."
        elif self.category == "event_reactive":
            text += f" Respond after the red cue, within {self.spec.max_latency:.2f} seconds; no pre-cue grasp."
            if self.response_mode == "removal":
                text += f" Complete placement within {self.spec.transfer_deadline:.2f} seconds."
        else:
            text += f" Process all {self.spec.num_samples} items in the fixed {self.spec.max_time:.1f}-second budget: at least {60 * self.spec.required_rate:.2f} correct items/minute, {100 * self.spec.required_accuracy:.1f}% accuracy."
        return text

    def violation(self, kind, error=0.0):
        if kind not in self.violations:
            self.violations.append(kind)
            self.event_log.append({"time": self.time, "type": kind})
        self.temporal_error = max(self.temporal_error, float(error))

    def record_action(self, timestamp=None):
        if self.action_time is None:
            self.action_time = self.time if timestamp is None else timestamp
            self.event_log.append({"time": self.action_time, "type": "task_action"})

    def advance(self, dt):
        if dt <= 0:
            raise ValueError("Physics dt must be positive")
        self.time += dt
        self.steps += 1
        if self.category == "temporal_window":
            color = BLUE if self.time < self.spec.valid_start else GREEN
            if self.time > self.spec.valid_end + 1e-9:
                color = RED
                if self.action_time is None:
                    self.violation("missed_window", self.time - self.spec.valid_end)
        elif self.category == "event_reactive":
            if self.event_time is None and self.time >= self.spec.event_time:
                self.event_time, self.alarm = self.time, True
                self.event_log.append({"time": self.time, "type": "event_triggered"})
            color = RED if self.event_time is not None else BLUE
            if self.event_time is not None:
                latency = self.time - self.event_time
                if self.action_time is None and latency >= self.spec.max_latency:
                    self.violation("late_reaction", latency - self.spec.max_latency)
                if (
                    self.response_mode == "removal"
                    and not self.goal_achieved
                    and latency > self.spec.transfer_deadline
                ):
                    self.violation("transfer_deadline", latency - self.spec.transfer_deadline)
        else:
            color = GREEN
        return [SceneCommand("indicator", "indicator", color)]

    def _extents(self, name, obj):
        radius, height = self.layout.dimensions[name]
        return np.abs(rotation_matrix(obj.orientation)) @ np.array([radius, radius, height / 2])

    def _fits(self, name, obj, target):
        return bool(
            np.all(
                np.abs(obj.position[:2] - target.position[:2]) + self._extents(name, obj)[:2]
                <= target.half_extents[:2] - 0.0005
            )
        )

    def _entry(self, name, obj, target):
        old = self.previous.objects.get(name) if self.previous else None
        if old is None or not self._fits(name, obj, target):
            return False
        bottom = obj.position[2] - self._extents(name, obj)[2]
        old_bottom = old.position[2] - self._extents(name, old)[2]
        plane = target.rim_z - target.insertion_depth
        return bool(
            old_bottom >= plane
            and target.support_z - 0.003 <= bottom < plane
            and bottom < old_bottom
        )

    def _seated(self, name, obj, target):
        bottom = obj.position[2] - self._extents(name, obj)[2]
        return bool(
            self._fits(name, obj, target)
            and target.support_z - 0.003 <= bottom <= target.support_z + 0.006
            and self.layout.assignments[name] in obj.fixture_contacts
            and not obj.grasped
            and np.linalg.norm(obj.velocity) < 0.08
        )

    def _source_extracted(self, obj):
        source = self.layout.heating_zone
        extent = self._extents("sample_0", obj)
        above = obj.position[2] - extent[2] > source.rim_z
        outside = np.any(
            np.abs(obj.position[:2] - source.position[:2]) - extent[:2] > source.half_extents[:2]
        )
        return bool(above or outside)

    def _check_action_time(self):
        if self.category == "temporal_window":
            error = window_error(self.action_time, self.spec.valid_start, self.spec.valid_end)
            if error > 1e-9:
                self.violation(
                    "early_action" if self.action_time < self.spec.valid_start else "late_action",
                    error,
                )
        elif self.category == "event_reactive":
            if self.event_time is None:
                self.violation("pre_event_action")
            elif self.action_time - self.event_time >= self.spec.max_latency:
                self.violation(
                    "late_reaction", self.action_time - self.event_time - self.spec.max_latency
                )

    def observe(self, state, dt):
        if self.event_time is not None and state.robot_moving and self.first_motion_time is None:
            self.first_motion_time = self.time
        for name, obj in state.objects.items():
            old = self.previous.objects.get(name) if self.previous else None
            if obj.position[2] < self.layout.table_height - 0.12:
                if name not in self.dropped:
                    self.event_log.append({"time": self.time, "type": "drop", "object": name})
                self.dropped.add(name)
            if obj.grasped:
                self.grasp_history.add(name)
                if self.category == "event_reactive" and self.event_time is None:
                    self.violation("pre_event_grasp")
            elif old is not None and old.grasped:
                self.released.add(name)
            target = self.layout.targets[self.layout.assignments[name]]
            if name not in self.insertion_times and self._entry(name, obj, target):
                if name not in self.grasp_history:
                    self.event_log.append(
                        {"time": self.time, "type": "unverified_target_entry", "object": name}
                    )
                    if self.category != "throughput":
                        self.violation("entry_without_verified_grasp")
                else:
                    self.insertion_times[name] = self.entry_times[name] = self.time
                    self.event_log.append(
                        {"time": self.time, "type": "target_entry", "object": name}
                    )
                    if self.response_mode == "entry" and self.category != "throughput":
                        self.record_action(self.time)
                        self._check_action_time()
            if self.response_mode == "extraction" and self.action_time is None:
                crossed = (
                    old is not None
                    and not self._source_extracted(old)
                    and self._source_extracted(obj)
                )
                if crossed:
                    if obj.grasped:
                        self.record_action()
                        self._check_action_time()
                    else:
                        self.violation("extracted_without_current_grasp")
            elif self.response_mode == "removal" and self.action_time is None:
                origin = self.layout.objects[name].position
                crossed = (
                    old is not None
                    and np.linalg.norm(old.position - origin) <= 0.08
                    and np.linalg.norm(obj.position - origin) > 0.08
                )
                if crossed:
                    if obj.grasped:
                        self.record_action()
                        self._check_action_time()
                    else:
                        self.violation("removed_without_current_grasp")
            placed = self._seated(name, obj, target) and name in self.grasp_history
            self.dwell[name] = self.dwell.get(name, 0.0) + dt if placed else 0.0
            if (
                self.dwell[name] + 1e-9 >= self.placement_dwell
                and name not in self.correct
                and name in self.insertion_times
            ):
                self.correct.add(name)
                self.attempted.add(name)
                self.event_log.append(
                    {"time": self.time, "type": "accepted_placement", "object": name}
                )
                if self.response_mode == "placement":
                    self.record_action()
                    self._check_action_time()
                elif self.category == "throughput":
                    self.record_action(self.entry_times[name])
            moved = np.linalg.norm(obj.position - self.layout.objects[name].position) > 0.05
            if (
                moved
                and name in self.released
                and not obj.grasped
                and np.linalg.norm(obj.velocity) < 0.08
                and not placed
            ):
                self.dwell[(name, "wrong")] = self.dwell.get((name, "wrong"), 0.0) + dt
                if self.dwell[(name, "wrong")] >= self.placement_dwell:
                    self.attempted.add(name)
                    self.failed_attempts.add(name)
            else:
                self.dwell[(name, "wrong")] = 0.0
        self.previous = state
        self.goal_achieved = len(self.correct) == len(self.layout.objects)
        if self.goal_achieved and self.completion_time is None:
            self.completion_time = self.time

    @property
    def physical_success(self):
        if (
            not self.goal_achieved
            or self.dropped
            or self.physical_failures
            or self.previous is None
        ):
            return False
        return all(
            self._seated(name, self.previous.objects[name], self.layout.targets[target])
            for name, target in self.layout.assignments.items()
        )

    @property
    def temporal_success(self):
        if self.category == "throughput":
            return bool(
                self.time >= self.spec.max_time - 1e-9
                and len(self.retained_correct) / self.spec.max_time + 1e-9
                >= self.spec.required_rate
                and self.accuracy >= self.spec.required_accuracy
            )
        return self.action_time is not None and not self.violations

    @property
    def retained_correct(self):
        if self.previous is None:
            return set()
        return {
            name
            for name in self.correct
            if self._seated(
                name,
                self.previous.objects[name],
                self.layout.targets[self.layout.assignments[name]],
            )
        }

    @property
    def attempted_operations(self):
        return len(self.correct) + len(self.failed_attempts | self.dropped)

    @property
    def accuracy(self):
        correct = len(self.retained_correct) if self.category == "throughput" else len(self.correct)
        return correct / max(1, self.attempted_operations)

    @property
    def joint_success(self):
        return self.physical_success and self.temporal_success

    @property
    def terminated(self):
        return self.category != "throughput" and (bool(self.dropped) or self.physical_success)

    def metrics(self):
        return {
            "task_id": self.task_id,
            "task_name": self.name,
            "domain": "household",
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
            "correct_operations": len(self.retained_correct)
            if self.category == "throughput"
            else len(self.correct),
            "attempted_operations": self.attempted_operations,
            "accepted_operations": len(self.correct),
            "failed_operations": len(self.failed_attempts | self.dropped),
            "accuracy": self.accuracy,
            "operations_per_minute": 60
            * (len(self.retained_correct) if self.category == "throughput" else len(self.correct))
            / max(self.time, 1e-9),
            "physical_failures": list(self.physical_failures),
            "drops": len(self.dropped),
            "drop_rate": len(self.dropped) / len(self.layout.objects),
            "collision_count": self.collision_count,
            "number_of_actions": self.action_count,
            "physics_steps": self.steps,
            "insertion_times": dict(self.insertion_times),
            "episode_spec": asdict(self.spec),
            "prototype_limitations": [
                "symbolic_household_processes",
                "rigid_graspable_proxies",
                "three_item_workload" if self.category == "throughput" else "single_item_scene",
            ],
        }
