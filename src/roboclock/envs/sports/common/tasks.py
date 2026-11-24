"""Independent physical and temporal scoring for Sports tasks 31–40."""

from dataclasses import dataclass, replace

import numpy as np

from roboclock.envs.sports.common.config import chess_config, load_config
from roboclock.envs.sports.common.layout import make_layout
from roboclock.physics.geometry import rotation_matrix, upright_error


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: str
    difficulty: str
    seed: int
    max_time: float
    limits: dict
    events: tuple


class SportsTask:
    discipline = "sports"
    category = "event_reactive"
    task_id = ""
    name = ""

    def __init__(self, difficulty, seed, config):
        self.config = config
        self.settings = config["tasks"][self.task_id]
        self.limits = config["difficulties"][difficulty].copy()
        self.layout = make_layout(self.task_id, seed, config)
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(self.task_id), 31]))
        self.events = self.schedule(rng)
        self.spec = EpisodeSpec(
            self.task_id,
            difficulty,
            seed,
            self.settings["max_time"],
            self.limits,
            tuple(self.events),
        )
        self.time = 0.0
        self.steps = self.action_count = self.collision_count = 0
        self.event_log = []
        self.violations = []
        self.physical_failures = []
        self.correct = set()
        self.attempted = set()
        self.dwell = {}
        self.grasped = set()
        self.touched = set()
        self.presses = {}
        self.rises = set()
        self._pressed = {}
        self._press_dwell = {}
        self._release_dwell = {}
        self._armed = {}
        self.goal_achieved = False
        self._terminated = False
        self.completion_time = None
        self.action_time = None
        self.event_time = None
        self.previous = None
        self.phase = "waiting"
        self.commands = []
        self.current_index = None
        self.mole_up_times = {}
        self.red_seen = False
        self.red_stopped = False
        self.red_start = None
        self.lift_time = self.down_time = self.lower_time = None
        self.hold_anchor = None
        self.card_index = 0
        self.bell_attempts = 0
        self.admitted = set()
        self.launched = set()
        self.exited = set()
        self.shots_due = 0
        self.residents = set()
        self.max_occupancy = 0
        self.occupancy = 0
        self.deadline_checked = False
        self._logged_events = set()
        self.hold_progress = {}
        self.trace_state = {}

    def schedule(self, rng):
        return []

    def event(self, name, **data):
        self.event_log.append({"time": self.time, "type": name, **data})

    def violation(self, name):
        if name not in self.violations:
            self.violations.append(name)
            self.event(name)

    def success(self, index):
        if index not in self.correct:
            self.correct.add(index)
            self.event("operation_complete", index=index)
            if self.action_time is None:
                self.action_time = self.time

    def stable(self, obj, name=None):
        p = self.config["physics"]
        pose_valid = (
            abs(rotation_matrix(obj["orientation"])[2, 1]) <= np.sin(p["max_tilt"])
            if name == "dumbbell"
            else upright_error(obj["orientation"]) <= p["max_tilt"]
        )
        return (
            obj["supported"]
            and not obj["robot_contact"]
            and np.linalg.norm(obj["velocity"]) <= p["max_linear_speed"]
            and np.linalg.norm(obj["angular_velocity"]) <= p["max_angular_speed"]
            and pose_valid
        )

    def placed(self, name, target, state, dt, tolerance=0.014):
        obj = state["objects"][name]
        valid = (
            name in self.grasped
            and self.stable(obj, name)
            and np.linalg.norm(obj["position"] - np.asarray(target)) <= tolerance
        )
        key = (name, tuple(target))
        self.dwell[key] = self.dwell.get(key, 0.0) + dt if valid else 0.0
        return self.dwell[key] + 1e-9 >= self.config["physics"]["placement_dwell"]

    def observe(self, state, dt):
        self.time += dt
        self.steps += 1
        self.rises = set()
        for name, obj in state["objects"].items():
            finite = np.all(
                np.isfinite(
                    np.r_[
                        obj["position"],
                        obj["orientation"],
                        obj["velocity"],
                        obj["angular_velocity"],
                    ]
                )
            )
            if not finite:
                if "nonfinite" not in self.physical_failures:
                    self.physical_failures.append("nonfinite")
                self._terminated = True
            if obj["grasped"]:
                self.grasped.add(name)
            if obj["robot_contact"]:
                self.touched.add(name)
            if (
                obj["position"][2] < self.config["scene"]["table_height"] - 0.08
                and name not in self.launched
            ):
                failure = f"drop:{name}"
                if failure not in self.physical_failures:
                    self.physical_failures.append(failure)
                    self.event("drop", object=name)
        for name, control in state["controls"].items():
            active = control["activated"]
            self._release_dwell[name] = 0.0 if active else self._release_dwell.get(name, 0.0) + dt
            if self._release_dwell[name] >= 0.10:
                self._armed[name] = True
            self._press_dwell[name] = self._press_dwell.get(name, 0.0) + dt if active else 0.0
            pressed = self._press_dwell[name] + 1e-9 >= self.config["physics"]["button_dwell"]
            if pressed and not self._pressed.get(name, False) and self._armed.get(name, True):
                self._armed[name] = False
                self.rises.add(name)
                self.presses[name] = self.time - self._press_dwell[name] + dt
                self.event("control_actuated", control=name, onset=self.presses[name])
            self._pressed[name] = pressed
        self.evaluate(state, dt)
        if self.physical_success and self.completion_time is None:
            self.completion_time = self.time
        self.previous = state

    def evaluate(self, state, dt):
        raise NotImplementedError

    @property
    def physical_success(self):
        return bool(self.goal_achieved and not self.physical_failures and self.collision_count == 0)

    @property
    def temporal_success(self):
        return bool(self.goal_achieved and not self.violations)

    @property
    def joint_success(self):
        return self.physical_success and self.temporal_success

    @property
    def terminated(self):
        if self.category == "throughput":
            return self._terminated
        return self._terminated or self.goal_achieved or bool(self.physical_failures)

    def gen_instruction(self):
        return self.name.replace("_", " ")

    def display(self):
        return self.phase.upper(), [0.15, 0.4, 0.9]

    def metrics(self):
        failures = self.physical_failures.copy()
        if self.collision_count > 0 and "arm_collision" not in failures:
            failures.append("arm_collision")
        count = len(self.correct)
        denominator = self.spec.max_time if self.category == "throughput" else max(self.time, 1e-9)
        return {
            "task_id": self.task_id,
            "task_name": self.name,
            "difficulty": self.spec.difficulty,
            "seed": self.spec.seed,
            "category": self.category,
            "task_success": self.physical_success,
            "manipulation_success": self.physical_success,
            "temporal_success": self.temporal_success,
            "joint_success": self.joint_success,
            "completion_time": self.completion_time,
            "temporal_violations": self.violations.copy(),
            "physical_failures": failures,
            "correct_operations": count,
            "attempted_operations": len(self.attempted),
            "operations_per_minute": 60 * count / denominator,
            "accuracy": count / max(1, count, len(self.attempted)),
            "elapsed_time": self.time,
            "number_of_actions": self.action_count,
            "collision_count": self.collision_count,
            "reaction_latency": None
            if self.action_time is None or self.event_time is None
            else self.action_time - self.event_time,
            "deadline_missed": any("late" in s or "miss" in s for s in self.violations),
            "phase": self.phase,
            "full_budget": self.time >= self.spec.max_time - 1e-8,
            "balls_admitted": len(self.admitted),
            "balls_fired": len(self.exited),
            "shots_due": self.shots_due,
            "max_hopper_occupancy": self.max_occupancy,
        }


class QuizBuzzer(SportsTask):
    task_id = "35"
    name = "quiz_buzzer_after_ready"

    def schedule(self, rng):
        return [{"time": float(rng.uniform(*self.settings["ready_range"]))}]

    def evaluate(self, state, dt):
        ready = self.events[0]["time"]
        self.phase = "ready" if self.time >= ready else "wait"
        if self.time >= ready and self.event_time is None:
            self.event_time = self.time
            self.event("ready_light")
        if "buzzer" in self.rises:
            self.attempted.add(0)
            self.action_time = self.presses["buzzer"]
            if self.action_time < ready - 1e-9:
                self.violation("early_press_lockout")
                self.phase = "locked_out"
                self._terminated = True
            else:
                self.success(0)
                self.goal_achieved = True
                if self.action_time - ready >= self.limits["deadline"] - 1e-9:
                    self.violation("late_press")
        if self.time >= ready + self.limits["deadline"] and self.action_time is None:
            self.violation("missed_ready_deadline")

    def gen_instruction(self):
        return f"Press the buzzer only after READY, within {self.limits['deadline']:g} s. An early press locks out the round."

    def display(self):
        return self.phase.upper(), [0.1, 0.9, 0.25] if self.phase == "ready" else [0.9, 0.12, 0.12]


class RhythmPads(SportsTask):
    task_id = "32"
    name = "rhythm_tap_and_hold"
    category = "temporal_window"

    def schedule(self, rng):
        return [
            {
                "time": 2.0 + i * self.settings["interval"],
                "control": f"pad_{int(rng.integers(3))}",
                "hold": self.settings["hold_duration"] if i % 2 else 0.08,
            }
            for i in range(self.settings["rounds"])
        ]

    def evaluate(self, state, dt):
        active = [
            i
            for i, e in enumerate(self.events)
            if e["time"] <= self.time < e["time"] + self.settings["interval"]
        ]
        self.current_index = active[-1] if active else None
        self.phase = "wait" if self.current_index is None else "note"
        for name in self.rises:
            i = self.current_index
            self.attempted.add((self.steps, name))
            if i is None or name != self.events[i]["control"] or i in self.correct:
                self.violation("wrong_or_extra_note")
                continue
            e = self.events[i]
            onset = self.presses[name]
            if not e["time"] - 1e-9 <= onset <= e["time"] + self.limits["note_window"] + 1e-9:
                self.violation(f"note_{i}_outside_window")
            self.hold_progress[i] = 0.0
        for i in list(self.hold_progress):
            if i in self.correct:
                continue
            if self._pressed.get(self.events[i]["control"], False):
                self.hold_progress[i] += dt
                if self.hold_progress[i] + 1e-9 >= self.events[i]["hold"]:
                    self.success(i)
            else:
                self.violation(f"note_{i}_hold_released_early")
                del self.hold_progress[i]
        for i, e in enumerate(self.events):
            if (
                self.time >= e["time"] + self.limits["note_window"] + e["hold"] + 0.1
                and i not in self.correct
            ):
                self.violation(f"note_{i}_missed")
        self.goal_achieved = len(self.correct) == len(self.events)

    def gen_instruction(self):
        return f"Hit each illuminated pad within {self.limits['note_window']:g} s. Hold HOLD notes for {self.settings['hold_duration']:g} s without releasing."

    def display(self):
        if self.current_index is None:
            return "WAIT FOR NOTE", [0.1, 0.3, 0.8]
        e = self.events[self.current_index]
        return f"{e['control'].upper()}  {'HOLD' if e['hold'] > 0.1 else 'TAP'}", [0.15, 0.8, 0.35]


class WhackMoles(SportsTask):
    task_id = "34"
    name = "whack_valid_moles_skip_decoys"

    def schedule(self, rng):
        # Shuffled bags cover all nine holes without a predictable spatial sweep.
        rounds = self.settings["rounds"]
        holes = []
        while len(holes) < rounds:
            holes.extend(rng.permutation(9).tolist())
        valid = np.ones(rounds, dtype=bool)
        valid[rng.choice(rounds, max(1, rounds // 3), replace=False)] = False
        # Avoid long stretches with no playable target while retaining random decoys.
        while any(not valid[i : i + 3].any() for i in range(rounds - 2)):
            rng.shuffle(valid)
        return [
            {
                "time": 2.0 + i * self.settings["interval"],
                "control": f"mole_{holes[i]}",
                "valid": bool(valid[i]),
            }
            for i in range(rounds)
        ]

    def mole_active(self, index, time=None):
        time = self.time if time is None else time
        event = self.events[index]
        end = (
            self.mole_up_times[index] + self.limits["deadline"]
            if index in self.mole_up_times
            else event["time"] + self.settings["rise_timeout"]
        )
        return event["time"] <= time < end

    def evaluate(self, state, dt):
        for i, e in enumerate(self.events):
            if i not in self.mole_up_times and self.mole_active(i):
                if state["controls"][e["control"]]["exposed"]:
                    self.mole_up_times[i] = self.time
                    self.event(
                        "mole_up",
                        index=i,
                        control=e["control"],
                        valid=e["valid"],
                        command_time=e["time"],
                    )
            if (
                i not in self.mole_up_times
                and self.time >= e["time"] + self.settings["rise_timeout"]
            ):
                failure = f"mole_failed_to_rise_{i}"
                if failure not in self.physical_failures:
                    self.physical_failures.append(failure)
                    self.event(failure)
        active = [i for i in range(len(self.events)) if self.mole_active(i)]
        self.current_index = active[-1] if active else None
        for name, control in state["controls"].items():
            if control["contact"] and any(
                self.events[i]["control"] == name and not self.events[i]["valid"] for i in active
            ):
                self.violation("decoy_hit")
        for name in self.rises:
            self.attempted.add((self.steps, name))
            i = self.current_index
            if i is None or self.events[i]["control"] != name or i not in self.mole_up_times:
                self.violation("inactive_mole_hit")
            elif not self.events[i]["valid"]:
                self.violation("decoy_hit")
            else:
                latency = self.presses[name] - self.mole_up_times[i]
                if latency >= self.limits["deadline"]:
                    self.violation(f"late_mole_{i}")
                self.success(i)
                self.event("mole_hit", index=i, latency=latency)
        for i, e in enumerate(self.events):
            if (
                e["valid"]
                and i in self.mole_up_times
                and self.time >= self.mole_up_times[i] + self.limits["deadline"]
                and i not in self.correct
            ):
                self.violation(f"missed_mole_{i}")
        last = len(self.events) - 1
        if (
            last in self.mole_up_times
            and self.time >= self.mole_up_times[last] + self.limits["deadline"]
        ):
            self.goal_achieved = len(self.correct) == sum(e["valid"] for e in self.events)
            self._terminated = True
        self.phase = (
            "wait"
            if self.current_index is None
            else "valid"
            if self.events[self.current_index]["valid"]
            else "decoy"
        )

    def gen_instruction(self):
        return f"Hit green moles while up, within {self.limits['deadline']:g} s. Never contact red decoy moles."

    def display(self):
        return self.phase.upper(), [0.9, 0.1, 0.1] if self.phase == "decoy" else [0.1, 0.8, 0.2]


class CommandGame(SportsTask):
    task_id = "36"
    name = "press_pull_twist_on_command"

    def schedule(self, rng):
        kinds = list(rng.permutation(["press", "pull", "twist"]))
        return [
            {
                "time": 2.0 + i * self.settings["interval"],
                "control": kinds[i % 3],
                "deadline": self.limits["command_deadline"] * self.settings["shrink"] ** i,
            }
            for i in range(self.settings["rounds"])
        ]

    def evaluate(self, state, dt):
        active = [
            i
            for i, e in enumerate(self.events)
            if e["time"] <= self.time < e["time"] + self.settings["interval"]
        ]
        self.current_index = active[-1] if active else None
        for name in self.rises:
            i = self.current_index
            self.attempted.add((self.steps, name))
            if i is None or self.events[i]["control"] != name:
                self.violation("wrong_command_action")
            elif i not in self.correct:
                self.success(i)
                if self.presses[name] < self.events[i]["time"] - 1e-9:
                    self.violation(f"early_command_{i}")
                if self.presses[name] - self.events[i]["time"] >= self.events[i]["deadline"]:
                    self.violation(f"late_command_{i}")
        for i, e in enumerate(self.events):
            if self.time >= e["time"] + e["deadline"] and i not in self.correct:
                self.violation(f"missed_command_{i}")
        self.goal_achieved = len(self.correct) == len(self.events)
        self.phase = (
            "wait" if self.current_index is None else self.events[self.current_index]["control"]
        )

    def gen_instruction(self):
        return "Perform only the displayed PRESS, PULL or TWIST command. Each round's response deadline shrinks; release between rounds."

    def display(self):
        if self.current_index is None:
            return "WAIT", [0.1, 0.3, 0.8]
        e = self.events[self.current_index]
        return f"{e['control'].upper()}  {e['deadline']:.2f}s", [0.9, 0.6, 0.1]


class DumbbellHold(SportsTask):
    task_id = "33"
    name = "lift_hold_lower_on_signal"
    category = "temporal_window"

    def evaluate(self, state, dt):
        o = state["objects"]["dumbbell"]
        initial = np.array(self.layout.bodies["dumbbell"].position)
        lifted = (
            o["position"][2] >= initial[2] + self.settings["lift_height"] - 0.012 and o["grasped"]
        )
        if self.lift_time is None and lifted:
            self.lift_time = self.time
            self.hold_anchor = o["position"].copy()
            self.event("lifted")
        if self.lift_time is None:
            self.phase = "lift"
            return
        if self.down_time is None:
            self.phase = "hold"
            still = (
                lifted
                and np.linalg.norm(o["velocity"]) < 0.025
                and np.linalg.norm(o["angular_velocity"])
                < self.config["physics"]["max_angular_speed"]
                and np.linalg.norm(o["position"] - self.hold_anchor) < 0.012
            )
            self.dwell["hold"] = self.dwell.get("hold", 0.0) + dt if still else 0.0
            if o["position"][2] < self.hold_anchor[2] - 0.03:
                self.violation("lowered_before_down")
            if self.dwell["hold"] >= self.settings["hold_duration"]:
                self.down_time = self.time
                self.event_time = self.time
                self.event("down_signal")
        else:
            self.phase = "down"
            if self.lower_time is None and o["position"][2] < self.hold_anchor[2] - 0.015:
                self.lower_time = self.time
                self.action_time = self.time
                self.event("lowering_started")
                if self.time > self.down_time + self.limits["lower_window"] + 1e-9:
                    self.violation("late_lowering")
            if self.lower_time is None and self.time > self.down_time + self.limits["lower_window"]:
                self.violation("missed_down_window")
            if self.placed("dumbbell", initial, state, dt):
                self.success(0)
                self.goal_achieved = True

    def gen_instruction(self):
        return f"Lift the dumbbell {self.settings['lift_height']:g} m, hold still until DOWN, and start lowering within {self.limits['lower_window']:g} s after DOWN. Release it on the platform."

    def display(self):
        return self.phase.upper(), [0.1, 0.9, 0.2] if self.phase == "down" else [0.95, 0.65, 0.1]


class RedLightPuck(SportsTask):
    task_id = "37"
    name = "push_puck_freeze_on_red"

    def is_red(self, time=None):
        t = self.time if time is None else time
        return (
            t % (self.settings["green_duration"] + self.settings["red_duration"])
            >= self.settings["green_duration"]
        )

    def evaluate(self, state, dt):
        red = self.is_red()
        o = state["objects"]["puck"]
        speed = max(np.linalg.norm(state["tool_velocity"]), np.linalg.norm(o["velocity"]))
        if red and self.phase != "red":
            self.red_seen = True
            self.red_stopped = False
            self.red_start = self.time
            self.dwell["red_stop"] = 0.0
            self.event("red_light")
        if red:
            if speed <= self.settings["stop_speed"]:
                self.dwell["red_stop"] += dt
                if not self.red_stopped and self.dwell["red_stop"] >= 0.08:
                    self.event("stopped", latency=self.time - self.red_start)
                    self.red_stopped = True
            elif self.red_stopped or self.time - self.red_start >= self.limits["stop_deadline"]:
                self.violation("moved_during_red")
            else:
                self.dwell["red_stop"] = 0.0
        self.phase = "red" if red else "green"
        target = np.array(self.layout.targets["puck"])
        in_goal = (
            "puck" in self.touched
            and np.all(
                np.abs(o["position"][:2] - target[:2]) + self.layout.bodies["puck"].radius <= 0.060
            )
            and abs(o["position"][2] - target[2]) <= 0.005
            and self.stable(o)
        )
        self.dwell["goal"] = self.dwell.get("goal", 0) + dt if in_goal else 0.0
        if self.red_seen and not red and self.dwell["goal"] >= 0.2:
            self.success(0)
            self.goal_achieved = True

    def gen_instruction(self):
        return f"Push the puck into the green goal. On RED stop the gripper and puck within {self.limits['stop_deadline']:g} s and keep both still until GREEN."

    def display(self):
        return self.phase.upper(), [0.9, 0.1, 0.1] if self.is_red() else [0.1, 0.9, 0.2]


class BowlingPins(SportsTask):
    task_id = "38"
    name = "set_ten_bowling_pins"
    category = "throughput"

    def evaluate(self, state, dt):
        all_valid = True
        for name, target in self.layout.targets.items():
            valid = self.placed(name, target, state, dt)
            if valid:
                self.success(name)
                self.attempted.add(name)
            all_valid = all_valid and valid
        self.goal_achieved = all_valid
        if self.time >= self.limits["ball_deadline"] and not self.deadline_checked:
            self.deadline_checked = True
            self.event("next_ball", all_ten_set=all_valid)
            if not all_valid:
                self.violation("pins_missing_at_ball")
        self.phase = "set_pins" if not all_valid else "rack_ready"
        if (
            self.time >= self.spec.max_time - 1e-9
            and 60 * len(self.correct) / self.spec.max_time < self.limits["pins_per_min"]
        ):
            self.violation("pin_rate_below_target")

    def gen_instruction(self):
        return f"Set all ten pins upright on their spots before the ball signal at {self.limits['ball_deadline']:g} s. Reach {self.limits['pins_per_min']:g} pins/min over {self.spec.max_time:g} s."

    def display(self):
        if self.time >= self.limits["ball_deadline"]:
            return f"BALL  PINS {len(self.correct)}/10", [0.95, 0.15, 0.1]
        return f"PINS {len(self.correct)}/10", [0.1, 0.8, 0.3] if self.goal_achieved else [
            0.9,
            0.6,
            0.1,
        ]


class SpeedCups(SportsTask):
    task_id = "39"
    name = "speed_cups_cards_and_bell"
    category = "throughput"

    def schedule(self, rng):
        cards = []
        previous = None
        for _ in range(self.settings["cards"]):
            while True:
                order = rng.permutation(5).tolist()
                if order != previous:
                    break
            cards.append({"order": order})
            previous = order
        return cards

    def targets(self):
        return {
            f"cup_{color}": self.layout.targets[f"slot_{slot}"]
            for slot, color in enumerate(self.events[self.card_index]["order"])
        }

    def evaluate(self, state, dt):
        valid = all([self.placed(n, p, state, dt) for n, p in self.targets().items()])
        if "bell" in self.rises:
            self.bell_attempts += 1
            self.attempted.add(self.bell_attempts)
            if valid:
                self.success(self.card_index)
                if self.card_index + 1 < len(self.events):
                    self.card_index += 1
                    self.dwell.clear()
                    self.event("next_card", index=self.card_index)
            else:
                self.violation("bell_with_wrong_arrangement")
        required = int(np.ceil(self.limits["cards_per_min"] * self.spec.max_time / 60 - 1e-9))
        self.goal_achieved = len(self.correct) >= required
        self.phase = "card_" + str(self.card_index + 1)
        if self.time >= self.spec.max_time - 1e-9 and not self.goal_achieved:
            self.violation("card_rate_below_target")

    def gen_instruction(self):
        return f"Match the card colors left-to-right to slots 1-5. Release the cups, ring the bell, then match the next card. At least {self.limits['cards_per_min']:g} correct cards/min."

    def display(self):
        return "CARD " + " ".join(str(i + 1) for i in self.events[self.card_index]["order"]), [
            0.2,
            0.65,
            0.9,
        ]


class BallFeeder(SportsTask):
    task_id = "40"
    name = "feed_ball_machine_without_starvation"
    category = "throughput"

    def schedule(self, rng):
        return [
            {"time": float(t)}
            for t in np.arange(
                self.settings["warmup"],
                self.settings["max_time"] + 1e-8,
                self.limits["fire_interval"],
            )
        ]

    def evaluate(self, state, dt):
        center = np.array(self.layout.targets["hopper"])
        h = self.config["scene"]["table_height"]
        inside = set()
        contained = set()
        for name, o in state["objects"].items():
            radius = self.layout.bodies[name].radius
            p = o["position"]
            tolerance = 0.001  # PhysX contact-offset tolerance, in metres.
            within = (
                np.all(np.abs(p[:2] - center[:2]) + radius <= 0.089 + tolerance)
                and p[2] - radius >= h + 0.012 - tolerance
                and p[2] + radius <= h + 0.13 + tolerance
            )
            if within:
                contained.add(name)
            # Capacity includes partially inserted and held spheres, not just admitted ones.
            nearest = np.clip(
                p,
                [center[0] - 0.089, center[1] - 0.089, h + 0.012],
                [center[0] + 0.089, center[1] + 0.089, h + 0.13],
            )
            if np.linalg.norm(p - nearest) < radius:
                inside.add(name)
            settled = within and not o["robot_contact"] and o["supported"] and name in self.grasped
            self.dwell[name] = self.dwell.get(name, 0) + dt if settled else 0.0
            if self.dwell[name] >= 0.12 and name not in self.admitted:
                self.admitted.add(name)
                self.success(name)
                self.attempted.add(name)
                self.event("ball_admitted", object=name)
            if name in self.launched and o["position"][2] > h + 0.19:
                if name not in self.exited:
                    self.exited.add(name)
                    self.event("ball_fired", object=name)
        self.residents = (contained & self.admitted) - self.launched
        occupancy = len(inside)
        self.occupancy = occupancy
        self.max_occupancy = max(self.max_occupancy, occupancy)
        if occupancy > self.settings["capacity"]:
            self.violation("hopper_over_capacity")
        for i, e in enumerate(self.events):
            if self.time + 1e-9 >= e["time"] and i not in self._logged_events:
                self._logged_events.add(i)
                self.shots_due += 1
                if self.residents:
                    name = sorted(self.residents)[0]
                    self.residents.remove(name)
                    self.launched.add(name)
                    self.commands.append(("fire", name))
                    self.event("fire_command", object=name)
                else:
                    self.violation(f"hopper_starved_{i}")
        if len(self.admitted) < self.shots_due:
            self.violation("fed_less_than_demand")
        self.goal_achieved = self.shots_due == len(self.events) and len(self.exited) == len(
            self.events
        )
        self.phase = "prefill" if self.time < self.settings["warmup"] else "feed"
        if self.time >= self.spec.max_time - 1e-9 and len(self.exited) < self.shots_due:
            self.violation("unverified_firing")

    def gen_instruction(self):
        return f"Load the hopper before firing begins at {self.settings['warmup']:g} s. Keep up with one shot every {self.limits['fire_interval']:g} s, and never exceed {self.settings['capacity']} balls in the hopper."

    def display(self):
        return (
            f"HOPPER {len(self.residents)}/{self.settings['capacity']}  FIRED {len(self.exited)}",
            [0.1, 0.8, 0.3],
        )


class ChessMove:
    name = "chess_move_then_clock"
    category = "event_reactive"
    discipline = "sports"


TASKS = {
    "31": ChessMove,
    "32": RhythmPads,
    "33": DumbbellHold,
    "34": WhackMoles,
    "35": QuizBuzzer,
    "36": CommandGame,
    "37": RedLightPuck,
    "38": BowlingPins,
    "39": SpeedCups,
    "40": BallFeeder,
}


def make_task(task_id, difficulty="easy", seed=0, config=None):
    task_id = str(task_id)
    config = config or load_config()
    if task_id not in TASKS or difficulty not in config["difficulties"]:
        raise ValueError("Unknown sports task or difficulty")
    if task_id == "31":
        from roboclock.envs.chess.task import make_task as make_chess

        c = chess_config(config)
        moves = config["tasks"]["31"]["chess_moves"]
        c["episode"]["move"] = moves[seed % len(moves)]
        task = make_chess("CH1", difficulty, seed, c)
        task.task_id = "31"
        task.spec = replace(task.spec, task_id="31")
        return task
    return TASKS[task_id](difficulty, seed, config)
