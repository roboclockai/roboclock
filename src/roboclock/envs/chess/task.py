"""Measured placement-before-clock predicates; all times are physics time."""

from dataclasses import dataclass

import chess
import numpy as np

from roboclock.envs.chess.config import load_config
from roboclock.envs.chess.rules import game_board, physical_plan
from roboclock.physics.geometry import rotation_matrix, upright_error


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: str
    difficulty: str
    seed: int
    max_time: float
    deadline: float
    opponent_press_time: float
    fen: str
    move_uci: str
    history: tuple[str, ...]


class ChessTask:
    name = "make_move_then_press_clock"
    category = "event_reactive"
    discipline = "chess"
    task_id = "CH1"

    def __init__(self, difficulty, seed, config):
        self.config = config
        self.board = game_board(config["episode"])
        rng = np.random.default_rng(seed)
        requested = config["episode"].get("move")
        self.move = (
            self.board.parse_uci(requested)
            if requested
            else sorted(self.board.legal_moves, key=lambda m: m.uci())[
                int(rng.integers(self.board.legal_moves.count()))
            ]
        )
        self.san = self.board.san(self.move)
        self.operations, self.expected = physical_plan(self.board, self.move)
        self.spec = EpisodeSpec(
            "CH1",
            difficulty,
            seed,
            config["episode"]["max_time"],
            config["difficulties"][difficulty]["deadline"],
            float(rng.uniform(*config["episode"]["opponent_press_range"])),
            self.board.fen(),
            self.move.uci(),
            tuple(config["episode"].get("history", [])),
        )
        self.initial_symbols = {
            "piece_" + chess.square_name(sq): p.symbol() for sq, p in self.board.piece_map().items()
        }
        self.time = 0.0
        self.steps = self.action_count = self.collision_count = 0
        self.event_log = []
        self.violations = []
        self.terminated = self.joint_success = self.physical_success = False
        self.opponent_time = self.placement_time = self.press_time = None
        self.first_motion_time = None
        self.placement_valid = False
        self._settled = self._press_dwell = 0.0
        self._previous_pressed = False
        self.failure_reason = None
        self.grasped = set()
        self.last_state = None

    def position(self, destination):
        scene = self.config["scene"]
        z = scene["table_height"] + scene["board_height"] + scene["piece_height"] / 2
        if destination == "capture":
            return np.array([0.35, 0.31, scene["table_height"] + 0.006 + scene["piece_height"] / 2])
        sq = chess.parse_square(destination)
        x, y = scene["board_center"]
        s = scene["square_size"]
        # Robot at negative X; white's back rank is nearest the robot.
        return np.array(
            [x + (chess.square_rank(sq) - 3.5) * s, y - (chess.square_file(sq) - 3.5) * s, z]
        )

    def event(self, name, **data):
        self.event_log.append({"event": name, "time": self.time, **data})

    def violation(self, name):
        if name not in self.violations:
            self.violations.append(name)
            self.event(name)

    def located(self, name, obj):
        target = self.position(self.expected[name])
        scene, physics = self.config["scene"], self.config["physics"]
        axis = rotation_matrix(obj["orientation"])[:, 2]
        extents = scene["piece_radius"] * np.sqrt(np.maximum(0.0, 1.0 - axis**2)) + scene[
            "piece_height"
        ] / 2 * np.abs(axis)
        clearance = scene["square_size"] / 2 - extents[:2] - 0.001
        if self.expected[name] == "capture":
            clearance = 0.0475 - extents[:2] - 0.001
        return bool(
            np.all(np.abs(obj["position"][:2] - target[:2]) <= clearance)
            and abs(obj["position"][2] - target[2]) <= 0.004
            and upright_error(obj["orientation"]) <= physics["max_tilt"]
            and np.linalg.norm(obj["velocity"]) <= physics["max_linear_speed"]
            and np.linalg.norm(obj["angular_velocity"]) <= physics["max_angular_speed"]
            and obj["supported"]
            and not obj["robot_contact"]
        )

    def observe(self, state, dt):
        if self.terminated:
            return
        self.time += dt
        self.steps += 1
        self.last_state = state
        if state.get("opponent_robot_contact", False):
            self.violation("wrong_clock_button")
        if self.opponent_time is None and state["opponent_pressed"]:
            self.opponent_time = self.time
            self.event("opponent_press", deadline=self.time + self.spec.deadline)
        for name, obj in state["objects"].items():
            if not np.all(
                np.isfinite(
                    np.r_[
                        obj["position"],
                        obj["orientation"],
                        obj["velocity"],
                        obj["angular_velocity"],
                    ]
                )
            ):
                self.failure_reason = "nonfinite_state"
                self.terminated = True
                self.event(self.failure_reason, piece=name)
                return
            if obj["position"][2] < self.config["scene"]["table_height"] - 0.03:
                self.failure_reason = "piece_dropped"
                self.terminated = True
                self.event(self.failure_reason, piece=name)
                return
            if obj["grasped"]:
                self.grasped.add(name)
            initial = self.position(name.removeprefix("piece_"))
            if np.linalg.norm(obj["position"] - initial) > 0.008:
                if self.first_motion_time is None:
                    self.first_motion_time = self.time
                    self.event("first_piece_motion")
                if self.opponent_time is None:
                    self.violation("move_before_opponent_press")
        candidate = set(state["objects"]) == set(self.expected) and all(
            self.located(n, obj) for n, obj in state["objects"].items()
        )
        candidate = candidate and all(op.piece in self.grasped for op in self.operations)
        self._settled = self._settled + dt if candidate else 0.0
        self.placement_valid = self._settled + 1e-9 >= self.config["physics"]["placement_dwell"]
        if self.placement_valid and self.placement_time is None:
            self.placement_time = self.time
            self.event("legal_move_placed", uci=self.move.uci(), san=self.san)
        pressed = state["robot_pressed"]
        if pressed and not self._previous_pressed:
            # The first physical contact/depression is what must follow settling.
            if not self.placement_valid:
                self.violation("clock_before_placement")
            if self.opponent_time is None:
                self.violation("clock_before_opponent_press")
            self._press_start = self.time
        self._previous_pressed = pressed
        self._press_dwell = self._press_dwell + dt if pressed else 0.0
        if self._press_dwell + 1e-9 >= self.config["physics"]["button_dwell"]:
            self.press_time = self._press_start
            self.physical_success = self.placement_valid and self.placement_time < self.press_time
            if (
                self.opponent_time is not None
                and self.press_time > self.opponent_time + self.spec.deadline + 1e-9
            ):
                self.violation("deadline_missed")
            self.joint_success = bool(
                self.physical_success and self.opponent_time is not None and not self.violations
            )
            self.event("robot_press", physical_success=self.physical_success)
            if self.physical_success:
                self.board.push(self.move)
                self.event("move_committed", fen=self.board.fen())
            self.terminated = True
        # Retain late physical completions separately from timing failures.
        if (
            self.opponent_time is not None
            and self.time > self.opponent_time + self.spec.deadline + 1e-9
        ):
            pending_on_time = (
                pressed and self._press_start <= self.opponent_time + self.spec.deadline + 1e-9
            )
            if self.press_time is None and not pending_on_time:
                self.violation("deadline_missed")
        if self.time >= self.spec.max_time - 1e-9 and not self.terminated:
            self.failure_reason = "time_budget"

    def gen_instruction(self):
        return (
            f"After the opponent presses the clock, play {self.san} ({self.move.uci()}). "
            f"Place and release all required pieces, then press YOUR clock within "
            f"{self.spec.deadline:g} s of the opponent's press. Captures go in the tray."
        )

    def metrics(self):
        latency = (
            None
            if self.press_time is None or self.opponent_time is None
            else self.press_time - self.opponent_time
        )
        outcome = self.board.outcome(claim_draw=self.config["episode"].get("claim_draw", False))
        return {
            "task_id": self.task_id,
            "difficulty": self.spec.difficulty,
            "seed": self.spec.seed,
            "category": self.category,
            "task_success": bool(self.physical_success),
            "manipulation_success": bool(self.physical_success),
            "joint_success": self.joint_success,
            "temporal_success": bool(
                self.press_time is not None
                and self.opponent_time is not None
                and not self.violations
            ),
            "temporal_violations": self.violations.copy(),
            "deadline_missed": "deadline_missed" in self.violations,
            "opponent_press_time": self.opponent_time,
            "placement_time": self.placement_time,
            "robot_press_time": self.press_time,
            "completion_time": self.press_time,
            "reaction_latency": latency,
            "deadline_seconds": self.spec.deadline,
            "temporal_error": None if latency is None else max(0.0, latency - self.spec.deadline),
            "number_of_actions": self.action_count,
            "collision_count": self.collision_count,
            "move_uci": self.move.uci(),
            "move_san": self.san,
            "initial_fen": self.spec.fen,
            "final_fen": self.board.fen(),
            "failure_reason": self.failure_reason,
            "game_result": None if outcome is None else outcome.result(),
            "game_termination": None if outcome is None else outcome.termination.name,
            "can_claim_threefold_repetition": self.board.can_claim_threefold_repetition(),
            "can_claim_fifty_moves": self.board.can_claim_fifty_moves(),
        }


TASKS = {"CH1": ChessTask}


def make_task(task_id, difficulty="easy", seed=0, config=None):
    if task_id not in TASKS:
        raise ValueError(f"Unknown chess task: {task_id}")
    config = config or load_config()
    if difficulty not in config["difficulties"]:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return ChessTask(difficulty, seed, config)
