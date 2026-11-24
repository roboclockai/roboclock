"""Legal alternating moves, released placement and physical clock handoffs."""

import numpy as np

from roboclock.envs.chess.rules import Operation, physical_plan
from roboclock.envs.sports.arcade.chess_backend import MeshChessTask

# The side the environment plays (the policy plays WHITE). It gets the turn only when
# WHITE's arm is back at its waiting pose, and that return has no time limit, so its
# deadline counts from the handoff: the time WHITE takes to hand over is not BLACK's
# lateness and is never booked as a missed deadline. WHITE's own deadline still counts
# from BLACK's press, as its instruction says.
ENVIRONMENT_ACTOR = "black"


class DuelTask(MeshChessTask):
    def __init__(self, difficulty, seed, config):
        super().__init__(difficulty, seed, config)
        self.sequence = config["duel"]["moves"]
        self.locations = {name: name[6:] for name in self.initial_symbols}
        self.turn_index = 0
        self.turn_results = []
        self.last_press = None
        self.turn_complete = False
        self.done = False
        self.begin_turn()

    def begin_turn(self):
        self.move = self.board.parse_uci(self.sequence[self.turn_index])
        self.san = self.board.san(self.move)
        operations, _ = physical_plan(self.board, self.move)
        by_square = {square: name for name, square in self.locations.items() if square != "capture"}
        self.operations = [
            Operation(by_square[op.piece[6:]], op.destination, op.kind) for op in operations
        ]
        self.expected = dict(self.locations)
        for op in self.operations:
            self.expected[op.piece] = op.destination
        self.actor = "white" if self.board.turn else "black"
        self.opponent_time = self.last_press if self.last_press is not None else 0.0
        # When this turn's deadline starts counting (see ENVIRONMENT_ACTOR).
        self.clock_start = self.time if self.actor == ENVIRONMENT_ACTOR else self.opponent_time
        self.turn_complete = False
        self.placement_valid = False
        self.placement_time = None
        self._settled = self._press_dwell = 0.0
        self._previous_pressed = False
        self.grasped = set()
        self.turn_violations = []
        self.event(
            "turn_start",
            actor=self.actor,
            uci=self.move.uci(),
            deadline=self.clock_start + self.spec.deadline,
        )

    def violation(self, name):
        if name not in self.turn_violations:
            self.turn_violations.append(name)
            self.violations.append(f"{self.turn_index}:{name}")
            self.event(name, actor=self.actor)

    def observe_duel(self, states, dt):
        self.time += dt
        if self.done:
            return
        state = states[self.actor]
        for name, obj in state["objects"].items():
            if (
                not np.isfinite(
                    np.r_[
                        obj["position"],
                        obj["velocity"],
                        obj["orientation"],
                        obj["angular_velocity"],
                    ]
                ).all()
                or obj["position"][2] < 0.72
            ):
                self.violation("invalid_or_dropped_piece")
                self.done = True
                return
            if obj["grasped"]:
                self.grasped.add(name)
            if (
                name not in {op.piece for op in self.operations}
                and np.linalg.norm(obj["position"] - self.position(self.locations[name])) > 0.012
            ):
                self.violation("unrelated_piece_moved")
        if self.turn_complete:
            if set(state["objects"]) != set(self.expected) or not all(
                self.located(n, o) for n, o in state["objects"].items()
            ):
                self.violation("placement_lost_after_clock")
            return
        candidate = (
            set(state["objects"]) == set(self.expected)
            and all(self.located(n, o) for n, o in state["objects"].items())
            and all(op.piece in self.grasped for op in self.operations)
        )
        self._settled = self._settled + dt if candidate else 0
        self.placement_valid = self._settled >= self.config["physics"]["placement_dwell"]
        if self.placement_valid and self.placement_time is None:
            self.placement_time = self.time
            self.event("legal_move_placed", actor=self.actor, uci=self.move.uci())
        if state["opponent_robot_contact"]:
            self.violation("wrong_clock_button")
        pressed = state["robot_pressed"]
        if pressed and not self._previous_pressed:
            self._press_start = self.time
            if not self.placement_valid:
                self.violation("clock_before_placement")
        self._previous_pressed = pressed
        self._press_dwell = self._press_dwell + dt if pressed else 0
        if self._press_dwell >= self.config["physics"]["button_dwell"]:
            physical = bool(self.placement_valid and self.placement_time < self._press_start)
            latency = self._press_start - self.opponent_time
            # The first black turn establishes the first actual opponent clock event.
            timed = bool(
                self.last_press is None
                or self._press_start - self.clock_start <= self.spec.deadline + 1e-9
            )
            if not timed:
                self.violation("deadline_missed")
            result = dict(
                actor=self.actor,
                move=self.move.uci(),
                san=self.san,
                placement_time=self.placement_time,
                press_time=self._press_start,
                opponent_press_time=self.last_press,
                clock_start=self.clock_start,
                latency=latency,
                deadline=self.spec.deadline,
                physical_success=physical,
                temporal_success=timed,
                success=physical and timed and not self.turn_violations,
                violations=self.turn_violations.copy(),
            )
            self.turn_results.append(result)
            self.event("clock_handoff", **result)
            if not physical:
                self.done = True
                return
            self.board.push(self.move)
            self.locations = dict(self.expected)
            self.last_press = self._press_start
            self.turn_complete = True
        if self.time > self.config["duel"]["max_time"]:
            self.violation("episode_timeout")
            self.done = True

    def result(self):
        complete = len(self.turn_results) == len(self.sequence)
        return dict(
            task="A1",
            variant="duel",
            full_episode=self.done,
            sim_time=self.time,
            physical_success=complete and all(r["physical_success"] for r in self.turn_results),
            temporal_success=complete and all(r["temporal_success"] for r in self.turn_results),
            success=complete
            and all(r["success"] for r in self.turn_results)
            and not self.violations,
            turns=self.turn_results,
            events=self.event_log,
            failures=self.violations,
            final_fen=self.board.fen(),
        )
