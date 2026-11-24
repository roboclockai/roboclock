"""A1 chess exchanges with clock handoffs (``chess/``).

Mirrors ``chess/demo.py``. Two Dual X5 rigs face each other. The policy's
16-dim action drives the WHITE rig; the BLACK rig is the opponent and stays part
of the environment, driven by the recording's privileged chess controller and
retire motion. The ``DuelTask`` evaluator observes every physics tick.

Turn handoff (environment rule): after a valid clock press the active side keeps
the turn until its arm is back within 0.025 rad of its reset joint pose, at least
1.5 s after the press was first observed. After the last turn the episode ends 2 s
after that return.

The scripted retire motion (BLACK's and the WHITE oracle's, ``retire_command``) first
lifts the tool above 1.09 m and then returns. Until 2026-11-10 the handoff also
required that lift, as the recording driver (``chess/demo.py``) still does. The
waiting pose holds the tool at about 1.06 m and the instruction only asks for the
return, so a policy that went straight back never handed the turn over. The lift is
how the scripted motion clears the pieces, not something the handoff needs: pieces or
fixtures an arm disturbs on its way back are judged by the task's own checks. The
scripted motion always lifts before it returns, so its handoff falls on the same
control step as before.
"""

import copy
from types import SimpleNamespace

import numpy as np

from roboclock.envs.sports.adapters.base import Adapter, ensure_extensions, episode_rates
from roboclock.envs.sports.joints import joint_step_speed

RETIRE_HEIGHT = 1.10
LIFTED_HEIGHT = 1.09
RETURN_TOLERANCE = 0.025
RETIRE_DWELL = 1.5
END_POST_ROLL = 2.0
START_DELAY = 2.0
# Retire motion: tool lift (m/s); joint step limits while lifting and while returning
# to the waiting pose (rad/s, tuned at RETIRE_TUNED_HZ).
RETIRE_SPEED = 0.22
RETIRE_LIFT_JOINT_SPEED = 1.5
RETIRE_RETURN_JOINT_SPEED = 1.2
RETIRE_TUNED_HZ = 30


class ChessAdapter(Adapter):
    family = "chess"

    def build(self):
        ensure_extensions()
        from pxr import UsdGeom

        from roboclock.envs.sports.arcade.chess_oracle import RefinedChessOracle
        from roboclock.envs.sports.chess.backend import DuelBackend
        from roboclock.envs.sports.chess.task import DuelTask
        from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_camera

        c = self.c
        t = DuelTask(self.spec.difficulty, self.spec.seed, c)
        b = DuelBackend(self.runtime, c)
        self.backend, self.t = b, t
        b.reset(t)
        apply_presentation_camera(b)
        t.piece_envelopes = b.piece_envelopes
        self.robots = dict(white=b.robot, black=b.opponent)
        self.home = {a: r.get_joint_positions().copy() for a, r in self.robots.items()}
        self.views, self.configs = {}, {}
        for actor, robot in self.robots.items():
            cfg = copy.deepcopy(c)
            # A parallel jaw grasp is invariant to 180 degrees about approach; this
            # keeps each X5 wrist roll inside its limits (as in the recordings).
            cfg["control"]["tool_yaw"] = 0.0 if actor == "white" else np.pi
            self.configs[actor] = cfg
            self.views[actor] = SimpleNamespace(
                robot=robot,
                state=lambda actor=actor: b.states[actor],
                button_rest={"robot": b.button_rest["robot" if actor == "white" else "opponent"]},
                rig=b.rig if actor == "white" else b.opponent_rig,
                active_side=b.active_side if actor == "white" else b.opponent_active_side,
            )
        # The opponent's controller is environment dynamics, not the policy.
        self.black = RefinedChessOracle(
            SimpleNamespace(backend=self.views["black"], task=t, config=self.configs["black"])
        )
        self.black.reset({})
        self.retire = None
        self.end_time = None
        self.previous_collisions = (0, 0, 0)
        self.since_control = 0
        title = b.stage.GetPrimAtPath("/World/Title")
        if title:
            UsdGeom.Imageable(title).MakeInvisible()
        self._shown = None
        self.cues()
        self.after_build()

    def rigs(self):
        return [self.backend.rig, self.backend.opponent_rig]

    @property
    def time(self):
        return float(self.t.time)

    @property
    def done(self):
        t = self.t
        return bool(
            (t.done and not t.turn_complete)
            or (self.end_time is not None and t.time >= self.end_time)
        )

    def state(self):
        return self.backend.states

    # -- environment side: opponent and turn handoff ---------------------------------
    def retire_state(self, actor):
        """Per-turn retire bookkeeping, started when the press is first observed."""
        t = self.t
        if self.retire is None or self.retire["turn"] != t.turn_index:
            origin = self.backend.states[actor]["tool_position"].copy()
            self.retire = dict(
                turn=t.turn_index, start=t.time, origin=origin, command=origin.copy(), lifted=False
            )
        return self.retire

    def handoff(self, actor):
        """Pass the turn once the retired arm is back at its waiting pose."""
        t, b = self.t, self.backend
        r = self.retire_state(actor)
        # Latch of the scripted retire motion (BLACK's reads it in ``retire_command``);
        # the handoff itself needs only the return (module docstring).
        r["lifted"] = r["lifted"] or b.states[actor]["tool_position"][2] > LIFTED_HEIGHT
        view = self.views[actor]
        ids = view.rig.arm_ids[view.active_side]
        q = self.robots[actor].get_joint_positions()
        if (
            t.time - r["start"] > RETIRE_DWELL
            and np.max(np.abs(q[ids] - self.home[actor][ids])) < RETURN_TOLERANCE
        ):
            if t.turn_index + 1 == len(t.sequence):
                t.done = True
                if self.end_time is None:
                    self.end_time = t.time + END_POST_ROLL
            else:
                t.turn_index += 1
                t.begin_turn()
                if t.actor == "black":
                    self.black.reset({})
                self.retire = None

    def opponent_control(self):
        """The recording driver's BLACK control step (acts only on BLACK's turn)."""
        t, b = self.t, self.backend
        if t.actor != "black":
            if t.turn_complete:
                self.handoff("white")
            return
        rig = b.opponent_rig
        if t.turn_complete:
            r = self.retire_state("black")
            command = retire_command(self, "black", self.black, r)
            rig.command = command
            rig.apply()
            self.handoff("black")
        elif t.time > START_DELAY:
            action = self.black.act({"joint_positions": b.opponent.get_joint_positions()})
            rig.command = np.asarray(action).copy()
            rig.apply()

    # -- policy side ---------------------------------------------------------------
    def apply(self, native):
        rig = self.backend.rig
        rig.command = np.array(native, copy=True)
        rig.apply()
        self.opponent_control()
        self.since_control = 0

    def tick(self):
        b, t = self.backend, self.t
        b.tick()
        collisions = (b.collision_count, b.black_collision_count, b.inter_arm_collision_steps)
        if any(now > before for now, before in zip(collisions, self.previous_collisions)):
            t.violation("arm_collision")
        self.previous_collisions = collisions
        t.observe_duel(b.states, self.physics_dt)
        self.ticks += 1
        self.since_control += 1

    def control_step(self):
        # The recording driver always completes every tick of a period.
        for _ in range(self.decimation):
            self.tick()
        self.after_control()

    def advance(self, count):
        # The opponent keeps its control rate while the policy's command is held.
        for _ in range(count):
            if self.done:
                break
            if self.since_control >= self.decimation:
                self.opponent_control()
                self.since_control = 0
            self.tick()
        self.after_control()

    def after_control(self):
        self.cues()

    def cues(self):
        """Turn and clock displays from the recording driver (visual only)."""
        from pxr import UsdGeom

        from roboclock.envs.sports.arcade.scenes import vertical_label

        t, b = self.t, self.backend
        complete = len(t.turn_results) == len(t.sequence)
        clock_start = t.last_press if t.turn_complete else t.opponent_time
        remaining = max(0, clock_start + t.spec.deadline - t.time)
        title = f"{t.actor.upper()} / {t.san}"
        if t.turn_complete and not complete:
            next_move = t.board.parse_uci(t.sequence[t.turn_index + 1])
            title = f"{'WHITE' if t.board.turn else 'BLACK'} / {t.board.san(next_move)}"
        if complete:
            title = "MATCH COMPLETE"
        timer = (
            "TWO EXCHANGES COMPLETE"
            if complete
            else ("FIRST MOVE / THEN CLOCK" if t.last_press is None else f"CLOCK {remaining:.1f}s")
        )
        value = (title, timer)
        if self._shown != value:
            vertical_label(b, "/World/Duel/Who", title, [0.48, 0.615, 1.435], [0.62, 0.060])
            vertical_label(b, "/World/Duel/Clock", timer, [0.48, 0.615, 1.355], [0.62, 0.045])
            for suffix in ["Title", "Subtitle"]:
                prim = b.stage.GetPrimAtPath("/World/Duel/TurnPanel/" + suffix)
                if prim:
                    UsdGeom.Imageable(prim).MakeInvisible()
            self._shown = value

    def events(self):
        return self.t.event_log

    def result(self):
        b, t = self.backend, self.t
        r = t.result()
        r.update(
            difficulty=self.spec.difficulty,
            seed=self.spec.seed,
            randomization=self.c["randomization"],
            arm_collision_count={"white": b.collision_count, "black": b.black_collision_count},
            inter_arm_collision_steps=b.inter_arm_collision_steps,
            controller="privileged scripted state-feedback, native articulation drives",
            opponent="BLACK Dual X5 driven by the environment's privileged chess controller",
            embodiment=self.c["embodiment"],
            **episode_rates(self),
        )
        return r

    def joint_success(self):
        return bool(self.t.result()["success"])

    def make_controller(self):
        self.controller = WhiteChessController(self)
        return self.controller


def retire_command(adapter, actor, oracle, r):
    """Lift above the board, then return to the reset joint pose (recording driver)."""
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    b = adapter.backend
    view = adapter.views[actor]
    rig, robot, side = view.rig, view.robot, view.active_side
    target = np.r_[r["origin"][:2], RETIRE_HEIGHT]
    ee = b.states[actor]["tool_position"]
    delta = target - r["command"]
    length = np.linalg.norm(delta)
    dt = oracle.dt
    step = r["command"] + delta * min(1, RETIRE_SPEED * dt / max(length, 1e-9))
    r["command"] = step.copy()
    action, solved = oracle.solver.compute_inverse_kinematics(
        step + [0, 0, 0.0034],
        euler_angles_to_quat(np.array([0, np.pi, oracle.env.config["control"]["tool_yaw"]])),
    )
    q = robot.get_joint_positions().copy()
    lifted = r["lifted"] or ee[2] > LIFTED_HEIGHT
    ids = rig.arm_ids[side]
    if lifted:
        delta_q = adapter.home[actor][ids] - q[ids]
        limit = joint_step_speed(RETIRE_RETURN_JOINT_SPEED, RETIRE_TUNED_HZ, dt) * dt
        q[ids] += delta_q * min(1, limit / max(1e-9, np.max(np.abs(delta_q))))
    elif solved:
        indices = action.joint_indices
        if indices is None:
            indices = range(len(action.joint_positions))
        moves = [(i, v - q[i]) for i, v in zip(indices, action.joint_positions) if v is not None]
        limit = joint_step_speed(RETIRE_LIFT_JOINT_SPEED, RETIRE_TUNED_HZ, dt) * dt
        scale = min(1, limit / max(1e-9, max(abs(d) for _, d in moves)))
        for i, d in moves:
            q[i] += d * scale
    command = np.array(rig.command, copy=True)
    command[ids] = q[ids]
    command[rig.grip_ids[side]] = np.clip(0.019, 0, 0.044)
    return command


class WhiteChessController:
    """WHITE's recorded privileged controller: move, clock press, then retire."""

    def __init__(self, adapter):
        from roboclock.envs.sports.arcade.chess_oracle import RefinedChessOracle

        self.adapter = adapter
        t = adapter.t
        self.oracle = RefinedChessOracle(
            SimpleNamespace(backend=adapter.views["white"], task=t, config=adapter.configs["white"])
        )
        self.oracle.reset({})
        self.turn = t.turn_index
        self.retire = None

    @property
    def phase(self):
        return self.oracle.phase

    def act(self):
        a = self.adapter
        t, rig = a.t, a.backend.rig
        if t.actor != "white":
            return np.array(rig.command, copy=True)
        if t.turn_complete:
            if self.retire is None or self.retire["turn"] != t.turn_index:
                origin = a.backend.states["white"]["tool_position"].copy()
                self.retire = dict(
                    turn=t.turn_index, origin=origin, command=origin.copy(), lifted=False
                )
            command = retire_command(a, "white", self.oracle, self.retire)
            ee = a.backend.states["white"]["tool_position"]
            self.retire["lifted"] = self.retire["lifted"] or ee[2] > LIFTED_HEIGHT
            return command
        if t.time <= START_DELAY:
            return np.array(rig.command, copy=True)
        if self.turn != t.turn_index:
            self.oracle.reset({})
            self.turn = t.turn_index
        action = self.oracle.act({"joint_positions": a.backend.robot.get_joint_positions()})
        return np.asarray(action).copy()
