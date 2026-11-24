"""Record uninterrupted physical chess exchanges; no scripted piece trajectories."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ, VIDEO_FPS
from roboclock.envs.sports.adapters.chess import (
    RETIRE_LIFT_JOINT_SPEED,
    RETIRE_RETURN_JOINT_SPEED,
    RETIRE_SPEED,
    RETIRE_TUNED_HZ,
)
from roboclock.envs.sports.chess.config import config
from roboclock.envs.sports.joints import joint_step_speed
from roboclock.envs.sports.arcade.demo import serial
from roboclock.simulators.isaac.version import simulator_label


def run(
    runtime, *, output, difficulty="easy", seed=0, duration=140, video=False, embodiment="dual_arx_x5", presentation_only=False
):
    """Run one complete physical exchange and retain evidence, including failures."""
    args = SimpleNamespace(
        output=Path(output), difficulty=difficulty, seed=seed, duration=duration, video=video
    )
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("src/roboclock").rglob("*.py")
    }
    c = config(args.difficulty, args.seed)
    from roboclock.simulators.isaac.embodiments.config import configure

    configure(c, "A1", embodiment)
    from roboclock.envs.sports.recording import configure as configure_recording

    configure_recording(c)
    (args.output / "config.json").write_text(json.dumps(c, indent=2))
    from roboclock.envs.sports.adapters.base import ensure_extensions

    # Isaac Sim 6.0.x does not enable the manipulator examples the backends import.
    ensure_extensions()
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import UsdGeom

    from roboclock.envs.sports.chess.backend import DuelBackend
    from roboclock.envs.sports.chess.task import DuelTask
    from roboclock.envs.sports.arcade.chess_oracle import RefinedChessOracle
    from roboclock.envs.sports.arcade.scenes import vertical_label
    from roboclock.simulators.isaac.rendering import capture_camera, renderer_state
    from roboclock.simulators.isaac.video import VideoEncoder

    t = DuelTask(args.difficulty, args.seed, c)
    b = DuelBackend(runtime, c)
    b.reset(t)
    from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_camera

    apply_presentation_camera(b)
    from roboclock.envs.sports import recording

    # Finger speed as in roboclock eval (both rigs); head camera before the first render.
    rig_setup = recording.prepare(b, video=video, task_id="A1", rigs=[b.rig, b.opponent_rig])
    t.piece_envelopes = b.piece_envelopes
    policies = {}
    robots = dict(white=b.robot, black=b.opponent)
    home = {actor: robot.get_joint_positions().copy() for actor, robot in robots.items()}
    for actor, robot in robots.items():
        cfg = copy.deepcopy(c)
        # A parallel jaw grasp is invariant to 180 degrees about approach.
        # Pick the equivalent orientation inside the X5 wrist-roll limits.
        cfg["control"]["tool_yaw"] = 0.0 if actor == "white" else np.pi
        view = SimpleNamespace(
            robot=robot,
            state=lambda actor=actor: b.states[actor],
            button_rest={"robot": b.button_rest["robot" if actor == "white" else "opponent"]},
        )
        view.rig = b.rig if actor == "white" else b.opponent_rig
        view.active_side = b.active_side if actor == "white" else b.opponent_active_side
        policies[actor] = RefinedChessOracle(SimpleNamespace(backend=view, task=t, config=cfg))
        policies[actor].reset({})
    title_prim = b.stage.GetPrimAtPath("/World/Title")
    if title_prim:
        UsdGeom.Imageable(title_prim).MakeInvisible()
    shown = None

    def cues():
        nonlocal shown
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
        if shown != value:
            vertical_label(b, "/World/Duel/Who", title, [0.48, 0.615, 1.435], [0.62, 0.060])
            vertical_label(b, "/World/Duel/Clock", timer, [0.48, 0.615, 1.355], [0.62, 0.045])
            for suffix in ["Title", "Subtitle"]:
                prim = b.stage.GetPrimAtPath("/World/Duel/TurnPanel/" + suffix)
                if prim:
                    UsdGeom.Imageable(prim).MakeInvisible()
            shown = value

    def frame():
        before = b.world.current_time
        for _ in range(3):
            b.world.render()
        capture_camera()
        assert before == b.world.current_time
        return np.asarray(b.camera.get_rgba()[:, :, :3], np.uint8).copy()

    cues()
    Image.fromarray(frame()).save(args.output / "initial.png")
    b.stage.GetRootLayer().Export(str((args.output / "scene.usda").resolve()))
    encoder = (
        VideoEncoder(args.output / "A1.mp4", c["sim"]["camera_resolution"], VIDEO_FPS)
        if args.video
        else None
    )
    encoder = recording.attach(encoder, b, args.output)
    if encoder:
        encoder.write(frame())
    next_frame = 1 / VIDEO_FPS
    video_frames = 1 if encoder else 0
    last_print = -5
    retire_start = None
    retire_origin = None
    retire_command = None
    lifted = False
    end_time = None
    previous_collisions = (0, 0, 0)
    with gzip.open(args.output / "trace.jsonl.gz", "wt") as trace:
        while t.time < args.duration:
            actor = t.actor
            robot = robots[actor]
            oracle = policies[actor]
            rig = getattr(oracle.env.backend, "rig", None)
            arm_ids = rig.arm_ids[oracle.env.backend.active_side] if rig else np.arange(7)
            if t.turn_complete:
                if retire_start is None:
                    retire_start = t.time
                    retire_origin = b.states[actor]["tool_position"].copy()
                    retire_command = retire_origin.copy()
                    lifted = False
                elapsed = t.time - retire_start
                target = np.r_[retire_origin[:2], 1.10]
                ee = b.states[actor]["tool_position"]
                delta = target - retire_command
                length = np.linalg.norm(delta)
                dt = oracle.dt
                step = retire_command + delta * min(1, RETIRE_SPEED * dt / max(length, 1e-9))
                retire_command = step.copy()
                action, solved = oracle.solver.compute_inverse_kinematics(
                    step + [0, 0, 0.0034],
                    euler_angles_to_quat(np.array([0, np.pi, oracle.env.config["control"]["tool_yaw"]])),
                )
                q = robot.get_joint_positions().copy()
                lifted = lifted or ee[2] > 1.09
                if lifted:
                    delta_q = home[actor][arm_ids] - q[arm_ids]
                    limit = joint_step_speed(RETIRE_RETURN_JOINT_SPEED, RETIRE_TUNED_HZ, dt) * dt
                    q[arm_ids] += delta_q * min(1, limit / max(1e-9, np.max(np.abs(delta_q))))
                elif solved:
                    indices = action.joint_indices
                    if indices is None:
                        indices = range(len(action.joint_positions))
                    moves = [
                        (i, v - q[i])
                        for i, v in zip(indices, action.joint_positions)
                        if v is not None
                    ]
                    limit = joint_step_speed(RETIRE_LIFT_JOINT_SPEED, RETIRE_TUNED_HZ, dt) * dt
                    scale = min(1, limit / max(1e-9, max(abs(d) for _, d in moves)))
                    for i, d in moves:
                        q[i] += d * scale
                if rig is None:
                    q[7:] = 0.019
                if rig is not None:
                    rig.set_arm_command(oracle.env.backend.active_side, q)
                    rig.grip(oracle.env.backend.active_side, 0.019)
                    rig.apply()
                else:
                    robot.apply_action(ArticulationAction(joint_positions=q))
                if (
                    lifted
                    and elapsed > 1.5
                    and np.max(np.abs(robot.get_joint_positions()[arm_ids] - home[actor][arm_ids]))
                    < 0.025
                ):
                    if t.turn_index + 1 == len(t.sequence):
                        t.done = True
                        if end_time is None:
                            end_time = t.time + 2
                    else:
                        t.turn_index += 1
                        t.begin_turn()
                        policies[t.actor].reset({})
                        retire_start = None
            elif t.time > 2:
                action = oracle.act({"joint_positions": robot.get_joint_positions()})
                rig = getattr(oracle.env.backend, "rig", None)
                if rig is not None:
                    rig.command = action.copy()
                    rig.apply()
                else:
                    robot.apply_action(ArticulationAction(joint_positions=action))
            for _ in range(c["sim"]["control_decimation"]):
                b.tick()
                collisions = (
                    b.collision_count,
                    b.black_collision_count,
                    b.inter_arm_collision_steps,
                )
                if any(
                    current > previous for current, previous in zip(collisions, previous_collisions)
                ):
                    t.violation("arm_collision")
                previous_collisions = collisions
                t.observe_duel(b.states, c["sim"]["physics_dt"])
                trace.write(
                    json.dumps(
                        serial(
                            dict(
                                time=t.time,
                                actor=t.actor,
                                phase=oracle.phase,
                                states=b.states,
                                joints={a: r.get_joint_positions() for a, r in robots.items()},
                            )
                        )
                    )
                    + "\n"
                )
            if t.done and not t.turn_complete:
                break
            if end_time is not None and t.time >= end_time:
                break
            if t.time - last_print >= 5:
                print(
                    "PROGRESS",
                    round(t.time, 2),
                    t.actor,
                    t.san,
                    oracle.phase,
                    "retire",
                    retire_start,
                    "placed",
                    t.placement_valid,
                    "events",
                    t.event_log[-2:],
                    flush=True,
                )
                last_print = t.time
            if encoder and t.time >= next_frame:
                cues()
                encoder.write(frame())
                video_frames += 1
                next_frame += 1 / VIDEO_FPS
        cues()
        Image.fromarray(frame()).save(args.output / "final.png")
    if encoder:
        encoder.close()
    r = t.result()
    r.update(
        **rig_setup,
        difficulty=args.difficulty,
        seed=args.seed,
        randomization=c["randomization"],
        video_frames=video_frames,
        source_hashes=hashes,
        renderer_settings=renderer_state(),
        arm_collision_count={"white": b.collision_count, "black": b.black_collision_count},
        inter_arm_collision_steps=b.inter_arm_collision_steps,
        simulator=simulator_label(),
        physics_hz=PHYSICS_HZ,
        control_hz=CONTROL_HZ,
        controller="privileged scripted state-feedback, native articulation drives",
        embodiment=c["embodiment"],
    )
    (args.output / "result.json").write_text(json.dumps(serial(r), indent=2) + "\n")
    print("RESULT", json.dumps(serial(r)), flush=True)
    b.close()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=140)
    p.add_argument("--video", action="store_true")
    p.add_argument("--presentation-only", action="store_true", help="Record only the presentation camera")
    p.add_argument("--embodiment", choices=["dual_arx_x5"],
                   default="dual_arx_x5")
    args = p.parse_args()
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        run(runtime, **vars(args))


if __name__ == "__main__":
    main()
