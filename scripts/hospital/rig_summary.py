"""Write results/hospital/<data ID>/oracle_summary.json for a task on the common dual-arm rig.

Reads the per-episode `roboclock eval --policy oracle --oracle` outputs (one directory per episode,
each with episodes.jsonl: Isaac Sim 6.0.1 runs one episode per process) of a headless run
(sim.no_render) and of the rendered acceptance run, and reports them separately.

    python scripts/hospital/rig_summary.py --task B4 --data-id B1 \
        --headless results/hospital/rig_eval/headless --rendered results/hospital/rig_eval/rendered
"""

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Placement before the rig (benchmark.yaml robot.task_mount_positions; arm 0 drives single-arm
# tasks) and the layout changes the rig needed. Kept here so every summary states them.
RIG_NOTES = {
    "A1": {
        "former_mounts": [[0.0, 0.0, 1.05], [0.0, -0.38, 1.05]],
        "changes": [
            "bed (scene-02 KR_007-KR_012) moved 0.65 m away from the window (robot +y): the 0.77 m "
            "gap between window wall and bed head could not hold the 0.86 m wide base with the idle "
            "arm now on the bed side; the IV line to the pillow follows the bed",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
        ],
    },
    "A2": {
        "former_mounts": [[0.15, 0.0, 1.05], [0.15, -0.38, 1.05]],
        "changes": [
            "robot and infusion demonstrator moved 0.45 m along the bed (ward ROBOT_WARD y -0.55 -> "
            "-0.10): the two-tier supply trolley beside the bed stood where the idle arm now goes",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
        ],
    },
    "A3": {"former_mounts": [[0.0, 0.0, 1.05], [-0.02, -0.36, 1.05]], "changes": [
        "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
        "RoboDojo rig: STOP WINDOW dial from the trolley's far -y side (behind arm 0's upper arm in the head view the whole episode) to its +y end (0.55, 0.20); the level scale card faces the head camera",
    ]},
    "B1": {
        "former_mounts": [[0.0, 0.0, 1.05], [0.0, 0.38, 1.05]],
        "changes": [
            "rig 0.12 m toward the pumps: the SILENCE keys sat 0.50-0.56 m out, past the X5's "
            "~0.53 m top-down reach",
            "room IV stand prop moved along the curtain line, clear of the idle arm",
            "presses measured from the X5 fingertips (12.6 mm below the tool point)",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
        ],
    },
    "B2": {
        "former_mounts": [[0.15, 0.05, 1.05], [0.84, 0.49, 1.05]],
        "changes": [
            "AED moved from the bench edge under the head camera onto an over-bed table across the "
            "patient's near hip (top 1.06 m, clear of the sheet's hip fold), mirrored so its lid stands "
            "away from the chest; pads, tab and SHOCK key move with it",
            "RESPONSE and SHOCK READY dials moved from the bench edge behind the camera to pole stands "
            "beyond the bed's far side, turned to the head camera",
            "rig 0.12 m to +y and raised to 1.20 m (arm 0 at (0.17, 0.12)) for the raised AED; carry "
            "height over the AED and chest; the hand waits beside its base before the arrest instead of "
            "over the AED",
            "X5 link2/base_link self-collision filtered (the lab's filter); presses measured from the "
            "fingertips; a press aims off by the measured hover miss (bounded)",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
            "RoboDojo rig: code-blue beacon from 1.9 m over the bed head (behind the 60-degree head camera) onto a pole stand beside the dials, in view",
        ],
    },
    "B3": {
        "former_mounts": [[0.15, -0.10, 1.05], [0.0, 0.38, 1.05]],
        "changes": [
            "fire alarm moved from the bench edge under the head camera to the bench's +y side in front "
            "of it (its red station behind the key, sign toward the robot); ALARM and FIRE dials moved "
            "from behind the camera to the bench's far +y corner, turned to the head camera",
            "pouch plate 25 mm further back: the hand no longer bumps its collider pulling tab 0",
            "X5 link2/base_link self-collision filtered (the alarm sits ~60 degrees to arm 0's side, "
            "0.28 m below its mount); presses measured from the fingertips",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
            "RoboDojo rig: fire-alarm strobe from 1.9 m over the bench (behind the head camera) onto the bench behind the dish, in view",
        ],
    },
    "B4": {
        "former_mounts": [[0.0, 0.0, 1.05], [-0.17, 0.30, 1.05]],
        "changes": [
            "ultrasound cart moved 0.50 m along the aisle toward the bed head and 0.12 m back: it "
            "stood where the idle arm and the base now go",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
        ],
    },
    "B5": {
        "former_mounts": [[0.0, 0.0, 1.22], [0.0, -0.38, 1.22]],
        "changes": ["ultrasound cart moved 0.30 m along the bed, clear of the base and the raised idle arm", "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)", "RoboDojo rig: response dial from the cabinet front (behind the tucked arm 0 in the head view, 70 % of the episode) onto a floor stand at (0.32, 0.25) on the bag's +y side"],
    },
    "A4": {
        "former_mounts": [[0.0, 0.0, 1.50], [0.0, 0.42, 1.50]],
        "changes": ["supply box and spare bag kept on the prep cart's lower shelf (the box overhung it)", "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)"],
    },
    "B6": {
        "former_mounts": [[0.15, 0.0, 1.05], [0.0, 0.38, 1.05]],
        "changes": [
            "bed moved 0.35 m away from the robot (+y) and the ultrasound cart behind it 0.33 m: "
            "the idle arm and the base stood over the bed's near side",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
        ],
    },
    "C1": {
        "former_mounts": [[0.10, 0.20, 1.05], [0.10, -0.20, 1.05]],
        "changes": [
            "arm 0 is now the -y arm: each arm's stock riser and two RX bins follow its own mount "
            "(RX 1-4 now run -y to +y)",
            "shift board 0.10 m further out (y -0.70), out of arm 0's elbow sweep",
            "arm 1's ready pose no longer depends on a camera existing: under sim.no_render it "
            "started from HOME and finished 0.04 s earlier than rendered",
            "RoboDojo rig: shift board from y -0.70 (past the 60-degree head view's right edge) to the wall between the stock risers (0.67, 0), low and tilted 55 degrees under the wall unit, facing the head camera",
        ],
    },
    "C2": {
        "former_mounts": [[0.10, -0.20, 1.05], [0.10, 0.20, 1.05]],
        "changes": [
            "no layout move (arm 0 was already the -y arm); a hand with nothing left to do now holds "
            "its last commanded point instead of following its own measured tool point, which let "
            "arm 1 sag onto its own mount column 4 s after the last instrument (one collision)",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
            "RoboDojo rig: the sorted-count / pace display turns to the head camera (it stood edge-on to it)",
        ],
    },
    "C3": {
        "former_mounts": [[0.0, 0.20, 1.05], [0.0, -0.20, 1.05]],
        "changes": [
            "arm 0 is now the -y arm: trays, pockets and parks follow each arm's mount "
            "(ward_c3.MOUNTS removed; the stands and base are built from the rig)",
            "rig 0.04 m toward the cart so the inner pockets stay within 0.5 m of each arm",
            "the two-arm insert lock now covers every pocket (it covered the inner two): on the "
            "wider rig the forearms angle further inward and touched (link4-link4) when one hand "
            "worked an inner and the other an outer pocket of the same lane; ~0.3 s slower",
            "initial finger yaw matches the planner's first diagonal (no wrist turn before work)",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
            "RoboDojo rig: shift dial and count board off the display post at (0.32, -0.56) (past the head view's right edge): dial on a post at (0.82, -0.38), count board on its own post at (0.90, -0.76), both facing the head camera and above arm 0's home pose",
        ],
    },
    "C4": {
        "former_mounts": [[0.08, 0.13, 1.05], [0.08, -0.37, 1.05]],
        "changes": [
            "arm 0 is now the -y arm: it works the split's -y half (tray and rack wells), arm 1 the +y half",
            "RoboDojo rig (2026-11-04): every countdown dial turns to the head camera (they were built facing +x, the head saw their backs)",
            "RoboDojo rig: readout board from (0.48, -0.85) (past the head view's right edge) to the back of the counter behind the center rack (0.70, -0.12), scaled 0.62 to stay under the wall unit, facing the head camera",
        ],
    },
}


def episodes(roots):
    rows = []
    for root in roots:
        for path in sorted(Path(root).glob("*/episodes.jsonl")) + sorted(Path(root).glob("episodes.jsonl")):
            for line in path.read_text().splitlines():
                rows.append(json.loads(line))
    return rows


def limit_of(row):
    spec = row["episode_spec"]
    for key in ("response_limit", "cycle_time", "oracle_time"):  # oracle_time: ward C, score_version 2
        if spec.get(key):
            return key, float(spec[key])
    return None, None


def response_of(row):
    return float(row["completion_time"]) - float(row.get("event_time") or 0.0)


def limit_use(row):
    spec = row["episode_spec"]
    limits = spec.get("alarm_limits") or []
    if row.get("response_times") and limits:
        # Keyed by pump, not alarm order: bound with the slowest response over the tightest limit.
        return max(float(v) for v in row["response_times"].values()) / min(limits)
    return response_of(row) / float(spec["response_limit"])


def summarize(rows, task_id):
    out = {"levels": {}, "episodes": []}
    for level in ("easy", "medium", "hard"):
        mine = [r for r in rows if r["task_id"] == task_id and r["difficulty"] == level]
        if not mine:
            continue
        times = [r["completion_time"] for r in mine if r.get("completion_time") is not None]
        key, _ = limit_of(mine[0])
        limits = sorted({round(limit_of(r)[1], 3) for r in mine if limit_of(r)[1]})
        entry = {
            "episodes": len(mine),
            "physical_success": sum(bool(r["manipulation_success"]) for r in mine),
            "on_time": sum(bool(r["temporal_success"]) for r in mine),
            "joint_success": sum(bool(r["joint_success"]) for r in mine),
            "completion_s": [min(times), max(times)] if times else None,
            "collision_count_max": max(int(r["collision_count"]) for r in mine),
        }
        if key:
            entry[key + "_s"] = limits
        if key == "response_limit":
            # B tasks: the response runs from the event/instruction to the scored completion; B1 has
            # one limit per alarm (the last one 0.7x), so its use is the worst alarm's ratio.
            responses = [response_of(r) for r in mine if r.get("completion_time") is not None]
            entry["response_s"] = [round(min(responses), 3), round(max(responses), 3)] if responses else None
            uses = [limit_use(r) for r in mine if r.get("completion_time") is not None]
            entry["limit_use_max"] = round(max(uses), 3) if uses else None
        elif key == "cycle_time" and times and limits:
            entry["shift_use_max"] = round(max(t / limit_of(r)[1] for t, r in zip(times, mine)), 3)
        elif key == "oracle_time" and times and limits:
            # t_Q in units of t_o: <= 1 passes hard, <= 2 medium, <= 4 easy.
            done = [r for r in mine if r.get("completion_time") is not None]
            entry["t_q_over_t_o_max"] = round(max(r["completion_time"] / limit_of(r)[1] for r in done), 3)
        if task_id.startswith("A"):
            entry["window_s"] = sorted({tuple(round(v, 3) for v in r["episode_spec"]["windows"][0])
                                        for r in mine if r["episode_spec"].get("windows")})
        out["levels"][level] = entry
    for r in sorted(rows, key=lambda r: (r["difficulty"], r["seed"])):
        if r["task_id"] != task_id:
            continue
        out["episodes"].append({
            "difficulty": r["difficulty"],
            "seed": r["seed"],
            "physical": bool(r["manipulation_success"]),
            "on_time": bool(r["temporal_success"]),
            "joint": bool(r["joint_success"]),
            "completion_s": r.get("completion_time"),
            "event_s": r.get("event_time"),
            "limit_s": limit_of(r)[1],
            "collision_count": r["collision_count"],
            "physical_failures": r["physical_failures"],
            "temporal_violations": r["temporal_violations"],
        })
    out["seeds"] = sorted({e["seed"] for e in out["episodes"]})
    total = len(out["episodes"])
    out["physical_success_total"] = f"{sum(e['physical'] for e in out['episodes'])}/{total}"
    out["on_time_total"] = f"{sum(e['on_time'] for e in out['episodes'])}/{total}"
    out["joint_success_total"] = f"{sum(e['joint'] for e in out['episodes'])}/{total}"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--data-id", required=True)
    p.add_argument("--headless", nargs="+", type=Path, default=[])
    p.add_argument("--rendered", nargs="+", type=Path, default=[])
    p.add_argument("--no-action", nargs="*", type=Path, default=[], help="hold-policy runs (A tasks)")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    from roboclock.envs import robodojo_rig
    from roboclock.envs.hospital.config import load_config
    from roboclock.envs.hospital.rig import head_camera_pose, rig_mounts
    from roboclock.envs.hospital.tasks.registry import TASKS

    config = load_config()
    mounts, yaw = rig_mounts(config, a.task)
    head, look, pitch = head_camera_pose(config, a.task)
    name = TASKS[a.task].name
    folder = REPO / "results/hospital" / a.data_id
    videos = sorted(p.name for p in folder.glob(f"{a.data_id}_*.mp4"))
    summary = {
        "task": f"{a.data_id} {name}",
        "env_task_id": a.task,
        "data_id": a.data_id,
        "environment": "hospital",
        "simulator": "isaacsim 6.0.1",
        "policy": "privileged oracle: roboclock eval --policy oracle --oracle (OraclePickPlace)",
        "rig": {
            "spec": robodojo_rig.describe(),
            "arm_mounts": {"arm0_right": [round(float(v), 4) for v in mounts[0]],
                           "arm1_left": [round(float(v), 4) for v in mounts[1]]},
            "yaw_deg": math.degrees(yaw),
            "origin": [round(float(v), 4) for v in 0.5 * (mounts[0] + mounts[1])],
            "head_camera": {
                "position": [round(float(v), 4) for v in head],
                "look_at": [round(float(v), 4) for v in look],
                "pitch_deg_down": math.degrees(pitch),
            },
            "convention": "internal arm 0 (ARXLeft, the arm single-arm tasks drive) on the rig's "
                          "right (-y): right_wrist_rgb; arm 1 (ARXRight) on its left: left_wrist_rgb. "
                          "Actions/joint_positions list the rig's left arm (arm 1) first: values "
                          "0-7 = arm 1, 8-15 = arm 0 (since 2026-11-04)",
            "measured_rig_check": next((r.get("rig_check") for r in episodes(a.rendered + a.headless)
                                        if r.get("rig_check")), None),
            "renderer_readback": next((r.get("renderer_readback") for r in episodes(a.rendered)
                                       if r.get("renderer_readback")), None),
            "camera_intrinsics_reported": next((r.get("camera_intrinsics_reported") for r in episodes(a.rendered)
                                                if r.get("camera_intrinsics_reported")), None),
            "camera_intrinsics_rendered": next((r.get("camera_intrinsics_rendered") for r in episodes(a.rendered)
                                                if r.get("camera_intrinsics_rendered")), None),
        },
        "former_mounts": RIG_NOTES[a.task]["former_mounts"],
        "layout_changes": RIG_NOTES[a.task]["changes"],
        "seeds": sorted({r["seed"] for r in episodes(a.headless + a.rendered)}),
        "headless": {"mode": "sim.no_render: no cameras, no render calls", **summarize(episodes(a.headless), a.task)},
        "rendered": {"mode": "RTX RaytracedLighting + DLAA, head and both wrist cameras rendered "
                             "once per policy step (25 Hz)",
                     **summarize(episodes(a.rendered), a.task)},
        "video": videos[0] if videos else None,
        "policy_camera_videos": ["head.mp4", "left_wrist.mp4", "right_wrist.mp4"],
        "superseded": ["superseded_2026-11-03/ (before the common rig)",
                       "superseded_2026-11-03_rig_head50/ (common rig, 50-degree 960x720 head camera, "
                       "no wrist cameras)"],
    }
    if a.no_action:
        rows = [r for r in episodes(a.no_action) if r["task_id"] == a.task]
        summary["no_action_run"] = [
            {"policy": r.get("policy"), "difficulty": r["difficulty"], "seed": r["seed"],
             "physical": bool(r["manipulation_success"]),
             "on_time": bool(r["temporal_success"]), "temporal_violations": r["temporal_violations"],
             "physical_failures": r["physical_failures"]}
            for r in rows
        ]
    output = a.output or folder / "oracle_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output, summary["headless"]["physical_success_total"], summary["rendered"].get("physical_success_total"))


if __name__ == "__main__":
    main()
