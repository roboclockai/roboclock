"""Replay RoboDojo demonstrations through the roboclock XPolicyLab client and score the chunks.

Integration check for a RoboDojo-trained policy (such as pi0.5 RoboDojo-sim-arx_x5-joint-0)
that needs no simulator. It reads RoboDojo episodes from the Hugging Face dataset
RoboDojo-Benchmark/RoboDojo (`data/RoboDojo/<task>/arx_x5/data/episode_*.hdf5`) and sends
every `--every`-th frame to the policy server as three requests:

- native: what RoboDojo's own eval client sends, i.e. the stored JPEG bytes and the stored
  joint and gripper state, unchanged.
- native again: the same request a second time. Its difference from the first measures the
  policy's own sampling noise.
- roboclock: the frame as a roboclock observation (decoded RGB images, 16 joint positions with
  each gripper as finger-slider positions), packed by `XPolicyLabPolicy.observation()`. Every
  roboclock evaluation sends this path.

Each predicted chunk is compared with the demonstrated actions that follow the frame. For
reference, the script also scores holding the current state. If the client converts
correctly, the roboclock error matches the native error within the sampling noise, and both
are well below hold.

Needs h5py and XPolicyLab's image decoder, so run it with the policy server's venv:

  PYTHONPATH=src:$ROBODOJO_DIR XPOLICYLAB_URL=ws://127.0.0.1:6011 \\
    $ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \\
    scripts/xpolicylab/replay_check.py episode_0000000.hdf5 ... --json replay.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from roboclock.envs.lab.policies import xpolicylab as xpl

ARM = np.r_[0:6, 7:13]  # arm joints of a packed 14-value RoboDojo vector
GRIPPER = np.array([6, 13])


def load_episode(path):
    import h5py
    from XPolicyLab.utils.process_data import decode_image_bit

    def packed(group, f):
        parts = [f[f"{group}/{side}_{part}_joint_states"][()] for side in xpl.SIDES for part in ("arm", "ee")]
        return np.concatenate(parts, axis=1).astype(np.float32)

    parts = Path(path).resolve().parts
    # Dataset layout <task>/arx_x5/data/episode_*.hdf5; otherwise name it by its folder.
    task = parts[parts.index("arx_x5") - 1] if "arx_x5" in parts else parts[-2]
    with h5py.File(path, "r") as f:
        instruction = f["instruction"][()]
        return {
            "name": f"{task}/{Path(path).stem}",
            "instruction": instruction.decode() if isinstance(instruction, bytes) else str(instruction),
            "hz": int(f["additional_info/frequency"][()]),
            "state": packed("state", f),
            "action": packed("action", f),
            "jpeg": {camera: [bytes(b) for b in f[f"vision/{camera}/colors"]] for camera, _ in xpl.CAMERAS},
            "decode": decode_image_bit,
        }


def native_request(episode, t):
    """RoboDojo's eval-client observation: stored JPEG bytes and stored state."""
    s = episode["state"][t]
    state = {}
    for k, side in enumerate(xpl.SIDES):
        state[f"{side}_arm_joint_state"] = s[7 * k : 7 * k + 6]
        state[f"{side}_ee_joint_state"] = s[7 * k + 6 : 7 * k + 7]
    return {
        "instruction": episode["instruction"],
        "env_idx": 0,
        "vision": {camera: {"color": episode["jpeg"][camera][t]} for camera, _ in xpl.CAMERAS},
        "state": state,
    }


def roboclock_observation(episode, t):
    """The frame as a roboclock environment observation (interface arm order, finger sliders)."""
    s = episode["state"][t]
    joints = np.zeros(16, dtype=np.float32)
    low, high = xpl.GRIPPER_SCALE
    for k, lab_arm in enumerate(xpl.ARM_ORDER):
        joints[8 * lab_arm : 8 * lab_arm + 6] = s[7 * k : 7 * k + 6]
        joints[8 * lab_arm + 6 : 8 * lab_arm + 8] = low + s[7 * k + 6] * (high - low)
    observation = {key: episode["decode"](episode["jpeg"][camera][t]) for camera, key in xpl.CAMERAS}
    observation.update(
        joint_positions=joints,
        instruction=episode["instruction"],
        sim_time=np.array([t / episode["hz"]], dtype=np.float32),
    )
    return observation


def chunk(reply):
    return np.stack([xpl.robodojo_vector(action) for action in reply["actions"]])


def errors(prediction, truth):
    n = min(len(prediction), len(truth))
    return (
        float(np.abs(prediction[:n, ARM] - truth[:n, ARM]).mean()),
        float(np.abs(prediction[:n, GRIPPER] - truth[:n, GRIPPER]).mean()),
    )


def check_episode(policy, episode, every, horizon):
    rows = []
    frames = range(0, len(episode["state"]) - horizon, every)
    for step, t in enumerate(frames):
        truth = episode["action"][t : t + horizon]
        native = chunk(policy.connection.request("infer", {"observation": native_request(episode, t)}, trial_id="native", step=step))
        again = chunk(policy.connection.request("infer", {"observation": native_request(episode, t)}, trial_id="native", step=step))
        observation = roboclock_observation(episode, t)
        if step == 0:
            policy.reset(observation)
        # The client reports the last commanded opening; RoboDojo's stored state is the same.
        policy.gripper = episode["state"][t][GRIPPER].copy()
        ours = chunk(
            policy.connection.request(
                "infer", {"observation": policy.observation(observation)}, trial_id=policy.trial_id, step=step
            )
        )
        hold = np.repeat(episode["state"][t][None], len(truth), axis=0)
        rows.append(
            {
                "frame": t,
                "native": errors(native, truth),
                "native_again_vs_native": errors(again, native[: len(truth)]),
                "roboclock": errors(ours, truth),
                "roboclock_vs_native": errors(ours, native[: len(truth)]),
                "hold": errors(hold, truth),
            }
        )
    return rows


def summary(rows):
    keys = ("native", "native_again_vs_native", "roboclock", "roboclock_vs_native", "hold")
    return {key: [float(np.mean([r[key][i] for r in rows])) for i in (0, 1)] for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", nargs="+", type=Path, help="RoboDojo episode_*.hdf5 files")
    parser.add_argument("--every", type=int, default=25, help="Frame stride (25 = one per second)")
    parser.add_argument("--horizon", type=int, default=25, help="Chunk entries scored (25 = 1 s)")
    parser.add_argument("--json", type=Path, help="Write per-frame errors here")
    args = parser.parse_args()

    policy = xpl.make()
    report = {"episodes": {}}
    print("Mean absolute error over the scored chunk: arm joints (rad) / gripper opening (0-1)")
    print(f"{'episode':<34} {'frames':>6} {'native':>15} {'native noise':>15} {'roboclock':>15} {'roboclock-native':>15} {'hold':>15}")
    for path in args.episodes:
        episode = load_episode(path)
        rows = check_episode(policy, episode, args.every, args.horizon)
        means = summary(rows)
        report["episodes"][episode["name"]] = {"instruction": episode["instruction"], "frames": rows, "mean": means}
        cells = " ".join(f"{means[k][0]:7.3f}/{means[k][1]:.3f}" for k in means)
        print(f"{episode['name']:<34} {len(rows):>6} {cells}")
    all_rows = [r for e in report["episodes"].values() for r in e["frames"]]
    report["mean"] = summary(all_rows)
    cells = " ".join(f"{report['mean'][k][0]:7.3f}/{report['mean'][k][1]:.3f}" for k in report["mean"])
    print(f"{'all':<34} {len(all_rows):>6} {cells}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
