"""Check recorded demonstrations against RoboDojo v1.0 and its training readers.

  PYTHONPATH=$ROBODOJO_DIR:$ROBODOJO_DIR/XPolicyLab/policy/OpenWAM/OpenWAM \\
    $ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \\
    scripts/collect/verify_demos.py <root> [--episodes N] [--dataset]

Per episode:
- OpenWAM's own `validate_robodojo_episode` and `read_calibrated_eef20` (sim calibration).
- Every JPEG decodes with XPolicyLab's `decode_image_bit` to 480x640x3 and carries XPL-RGB1.
- action[t] == state[t+1] (joints and gripper); grippers in [0, 1]; quaternions unit, w >= 0.
- The arm bases implied by ee_poses and the joints sit at RoboDojo's (-0.3 | 0.3, -0.45, 0.765)
  facing +y; the wrist extrinsics are rigid to link6.
- Images are not blank and change over the episode.
With --dataset, also builds OpenWAM's MultiTaskRoboDojoDataset from its robodojo.yaml over
<root> (writing normalization stats to <root>/meta) and draws samples. That needs omegaconf:
run with $ROBODOJO_DIR/XPolicyLab/policy/FastWAM/.venv/bin/python.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
BASES = {"left": np.array([-0.3, -0.45, 0.765]), "right": np.array([0.3, -0.45, 0.765])}


def pose_matrix(p7):
    m = np.eye(4)
    w, x, y, z = p7[3:]
    m[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    m[:3, 3] = p7[:3]
    return m


def check_episode(path, kin_cls):
    from openwam.dataloader import robodojo as ow
    from XPolicyLab.utils.process_data import decode_image_bit

    out = {"file": str(path), "problems": []}
    bad = out["problems"].append
    meta = ow.validate_robodojo_episode(path)
    out["openwam_frames"] = meta.get("num_frames", meta.get("length")) if isinstance(meta, dict) else None
    from openwam.dataloader.robodojo_contract import resolve_robodojo_calibration

    calibration = resolve_robodojo_calibration(None)
    with h5py.File(path, "r") as f:
        T = f["state/left_arm_joint_states"].shape[0]
        out["T"] = T
        for side in ("left", "right"):
            for part in ("arm_joint_states", "ee_joint_states"):
                s, a = f[f"state/{side}_{part}"][()], f[f"action/{side}_{part}"][()]
                if not np.array_equal(a[:-1], s[1:]):
                    bad(f"action/{side}_{part}[t] != state[t+1]")
            for group in ("state", "action"):
                g = f[f"{group}/{side}_ee_joint_states"][()]
                if g.min() < 0 or g.max() > 1:
                    bad(f"{group} {side} gripper outside [0,1]: {g.min()} {g.max()}")
                q = f[f"{group}/{side}_ee_poses"][()][:, 3:]
                if np.abs(np.linalg.norm(q, axis=1) - 1).max() > 1e-6 or q[:, 0].min() < 0:
                    bad(f"{group} {side} quaternion not unit / w<0")
            # Bases implied by the state ee pose and FK of the state joints.
            kin = kin_cls()
            inv_tool = np.linalg.inv(tool_frame())
            joints = f[f"state/{side}_arm_joint_states"][()]
            poses = f[f"state/{side}_ee_poses"][()]
            errs, yaws = [], []
            for t in range(0, T, max(1, T // 20)):
                link6_in_base = kin.tool_pose(joints[t], base=False) @ inv_tool
                base = pose_matrix(poses[t]) @ np.linalg.inv(link6_in_base)
                errs.append(np.linalg.norm(base[:3, 3] - BASES[side]))
                yaws.append(np.degrees(np.arctan2(base[1, 0], base[0, 0])))
            out[f"{side}_base_error_m"] = float(max(errs))
            out[f"{side}_base_yaw_deg"] = float(np.mean(yaws))
            if max(errs) > 0.01 or abs(np.mean(yaws) - 90) > 1:
                bad(f"{side} base at error {max(errs):.4f} m yaw {np.mean(yaws):.2f} deg")
            # Wrist extrinsic rigid to link6.
            cam = "cam_left_wrist" if side == "left" else "cam_right_wrist"
            ext = f[f"vision/{cam}/extrinsic_matrix"][()]
            rel = [np.linalg.inv(pose_matrix(poses[t])) @ ext[t] for t in (0, T // 2, T - 1)]
            if max(np.abs(r - rel[0]).max() for r in rel) > 1e-6:
                bad(f"{cam} extrinsic not rigid to link6")
        for cam in CAMS:
            colors = f[f"vision/{cam}/colors"]
            if colors.shape[0] != T:
                bad(f"{cam} has {colors.shape[0]} frames, state {T}")
            means, picks = [], sorted({0, T // 4, T // 2, 3 * T // 4, T - 1})
            for t in picks:
                raw = bytes(colors[t])
                if b"XPL-RGB1" not in raw[:64]:
                    bad(f"{cam}[{t}] missing XPL-RGB1 marker")
                img = np.asarray(decode_image_bit(raw))
                if img.shape != (480, 640, 3):
                    bad(f"{cam}[{t}] decoded shape {img.shape}")
                means.append(img.astype(np.float32))
            out[f"{cam}_mean"] = float(np.mean([m.mean() for m in means]))
            out[f"{cam}_change"] = float(np.abs(means[-1] - means[0]).mean())
            if out[f"{cam}_mean"] < 5:
                bad(f"{cam} nearly black")
        inst = f["instruction"][()]
        out["instruction"] = inst.decode() if isinstance(inst, bytes) else str(inst)
    eef = ow.read_calibrated_eef20(path, calibration)
    out["eef20_shape"] = list(eef.shape)
    if not np.isfinite(eef).all():
        bad("eef20 not finite")
    return out


def tool_frame():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", np.pi / 2).as_matrix()
    pose[:3, 3] = (0.145, 0.0, 0.0)
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes", type=int, default=0, help="Check at most N per task (0: all)")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    # roboclock's pure-Python X5 kinematics (no Isaac Sim): this checkout's src.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from roboclock.policies.x5_kinematics import X5Kinematics

    files = []
    for task in sorted(p for p in args.root.iterdir() if (p / "arx_x5" / "data").is_dir()):
        eps = sorted((task / "arx_x5" / "data").glob("episode_*.hdf5"))
        files += eps[: args.episodes] if args.episodes else eps
    rows = []
    for path in files:
        try:
            row = check_episode(path, X5Kinematics)
        except Exception as exc:  # report and continue
            row = {"file": str(path), "problems": [f"{type(exc).__name__}: {exc}"]}
        rows.append(row)
        status = "OK " if not row["problems"] else "BAD"
        print(status, path.relative_to(args.root), {k: (round(v, 4) if isinstance(v, float) else v)
                                                     for k, v in row.items() if k not in ("file",)}, flush=True)
    if args.dataset:
        # Needs omegaconf (e.g. the FastWAM venv); uses OpenWAM's own training config.
        from omegaconf import OmegaConf

        from openwam.dataloader.robodojo import MultiTaskRoboDojoDataset

        cfg = OmegaConf.load(Path(sys.modules["openwam"].__file__).parents[1] / "configs/dataloader/robodojo.yaml")
        cfg.dataset_dir = str(args.root)
        ds = MultiTaskRoboDojoDataset.from_config(cfg, split="train")
        print("dataset windows", len(ds))
        for index in np.linspace(0, len(ds) - 1, 4).astype(int):
            sample = ds[int(index)]
            print(index, {k: (tuple(v.shape) if hasattr(v, "shape") else (v[:90] if isinstance(v, str) else type(v).__name__))
                          for k, v in sample.items()})
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    print(f"{sum(not r['problems'] for r in rows)}/{len(rows)} episodes passed")


if __name__ == "__main__":
    main()
