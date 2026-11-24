"""Headless rig check of one hospital task: one oracle episode plus an interpenetration scan.

    PYTHONPATH=src OMNI_KIT_ACCEPT_EULA=YES python scripts/hospital/rig_probe.py \
        B4 hard 0 results/hospital/rig_probe/B4_hard_0.json [--every 5] [--override over.yaml]

Runs the evaluator's loop (oracle policy, --oracle) on one episode with sim.no_render (no cameras,
no render calls; --render keeps them) and writes the task metrics plus, sampled every N control
steps, every place where the robot visibly passes through the scene: surface samples of every
scene-02 ward part and of every other authored Gprim (furniture, devices, props) that fall inside
(a) the convex hull of an X5 link mesh of either arm or (b) a robot visual box (mount columns,
wheeled base, rig bridge/mast/head camera). Ward parts have no colliders, so the PhysX collision
count cannot see them; this scan can. It also reports moved furniture (ward_pose "move") that
intersects other furniture, the arms' extents, an end-effector trace, the contact pairs at each
collision onset and, per key object (task objects, keys, dials), how often an arm stands between the
head camera and it. Isaac Sim 6.0.1 hangs on a second reset: one episode per process.
"""

import argparse
import json
import re
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("task")
p.add_argument("difficulty")
p.add_argument("seed", type=int)
p.add_argument("out")
p.add_argument("--override", default=None)
p.add_argument("--render", action="store_true")
p.add_argument("--every", type=int, default=5)
p.add_argument("--margin", type=float, default=0.0)
p.add_argument("--no-run", action="store_true", help="only scan the reset state")
args = p.parse_args()
__import__("pathlib").Path(args.out).parent.mkdir(parents=True, exist_ok=True)


def deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


import yaml

from roboclock.envs.hospital.config import load_config

config = load_config()
if args.override:
    deep_merge(config, yaml.safe_load(open(args.override)))
config["sim"]["no_render"] = not args.render
cfg_path = args.out + ".cfg.yaml"
open(cfg_path, "w").write(yaml.safe_dump(config, sort_keys=False))

from roboclock.envs.registry import create_env, get_environment, import_symbol

spec = get_environment("hospital")
t0 = time.time()
with import_symbol(spec.runtime_factory)(headless=True, renderer="RaytracedLighting") as runtime:
    env = create_env(
        "hospital", args.task, args.difficulty, runtime=runtime, config_path=cfg_path, oracle=True
    )
    import trimesh
    from pxr import Usd, UsdGeom
    from scipy.spatial import ConvexHull

    from roboclock.policies.base import load_policy

    obs, _ = env.reset(seed=args.seed, options={"task_id": args.task, "difficulty": args.difficulty,
                                                "seed": args.seed, "inference_latency": 0.0})
    b = env.backend
    stage = b.stage
    t_reset = time.time() - t0
    xcache = UsdGeom.XformCache()

    def world(prim):
        return np.array(xcache.GetLocalToWorldTransform(prim), dtype=float)  # row-vector convention

    # --- robot sources: convex hulls of the X5 link meshes ----------------------------------------
    hulls = []  # (label, prim, local hull vertices, equations)
    for side, root in enumerate(("/World/ARXLeft", "/World/ARXRight")):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=float)
            if len(pts) < 4:
                continue
            try:
                h = ConvexHull(pts)
            except Exception:
                continue
            link = re.search(r"/(link\d+|base_link)", str(prim.GetPath()))
            hulls.append((f"arm{side}:{link.group(1) if link else 'x'}", prim, pts[h.vertices], h.equations))

    robot_vis = []
    targets = []  # (name, world sample points, lo, hi)
    ward_prim = stage.GetPrimAtPath("/World/BedsideWard")
    skip = ("/World/ARXLeft", "/World/ARXRight", "/World/defaultGroundPlane", "/World/Lighting",
            "/World/HeadCamera", "/World/Camera", "/World/Table", "/World/TableVisual", "/Render",
            "/OmniverseKit")
    ward_names = {}
    if ward_prim.IsValid():
        from roboclock.envs.hospital.bedside.ward import ward_path

        pose = (b.layout.info or {}).get("ward_pose") or {}
        meta = ward_path(config, pose.get("usd")).with_suffix(".json")
        for part in json.loads(meta.read_text())["parts"]:
            ward_names["/World/BedsideWard/" + part["prim"].split("/", 2)[2]] = part["source_object"]

    rng = np.random.default_rng(0)

    def sample_mesh(prim):
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=int)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=int)
        if len(pts) == 0 or len(counts) == 0:
            return None
        faces = []
        k = 0
        for c in counts:
            for j in range(1, c - 1):
                faces.append((idx[k], idx[k + j], idx[k + j + 1]))
            k += c
        m = world(prim)
        wp = pts @ m[:3, :3] + m[3, :3]
        if not faces:
            return wp
        tm = trimesh.Trimesh(wp, np.asarray(faces), process=False)
        area = float(tm.area)
        n = int(np.clip(area / 0.0002, 200, 30000))  # ~1 sample per 2 cm^2
        s, _ = trimesh.sample.sample_surface(tm, n, seed=0)
        return np.concatenate([s, wp[:: max(1, len(wp) // 5000)]])

    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def sample_box(prim):
        r = bb.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            return None
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        g = rng.uniform(lo, hi, size=(600, 3))
        for ax in range(3):  # put samples on the box faces
            g[200 * ax:200 * ax + 100, ax] = lo[ax]
            g[200 * ax + 100:200 * ax + 200, ax] = hi[ax]
        return g

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if any(path.startswith(s) for s in skip):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        robotish = path.startswith("/World/Rig/") or re.search(
            r"/(Robot|ARXMount|Mobile|c3_robot_)[^/]*|/World/C4/(Wheel|Hub|Column|Plinth|Collar)_|"
            r"/World/WardC1/(Pedestal|Collar|Base|Wheel)", path)
        if robotish:
            robot_vis.append(prim)
            continue
        if path.startswith("/World/BedsideWard"):
            src = next((ward_names[k] for k in ward_names if path == k or path.startswith(k + "/")), None)
            if src in ("Rectangle008",):
                continue
            name = f"ward:{src}"
        else:
            name = path
        pts = sample_mesh(prim) if prim.IsA(UsdGeom.Mesh) else sample_box(prim)
        if pts is None or not len(pts):
            continue
        lo, hi = pts.min(0), pts.max(0)
        if name.startswith("ward:") and (hi - lo).max() > 4.0:
            continue  # room shell slabs spanning the bay
        targets.append((name, pts, lo, hi))
    print(f"probe: {len(hulls)} link hulls, {len(robot_vis)} robot visuals, {len(targets)} targets", flush=True)

    vis_boxes = []
    for prim in robot_vis + [stage.GetPrimAtPath(f"/World/ARXMount{s}") for s in (0, 1)]:
        if not prim.IsValid():
            continue
        r = bb.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        vis_boxes.append((f"vis:{str(prim.GetPath()).rsplit('/', 1)[-1]}", np.array(r.GetMin()), np.array(r.GetMax())))

    tlo = np.array([t[2] for t in targets])
    thi = np.array([t[3] for t in targets])
    hits = {}

    def record(key, t, depth):
        h = hits.setdefault(key, {"first": t, "last": t, "depth": 0.0, "n": 0})
        h["last"] = t
        h["n"] += 1
        h["depth"] = max(h["depth"], depth)

    run_extent = {}

    def scan(t, with_vis):
        xcache.Clear()
        m = args.margin
        for name, prim, verts, eq in hulls:
            M = world(prim)
            wv = verts @ M[:3, :3] + M[3, :3]
            lo, hi = wv.min(0) - m, wv.max(0) + m
            arm = name.split(":")[0]
            elo, ehi = run_extent.get(arm, (np.full(3, 9.0), np.full(3, -9.0)))
            run_extent[arm] = (np.minimum(elo, wv.min(0)), np.maximum(ehi, wv.max(0)))
            cand = np.nonzero(np.all(thi > lo, 1) & np.all(tlo < hi, 1))[0]
            if not len(cand):
                continue
            inv = np.linalg.inv(M)
            for j in cand:
                tname, pts = targets[j][0], targets[j][1]
                sel = pts[np.all((pts > lo) & (pts < hi), 1)]
                if not len(sel):
                    continue
                local = sel @ inv[:3, :3] + inv[3, :3]
                d = local @ eq[:, :3].T + eq[:, 3]  # <= 0 inside each facet plane
                inside = np.all(d <= m, 1)
                if inside.any():
                    depth = float(np.max(-np.max(d[inside], 1)))
                    arm, link = name.split(":")
                    record((arm, link, tname), t, depth)
        if not with_vis:
            return
        for name, lo, hi in vis_boxes:
            cand = np.nonzero(np.all(thi > lo, 1) & np.all(tlo < hi, 1))[0]
            for j in cand:
                tname, pts = targets[j][0], targets[j][1]
                inside = np.all((pts > lo + 0.002) & (pts < hi - 0.002), 1)
                if inside.any():
                    q = pts[inside]
                    depth = float(np.max(np.min(np.minimum(q - lo, hi - q), 1)))
                    record((name, "", tname), t, depth)

    def arm_extent():
        xcache.Clear()
        res = {}
        for name, prim, verts, eq in hulls:
            M = world(prim)
            wv = verts @ M[:3, :3] + M[3, :3]
            arm = name.split(":")[0]
            lo, hi = res.get(arm, (np.full(3, 9.0), np.full(3, -9.0)))
            res[arm] = (np.minimum(lo, wv.min(0)), np.maximum(hi, wv.max(0)))
        return {k: [v[0].round(3).tolist(), v[1].round(3).tolist()] for k, v in res.items()}

    arm_reset = arm_extent()

    # Moved furniture vs everything else: shared 2 cm surface voxels (visible interpenetration).
    move_conflicts = {}
    moves = set(((b.layout.info or {}).get("ward_pose") or {}).get("move", {}))
    if moves:
        def vox(pts):
            return set(map(tuple, np.floor(pts / 0.02).astype(int)))
        moved = {}
        rest = {}
        for name, pts, lo, hi in targets:
            src = name[5:] if name.startswith("ward:") else None
            (moved if src in moves else rest).setdefault(name, []).append(pts)
        mv = vox(np.concatenate([np.concatenate(v) for v in moved.values()])) if moved else set()
        for name, chunks in rest.items():
            shared = mv & vox(np.concatenate(chunks))
            if len(shared) > 2:
                move_conflicts[name] = len(shared)
    # --- head-camera occlusion of the key objects by the arms --------------------------------------
    from roboclock.envs.hospital.rig import head_camera_pose

    head_pos, _, _ = head_camera_pose(config, args.task)
    key_names = [n for n in b.layout.objects] + [i.name for i in b.layout.indicators if i.style == "dial"]
    key_names += [f.name for f in b.layout.fixtures if f.name in ("aed", "shock_button", "alarm_button", "fire_dish")]
    occl = {n: {"samples": 0, "arm0": 0, "arm1": 0} for n in key_names}
    fixture_pos = {f.name: np.asarray(f.position, float) for f in b.layout.fixtures}
    dial_pos = {i.name: np.asarray(i.position, float) + [0, 0, 0.005 + 0.05 + 0.078] for i in b.layout.indicators}

    def key_point(name):
        st = b.state()
        if name in st.objects:
            return np.asarray(st.objects[name].position, float)
        return dial_pos.get(name, fixture_pos.get(name))

    def occlusion():
        xcache.Clear()
        mats = [(name.split(":")[0], np.linalg.inv(world(prim)), eq) for name, prim, verts, eq in hulls]
        for kname in key_names:
            p = key_point(kname)
            if p is None:
                continue
            occl[kname]["samples"] += 1
            v = p - head_pos
            blocked = set()
            for arm, inv, eq in mats:
                if arm in blocked:
                    continue
                c = head_pos @ inv[:3, :3] + inv[3, :3]
                dv = v @ inv[:3, :3]
                num = -(eq[:, :3] @ c + eq[:, 3])
                den = eq[:, :3] @ dv
                t0, t1 = 0.0, 0.97  # stop short of the object itself
                ok = True
                for n_, d_ in zip(num, den):
                    if abs(d_) < 1e-12:
                        if n_ < 0:
                            ok = False
                            break
                        continue
                    t = n_ / d_
                    if d_ > 0:
                        t1 = min(t1, t)
                    else:
                        t0 = max(t0, t)
                    if t0 > t1:
                        ok = False
                        break
                if ok:
                    blocked.add(arm)
            for arm in blocked:
                occl[kname][arm] += 1
                occl[kname].setdefault(arm + "_times", []).append(round(env.task.time, 2))

    occlusion()
    occl_reset = {k: dict(v) for k, v in occl.items()}
    scan(0.0, True)
    reset_hits = {"|".join(k): dict(v) for k, v in hits.items()}
    coll_at_reset = b.collision_count
    arm_filter = (["/World/Table"] + [body.prim_path for parts in b.target_parts.values() for body, _ in parts]
                  + [body.prim_path for body in b.fixture_bodies.values()]
                  + [p for p in ("/World/ARXMount0", "/World/ARXMount1") if stage.GetPrimAtPath(p).IsValid()]
                  + [body.prim_path for body in b.clutter.values()])
    coll_times = []
    ee_trace = []
    coll_detail = []
    if not args.no_run:
        policy = load_policy("oracle", env)
        policy.reset(obs)
        step = 0
        last = False
        while not env.done:
            action = policy.act(obs)
            obs, *_ = env.step(np.asarray(action, dtype=np.float32))
            step += 1
            if b.last_collision and not last:
                coll_times.append(round(env.task.time, 3))
                detail = {"t": round(env.task.time, 3)}
                dtp = config["sim"]["physics_dt"]
                for label, view in (("arm0", getattr(b, "arm_contacts", None)),
                                    ("arm1", getattr(b, "secondary_contacts", None)),
                                    ("inter", getattr(b, "inter_arm_contacts", None))):
                    if view is None:
                        continue
                    f = np.linalg.norm(view.get_contact_force_matrix(dt=dtp), axis=-1)
                    hit = np.argwhere(f > 0.5)
                    if len(hit):
                        detail[label] = [[int(i), int(j), round(float(f[i, j]), 2)] for i, j in hit[:6]]
                coll_detail.append(detail)
            last = b.last_collision
            if step % args.every == 0:
                scan(round(env.task.time, 3), False)
                occlusion()
            if step % 10 == 0:
                st = b.state()
                ee_trace.append([round(env.task.time, 2), *[round(float(v), 3) for v in st.ee_position],
                                 *[round(float(v), 3) for v in b._tool_pose(b.secondary_robot)[0]]])
    row = env.task.metrics()
    from roboclock.envs.lab.robot import mount_positions

    out = {
        "metrics": {k: v for k, v in row.items() if k != "events"},
        "collision_count_at_reset": coll_at_reset,
        "collision_onsets": coll_times[:40],
        "sim_time": env.task.time,
        "wall_reset_s": t_reset,
        "wall_total_s": time.time() - t0,
        "reset_overlaps": reset_hits,
        "overlaps": {"|".join(k): v for k, v in sorted(hits.items(), key=lambda kv: -kv[1]["depth"])},
        "events": env.task.event_log,
        "mounts": [list(map(float, m)) for m in mount_positions(config, args.task)],
        "ward_pose": (b.layout.info or {}).get("ward_pose"),
        "targets_aabb": {t[0] + f"#{i}": [t[2].round(3).tolist(), t[3].round(3).tolist()] for i, t in enumerate(targets)},
        "arm_extent_reset": arm_reset,
        "move_conflicts": move_conflicts,
        "ee_trace": ee_trace,
        "occlusion_reset": occl_reset,
        "occlusion": occl,
        "collision_detail": coll_detail[:20],
        "arm_filter_paths": arm_filter,
        "rows_arm0": [f"/World/ARXLeft/link{i}" for i in range(1, 7)],
        "arm_extent_end": arm_extent(),
        "arm_extent_run": {k: [v[0].round(3).tolist(), v[1].round(3).tolist()] for k, v in run_extent.items()},
        "robot_vis": {n: [lo.round(3).tolist(), hi.round(3).tolist()] for n, lo, hi in vis_boxes},
    }
    json.dump(out, open(args.out, "w"), indent=1, default=str)
    print("PROBE_DONE", json.dumps({k: row.get(k) for k in ("task_success", "temporal_success", "joint_success",
                                                             "completion_time", "collision_count")}), flush=True)
