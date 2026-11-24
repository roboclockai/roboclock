"""Visible-event routing between two physically grasped mallets.

The oracle uses privileged measured state. Routing never reads a future mole's
identity or color; a new target is assigned only after its measured rise event.
Until then both arms stay where the pickup left them, not over the board.
"""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .dual_taiko import _ArmView
from .oracle import ToolOracle, pitch_matrix


class _MoleTaskView:
    def __init__(self, router, side):
        self.router, self.side = router, side

    @property
    def current_index(self):
        task = self.router.env.task
        index = task.current_index
        return index if self.router.assignments.get(index) == self.side else None

    def __getattr__(self, name):
        return getattr(self.router.env.task, name)


class DualMoleOracle:
    def __init__(self, env):
        self.env = env
        self.assignments, self.route_scores = {}, {}
        self.arms = {}
        for side in ("left", "right"):
            view = _ArmView(env.backend, side)
            # Between moles each arm holds the pose its pickup ended in (lifted above its
            # mallet's rest): it leaves that pose only for a mole whose rise was measured.
            # ``mole_idle_target`` ([x, |y|, z]) names another waiting pose.
            idle = env.config["v2"].get("mole_idle_target")
            view.idle_target = idle and [
                idle[0], idle[1] if side == "left" else -idle[1], idle[2]
            ]
            self.arms[side] = ToolOracle(
                SimpleNamespace(
                    backend=view,
                    task=_MoleTaskView(self, side),
                    config=env.config,
                )
            )
        self.phase = "prepare"

    def route(self, index):
        b, t = self.env.backend, self.env.task
        sp = t.layout.controls[t.events[index]["control"]]
        contact = np.array(
            [
                *sp.position[:2],
                sp.position[2]
                + sp.height / 2
                + b.head_radius
                - self.env.config["v2"].get("mole_contact_depth", 0.005),
            ]
        )
        scores = {}
        for side, policy in self.arms.items():
            object_name = "mallet_left" if side == "left" else "mallet"
            held = b.state()["objects"][object_name]["grasped"]
            if policy.offset is None or not policy.had_grasp or not held:
                scores[side] = dict(score=1e6, available=False)
                continue
            kin = b.rig.kin[side]
            seed = b.rig.robot.get_joint_positions()[b.rig.arm_ids[side]]
            hit = self.env.config["v2"].get("mole_hit_angle", 22)
            ready = self.env.config["v2"].get("mole_ready_angle", -25)
            wrist = contact - pitch_matrix(np.deg2rad(hit)) @ policy.offset
            errors = []
            for angle in (ready, hit):
                q, error = kin.solve(
                    wrist,
                    Rotation.from_euler("y", 180 + angle, degrees=True).as_matrix(),
                    seed=seed,
                )
                errors.append(error)
            ee, _ = b.rig.pose(side)
            distance = float(np.linalg.norm(wrist - ee))
            scores[side] = dict(
                score=20 * max(errors) + distance,
                available=True,
                ready_ik_error=errors[0],
                strike_ik_error=errors[1],
                distance=distance,
            )
        side = min(scores, key=lambda s: scores[s]["score"])
        self.assignments[index] = side
        self.route_scores[index] = scores
        t.event("mallet_assigned", index=index, side=side, candidates=scores)

    def act(self, observation):
        t = self.env.task
        index = t.current_index
        if (
            index is not None
            and index in t.mole_up_times
            and t.events[index]["valid"]
            and index not in self.assignments
        ):
            self.route(index)
        for policy in self.arms.values():
            policy.act(observation)
        self.phase = "/".join(p.phase for p in self.arms.values())
        self.diagnostics = {side: p.diagnostics for side, p in self.arms.items()}
        self.diagnostics["assignment"] = self.assignments.get(index)
        return self.env.backend.rig.command.copy()
