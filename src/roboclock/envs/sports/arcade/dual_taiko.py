"""Two independently grasped drumsticks, with native bimanual feedback control."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.sports.arcade.backend import ToolBackend
from roboclock.envs.sports.arcade.contacts import head_contacts, mole_head_activated
from roboclock.envs.sports.arcade.oracle import ToolOracle
from roboclock.envs.sports.arcade.refinement import bind
from roboclock.envs.sports.arcade.scenes import vertical_label
from roboclock.physics.contacts import GraspTracker
from roboclock.physics.geometry import rotation_matrix


class DualTaikoBackend(ToolBackend):
    def initialize_robot(self, task):
        self.left_grasps = GraspTracker(
            minimum_time=0.04, force_threshold=self.config["physics"]["contact_force"]
        )
        self.previous_left_tool = None
        rotation = Rotation.from_euler("y", 180, degrees=True).as_matrix()
        self.rig.gripper_effort = self.config["v2"].get("native_gripper_effort", 2.0)
        self.rig.initialize(
            {
                side: (np.array([0.30, y, 1.06]), rotation, 0.04)
                for side, y in (("left", 0.30), ("right", -0.30))
            },
            zero_arms=self.config["embodiment"].get("start_pose") == "zero",
        )
        self.world.reset()
        self.rig.apply()

    def customize_scene(self):
        super().customize_scene()
        from isaacsim.core.prims import RigidPrim

        p = np.asarray(self.layout.bodies["mallet_left"].position)
        for j, dx in enumerate((0.025, 0.090)):
            for side in (-1, 1):
                self.collider(
                    f"/World/LeftToolRack/Saddle{j}_{side}".replace("-", "n"),
                    "cube",
                    [p[0] + dx, p[1] + side * 0.030, p[2] - 0.022],
                    [0.016, 0.010, 0.052],
                    [0.11, 0.14, 0.18],
                )
            self.collider(
                f"/World/LeftToolRack/Support{j}",
                "cube",
                [p[0] + dx, p[1], p[2] - 0.032],
                [0.016, 0.024, 0.042],
                [0.11, 0.14, 0.18],
            )
        self.collider(
            "/World/LeftToolRack/Foot",
            "cube",
            [p[0] + 0.015, p[1], 0.757],
            [0.15, 0.075, 0.014],
            [0.08, 0.1, 0.13],
        )
        self.surface_paths.append("/World/LeftToolRack/Foot")
        for part in ("Shaft", "Head") if self.task.task_id == "32" else ("Shaft",):
            bind(self, "/World/Objects/mallet_left/" + part, self.appearance.material("clean_wood"))
        for part in ("GripSleeve", *(f"GripBand{i}" for i in range(7))):
            bind(self, "/World/Objects/mallet_left/" + part, self.appearance.material("rubber"))
        self.left_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=[self.objects["mallet_left"].prim_path],
                name="left_grasp_contacts",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=[
                    self.rig.fingers["left"] + [self.rig.tools["left"].prim_path]
                ],
                max_contact_count=8192,
            )
        )
        self.left_strikes = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=[c.prim_path for c in self.controls.values()],
                name="left_strike_contacts",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=[
                    [self.objects["mallet_left"].prim_path] for _ in self.controls
                ],
                max_contact_count=8192,
            )
        )
        if self.task.task_id == "34":
            self.left_hand_contacts = self.world.scene.add(
                RigidPrim(
                    prim_paths_expr=[body.prim_path for body in self.controls.values()],
                    name="left_hand_mole_contacts",
                    reset_xform_properties=False,
                    track_contact_forces=True,
                    contact_filter_prim_paths_expr=[
                        self.rig.fingers["left"] + [self.rig.tools["left"].prim_path]
                        for _ in self.controls
                    ],
                    max_contact_count=8192,
                )
            )
        if self.task.task_id != "32":
            return
        for side, z in (("L", 1.205), ("R", 1.135)):
            vertical_label(
                self,
                f"/World/Game/Lane{side}",
                side,
                self.screen_point([0.335, 0.319, z]),
                [0.025, 0.026],
            )
        for name, sp in self.layout.controls.items():
            label = "ROBOT LEFT" if sp.position[1] > 0 else "ROBOT RIGHT"
            vertical_label(
                self,
                f"/World/Drums/{name}/HandLabel",
                label,
                [sp.position[0], sp.position[1] - 0.112, sp.position[2] - 0.08],
                [0.15, 0.025],
            )

    def measure(self, dt):
        s = super().measure(dt)
        p, _ = self.rig.pose("left")
        velocity = (
            np.zeros(3)
            if self.previous_left_tool is None or dt == 0
            else (p - self.previous_left_tool) / dt
        )
        self.previous_left_tool = p.copy()
        obj = s["objects"]["mallet_left"]
        forces = self.left_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])[0]
        width = self.rig.width("left")
        spec = self.layout.bodies["mallet_left"]
        held, _ = self.left_grasps.update(
            "mallet_left",
            forces[:2],
            width,
            obj["position"],
            p,
            obj["velocity"],
            velocity,
            spec.radius,
            spec.height,
            dt,
        )
        obj.update(
            grasped=held,
            finger_contact_forces=forces[:2].copy(),
            gripper_width=width,
            robot_contact=bool(np.linalg.norm(forces) > self.config["physics"]["contact_force"]),
        )
        head = obj["position"] + rotation_matrix(obj["orientation"]) @ np.array(
            [0, 0, -self.head_offset]
        )
        s["tools"] = {
            "left": {"tool_position": p, "tool_head_position": head, "object": "mallet_left"},
            "right": {
                "tool_position": s["tool_position"],
                "tool_head_position": s["tool_head_position"],
                "object": "mallet",
            },
        }
        rows = {p.rsplit("/", 1)[-1]: i for i, p in enumerate(self.left_strikes.prim_paths)}
        normal, points, _, _, counts, starts = self.left_strikes.get_contact_force_data(
            dt=self.config["sim"]["physics_dt"]
        )
        hand_forces = (
            self.left_hand_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
            if self.task.task_id == "34"
            else None
        )
        hand_rows = (
            {
                path.rsplit("/", 1)[-1]: i
                for i, path in enumerate(self.left_hand_contacts.prim_paths)
            }
            if hand_forces is not None
            else {}
        )
        for name, c in s["controls"].items():
            row = rows[name]
            first, count = int(starts[row, 0]), int(counts[row, 0])
            left_force, left_points = head_contacts(
                points[first : first + count],
                normal[first : first + count],
                obj["position"],
                obj["orientation"],
                radius=self.head_radius,
                head_offset=self.head_offset,
                lower_only=False,
            )
            right_force = c["tool_force"]
            c["head_forces"] = {"left": left_force, "right": right_force}
            c["striking_hand"] = "left" if left_force > right_force else "right"
            c["tool_force"] = max(left_force, right_force)
            c["tool_contact"] = c["tool_force"] > 0.05
            c["contact"] |= left_force > 0.05
            c["activated"] = bool(
                (left_force > 0.05 and held)
                or (right_force > 0.05 and s["objects"]["mallet"]["grasped"])
            )
            if self.task.task_id == "34":
                # Decoy avoidance counts every tool/hand contact, including shaft
                # contact that correctly cannot score a valid head strike.
                any_left_tool = float(np.abs(normal[first : first + count]).sum())
                left_hand = bool(
                    np.any(
                        np.linalg.norm(hand_forces[hand_rows[name]], axis=-1)
                        > self.config["physics"]["contact_force"]
                    )
                )
                c["contact"] |= any_left_tool > 0.05 or left_hand
                c["hand_contact"] |= left_hand
                c["head_contact_points"] = {"left": left_points, "right": c["tool_contact_points"]}
                if c["striking_hand"] == "left":
                    c["tool_contact_points"] = left_points
                c["activated"] = mole_head_activated(
                    left_force, held, c["exposed"], c["travel"]
                ) or mole_head_activated(
                    right_force, s["objects"]["mallet"]["grasped"], c["exposed"], c["travel"]
                )
        return s

    def update_cues(self):
        super().update_cues()
        from .feedback import mole_score, taiko_instruction

        if self.task.task_id == "34":
            from pxr import UsdGeom

            text = mole_score(self.task)
            if text != getattr(self, "_mole_score", None):
                UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/Refined/MoleBrand")).MakeInvisible()
                for name, line, z in (("Title", text[0], .827), ("Count", text[1], .790)):
                    vertical_label(
                        self, "/World/Refined/MoleScore" + name, line,
                        [.62, -.237, z], [.32, .026],
                    )
                self._mole_score = text
            return
        if self.task.task_id != "32":
            return
        text = taiko_instruction(self.task)
        if text != getattr(self, "_dual_cue", None):
            vertical_label(
                self,
                "/World/Game/Instruction",
                text,
                self.screen_point([0.53, 0.319, 1.07]),
                [0.39, 0.036],
            )
            self._dual_cue = text


class _TaskView:
    def __init__(self, task, side):
        self.task = task
        self.ids = [
            i
            for i, e in enumerate(task.events)
            if e["control"] == ("pad_0" if side == "left" else "pad_1")
        ]
        self.events = [task.events[i] for i in self.ids]

    @property
    def correct(self):
        return {i for i, original in enumerate(self.ids) if original in self.task.correct}

    def __getattr__(self, name):
        return getattr(self.task, name)


class _ArmView:
    def __init__(self, backend, side):
        self.backend = backend
        self.active_side = side
        self.idle_target = [0.33, 0.26 if side == "left" else -0.26, 1.06]

    def state(self):
        s = self.backend.state()
        tool = s["tools"][self.active_side]
        return dict(
            s,
            tool_position=tool["tool_position"],
            tool_head_position=tool["tool_head_position"],
            objects={"mallet": s["objects"][tool["object"]]},
        )

    def __getattr__(self, name):
        return getattr(self.backend, name)


class DualToolOracle:
    def __init__(self, env):
        self.env = env
        self.arms = {
            side: ToolOracle(
                SimpleNamespace(
                    backend=_ArmView(env.backend, side),
                    task=_TaskView(env.task, side),
                    config=env.config,
                )
            )
            for side in ("left", "right")
        }
        self.phase = "approach/approach"

    def act(self, observation):
        for policy in self.arms.values():
            policy.act(observation)
        self.phase = "/".join(p.phase for p in self.arms.values())
        self.diagnostics = {side: p.diagnostics for side, p in self.arms.items()}
        return self.env.backend.rig.command.copy()
