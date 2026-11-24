"""Physical held tools and refined room geometry for the v2 demo tasks."""

import numpy as np

from roboclock.envs.sports.arcade.scenes import build_room, vertical_label
from roboclock.envs.sports.arcade.tasks import station_offset
from roboclock.envs.sports.common.backend import SportsBackend
from roboclock.envs.sports.common.equipment import annulus
from roboclock.physics.geometry import rotation_matrix


class ToolBackend(SportsBackend):
    def reset(self, task):
        super().reset(task)

    def make_body(self, spec):
        if spec.shape != "mallet":
            return super().make_body(spec)
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import PhysxSchema, UsdGeom

        path = f"/World/Objects/{spec.name}"
        UsdGeom.Xform.Define(self.stage, path)
        # A compact game tool: handle along local Z, striking end at -Z.
        # It rests horizontally in an open saddle and is grasped across the handle.
        self.head_offset = self.config["v2"]["tool_offset"]
        extension = self.head_offset - 0.125
        self.collider(
            path + "/Shaft",
            "cylinder",
            [0, 0, -0.035 - extension / 2],
            [0.011, 0.17 + extension],
            [0.60, 0.36, 0.15],
        )
        self.collider(
            path + "/GripSleeve",
            "cube",
            [0, 0, 0.014],
            self.config["v2"]["grip_sleeve_size"],
            [0.045, 0.065, 0.072],
        )
        if self.config["v2"].get("dual_sticks"):
            from isaacsim.core.api.materials import PhysicsMaterial
            from pxr import UsdShade

            rubber = PhysicsMaterial(
                path + "/GripPhysics",
                static_friction=2.0,
                dynamic_friction=1.5,
                restitution=0.0,
            )
            UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath(path + "/GripSleeve")).Bind(
                rubber.material, materialPurpose="physics"
            )
        if self.task.task_id == "34":
            self.collider(path + "/Pommel", "sphere", [0, 0, 0.048], [0.017], [0.08, 0.12, 0.13])
        self.head_radius = 0.032 if self.task.task_id == "34" else 0.016
        self.collider(
            path + "/Head",
            "sphere",
            [0, 0, -self.head_offset],
            [self.head_radius],
            [0.88, 0.34, 0.12],
        )
        for i in range(7):
            annulus(
                self,
                path + f"/GripBand{i}",
                [0, 0, 0.005 + i * 0.004],
                0.0109,
                0.0114,
                0.0015,
                [0.20, 0.14, 0.085],
            )
        body = self.world.scene.add(
            SingleRigidPrim(
                path,
                name=spec.name,
                position=np.array(spec.position),
                orientation=np.array([2**-0.5, 0, -(2**-0.5), 0]),
                mass=spec.mass,
            )
        )
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(body.prim)
        api.CreateEnableCCDAttr(True)
        return body

    def screen_point(self, point):
        """One transform for every display element, including animated notes."""
        return np.asarray(point) + np.asarray(self.config["v2"]["screen_offset"])

    def customize_scene(self):
        from isaacsim.core.prims import RigidPrim
        from pxr import UsdGeom

        a = build_room(self, "moles" if self.task.task_id == "34" else "taiko")
        self.appearance = a
        # Open horizontal saddle: support beneath the shaft; no clamp or attachment.
        p = np.array(self.layout.bodies["mallet"].position)
        for j, dx in enumerate((0.025, 0.090)):
            for side in (-1, 1):
                self.collider(
                    f"/World/ToolRack/Saddle{j}_{side}".replace("-", "n"),
                    "cube",
                    [p[0] + dx, p[1] + side * 0.030, p[2] - 0.022],
                    [0.016, 0.010, 0.052],
                    [0.11, 0.14, 0.18],
                )
            self.collider(
                f"/World/ToolRack/Support{j}",
                "cube",
                [p[0] + dx, p[1], p[2] - 0.032],
                [0.016, 0.024, 0.042],
                [0.11, 0.14, 0.18],
            )
        self.collider(
            "/World/ToolRack/Foot",
            "cube",
            [p[0] + 0.015, p[1], 0.757],
            [0.15, 0.075, 0.014],
            [0.08, 0.1, 0.13],
        )
        self.surface_paths.append("/World/ToolRack/Foot")
        if self.task.task_id == "32":
            for path in ("/World/PadInstructions", "/World/PadProgress", "/World/PadLED"):
                prim = self.stage.GetPrimAtPath(path)
                if prim:
                    UsdGeom.Imageable(prim).MakeInvisible()
            for name, sp in self.layout.controls.items():
                x, y, z = sp.position
                prim = self.stage.GetPrimAtPath("/World/Controls/" + name + "/Label")
                if prim:
                    UsdGeom.Imageable(prim).MakeInvisible()
                self.collider(
                    f"/World/Drums/{name}/Shell",
                    "cylinder",
                    [x, y, z - 0.079],
                    [0.110, 0.14],
                    [0.50, 0.075, 0.035],
                )
                self.surface_paths.append(f"/World/Drums/{name}/Shell")
                for dz in (-0.138, -0.025):
                    annulus(
                        self,
                        f"/World/Drums/{name}/Rim{int(abs(dz) * 1000)}",
                        [x, y, z + dz],
                        0.098,
                        0.105,
                        0.012,
                        [0.075, 0.045, 0.025],
                    )
                for j in range(12):
                    theta = j * np.pi / 6
                    self.collider(
                        f"/World/Drums/{name}/Stud{j}",
                        "sphere",
                        [x + 0.099 * np.cos(theta), y + 0.099 * np.sin(theta), z - 0.035],
                        [0.004],
                        [0.76, 0.60, 0.26],
                        False,
                    )
                # Drum skin and dark edge are attached to the spring-supported physical head.
                annulus(
                    self,
                    f"/World/Controls/{name}/SkinRim",
                    [0, 0, 0.0065],
                    0.087,
                    0.095,
                    0.001,
                    [0.76, 0.61, 0.40],
                )
                self.control_materials[name].set_color(np.array([0.91, 0.84, 0.69]))
            a.rounded(
                "/World/Game/ScreenHousing",
                self.screen_point([0.53, 0.36, 1.19]),
                [0.48, 0.05, 0.32],
                "navy",
                radius=0.018,
            )
            # Horizontal timeline near the drum, visible to the fixed task camera.
            a.rounded(
                "/World/Game/Timeline",
                self.screen_point([0.53, 0.331, 1.18]),
                [0.42, 0.008, 0.18],
                "navy",
                radius=0.006,
            )
            self.note_dots = []
            for i, e in enumerate(self.task.events):
                shape, mat = self.collider(
                    f"/World/Game/Note{i}",
                    "sphere",
                    self.screen_point([0.37, 0.32, 1.17]),
                    [0.012],
                    [0.9, 0.1, 0.1],
                    False,
                )
                self.note_dots.append((shape, mat))
            self.collider(
                "/World/Game/HitLine",
                "cube",
                self.screen_point([0.37, 0.32, 1.17]),
                [0.003, 0.002, 0.12],
                [0.97, 0.9, 0.65],
                False,
            )
            vertical_label(
                self,
                "/World/Game/Title",
                "TAIKO",
                self.screen_point([0.53, 0.327, 1.31]),
                [0.34, 0.045],
            )
        else:
            # Cabinet trim moves with the seeded station (``tasks.station_offset``).
            dx, dy, _ = station_offset(self.config)
            a.rounded(
                "/World/Game/Front",
                [0.62 + dx, -0.212 + dy, 0.80],
                [0.39, 0.04, 0.18],
                "navy",
                radius=0.012,
            )
            vertical_label(
                self,
                "/World/Game/FrontTitle",
                "MOLE CLUB",
                [0.62 + dx, -0.235 + dy, 0.81],
                [0.34, 0.055],
            )
            # CC0 wooden mallet visual; conservative compound shaft/head collision.
        from roboclock.envs.sports.arcade.refinement import finish_equipment

        finish_equipment(self, a)
        self.strike_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=[b.prim_path for b in self.controls.values()],
                name="tool_strike_contacts",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=[["/World/Objects/mallet"] for _ in self.controls],
                max_contact_count=8192,
            )
        )
        self._last_contact = {}
        self._cue_text = None

    def measure(self, dt):
        s = super().measure(dt)
        obj = s["objects"]["mallet"]
        head = obj["position"] + rotation_matrix(obj["orientation"]) @ np.array(
            [0, 0, -self.head_offset]
        )
        s["tool_head_position"] = head
        forces = self.strike_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        rows = {p.rsplit("/", 1)[-1]: i for i, p in enumerate(self.strike_contacts.prim_paths)}
        from roboclock.envs.sports.arcade.contacts import head_contacts, mole_head_activated

        normal_forces, points, _, _, counts, starts = self.strike_contacts.get_contact_force_data(
            dt=self.config["sim"]["physics_dt"]
        )
        for name, c in s["controls"].items():
            sp = self.layout.controls[name]
            force = float(np.linalg.norm(forces[rows[name], 0]))
            row = rows[name]
            first, count = int(starts[row, 0]), int(counts[row, 0])
            head_force, contact_points = head_contacts(
                points[first : first + count],
                normal_forces[first : first + count],
                obj["position"],
                obj["orientation"],
                radius=self.head_radius,
                head_offset=self.head_offset,
                lower_only=False,
            )
            c["hand_contact"] = c["contact"]
            c["tool_force"] = head_force
            c["tool_contact_points"] = contact_points
            c["tool_contact"] = bool(head_force > 0.05)
            c["contact"] = bool(force > 0.05 or c["hand_contact"])
            # Grasp validity is based on opposing finger forces, not an attachment.
            c["activated"] = bool(c["tool_contact"] and obj["grasped"] and c["exposed"])
            if sp.kind == "mole":
                c["activated"] = mole_head_activated(
                    head_force, obj["grasped"], c["exposed"], c["travel"]
                )
        return s

    def update_cues(self):
        if self.task.task_id == "34":
            return super().update_cues()
        from pxr import Gf, UsdGeom

        t = self.task
        for (shape, mat), e in zip(self.note_dots, t.events):
            dt = e["time"] - t.time
            visible = -0.5 <= dt <= 5
            im = UsdGeom.Imageable(shape.GetPrim())
            im.MakeVisible() if visible else im.MakeInvisible()
            if visible:
                op = shape.GetOrderedXformOps()[0]
                op.Set(
                    Gf.Vec3d(
                        *map(
                            float,
                            self.screen_point(
                                [
                                    0.37 + dt * 0.064,
                                    0.32,
                                    1.205 if e["control"] == "pad_0" else 1.135,
                                ]
                            ),
                        )
                    )
                )
                mat.set_color(
                    np.array([0.95, 0.22, 0.08] if e["control"] == "pad_0" else [0.08, 0.5, 0.95])
                )
        idx = t.current_index
        label = (
            "PICK UP STICK"
            if t.time < 10
            else "READY"
            if idx is None
            else ("LEFT" if t.events[idx]["control"] == "pad_0" else "RIGHT")
        )
        if label != self._cue_text:
            vertical_label(
                self,
                "/World/Game/Instruction",
                label,
                self.screen_point([0.53, 0.327, 1.075]),
                [0.34, 0.04],
            )
            self._cue_text = label
