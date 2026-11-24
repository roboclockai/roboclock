"""Resuscitation-bay dressing: a patient lying on a bed, an AED with its pads, a crash cart, a bedhead wall.

Visual geometry only; the collision shapes (chest block, AED body, SHOCK button) come from the layout. Built
to look like a real hospital room from photographs: the patient lies with the head toward the wall, the chest
is bare where the pads go, the lower body is under a draped sheet, and the unit's buttons are color coded.
"""

import numpy as np

from roboclock.envs.hospital.scene_meshes import cloth_arrays, torso_arrays, tube_arrays

SKIN = (0.86, 0.66, 0.55)


TORSO_N = 3.2  # cross-section exponent: 2 is an ellipse, 6 a box; a rounded chest sits between


class EmergencyScenes:
    """Mixed into HospitalAppearance; uses its cube, rounded, cylinder, sphere, label and material helpers."""

    # -- small helpers ----------------------------------------------------------------------

    def tint(self, name, color, roughness=0.5):
        return self.material(name, color=color, roughness=roughness)

    def ellipsoid(self, path, center, radii, material, yaw=0.0):
        from pxr import Gf, UsdGeom, UsdShade

        shape = UsdGeom.Sphere.Define(self.stage, path)
        shape.CreateRadiusAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        if yaw:
            shape.AddRotateZOp().Set(float(np.degrees(yaw)))
        shape.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in radii]))
        visual = self.material(material) if isinstance(material, str) else material
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(visual.material)
        return shape

    def outline(self, path, center, size, thickness, material, z):
        """A flat rectangular outline lying on a horizontal surface (a decal ring)."""
        cx, cy = center
        sx, sy = size
        bars = (
            (0, sy / 2 - thickness / 2, sx, thickness),
            (0, -sy / 2 + thickness / 2, sx, thickness),
            (sx / 2 - thickness / 2, 0, thickness, sy - 2 * thickness),
            (-sx / 2 + thickness / 2, 0, thickness, sy - 2 * thickness),
        )
        for k, (dx, dy, w, h) in enumerate(bars):
            self.cube(f"{path}_{k}", [cx + dx, cy + dy, z], [w, h, 0.0012], material)

    def mesh(self, path, arrays, position, material, along_y=False, center_x=0.0, y0=0.0):
        """Author a generated mesh (cloth) into the stage with a material."""
        from pxr import UsdShade

        from roboclock.simulators.isaac.visual_geometry import author_mesh

        points, uv, indices, normals = arrays
        if (
            along_y
        ):  # the generator's length axis (x) becomes world y, its width axis (y) becomes world x
            points = np.stack([center_x + points[:, 1], y0 + points[:, 0], points[:, 2]], axis=1)
            normals = np.stack([normals[:, 1], normals[:, 0], normals[:, 2]], axis=1)
        mesh = author_mesh(self.stage, path, (points, uv, indices), position, normals)
        mesh.CreateDoubleSidedAttr(True)
        visual = self.material(material) if isinstance(material, str) else material
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual.material)
        return mesh

    # -- the patient ------------------------------------------------------------------------

    def patient(self, cx, cy, chest_top, mattress_top, seed=0, bed_model=False):
        """An adult lying on their back, head toward -y. The chest block is the layout's collision shape."""
        skin = self.tint("patient_skin", SKIN, 0.55)
        gown = self.tint("patient_gown", (0.58, 0.74, 0.86), 0.85)
        hair = self.tint("patient_hair", (0.20, 0.14, 0.10), 0.7)
        lip = self.tint("patient_lip", (0.72, 0.38, 0.38), 0.5)
        dark = self.tint("patient_dark", (0.10, 0.07, 0.06), 0.8)
        linen = self.tint("bed_linen", (0.93, 0.94, 0.95), 0.9)
        pad = self.tint("bed_mattress", (0.50, 0.58, 0.64), 0.6)
        base = "/World/Patient"
        # Mattress and pillow.
        if not bed_model:
            self.rounded(
                f"{base}/Mattress", [cx, 0.0, mattress_top - 0.03], [0.60, 1.16, 0.06], pad, 0.02
            )
        head_y = cy - 0.40
        self.rounded(
            f"{base}/Pillow",
            [cx, head_y - 0.01, mattress_top + 0.04],
            [0.40, 0.30, 0.08],
            linen,
            0.03,
        )
        # A rounded torso: its flat top is where the pads rest (the layout's invisible chest block).
        torso = torso_arrays(cx, cy - 0.26, cy + 0.27, chest_top, 0.20, 0.225, 0.185, n=TORSO_N)
        self.mesh(f"{base}/Torso", torso, (0.0, 0.0, 0.0), skin)
        # Surface detail, flush with the flat top so the pads still rest on it: collarbones and nipples.
        darker = self.tint("patient_skin_dark", (0.72, 0.50, 0.42), 0.6)
        for side, sign in (("L", -1), ("R", 1)):
            self.ellipsoid(
                f"{base}/Collarbone{side}",
                [cx + sign * 0.075, cy - 0.205, chest_top - 0.0015],
                [0.085, 0.011, 0.004],
                darker,
                yaw=-sign * 0.12,
            )
            self.ellipsoid(
                f"{base}/Nipple{side}",
                [cx + sign * 0.100, cy + 0.015, chest_top - 0.0005],
                [0.013, 0.013, 0.0035],
                darker,
            )
        self.ellipsoid(
            f"{base}/Sternum", [cx, cy - 0.02, chest_top - 0.0012], [0.012, 0.17, 0.003], darker
        )
        # The open gown lies over the rim of the chest on both sides, softening its edges.
        for side, sign in (("L", -1), ("R", 1)):
            self.rounded(
                f"{base}/GownFold{side}",
                [cx + sign * 0.198, cy - 0.01, chest_top - 0.012],
                [0.060, 0.47, 0.048],
                gown,
                0.022,
            )
        # Neck, head and face (face up, forehead toward -y).
        head_z = mattress_top + 0.08 + 0.085
        self.cylinder(
            f"{base}/Neck", [cx, cy - 0.285, mattress_top + 0.105], 0.05, 0.09, skin, axis="Y"
        )
        self.ellipsoid(f"{base}/Head", [cx, head_y, head_z], [0.078, 0.105, 0.092], skin)
        self.ellipsoid(
            f"{base}/Hair", [cx, head_y - 0.025, head_z - 0.004], [0.083, 0.100, 0.090], hair
        )
        self.ellipsoid(
            f"{base}/Jaw", [cx, head_y + 0.055, head_z - 0.02], [0.060, 0.060, 0.065], skin
        )
        top = head_z + 0.092
        self.ellipsoid(
            f"{base}/Nose", [cx, head_y + 0.014, top - 0.002], [0.011, 0.020, 0.014], skin
        )
        self.ellipsoid(
            f"{base}/Lips", [cx, head_y + 0.058, top - 0.016], [0.022, 0.007, 0.006], lip
        )
        for side, sign in (("L", -1), ("R", 1)):
            self.ellipsoid(
                f"{base}/Ear{side}",
                [cx + sign * 0.080, head_y + 0.005, head_z],
                [0.010, 0.020, 0.028],
                skin,
            )
            self.cube(
                f"{base}/Eye{side}",
                [cx + sign * 0.030, head_y - 0.020, top - 0.004],
                [0.022, 0.004, 0.0025],
                dark,
            )
            self.cube(
                f"{base}/Brow{side}",
                [cx + sign * 0.031, head_y - 0.036, top - 0.008],
                [0.028, 0.005, 0.003],
                hair,
            )
        # Shoulders, arms along the body, hands; short gown sleeves.
        for side, sign in (("L", -1), ("R", 1)):
            ax = cx + sign * 0.248
            self.ellipsoid(
                f"{base}/Shoulder{side}",
                [cx + sign * 0.205, cy - 0.215, mattress_top + 0.12],
                [0.055, 0.060, 0.065],
                skin,
            )
            self.cylinder(
                f"{base}/UpperArm{side}",
                [ax, cy - 0.02, mattress_top + 0.047],
                0.047,
                0.34,
                skin,
                axis="Y",
            )
            self.cylinder(
                f"{base}/Sleeve{side}",
                [ax, cy - 0.10, mattress_top + 0.050],
                0.052,
                0.17,
                gown,
                axis="Y",
            )
            self.cylinder(
                f"{base}/Forearm{side}",
                [ax + sign * 0.02, cy + 0.285, mattress_top + 0.038],
                0.038,
                0.30,
                skin,
                axis="Y",
            )
            self.ellipsoid(
                f"{base}/Hand{side}",
                [ax + sign * 0.03, cy + 0.475, mattress_top + 0.030],
                [0.040, 0.060, 0.022],
                skin,
            )
        for side, sign in (("L", -1), ("R", 1)):  # gown straps over the shoulders
            self.rounded(
                f"{base}/Strap{side}",
                [cx + sign * 0.19, cy - 0.215, mattress_top + 0.182],
                [0.09, 0.10, 0.018],
                gown,
                0.006,
            )
        # Gown flap over the abdomen, bare chest above it.
        self.rounded(
            f"{base}/GownFlap",
            [cx, cy + 0.205, chest_top + 0.002],
            [0.40, 0.085, 0.006],
            gown,
            0.003,
        )
        # Sheet over the hips and both legs to the feet, which stick out at the end of the bed.
        sheet_x0, sheet_x1, half = cy + 0.17, cy + 1.12, 0.33

        def surface(x, y):
            u = (x - sheet_x0) / (sheet_x1 - sheet_x0)  # 0 at the hips, 1 at the feet
            hips = 0.20 * np.exp(-((y / 0.21) ** 4))
            legs = np.clip(
                np.exp(-(((y - 0.10) / 0.07) ** 2)) + np.exp(-(((y + 0.10) / 0.07) ** 2)), 0.0, 1.0
            )
            blend = np.clip(u / 0.22, 0.0, 1.0)
            toes = (
                0.06
                * np.clip((u - 0.84) / 0.08, 0.0, 1.0)
                * (1.0 - np.clip((u - 0.96) / 0.04, 0.0, 1.0))
            )
            return mattress_top + 0.005 + (1 - blend) * hips + blend * (0.115 * legs + toes * legs)

        arrays = cloth_arrays(
            sheet_x0, sheet_x1, -half, half, surface, 0.30, folds=0.004, seed=seed, end_drop=0.025
        )
        self.mesh(
            f"{base}/Sheet", arrays, (0.0, 0.0, 0.0), linen, along_y=True, center_x=cx, y0=0.0
        )
        for side, sign in (("L", -1), ("R", 1)):  # bare feet sticking out of the sheet, toes up
            self.ellipsoid(
                f"{base}/Foot{side}",
                [cx + sign * 0.10, cy + 1.15, mattress_top + 0.075],
                [0.045, 0.105, 0.040],
                skin,
            )
        # Side rails on the bed, the +x side toward the room (the real bed has its own).
        if not bed_model:
            for k, y in enumerate((-0.30, 0.30)):
                self.cylinder(
                    f"{base}/RailPost_{k}",
                    [cx + 0.325, y, mattress_top + 0.07],
                    0.012,
                    0.16,
                    "stainless",
                )
            self.cylinder(
                f"{base}/Rail",
                [cx + 0.325, 0.0, mattress_top + 0.15],
                0.012,
                0.62,
                "stainless",
                axis="Y",
            )

    # -- the AED ----------------------------------------------------------------------------

    def aed_unit(self, center, top, marks, patient_x, patient_y, mattress_top, chest_top, side=-1.0):
        """A Philips-style AED: blue body, pad tray, color-coded buttons, and the chest marks.

        side: which way (y) the open case's lid stands; -1 is the original unit, +1 its mirror image
        (the B2 unit on the over-bed table keeps its lid away from the chest and the head camera).
        """
        cx, cy = center
        m = -float(side)  # multiplies every y offset of the original (lid at -y) unit
        # The mirrored unit's lid leans 18 mm further back: the hand taking the pad next to it
        # brushed the upright lid (B2 on the over-bed table).
        lid = 0.0 if m > 0 else 0.018
        base = "/World/AED"
        case = self.tint("aed_case", (0.07, 0.07, 0.08), 0.7)
        dark_blue = self.tint("aed_dark_blue", (0.05, 0.14, 0.34), 0.45)
        tray = self.tint("aed_tray", (0.20, 0.22, 0.25), 0.6)
        green = self.tint("aed_green", (0.10, 0.75, 0.30), 0.35)
        info = self.tint("aed_info", (0.15, 0.40, 0.90), 0.35)
        grip = self.tint("aed_grip", (0.12, 0.30, 0.62), 0.5)
        self.rounded(f"{base}/Case", [cx, cy, top - 0.066], [0.232, 0.272, 0.022], case, 0.006)
        self.rounded(
            f"{base}/Handle", [cx - 0.092, cy, top - 0.030], [0.034, 0.200, 0.062], grip, 0.010
        )
        self.rounded(
            f"{base}/Tray", [cx + 0.005, cy, top + 0.001], [0.130, 0.215, 0.002], tray, 0.001
        )
        # The open carrying case: a black rim around the unit and a lid standing up behind it with the instruction card.
        for k, (px, py, w, d) in enumerate(
            (
                (0.0, 0.140, 0.262, 0.012),
                (0.0, -0.140, 0.262, 0.012),
                (0.125, 0.0, 0.012, 0.268),
                (-0.125, 0.0, 0.012, 0.268),
            )
        ):
            self.rounded(
                f"{base}/CaseWall_{k}", [cx + px, cy + m * py, top - 0.045], [w, d, 0.060], case, 0.004
            )
        self.rounded(
            f"{base}/Lid", [cx, cy - m * (0.158 + lid), top + 0.080], [0.262, 0.014, 0.220], case, 0.008
        )
        self.rounded(
            f"{base}/Card",
            [cx, cy - m * (0.149 + lid), top + 0.085],
            [0.200, 0.004, 0.150],
            "cabinet_white",
            0.002,
        )
        if self.config["visuals"]["labels"]:
            for k, (text, dx) in enumerate(
                (("1 PULL", -0.062), ("2 PLACE", 0.0), ("3 PRESS", 0.062))
            ):
                text_mesh = self.label(
                    f"{base}/CardText_{k}",
                    text,
                    [cx + dx, cy - m * (0.1467 + lid), top + 0.085],
                    [0.060, 0.050],
                    (0.05, 0.14, 0.34),
                )
                if m < 0 and text_mesh is not None:  # read from the unit's side of the mirrored lid
                    text_mesh.AddRotateZOp().Set(180.0)
        self.cylinder(f"{base}/Power", [cx + 0.075, cy - m * 0.060, top + 0.003], 0.0125, 0.006, green)
        self.cylinder(f"{base}/Info", [cx + 0.075, cy - m * 0.012, top + 0.003], 0.0095, 0.006, info)
        self.rounded(
            f"{base}/Speaker",
            [cx + 0.075, cy + m * 0.075, top + 0.001],
            [0.020, 0.040, 0.002],
            dark_blue,
            0.001,
        )
        self.cube(f"{base}/Seam", [cx, cy, top + 0.0005], [0.2, 0.001, 0.001], dark_blue)
        if self.config["visuals"]["labels"]:
            name_mesh = self.label(
                f"{base}/Name",
                "HEARTSTART",
                [cx, cy + m * 0.1205, top - 0.032],
                [0.15, 0.030],
                (0.05, 0.14, 0.34),
            )
            if m < 0 and name_mesh is not None:
                name_mesh.AddRotateZOp().Set(180.0)
        # Cables from the unit's connector: over the case wall and the patient's arm, up the side of the torso
        # and across the chest to each mark. Every point stays outside the body and clear of the arm.
        yellow = self.tint("aed_connector", (0.95, 0.78, 0.10), 0.5)
        blue_cable = self.tint("aed_cable", (0.20, 0.42, 0.85), 0.5)
        port = np.array([cx + 0.030, cy - m * 0.105, top + 0.006])
        self.rounded(f"{base}/Port", [*port[:2], top + 0.004], [0.030, 0.022, 0.012], yellow, 0.003)
        zc, half_h = chest_top - 0.10, 0.10

        def width_at(y):
            return 0.225 + (0.185 - 0.225) * np.clip((y - (patient_y - 0.26)) / 0.53, 0.0, 1.0)

        def surface_point(y, t, offset=0.008):
            """A point on the near side of the torso at angle t (0 = middle of the side, pi/2 = the top), pushed out."""
            width = width_at(y)
            x = width * np.cos(t) ** (2 / TORSO_N)
            z = zc + half_h * np.sin(t) ** (2 / TORSO_N)
            gx = TORSO_N * (x / width) ** (TORSO_N - 1) / width
            gz = TORSO_N * (abs(z - zc) / half_h) ** (TORSO_N - 1) / half_h
            norm = np.hypot(gx, gz)
            return [patient_x - (x + gx / norm * offset), y, z + gz / norm * offset]

        for k, (mx, my, _) in enumerate(marks):
            y_c = my + 0.14  # where this cable climbs onto the torso
            # Arrive from outside the body, then follow the surface a centimeter out from it.
            climb = [surface_point(y_c, 0.01, 0.03)] + [
                surface_point(y_c, t, 0.011) for t in (0.01, 0.05, 0.2, 0.6, 1.0, 1.3, 1.5)
            ]
            control = [
                port,
                [port[0] + 0.07, port[1] - m * 0.01, top + 0.045],  # up and over the case wall
                [patient_x - 0.30, y_c + 0.03, mattress_top + 0.118],  # above the near arm
                *climb,
                [patient_x - 0.08, y_c - 0.04, chest_top + 0.005],
                [mx - 0.09, my + 0.02, chest_top + 0.005],
                [mx - 0.065, my + 0.01, chest_top + 0.005],
            ]
            self.mesh(
                f"{base}/Cable_{k}", tube_arrays(control, 0.0035, sides=8), (0, 0, 0), blue_cable
            )
        # Chest marks: orange for pad 1, blue for pad 2, matching the stripes on the pads.
        colors = ((1.00, 0.50, 0.10), (0.20, 0.45, 0.90))
        for k, (mx, my, _) in enumerate(marks):
            ring = self.tint(f"mark_{k + 1}", colors[k], 0.5)
            self.outline(
                f"/World/Marks/Mark_{k}", (mx, my), (0.108, 0.080), 0.005, ring, chest_top + 0.0008
            )

    # -- the crash cart -----------------------------------------------------------------------

    def crash_cart(self, x, y):
        """A red crash cart with labeled drawers, a defibrillator on top and a suction canister (front faces +y)."""
        base = "/World/CrashCart"
        red = self.tint("cart_red", (0.62, 0.06, 0.09), 0.45)
        cream = self.tint("cart_cream", (0.93, 0.90, 0.80), 0.6)
        rail = self.tint("cart_rail", (0.88, 0.88, 0.86), 0.4)
        front = y + 0.25
        self.rounded(f"{base}/Body", [x, y, 0.45], [0.62, 0.50, 0.80], red, 0.012)
        self.rounded(f"{base}/Top", [x, y, 0.875], [0.66, 0.54, 0.035], cream, 0.012)
        self.rounded(f"{base}/Plinth", [x, y, 0.045], [0.60, 0.48, 0.07], "stainless", 0.01)
        for k, dx in enumerate((-0.26, 0.26)):
            for j, dy in enumerate((-0.19, 0.19)):
                self.cylinder(
                    f"{base}/Wheel_{k}{j}", [x + dx, y + dy, 0.03], 0.03, 0.03, "rubber", axis="X"
                )
        heights = (0.115, 0.115, 0.115, 0.115, 0.220)
        labels = ("AIRWAY", "SUCTION", "DRUGS", "IV SUPPLIES", "BROSELOW BAGS")
        z = 0.82
        for k, (h, text) in enumerate(zip(heights, labels)):
            z -= h / 2
            self.rounded(
                f"{base}/Drawer_{k}", [x, front + 0.002, z], [0.575, 0.008, h - 0.012], red, 0.003
            )
            self.cube(
                f"{base}/Handle_{k}",
                [x, front + 0.012, z - h * 0.18],
                [0.20, 0.010, 0.010],
                "stainless",
            )
            if self.config["visuals"]["labels"]:
                self.label(
                    f"{base}/Label_{k}",
                    text,
                    [x, front + 0.0075, z + h * 0.18],
                    [0.20, 0.034],
                    (0.45, 0.08, 0.08),
                )
            z -= h / 2
        # Defibrillator monitor, binder and suction canister.
        self.rounded(
            f"{base}/Defib", [x - 0.10, y, 0.99], [0.30, 0.24, 0.20], "cabinet_white", 0.015
        )
        self.rounded(
            f"{base}/DefibScreen",
            [x - 0.10, y + 0.1215, 1.02],
            [0.13, 0.004, 0.09],
            "screen_dark",
            0.002,
        )
        self.cylinder(
            f"{base}/DefibKnob", [x - 0.03, y + 0.1215, 0.96], 0.020, 0.012, "rubber", axis="Y"
        )
        self.rounded(
            f"{base}/Binder", [x + 0.16, y, 0.904], [0.24, 0.30, 0.022], "sterile_blue", 0.004
        )
        self.cylinder(f"{base}/RailTube", [x + 0.34, y, 0.62], 0.014, 0.60, rail)
        self.cylinder(f"{base}/Canister", [x + 0.375, y + 0.06, 0.78], 0.07, 0.19, "glass")
        self.cylinder(
            f"{base}/CanisterLid", [x + 0.375, y + 0.06, 0.885], 0.075, 0.025, "sterile_blue"
        )
        self.cylinder(f"{base}/IVPole", [x + 0.24, y - 0.20, 1.30], 0.009, 0.85, "stainless")
        self.rounded(
            f"{base}/IVBag", [x + 0.24, y - 0.20, 1.55], [0.10, 0.03, 0.18], "iv_bag", 0.02
        )

    # -- the wall and the whole bay -------------------------------------------------------------

    def bedhead_wall(self, cx, wall_y):
        """A bedhead rail with gas outlets and a wall monitor showing the arrest, above the patient."""
        base = "/World/Bedhead"
        panel = self.tint("bedhead_panel", (0.86, 0.85, 0.80), 0.6)
        self.rounded(f"{base}/Rail", [cx, wall_y + 0.016, 1.32], [1.40, 0.032, 0.40], panel, 0.01)
        for k, dx in enumerate((-0.52, -0.40, -0.28, 0.28, 0.40, 0.52)):
            color = ("sterile_blue", "gas_green", "gas_yellow")[k % 3]
            self.cylinder(
                f"{base}/Outlet_{k}", [cx + dx, wall_y + 0.042, 1.30], 0.030, 0.020, color, axis="Y"
            )
        self.rounded(
            f"{base}/Arm",
            [cx + 0.30, wall_y + 0.12, 1.66],
            [0.025, 0.20, 0.025],
            "stainless",
            0.005,
        )
        self.rounded(
            f"{base}/MonitorFrame",
            [cx + 0.30, wall_y + 0.24, 1.66],
            [0.52, 0.04, 0.34],
            "rubber",
            0.01,
        )
        self.rounded(
            f"{base}/MonitorScreen",
            [cx + 0.30, wall_y + 0.262, 1.66],
            [0.48, 0.004, 0.30],
            "screen_dark",
            0.004,
        )
        if self.config["visuals"]["labels"]:
            self.label(
                f"{base}/MonitorText",
                "CODE BLUE",
                [cx + 0.30, wall_y + 0.265, 1.66],
                [0.46, 0.28],
                (0.9, 0.15, 0.15),
            )

    def dress_cardiac_arrest(self, layout):
        fixtures = {f.name: f for f in layout.fixtures}
        chest, aed = fixtures["chest"], fixtures["aed"]
        cx, cy = float(chest.position[0]), float(chest.position[1])
        chest_top = float(chest.position[2] + chest.size[2] / 2)
        mattress_top = chest_top - float(chest.size[2])
        from roboclock.envs.hospital.props import add_model, model_info, model_size

        info = model_info("hospital_bed_flat")
        bed_model = info is not None
        self.patient(cx, cy, chest_top, mattress_top, bed_model=bed_model)
        if bed_model:
            # The real hospital bed with its backrest laid flat: head toward the wall, mattress at the
            # patient's back. 2.2 m long, 1.49 m wide with its rails at native scale; it is slimmed to
            # leave the bench free and stretched so its mattress top meets the patient.
            native_top = info["top_z"] + 0.02
            add_model(
                self.stage,
                "hospital_bed_flat",
                "/World/Patient/Bed",
                (cx + 0.05, cy + 0.42, 0.0),
                scale=(0.95, 0.50, mattress_top / native_top),
                yaw=-1.5708,
            )
        marks = [(*map(float, layout.zones[f"mark_{k}"].center), chest_top) for k in range(2)]
        aed_info = layout.info.get("aed", {})
        self.aed_unit(
            aed.position[:2],
            float(aed.position[2] + aed.size[2] / 2),
            marks,
            cx,
            cy,
            mattress_top,
            chest_top,
            side=float(aed_info.get("side", -1.0)),
        )
        if "table" in aed_info:
            self.overbed_table(aed_info["table"])
        if layout.info.get("dial_stands"):
            self.dial_stands(layout)
        from roboclock.envs.hospital.props import add_model

        if model_size("crash_cart") is not None:  # the real cart; the built one is the fallback
            add_model(self.stage, "crash_cart", "/World/Cart/Model", (1.22, -0.42, 0.0), yaw=3.1416)
        else:
            self.crash_cart(1.22, -0.42)
        self.bedhead_wall(cx, -0.80)

    def dial_stands(self, layout):
        """A slim floor stand under each cue dial's housing and the beacon (they stand off the bench)."""
        frame = self.tint("dial_stand", (0.70, 0.72, 0.75), 0.35)
        for indicator in layout.indicators:
            x, y, z = (float(v) for v in indicator.position)
            if indicator.style == "beacon":
                bottom = z - 0.015  # the beacon's base disc (30 mm) is centered on its position
            else:
                bottom = z + 0.005 - 0.05  # the dial housing (0.10 m tall) is centered 5 mm above
            self.cylinder(f"/World/DialStands/{indicator.name}_Pole", [x, y, bottom / 2], 0.012, bottom, frame)
            self.cylinder(f"/World/DialStands/{indicator.name}_Base", [x, y, 0.008], 0.10, 0.016, frame)

    def overbed_table(self, table):
        """A hospital over-bed table across the patient's near hip: a laminate top on a column that
        rises from a low wheeled base slid under the bed, at the top's +y end on the robot's side."""
        (tx, ty), (sx, sy), top = table["center"], table["size"], float(table["top"])
        laminate = self.tint("overbed_laminate", (0.78, 0.72, 0.62), 0.55)
        frame = self.tint("overbed_frame", (0.70, 0.72, 0.75), 0.35)
        base = "/World/OverbedTable"
        self.rounded(f"{base}/Top", [tx, ty, top - 0.0125], [sx, sy, 0.025], laminate, 0.006)
        self.rounded(f"{base}/Edge", [tx, ty, top - 0.030], [sx - 0.02, sy - 0.02, 0.012], frame, 0.003)
        # Column at the +y end, just outside the bed's near edge; arm under the top.
        px, py = tx - sx / 2 + 0.018, ty + sy / 2 + 0.018
        self.rounded(f"{base}/Column", [px, py, (top - 0.03) / 2], [0.030, 0.030, top - 0.03], frame, 0.006)
        self.rounded(f"{base}/Arm", [px + 0.06, py - 0.008, top - 0.040], [0.15, 0.025, 0.016], frame, 0.004)
        self.rounded(f"{base}/Foot", [px + 0.14, py, 0.025], [0.32, 0.045, 0.03], frame, 0.006)
        for k, dx in enumerate((0.0, 0.28)):
            self.cylinder(f"{base}/Caster_{k}", [px + dx, py, 0.012], 0.012, 0.024, "rubber", axis="Y")

    # -- B3: fire ---------------------------------------------------------------------------

    def dress_fire(self, layout):
        """A steel dish that catches fire, and the red fire-blanket pouch on its plate with two black tabs.

        The flames and the spread blanket are hidden groups the task shows at ignition and when the fire is
        smothered; the pouch is the real model when it is installed (CC BY-NC) and a built red box otherwise.
        """
        from pxr import Gf, UsdGeom, UsdLux

        fire = layout.info["fire"]
        cx, cy = fire["center"]
        top, z = fire["dish_top"], layout.table_height
        px, py = fire["pouch"]
        char = self.tint("char", (0.05, 0.04, 0.035), 0.9)
        # Dish: a steel bowl with a charred bottom and a rim.
        self.cylinder("/World/Fire/Dish", [cx, cy, z + 0.0225], 0.088, 0.045, "stainless")
        self.cylinder("/World/Fire/Char", [cx, cy, top + 0.0012], 0.074, 0.003, char)
        # Wall plate with its sign, and an open red holder on the bench for the folded blanket (no pouch).
        self.cube("/World/Fire/Plate", [px, py, z + 0.17], [0.30, 0.02, 0.34], "cabinet_white")
        self.label(
            "/World/Fire/Sign",
            "FIRE|BLANKET",
            [px, py + 0.0105, z + 0.30],
            [0.20, 0.07],
            [0.85, 0.1, 0.1],
        )
        bx, by = fire["holder"]
        red = self.material("fire_red")
        self.rounded("/World/Fire/Holder", [bx, by, z + 0.011], [0.17, 0.095, 0.022], red, 0.006)
        for k, sx in enumerate((-0.083, 0.083)):  # the straps the black tabs are the ends of
            self.rounded(
                f"/World/Fire/Strap_{k}",
                [bx + sx, by - 0.035, z + 0.006],
                [0.012, 0.075, 0.004],
                "rubber",
                0.002,
            )
        # The fire alarm station behind the button: a red box with its label and a ring around the button.
        ax, ay, az = fire["alarm"]
        station = self.material("fire_red")
        if fire.get("alarm_station") == "+x":
            # Behind the button as seen from the robot (rig head camera), its sign facing the robot.
            # 25 mm clear of the key's far edge, so the pressing hand passes in front of the sign.
            self.rounded(
                "/World/Fire/AlarmStation", [ax + 0.0625, ay, z + 0.07], [0.025, 0.11, 0.14], station, 0.008
            )
            sign = self.label(
                "/World/Fire/AlarmLabel", "FIRE|ALARM", [ax + 0.049, ay, z + 0.09], [0.09, 0.07], [1.0, 1.0, 1.0]
            )
            if sign is not None:
                sign.AddRotateZOp().Set(-90.0)
        else:
            self.rounded(
                "/World/Fire/AlarmStation",
                [ax, ay - 0.045, z + 0.07],
                [0.11, 0.025, 0.14],
                station,
                0.008,
            )
            self.label(
                "/World/Fire/AlarmLabel",
                "FIRE|ALARM",
                [ax, ay - 0.0315, z + 0.09],
                [0.09, 0.07],
                [1.0, 1.0, 1.0],
            )
        self.cylinder("/World/Fire/AlarmRing", [ax, ay, az + 0.0008], 0.034, 0.002, "stainless")
        # Flames: tapered tongues leaning outward, a yellow core and an orange light; hidden until ignition.
        UsdGeom.Xform.Define(self.stage, "/World/Fire/Flames")
        red_f = self.material("flame_red", color=[1.0, 0.16, 0.02], roughness=0.5, emissive=True)
        orange = self.material(
            "flame_orange", color=[1.0, 0.42, 0.04], roughness=0.5, emissive=True
        )
        yellow = self.material(
            "flame_yellow", color=[1.0, 0.85, 0.22], roughness=0.5, emissive=True
        )
        rng = np.random.default_rng(3)
        tongues = []
        for k in range(9):  # outer ring, red, leaning out
            a = 2 * np.pi * k / 9 + 0.2
            tongues.append(
                (
                    0.050 * np.cos(a),
                    0.050 * np.sin(a),
                    0.020,
                    rng.uniform(0.10, 0.15),
                    0.22,
                    a,
                    red_f,
                )
            )
        for k in range(6):  # middle ring, orange
            a = 2 * np.pi * k / 6 + 0.6
            tongues.append(
                (
                    0.026 * np.cos(a),
                    0.026 * np.sin(a),
                    0.020,
                    rng.uniform(0.15, 0.21),
                    0.12,
                    a,
                    orange,
                )
            )
        tongues.append((0.0, 0.0, 0.026, 0.25, 0.0, 0.0, orange))
        tongues.append((0.004, 0.0, 0.015, 0.17, 0.0, 0.0, yellow))
        for k, (dx, dy, radius, height, lean, a, material) in enumerate(tongues):
            cone = UsdGeom.Cone.Define(self.stage, f"/World/Fire/Flames/Tongue_{k}")
            cone.CreateRadiusAttr(float(radius))
            cone.CreateHeightAttr(float(height))
            cone.CreateAxisAttr("Z")
            cone.AddTranslateOp().Set(Gf.Vec3d(cx + dx, cy + dy, top + 0.004 + height / 2))
            # Lean away from the middle: rotate about the horizontal axis perpendicular to the lean direction.
            cone.AddRotateXYZOp().Set(
                Gf.Vec3f(
                    float(-np.degrees(lean * np.sin(a))), float(np.degrees(lean * np.cos(a))), 0.0
                )
            )
            visual = material
            from pxr import UsdShade

            UsdShade.MaterialBindingAPI.Apply(cone.GetPrim()).Bind(visual.material)
        light = UsdLux.SphereLight.Define(self.stage, "/World/Fire/Flames/Glow")
        light.CreateIntensityAttr(30000.0)
        light.CreateRadiusAttr(0.04)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.45, 0.1))
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(cx, cy, top + 0.10))
        # The blanket once it covers the dish: a cloth draped over the dish and falling down its sides.
        UsdGeom.Xform.Define(self.stage, "/World/Fire/Cover")
        cloth = self.tint(
            "fire_blanket_cloth", (0.58, 0.50, 0.34), 0.95
        )  # tan glass-fiber, clearly darker than the bench

        def draped(x, y):
            r = np.hypot(x - cx, y - cy)
            tent = np.clip(
                (0.125 - r) / 0.035, 0.0, 1.0
            )  # 1 over the dish, falling steeply down its sides
            return z + 0.003 + (top + 0.050 - z) * tent**1.2

        arrays = cloth_arrays(
            cx - 0.14, cx + 0.14, cy - 0.14, cy + 0.14, draped, 0.03, folds=0.007, seed=2
        )
        self.mesh("/World/Fire/Cover/Cloth", arrays, (0.0, 0.0, 0.0), cloth)
        for path in ("/World/Fire/Flames", "/World/Fire/Cover"):
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).MakeInvisible()
        self.callouts = {
            **getattr(self, "callouts", {}),
            "flames": "/World/Fire/Flames",
            "blanket_cover": "/World/Fire/Cover",
        }

    # -- operating room: the patient on the bed and the surgical team ------------------------

    def surgical_patient(self, tx, ty, top):
        """A patient under a green drape on the table, head at +x: legs, abdomen with the field, chest, head."""
        skin = self.tint("patient_skin", SKIN, 0.55)
        cap = self.tint("or_cap", (0.20, 0.52, 0.48), 0.8)
        mask = self.tint("or_mask", (0.88, 0.92, 0.94), 0.7)
        base = "/World/OR/Patient"

        def surface(x, y):
            u = x - tx
            legs = np.clip(
                np.exp(-(((y - 0.095) / 0.075) ** 2)) + np.exp(-(((y + 0.095) / 0.075) ** 2)),
                0.0,
                1.0,
            )
            body = np.exp(-((y / 0.21) ** 4))
            to_legs = np.clip((-u - 0.15) / 0.20, 0.0, 1.0)
            to_chest = np.clip((u - 0.25) / 0.20, 0.0, 1.0)
            toes = (
                0.05
                * np.clip((-u - 0.82) / 0.05, 0.0, 1.0)
                * (1 - np.clip((-u - 0.92) / 0.05, 0.0, 1.0))
            )
            core = (1 - to_chest) * 0.17 * body + to_chest * 0.20 * body
            return top + 0.004 + (1 - to_legs) * core + to_legs * (0.115 * legs + toes * legs)

        arrays = cloth_arrays(
            tx - 0.95, tx + 0.60, -0.25, 0.25, surface, 0.30, folds=0.004, end_drop=0.03
        )
        self.mesh(f"{base}/Drape", arrays, (0.0, ty, 0.0), "drape_green")
        self.rounded(
            f"{base}/Pillow", [tx + 0.86, ty, top + 0.03], [0.32, 0.30, 0.06], "cabinet_white", 0.03
        )
        self.cylinder(f"{base}/Neck", [tx + 0.68, ty, top + 0.07], 0.05, 0.14, skin, axis="X")
        self.ellipsoid(f"{base}/Head", [tx + 0.83, ty, top + 0.11], [0.105, 0.085, 0.085], skin)
        self.ellipsoid(f"{base}/Cap", [tx + 0.84, ty, top + 0.135], [0.108, 0.090, 0.070], cap)
        self.rounded(f"{base}/Mask", [tx + 0.74, ty, top + 0.095], [0.06, 0.13, 0.07], mask, 0.02)

    def surgeon(self, base, x, y, toward, hands, gown_color=(0.30, 0.60, 0.78)):
        """A surgeon in scrubs, gown, cap, mask and gloves, leaning over the table with both hands on the field.

        toward is +1 when the table is at larger y than the surgeon; hands is the (x, y, z) between the gloves.
        """
        d = float(toward)
        skin = self.tint("patient_skin", SKIN, 0.55)
        scrubs = self.tint("scrubs_green", (0.18, 0.55, 0.50), 0.8)
        gown = self.tint(f"gown_{int(sum(gown_color) * 100)}", gown_color, 0.8)
        shoe = self.tint("clog_white", (0.92, 0.93, 0.93), 0.6)
        glove = self.tint("glove_cream", (0.88, 0.80, 0.68), 0.5)
        dark = self.tint("visor_dark", (0.06, 0.09, 0.16), 0.2)
        mask = self.tint("or_mask", (0.88, 0.92, 0.94), 0.7)
        for k, s in enumerate((-1, 1)):
            self.rounded(
                f"{base}/Shoe_{k}",
                [x + s * 0.09, y + d * 0.03, 0.04],
                [0.10, 0.26, 0.08],
                shoe,
                0.03,
            )
            self.cylinder(f"{base}/Leg_{k}", [x + s * 0.09, y, 0.50], 0.062, 0.84, scrubs)
        self.rounded(f"{base}/Hips", [x, y, 0.93], [0.38, 0.24, 0.20], scrubs, 0.06)
        self.rounded(f"{base}/Waist", [x, y + d * 0.03, 1.09], [0.37, 0.25, 0.18], gown, 0.07)
        self.rounded(f"{base}/Chest", [x, y + d * 0.07, 1.26], [0.43, 0.27, 0.22], gown, 0.08)
        self.rounded(f"{base}/Shoulders", [x, y + d * 0.10, 1.40], [0.47, 0.22, 0.10], gown, 0.05)
        hy = y + d * 0.17
        self.cylinder(f"{base}/Neck", [x, hy - d * 0.04, 1.47], 0.05, 0.08, skin)
        self.sphere(f"{base}/Head", [x, hy, 1.57], 0.095, skin)
        self.ellipsoid(f"{base}/Cap", [x, hy - d * 0.005, 1.61], [0.10, 0.105, 0.075], scrubs)
        self.rounded(f"{base}/Mask", [x, hy + d * 0.075, 1.54], [0.12, 0.05, 0.075], mask, 0.02)
        self.rounded(f"{base}/Visor", [x, hy + d * 0.09, 1.595], [0.13, 0.012, 0.04], dark, 0.004)
        hx, hy2, hz = hands
        for k, s in enumerate((-1, 1)):
            shoulder = [x + s * 0.22, y + d * 0.10, 1.40]
            elbow = [x + s * 0.28, y + d * 0.30, 1.22]
            hand = [hx + s * 0.055, hy2, hz]
            self.mesh(
                f"{base}/Arm_{k}",
                tube_arrays([shoulder, elbow, hand], 0.045, sides=10),
                (0, 0, 0),
                gown,
            )
            self.ellipsoid(f"{base}/Glove_{k}", hand, [0.040, 0.060, 0.030], glove)
