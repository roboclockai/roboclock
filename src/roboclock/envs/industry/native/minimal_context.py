"""Minimal visible context, separate from task instructions and evaluator state.

Only authored nonphysical label meshes are affected. Original fixtures, products,
actuators, contacts and task logic are untouched. Imported product markings remain.

Merged from three recorded variants: the Task45/49 version (machine-signal lamps
on reused display quads) plus the Task47 two-kit labels. That version's per-index
Task48 slot relabelling was dropped: it was never recorded and would relabel the
throughput scene's B-01 shelf sign as A-01. The Task44 spill-limit label is the
resolved limit, which equals the static "MAX 2.0 L" for the reviewed episode.
"""

import json
from pathlib import Path

# Static equipment/container identifiers, never next-action instructions.
LABELS = {
    41: {"Bath0": "ZINCATE DIP", "Bath1": "RINSE"},
    42: {"Supply": "INFEED", "Output": "OUTFEED"},
    43: {"MixUnit": "ADHESIVE", "Input": "COVERS", "Output": "OUTPUT"},
    44: {"Valve": "ISOLATION", "SpillLimit": "MAX 2.0 L"},
    45: {"StopText": "STOP"},
    46: {"Rework": "REWORK", "Batch": "COLLECTION"},
    47: {
        "Feed_bearingLabel": "BEARING",
        "Feed_bracketLabel": "BRACKET",
        "Feed_pinLabel": "PIN",
        "KitID": "KIT",
        # Two-kit layout (recorded Task47 scene).
        "Feed_0_bearingLabel": "BEARING",
        "Feed_0_bracketLabel": "BRACKET",
        "Feed_0_pinLabel": "PIN",
        "Feed_1_bearingLabel": "BEARING",
        "Feed_1_bracketLabel": "BRACKET",
        "Feed_1_pinLabel": "PIN",
        "Kit0ID": "KIT 1",
        "Kit1ID": "KIT 2",
    },
    48: {"Slot0": "A-01", "Slot1": "B-01", "MagazineID": "INFEED"},
    49: {f"{label}_{side}Back": label for label in ("PASS", "REJECT") for side in ("left", "right")},
    50: {
        "Feed_Color0Label": "CHAMPAGNE",
        "Feed_Color1Label": "ROSE",
        "Feed_Color2Label": "COPPER",
        "Feed_Color3Label": "DEEP BROWN",
        "leftOutputLabel": "OUTPUT",
        "rightOutputLabel": "OUTPUT",
    },
}


class MinimalContext:
    """Hide instruction/phase panels; retain necessary physical process cues.

    Tasks45/49 reuse existing nonphysical status-screen quads as small machine signals.
    It is dark outside a revealed result, green for PASS and red for REJECT. Its
    state is a simulated machine observation, not an action or progress message.
    """

    def __init__(self, scene, task, output):
        from pxr import UsdGeom, UsdPhysics

        self.scene, self.task = scene, task
        self.output = Path(output)
        hidden, retained = [], {}
        labels = dict(LABELS[task])
        if task == 44:
            labels["SpillLimit"] = f"MAX {scene.spill.limit_l:.1f} L"
        for prim in scene.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/Signs/") or not prim.IsA(UsdGeom.Gprim):
                continue
            # A label may be hidden, but its collider must never be disabled.
            assert not prim.HasAPI(UsdPhysics.CollisionAPI), path
            name = prim.GetName()
            if name in labels:
                retained[path] = labels[name]
                self._replace_label(name, labels[name])
            else:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden.append(path)
        self.receipt = dict(
            task_id=task,
            profile="minimal",
            hud=False,
            hidden_nonphysical_label_paths=hidden,
            retained_static_identifiers=retained,
            preserved="Original geometry/contact/actuator/scoring/clock and physical lamps, liquids, motion and safety graduations",
            instructions="External task description; no target action, phase, clock, budget, reward, completion count or evaluator result HUD in RGB",
            qc_signal=None,
        )
        if task == 45:
            self._inspection_signal()
            if scene.camera is not None:  # no camera in no-render runs
                self.receipt["signal_view_intrinsics"] = scene.camera.get_intrinsics_matrix().tolist()
        (self.output / "visual-context.json").write_text(json.dumps(self.receipt, indent=2) + "\n")

    def _replace_label(self, name, text):
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Sdf, UsdGeom, UsdShade

        stage = self.scene.stage
        prim = stage.GetPrimAtPath("/World/Signs/" + name)
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        width_m = max(p[0] for p in points) - min(p[0] for p in points)
        height_m = max(p[2] for p in points) - min(p[2] for p in points)
        width, height = 1024, max(96, round(1024 * height_m / width_m))
        image = Image.new("RGB", (width, height), (23, 43, 49))
        draw = ImageDraw.Draw(image)
        size = round(height * 0.62)
        while True:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            bounds = draw.textbbox((0, 0), text, font=font)
            if bounds[2] - bounds[0] < width * 0.94:
                break
            size -= 1
        draw.text(
            ((width - bounds[2] + bounds[0]) / 2, (height - bounds[3] + bounds[1]) / 2 - bounds[1]),
            text,
            font=font,
            fill=(225, 237, 237),
        )
        directory = self.output / "labels/minimal"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".png")
        image.save(path)
        shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Signs/" + name + "_material/Texture"))
        assert shader, name
        shader.GetInput("file").Set(Sdf.AssetPath(str(path)))

    def _fault_textures(self, aspect):
        """Task 45 fault-panel faces: state colour, a symbol and one state word (no action or
        deadline text). Written once per build, so no texture changes during an episode."""
        from PIL import Image, ImageDraw, ImageFont

        bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        height = 320
        width = round(height * aspect)
        faces = {
            "NORMAL": ((12, 150, 40), "check", ("RUN",)),
            "WARNING": ((225, 12, 6), "triangle", ("MOTOR", "FAULT")),
            "STOPPED": ((14, 16, 18), None, ("STOPPED",)),
        }
        directory = self.output / "labels/minimal"
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for state, (background, symbol, words) in faces.items():
            image = Image.new("RGB", (width, height), background)
            draw = ImageDraw.Draw(image)
            white = (250, 250, 250)
            draw.rectangle((6, 6, width - 7, height - 7), outline=white, width=10)
            left = 24
            if symbol:
                # Symbol in a height-sized square on the left; text fills the rest.
                s = round(height * 0.6)
                x0, y0 = 30, (height - s) // 2
                if symbol == "triangle":
                    draw.polygon([(x0 + s / 2, y0), (x0 + s, y0 + s), (x0, y0 + s)], fill=white)
                    inner = ImageFont.truetype(bold, round(s * 0.62))
                    draw.text((x0 + s / 2, y0 + s * 0.66), "!", fill=background, font=inner, anchor="mm")
                else:
                    draw.line([(x0 + 0.1 * s, y0 + 0.55 * s), (x0 + 0.4 * s, y0 + 0.85 * s), (x0 + 0.92 * s, y0 + 0.15 * s)], fill=white, width=round(s * 0.16), joint="curve")
                left = x0 + s + 12
            box = (left, 30, width - 26, height - 30)
            size = round((box[3] - box[1]) / len(words) * 0.95)
            while True:
                font = ImageFont.truetype(bold, size)
                widest = max(draw.textbbox((0, 0), w, font=font)[2] for w in words)
                if widest <= box[2] - box[0] or size < 12:
                    break
                size -= 2
            step = size * 1.05
            top = (box[1] + box[3]) / 2 - step * len(words) / 2
            for i, word in enumerate(words):
                draw.text(((box[0] + box[2]) / 2, top + step * (i + 0.5)), word, fill=white, font=font, anchor="mm")
            path = directory / f"FaultPanel{state}.png"
            image.save(path)
            paths[state] = path
        return paths

    def _inspection_signal(self):
        import numpy as np
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        from .arx_cell_task import set_visual_target

        scene = self.scene
        display = scene.display
        # 49: QC lamp on the D455 post below the ring light, inside the robodojo_rig head view
        # (was 1.525 m, above it).
        position = np.array([0.0, 0.9988, 1.20]) if self.task == 49 else display.front.copy()
        # 45: a 12 x 8 cm fault panel (was a 4.5 cm square) on the HMI front, grown to the left
        # of the old lamp, where the yellow stop housing does not cover it in the head view; its
        # bottom stays above the bench top (0.94 m). Each state shows a symbol + word texture.
        half = (0.0225, 0.0225)
        if self.task == 45:
            half = (0.06, 0.04)
            position = position + [-0.07, 0.0, 0.005]
            textures = self._fault_textures(2 * half[0] / (2 * half[1]))
        lamps = {}
        paths = {}
        colors = ({"INPUT": [0.025, 0.03, 0.035], "PASS": [0.025, 0.55, 0.12], "REJECT": [0.85, 0.025, 0.012]} if self.task == 49 else {"NORMAL": [0.025, 0.55, 0.12], "WARNING": [0.85, 0.025, 0.012], "STOPPED": [0.025, 0.03, 0.035]})
        for state, color in colors.items():
            body = display.bodies[state]
            path = list(body.prim_paths)[0]
            prim = scene.stage.GetPrimAtPath(path)
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
            # Existing display quads/materials; no new product/device mesh.
            UsdGeom.Mesh(prim).GetPointsAttr().Set([
                (-half[0], 0, -half[1]), (-half[0], 0, half[1]),
                (half[0], 0, -half[1]), (half[0], 0, half[1]),
            ])
            material_path = "/World/MinimalContext/InspectionSignal" + state
            material = UsdShade.Material.Define(scene.stage, material_path)
            shader = UsdShade.Shader.Define(scene.stage, material_path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            if self.task == 45:
                # Texture-driven colour on the sign quad's existing (camera-readable) st primvar.
                tex = UsdShade.Shader.Define(scene.stage, material_path + "/Texture")
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(textures[state])))
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                reader = UsdShade.Shader.Define(scene.stage, material_path + "/UV")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                for name in ["diffuseColor", "emissiveColor"]:
                    shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            else:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(2 * v for v in color)))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            UsdGeom.Imageable(prim).MakeVisible()
            lamps[state] = body
            paths[state] = path
            self.receipt["hidden_nonphysical_label_paths"].remove(str(path))
        original_set = display.set
        last = [None]

        def update(state):
            original_set(state)
            if state == last[0]:
                return
            last[0] = state
            active = state if self.task == 45 else state if state in ["PASS", "REJECT"] else "INPUT"
            # Preloaded colors avoid stale RT2 material edits. Supported Fabric
            # transforms put exactly one quad on the post and the others below
            # the opaque factory floor. No USD-only live visibility switching.
            for index, (key, body) in enumerate(lamps.items()):
                set_visual_target(body, position if key == active else np.array([0.0, 1.03, -10.0 - index]))

        display.set = update
        update(display.state)
        self.receipt["qc_signal" if self.task == 49 else "fault_signal"] = dict(
            reused_nonphysical_quads=paths,
            position_m=position.tolist(), size_m=[2 * half[0], 2 * half[1]],
            states=({"dark": "no current revealed QC result", "green": "PASS", "red": "REJECT"} if self.task == 49 else {"green": "normal motor test (check mark, RUN)", "red": "fault warning (warning triangle, MOTOR FAULT)", "dark": "motor stopped (STOPPED)"}),
            availability=("Original controller reveals result only after all three actual optical acquisitions" if self.task == 49 else "Original symbolic motor warning and measured native stop state; no action/deadline text"),
            renderer="Fixed preloaded per-state materials, supported Fabric transforms; inactive quads below opaque floor",
            limitation="Symbolic simulated inspection outcome, not a learned or calibrated defect detector",
        )
