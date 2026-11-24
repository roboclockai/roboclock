"""Shared X5 rig pose, action, camera and text-label helpers; no domain semantics."""

import numpy as np


class RigView:
    def label(self, path, text, position, size, white=True):
        import hashlib
        from pathlib import Path

        from PIL import Image, ImageDraw, ImageFont
        from pxr import Sdf, UsdGeom, UsdShade

        # Text-only raster labels, generated locally; no external image service.
        folder = Path("results/shared/labels").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        width = 1024
        height = max(48, round(width * float(size[1]) / float(size[0])))
        key = repr(("aspect-fit-v2", text, white, width, height))
        filename = folder / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".png")
        if not filename.exists():
            im = Image.new("RGB", (width, height), "#e2d5ad" if white else "#17202a")
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, int(height * 0.68)))
                if font.getlength(text) > width * 0.94:
                    font = ImageFont.truetype(
                        "DejaVuSans-Bold.ttf",
                        max(10, int(font.size * width * 0.94 / font.getlength(text))),
                    )
            except OSError:
                font = ImageFont.load_default()
            draw.text(
                (im.width / 2, im.height / 2),
                text,
                font=font,
                anchor="mm",
                fill="#18212b" if white else "#f5e4bc",
            )
            im.save(filename)
        x, y, z = map(float, position)
        w, h = size
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(
            [
                (x - w / 2, y - h / 2, z),
                (x + w / 2, y - h / 2, z),
                (x + w / 2, y + h / 2, z),
                (x - w / 2, y + h / 2, z),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
        ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
        mat = UsdShade.Material.Define(self.stage, path + "/Material")
        shader = UsdShade.Shader.Define(self.stage, path + "/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        tex = UsdShade.Shader.Define(self.stage, path + "/Material/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(filename))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        uv = UsdShade.Shader.Define(self.stage, path + "/Material/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv.ConnectableAPI(), "result"
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    def tool_pose(self):
        return self.rig.pose(self.active_side)

    def apply_action(self, joints):
        self.rig.command = np.asarray(joints).copy()
        self.rig.apply()

    def state(self):
        return self._state

    def observation(self, depth=False):
        from roboclock.simulators.isaac.rendering import capture_camera

        capture_camera()
        rgba = self.camera.get_rgba()
        if rgba is None or not rgba.size:
            raise RuntimeError("Camera has no fresh frame")
        p, q = self.tool_pose()
        joints = self.robot.get_joint_positions()
        from roboclock.simulators.isaac.embodiments.single import width
        result = {
            "rgb": np.asarray(rgba[:, :, :3], dtype=np.uint8).copy(),
            "joint_positions": np.asarray(joints, dtype=np.float32),
            "ee_pose": np.asarray([*p, *q], dtype=np.float32),
            "gripper_width": np.array([width(self)], dtype=np.float32),
            "sim_time": np.array([self.task.time], dtype=np.float32),
            "instruction": self.task.gen_instruction(),
        }
        if depth:
            result["depth"] = np.asarray(
                self.camera.get_current_frame()["distance_to_image_plane"], dtype=np.float32
            ).copy()
        return result

    def close(self):
        if self.camera is not None:
            self.camera.destroy()
            self.camera = None
        if self.world is not None:
            self.world.stop()
            self.world.clear()
            self.world = None
