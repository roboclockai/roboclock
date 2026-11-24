"""Recognizable CC0 chess pieces with mesh collision and a furnished chess room."""

import json

import numpy as np

from roboclock.envs.chess.backend import ChessBackend
from roboclock.envs.chess.task import ChessTask
from roboclock.envs.sports.arcade.scenes import ROOT, bind_isaac_materials, build_room
from roboclock.physics.geometry import rotation_matrix


class RoomChessBackend(ChessBackend):
    def customize_scene(self):
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        catalog = json.loads((ROOT / "chess/pieces.json").read_text())
        kinds = dict(p="pawn", r="rook", n="knight", b="bishop", q="queen", k="king")
        self.piece_envelopes = {}
        for name, body in self.objects.items():
            symbol = self.task.initial_symbols[name]
            item = catalog[("white" if symbol.isupper() else "black") + "_" + kinds[symbol.lower()]]
            root = self.stage.GetPrimAtPath(body.prim_path)
            root.SetTypeName("Xform")
            root.RemoveAPI(UsdPhysics.CollisionAPI)
            root.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
            self.stage.RemovePrim(body.prim_path + "/Identity")
            prim = add_reference_to_stage(
                str((ROOT / "chess" / item["file"]).resolve()), body.prim_path + "/Model"
            )
            low, high = np.array(item["min"]), np.array(item["max"])
            scale = 2 * self.config["scene"]["piece_radius"] / max((high - low)[:2])
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            center = np.r_[(low[:2] + high[:2]) / 2, low[2]]
            xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -self.config["scene"]["piece_height"] / 2))
            xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
            xf.AddTranslateOp(opSuffix="normalize").Set(Gf.Vec3d(*map(float, -center)))
            bind_isaac_materials(self.stage, prim)
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(child)
                    UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr(
                        "convexDecomposition"
                    )
                    api = PhysxSchema.PhysxCollisionAPI.Apply(child)
                    api.CreateContactOffsetAttr(0.001)
                    api.CreateRestOffsetAttr(0)
            self.piece_envelopes[name] = dict(
                symbol=symbol, size=((high - low) * scale).tolist(), collision="convexDecomposition"
            )
        build_room(self, "chess")


class MeshChessTask(ChessTask):
    """Keep chess rules and add complete mesh-envelope containment to settling."""

    def located(self, name, obj):
        if not super().located(name, obj):
            return False
        size = np.asarray(self.piece_envelopes[name]["size"])
        rotation = rotation_matrix(obj["orientation"])
        offset = np.array([0, 0, (size[2] - self.config["scene"]["piece_height"]) / 2])
        center = obj["position"] + rotation @ offset
        extents = np.abs(rotation) @ (size / 2)
        half = (
            0.0475 if self.expected[name] == "capture" else self.config["scene"]["square_size"] / 2
        )
        return bool(
            np.all(
                np.abs(center[:2] - self.position(self.expected[name])[:2]) + extents[:2]
                <= half - 0.001
            )
        )
