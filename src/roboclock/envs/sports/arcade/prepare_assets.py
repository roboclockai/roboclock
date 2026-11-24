"""Prepare individually referenceable chess pieces from the preserved CC0 USD."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent / "assets"


def prepare():
    from pxr import Usd, UsdGeom, UsdUtils

    src = ROOT / "chess/chess_set_1k.usdc"
    original = Usd.Stage.Open(str(src.resolve()))
    root = original.GetDefaultPrim()
    pieces = {}
    for color in ("white", "black"):
        for kind in ("pawn", "rook", "knight", "bishop", "queen", "king"):
            chosen = next(p for p in root.GetChildren() if f"piece_{kind}_{color}" in p.GetName())
            stage = Usd.Stage.Open(original.Flatten())
            for child in list(stage.GetDefaultPrim().GetChildren()):
                if child.GetName().startswith("piece_") and child.GetName() != chosen.GetName():
                    stage.RemovePrim(child.GetPath())
                elif (
                    any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(child))
                    and child.GetName() != chosen.GetName()
                ):
                    stage.RemovePrim(child.GetPath())
            box = (
                UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                .ComputeWorldBound(stage.GetPrimAtPath(chosen.GetPath()))
                .ComputeAlignedRange()
            )
            # Export in the original folder so relative texture references remain valid.
            out = ROOT / f"chess/{color}_{kind}.usdc"
            UsdUtils.ModifyAssetPaths(
                stage.GetRootLayer(),
                lambda p: os.path.relpath(p, src.parent.resolve()) if Path(p).is_absolute() else p,
            )
            stage.GetRootLayer().Export(str(out))
            pieces[f"{color}_{kind}"] = dict(
                file=out.name, min=list(box.GetMin()), max=list(box.GetMax())
            )
    (ROOT / "chess/pieces.json").write_text(json.dumps(pieces, indent=2) + "\n")
    props = {}
    for name in ("wooden_hammer_01", "dining_chair_02", "side_table_01"):
        p = next((ROOT / name).glob("*.usdc"))
        stage = Usd.Stage.Open(str(p.resolve()))
        box = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeWorldBound(stage.GetDefaultPrim())
            .ComputeAlignedRange()
        )
        props[name] = dict(
            file=str(p.relative_to(ROOT)),
            min=list(box.GetMin()),
            max=list(box.GetMax()),
            axis=UsdGeom.GetStageUpAxis(stage),
        )
    (ROOT / "props.json").write_text(json.dumps(props, indent=2) + "\n")
    print("ASSETS_PREPARED", json.dumps(props), flush=True)


if __name__ == "__main__":
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime():
        prepare()
