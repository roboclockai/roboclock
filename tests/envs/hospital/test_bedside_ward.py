"""Source-name encoding and local texture reuse in the shared hospital importer."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_gb18030_obj_binds_utf8_mtl_and_sibling_texture(tmp_path):
    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdShade

    script = Path(__file__).resolve().parents[3] / "scripts/lab/import_background_assets.py"
    spec = importlib.util.spec_from_file_location("ward_import_regression", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = tmp_path / "model"
    textures = tmp_path / "textures"
    model.mkdir()
    textures.mkdir()
    Image.new("RGB", (2, 2), (230, 235, 240)).save(textures / "wall.png")
    source = model / "02.obj"
    source.write_bytes(
        (
            "v 0 0 0\nv 1000 0 0\nv 0 1000 0\n"
            "vt 0 0\nvt 1 0\nvt 0 1\no 左病房\n"
            "usemtl 墙面\nf 1/1 2/2 3/3\n"
        ).encode("gb18030")
    )
    source.with_suffix(".mtl").write_text(
        "newmtl 墙面\nKd .8 .8 .8\nmap_Kd ../textures/wall.png\n", encoding="utf8"
    )
    output = tmp_path / "ward.usdc"
    module.convert(
        source,
        output,
        0.001,
        object_names={"左病房"},
        source_encoding="gb18030",
        origin_zup_source_units=[0, 0, 0],
    )
    stage = Usd.Stage.Open(str(output))
    mesh = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    mat, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    assert mat
    texture = UsdShade.Shader(stage.GetPrimAtPath(str(mat.GetPath()) + "/Texture"))
    assert Path(texture.GetInput("file").Get().resolvedPath).is_file()
    points = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
    assert tuple(points[2]) == (0.0, 0.0, 1.0)
    record = json.loads(output.with_suffix(".json").read_text())
    assert record["source_encoding"] == "gb18030"
    assert record["materials"]["墙面"]["texture"].endswith("wall.png")


def test_missing_requested_ward_fails_with_rebuild_instruction(tmp_path):
    pytest.importorskip("pxr")
    from types import SimpleNamespace

    from roboclock.envs.hospital.bedside.ward import build_ward

    look = SimpleNamespace(config={"scene": {"bedside_ward_usd": str(tmp_path / "missing.usdc")}})
    with pytest.raises(FileNotFoundError, match="prepare_bedside_ward.py"):
        build_ward(look, None)
