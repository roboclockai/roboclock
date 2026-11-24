"""Derived chess models must load textures after a checkout is moved."""

from pathlib import Path

import pytest


def test_derived_chess_texture_references_are_portable():
    pytest.importorskip("pxr.Usd")
    from pxr import Sdf, Usd

    from roboclock.envs.sports.arcade.prepare_assets import ROOT

    checked = 0
    for color in ("white", "black"):
        for kind in ("pawn", "rook", "knight", "bishop", "queen", "king"):
            model = ROOT / "chess" / f"{color}_{kind}.usdc"
            stage = Usd.Stage.Open(str(model.resolve()))
            for prim in stage.Traverse():
                for attr in prim.GetAttributes():
                    if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                        continue
                    value = attr.Get()
                    if not value or not value.path:
                        continue
                    assert not Path(value.path).is_absolute(), (model, value.path)
                    assert (model.parent / value.path).is_file(), (model, value.path)
                    checked += 1
    assert checked > 0
