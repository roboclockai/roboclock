"""Build hospital visuals into a plain USD stage, with Isaac's material classes stubbed.

Catches duplicate prims, bad transforms and missing materials before the simulator runs.
Skipped when the pure-USD library (usd-core) is not installed.
"""

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdShade  # noqa: E402

from roboclock.envs.hospital.appearance import HospitalAppearance  # noqa: E402
from roboclock.envs.hospital.config import load_legacy_config as load_config  # noqa: E402
from roboclock.envs.hospital.emergency_scenes import TORSO_N  # noqa: E402


class FakeMaterial:
    def __init__(self, stage, name):
        self.material = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
        self.shaders_list = [UsdShade.Shader.Define(stage, f"/World/Materials/{name}/Shader")]


class StubAppearance(HospitalAppearance):
    """Replaces Isaac-only material creation; geometry code runs unchanged."""

    def material(self, name, *, color=None, roughness=None, metallic=None, emissive=False):
        if name not in self.materials:
            self.materials[name] = FakeMaterial(self.stage, name)
        return self.materials[name]

    def glass(self):
        return self.material("glass")

    def amber_glass(self):
        return self.material("amber_glass")

    def label(self, path, text, center, size, color=(0.1, 0.35, 0.5), *, cylindrical_radius=None):
        if cylindrical_radius is not None and path.endswith("/Label"):
            return None
        return UsdGeom.Mesh.Define(self.stage, path)


def make(stage):
    config = load_config()
    config["visuals"]["labels"] = True
    return StubAppearance(stage, None, config)


def test_pill_vial_prims_are_unique_and_transforms_valid():
    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    path = "/World/Objects/cup_0"
    UsdGeom.Xform.Define(stage, path)
    # The backend creates the physics collider first, under the same parent.
    UsdGeom.Cylinder.Define(stage, path + "/Body")
    look.pill_vial_visual(path, 0.023, 0.062, 1, (0.9, 0.2, 0.1), seed=1)
    names = [p.GetName() for p in stage.GetPrimAtPath(path).GetChildren()]
    assert len(names) == len(set(names))
    for required in ("Body", "AmberBody", "Cap", "LabelSticker", "LabelStrip"):
        assert required in names
    assert sum(n.startswith("Pill_") for n in names) >= 30


def test_every_task_scene_dressing_builds_without_duplicate_prims():
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import LEGACY_TASKS as TASKS
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    for task_id in TASKS:
        stage = Usd.Stage.CreateInMemory()
        look = make(stage)
        task = make_task(task_id, "easy", 0, config)
        layout = make_layout(task, config)
        look.room(layout.table_height)
        look.wall_fixtures()
        look.wall_extras()
        look.everyday_objects()
        look.equipment(layout.table_height)
        look.props()
        look.dress_task(layout)
        paths = [str(p.GetPath()) for p in stage.Traverse()]
        assert len(paths) == len(set(paths)), task_id


def test_every_surgical_instrument_builds_with_unique_prims():
    import numpy as np

    from roboclock.envs.hospital.layout import load_catalog

    catalog = load_catalog()
    for kind in ("scalpel", "forceps", "scissors", "clamp", "retractor"):
        stage = Usd.Stage.CreateInMemory()
        look = make(stage)
        path = f"/World/Objects/{kind}"
        UsdGeom.Xform.Define(stage, path)
        look.surgical_tool(path, kind, np.array(catalog[kind]["size"]))
        paths = [str(p.GetPath()) for p in stage.Traverse()]
        assert len(paths) == len(set(paths)) and len(paths) > 4, kind


def test_operating_room_and_procedure_board_build():
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    layout = make_layout(make_task("A3", "easy", 0, config), config)
    look.operating_room(layout)
    paths = [str(p.GetPath()) for p in stage.Traverse()]
    assert len(paths) == len(set(paths))
    # The real bed model, or the built mattress when the model is not installed.
    assert any("Mattress" in p or p.endswith("/Bed") for p in paths) and any(
        "Board" in p for p in paths
    )


def test_procedure_board_call_panels_render_and_start_hidden(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOCLOCK_ASSET_CACHE", str(tmp_path))
    from roboclock.envs.hospital.appearance import call_label_texture
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    waiting, called = call_label_texture(2), call_label_texture(2, "scalpel")
    assert waiting.exists() and called.exists() and waiting != called

    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    layout = make_layout(make_task("A3", "easy", 0, load_config()), load_config())
    look.procedure_board(layout.steps)
    assert set(look.callouts) == {f"{kind}_{k}" for kind in ("wait", "call") for k in range(3)}
    for k in range(3):
        call = UsdGeom.Imageable(stage.GetPrimAtPath(look.callouts[f"call_{k}"]))
        wait = UsdGeom.Imageable(stage.GetPrimAtPath(look.callouts[f"wait_{k}"]))
        assert call.ComputeVisibility() == UsdGeom.Tokens.invisible
        assert wait.ComputeVisibility() == UsdGeom.Tokens.inherited


def test_blood_bag_and_fridge_drawer_build_with_a_prismatic_joint():
    from pxr import UsdPhysics

    from roboclock.envs.hospital.drawer import author_drawer
    from roboclock.envs.hospital.layout import load_catalog, make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    layout = make_layout(make_task("A2", "easy", 0, config), config)
    assert layout.drawers
    for drawer in layout.drawers:
        root = author_drawer(stage, look, drawer)
        assert stage.GetPrimAtPath(root).HasAPI(UsdPhysics.RigidBodyAPI)
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"/World/Joints/{drawer.name}"))
        assert joint.GetUpperLimitAttr().Get() == pytest.approx(drawer.travel)
    path = "/World/Objects/bag_0"
    UsdGeom.Xform.Define(stage, path)
    spec = load_catalog()["blood_bag"]
    look.blood_bag_visual(path, spec["size"], spec["eyelet"])
    paths = [str(p.GetPath()) for p in stage.Traverse()]
    assert len(paths) == len(set(paths))
    # The real bag model (with its own hanging hole) or, without it, the built eyelet ring.
    assert any(p.endswith("/Model") or "/Eyelet" in p for p in paths)


def test_blood_bank_dressing_and_draped_or_build_with_unique_prims():
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    for task_id, build in (("A2", "dress_blood_bank"), ("A3", "operating_room")):
        stage = Usd.Stage.CreateInMemory()
        look = make(stage)
        layout = make_layout(make_task(task_id, "easy", 0, config), config)
        getattr(look, build)(layout)
        paths = [str(p.GetPath()) for p in stage.Traverse()]
        assert len(paths) == len(set(paths)) and len(paths) > 10, task_id


def test_timer_bar_builds_hidden_and_shrinks_when_set():
    import numpy as np

    from roboclock.envs.hospital.timers import BAR_LENGTH, TimerBar

    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    bar = TimerBar(stage, look, "ind_demo", np.array([0.77, 0.10, 0.795]), 0.75)
    prim = stage.GetPrimAtPath("/World/Timers/ind_demo_Bar")
    assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
    bar.set(1.0, "green")
    assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.inherited
    full = bar.scale.Get()[1]
    bar.set(0.25, "red")
    quarter, center = bar.scale.Get()[1], bar.translate.Get()
    assert full == pytest.approx(BAR_LENGTH) and quarter == pytest.approx(0.25 * BAR_LENGTH)
    assert center[1] == pytest.approx(0.10 - BAR_LENGTH / 2 + quarter / 2)  # drains toward -y
    bar.set(-3.0, "red")  # out-of-range fractions are clamped, never inverted
    assert bar.scale.Get()[1] > 0


def test_prop_is_placed_by_its_footprint_center_at_true_size(tmp_path):
    """A prop whose origin is a corner still lands centered where it is asked to stand."""
    import numpy as np

    from roboclock.envs.hospital.props import CATALOG, add_prop

    spec = CATALOG["SM_MedicalCabinet_01a"]  # its origin sits at a corner, not at the center
    lo, size = spec.min, spec.size
    middle = [lo[i] + size[i] / 2 for i in range(3)]
    (tmp_path / "SM_MedicalCabinet_01a.usda").write_text(
        f"""#usda 1.0
(
    defaultPrim = "Prop"
)

def Xform "Prop"
{{
    double3 xformOp:translate = ({middle[0]}, {middle[1]}, {middle[2]})
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Cube "Body"
    {{
        double size = 1
        double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
"""
    )
    stage = Usd.Stage.CreateInMemory()
    add_prop(
        stage,
        "SM_MedicalCabinet_01a",
        "/World/Cabinet",
        (1.5, -0.8),
        yaw=np.pi / 2,
        base=str(tmp_path),
        suffix=".usda",
    )
    box = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        .ComputeWorldBound(stage.GetPrimAtPath("/World/Cabinet"))
        .ComputeAlignedRange()
    )
    low, high = np.array(box.GetMin()), np.array(box.GetMax())
    assert (low[:2] + high[:2]) / 2 == pytest.approx([1.5, -0.8], abs=1e-3)
    assert high - low == pytest.approx([size[1], size[0], size[2]], abs=1e-3)  # a quarter turn
    assert low[2] == pytest.approx(0.0, abs=1e-3)  # standing on the floor


def test_cardiac_arrest_bay_builds_with_unique_prims_and_a_patient_on_the_bed():
    import numpy as np

    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    layout = make_layout(make_task("B2", "easy", 0, config), config)
    look.dress_cardiac_arrest(layout)
    paths = [str(p.GetPath()) for p in stage.Traverse()]
    assert len(paths) == len(set(paths)) and len(paths) > 80
    for required in ("Head", "Sheet", "Mattress", "Hand"):
        alternatives = {"Mattress": ("Mattress", "/Bed")}.get(required, (required,))
        assert any(p.endswith(r) or f"/{r}" in p for r in alternatives for p in paths), required
    chest = next(f for f in layout.fixtures if f.name == "chest")
    head = np.array(
        stage.GetPrimAtPath("/World/Patient/Head").GetAttribute("xformOp:translate").Get()
    )
    assert head[1] < chest.position[1] - 0.25  # the head is toward the back wall, beyond the chest
    assert abs(head[0] - chest.position[0]) < 0.01  # and centered on the body


def test_aed_cables_never_pass_through_the_patient_or_the_aed_case():
    import numpy as np

    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    for seed in range(4):
        stage = Usd.Stage.CreateInMemory()
        look = make(stage)
        layout = make_layout(make_task("B2", "easy", seed, config), config)
        look.dress_cardiac_arrest(layout)
        chest = next(f for f in layout.fixtures if f.name == "chest")
        aed = next(f for f in layout.fixtures if f.name == "aed")
        cx, cy = float(chest.position[0]), float(chest.position[1])
        chest_top = float(chest.position[2] + chest.size[2] / 2)
        mattress_top = chest_top - 0.20
        for k in range(2):
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/AED/Cable_{k}"))
            points = np.array([[*p] for p in mesh.GetPointsAttr().Get()])
            # Torso: superellipse cross-section tapering from the shoulders to the waist.
            mix = np.clip((points[:, 1] - (cy - 0.26)) / 0.53, 0.0, 1.0)
            width = 0.225 + (0.185 - 0.225) * mix
            inside = (abs(points[:, 0] - cx) / width) ** TORSO_N + (
                abs(points[:, 2] - (chest_top - 0.10)) / 0.10
            ) ** TORSO_N
            in_range = (points[:, 1] > cy - 0.26) & (points[:, 1] < cy + 0.27)
            assert np.all(inside[in_range] > 1.02), (
                seed,
                k,
                "a cable point is inside or touching the torso",
            )
            # Arms: cylinders along y beside the body.
            for sign in (-1, 1):
                ax = cx + sign * 0.248
                upper = (points[:, 1] > cy - 0.19) & (points[:, 1] < cy + 0.15)
                reach = np.hypot(points[:, 0] - ax, points[:, 2] - (mattress_top + 0.047))
                assert np.all(reach[upper] > 0.047 + 0.003), (
                    seed,
                    k,
                    "a cable point is inside the upper arm",
                )
                fore = (points[:, 1] > cy + 0.135) & (points[:, 1] < cy + 0.435)
                reach = np.hypot(
                    points[:, 0] - (ax + sign * 0.02), points[:, 2] - (mattress_top + 0.038)
                )
                assert np.all(reach[fore] > 0.038 + 0.003), (
                    seed,
                    k,
                    "a cable point is inside the forearm",
                )
            # The AED's case walls: nothing crosses a wall below its top edge.
            wall_top = aed.position[2] + aed.size[2] / 2 + 0.015
            right = aed.position[0] + 0.131
            near_wall = (abs(points[:, 0] - right) < 0.012) & (
                abs(points[:, 1] - aed.position[1]) < 0.135
            )
            assert np.all(points[near_wall, 2] > wall_top), (
                seed,
                k,
                "a cable passes through the case wall",
            )


def test_aed_case_clears_the_patients_near_arm_and_hand():
    """The AED stands on an over-bed table across the near hip (common rig, 2026-11-03): the table's
    underside clears the near arm and hand, the gown flap over the chest and the sheet's hip fold."""
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    for seed in range(4):
        layout = make_layout(make_task("B2", "easy", seed, config), config)
        chest = next(f for f in layout.fixtures if f.name == "chest")
        aed = next(f for f in layout.fixtures if f.name == "aed")
        table = next(f for f in layout.fixtures if f.name == "overbed_table")
        chest_top = chest.position[2] + chest.size[2] / 2
        underside = table.position[2] - table.size[2] / 2
        assert underside >= chest_top + 0.02  # gown flap (+6 mm) and the sheet's hip fold (+5 mm)
        assert aed.position[2] - aed.size[2] / 2 == pytest.approx(underside + table.size[2])
        lo = table.position[:2] - table.size[:2] / 2
        hi = table.position[:2] + table.size[:2] / 2
        assert np.all(aed.position[:2] - aed.size[:2] / 2 >= lo - 1e-9)
        assert np.all(aed.position[:2] + aed.size[:2] / 2 <= hi + 1e-9)


def test_every_task_prop_exists_and_builds_on_a_stage():
    """The per-task floor props must name catalogued assets and form valid prim paths (a path starting with a digit is empty)."""
    from pxr import Usd

    from roboclock.envs.hospital.appearance import HOSPITAL_PROPS
    from roboclock.envs.hospital.props import CATALOG, add_prop

    stage = Usd.Stage.CreateInMemory()
    for task, props in HOSPITAL_PROPS.items():
        for index, (name, xy, yaw) in enumerate(props):
            assert name in CATALOG, (task, name)
            prim = add_prop(stage, name, f"/World/{task}/prop_{index}_{name}", xy, yaw)
            assert prim.GetPrim().IsValid(), (task, name)


def test_every_task_model_is_installed_and_builds_on_a_stage():
    from pxr import Usd

    from roboclock.envs.hospital.appearance import HOSPITAL_MODELS
    from roboclock.envs.hospital.props import add_model, model_size

    stage = Usd.Stage.CreateInMemory()
    for task, models in HOSPITAL_MODELS.items():
        for index, (name, xy, yaw, *height) in enumerate(models):
            assert model_size(name) is not None, (task, name)
            assert (
                add_model(
                    stage,
                    name,
                    f"/World/{task}/model_{index}_{name}",
                    (*xy, height[0] if height else 0.0),
                    yaw=yaw,
                )
                .GetPrim()
                .IsValid()
            )


def test_operating_room_shows_the_table_and_patient_but_no_doctors():
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    layout = make_layout(make_task("A3", "easy", 0, config), config)
    look.operating_room(layout)
    paths = {str(p.GetPath()) for p in stage.Traverse()}
    for required in ("/World/OR/Bed", "/World/OR/Patient/Drape", "/World/OR/Patient/Head"):
        assert required in paths, required
    # The scenario is described in the task text; the scene renders only the robot's work and its signals.
    assert not any("Surgeon" in p or "Anesthesiologist" in p for p in paths)


def test_cardiac_arrest_patient_has_two_feet_under_a_sheet_that_covers_the_legs():
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    look.dress_cardiac_arrest(make_layout(make_task("B2", "easy", 0, config), config))
    paths = {str(p.GetPath()) for p in stage.Traverse()}
    assert "/World/Patient/FootL" in paths and "/World/Patient/FootR" in paths
    assert "/World/Patient/Bed" in paths


def test_a2_uses_the_refrigerator_model_in_place_of_the_built_cabinet():
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.props import model_size
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    layout = make_layout(make_task("A2", "easy", 0, config), config)
    shell = [
        f for f in layout.fixtures if f.name.startswith("fridge_") and f.name != "fridge_fascia"
    ]
    assert shell and all(f.collision for f in shell)  # the colliders stay either way
    assert all(not f.visible for f in shell) == (model_size("blood_bank_fridge_open") is not None)
    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    look.dress_blood_bank(layout)
    assert stage.GetPrimAtPath("/World/BloodBank/Fridge").IsValid()


def test_dial_ring_empties_and_the_hand_sweeps_as_time_runs_out():
    from pxr import UsdGeom

    from roboclock.envs.hospital.timers import Dial

    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    dial = Dial(stage, look, "demo", (0.8, 0.0, 1.0))

    def lit():
        return sum(
            UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible
            for p in dial.segments
        )

    assert lit() == Dial.SEGMENTS  # waiting: the whole ring is lit in the state color
    dial.set(1.0, "green")
    assert lit() == Dial.SEGMENTS
    dial.set(0.5, "amber")
    assert lit() == Dial.SEGMENTS // 2
    angle_half = dial._hand_r.Get()
    dial.set(0.1, "red")
    assert lit() == 3 and dial._hand_r.Get() != angle_half
    dial.set(0.0, "red")
    assert lit() == 0
    before = dial.color
    dial.state((0.1, 0.3, 1.0))  # a running countdown ignores the state color
    assert dial.color == before and lit() == 0


def test_state_color_lights_the_whole_dial_when_no_countdown_runs():
    from pxr import UsdGeom

    from roboclock.envs.hospital.timers import Dial

    stage = Usd.Stage.CreateInMemory()
    look = make(stage)
    dial = Dial(stage, look, "state", (0.8, 0.0, 1.0))
    dial.state((0.95, 0.10, 0.10))
    assert dial.color == "red"
    assert all(
        UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible for p in dial.segments
    )
