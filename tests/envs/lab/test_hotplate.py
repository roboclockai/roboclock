from dataclasses import replace

import numpy as np
import pytest
from helpers import at_target, observe_for, snapshot

from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.mdp.state import ObjectState
from roboclock.envs.lab.tasks.registry import make_task


def lifted(task, **changes):
    source = task.layout.objects["sample_0"]
    obj = ObjectState(source.position + [0, 0, 0.04], source.orientation, grasped=True)
    return snapshot(task, sample_0=replace(obj, **changes))


@pytest.mark.parametrize(
    "difficulty,window,placement",
    [("easy", 4.69, 20.32), ("medium", 2.63, 11.36), ("hard", 1.48, 6.35)],
)
def test_hotplate_windows_and_legacy_name(difficulty, window, placement):
    task = make_task("B1", difficulty)
    assert task.spec.max_latency == window
    assert task.spec.transfer_deadline == placement
    assert make_task("alarm_transfer").name == "hotplate_isolation"


def test_hotplate_support_and_clutter(task_factory):
    from roboclock.envs.lab.clutter import reserved_regions, sample_clutter

    task = task_factory("B1")
    source = task.layout.objects["sample_0"]
    bottom = source.position[2] - task.layout.dimensions["sample_0"][1] / 2
    assert bottom == pytest.approx(task.layout.heating_zone.support_z + 0.002)
    assert task.layout.heating_zone.support_z == pytest.approx(task.layout.table_height + 0.08)
    config = load_config()
    regions = reserved_regions(task, task.layout, config)
    for item in sample_clutter(task, task.layout, config):
        assert not any(
            np.all(np.abs(item.position[:2] - center) < item.size[:2] / 2 + extent)
            for center, extent in regions
        )


def test_grasp_on_plate_is_not_isolation(task_factory):
    task = task_factory("B1", max_latency=0.1)
    task.event_time = task.time = 1.0
    source = task.layout.objects["sample_0"]
    obj = ObjectState(
        source.position, source.orientation, grasped=True, fixture_contacts=frozenset(["heater"])
    )
    observe_for(task, snapshot(task, sample_0=obj), duration=0.2)
    assert task.grasp_time is not None and task.action_time is None
    assert task.metrics()["overheat_failure"]


@pytest.mark.parametrize(
    "changes",
    [
        {"grasped": False},
        {"fixture_contacts": frozenset(["heater"])},
        {"orientation": np.array([0.70710678, 0.70710678, 0, 0])},
        {"position": np.array([2.0, 0, 1.0])},
    ],
)
def test_invalid_lifts_do_not_count(task_factory, changes):
    task = task_factory("B1")
    task.event_time = task.time = 1.0
    observe_for(task, lifted(task, **changes), duration=0.2)
    assert task.action_time is None


def test_lift_dwell_resets_after_loss_of_grasp(task_factory):
    task = task_factory("B1")
    task.event_time = task.time = 1.0
    observe_for(task, lifted(task), duration=0.03)
    observe_for(task, lifted(task, grasped=False), duration=0.01)
    observe_for(task, lifted(task), duration=0.03)
    assert task.action_time is None
    observe_for(task, lifted(task), duration=0.02)
    assert task.action_time == pytest.approx(1.09)
    assert not task.physical_success  # Still must release into the safety tray.


def test_pre_event_lift_and_reheating_stay_invalid(task_factory):
    task = task_factory("B1")
    observe_for(task, lifted(task), duration=0.05)
    assert "pre_event_action" in task.violations
    task.event_time = task.time
    observe_for(task, lifted(task), duration=0.05)
    task.observe(snapshot(task), 0.01)
    assert "returned_to_hotplate" in task.violations
    observe_for(task, snapshot(task, sample_0=at_target(task)))
    assert task.physical_success and not task.joint_success


def test_warning_and_timeout_advance_without_action(task_factory):
    task = task_factory("B1", event_time=0.01, max_latency=0.02)
    assert not task.visual_state()["alarm"]
    for _ in range(4):
        commands = task.advance(0.01)
        task.observe(snapshot(task), 0.01)
    assert any(c.kind == "hotplate_visual" for c in commands)
    assert task.visual_state()["alarm"] and task.visual_state()["failed"]
    assert sum(e["type"] == "event_triggered" for e in task.event_log) == 1
    observe_for(task, lifted(task), duration=0.05)
    observe_for(task, snapshot(task, sample_0=at_target(task)))
    assert task.physical_success and not task.temporal_success
    assert task.metrics()["overheat_failure"]


def test_hotplate_asset_is_portable_and_has_no_embedded_rigid_bodies():
    from pathlib import Path

    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics

    root = Path(__file__).resolve().parents[3] / "src/roboclock/envs/lab/assets/hotplate"
    stage = Usd.Stage.Open(str(root / "hotplate.usdc"))
    assert stage.GetDefaultPrim().GetName() == "Hotplate"
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 5
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    bounds = (
        UsdGeom.BBoxCache(0, ["default"])
        .ComputeWorldBound(stage.GetDefaultPrim())
        .ComputeAlignedRange()
    )
    assert bounds.GetMax()[2] == pytest.approx(0.08)
    assert (root / "atlas.png").exists()
    assert "CC0" in (root / "LICENSE.txt").read_text()


def test_warning_audio_matches_event_and_isolation():
    from roboclock.envs.lab.hotplate_audio import SAMPLE_RATE, warning_samples

    times = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    audio = warning_samples(times, 0.5, 1.1)
    assert np.max(np.abs(audio)) <= 0.25
    assert not audio[times < 0.5].any()
    assert not audio[times >= 1.1].any()
    assert not warning_samples(times, None).any()
    for start, frequency in [(0.50, 880), (0.65, 1320)]:
        segment = audio[(times >= start) & (times < start + 0.15)]
        frequencies = np.fft.rfftfreq(len(segment), 1 / SAMPLE_RATE)
        assert frequencies[np.argmax(np.abs(np.fft.rfft(segment)))] == pytest.approx(
            frequency, abs=8
        )
