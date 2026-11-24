"""Explicit hospital task registry."""

from roboclock.envs.hospital.config import load_config, sample_episode
from roboclock.envs.hospital.tasks.bedside.autoinjector import AnaphylaxisAutoInjector
from roboclock.envs.hospital.tasks.bedside.iv_bag import IVBagSpikingHanging
from roboclock.envs.hospital.tasks.bedside.pump_panel import PumpPanel
from roboclock.envs.hospital.tasks.bedside.roller_clamp import RollerClamp
from roboclock.envs.hospital.tasks.bedside.stopcock import StopcockSwitch
from roboclock.envs.hospital.tasks.bedside.tourniquet import HaemorrhageTourniquet
from roboclock.envs.hospital.tasks.bedside.ventilator import VentilatorReconnect
from roboclock.envs.hospital.tasks.reactive.aed_response import CardiacArrest
from roboclock.envs.hospital.tasks.reactive.drug_spill import HazardousDrugSpill
from roboclock.envs.hospital.tasks.reactive.fire_response import FireBlanket
from roboclock.envs.hospital.tasks.reactive.oxygen_desaturation import OxygenDesaturation
from roboclock.envs.hospital.tasks.reactive.supply_alarm import SupplyAlarm
from roboclock.envs.hospital.tasks.temporal_window.blood_bag_transfer import BloodBagTransfer
from roboclock.envs.hospital.tasks.temporal_window.instrument_handoff import InstrumentHandoff
from roboclock.envs.hospital.tasks.temporal_window.pill_delivery import PillDelivery
from roboclock.envs.hospital.tasks.temporal_window.syringe_dose import SyringeDose
from roboclock.envs.hospital.tasks.throughput.cart_restock import CartRestock
from roboclock.envs.hospital.tasks.throughput.pharmacy_bins import PharmacyBins
from roboclock.envs.hospital.tasks.throughput.sterilizer_loading import SterilizerLoading
from roboclock.envs.hospital.tasks.window.bag_switch import SequentialBagSwitch
from roboclock.envs.hospital.tasks.window.bvm_ventilation import BagValveVentilation
from roboclock.envs.hospital.tasks.window.drip_chamber import DripChamberLevel
from roboclock.envs.hospital.tasks.window.syringe_cutoff import SyringePumpCutoff

LEGACY_TASKS = {
    cls.task_id: cls
    for cls in (
        PillDelivery,
        BloodBagTransfer,
        InstrumentHandoff,
        SyringeDose,
        SupplyAlarm,
        CardiacArrest,
        FireBlanket,
        PharmacyBins,
        CartRestock,
        SterilizerLoading,
    )
}


# Ward throughput tasks C1-C4 (scene-02 left ward, two arms).
WARD_TASKS = {
    "C1": "roboclock.envs.hospital.tasks.ward.pharmacy_order_bins:PharmacyOrderBins",
    "C2": "roboclock.envs.hospital.tasks.ward.instrument_tray_sort:InstrumentTraySort",
    "C3": "roboclock.envs.hospital.tasks.ward.crash_cart_vials:CrashCartVials",
    "C4": "roboclock.envs.hospital.tasks.ward.blood_tube_triage:BloodTubeTriage",
}


def ward_task_class(task_id):
    from importlib import import_module

    module, name = WARD_TASKS[task_id].split(":")
    return getattr(import_module(module), name)


# Ward bedside tasks (bedside_profile): a response limit from the instruction. A4 (start the
# infusion, B6 until 2026-11-19) is counted with the A class; B4-B6 are bedside reactions (the
# pump panel was B7 until then).
# B4 is the ventilator circuit reconnection since 2026-11-19; the infusion-line switch
# (StopcockSwitch) that was B4 until then left the suite and stays importable for old recordings
# (results.py), like the other retired tasks.
# B1 is the oxygen desaturation response since 2026-11-19 (alarm at a seeded event time, limit from
# the alarm); the ICU pump alarms (SupplyAlarm) that were B1 until then stay the legacy profile's
# B1 (LEGACY_TASKS) and map old recordings (results.py).
BEDSIDE_TASKS = {
    cls.task_id: cls
    for cls in (
        IVBagSpikingHanging, OxygenDesaturation, VentilatorReconnect, AnaphylaxisAutoInjector,
        HaemorrhageTourniquet,
    )
}
# B5 is the anaphylaxis auto-injector since 2026-11-20 (alarm at a seeded event time, limit from
# the alarm); "hang the bag, open the clamp" (RollerClamp) that was B5 until then left the suite
# and stays importable for old recordings (results.py).
RETIRED_BEDSIDE_TASKS = {StopcockSwitch.name: StopcockSwitch, RollerClamp.name: RollerClamp}
# B6 is the haemorrhage tourniquet since 2026-11-20 (bleeding at a seeded event time, limit from
# its start); the pump-panel key press (PumpPanel) that was B6 until then left the suite and
# stays importable for old recordings (results.py).
RETIRED_BEDSIDE_TASKS[PumpPanel.name] = PumpPanel

# Ward process-window tasks A1-A2: act while a monitored quantity is inside its band. The syringe
# pump cutoff was A3 until 2026-11-19 (replaced by the ventilation rhythm below); its class stays
# importable for old recordings (results.py) but is no longer registered.
PROCESS_TASKS = {cls.task_id: cls for cls in (DripChamberLevel, SequentialBagSwitch)}
RETIRED_TASKS = {SyringePumpCutoff.name: SyringePumpCutoff}
# Ward rhythm task A3 (window_profile: rhythm): one bag squeeze inside each 6 s breath window.
RHYTHM_TASKS = {BagValveVentilation.task_id: BagValveVentilation}

# The v2 suite: A1-A3 process windows and A4, B1-B3 emergencies, B4-B6 bedside reactions, C1-C4
# ward throughput. The archived A1-A4 window tasks and the tabletop C1-C3 remain reachable through
# the legacy profile.
# Ward emergency B3 (spill_profile: spill): the hazardous drug spill. The fire blanket was B3 until
# 2026-11-19; it stays the legacy profile's B3 (LEGACY_TASKS) and maps old recordings (results.py).
SPILL_TASKS = {HazardousDrugSpill.task_id: HazardousDrugSpill}

TASKS = {
    **PROCESS_TASKS,
    **RHYTHM_TASKS,
    "A4": IVBagSpikingHanging,
    **{key: cls for key, cls in LEGACY_TASKS.items() if key.startswith("B")},
    **SPILL_TASKS,
    **{key: cls for key, cls in BEDSIDE_TASKS.items() if key.startswith("B")},
    **{task_id: ward_task_class(task_id) for task_id in WARD_TASKS},
}


def make_task(task_id: str, difficulty="easy", seed=0, config=None):
    config = config or load_config()
    configured = config["tasks"]
    tasks = {}
    for key, cls in {**LEGACY_TASKS, **TASKS}.items():
        if key not in configured:
            continue
        if key in WARD_TASKS and configured[key].get("ward_profile") != "ward":
            cls = LEGACY_TASKS.get(key)  # an archived profile keeps the tabletop C1-C3
        if key in PROCESS_TASKS and configured[key].get("window_profile") != "process":
            cls = LEGACY_TASKS.get(key)  # and the historical A1-A4 window tasks
        if key in RHYTHM_TASKS and configured[key].get("window_profile") != "rhythm":
            cls = LEGACY_TASKS.get(key)
        if key in SPILL_TASKS and configured[key].get("spill_profile") != "spill":
            cls = LEGACY_TASKS.get(key)  # the archived fire blanket shares the v2 B3's ID
        if key in BEDSIDE_TASKS and configured[key].get("bedside_profile") != "bedside":
            cls = LEGACY_TASKS.get(key)  # the archived A4 (syringe dose) shares the v2 A4's ID
        if cls is not None:
            tasks[key] = cls
    matches = [cls for key, cls in tasks.items() if task_id in (key, cls.name)]
    if not matches:
        raise ValueError(f"Unknown task {task_id!r}; choose one of {', '.join(tasks)}")
    cls = matches[0]
    return cls(sample_episode(cls.task_id, difficulty, seed, config))


def make_legacy_task(task_id, difficulty="easy", seed=0, config=None):
    from roboclock.envs.hospital.config import load_legacy_config

    config = config or load_legacy_config()
    matches = [cls for key, cls in LEGACY_TASKS.items() if task_id in (key, cls.name)]
    if not matches:
        raise ValueError(f"Unknown historical hospital task: {task_id}")
    cls = matches[0]
    return cls(sample_episode(cls.task_id, difficulty, seed, config))
