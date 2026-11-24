"""Benchmark scoring for the Industry tasks from measured physical state.

The native scenes measure process events and most success predicates themselves
(``scene.observe()``). Where the recorded diagnostic controller also decided part of
the outcome - grasp/clearance events read by a scene predicate, arrest and tray
retention windows (B3), credited kit parts / shelved cartons / sorted parts /
assembled palettes (C1-C4) - the same native predicates and stability windows are
evaluated here from scene state, independent of any controller phase. The scene
only ever sees events written by itself or by these monitors; the oracle's
controller writes into a separate overlay (see ``oracle.py``).

Monitors run inside the environment for every policy:

* ``physics(o)`` after each native ``scene.observe()`` call;
* ``tick(o)`` on each control tick, where the native controllers measured;
* ``outcome(o)`` the result row fields at any time.

``hold_s`` is how long the recorded runner kept simulating after completion; the
environment terminates that long after the measured completion. Throughput tasks (C)
follow ``docs/c-class-timing.md`` (``ThroughputMonitor``): they run to the horizon H = 5 t_o
of their frozen oracle time unless all Q jobs are done and the scene has stayed valid for
``SETTLE_S``, or the stall stop fires; time plays no part in their physical success.
"""

import math
import re
from fnmatch import fnmatchcase

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.evaluation import throughput


def jsonable(value):
    """Plain JSON values: numpy scalars/arrays converted, non-finite floats dropped."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _body_name(path):
    """A contact actor's short name: an arm body as ``<side>/<link>``, anything else
    without the ``/World/`` prefix."""
    parts = path.strip().strip("/").split("/")
    if parts[:2] == ["World", "ARX"] and len(parts) > 3:
        link = next((p for p in reversed(parts) if re.fullmatch(r"base_link|link\d+", p)), parts[-1])
        return f"{parts[2]}/{link}"
    return "/".join(parts[1:] if parts[0] == "World" else parts)


# An arm collision: an arm link other than the gripper's fingers pressing on anything (a fixture,
# the bench, a mount, a product, the other arm, the arm itself) with more than this normal force.
# It is the sports rule (``envs/sports/c_class.ARM_COLLISION_N``): policy runs there showed
# grazes at 4 N or less and collisions at 96 N or more, and a collaborative arm's collision
# detection stops it at tens of newtons.
#
# Until 2026-11-10 every contact voided physical success, whatever its force, except a finger on
# the product: a finger resting on the fixture that holds the product, a link brushing a mount at
# a fraction of a newton, the palm on the handle it turns. In the GPT 6 Astra sweep that rule
# voided 18 of 20 episodes, three of which had completed their task. The scripted oracles make
# no such contact at all, and their validation (``native/runner.py`` and each task's validator)
# still requires none: this constant is the verdict for a policy.
ARM_COLLISION_N = 20.0
FINGER_LINKS = ("link7", "link8")


def _arm_link(path):
    """The link name of an arm body path ("link6", "base_link"), or None for any other body."""
    if not path.startswith("/World/ARX/"):
        return None
    return next((p for p in reversed(path.split("/")) if re.fullmatch(r"base_link|link\d+", p)), "")


def contact_voids(pair, record):
    """Whether one recorded arm contact voids physical success (see ``ARM_COLLISION_N``).

    ``pair`` is the record's key, ``"<body> | <body>"``; ``record`` holds what the contact
    callback measured. Fingers work by contact, so a pair whose only arm bodies are fingers never
    voids, however hard they press: what a finger does to a part or a fixture is judged by the
    task's own predicates. Any other arm link voids once the pair's peak normal force exceeds
    ``ARM_COLLISION_N``. The zero-impulse arm-to-arm log (``ContactBook.interarm_events``)
    carries no force and does not void by itself: a real touch is also recorded with its force.
    A record without a force (a log made before forces were kept) keeps the old strict verdict.
    """
    if "force_threshold_applied" in record:
        return False
    bodies = record.get("actors") or pair.split(" | ")
    links = [link for link in map(_arm_link, bodies) if link is not None]
    if not links or all(link in FINGER_LINKS for link in links):
        return False
    force = record.get("peak_force_N")
    return True if force is None else float(force) > ARM_COLLISION_N


def voiding_contacts(unexpected):
    """The recorded contacts that void physical success, of ``{pair: record}``."""
    return {pair: record for pair, record in unexpected.items() if contact_voids(pair, record)}


def arm_contact_peak(unexpected):
    """Largest normal force (N) any non-finger arm link pressed with, 0 without such a contact."""
    forces = [
        float(record.get("peak_force_N") or 0.0)
        for pair, record in unexpected.items()
        if "force_threshold_applied" not in record
        and any(
            link not in FINGER_LINKS
            for link in map(_arm_link, record.get("actors") or pair.split(" | "))
            if link is not None
        )
    ]
    return max(forces, default=0.0)


def physical_failures(completed, voiding):
    """Why a row's physical verdict is false, as the codes the other domains report under
    ``physical_failures`` (cross-domain summaries read that field): ``task_not_completed`` when
    the task's own physical predicate does not hold, then one ``arm_collision:<a>|<b>`` per
    contact pair that voided it (``voiding_contacts``; empty when the verdict is true)."""
    codes = [] if completed else ["task_not_completed"]
    for pair in sorted(voiding):
        codes.append("arm_collision:" + "|".join(_body_name(a) for a in pair.split(" | ")))
    return codes


class PoseWindow:
    """Consecutive measured poses that must stay within a position/orientation span.

    Mirrors the native controllers' windows: samples older than ``dwell`` are dropped
    (keeping the oldest one that still spans the dwell), and the window is stable once
    it spans ``dwell`` with the summed per-body position span and per-body rotation
    from the first sample inside the limits.
    """

    def __init__(self, dwell, span_m, angle_deg, *, min_samples=2, tolerance=0.0):
        self.dwell, self.span, self.angle = dwell, span_m, np.deg2rad(angle_deg)
        self.min_samples, self.tolerance = min_samples, tolerance
        self.rows = []

    def clear(self):
        self.rows.clear()

    def add(self, t, positions, rotations):
        """positions (N,3) / rotations (N,3,3) of the bodies; returns stability or None."""
        self.rows.append((t, np.atleast_2d(positions), np.reshape(rotations, (-1, 3, 3))))
        while len(self.rows) > 1 and self.rows[1][0] <= t - self.dwell:
            self.rows.pop(0)
        if len(self.rows) < self.min_samples or t - self.rows[0][0] < self.dwell - self.tolerance:
            return None
        positions = np.array([r[1] for r in self.rows])
        rotations = np.array([r[2] for r in self.rows])
        span = float(np.linalg.norm(np.ptp(positions, axis=0), axis=-1).max())
        angle = float(
            Rotation.from_matrix(
                (rotations @ rotations[0].transpose(0, 2, 1)).reshape(-1, 3, 3)
            ).magnitude().max()
        )
        if span > self.span or angle > self.angle:
            return None
        return dict(
            time_s=t,
            samples=len(self.rows),
            position_span_m=span,
            orientation_from_first_rad=angle,
        )


class Dwell:
    """Time a condition has held continuously, as the controllers' ``since`` timers."""

    def __init__(self):
        self.since = None

    def update(self, t, condition):
        self.since = (self.since if self.since is not None else t) if condition else None
        return 0.0 if self.since is None else t - self.since


class Monitor:
    category = "temporal_window"
    hold_s = None
    required_operations = 1

    def __init__(self, scene, resolved, native):
        self.scene, self.resolved, self.native = scene, resolved, native
        self.events = {}
        self.completed_at = None
        self.termination = None

    # -- hooks -------------------------------------------------------------
    def physics(self, o):
        pass

    def tick(self, o):
        pass

    def mark(self, name, t, **detail):
        if name not in self.events:
            self.events[name] = dict(time_s=float(t), **detail) if detail else float(t)

    def complete(self, t):
        if self.completed_at is None:
            self.completed_at = float(t)
            self.mark("physical_completion_s", t)

    def finished(self, t):
        return (
            self.hold_s is not None
            and self.completed_at is not None
            and t - self.completed_at > self.hold_s
        )

    def end(self, reason):
        """The environment ended the episode for `reason` (it reached its length: "horizon")."""
        if self.termination is None:
            self.termination = reason

    # -- results -----------------------------------------------------------
    def unexpected(self):
        contacts = getattr(self.scene, "contacts", None)
        unexpected = contacts.unexpected if hasattr(contacts, "unexpected") else None
        return dict(unexpected if unexpected is not None else getattr(self.scene, "unexpected", {}))

    def result(self, o):
        """Task-specific physical/temporal predicates."""
        raise NotImplementedError

    def outcome(self, o):
        result = self.result(o)
        unexpected = self.unexpected()
        voiding = voiding_contacts(unexpected)
        completed = bool(result.pop("physical"))
        physical = completed and not voiding
        temporal = bool(result.pop("temporal"))
        violations = list(result.pop("violations", []))
        correct = int(result.pop("correct_operations", int(physical)))
        # These are the scalars aggregate() averages; None means not applicable.
        result.setdefault("reaction_latency", None)
        row = dict(
            task_success=physical,
            manipulation_success=physical,
            temporal_success=temporal,
            joint_success=physical and temporal,
            temporal_violations=violations,
            correct_operations=correct,
            required_operations=self.required_operations,
            completion_time=self.completed_at,
            # Every arm contact recorded, whatever its force (what the oracle's validation
            # forbids); the ones that void this verdict are `arm_collisions`.
            collision_count=len(unexpected),
            unexpected_contacts=sorted(unexpected),
            arm_collisions=sorted(voiding),
            arm_contact_peak_N=arm_contact_peak(unexpected),
            physical_failures=physical_failures(completed, voiding),
            measured_events=self.events,
            # Scene/process event times (scene.observe() and these monitors only).
            scene_events={
                k: v for k, v in self.scene.events.items() if isinstance(v, (int, float, bool))
            },
            **result,
        )
        return jsonable(row)


# --------------------------------------------------------------------- A tasks
class PlatingMonitor(Monitor):
    """A1 (41): whole-part exit inside the recipe window after immersion (reviewed 20-40 s;
    levels in configs/difficulty.yaml); released rinse retention."""

    hold_s = 3.0

    def physics(self, o):
        if self.scene.success:
            self.complete(self.scene.events.get("supported_released_retention_s", o["time_s"]))

    def result(self, o):
        s, r = self.scene, self.resolved
        duration = None if s.final_exit is None else s.final_exit - s.process_start
        temporal = duration is not None and r["process_min_s"] <= duration <= r["process_max_s"]
        violations = []
        if duration is None:
            violations.append("part never cleared the plating liquid after immersion")
        elif not temporal:
            violations.append(
                f"plating duration {duration:.2f}s outside [{r['process_min_s']}, {r['process_max_s']}] s"
            )
        return dict(
            physical=s.success, temporal=temporal, violations=violations, plating_duration_s=duration
        )


class GraspEvents:
    """Bilateral grasp as the native controllers measured it: both finger contact
    forces above a threshold for longer than ``dwell`` (control ticks)."""

    DWELL_S = 0.25

    def __init__(self, threshold, dwell=DWELL_S):
        self.threshold, self.dwell, self.timer = threshold, dwell, Dwell()

    def update(self, t, forces):
        return self.timer.update(t, min(forces) > self.threshold) > self.dwell


class RotaryMonitor(Monitor):
    """A2 (42): supported upright release in the moving carrier inside the sector.

    The scene credits source clearance only after a bilateral grasp; that grasp is
    measured here (both fingers > 10 N for > 0.25 s, the controller's criterion).
    """

    hold_s = 3.0
    grasp_force_n = 10.0  # stated in the instruction (tasks.judged_rules)

    def __init__(self, *args):
        super().__init__(*args)
        self.grasp = GraspEvents(self.grasp_force_n)

    def tick(self, o):
        forces = [o["forces"].get("finger7", 0), o["forces"].get("finger8", 0)]
        if "bilateral_grasp_s" not in self.scene.events and self.grasp.update(o["time_s"], forces):
            self.scene.events["bilateral_grasp_s"] = o["time_s"]
            self.mark("bilateral_grasp_s", o["time_s"])

    def physics(self, o):
        if "retention_confirmed_s" in self.scene.events:
            self.complete(self.scene.events["retention_confirmed_s"])

    def result(self, o):
        events = self.scene.events
        placed = "supported_release_s" in events and o["supported"] and o["released"]
        violations = []
        if "release_angle_deg" in events and not events.get("temporal_success"):
            violations.append(
                f"released at carrier angle {events['release_angle_deg']:.1f} deg outside "
                f"[{self.resolved['loading_sector_deg'][0]:g}, {self.resolved['loading_sector_deg'][1]:g}]"
            )
        elif "release_angle_deg" not in events:
            violations.append("no supported release in the carrier")
        return dict(
            physical=placed,
            temporal=bool(self.scene.success),
            violations=violations,
            release_angle_deg=events.get("release_angle_deg"),
        )


class BondingMonitor(Monitor):
    """A3 (43): released aligned seating confirmed 0.5 s, beginning inside the window after
    the adhesive bead completes (reviewed 2-14 s; levels in configs/difficulty.yaml). The scene's timing predicate also needs the lid to have
    been grasped (both fingers > 8 N for > 0.25 s) and lifted clear of its source
    (lowest point >= 0.86 m, source support <= 0.1 N); both are measured here."""

    hold_s = 2.5
    grasp_force_n = 8.0  # stated in the instruction (tasks.judged_rules)

    def __init__(self, *args):
        super().__init__(*args)
        self.grasp = GraspEvents(self.grasp_force_n)

    def tick(self, o):
        events, t = self.scene.events, o["time_s"]
        if "bilateral_grasp_s" not in events and self.grasp.update(t, o["finger_forces_N"]):
            events["bilateral_grasp_s"] = t
            self.mark("bilateral_grasp_s", t)
        if (
            "bilateral_grasp_s" in events
            and "source_cleared_s" not in events
            and o["lowest_point_world_z_m"] >= 0.86
            and o["source_support_force_N"] <= 0.1
        ):
            events["source_cleared_s"] = t
            self.mark("source_cleared_s", t)

    def physics(self, o):
        if self.scene.physical_completed:
            self.complete(self.scene.events["seating_confirmed_s"])

    def result(self, o):
        s, events = self.scene, self.scene.events
        glue, seated = events.get("glue_completed_s"), events.get("seated_release_s")
        first = events.get("first_assembly_contact_s")
        lo, hi = self.resolved["window_relative_s"]
        violations = []
        if glue is None:
            violations.append("adhesive bead never completed")
        elif seated is not None and not glue + lo <= seated <= glue + hi:
            violations.append(f"seating began {seated - glue:.2f}s after the bead, outside [{lo}, {hi}] s")
        if glue is not None and first is not None and first < glue + lo:
            violations.append("cover touched the housing before the bonding window")
        if seated is None:
            violations.append("cover never seated")
        return dict(
            physical=s.physical_completed and o["physical_seating"],
            temporal=bool(s.success),
            violations=violations,
            seating_after_bead_s=None if glue is None or seated is None else seated - glue,
        )


# --------------------------------------------------------------------- B tasks
class IsolationMonitor(Monitor):
    """B1 (44): valve closed (>= 89 deg, confirmed 0.15 s, with post-leak finger
    contact) within the deadline of leak onset (reviewed 5 s; levels in
    configs/difficulty.yaml); still closed and not leaking at the end."""

    category = "event_reactive"
    hold_s = 2.5

    def physics(self, o):
        if self.scene.physical_completed:
            self.complete(self.scene.events["closure_confirmed_s"])

    def result(self, o):
        s, events = self.scene, self.scene.events
        violations = []
        if "pre_event_motion_s" in events:
            violations.append("valve moved before the leak")
        response = events.get("response_s")
        if response is None:
            violations.append("valve never confirmed closed after the leak")
        elif response > self.resolved["response_deadline_s"]:
            violations.append(f"closed {response:.2f}s after leak onset (> {self.resolved['response_deadline_s']} s)")
        return dict(
            physical=s.physical_completed and o["closed"] and not o["leaking"],
            temporal=bool(s.success),
            violations=violations,
            reaction_latency=response,
            estimated_spill_l=o.get("estimated_spill_l"),
        )


class MotorMonitor(Monitor):
    """B2 (45): 8 mm stop-button press with finger contact sustained 0.2 s within
    3 s of the warning, then the braked motor stays stopped."""

    category = "event_reactive"
    hold_s = 2.5

    def physics(self, o):
        if self.scene.physical_completed:
            self.complete(self.scene.events["motor_stopped_s"])

    def result(self, o):
        s, events = self.scene, self.scene.events
        violations = []
        if "premature_press_s" in events:
            violations.append("button pressed before the warning")
        response = events.get("response_s")
        if "button_confirmed_s" not in events:
            violations.append("stop button never confirmed")
        elif response is not None and response > self.resolved["response_deadline_s"]:
            violations.append(f"pressed {response:.2f}s after the warning (> {self.resolved['response_deadline_s']} s)")
        return dict(
            physical=s.physical_completed and s.final_valid(o),
            temporal=bool(s.success),
            violations=violations,
            reaction_latency=response,
        )


class BearingMonitor(Monitor):
    """B3 (46): stable arrest wholly inside the safety zone within 3 s of measured
    zone entry, then the released bearing retained on the rework tray.

    Arrest (controller window in the recording): bearing inside the zone, lane
    support > 0.2 N and a robot fingertip braking it (> 0.2 N; the recorded oracle
    brakes with the right hand, either hand counts here), pose stable over 0.15 s
    (0.2 mm / 0.2 deg). A hand contact before zone entry voids the timing.
    Recovery: ``scene.tray_valid`` held with a 0.5 s pose window (0.2 mm / 0.5 deg).
    """

    category = "event_reactive"
    hold_s = 2.5

    def __init__(self, *args):
        super().__init__(*args)
        self.arrest = PoseWindow(0.15, 0.0002, 0.2)
        gate = self.resolved["tray_retention_stability"]
        self.retention = PoseWindow(gate["dwell_s"], gate["position_span_m"], gate["orientation_from_first_deg"])
        self.retention_valid = False

    def inside(self, o):
        zone = self.resolved["zone_x_m"]
        return (
            o["bounds_min_m"][0] >= zone[0]
            and o["bounds_max_m"][0] <= zone[1]
            and o["bounds_min_m"][1] >= 0.80
            and o["bounds_max_m"][1] <= 1.02
        )

    def tick(self, o):
        s, t = self.scene, o["time_s"]
        pose = (np.array(o["bearing_position"]), np.array(o["bearing_rotation"]))
        entry = s.entry_at
        if entry is not None and "arrest_confirmed_s" not in self.events:
            fingers = [f for pair in o["finger_forces_N"].values() for f in pair]
            # Stopped by a hand, wholly inside the zone. Whether the lane still carries the
            # bearing is not part of an arrest: a firm two-finger catch takes its weight off
            # the lane, and used to go uncredited until the grip was relaxed.
            supported = self.inside(o) and max(fingers) > 0.2
            stable = self.arrest.add(t, *pose) if supported else self.arrest.clear()
            if stable:
                self.mark("arrest_window", t, window=stable)
                self.mark("arrest_confirmed_s", t)
                self.events["response_s"] = t - entry
        eligible = s.tray_valid(o)
        stable = self.retention.add(t, *pose) if eligible else self.retention.clear()
        self.retention_valid = stable is not None
        if stable and "released_tray_retention_s" not in s.events:
            # The outfeed QC lamp turns green on this scene event.
            s.events["released_tray_retention_s"] = t
            self.mark("released_tray_retention_s", t, window=stable)
            self.complete(t)

    def result(self, o):
        s = self.scene
        response = self.events.get("response_s")
        premature = s.first_hand_at is not None and (s.entry_at is None or s.first_hand_at < s.entry_at)
        violations = []
        if premature:
            violations.append("hand contacted the bearing before it entered the safety zone")
        if s.entry_at is not None and response is None:
            violations.append("bearing was not arrested inside the safety zone")
        elif response is not None and response > self.resolved["response_deadline_s"]:
            violations.append(f"arrested {response:.2f}s after zone entry (> {self.resolved['response_deadline_s']} s)")
        return dict(
            physical="released_tray_retention_s" in s.events and s.tray_valid(o) and self.retention_valid,
            temporal=response is not None and response <= self.resolved["response_deadline_s"] and not premature,
            violations=violations,
            reaction_latency=response,
        )


# --------------------------------------------------------------------- C tasks
def progress_steps(task_id, resolved):
    """Credited sub-steps in a whole C task (sets the stall window, which the instruction
    states: tasks.stall_rule)."""
    if task_id == "C3":
        return 6  # every part's weighing, then its sort
    return {"C1": 6, "C2": 3, "C4": 10}[task_id]  # 2 kits x 3 components; cartons; 8 pans + 2 deliveries


class ThroughputMonitor(Monitor):
    """A C task as Q jobs (docs/c-class-timing.md, ``roboclock.evaluation.throughput``).

    - ``job_times()``: each job's credit time (None until done); ``completed_at`` is the
      tick of the Q-th credit, t_Q, the quantity the frozen oracle time t_o was measured as.
    - physical: ``final_valid(o)`` at the end of the episode (all Q jobs done, every one
      still valid; no time term) and no arm collision (``Monitor.outcome``, ``contact_voids``).
    - temporal: per tier from t_Q, judged in ``tasks.IndustryTask.metrics`` with the
      episode's difficulty (``throughput.judge``); the monitor does not know the tier.
    - Early end (``finished``): "all_done" once all Q jobs are done and ``final_valid`` has
      held for ``SETTLE_S``; "stall" once ``throughput.stalled`` (from T_medium on, no
      progress for half of t_o). Progress is a done job or any credited sub-step
      (``progress_patterns``, the stage credits): one industry job takes the oracle up to
      ~40 s, so whole jobs alone would stop a policy that is still placing parts.
      The environment ends the rest at the horizon ("horizon").
    """

    category = "throughput"
    task_id = None  # class ID, for the frozen oracle time
    progress_patterns = ()  # measured_events that count as progress for the stall stop

    def __init__(self, *args):
        super().__init__(*args)
        self.oracle_time = throughput.oracle_time("industry", self.task_id)
        self.settled = Dwell()

    def job_times(self):
        """Credit time of each of the Q jobs, None while not done."""
        raise NotImplementedError

    def event_time(self, name):
        value = self.events.get(name)
        return value["time_s"] if isinstance(value, dict) else value

    def result(self, o):
        correct = self.correct()
        valid = self.final_valid(o)
        violations = []
        if correct < self.required_operations:
            violations.append(f"{correct}/{self.required_operations} jobs done")
        elif not valid:
            violations.append("done work no longer valid at the end of the episode")
        elapsed = max(float(o["time_s"]), 1e-9)
        return dict(
            physical=valid,
            # Placeholder: all Q done. The tier's verdict replaces it (tasks.metrics).
            temporal=self.completed_at is not None,
            violations=violations,
            correct_operations=correct,
            operations_per_minute=correct * 60.0 / elapsed,
        )

    def outcome(self, o):
        row = super().outcome(o)
        row["job_times_s"] = [None if t is None else float(t) for t in self.job_times()]
        row["termination_reason"] = self.termination
        return row

    def credit_complete(self, o):
        """End of a control tick: mark t_Q at the Q-th credit, then check the early ends."""
        t = float(o["time_s"])
        if self.correct() >= self.required_operations:
            self.mark("required_operations_s", t)
            self.complete(t)
        if self.termination is not None:
            return
        done = self.completed_at is not None
        settled = self.settled.update(t, done and self.final_valid(o))
        if done and settled >= throughput.SETTLE_S - 1e-6:
            self.termination = "all_done"
        elif self.settled.since is None and throughput.stalled(  # no stall while settling
            t, self.last_progress_time(), self.oracle_time, steps=self.progress_steps()
        ):
            self.termination = "stall"

    def last_job_time(self):
        done = [t for t in self.job_times() if t is not None]
        return max(done) if done else None

    def progress_steps(self):
        """Progress steps in the whole task (sets the stall window): one per credited sub-step."""
        return progress_steps(self.task_id, self.resolved)

    def last_progress_time(self):
        """Latest done job or credited sub-step, None if nothing yet."""
        times = [t for t in self.job_times() if t is not None]
        times += [
            self.event_time(name)
            for name in self.events
            if any(fnmatchcase(name, pattern) for pattern in self.progress_patterns)
        ]
        times = [float(t) for t in times if t is not None]
        return max(times) if times else None

    def finished(self, t):
        return self.termination is not None


class KittingMonitor(ThroughputMonitor):
    """C1 (47): a component is credited once released whole inside its kit insert on
    the insert floor, supported and free of finger contact, with a 0.3 s pose window
    (0.2 mm / 0.2 deg). A job is a kit of bearing+bracket+pin, done at its third credit."""

    task_id = "C1"
    progress_patterns = ("kit?_*_credited_s",)

    required_operations = 2

    def __init__(self, *args):
        super().__init__(*args)
        from roboclock.envs.industry.native.kitting_batch import NAMES

        self.names = NAMES
        gate = self.resolved["placement_stability"]
        self.windows = {
            (name, j): PoseWindow(
                gate["dwell_s"], gate["position_span_m"], gate["orientation_from_first_deg"],
                min_samples=1, tolerance=1e-6,
            )
            for name in NAMES
            for j in (0, 1)
        }
        self.credited = set()

    def tick(self, o):
        t, floor = o["time_s"], self.resolved["kit_floor_m"]
        for (name, j), window in self.windows.items():
            if (j, name) in self.credited:
                continue
            part = o["parts"][f"{name}{j}"]
            valid = self.placed(part, floor)
            stable = window.add(t, part["position"], part["rotation"]) if valid else window.clear()
            if stable:
                self.credited.add((j, name))
                self.mark(f"kit{j}_{name}_credited_s", t, window=stable)
                if all((j, n) in self.credited for n in self.names):
                    self.mark(f"kit{j}_complete_s", t)
        self.credit_complete(o)

    @staticmethod
    def placed(part, floor):
        """Released whole inside its insert, on the floor, supported, no finger contact."""
        return bool(
            part["whole_contained"]
            and floor - 0.001 <= part["bounds_min_m"][2] <= floor + 0.002
            and (part["kit_support_N"] > 0.005 or part["kit_touching"])
            and max(f for pair in part["finger_forces_N"].values() for f in pair) < 0.01
        )

    def correct(self):
        return sum(all((j, n) in self.credited for n in self.names) for j in (0, 1))

    def job_times(self):
        return [self.event_time(f"kit{j}_complete_s") for j in (0, 1)]

    def final_valid(self, o):
        """All 6 credited and, on this frame, still contained and released as when credited (the credit
        test without its pose window), so a part still held in the insert does not count."""
        s, floor = self.scene, self.resolved["kit_floor_m"]
        return bool(
            len(self.credited) == 6
            and all(s.contained(n, j) for n in self.names for j in range(2))
            and all(self.placed(o["parts"][f"{n}{j}"], floor) for n in self.names for j in range(2))
        )

    def result(self, o):
        row = super().result(o)
        row["credited_components"] = sorted(f"kit{j}:{n}" for j, n in self.credited)
        return row


class PutawayMonitor(ThroughputMonitor):
    """C2 (48): a carton is credited when ``scene.valid_product`` (inside its SKU's
    shelf bounds, on the shelf floor, supported, released, at rest) holds > 0.5 s."""

    task_id = "C2"
    progress_patterns = ("*_correct_slot_credit_s",)
    required_operations = 3

    def __init__(self, *args):
        super().__init__(*args)
        self.timers = {i: Dwell() for i in range(len(self.resolved["products"]))}
        self.placed = set()

    def tick(self, o):
        t = o["time_s"]
        for index, timer in self.timers.items():
            if index not in self.placed and timer.update(t, self.scene.valid_product(index)) > 0.5:
                self.placed.add(index)
                self.mark(f"{index}_correct_slot_credit_s", t)
        self.credit_complete(o)

    def correct(self):
        return len(self.placed)

    def job_times(self):
        return [self.event_time(f"{i}_correct_slot_credit_s") for i in self.timers]

    def final_valid(self, o):
        return bool(
            len(self.placed) >= self.required_operations
            and all(self.scene.valid_product(i) for i in self.placed)
        )

    def result(self, o):
        row = super().result(o)
        row["placed_ids"] = sorted(self.placed)
        return row


class WeighingMonitor(ThroughputMonitor):
    """C3 (49): a part is weighed once it rests alone, released and at rest on the bench scale
    for 0.5 s (``weigh_scale.WeighTracker``); its result is PASS at ``weigh_scale.MIN_G`` or
    more, else REJECT. A part is sorted once, after its weighing, it rests released in a tray of
    its result (``scene.contained``: either arm's receiver of that result, support, at rest) for
    > 0.3 s."""

    task_id = "C3"
    progress_patterns = ("*_weighed", "*_sorted_s")

    required_operations = 3

    def __init__(self, *args):
        super().__init__(*args)
        from roboclock.envs.industry.native.weigh_scale import WeighTracker

        self.weigh = WeighTracker()
        self.sort_timers = {i: Dwell() for i in range(3)}
        self.sorted = set()

    def unexpected(self):
        # Every arm-to-arm touch is listed, the zero-impulse ones too (the oracle's validation
        # forbids them all). Which of them void a policy's verdict is `contact_voids`.
        return {**super().unexpected(), **self.scene.contacts.interarm_events}

    def tick(self, o):
        t = o["time_s"]
        for index, grams in self.weigh.update(o):
            self.mark(f"{index}_weighed", t, reading_g=grams)
        for index in range(3):
            part = o["parts"][str(index)]
            good = (
                index in self.weigh.weighed
                and part["contained"]
                and part["support_N"] > 0.01
                and np.linalg.norm(part["linear_velocity"]) < 0.012
                and np.linalg.norm(part["angular_velocity"]) < 0.1
                and max(f for forces in part["finger_forces_by_arm"].values() for f in forces) < 0.01
            )
            if index not in self.sorted and self.sort_timers[index].update(t, good) > 0.3:
                self.sorted.add(index)
                self.mark(f"{index}_sorted_s", t)
        self.credit_complete(o)

    def correct(self):
        return len(self.sorted)

    def job_times(self):
        return [self.event_time(f"{i}_sorted_s") for i in range(3)]

    def final_valid(self, o):
        s = self.scene
        return bool(
            len(self.sorted) == 3
            and all(
                s.contained(i)
                and i in self.weigh.weighed
                and (
                    o["parts"][str(i)]["support_N"] > 0.01
                    or s.receiver_touching(i)
                )
                and np.linalg.norm(s.parts[i].get_linear_velocity()) < 0.012
                and np.linalg.norm(s.parts[i].get_angular_velocity()) < 0.1
                and max(v for forces in o["parts"][str(i)]["finger_forces_by_arm"].values() for v in forces) < 0.01
                for i in range(3)
            )
        )

    def result(self, o):
        row = super().result(o)
        row["weighed_g"] = {str(k): v for k, v in self.weigh.weighed.items()}
        row["sorted"] = sorted(self.sorted)
        return row


class PaletteMonitor(ThroughputMonitor):
    """C4 (50): a pan is credited once seated in its pocket (``scene.contained``),
    supported by the case and released, with a 0.2 s pose window over pan and case;
    a palette is delivered when ``scene.payload_native_valid`` holds for its case and
    four credited pans with a 0.5 s window over all five bodies.

    Any of a colour's six feed pans may fill that colour's pocket: they are identical, and the
    instruction names none. (Until 2026-11 only the pan the oracle picks was credited, index 0
    for the left case and 2 for the right; ``pan_of`` records which pan filled each pocket.)

    Task rule (not expressible as a joint target): once a palette's four pans are
    credited, that palette's arm switches its finger drives to the declared case-lip
    grip gains (``case_grip.finger_position_stiffness_N_m`` / ``finger_damping_N_s_m``);
    the recorded controller set the same gains when it started the case transfer.
    """

    task_id = "C4"
    progress_patterns = ("pan?_?_credited_s", "delivery?_s")

    required_operations = 2

    def __init__(self, *args):
        super().__init__(*args)
        gate, output = self.resolved["pan_release_stability"], self.resolved["carrier_output_stability"]
        self.pan_windows = {
            (c, color): PoseWindow(gate["dwell_s"], gate["position_span_m"], gate["orientation_from_first_deg"])
            for c in (0, 1)
            for color in range(4)
        }
        self.output_windows = {
            c: PoseWindow(output["dwell_s"], output["position_span_m"], output["orientation_from_first_deg"])
            for c in (0, 1)
        }
        self.credited, self.delivered, self.case_gains = set(), set(), set()
        self.pan_of, self.candidate = {}, {}  # (case, colour) -> the pan index in or entering its pocket

    def seated(self, o, c, color):
        """Index of a pan of `color` seated in case c's pocket, supported and released, that
        fills no other pocket; the oracle's pan first. None when there is none."""
        taken = {(col, j) for (_, col), j in self.pan_of.items()}
        for j in sorted(range(6), key=lambda j: j != (0 if c == 0 else 2)):
            seat = o["pans"][f"{color}_{j}"].get("cases", {}).get(str(c))
            if (
                seat is not None
                and (color, j) not in taken
                and (seat["support_N"] > 0.005 or seat["touching"])
                and seat["contained"]
                and max(o["pans"][f"{color}_{j}"]["finger_forces_N"]) < 0.01
            ):
                return j
        return None

    def pans(self, c):
        """{colour: pan index} of case c's credited pans."""
        return {color: j for (case, color), j in self.pan_of.items() if case == c}

    def case_gains_for(self, cycle, t):
        if cycle in self.case_gains:
            return
        s, side = self.scene, "left" if cycle == 0 else "right"
        grip = self.resolved["case_grip"]
        robot = s.dual.robots[side]
        kp = np.array([700 if int(n[5:]) <= 6 else grip["finger_position_stiffness_N_m"] for n in s.dual.names[side]])
        kd = np.array([70 if int(n[5:]) <= 6 else grip["finger_damping_N_s_m"] for n in s.dual.names[side]])
        robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
        self.case_gains.add(cycle)
        self.mark(f"{side}_case_grip_gains_s", t, stiffness_N_m=grip["finger_position_stiffness_N_m"])

    def tick(self, o):
        s, t = self.scene, o["time_s"]
        for (c, color), window in self.pan_windows.items():
            if (c, color) in self.credited:
                continue
            j = self.seated(o, c, color)
            if j != self.candidate.get((c, color)):
                window.clear()  # another pan, or none: its window starts over
                self.candidate[c, color] = j
            if j is None:
                continue
            pan = o["pans"][f"{color}_{j}"]
            case_p, case_R = s.pose(s.case[c])
            stable = window.add(t, np.array([pan["position"], case_p]), np.array([pan["rotation"], case_R]))
            if stable:
                self.credited.add((c, color))
                self.pan_of[c, color] = j
                self.mark(f"pan{c}_{color}_credited_s", t, window=stable, pan=j)
        for c in (0, 1):
            if all((c, color) in self.credited for color in range(4)):
                self.case_gains_for(c, t)
            if c in self.delivered:
                continue
            valid = all((c, color) in self.credited for color in range(4)) and s.payload_native_valid(o, c, self.pans(c))
            if valid:
                bodies = [o["cases"][str(c)]] + [o["pans"][f"{color}_{j}"] for color, j in sorted(self.pans(c).items())]
                stable = self.output_windows[c].add(
                    t, np.array([b["position"] for b in bodies]), np.array([b["rotation"] for b in bodies])
                )
            else:
                stable = self.output_windows[c].clear()
            if stable:
                self.delivered.add(c)
                self.mark(f"delivery{c}_s", t, window=stable)
        self.credit_complete(o)

    def correct(self):
        return len(self.delivered)

    def job_times(self):
        return [self.event_time(f"delivery{c}_s") for c in (0, 1)]

    def final_valid(self, o):
        s = self.scene
        return bool(
            len(self.delivered) >= 2
            and all(
                len(self.pans(c)) == 4 and s.payload_native_valid(o, c, self.pans(c))
                for c in (0, 1)
            )
        )

    def result(self, o):
        row = super().result(o)
        row["credited_pans"] = sorted(f"{c}_{color}" for c, color in self.credited)
        row["pan_of"] = {f"{c}_{color}": j for (c, color), j in sorted(self.pan_of.items())}
        row["delivered"] = sorted(self.delivered)
        return row


MONITORS = {
    41: PlatingMonitor,
    42: RotaryMonitor,
    43: BondingMonitor,
    44: IsolationMonitor,
    45: MotorMonitor,
    46: BearingMonitor,
    47: KittingMonitor,
    48: PutawayMonitor,
    49: WeighingMonitor,
    50: PaletteMonitor,
}
