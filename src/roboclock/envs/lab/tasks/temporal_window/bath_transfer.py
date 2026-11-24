"""A5/A6: measured immersion duration followed by contact-backed bath transfer."""

from dataclasses import replace

import numpy as np

from roboclock.envs.lab.mdp.events import SceneCommand
from roboclock.envs.lab.station_frame import station_of
from roboclock.envs.lab.tasks.base import LabTask

# A5's reel counts as lifting once it rises this far above its stop-bath position.
LIFT_START_RISE = 0.01
# Seconds the timed event may miss its instant, by level. A5's lift may only be late; A6's bath
# exit may be early or late by this much. (A6 used to allow one 120 Hz physics tick, 8 ms, which
# only a controller reading the liquid surface from simulator state can meet.)
# These are benchmark settings with no chemical or biological motivation: A5's stop bath only
# has a minimum time (Ilford: "The time in the stop bath is not critical. It must be at least
# 10 seconds."), and published heat-shock efficiencies are insensitive to deviations of many
# seconds (Chan et al., Biosci Rep 2013).
# Every level allows at least 1 s (2026-11-17; was 0.25 / 0.15 / 0.08 s): a sub-second limit
# approaches human reaction time, so a miss could not separate a weak policy from an unreachable
# constraint. Medium is sqrt(easy * hard).
TOLERANCE_S = {"easy": 4.0, "medium": 2.0, "hard": 1.0}
# The sample must be immersed in the first bath within this long of the episode start.
LOADING_LIMIT_S = 15.0
# The sample must rest released in the second bath within this long of the timed event (A5: the
# start of the lift; A6: the warm-bath exit). A5 used to count from the countdown's end, which
# left a lift at the end of a multi-second late window too little time for the transfer.
TRANSFER_LIMIT_S = 5.0
# Seconds the episode runs past the latest valid release (at the most lenient level).
EPISODE_SLACK_S = 3.0


class BathTransfer(LabTask):
    laboratory_v2 = True
    bath_task = True
    category = "temporal_window"
    discipline = "laboratory"

    def __init__(self, spec, parameters=None):
        super().__init__(spec)
        parameters = parameters or {}
        self.duration = float(
            parameters.get("duration_s", 12.0 if self.task_id == "A5" else 30.0)
        )
        # The level's tolerance, unless the configuration names one. The override is what the
        # timing-only ablation varies (docs/ablations.md): it holds the level - and with it the
        # position and orientation tolerances and every other physical draw - fixed while the
        # timed instant's tolerance moves. The benchmark's own configs never set it.
        self.tolerance = float(parameters.get("tolerance_s", TOLERANCE_S[spec.difficulty]))
        if not 0 < self.tolerance <= 10:
            raise ValueError(f"{self.task_id} tolerance_s must be in (0, 10] s")
        self.spec = replace(
            spec,
            valid_start=self.duration - self.early_tolerance,
            valid_end=self.duration + self.tolerance,
            event_time=self.duration,
            num_samples=1,
            max_time=max(
                self.duration + (25 if self.task_id == "A5" else 12),
                LOADING_LIMIT_S + self.after_deadline,
            ),
        )
        self.grasp_seen = False
        self.failed = False
        self.exit_time = None
        self.lift_time = None
        self.lowest_bottom = float("inf")
        self.immersion_time = None
        self.release_time = None
        self.retention = 0.0
        self.controls = {}
        # Both samples start dry on the bench stand; the countdown starts at immersion.
        self.source_entry_time = None

    @property
    def deadline(self):
        return None if self.source_entry_time is None else self.source_entry_time + self.duration

    @property
    def after_deadline(self):
        """Episode time from the countdown's start: latest event, transfer, slack. It uses the
        largest level tolerance, so the episode length is the same at every level."""
        tolerance = max(self.tolerance, *TOLERANCE_S.values())
        return self.duration + tolerance + TRANSFER_LIMIT_S + EPISODE_SLACK_S

    @property
    def early_tolerance(self):
        # The film stays in the stop bath for the whole countdown, so A5 can only be late.
        return 0.0 if self.task_id == "A5" else self.tolerance

    def process(self, dt):
        return [SceneCommand("bath_visual", "equipment", self.visual_state())]

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        self.controls = dict(state.controls)
        self.grasp_seen |= obj.grasped
        if self.failed:
            return
        if self.source_entry_time is None:
            if self.controls.get("source_fully_immersed", False) and obj.grasped:
                self.source_entry_time = self.time
                self.spec = replace(
                    self.spec,
                    valid_start=self.deadline - self.early_tolerance,
                    valid_end=self.deadline + self.tolerance,
                    event_time=self.deadline,
                    max_time=self.source_entry_time + self.after_deadline,
                )
                self.event_log.append({"time": self.time, "type": f"{self.first_bath}_immersion_started"})
            elif self.time > LOADING_LIMIT_S:
                self.physical_failures.append(f"{self.first_bath}_loading_timeout")
                self.failed = True
            return
        if self.task_id == "A5" and self.lift_time is None and self.film_lifted():
            self.lift_time = self.time
            self.event_log.append({"time": self.time, "type": "developer_lift_started"})
            self.check_timing(self.lift_time)
        outside = not self.controls.get("source_immersed", True)
        if self.exit_time is None and outside:
            self.exit_time = self.time
            self.event_log.append({"time": self.time, "type": "sample_exited_source_bath"})
            if self.task_id == "A6":
                self.check_timing(self.exit_time)
            if not self.grasp_seen:
                self.physical_failures.append("uncontrolled_removal")
        if self.action_time is None and self.time > self.deadline + self.tolerance + 1e-9:
            self.violation(
                "overdevelopment" if self.task_id == "A5" else "heat_exposure_exceeded",
                self.time - self.deadline,
            )
            self.failed = True
        if self.exit_time is not None:
            if self.controls.get("destination_immersed", False) and self.immersion_time is None:
                self.immersion_time = self.time
                self.event_log.append({"time": self.time, "type": "destination_immersion"})
            supported = "destination_floor" in obj.fixture_contacts
            released = not obj.grasped and state.gripper_open
            valid = (
                self.controls.get("destination_immersed", False)
                and supported
                and released
                and np.linalg.norm(obj.velocity) < 0.04
            )
            self.retention = self.retention + dt if valid else 0.0
            if valid and self.release_time is None:
                self.release_time = self.time
            if self.retention >= 0.25 - 1e-9:
                self.goal_achieved = True
                self.correct.add("sample_0")
                self.attempted.add("sample_0")
            if (
                self.time - self.transfer_start > TRANSFER_LIMIT_S
                and not self.goal_achieved
            ):
                self.physical_failures.append("transfer_timeout")
                self.failed = True

    @property
    def first_bath(self):
        """Event-name prefix of the timed bath (A5 keeps its earlier "developer" spelling)."""
        return "developer" if self.task_id == "A5" else "warm_bath"

    @property
    def transfer_start(self):
        """When TRANSFER_LIMIT_S starts: A5's lift (set before the exit), A6's bath exit."""
        if self.task_id == "A5":
            return self.lift_time if self.lift_time is not None else self.deadline
        return self.exit_time

    def film_lifted(self):
        """True once the reel rises from where it sat in the stop bath, or the film emerges."""
        bottom = self.controls.get("sample_bottom_z")
        if bottom is not None:
            self.lowest_bottom = min(self.lowest_bottom, bottom)
            if bottom > self.lowest_bottom + LIFT_START_RISE:
                return True
        return not self.controls.get("source_fully_immersed", True)

    def check_timing(self, timestamp):
        self.event_time = self.deadline
        self.record_action(timestamp)
        error = timestamp - self.deadline
        if error < -self.early_tolerance - 1e-9:
            self.violation("early_transfer", -error)
        elif error > self.tolerance + 1e-9:
            self.violation("late_transfer", error)

    def visual_state(self):
        return dict(
            time=self.time,
            duration=self.duration,
            remaining=self.duration
            if self.deadline is None
            else max(0.0, self.deadline - self.time),
            phase="failed"
            if self.failed or self.violations
            else "complete"
            if self.goal_achieved
            else "transfer"
            if self.exit_time is not None
            else "loading"
            if self.source_entry_time is None
            else "timing",
            source_entry_time=self.source_entry_time,
            deadline=self.deadline,
            source_exit_time=self.exit_time,
            destination_immersion_time=self.immersion_time,
            grasp_seen=self.grasp_seen,
            controls=self.controls,
        )

    def metrics(self):
        return dict(
            super().metrics(),
            prescribed_duration_s=self.duration,
            timing_tolerance_s=self.tolerance,
            timing_event="developer_lift_start" if self.task_id == "A5" else "warm_bath_exit",
            source_entry_time=self.source_entry_time,
            timing_deadline=self.deadline,
            development_interval_s=(
                self.lift_time - self.source_entry_time if self.lift_time is not None else None
            ),
            developer_lift_time=self.lift_time,
            source_exit_time=self.exit_time,
            destination_immersion_time=self.immersion_time,
            destination_release_time=self.release_time,
            destination_retention_s=self.retention,
            grasp_observed=self.grasp_seen,
            final_controls=self.controls,
            process_model="Rigid-body transfer; bath immersion is geometric. No chemical or biological response is simulated.",
            # The episode's seeded station placement relative to the rig (station_frame).
            apparatus_placement=station_of(self).info(),
        )

    @property
    def terminated(self):
        return super().terminated or self.failed


class FilmDevelopment(BathTransfer):
    task_id = "A5"
    name = "photographic_film_development"
    asset = "film_carrier"
    # The timed step is the stop bath that follows development, then the transfer to the fixer.
    # 12 s is a real stop-bath time: Ilford's "Processing your first black & white film" gives
    # ILFOSTOP 10 seconds ("It must be at least 10 seconds"), Kodak's ED-BWF "Rinse/stop bath
    # 0:30". Development itself takes at least 5 minutes and is not part of this task. The event
    # names, metrics and violation keys below (developer_*, development_interval_s,
    # overdevelopment) keep their earlier spelling for result compatibility: they now refer to
    # the first, timed bath, which is the stop bath.
    instruction = "Pick up the developed film reel by its center spindle from the stand, immerse it in STOP BATH to start the countdown, and keep it there for the full 12 seconds. Only when the countdown has ended, lift it out, transfer it into FIXER and release it onto the support. The interval runs from full stop-bath immersion to the start of the lift; the lift limit is a benchmark setting, not a chemical one."

    def gen_instruction(self):
        # Kept short: the observation's instruction holds 512 characters.
        return (
            f"{super().gen_instruction()} Limits: immersed by {LOADING_LIMIT_S:g} s; lift begun "
            f"within {self.tolerance:.2f} s after the countdown ends; released within "
            f"{TRANSFER_LIMIT_S:g} s of the lift."
        )


class HeatShockTransfer(BathTransfer):
    task_id = "A6"
    name = "heat_shock_ice_transfer"
    asset = "microtube"
    # The 30 s heat shock follows the Thermo Fisher One Shot TOP10 user guide (MAN0000633):
    # "Incubate for exactly 30 seconds in the 42°C water bath. Do not mix or shake." The exit
    # tolerance (TOLERANCE_S) is a benchmark setting, not a biological one.
    # The countdown display (bath_scene.countdown_display) shows the seconds left: say so.
    # The tube starts on the bench stand (2026-11-19; it used to start immersed), and the baths
    # are unlabelled: the instruction tells them apart by their contents.
    instruction = "Heat-shock the capped tube as in the Thermo Fisher One Shot TOP10 protocol: pick it up from the bench stand, immerse it in the warm water bath to start the countdown and keep it there for 30 seconds, then transfer it into the ice bucket and release it onto the support. The display behind the baths counts the 30 seconds down to 00.0."

    def gen_instruction(self):
        low, high = self.duration - self.tolerance, self.duration + self.tolerance
        return (
            f"{super().gen_instruction()} Limits: immersed by {LOADING_LIMIT_S:g} s; the tube must "
            f"leave the warm bath between {low:.2f} and {high:.2f} seconds after immersion, and "
            f"rest released in the ice bucket within {TRANSFER_LIMIT_S:g} seconds of leaving."
        )
