"""Observable liquid timing tasks, driven by measured fixture motion.

Volumes and chemistry are lumped process models, not particle/CFD simulation.
Robot actions only affect them through PhysX valve/rack measurements.
"""

from dataclasses import replace

import numpy as np

from roboclock.envs.lab.mdp.events import SceneCommand
from roboclock.envs.lab.station_frame import station_of
from roboclock.envs.lab.tasks.base import WindowTask


class LiquidTask(WindowTask):
    liquid_process = True
    discipline = "analytical_chemistry"
    asset = "container"
    completion_dwell = 0.4
    amplitude = 0.0  # Legacy geometry helpers must not introduce a moving slot.

    def __init__(self, spec):
        super().__init__(spec)
        self.opening = 1.0
        self.contact_seen = False
        self.stopped_at = None
        self.volume = 0.0
        self.flow = 0.0
        self.process_failure = False
        self.final_dwell = 0.0

    def observe(self, state, dt):
        self.previous = state
        self.opening = float(np.clip(state.controls.get("opening", 1.0), 0, 1))
        self.contact_seen |= state.controls.get("robot_contact", False)
        if self.opening <= 0.04 and self.stopped_at is None:
            self.stopped_at = self.time
            self.record_action()
            self.attempted.add("valve")
            if not self.contact_seen:
                self.physical_failures.append("valve_closed_without_robot_contact")
            self.check_stop()
        if self.stopped_at is not None and self.opening > 0.04:
            self.final_dwell = 0.0
        if self.stopped_at is not None and self.opening <= 0.04:
            self.final_dwell += dt
            if self.final_dwell >= self.completion_dwell and not self.physical_failures:
                self.goal_achieved = self.valid_amount()
                if self.goal_achieved:
                    self.correct.add("valve")
        if self.physical_success and self.completion_time is None:
            self.completion_time = self.time

    def fail(self, kind):
        if not self.process_failure:
            self.event_log.append({"time": self.time, "type": kind})
            self.physical_failures.append(kind)
        self.process_failure = True
        self.violation(kind)

    def process(self, dt):
        self.update_process(dt)
        return (
            [SceneCommand("liquid_visual", "process", self.visual_state())]
            if self.steps % 4 == 0
            else []
        )

    def check_stop(self):
        # Deadlines follow delivered volume; throttling can extend the nominal clock.
        if self.volume < self.minimum_volume - 1e-6:
            self.violation("early_action", (self.minimum_volume - self.volume) / self.nominal_flow)
        elif self.volume >= self.maximum_volume - 1e-7:
            self.violation("late_action", (self.volume - self.maximum_volume) / self.nominal_flow)
        if not self.valid_amount():
            self.physical_failures.append("incorrect_collected_amount")

    def visual_state(self):
        return {"volume_ml": self.volume, "opening": self.opening, "flow_ml_s": self.flow}

    @property
    def terminated(self):
        return (
            self.physical_success
            or self.process_failure
            or self.final_dwell >= self.completion_dwell + 0.4
        )

    def metrics(self):
        result = super().metrics()
        result.update(
            process_model="lumped_volume_and_color; no CFD",
            volume_ml=self.volume,
            flow_ml_s=self.flow,
            valve_opening=self.opening,
            robot_contact_observed=bool(self.contact_seen),
        )
        # The episode's seeded station placement relative to the rig (station_frame).
        result["apparatus_placement"] = station_of(self).info()
        return result


class WaterCollection(LiquidTask):
    task_id = "A1"
    name = "water_collection"
    completion_dwell = 1.8
    instruction = (
        "Fill the 500 mL beaker from the low-flow metering faucet at the laboratory sink. "
        "Close the handle once the water is between the lower bound and the 500 mL "
        "graduation, before it overflows."
    )

    def __init__(self, spec, parameters=None):
        # One faucet for every difficulty. It is a low-flow metering outlet, not an ordinary
        # laboratory faucet: with the handle fully open it gives a gentle ~3 mm stream
        # (10 mL/s = 0.6 L/min), where the lowest-flow laboratory outlet found delivers
        # 2.8 L/min (Chicago Faucets E7FCJKCP, flow-controlled nozzle) and retailer figures
        # for other laboratory faucets are higher still. Difficulty narrows the accepted
        # volume band, as beaker graduations are only about +/-5 % accurate.
        parameters = parameters or {}
        flow = float(parameters.get("flow_ml_s", 10.0))
        minimum = float(
            parameters.get("minimum_volume_ml", {"easy": 440.0, "medium": 460.0, "hard": 480.0})[
                spec.difficulty
            ]
        )
        initial = float(parameters.get("initial_volume_ml", 0.0))
        if not np.isfinite(flow) or flow <= 0:
            raise ValueError("flow_ml_s must be finite and positive")
        if not (np.isfinite(initial) and np.isfinite(minimum) and 0 <= initial < minimum < 500):
            raise ValueError("Require 0 <= initial_volume_ml < minimum_volume_ml < 500")
        self.nominal_flow = flow
        start = (minimum - initial) / flow
        end = (500 - initial) / flow
        super().__init__(
            replace(
                spec,
                valid_start=start,
                valid_end=end,
                max_time=max(spec.max_time, end + 20.0),
            )
        )
        self.opening = 0.0
        self.opened_at = None
        self.fully_open_at = None
        self.volume = initial
        self.initial_volume = initial
        self.minimum_volume, self.maximum_volume = minimum, 500.0
        # The handle is the round knob on the outlet over the beaker; wet_bench.build_station
        # starts it closed at +CLOSE_ANGLE, so it opens clockwise seen from above.
        self.instruction = (
            f"Open the closed low-flow metering faucet at the laboratory sink and fill the 500 mL "
            f"beaker. The handle is "
            f"the round knob on top of the outlet above the beaker: it opens with a quarter turn "
            f"clockwise, seen from above, and closes by turning it back. "
            f"Close the handle once the water is between {minimum:g} mL and the 500 mL "
            f"graduation, before overflow."
        )

    def observe(self, state, dt):
        opening = float(np.clip(state.controls.get("opening", 0.0), 0, 1))
        if self.opened_at is None:
            self.previous = state
            self.opening = opening
            self.contact_seen |= state.controls.get("robot_contact", False)
            if opening <= 0.04:
                return  # An initially closed faucet is not a closure attempt.
            self.opened_at = self.time
            self.event_log.append({"time": self.time, "type": "faucet_opened"})
            if not self.contact_seen:
                self.physical_failures.append("valve_opened_without_robot_contact")
        if opening >= 0.96 and self.fully_open_at is None:
            self.fully_open_at = self.time
            self.event_log.append({"time": self.time, "type": "faucet_fully_open"})
        stopped = self.stopped_at
        super().observe(state, dt)
        if stopped is None and self.stopped_at is not None:
            self.event_log.append({"time": self.time, "type": "faucet_closed"})

    def metrics(self):
        return dict(
            super().metrics(),
            faucet_opened_at=self.opened_at,
            faucet_fully_open_at=self.fully_open_at,
            faucet_closed_at=self.stopped_at,
            initial_volume_ml=self.initial_volume,
            minimum_volume_ml=self.minimum_volume,
            nominal_flow_ml_s=self.nominal_flow,
        )

    def update_process(self, dt):
        self.flow = self.nominal_flow * (self.opening if self.opening > 0.04 else 0.0)
        self.volume += self.flow * dt
        if self.volume >= 500.0 - 1e-7:
            self.fail("overflow")

    def valid_amount(self):
        return self.minimum_volume - 1e-6 <= self.volume < self.maximum_volume - 1e-7


class PhaseSeparation(LiquidTask):
    task_id = "A2"
    name = "phase_separation"
    # The valve is not on the glassware: liquid_scene.valve() builds it on a pedestal beside the
    # stand (PIVOTS), piped toward the funnel. The instruction says which control is meant and
    # which way it closes: toward the CLOSE plate at the end of the lever's travel.
    instruction = (
        "Drain the blue lower phase from the separatory funnel into the receiving beaker. "
        "The stopcock is the steel quarter-turn lever valve with the black grip, on the pedestal "
        "to the right of the funnel and piped toward it; it starts open. Close it by swinging the "
        "lever a quarter turn toward the plate marked CLOSE, near the phase boundary, before the "
        "amber upper phase enters the beaker."
    )

    def __init__(self, spec, parameters=None):
        margin = float(
            (parameters or {}).get("margin_seconds", {"easy": 4.0, "medium": 2.5, "hard": 1.5})[
                spec.difficulty
            ]
        )
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("margin_seconds must be finite and positive")
        super().__init__(replace(spec, valid_start=24.0 - margin, valid_end=24.0))
        self.nominal_flow = 5.0
        self.lower_remaining = 120.0
        self.upper_volume = 100.0
        self.upper_received = 0.0
        self.failed_at = None
        self.minimum_volume = self.nominal_flow * self.spec.valid_start
        self.maximum_volume = 120.0

    # Once the lower phase is gone the failure is latched at that instant, but the episode runs
    # on for this long so the amber upper phase is seen running into the beaker (it used to end
    # on the same tick, with only blue in the beaker and the stream). Scoring is unchanged.
    failure_display = 1.5

    def update_process(self, dt):
        self.flow = self.nominal_flow * (self.opening if self.opening > 0.04 else 0.0)
        drained = self.flow * dt
        self.volume += drained
        lower = min(drained, max(self.lower_remaining, 0.0))
        self.lower_remaining -= lower
        upper = min(drained - lower, self.upper_volume)
        self.upper_volume -= upper
        self.upper_received += upper
        if self.lower_remaining <= 1e-7 and self.failed_at is None:
            self.failed_at = self.time
            self.fail("upper_phase_contamination")

    @property
    def terminated(self):
        if self.process_failure and self.failed_at is not None:
            return self.physical_success or self.time >= self.failed_at + self.failure_display - 1e-9
        return super().terminated

    def valid_amount(self):
        minimum = self.nominal_flow * self.spec.valid_start
        return minimum - 1e-6 <= self.volume < 120.0 - 1e-7

    def visual_state(self):
        return dict(
            super().visual_state(),
            lower_ml=max(0, self.lower_remaining),
            upper_ml=self.upper_volume,
            received_lower_ml=self.volume - self.upper_received,
            received_upper_ml=self.upper_received,
        )


class Chromatography(LiquidTask):
    task_id = "A3"
    name = "chromatography_fractions"
    # The vessels ride on one collector (liquid_scene.column_station): the instruction names it
    # and its handle, and says which tube takes which fraction (observe() credits fraction i only
    # in tube i). Without this a model grasped the fixed reagent bottles 110 times.
    instruction = (
        "Collect the blue, green, orange, and purple column fractions in the fresh tubes F1, F2, "
        "F3 and F4, in that order. The waste beaker and the tubes stand in a row on a sliding "
        "collector under the column's outlet; the collector slides along its rail when moved by "
        "its grip, the black knob with the yellow collar at the collector's near corner. "
        "Move a fresh tube under the outlet before each colored band elutes; "
        "return the waste beaker under the outlet after each fraction finishes."
    )
    colors = ((0.04, 0.43, 0.85), (0.06, 0.68, 0.25), (0.95, 0.35, 0.025), (0.57, 0.055, 0.72))

    def __init__(self, spec, parameters=None):
        gap = float(
            (parameters or {}).get("switch_seconds", {"easy": 20.0, "medium": 12.0, "hard": 5.0})[
                spec.difficulty
            ]
        )
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError("switch_seconds must be finite and positive")
        super().__init__(replace(spec, valid_start=12.0, valid_end=12.0 + 9.0))
        self.intervals = [
            (12.0 + i * (9 + gap), 21.0 + i * (9 + gap)) for i in range(len(self.colors))
        ]
        self.collection = np.zeros((len(self.colors) + 1, len(self.colors)))
        self.tube_volumes = np.zeros(len(self.colors) + 1)
        self.receiver = 0
        self.released = [False] * len(self.colors)
        self.completed = [False] * len(self.colors)
        self.transition_times = []
        self.last_receiver = 0
        self.flow = 1.5
        self.nominal_flow = 1.5

    def observe(self, state, dt):
        self.previous = state
        self.contact_seen |= state.controls.get("robot_contact", False)
        self.receiver = int(state.controls.get("receiver", -1))
        if self.receiver != self.last_receiver:
            self.transition_times.append({"time": self.time, "receiver": self.receiver})
            self.event_log.append(
                {"time": self.time, "type": "receiver_changed", "receiver": self.receiver}
            )
            self.last_receiver = self.receiver
        # Credit complete fractions only after removing their collection tube.
        for i, (start, end) in enumerate(self.intervals):
            if self.time > end and self.receiver != i + 1 and not self.released[i]:
                self.released[i] = True
                amount = self.collection[i + 1, i]
                contamination = float(self.collection[i + 1].sum() - amount)
                self.attempted.add(f"fraction_{i}")
                if amount >= 0.98 * (end - start) * self.flow and contamination < 0.05:
                    self.correct.add(f"fraction_{i}")
                else:
                    self.violation(f"fraction_{i + 1}_missed_or_mixed")
        self.goal_achieved = (
            len(self.correct) == len(self.colors) and self.contact_seen and self.receiver == 0
        )
        if self.goal_achieved:
            self.record_action()
            self.completion_time = self.time

    def update_process(self, dt):
        active = next((i for i, (s, e) in enumerate(self.intervals) if s <= self.time < e), None)
        if self.receiver >= 0:
            self.tube_volumes[self.receiver] += self.flow * dt
            if active is not None:
                self.collection[self.receiver, active] += self.flow * dt
        if active is not None and self.receiver != active + 1:
            self.violation(f"fraction_{active + 1}_collection_gap")
        if self.time > self.intervals[-1][1] + self.spec.valid_start and not self.goal_achieved:
            self.fail("fraction_collection_timeout")

    def visual_state(self):
        return {
            "time": self.time,
            "intervals": self.intervals,
            "receiver": self.receiver,
            "tube_volumes": self.tube_volumes.tolist(),
            "collection": self.collection.tolist(),
            "flow_ml_s": self.flow,
        }

    @property
    def terminated(self):
        return self.goal_achieved or self.process_failure

    def metrics(self):
        result = super().metrics()
        result.update(
            fraction_intervals=self.intervals,
            collection_ml=self.collection.tolist(),
            receiver_transitions=self.transition_times,
        )
        return result


class Titration(LiquidTask):
    task_id = "A4"
    name = "titration_endpoint"
    instruction = (
        "Deliver titrant from the burette into the stirred Erlenmeyer flask. "
        "The stopcock is the steel quarter-turn lever valve with the black grip, on the pedestal "
        "to the right of the flask and piped toward it; it starts open. Slow the flow by swinging "
        "the lever part of the way toward the plate marked CLOSE, or close it with a quarter "
        "turn, when the solution in the flask turns pale pink; stop before it becomes strongly "
        "pink from excess titrant."
    )

    def __init__(self, spec, parameters=None):
        margin = float(
            (parameters or {}).get("margin_seconds", {"easy": 6.0, "medium": 4.0, "hard": 2.0})[
                spec.difficulty
            ]
        )
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("margin_seconds must be finite and positive")
        # Full-open flow is 0.33 / 0.50 / 1.0 mL/s (easy / medium / hard). Easy and medium are
        # within a Class A 50 mL buret (ASTM E287 Table 2: at most 0.42-0.56 mL/s); hard's
        # 1.0 mL/s is within Class B / ISO 385 type II 50 mL buret delivery (ASTM E287 Table 3:
        # 50 mL in no less than 50 s).
        self.nominal_flow = 2.0 / margin
        start = 18.0 / self.nominal_flow
        super().__init__(replace(spec, valid_start=start, valid_end=start + margin))
        self.minimum_volume, self.maximum_volume = 18.0, 20.0

    def update_process(self, dt):
        self.flow = self.nominal_flow * (self.opening if self.opening > 0.04 else 0.0)
        self.volume += self.flow * dt
        if self.volume >= 20.0 - 1e-7:
            self.fail("excess_titrant")

    def valid_amount(self):
        return 18.0 - 1e-6 <= self.volume < 20.0 - 1e-7

    def visual_state(self):
        return dict(
            super().visual_state(),
            endpoint_fraction=float(np.clip((self.volume - 17.6) / 2.4, 0, 1)),
        )
