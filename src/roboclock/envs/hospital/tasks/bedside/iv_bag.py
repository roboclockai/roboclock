"""A4 (B6 until 2026-11-19): start the new infusion. 1) spike and hang the prepared saline bag, 2) press pump START.

The instruction is issued right after the scene settles; the robot starts at rest, tucked near its
own base. Step 1 is the existing fixture-assisted insertion and released hanging (aligned port
entry, at least 18 mm insertion, real hook contact, released support verified for 200 ms). Step 2
is one physical press of the pump's START key (at least 4 mm of its 6 mm stroke with robot contact,
held 40 ms). START counts only once step 1 is retained; a START press before that is a
wrong-order failure. The response limit runs from the instruction to the end of the START press,
and the bag must still hang when START completes. No drug, rate or flow is simulated.
"""

import numpy as np

from roboclock.envs.hospital.bedside.iv_bag import (
    BAG,
    INSERT_DEPTH,
    INSTRUCTION,
    START_HOLD,
    START_KEY,
    START_PRESS,
    START_STROKE,
    geometry,
    start_key_state,
)
from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask


class IVBagSpikingHanging(HospitalTask):
    """Hang and start an infusion (the class keeps its historical name for importers)."""

    task_id = "A4"
    name = "hang_and_start_infusion"
    category = "temporal_window"  # A class since 2026-11-19 (was B6)
    discipline = "infusion"
    bedside_task = "iv_bag"
    bedside_module = "iv_bag"
    bag = BAG
    start_key = START_KEY
    hang_dwell = 0.20
    start_hold = START_HOLD

    def __init__(self, spec):
        super().__init__(spec)
        self.entry_time = self.insertion_time = self.hanging_time = None
        self.start_time = None  # end of the START press: the pump accepts START
        self.pressed_for = 0.0
        self._prior_measurement = None
        self._hanging_since = None
        self._retained = False
        self._hung = False
        self.device_state = {"bag_hung": False, "pump_running": False}

    def gen_instruction(self):
        return (
            f"{INSTRUCTION}. The uncapped saline bag waits in its cradle on the prep cart; you "
            "start at rest. After the green request, 1) lower the bag's trainer port fully over "
            "the IV pole's spike and leave its eyelet released on the hook, then 2) press START "
            "on the infusion pump below it, completing both within "
            f"{self.spec.response_limit:.1f} seconds. START before the bag hangs is a wrong-order "
            "failure. This rigid trainer models insertion, hanging and one key press, not membrane "
            "puncture, fluid flow or an infusion rate."
        )

    def process(self, dt):
        if self.event_time is None and self.time + EPS >= self.spec.event_time:
            self.event_time = self.time
            self.event_log.append({"time": self.event_time, "type": "start_infusion_request"})
        if self.event_time is None:
            return [SceneCommand("indicator", "indicator", BLUE)]
        deadline = self.event_time + self.spec.response_limit
        if self.time > deadline + EPS and self.start_time is None:
            self.violation("late_completion", self.time - deadline)
        color = RED if self.violations else GREEN
        commands = [SceneCommand("indicator", "indicator", color)]
        if self.start_time is None:
            commands += self.timer("indicator", (deadline - self.time) / self.spec.response_limit)
        return commands

    def evaluate(self, state, dt):
        obj = state.objects[BAG]
        measured = geometry(self.layout, obj)
        prior = self._prior_measurement
        # The robot starts at rest: any bag contact or motion before the request is early.
        if self.event_time is None and (
            obj.robot_contact
            or np.linalg.norm(obj.position - self.layout.objects[BAG].position) > 0.008
        ):
            self.violation("early_action", max(0.0, self.spec.event_time - self.time))
        if prior is not None:
            crossed = prior["depth"] <= 0.0 < measured["depth"]
            if crossed and measured["lateral_error"] < 0.020:
                self.attempted.add("spike")
                if not measured["aligned"]:
                    self.physical_failure("misaligned_spike_entry")
                elif obj.grasped:
                    self.entry_time = self.time
                    self.event_log.append({"time": self.time, "type": "spike_entered_port"})
                    if self.event_time is None:
                        self.violation("early_action", max(0.0, self.spec.event_time - self.time))
        if (
            self.entry_time is not None
            and self.insertion_time is None
            and measured["inserted"]
            and obj.grasped
        ):
            self.insertion_time = self.time
            self.correct.add("spike")
            self.event_log.append({"time": self.time, "type": "spike_fully_inserted"})
        if measured["hanging"] and self.insertion_time is None:
            self.physical_failure("hanging_before_insertion")
        retained = (
            measured["hanging"]
            and measured["inserted"]
            and self.insertion_time is not None
            and self.time > self.insertion_time + EPS
        )
        self._retained = False
        if retained:
            if self._hanging_since is None:
                self._hanging_since = self.time
            self.dwell["hang"] = self.dwell.get("hang", 0.0) + dt
            if self.dwell["hang"] + EPS >= self.hang_dwell:
                # Step 1 is scored at its first released support, verified by the dwell.
                self._retained = True
                if self.hanging_time is None:
                    self.hanging_time = self._hanging_since
                    self.event_log.append({"time": self.hanging_time, "type": "bag_hung"})
                    if self.event_time is None or self.hanging_time < self.event_time - EPS:
                        self.violation("early_action")
                self.correct.add("hang")
                self.attempted.add("hang")
        else:
            self._hanging_since = None
            self.dwell["hang"] = 0.0
        # After its verification, the bag still counts as hung while it rests released on the
        # hook with the spike fully in its port (the port's walls keep the spike inside; the strict
        # entry envelope above flickers while the hanging pouch settles).
        self._hung = (
            self.hanging_time is not None
            and measured["hanging"]
            and measured["depth"] >= INSERT_DEPTH - 0.003
        )
        self.device_state["bag_hung"] = self._hung
        if self.insertion_time is not None and measured["depth"] < -0.002:
            self.physical_failure("spike_disconnected")
        self._prior_measurement = measured
        self._evaluate_start(state, dt)
        self.goal_achieved = self._hung and self.start_time is not None

    def _evaluate_start(self, state, dt):
        """Step 2: one physical START press (B6's key rule), valid only after step 1 is retained."""
        key = state.objects.get(START_KEY)
        if key is None:
            return
        measured = start_key_state(self.layout, key)
        travel, contact = measured["depression_m"], measured["contact"]
        if not np.isfinite(travel):
            return
        if self.event_time is None and contact and travel > 0.0015:
            self.violation("early_action", max(0.0, self.spec.event_time - self.time))
        pressed = contact and measured["aligned"] and travel >= START_PRESS
        if self.start_time is not None:
            return
        if pressed:
            self.attempted.add(START_KEY)
            if self.hanging_time is None:
                # Starting the pump before the bag is spiked and hung (step 1 verified): wrong order.
                self.physical_failure("start_before_hang")
        valid = (
            pressed
            and self.event_time is not None
            and self._hung
            and not self.physical_failures
            and travel <= START_STROKE + 0.0006
        )
        self.pressed_for = self.pressed_for + dt if valid else 0.0
        if self.pressed_for + EPS >= self.start_hold:
            self.start_time = self.time
            self.record_action(self.start_time)
            self.correct.add(START_KEY)
            self.device_state["pump_running"] = True
            self.event_log.append({"time": self.time, "type": "pump_started"})
            if self.time > self.event_time + self.spec.response_limit + EPS:
                self.violation(
                    "late_completion", self.time - self.event_time - self.spec.response_limit
                )

    @property
    def temporal_success(self):
        return self.start_time is not None and not self.violations

    def metrics(self):
        result = super().metrics()
        result.update(
            {
                "instruction": INSTRUCTION,
                "spike_entry_time": self.entry_time,
                "insertion_time": self.insertion_time,
                "hanging_time": self.hanging_time,
                "start_time": self.start_time,
                "response_limit": self.spec.response_limit,
                "insertion_depth_required": INSERT_DEPTH,
                "start_press_required_m": START_PRESS,
                "start_hold_required_s": self.start_hold,
                "hang_latency": None
                if self.hanging_time is None or self.event_time is None
                else self.hanging_time - self.event_time,
                "completion_latency": None
                if self.start_time is None or self.event_time is None
                else self.start_time - self.event_time,
                "device_state": dict(self.device_state),
                "physical_model": "fixture_assisted_rigid_port_insertion_peg_support_and_pump_key",
            }
        )
        return result


HangAndStartInfusion = IVBagSpikingHanging
