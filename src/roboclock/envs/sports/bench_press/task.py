"""Event onset, grasp, arrested fall and released rack support are separate."""
import numpy as np
from scipy.spatial.transform import Rotation

# Judged distances the instruction states (``roboclock.envs.sports.tasks``); metres.
# Before the bar falls, a jaw gap below STANDBY_GAP counts as an early grasp. A catch needs
# each hand within GRIP_TOLERANCE of its grip point (``bench_grip_y`` either side of the bar's
# centre), and the bar counts as released on the hooks with both jaw gaps above RELEASE_GAP.
STANDBY_GAP = 0.045
GRIP_TOLERANCE = 0.045
RELEASE_GAP = 0.065


class BenchPress:
    def __init__(self, b):
        self.b, self.c = b, b.config["workflow"]
        self.done = False
        self.phase = "STANDBY"
        self.events, self.failures = [], []
        self.event_time = self.response_time = self.catch_time = self.rack_time = None
        self.ascended = self.stalled = False
        self.ready_to_press = False
        self.stall_start = self.grip_start = self.rack_start = self.early_start = None
        self.previous_time = 0.
        self.standby_peak_error = 0.
        self.departure_start = self.precontact_start = self.motion_response_time = None
        self.min_height = float("inf")
        self.peak_arm_contact = 0.
        self.last_retained = 0.

    def event(self, name, **values):
        self.events.append(dict(time=self.b.time, event=name, **values))

    def fail(self, name):
        if name not in self.failures:
            self.failures.append(name)
            self.event(name)

    def update(self, s):
        t, c = float(s["time"]), self.c
        o = s["objects"]["bar"]
        p, v, f = np.asarray(o["position"]), np.asarray(o["velocity"]), np.asarray(o["forces"])
        dt = t - self.previous_time
        self.previous_time = t
        gap = np.asarray(s["finger_gaps"])
        self.peak_arm_contact = max(self.peak_arm_contact, s["arm_contact_peak"])
        if s["arm_contact_peak"] > 12.: self.fail("arm_collision")
        self.min_height = min(self.min_height, float(p[2]))
        hands = np.array([s["hands"][side]["position"] for side in ("left", "right")])
        bar_rotation = Rotation.from_quat(np.roll(o.get("orientation", [1., 0., 0., 0.]), -1))
        targets = p + bar_rotation.apply([[0., c["bench_grip_y"], 0.], [0., -c["bench_grip_y"], 0.]])
        errors = np.linalg.norm(hands - targets, axis=1)
        standby = np.asarray(c["standby_position"]) + np.array([[0.,c["bench_grip_y"],0.],[0.,-c["bench_grip_y"],0.]])
        standby_error = float(np.max(np.linalg.norm(hands-standby, axis=1)))
        if self.event_time is None:
            if t > 0.2:
                self.standby_peak_error = max(self.standby_peak_error, standby_error)
                # Leaving standby before the stall is allowed (standby_peak_error records it);
                # touching the bar or closing the jaws early still fails.
                if np.max(f[:4]) > .08:
                    if self.precontact_start is None: self.precontact_start = t
                    if t-self.precontact_start > .025: self.fail("premature_bar_contact")
                else: self.precontact_start = None
            if np.any(gap < STANDBY_GAP):
                if self.early_start is None: self.early_start = t
                if t-self.early_start > .08: self.fail("premature_grasp")
            else: self.early_start = None
            if p[2] >= c["bench_top"] - 0.022:
                self.ascended = self.stalled = False
                self.ready_to_press = False
                self.stall_start = None
            elif p[2] < c["bench_bottom"] + 0.035:
                self.ready_to_press = True
            if self.ready_to_press and t > 3. and v[2] > 0.025:
                self.ascended = True
            if self.ascended and abs(v[2]) < 0.018 and p[2] < c["bench_top"] - 0.035:
                if self.stall_start is None: self.stall_start = t
                if t - self.stall_start >= c["stall_dwell"] and not self.stalled:
                    self.stalled = True
                    self.event("press_stall", measured_height=float(p[2]))
            elif not self.stalled:
                self.stall_start = None
            if self.ascended and self.stalled and v[2] < -c["descent_threshold"]:
                self.event_time = t
                self.phase = "CATCH"
                self.event("exhaustion_descent", measured_height=float(p[2]), velocity_z=float(v[2]))
        else:
            if self.motion_response_time is None and standby_error > c["standby_tolerance"]:
                self.motion_response_time = t
                self.event("approach_response", latency=t-self.event_time)
            if self.response_time is None and np.all(gap < 0.065) and np.all(f[:4] > .08):
                self.response_time = t
                self.event("gripper_response", latency=t - self.event_time)
            retained = np.all(f[:4] > 0.08) and np.max(errors) < GRIP_TOLERANCE and np.all(gap < 0.05)
            if retained: self.last_retained = t
            arrested = retained and v[2] > -0.025
            if self.catch_time is None:
                if arrested:
                    if self.grip_start is None: self.grip_start = t
                    if t - self.grip_start >= c["catch_dwell"]:
                        self.catch_time = t
                        self.phase = "RERACK"
                        self.event("bilateral_catch", latency=t - self.event_time)
                        self.b.human_effort.released = True
                        self.b.human_effort.release_time = t
                else: self.grip_start = None
                if t - self.event_time > c["catch_deadline"]: self.fail("late_catch")
            support = o.get("support_contacts", {})
            supported = (support.get("RackLeft", 0.) > 0.2 and support.get("RackRight", 0.) > 0.2
                         and support.get("RackLeft", 0.) + support.get("RackRight", 0.) >= 0.85*c["bench_mass"]*9.81)
            if self.catch_time is not None and not supported and t-self.last_retained > 0.25:
                self.fail("lost_robot_grasp")
            seated = (supported
                      and abs(p[0] - c["rack_x"]) < 0.025 and abs(p[2] - c["rack_height"]) < 0.022
                      and np.linalg.norm(v) < 0.025 and np.all(gap > RELEASE_GAP) and np.max(f[:4]) < 0.04)
            if self.catch_time is not None and seated:
                if self.rack_start is None: self.rack_start = t
                if t - self.rack_start >= c["rack_dwell"]:
                    self.rack_time = t
                    self.phase, self.done = "COMPLETE", True
                    self.event("supported_rerack")
            else: self.rack_start = None
        if p[2] < c["human_danger_height"] or o.get("support_contacts", {}).get("HumanTorso", 0.) > 0.05:
            self.fail("human_safety_contact")
            self.phase, self.done = "FAILED", True
        if t >= c["bench_timeout"]:
            self.fail("episode_timeout")
            self.phase, self.done = "FAILED", True

    def display(self):
        if self.event_time is None:
            return ("STANDBY / WAIT FOR EXHAUSTION", "HANDS OFF THE BAR / WAIT FOR FALL")
        latency = (self.catch_time or self.b.time) - self.event_time
        return (f"{self.phase} / REACTION {latency:.2f}s", f"LIMIT {self.c['catch_deadline']:.2f}s / RETURN TO HOOKS")

    def result(self):
        physical = self.rack_time is not None and not any(x in self.failures for x in ("human_safety_contact", "arm_collision", "lost_robot_grasp"))
        temporal = self.event_time is not None and self.catch_time is not None and self.catch_time - self.event_time <= self.c["catch_deadline"]
        standby = not any(x in self.failures for x in ("premature_bar_contact", "premature_grasp"))
        return dict(task="B3", name="Bench Press Assist", primary_temporal_category="event_reactive",
                    success=physical and temporal and standby, physical_success=physical, temporal_success=temporal,
                    standby_success=standby, standby_peak_error=self.standby_peak_error,
                    motion_response_time=self.motion_response_time, exhaustion_time=self.event_time,
                    response_time=self.response_time, catch_time=self.catch_time, rack_time=self.rack_time,
                    reaction_latency=None if self.catch_time is None else self.catch_time-self.event_time,
                    catch_deadline=self.c["catch_deadline"], minimum_bar_height=self.min_height,
                    peak_arm_contact=self.peak_arm_contact, events=self.events, failures=self.failures,
                    controller="privileged scripted feedback baseline", human_model="bounded wrench with skinned visual retargeting")
