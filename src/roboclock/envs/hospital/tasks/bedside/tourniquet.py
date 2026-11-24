"""B6 (since 2026-11-20): massive limb haemorrhage; tighten the tourniquet and secure its windlass.

A bleeding-control training manikin lies on the trauma bed. A windlass tourniquet (band, windlass
rod, retaining clip) is already placed high on the injured thigh with its strap pulled tight, but
the windlass has not been turned. At a seeded time (`spec.event_time`, 4-10 s) the thigh wound
starts to bleed heavily: blood runs from the wound, a pool grows on the absorbent pad under the
leg and the trainer's display turns red (MASSIVE BLEEDING). The robot starts at rest, tucked near
its own base, and has to pinch the windlass rod, turn it one half-turn (180 degrees nominal; the
episode's seeded rod start and clip angles make it 164-196) so its marked tip ends under the
retaining clip, and let go.

What is measured and what is modelled:
- The rod is a rigid body on a revolute joint; only finger contact turns it. The apparatus module
  (bedside/tourniquet.py) reports its measured angle, angular speed and contact.
- The twisted strap unwinds a released rod: a sprung, damped joint drive pulls it back to its
  start. The retaining clip is a latch on that same drive, engaged by the measured angle (within
  CLIP_TOLERANCE below the half-turn): there the rod is held at the clip instead. The clip's hook
  geometry is not simulated.
- Bleeding is a process model of the measured angle (`bleed_rate`): BLEED_RATE with the windlass
  untwisted, falling linearly to zero at OCCLUDE_ANGLE, before the clip; the lost volume is its
  integral and the pool on the pad is drawn from it. No blood pressure, shock or tissue.

The tourniquet is "secured" when the rod sits in the clip, released, out of robot contact and at
rest for SECURE_DWELL; it is scored at the first instant of that dwell and the response limit runs
from the start of the bleeding to that instant. Touching or turning the rod before the bleeding
is an early action. Letting go of a part-turned rod outside the clip (it unwinds, the bleeding
resumes) and forcing it OVERTURN past the clip are physical failures.

Real-world references (fetched 2026-11-20):
- One 180-degree windlass turn occludes a correctly strapped Combat Application Tourniquet, and
  the rod must be secured in the clip at a rotation beyond that at which bleeding stops (Wall,
  Buising and Sahr, J Spec Oper Med 19(3):52, 2019, doi:10.55460/RYU9-YZSV).
- Experienced providers apply a CAT to a thigh bleeding trainer in 24 +- 4 s, the trainer losing
  99 +- 17 mL of simulated blood meanwhile, about 4 mL/s (Lechner et al., BMC Emerg Med 24:90,
  2024, doi:10.1186/s12873-024-01005-x): the easy limit and BLEED_RATE.

Until 2026-11-20 B6 was the pump-panel key press (tasks/bedside/pump_panel.py,
`pump_panel_response`), which left the suite; results recorded for it keep its name (results.py).
"""

import math

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask

ROD = "windlass_rod"
BLEED_RATE = 4.0  # mL/s with the windlass untwisted (trainer average: 99 mL in 24 s)
TURN_ANGLE = math.pi  # one windlass turn (nominal): the rod's marked tip under the retaining clip
OCCLUDE_BEFORE_CLIP = math.radians(30.0)  # the bleeding has stopped this far before the clip
OCCLUDE_ANGLE = TURN_ANGLE - OCCLUDE_BEFORE_CLIP  # nominal: 150 degrees
CLIP_TOLERANCE = math.radians(12.0)  # the clip takes the rod from this far below the half-turn
OVERTURN = math.radians(20.0)  # forced this far past the clip: over-tightened (the X5 wrist reaches 204 degrees)
MOTION_THRESHOLD = math.radians(5.0)  # rod turn that counts as moved
COMMIT_ANGLE = math.radians(30.0)  # a rod turned this far and let go outside the clip unwinds
RELEASE_GRACE = 0.10  # s without contact before a part-turned rod counts as let go
REST_SPEED = 0.5  # rad/s: "at rest" for the released rod in the clip
SECURE_DWELL = 0.30  # s the released rod must stay in the clip
HOLD_AFTER = 1.5  # s shown after the rod is secured: display green, pool no longer growing
POOL_DEPTH = 0.001  # m: the stain on the absorbent pad is drawn as a disc of this depth
INSTRUCTION = "MASSIVE BLEEDING: TURN THE WINDLASS HALF A TURN INTO ITS CLIP"


def bleed_rate(angle: float, occlude: float = OCCLUDE_ANGLE) -> float:
    """Blood loss (mL/s) for a windlass angle (rad from the untwisted start); zero from `occlude`."""
    if not math.isfinite(angle):
        return BLEED_RATE
    return BLEED_RATE * min(max(1.0 - angle / occlude, 0.0), 1.0)


def clip_engaged(angle: float, turn: float = TURN_ANGLE) -> bool:
    """The retaining clip holds a rod turned to within CLIP_TOLERANCE below the clip angle `turn`
    (the episode's turn from the rod's start to the clip; the nominal half-turn by default)."""
    return math.isfinite(angle) and angle >= turn - CLIP_TOLERANCE


def pool_radius(volume_ml: float) -> float:
    """Radius (m) of the drawn pool for a lost volume: a disc POOL_DEPTH deep."""
    return math.sqrt(max(volume_ml, 0.0) * 1e-6 / (math.pi * POOL_DEPTH))


class HaemorrhageTourniquet(HospitalTask):
    task_id = "B6"
    name = "haemorrhage_tourniquet"
    category = "event_reactive"
    discipline = "trauma_care"
    bedside_task = "tourniquet"
    bedside_module = "tourniquet"
    rod = ROD
    secure_dwell = SECURE_DWELL
    hold_after = HOLD_AFTER

    def __init__(self, spec):
        super().__init__(spec)
        # The episode's turn from the rod's start to the clip and where the bleeding stops; the
        # layout's seeded placement replaces these nominal values in `bind`.
        self.turn_required = TURN_ANGLE
        self.occlude_angle = OCCLUDE_ANGLE
        self.placement = {}
        self.angle = 0.0  # measured windlass angle
        self.max_angle = 0.0
        self.blood_lost = 0.0  # mL since the bleeding started
        self.rate = 0.0  # mL/s now
        self.turn_time = None  # first contact-driven turn after the bleeding started
        self.occlusion_time = None  # first instant the bleeding has stopped
        self.secure_time = None  # first released, retained instant in the clip
        self.blood_at_secure = None
        self._retained_since = None
        self._free_for = 0.0  # time the part-turned rod has been out of contact, outside the clip
        self.device_state = {"bleeding": False, "alarm_active": False, "windlass_secured": False}

    def bind(self, layout, placement_dwell: float = 0.2):
        super().bind(layout, placement_dwell)
        info = getattr(layout, "info", {}).get("tourniquet") or {}
        self.turn_required = float(info.get("turn_required_rad", TURN_ANGLE))
        self.occlude_angle = self.turn_required - OCCLUDE_BEFORE_CLIP
        self.placement = dict(info.get("placement") or {})

    def gen_instruction(self):
        return (
            "Trauma bay: a bleeding-control manikin has a windlass tourniquet strapped on its "
            "thigh, not yet tightened. When the thigh wound bleeds (blood pools on the pad, the "
            "display turns red: MASSIVE BLEEDING), pinch the windlass rod, turn it "
            "counterclockwise (seen from above) about half a turn until its red tip is in the retaining "
            f"clip, and let go, within {self.spec.response_limit:.1f} s of the start of the "
            "bleeding. Do not touch the rod before the bleeding, let go of it before the clip "
            "(it unwinds) or force it past the clip."
        )

    # --- simulation loop -------------------------------------------------------------------

    @property
    def bleeding(self) -> bool:
        return self.event_time is not None and self.rate > 0.0

    def process(self, dt):
        if self.event_time is None and self.time + EPS >= self.spec.event_time:
            self.event_time = self.time
            self.device_state.update(bleeding=True, alarm_active=True)
            self.event_log.append({"time": self.time, "type": "bleeding_started"})
        if self.event_time is None:
            return [SceneCommand("indicator", "indicator", BLUE)]
        deadline = self.event_time + self.spec.response_limit
        # Late once the deadline has passed without the rod released in the clip; a retention that
        # began in time and is still being verified by its dwell is not late.
        pending = self._retained_since is not None and self._retained_since <= deadline + EPS
        if self.time > deadline + EPS and self.secure_time is None and not pending:
            self.violation("late_response", self.time - deadline)
        done = self.secure_time is not None
        color = GREEN if done and not self.violations else RED
        commands = [SceneCommand("indicator", "indicator", color)]
        if not done:
            commands += self.timer("indicator", (deadline - self.time) / self.spec.response_limit)
        return commands

    def evaluate(self, state, dt):
        controls = state.controls.get("tourniquet") or {}
        angle = float(controls.get("angle_rad", 0.0))
        speed = float(controls.get("speed_rad_s", 0.0))
        contact = bool(controls.get("contact", False))
        if math.isfinite(angle):
            self.angle = angle
        self.max_angle = max(self.max_angle, self.angle)
        turned = abs(self.angle) > MOTION_THRESHOLD
        if self.event_time is None:
            # The robot starts at rest: any contact with or turn of the windlass is early.
            if contact or turned:
                self.violation("early_action", max(0.0, self.spec.event_time - self.time))
                self.attempted.add("windlass")
            return
        self.rate = bleed_rate(self.angle, self.occlude_angle)
        self.blood_lost += self.rate * dt
        if turned and contact and self.turn_time is None:
            self.turn_time = self.time
            self.attempted.add("windlass")
            self.event_log.append({"time": self.time, "type": "windlass_turned"})
        if self.occlusion_time is None and self.turn_time is not None and self.rate <= 0.0:
            self.occlusion_time = self.time
            self.event_log.append(
                {"time": self.time, "type": "bleeding_stopped", "blood_lost_ml": self.blood_lost}
            )
        if self.angle > self.turn_required + OVERTURN:
            self.physical_failure("over_tightened")
        in_clip = clip_engaged(self.angle, self.turn_required)
        # A part-turned rod let go outside the clip unwinds and the bleeding resumes.
        if self.max_angle >= COMMIT_ANGLE and not in_clip and not contact and self.secure_time is None:
            self._free_for += dt
            if self._free_for + EPS >= RELEASE_GRACE:
                self.physical_failure("released_unsecured")
        else:
            self._free_for = 0.0
        retained = in_clip and not contact and abs(speed) < REST_SPEED and self.turn_time is not None
        if retained:
            if self._retained_since is None:
                self._retained_since = self.time
                self.attempted.add("clip")
            self.dwell["clip"] = self.dwell.get("clip", 0.0) + dt
            if self.dwell["clip"] + EPS >= self.secure_dwell and self.secure_time is None:
                self.secure_time = self._retained_since
                self.blood_at_secure = self.blood_lost
                self.record_action(self.secure_time)
                self.correct.update(("windlass", "clip"))
                self.device_state.update(bleeding=False, alarm_active=False, windlass_secured=True)
                self.event_log.append(
                    {"time": self.secure_time, "type": "windlass_secured", "blood_lost_ml": self.blood_lost}
                )
                limit = self.event_time + self.spec.response_limit
                if self.secure_time > limit + EPS:
                    self.violation("late_response", self.secure_time - limit)
        else:
            self._retained_since = None
            self.dwell["clip"] = 0.0
        # Secured for as long as the rod stays in the clip.
        if self.secure_time is not None and not in_clip:
            self.physical_failure("rod_left_clip")
            self.device_state.update(bleeding=True, alarm_active=True, windlass_secured=False)
        self.goal_achieved = self.secure_time is not None and in_clip

    # --- outcomes ----------------------------------------------------------------------------

    @property
    def cue_observed(self):
        return self.event_time is not None

    @property
    def temporal_success(self):
        return self.secure_time is not None and not self.violations

    @property
    def terminated(self):
        if bool(self.dropped) or bool(self.physical_failures):
            return True
        if self.completion_time is not None:
            return self.time + EPS >= self.completion_time + self.hold_after
        return False

    def metrics(self):
        result = super().metrics()

        def after_event(value):
            return None if value is None or self.event_time is None else value - self.event_time

        result.update(
            {
                "instruction": INSTRUCTION,
                "bleed_start_time": self.event_time,
                "turn_time": self.turn_time,
                "occlusion_time": self.occlusion_time,
                "secure_time": self.secure_time,
                "response_limit": self.spec.response_limit,
                "response_latency": after_event(self.secure_time),
                "turn_latency": after_event(self.turn_time),
                "occlusion_latency": after_event(self.occlusion_time),
                "windlass_angle_rad": self.angle,
                "max_windlass_angle_rad": self.max_angle,
                "blood_lost_ml": self.blood_lost,
                "blood_lost_at_secure_ml": self.blood_at_secure,
                "bleed_rate_ml_s": BLEED_RATE,
                "turn_required_rad": self.turn_required,
                "occlusion_angle_rad": self.occlude_angle,
                "placement": dict(self.placement),
                "secure_dwell_required_s": self.secure_dwell,
                "device_state": dict(self.device_state),
                "physical_model": (
                    "contact-turned windlass rod on a sprung joint; clip = angle-engaged latch on "
                    "the joint drive; bleeding and pool are process models of the measured angle"
                ),
            }
        )
        return result
