"""B2: catch the object that rolls down the board toward the counter edge, before it falls."""

from dataclasses import dataclass

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task

OBJECTS = ("cup", "egg", "pill_bottle", "apple")


@dataclass(frozen=True)
class B2Spec(X5Spec):
    max_time: float = 12.0
    release_time: float = 2.0     # hidden: when the peg holding the object retracts (s)
    target: float = 1.0           # hidden: floor() indexes OBJECTS, the one object that is let go
    ramp_deg: float = 6.0         # board slope at medium; multiplied by time_scale (4.5 / 6 / 9 deg)
    stop_speed: float = 0.03      # m/s: the object counts as stopped below this
    stop_hold_s: float = 0.2      # ... for this long
    hold_s: float = 1.0           # held clear of the counter this long = secured
    crush_limit_n: float = 30.0   # egg-class limit, applied to every object
    others_limit_m: float = 0.01  # the three objects that were not let go may move this far

    # Per-seed placement (sim.py Oracle.setup): the board with its four objects and pegs, as one unit, turned about the
    # middle of its thin end (TOE) and moved. The yaw turns the rolling direction too; the oracle's catch line stays.
    # One-sided ranges: the cup lane's catch pose is at the left arm's reach (nominal 0.30 rad joint margin); moving
    # the board toward -x or +y or turning it clockwise brings it closer to the left base. Within these ranges the
    # catch pose keeps >= ~0.12 rad (IK screen, 2026-11-20); with +-1.5 cm / +-5 deg it was unreachable.
    LAYOUT_JITTER = {"board": ((-0.005, 0.02), (-0.015, 0.008), (-1.5, 5.0))}

    @classmethod
    def hidden_ranges(cls):
        return {"release_time": (1.0, 3.0), "target": (0.0, len(OBJECTS)-1e-6)}

    @property
    def target_name(self):
        return OBJECTS[int(self.target)]

    @property
    def board_deg(self):
        return self.ramp_deg*self.time_scale


class RollingEdgeCatch(X5Task):
    task_id, name, category = "B2", "Catch the object rolling off the counter", "event_reactive"
    Spec = B2Spec
    # States what "caught" and "held" are judged as, the crush limit and the limit on the other objects.
    instruction = ("Something may roll off the counter. Catch it before it falls, and hold on to it: bring it to "
                   "rest before its centre passes the counter's edge on the robot's side, then lift it at least 1.5 "
                   "cm off the counter between both jaws of one gripper and keep it there for 1 second. Hold it "
                   "gently: more than 30 N on a jaw crushes it. The other objects must stay where they are (within "
                   "1 cm).")
    SIGNALS = {
        "released": ("the peg holding the target has retracted", False),
        "object_speed": ("target speed (m/s)", 0.0),
        "edge_distance": ("target centre to the counter edge along its heading (m); < 0 past the edge", 1.0),
        "object_fell": ("target below the counter top", False),
        "object_held": ("both jaws of one gripper (either arm) on the target and the target clear of the counter", False),
        "peak_contact_n": ("largest contact force of any jaw (either arm) on the target so far (N)", 0.0),
        "others_moved_m": ("largest displacement of the objects that were not let go (m)", 0.0),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.held, self.still = Hold(), Hold()
        self.min_edge = None

    def process(self, dt):
        s, spec = self.signals, self.spec
        if s["released"]:
            self.mark("released")
            self.min_edge = s["edge_distance"] if self.min_edge is None else min(self.min_edge, s["edge_distance"])
            if s["edge_distance"] < 0:
                self.mark("crossed_edge")
            rolling = "rolling" in self.events or s["object_speed"] > 2*spec.stop_speed
            if rolling:
                self.mark("rolling")
            slow = rolling and s["object_speed"] < spec.stop_speed
            if self.still.step(slow, dt) >= spec.stop_hold_s and s["edge_distance"] > 0:
                self.mark("stopped", edge_distance=s["edge_distance"])
        if self.held.step(s["object_held"], dt) >= spec.hold_s:
            self.mark("secured")
        if s["object_fell"]:
            self.failure("object_fell")
        if s["peak_contact_n"] > spec.crush_limit_n:
            self.failure("object_crushed")
        if s["others_moved_m"] > spec.others_limit_m:
            self.failure("wrong_object_disturbed")
        self.done = "secured" in self.events or bool(self.physical_failures)

    def cues(self):
        return {}

    @property
    def physical_success(self):
        return "secured" in self.events and not self.physical_failures

    @property
    def temporal_success(self):
        return "stopped" in self.events and "crossed_edge" not in self.events

    def task_metrics(self):
        released = self.events.get("released")
        stopped = self.events.get("stopped")
        return {"object": self.spec.target_name, "board_deg": self.spec.board_deg, "release_time": released,
                "stop_latency": None if stopped is None or released is None else stopped-released,
                "edge_margin_m": self.min_edge, "peak_contact_n": self.signals["peak_contact_n"]}


TASK = RollingEdgeCatch
