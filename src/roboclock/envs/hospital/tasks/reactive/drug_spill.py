"""B3: a bottle of a hazardous (cytotoxic) drug falls off the rack and shatters; clean up the spill.

The bottle stands on the counter rack until a seeded time, then tips off the shelf and falls (rigid-
body physics). When it first touches the counter it breaks: the scene replaces it with two glass
pieces and a puddle that keeps spreading (spill_scene.py). The response clock starts at the break.

Hazardous-drug spill procedure (USP <800> section 16, ASHP guidelines on handling hazardous drugs,
OSHA hazardous-drug guidance): contain the liquid with an absorbent pad first, then pick up broken
glass with a tool, never by hand, and discard it in a puncture-proof cytotoxic container.

  1. contain: the absorbent towel pad from the spill kit rests, released and flat, on the puddle.
     Until then the puddle grows; it reaches its full extent (PUDDLE_FULL) exactly at the
     containment limit, CONTAIN_SHARE of the response limit, as the CONTAIN dial runs out.
  2. recover: both glass pieces rest, released, inside the cytotoxic sharps bin (the robot's
     gripper is the tool). A piece gripped before the liquid is contained is a wrong-order failure.

t_contain - t_break < CONTAIN_SHARE * limit and t_done - t_break < limit, both strict.
"""

import math

import numpy as np

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.mdp.geometry import bottom_z, fits_xy
from roboclock.envs.hospital.tasks.base import EPS, ReactiveTask

SHOW_DELAY = 1.5  # seconds the scene keeps running after the last piece is in the bin
CONTAIN_SHARE = 0.4  # the liquid must be contained within this share of the response limit
FALL_TIMEOUT = 1.5  # the bottle counts as broken this long after it tipped, wherever it is
PUDDLE_START = 0.035  # m, puddle radius at the break
PUDDLE_FULL = 0.10  # m, its full extent, reached at the containment limit; it spreads no further
LANDING_RADIUS = 0.06  # m, a break measured this close to the planned landing point is used as is
PUDDLE_STEP = 0.0005  # m, radius change that is worth a scene update


def _seconds(limit: float) -> str:
    """A limit in seconds for the instruction, rounded down to 0.1 s so it never overstates."""
    return f"{math.floor(limit * 10 + 1e-9) / 10:g}"


class HazardousDrugSpill(ReactiveTask):
    task_id = "B3"
    name = "hazardous_drug_spill"
    discipline = "emergency"
    bottle, pad, shards, bin = "bottle", "pad", ("shard_0", "shard_1"), "hd_bin"
    consumables = (bottle,)  # the intact bottle no longer exists after the break

    def __init__(self, spec):
        super().__init__(spec)
        self.fall_time = None  # the bottle tips off the shelf
        self.spill_center = None  # xy of the break, measured
        self.contain_time = None
        self.shard_times = {}
        self.cleanup_time = None
        self._sent = set()
        self._radius_sent = None

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        spill = layout.info["spill"]
        self.landing = np.array(spill["center"], dtype=float)  # where the bottle is meant to land
        self.cover_radius = float(spill["cover_radius"])
        self.tip_velocity = [float(v) for v in spill["tip_velocity"]]
        self.tip_spin = [float(v) for v in spill["tip_spin"]]

    # -- limits and the puddle ----------------------------------------------------------------

    @property
    def containment_limit(self):
        return CONTAIN_SHARE * self.spec.response_limit

    @property
    def contained(self):
        return self.contain_time is not None

    def puddle_radius(self, time=None):
        """Radius of the liquid: linear from the break to its full extent at the containment
        limit, where it stops; frozen earlier once the pad is on it."""
        if self.event_time is None:
            return 0.0
        t = self.time if time is None else time
        if self.contain_time is not None:
            t = min(t, self.contain_time)
        rate = (PUDDLE_FULL - PUDDLE_START) / self.containment_limit
        return float(min(PUDDLE_START + rate * (t - self.event_time), PUDDLE_FULL))

    def gen_instruction(self):
        return (
            "A glass bottle of a hazardous (cytotoxic) drug falls off the rack and shatters on the "
            "counter. First lay the absorbent towel pad from the spill kit on the spreading liquid, "
            "before the CONTAIN dial runs out and the puddle reaches its full size; then pick up both "
            "glass pieces and drop them into the yellow cytotoxic sharps bin. Contain the liquid within "
            f"{_seconds(self.containment_limit)} seconds and finish within "
            f"{_seconds(self.spec.response_limit)} seconds of the break. Touch nothing before the "
            "bottle breaks."
        )

    # -- scene --------------------------------------------------------------------------------

    def _once(self, key):
        if key in self._sent:
            return False
        self._sent.add(key)
        return True

    def process(self, dt):
        commands = []
        if self.fall_time is None and self.time >= self.spec.event_time:
            self.fall_time = self.time
            self.event_log.append({"time": self.time, "type": "bottle_tips"})
            commands.append(SceneCommand("release", self.bottle, self.tip_velocity))
            commands.append(SceneCommand("spin", self.bottle, self.tip_spin))
        limit = self.spec.response_limit
        if self.event_time is not None:
            if self._once("break"):
                commands.append(
                    SceneCommand("spill", "break", [float(v) for v in self.spill_center])
                )
            elapsed = self.time - self.event_time
            if self.cleanup_time is None and elapsed + EPS >= limit:
                self.violation("late_cleanup", elapsed - limit)
            if self.contain_time is None and elapsed + EPS >= self.containment_limit:
                self.violation("late_containment", elapsed - self.containment_limit)
            radius = self.puddle_radius()
            if self._radius_sent is None or abs(radius - self._radius_sent) >= PUDDLE_STEP:
                self._radius_sent = radius
                commands.append(SceneCommand("spill", "puddle", radius))
            if self.contained and self._once("contain"):
                commands.append(SceneCommand("spill", "contain", radius))
        if self.event_time is None:
            light = contain_light = BLUE
        else:
            light = GREEN if self.cleanup_time is not None else RED
            contain_light = GREEN if self.contained else RED
            if self.cleanup_time is None:
                commands += self.timer(
                    "indicator", (self.event_time + limit - self.time) / limit
                )
            if not self.contained:
                commands += self.timer(
                    "contain",
                    (self.event_time + self.containment_limit - self.time) / self.containment_limit,
                )
        commands.append(SceneCommand("indicator", "indicator", light))
        commands.append(SceneCommand("indicator", "contain", contain_light))
        return commands

    # -- scoring ------------------------------------------------------------------------------

    def _dwell(self, key, condition, dt):
        self.dwell[key] = self.dwell.get(key, 0.0) + dt if condition else 0.0
        return self.dwell[key] + EPS >= self.placement_dwell

    def _at_rest(self, name, state):
        obj = state.objects[name]
        return not obj.grasped and float(np.linalg.norm(obj.velocity)) < 0.08

    def _in_bin(self, name, state):
        """Released and at rest inside the sharps bin: within its walls and below its rim. A piece
        dropped in may lean on a wall or on the other piece (1 cm of lean allowed, as in the ward
        bins), which is in the bin."""
        obj, shape = state.objects[name], self.layout.shapes[name]
        target = self.layout.targets[self.bin]
        return (
            self._at_rest(name, state)
            and name in self.moved
            and fits_xy(shape, obj, target, clearance=-0.01)
            and target.support_z - 0.003 <= bottom_z(shape, obj) < target.rim_z - 0.012
        )

    def _broke(self, state):
        """The falling bottle first touches the counter (or never arrives: it still breaks)."""
        if self.fall_time is None:
            return False
        obj = state.objects[self.bottle]
        low = bottom_z(self.layout.shapes[self.bottle], obj) <= self.layout.table_height + 0.008
        return low or self.time - self.fall_time >= FALL_TIMEOUT

    def evaluate(self, state, dt):
        movable = (self.pad, *self.shards)
        if self.event_time is None:
            if any(state.objects[n].grasped for n in (*movable, self.bottle)):
                self.violation("pre_event_action")
            if self._broke(state):
                self.event_time = self.time
                position = state.objects[self.bottle].position[:2]
                # A bottle that bounced off somewhere else still spills where it was meant to land.
                near = float(np.linalg.norm(position - self.landing)) < LANDING_RADIUS
                self.spill_center = np.array(position if near else self.landing, dtype=float)
                self.event_log.append(
                    {
                        "time": self.time,
                        "type": "event_triggered",
                        "spill_center": [round(float(v), 4) for v in self.spill_center],
                    }
                )
            return
        if not self.contained:
            if any(state.objects[n].grasped for n in self.shards):
                self.physical_failure("glass_before_containment")
            pad = state.objects[self.pad]
            height = self.layout.table_height + self.layout.shapes[self.pad].half[2]
            covering = (
                float(np.linalg.norm(pad.position[:2] - self.spill_center)) < self.cover_radius
                and abs(pad.position[2] - height) < 0.02
                and self._at_rest(self.pad, state)
                and self.pad in self.moved
                and self.tilt_ok(pad)
            )
            if self._dwell("contain", covering, dt):
                self.contain_time = self.time
                self.correct.add(self.pad)
                self.attempted.add(self.pad)
                if self.time - self.event_time + EPS >= self.containment_limit:
                    self.violation(
                        "late_containment", self.time - self.event_time - self.containment_limit
                    )
                self.event_log.append(
                    {
                        "time": self.time,
                        "type": "spill_contained",
                        "puddle_radius": round(self.puddle_radius(), 4),
                    }
                )
        for name in self.shards:
            if name in self.shard_times:
                continue
            if self._dwell(("bin", name), self._in_bin(name, state), dt):
                self.shard_times[name] = self.time
                self.correct.add(name)
                self.attempted.add(name)
                self.event_log.append({"time": self.time, "type": "glass_in_bin", "object": name})
        done = self.contained and len(self.shard_times) == len(self.shards)
        if done and self.cleanup_time is None:
            self.cleanup_time = self.time
            self.record_action(self.time)
            if self.time - self.event_time + EPS >= self.spec.response_limit:
                self.violation(
                    "late_cleanup", self.time - self.event_time - self.spec.response_limit
                )
            self.event_log.append({"time": self.time, "type": "spill_cleaned_up"})
        # Keep the episode running briefly so the soaked pad and the emptied counter are on screen.
        if self.cleanup_time is not None and self.time - self.cleanup_time >= SHOW_DELAY:
            self.goal_achieved = True

    def metrics(self):
        result = super().metrics()
        since = None if self.event_time is None else self.event_time
        result.update(
            fall_time=self.fall_time,
            break_time=self.event_time,
            contain_time=self.contain_time,
            shard_times=dict(self.shard_times),
            cleanup_time=self.cleanup_time,
            containment_limit=self.containment_limit,
            containment_latency=None
            if self.contain_time is None
            else self.contain_time - since,
            response_time=None if self.cleanup_time is None else self.cleanup_time - since,
            puddle_radius=self.puddle_radius(),
            puddle_full_radius=PUDDLE_FULL,
        )
        return result
