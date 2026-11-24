"""V2 tool-use task contracts, distinct from the frozen v1 benchmark."""

from dataclasses import replace

from roboclock.envs.sports.common.config import load_config
from roboclock.envs.sports.common.layout import Body, Control
from roboclock.envs.sports.common.tasks import SportsTask, WhackMoles


def configuration(task_id, variant="basic", room_style="v5"):
    if room_style not in {"v5", "v6"}:
        raise ValueError("Unknown room style")
    c = load_config()
    c["scene"]["room_style"] = room_style
    c["sim"]["camera_resolution"] = [1280, 960]
    c["scene"]["spatial_jitter"] = 0.0
    c["scene"]["camera_position"] = [1.90, -2.35, 1.95]
    c["scene"]["camera_look_at"] = [0.46, 0.12, 1.02]
    c["control"]["transit_height"] = 1.16
    c["control"]["cartesian_speed"] = 0.30
    c["control"]["joint_target_speed"] = 1.8
    c["physics"]["button_dwell"] = 2 * c["sim"]["physics_dt"]
    c["v2"] = dict(
        variant=variant,
        tool_offset=0.125,
        tool_radius=0.032 if task_id == "B1" else 0.016,
        tool_mass=0.045 if task_id == "B1" else 0.055,
        revision="v6-daylight-rooms",
        grip_sleeve_size=[0.024, 0.026, 0.064],
        finger_force_limit=40.0 if task_id == "B1" else 20.0,
        finger_stiffness=4000.0 if task_id == "B1" else 2400.0,
        preparation=10.0,
        scene="garden_arcade" if task_id == "B1" else "percussion_studio",
        screen_offset=[0.38, 0.07, 0.30],
    )
    if room_style == "v5":
        c["scene"]["camera_position"] = [1.65, -1.7, 1.75]
        c["scene"]["camera_look_at"] = [0.40, 0.08, 0.92]
        c["v2"]["screen_offset"] = [0.0, 0.0, 0.0]
        c["v2"]["revision"] = "v5.1-chess-placement"
        c["v2"]["scene"] = "chess_club" if task_id == "A1" else "arcade"
    if task_id == "B1":
        c["tasks"]["34"].update(rounds=9, interval=4.0, max_time=51.0)
        # Mole time-up, measured from the rise (2026-11-16 recalibration; was 2.8 / 1.4 /
        # 1.10 s): the reference oracle, waiting above the mallet rests, hits a mole
        # 1.16-1.76 s after its rise (easy seeds 0-9, 60 moles; the schedule and motion
        # are the same at every tier). Hard = max(t / 0.8, t + 0.3) with t = 1.756 s,
        # rounded up to 0.1 s; medium = sqrt(easy * hard). Moles still rise every 4 s.
        # Easy since 2026-11-23 (user rule): no document gives an upper time and 4 x t = 7.0 s
        # would outlast the 4 s between moles, so easy is capped at that interval; medium =
        # sqrt(easy * hard) = 2.97 -> 3.0 s (was 2.8 / 2.5 s).
        c["difficulties"]["easy"]["deadline"] = 4.0
        c["difficulties"]["medium"]["deadline"] = 3.0
        c["difficulties"]["hard"]["deadline"] = 2.2
        # Mallet strike motion (2026-11-04; was 0.23 m/s, 1.7 rad/s joints, 2.3 rad/s
        # wrist pitch). No ``mole_idle_target``: between moles the arms wait where the
        # pickup ended, above the mallet rests (2026-11-15; they waited over the board at
        # [0.50, +/-0.12, 1.07] before, a pose the instruction does not ask for).
        c["v2"].update(
            mole_cartesian_speed=0.60,
            mole_joint_speed=3.5,
            mole_pitch_speed=5.0,
            # Start the swing while the last few centimetres of the approach finish.
            mole_aim_tolerance=0.035,
            mole_pitch_tolerance=0.10,
        )
    else:
        # Note window, +/- around each note (2026-11-17; was +/-0.5 / 0.25 / 0.12 s every
        # 2.5 s): every policy-facing limit is at least 1 s, so hard is +/-1.0 s, easy
        # +/-2.0 s and medium sqrt(easy * hard) = +/-1.41 s. Notes (a chord's two notes share
        # a time) now come every 5 s so consecutive windows never overlap (a 1 s gap at
        # easy) and each hit maps to one note; the last of the 8 notes is at 47 s, easy ends
        # at 47 + 2.0 + 0.7 = 49.7 s (was 30.7 s), max_time 34 -> 53 s keeps the 3.3 s slack.
        # The reference oracle strikes every note about 0.10 s early.
        c["tasks"]["32"].update(rounds=8, interval=5.0, max_time=53.0)
        for level, window in (("easy", 2.0), ("medium", 1.41), ("hard", 1.0)):
            c["difficulties"][level]["note_window"] = window
    return c


def station_offset(config):
    """Seeded [dx, dy, 0] shift (m) of the whole station: drums or mole cabinet and both tool
    rests (``v2.station_offset``, written by ``roboclock.envs.sports.spatial``; zero without)."""
    dx, dy = config["v2"].get("station_offset", (0.0, 0.0))
    return [float(dx), float(dy), 0.0]


def shift_station(task):
    """Move every control, fixture and tool of the task's layout by ``station_offset``."""
    offset = station_offset(task.config)
    if not any(offset):
        return
    for item in [*task.layout.controls.values(), *task.layout.bodies.values()]:
        item.position = [float(p + d) for p, d in zip(item.position, offset)]
    for fixture in task.layout.fixtures:
        fixture["position"] = [float(p + d) for p, d in zip(fixture["position"], offset)]


def tool_layout(task):
    h = task.config["scene"]["table_height"]
    # The tool rests horizontally in an open saddle; there is no attachment.
    task.layout.bodies["mallet"] = Body(
        "mallet",
        [0.30, -0.30, h + 0.074],
        radius=0.011,
        height=0.26 + task.config["v2"]["tool_offset"] - 0.125,
        mass=task.config["v2"]["tool_mass"],
        shape="mallet",
        label="",
    )
    task.layout.friction = 1.0


class TaikoTask(SportsTask):
    task_id = "32"
    name = "taiko_stick_drumming"
    category = "temporal_window"

    def __init__(self, difficulty, seed, config):
        super().__init__(difficulty, seed, config)
        self.layout.fixtures = []
        self.layout.controls = {
            f"pad_{i}": Control(
                f"pad_{i}",
                [0.51, (0.5 - i) * 0.24, 0.91],
                radius=0.095,
                height=0.012,
                travel=0.009,
                threshold=0.001,
                color=[0.88, 0.80, 0.62],
            )
            for i in range(2)
        }
        tool_layout(self)
        if config["v2"].get("dual_sticks"):
            self.layout.bodies["mallet_left"] = replace(
                self.layout.bodies["mallet"],
                name="mallet_left",
                position=[0.30, 0.30, config["scene"]["table_height"] + 0.074],
            )
        shift_station(self)
        self.hits = []

    def schedule(self, rng):
        interval = self.settings["interval"]
        events = [
            dict(
                time=12.0 + i * interval,
                control=f"pad_{int(rng.integers(2))}",
                hold=0.0,
                valid=True,
                kind="beat",
            )
            for i in range(self.settings["rounds"])
        ]
        if self.config["v2"].get("dual_sticks"):
            # Paired notes share a timestamp; lane 0 is robot-left (+Y).
            chords = []
            for i, event in enumerate(events):
                if i % 3 == 0:
                    event["kind"] = "chord"
                    chords.append(
                        dict(event, control="pad_1" if event["control"] == "pad_0" else "pad_0")
                    )
            events = sorted(events + chords, key=lambda e: (e["time"], e["control"]))
        if self.config["v2"]["variant"] == "roll":
            # A roll is three separated impacts, not an impossible sustained drum HOLD.
            last = events[-1]
            events.extend(
                [
                    dict(
                        time=last["time"] + j * 0.7,
                        control=last["control"],
                        hold=0.0,
                        valid=True,
                        kind="roll",
                    )
                    for j in (1, 2)
                ]
            )
        return events

    def evaluate(self, state, dt):
        self.current_index = next(
            (
                i
                for i, e in enumerate(self.events)
                if e["time"] + self.limits["note_window"] >= self.time and i not in self.correct
            ),
            None,
        )
        for name in self.rises:
            onset = self.presses[name]
            self.attempted.add((self.steps, name))
            candidates = [
                i
                for i, e in enumerate(self.events)
                if i not in self.correct
                and e["control"] == name
                and abs(onset - e["time"]) <= self.limits["note_window"]
            ]
            if candidates:
                i = min(candidates, key=lambda i: abs(onset - self.events[i]["time"]))
                self.success(i)
                hit = dict(
                    index=i,
                    time=onset,
                    error=onset - self.events[i]["time"],
                    contact_force=state["controls"][name]["tool_force"],
                    hand=state["controls"][name].get("striking_hand", "right"),
                )
                self.hits.append(hit)
                self.event("stick_hit", **hit)
            else:
                self.violation(f"wrong_or_offbeat_hit_{self.steps}")
        for i, e in enumerate(self.events):
            if self.time > e["time"] + self.limits["note_window"] and i not in self.correct:
                self.violation(f"missed_note_{i}")
        self.phase = "prepare" if self.time < 10 else "play"
        if self.time > self.events[-1]["time"] + self.limits["note_window"] + 0.7:
            self.goal_achieved = len(self.correct) == len(self.events)
            self._terminated = True

    def gen_instruction(self):
        if self.config["v2"].get("dual_sticks"):
            return "Pick up both sticks. LEFT is robot-left (+Y), RIGHT is robot-right (-Y). Strike each cued drum; paired notes are simultaneous. Lift the sticks between hits."
        return "Pick up the stick. Strike the indicated drum on each visible note. Lift the stick between strikes."


class MalletMoles(WhackMoles):
    name = "mallet_whack_a_mole"

    def __init__(self, difficulty, seed, config):
        super().__init__(difficulty, seed, config)
        tool_layout(self)
        if config["v2"].get("dual_mallets"):
            self.layout.bodies["mallet_left"] = replace(
                self.layout.bodies["mallet"], name="mallet_left",
                position=[0.30, 0.30, config["scene"]["table_height"] + 0.074],
            )
        # Reserve wrist-arc clearance: the near row must not force a folded IK branch.
        for control in self.layout.controls.values():
            control.position[0] += 0.10
        for fixture in self.layout.fixtures:
            fixture["position"][0] += 0.10
        shift_station(self)
        for e in self.events:
            e["time"] += config["v2"]["preparation"]
        self.spec = replace(self.spec, events=tuple(self.events))

    def gen_instruction(self):
        tools = "both mallets" if self.config["v2"].get("dual_mallets") else "the mallet"
        return f"Pick up {tools}. Hit green moles with a held mallet head before they retract. Never hit red moles."
