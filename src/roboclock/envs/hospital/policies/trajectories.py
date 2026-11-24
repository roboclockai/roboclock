"""Cartesian waypoints with contact-driven phase transitions for the hospital tasks.

Privileged diagnostic planner. Its output always becomes articulation motor commands.
"""

import numpy as np

from roboclock.envs.hospital.policies.jobs import HospitalJobs, object_yaw
from roboclock.envs.lab.mdp.state import orientation_error
from roboclock.envs.lab.policies.trajectories import CartesianCommand, limit_step

YAW_RATE = 2.5  # rad/s
CLOSE_SPEED = 0.05  # m/s per finger when closing gently on a light, round item
OPEN_SPEED = 0.25


def ramp_finger(current: float, target: float, dt: float) -> float:
    """Move the commanded finger position toward the target at a bounded speed.

    Closing slowly means the first finger to touch an item pushes with only a small force, which
    then builds up to the squeeze set by the closing target, so light round items are not shot out.
    """
    speed = CLOSE_SPEED if target < current else OPEN_SPEED
    step = speed * dt
    return float(min(max(target, current - step), current + step))


class HospitalTrajectory:
    def __init__(self, config, dt):
        self.config, self.dt = config, dt
        self.jobs = HospitalJobs()
        self.reset()

    def reset(self):
        self.jobs.reset()
        self.phase, self.sample, self.job = "select", None, None
        self.phase_time = 0.0
        self.retries, self.released_at = {}, {}
        self.last_goal = None
        self.hold = None
        self.grasp_offset = np.zeros(3)
        self.yaw = 0.0
        self.open_width = 0.04
        self.close_width = 0.0
        self.gentle = False
        self.drawer_phase, self.drawer_time = "hover", 0.0
        self.path_index = 0
        self._press_key, self._press_top = None, None

    def trace(self, task, name, ee, goal, **extra):
        """Once a second: how far the hand is from where the planner wants it (for diagnosis)."""
        if task.time - getattr(self, "_trace_time", -9.0) < 1.0:
            return
        self._trace_time = task.time
        task.event_log.append(
            {
                "time": round(task.time, 2),
                "type": "planner_trace",
                "object": name,
                "phase": self.phase,
                "dxy": round(float(np.linalg.norm(ee[:2] - goal[:2])), 4),
                "dz": round(float(ee[2] - goal[2]), 4),
                **extra,
            }
        )

    def transition(self, phase, time):
        self.phase, self.phase_time = phase, time

    @staticmethod
    def open_width_for(name, layout):
        return float(np.clip(layout.dimensions[name][0] + 0.014, 0.018, 0.04))

    def step_yaw(self, target):
        delta = (target - self.yaw + np.pi / 2) % np.pi - np.pi / 2
        self.yaw += float(np.clip(delta, -YAW_RATE * self.dt, YAW_RATE * self.dt))

    @staticmethod
    def awaiting_cue(task):
        """True while an emergency task (B1-B3) has shown nothing a policy could react to.

        The cue is the simulation step at which the task turns it on: B1 the first pump light goes
        red (its alarm time), B2 the arrest (`event_time` set: response dial red, code-blue lamp
        flashing), B3 the ignition (`event_time` set: flames shown, FIRE dial red). Until then the
        oracle holds the pose the episode started in, like a policy that cannot know where or when
        the event will be; it used to hover over the next pump (B1), wait at a pose beside the AED
        (B2) and hover over the fire alarm (B3).
        """
        if task.category != "event_reactive":
            return False
        observed = getattr(task, "cue_observed", None)
        return task.event_time is None if observed is None else not observed

    def command(self, task, state):
        """The tool-pose command, or None to hold the start pose (B1-B3 before their cue)."""
        from roboclock.envs.hospital.bedside import module_for

        bedside = module_for(task)
        if bedside is not None:
            if self.last_goal is None:
                self.last_goal = state.ee_position.copy()
                self.yaw = float(task.layout.info.get("initial_ee_yaw", 0.0))
            return bedside.command(self, task, state)
        if self.awaiting_cue(task):
            self.last_goal = None  # start from wherever the hand is when the cue comes
            return None
        layout = task.layout
        # Carry height: 0.30 m over the bench, or a task's own (B2: over its raised AED table).
        clearance = float(layout.info.get("clearance_z", layout.table_height + 0.30))
        if self.phase != getattr(self, "_logged_phase", None):
            self._logged_phase = self.phase
            task.event_log.append(
                {
                    "time": task.time,
                    "type": "planner_phase",
                    "object": self.sample,
                    "phase": self.phase,
                }
            )
        if self.last_goal is None:
            self.last_goal = state.ee_position.copy()
            self.yaw = 0.0
        if task.task_id == "A2" and self.drawer_phase != "done":
            return self.pull_drawer(task, state)
        if self.phase == "select":
            recent = [n for n, t in self.released_at.items() if task.time - t < 1.2]
            exhausted = [n for n, k in self.retries.items() if k >= 3]
            name = self.jobs.choose(task, state, exclude=recent + exhausted)
            if (
                name is None
                and hasattr(task, "button")
                and not task.press_done
                and not task.goal_achieved
            ):
                return self.press_button(task, state)
            if name is None or task.goal_achieved:
                self.open_width = 0.04
                return self.finish(state, np.array([*state.ee_position[:2], clearance]), False)
            self.sample = name
            self.transition("approach", task.time)
        name = self.sample
        job = self.jobs.describe(task, state, name)
        obj = state.objects[name]
        pick, close = (
            job.pick.copy(),
            self.phase in ("grasp", "lift", "transfer", "insert", "retain"),
        )
        above = pick.copy()
        above[2] = max(pick[2] + 0.10, clearance - 0.08)
        goal = above
        if not job.grasp_ok and job.hover is not None and self.phase in ("approach", "descend"):
            goal = job.hover.copy()
        if task.task_id == "A2" and self.phase == "approach":
            # Cross over the drawer handle high enough that the fingers do not push it shut, and only
            # lower the hand once it is above the bag.
            if np.linalg.norm(state.ee_position[:2] - pick[:2]) > 0.04:
                goal = goal.copy()
                goal[2] = max(goal[2], layout.info["handle"][2] + 0.17)
        target_yaw = (
            object_yaw(obj.orientation)
            if self.phase in ("approach", "descend", "grasp")
            else job.yaw
        )
        self.step_yaw(target_yaw)
        ready = (
            orientation_error(
                state.ee_orientation,
                np.array([0.0, -np.sin(self.yaw / 2), np.cos(self.yaw / 2), 0.0]),
            )
            < 0.10
        )

        def near(point, tol=0.012):
            return np.linalg.norm(state.ee_position - point) < tol

        if self.phase == "approach":
            # Descend only when the open fingers straddle the item with margin to spare.
            margin = self.open_width_for(name, layout) - layout.dimensions[name][0]
            aligned = np.linalg.norm(state.ee_position[:2] - goal[:2]) < min(0.02, 0.5 * margin)
            self.trace(
                task,
                name,
                state.ee_position,
                goal,
                ready=bool(ready),
                tol=round(float(min(0.02, 0.5 * margin)), 4),
            )
            if job.grasp_ok and aligned and abs(state.ee_position[2] - goal[2]) < 0.02 and ready:
                self.transition("descend", task.time)
        elif self.phase == "descend":
            goal = pick
            if not job.grasp_ok:
                self.transition("approach", task.time)
            elif near(goal) and ready:
                self.transition("grasp", task.time)
        elif self.phase == "grasp":
            goal, close = pick, True
            # The bag is heavy: let the pinch force build before lifting it.
            held_long_enough = task.task_id != "A2" or task.time - self.phase_time > 0.5
            # A thin pad can be pinched and lifted before the contact reading catches up: counts as held once
            # it rides with the hand.
            riding = (
                np.linalg.norm(obj.position - state.ee_position) < 0.03
                and obj.position[2] - layout.objects[name].position[2] > 0.02
            )
            if (obj.grasped or riding) and held_long_enough:
                self.grasp_offset = state.ee_position - obj.position
                self.hold = state.ee_position.copy()
                self.hold[2] = max(self.hold[2], clearance)
                if task.task_id == "A2":
                    self.hold[2] = max(self.hold[2], layout.info["bag_lift_ee_z"])
                self.transition("lift", task.time)
            elif task.time - self.phase_time > 1.4:
                self.retries[name] = self.retries.get(name, 0) + 1
                task.event_log.append(
                    {"time": task.time, "type": "planner_grasp_failed", "object": name}
                )
                # Give up on an item after three failed grasps and move on to the next one.
                self.transition("select" if self.retries[name] >= 3 else "approach", task.time)
                close = False
        elif self.phase == "lift":
            goal = self.hold
            if near(goal, 0.02):
                self.transition("transfer", task.time)
        elif self.phase == "transfer":
            goal = job.destination + self.grasp_offset
            goal[2] = clearance
            if job.path:
                goal = job.path[0] + self.grasp_offset
            self.trace(task, name, state.ee_position, goal)
            if near(goal, 0.02) and job.transfer_ok:
                self.path_index = 0
                self.waypoint_time = task.time
                self.transition("insert", task.time)
        elif self.phase == "insert":
            held = state.ee_position - obj.position
            # An arm at full reach can settle a centimeter short of its target: after it has had time
            # to settle, a slightly looser tolerance is accepted instead of stalling until the timeout.
            settled_for = task.time - self.phase_time
            if job.path:
                # Visit each waypoint in turn: line up first, then slide along the peg.
                goal = job.path[self.path_index] + held
                self.trace(task, name, state.ee_position, goal, wp=self.path_index)
                if near(goal, 0.006) or (
                    task.time - self.waypoint_time > 3.0 and near(goal, 0.012)
                ):
                    if self.path_index < len(job.path) - 1:
                        self.path_index += 1
                        self.waypoint_time = task.time
                    elif job.release_ok:
                        self.hold = state.ee_position.copy()
                        self.transition("release", task.time)
            else:
                goal = job.destination + held
                self.trace(task, name, state.ee_position, goal)
                # A pull-tab the task already counts as pulled needs no more moving: let go at once instead of
                # holding on until the timeout (the grasp reading can drop while the thin tab is carried).
                consumable = name in getattr(task, "consumables", ())
                pulled = consumable and name in task.correct
                # A tab that has not come free after three seconds is let go and grasped again, not held for the
                # whole phase timeout.
                stuck = consumable and settled_for > 3.0 and job.release_ok
                if (
                    near(goal, 0.009) or (settled_for > 2.0 and near(goal, 0.02)) or pulled or stuck
                ) and job.release_ok:
                    self.hold = state.ee_position.copy()
                    self.transition("release", task.time)
        elif self.phase == "release":
            goal, close = self.hold.copy(), False
            if task.time - self.phase_time >= 0.15:
                goal[2] += 0.04
            if not obj.grasped and task.time - self.phase_time >= 0.45:
                self.released_at[name] = task.time
                self.transition("retreat", task.time)
        elif self.phase == "retreat":
            # Only ever move up and away: after hanging the bag the hand is already above clearance.
            close = False
            if name in getattr(task, "consumables", ()):
                # A released pull-tab needs no retreat: the next approach already rises as it moves. Climbing here
                # swung the arm sideways and it sat there until the timeout with the next tab waiting.
                goal, done = state.ee_position.copy(), True
            else:
                goal = np.array([*state.ee_position[:2], max(clearance, state.ee_position[2])])
                done = near(goal, 0.03)
            if done:
                self.retries[name] = self.retries.get(name, 0) + 1
                self.transition("select", task.time)
        # Waiting is only exempt from the timeout when the task itself withholds permission
        # (a light, an alarm, a window). Otherwise a single unreachable item must not stall a run.
        waiting = (
            (not job.grasp_ok and self.phase in ("approach", "descend"))
            or (not job.transfer_ok and self.phase == "transfer")
            or (not job.release_ok and self.phase == "insert")
        )
        if task.time - self.phase_time > self.config["phase_timeout"] and not waiting:
            self.retries[name] = self.retries.get(name, 0) + 1
            task.event_log.append(
                {"time": task.time, "type": "planner_timeout", "object": name, "phase": self.phase}
            )
            if self.phase in ("lift", "transfer", "insert"):
                # Holding the item: set it down here and let go instead of dropping it later.
                self.hold = state.ee_position.copy()
                self.transition("release", task.time)
            else:
                self.transition("select", task.time)
                close = False
        carrying = self.phase in ("lift", "transfer", "insert", "release", "retreat")
        quick = job.fast if carrying else job.rush
        speed = self.config["reactive_speed"] if quick else self.config["max_cartesian_speed"]
        if task.task_id == "B2" and carrying and name.startswith("pad"):
            speed = min(speed, 0.30)  # a pad pinched by two fingertips is flung out by a fast carry
        if task.task_id == "A2" and carrying:
            speed = min(speed, 0.25)  # a swinging 0.35 kg bag held by its loop: carry it gently
        # Open the fingers only as wide as the item needs, so neighbors are not hit.
        self.open_width = self.open_width_for(name, layout)
        # Closing to a target just inside the item gives a controlled squeeze (stiffness 1000 N/m).
        force = layout.shapes[name].grip_force
        self.gentle = force is not None
        self.close_width = (
            0.0 if force is None else max(0.0, layout.dimensions[name][0] - force / 1000)
        )
        if self.phase == "descend":
            speed = self.config["insertion_speed"] * 3
        if self.phase == "insert" and job.path:
            speed = 0.06  # threading an eyelet onto a peg needs care
        return self.finish(state, goal, close, speed)

    def press_button(self, task, state):
        """Go over the button (a pump's SILENCE key, SHOCK, or the fire alarm) with the fingers closed, press it once
        ready, then lift off. It is only called after the task's cue (`awaiting_cue`)."""
        button, ee = task.button, state.ee_position
        self.gentle, self.close_width, self.open_width = False, 0.0, 0.0
        self.step_yaw(0.0)
        if button is None:
            # B1 between alarms: no pump is alarming and the next one is not known. Lift straight
            # off the key just pressed and stay there; never move toward a pump that has not alarmed.
            goal = self.last_goal.copy()
            goal[2] = max(goal[2], (goal[2] - 0.14 if self._press_top is None else self._press_top) + 0.14)
            return self.finish(state, goal, True, 0.2)
        self._press_top = float(button[2])
        key = tuple(np.round(np.asarray(button, dtype=float), 4))
        if self._press_key != key:
            self._press_key, self.press_bias = key, np.zeros(2)
        error = np.asarray(button[:2]) - ee[:2]
        aligned = np.linalg.norm(error) < 0.008
        if not aligned and np.linalg.norm(error) < 0.03:
            # A stretched arm hovers a centimeter off its goal (it sags): aim off by the measured
            # miss, bounded, until the hand is over the key.
            self.press_bias = np.clip(self.press_bias + 1.5 * error * self.dt, -0.025, 0.025)
        aim = np.asarray(button[:2]) + self.press_bias
        over = np.array([*aim, button[2] + 0.14])
        if task.press_done:
            goal = np.array([ee[0], ee[1], button[2] + 0.14])
            return self.finish(state, goal, True, 0.2)
        if task.press_ready and aligned:
            return self.finish(state, np.array([*aim, button[2] + 0.004]), True, 0.06)
        # The deadline is running whenever a press is due or about to be: go at the reactive speed.
        return self.finish(state, over, True, self.config["reactive_speed"])

    def pull_drawer(self, task, state):
        """Grip the drawer handle, pull the drawer open along +y, let go and lift clear."""
        layout = task.layout
        handle = layout.info["handle"]
        travel = layout.info["drawer_travel"]
        bag = layout.objects[task.bag].position
        shift = float(state.objects[task.bag].position[1] - bag[1])  # the bag rides in the drawer
        ee, now = state.ee_position, task.time
        self.gentle, self.close_width, self.open_width = False, 0.0, 0.04  # the bar is 18 mm thick
        self.step_yaw(0.0)
        close, speed = False, self.config["max_cartesian_speed"]
        phase = self.drawer_phase
        goal = handle + np.array([0.0, 0.0, 0.16])
        if phase != getattr(self, "_logged_drawer_phase", None):
            self._logged_drawer_phase = phase
            task.event_log.append(
                {"time": now, "type": f"drawer_{phase}", "shift": round(shift, 3)}
            )
        if phase == "hover":
            if np.linalg.norm(ee - goal) < 0.012:
                self.drawer_phase, self.drawer_time = "descend", now
        elif phase == "descend":
            goal, speed = handle.copy(), 0.10
            if np.linalg.norm(ee[:2] - handle[:2]) > 0.025:
                self.drawer_phase = "hover"
            elif np.linalg.norm(ee - goal) < 0.008:
                self.drawer_phase, self.drawer_time = "close", now
        elif phase == "close":
            goal, close = handle.copy(), True
            if now - self.drawer_time > 0.7:
                self.drawer_phase, self.drawer_time = "pull", now
        elif phase == "pull":
            goal, close, speed = handle + np.array([0.0, travel - 0.01, 0.0]), True, 0.10
            if shift >= travel - 0.02 or now - self.drawer_time > 8.0:
                self.drawer_phase, self.drawer_time = "let_go", now
        elif phase == "let_go":
            goal, self.open_width = ee.copy(), 0.04
            if now - self.drawer_time > 0.5:
                self.drawer_phase = "rise"
        elif phase == "rise":
            self.open_width = 0.04
            goal = np.array([ee[0], ee[1], handle[2] + 0.16])
            if ee[2] > handle[2] + 0.12:
                self.drawer_phase = "done"
        return self.finish(state, goal, close, speed)

    def finish(self, state, goal, close, speed=None):
        speed = speed or self.config["max_cartesian_speed"]
        self.last_goal = limit_step(self.last_goal, goal, speed, self.dt)
        return CartesianCommand(self.last_goal.copy(), close, self.yaw)
