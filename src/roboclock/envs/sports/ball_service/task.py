"""Contact-based service evaluation, independent of controller phase labels.

C-class stream task (score_version 2, docs/c-class-timing.md). Each of the ``demand_count``
demands is a job whose deadline is its scheduled firing time (config.py: m x the hard
schedule, m = 4 / 2 / 1 for easy / medium / hard). At that time the launcher takes
the earliest-accepted ball waiting free in its accepted cell; the job time is that ball's
acceptance time. A demand that finds no ball (starvation) is owed: the next ball that becomes
available is fired for it, and the job time is that moment, so the job is late but can still
be served. The episode ends SETTLE_S after the last demand is served (all_done) or at the
horizon, 1.25 x the last scheduled firing time. Physical success needs every demand served
through the outlet and no drop, arm collision, occupied cell or overflow; time plays no part.
"""

import numpy as np

from roboclock.envs.sports import c_class
from roboclock.evaluation.throughput import TIME_TOLERANCE_S

# Failure codes that rule out physical success (starvation and gate retries only delay jobs).
PHYSICAL = ("dropped_", "arm_collision", "occupied_cell", "overflow")


def retained(obj, side):
    i = 0 if side == "left" else 2
    return min(obj["forces"][i : i + 2]) > 0.025


def cell_at(position, c, *, full=True):
    p = np.asarray(position)
    r = c["radius"] if full else 0.0
    for i, center in enumerate(c["cells"]):
        if (
            np.max(np.abs(p[:2] - np.asarray(center[:2]))) + r <= c["cell_half_width"] + 0.001
            and c["floor_z"] - 0.005 <= p[2] - r
            and p[2] + r <= c["lip_z"] + 0.001
        ):
            return i
    return None


class Task:
    def __init__(self, b):
        self.b = b
        self.c = b.config["service"]
        self.done = False
        self.events = []
        self.failures = []
        self.termination_reason = None
        self.accepted = {}
        self.accepted_times = {}
        self.fired = set()
        self.jobs = []  # one per issued demand: scheduled_time, ball, job_time, served_time
        self.last_served = None
        self.grasped = {}
        self.grasp_since = {}
        self.grasp_offset = {}
        self.dwell = {}
        self.entry = set()
        self.previous = {}
        self.pending = None
        self.demand_index = 0
        self.max_occupancy = 0
        self.occupancy = []
        self.peak_arm_contact = 0.0

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)

    def emit(self, event, **kw):
        self.events.append(dict(event=event, time=self.b.time, **kw))

    def finish(self, reason):
        self.done, self.termination_reason = True, reason

    def occupants(self, s):
        return {
            n: cell_at(s["objects"][n]["position"], self.c)
            for n in self.c["sources"]
            if n not in self.fired and cell_at(s["objects"][n]["position"], self.c) is not None
        }

    def available(self, s):
        """Unfired balls resting free in the cell they were accepted into."""
        return [
            n
            for n, cell in self.occupants(s).items()
            if self.accepted.get(n) == cell
            and not retained(s["objects"][n], "left")
            and not retained(s["objects"][n], "right")
        ]

    def next_ball(self, s):
        """The earliest-accepted available ball not yet assigned to a demand."""
        assigned = {job["ball"] for job in self.jobs}
        names = [n for n in self.available(s) if n not in assigned]
        return min(names, key=self.accepted_times.__getitem__) if names else None

    def update(self, s):
        if self.done:
            return
        now = s["time"]
        c = self.c
        self.peak_arm_contact = max(self.peak_arm_contact, s.get("arm_contact_peak", 0))
        if c_class.arm_collision(s):
            self.fail("arm_collision")
        entries = 0
        for name in c["sources"]:
            o = s["objects"][name]
            p = np.asarray(o["position"])
            r = c["radius"]
            for side in ("left", "right"):
                key = (name, side)
                offset = p - np.asarray(s["hands"][side]["position"])
                stable = retained(o, side) and (
                    key not in self.grasp_offset
                    or np.linalg.norm(offset - self.grasp_offset[key]) < 0.008
                )
                if stable:
                    if key not in self.grasp_since:
                        self.grasp_since[key] = now
                        self.grasp_offset[key] = offset.copy()
                    if now - self.grasp_since[key] >= c["grasp_dwell"] and name not in self.grasped:
                        self.grasped[name] = side
                        self.emit("retained_grasp", ball=name, side=side)
                else:
                    self.grasp_since.pop(key, None)
                    self.grasp_offset.pop(key, None)
            # Count any ball overlapping the hopper interior, including supported
            # stacks above the rim, before acceptance; overflow cannot be hidden.
            if name not in self.fired:
                for center in c["cells"]:
                    if (
                        np.max(np.abs(p[:2] - np.array(center[:2]))) < c["cell_half_width"] + r
                        and c["floor_z"] < p[2] + r < c["lip_z"] + 3 * r
                    ):
                        entries += 1
                        break
            old = self.previous.get(name, p)
            for i, center in enumerate(c["cells"]):
                if (
                    old[2] - r > c["lip_z"] >= p[2] - r
                    and np.max(np.abs(p[:2] - np.array(center[:2]))) + r
                    < c["cell_half_width"] + 0.001
                ):
                    self.entry.add((name, i))
            self.previous[name] = p.copy()
            cell = cell_at(p, c)
            free = not retained(o, "left") and not retained(o, "right")
            support = o.get("support_contacts", {}).get(f"gate{cell}", 0) if cell is not None else 0
            valid = (
                cell is not None
                and (name, cell) in self.entry
                and name in self.grasped
                and free
                and support > 0.15
                and np.linalg.norm(o["velocity"]) < 0.035
            )
            if valid and name not in self.accepted:
                self.dwell.setdefault(name, now)
                if now - self.dwell[name] >= c["support_dwell"]:
                    if cell in self.accepted.values() and any(
                        v == cell and n not in self.fired for n, v in self.accepted.items()
                    ):
                        self.fail("occupied_cell")
                    self.accepted[name] = cell
                    self.accepted_times[name] = now
                    self.emit("accepted", ball=name, cell=cell, side=self.grasped[name])
            else:
                self.dwell.pop(name, None)
            if (
                name not in self.fired
                and p[2] < 0.40
                and not (self.pending and self.pending[0] == name)
            ):
                self.fail("dropped_" + name)
        self.max_occupancy = max(self.max_occupancy, entries)
        if entries > c["capacity"]:
            self.fail("overflow")
        if not self.occupancy or now - self.occupancy[-1][0] >= 0.1:
            self.occupancy.append([now, entries])
        self.issue_demand(s, now)
        self.assign_owed(s, now)
        self.open_gate(s, now)
        self.discharge(s, now)
        served = sum(job["served_time"] is not None for job in self.jobs)
        if (
            served == c["demand_count"]
            and self.pending is None
            and now - self.last_served >= c_class.SETTLE_S
        ):
            self.finish("all_done")
        elif now >= c["horizon"] - 1e-8:  # the env stops at max_time - 1e-9, as in C1/C2
            self.finish("horizon")

    def issue_demand(self, s, now):
        """At its scheduled time a demand takes a stocked ball or, if none, is owed."""
        c = self.c
        due = c["preparation"] + self.demand_index * c["firing_interval"]
        if self.demand_index >= c["demand_count"] or now < due:
            return
        job = dict(
            index=self.demand_index, scheduled_time=due, ball=None, job_time=None, served_time=None
        )
        self.jobs.append(job)
        self.emit("demand", scheduled_time=due, index=self.demand_index)
        name = self.next_ball(s)
        if name is None:
            self.fail("starvation")
            self.emit("missed_demand", scheduled_time=due, index=self.demand_index)
        else:
            job.update(ball=name, job_time=self.accepted_times[name])
        self.demand_index += 1

    def assign_owed(self, s, now):
        """Owed demands, oldest first, take the next ball that becomes available (late)."""
        for job in self.jobs:
            if job["ball"] is None:
                name = self.next_ball(s)
                if name is None:
                    return
                job.update(ball=name, job_time=now)
                self.emit(
                    "late_supply",
                    index=job["index"],
                    ball=name,
                    scheduled_time=job["scheduled_time"],
                )

    def open_gate(self, s, now):
        """Fire the oldest assigned, unserved demand's ball when the gate is free."""
        if self.pending is not None:
            return
        for job in self.jobs:
            if job["ball"] is None or job["served_time"] is not None:
                continue
            name = job["ball"]
            if name not in self.available(s):
                job.update(ball=None, job_time=None)  # gone before its gate opened: owed again
                continue
            cell = self.accepted[name]
            self.b.gate_drives[cell].GetTargetPositionAttr().Set(-0.13)
            self.pending = (name, cell, now, job["index"])
            return

    def discharge(self, s, now):
        """Count the ball once it passes the outlet; close the gate after its timeout."""
        if not self.pending:
            return
        c = self.c
        name, cell, opened, index = self.pending
        job = self.jobs[index]
        p = np.asarray(s["objects"][name]["position"])
        oldcell = np.array(c["cells"][cell])
        if (
            p[2] + c["radius"] < 0.79
            and p[2] > 0.55
            and np.max(np.abs(p[:2] - oldcell[:2])) < 0.08
            and name not in self.fired
        ):
            self.fired.add(name)
            job["served_time"] = self.last_served = now
            self.emit(
                "outlet_passage",
                ball=name,
                cell=cell,
                scheduled_time=job["scheduled_time"],
                demand=index,
                late=job["job_time"] > job["scheduled_time"] + TIME_TOLERANCE_S,
            )
        if now - opened >= c["gate_timeout"]:
            if name not in self.fired:
                self.fail("unverified_discharge")
                job.update(ball=None, job_time=None)  # not served: owed again
            self.b.gate_drives[cell].GetTargetPositionAttr().Set(0)
            self.pending = None

    def display(self):
        stock = len(self.occupants(self.b.last_state))
        count = self.c["demand_count"]
        owed = sum(job["ball"] is None for job in self.jobs)
        due = self.c["preparation"] + self.demand_index * self.c["firing_interval"]
        if self.done:
            return (
                f"COMPLETE / SERVED {len(self.fired)}/{count}",
                "CONTINUOUS FEED / NO OVERFLOW"
                if not self.failures
                else "CHECK FAILED / SEE EVALUATION",
            )
        if self.demand_index >= count:
            return (
                f"FINAL CYCLE / SERVED {len(self.fired)}/{count}",
                f"OWED {owed} / STOCK A BALL" if owed else "VERIFYING OUTLET / FINISHING",
            )
        return (
            f"STOCK {stock}/{self.c['capacity']} / SERVED {len(self.fired)}"
            + (f" / OWED {owed}" if owed else ""),
            f"{'START' if self.demand_index == 0 else 'NEXT'} IN {max(0, due - self.b.time):04.1f}s / {60 / self.c['firing_interval']:.1f} BALLS/MIN",
        )

    def deadlines(self):
        c = self.c
        return [c["preparation"] + i * c["firing_interval"] for i in range(c["demand_count"])]

    def job_times(self):
        """Per demand: the stocking time of the ball that served it, None if not served."""
        times = [None] * self.c["demand_count"]
        for job in self.jobs:
            if job["served_time"] is not None:
                times[job["index"]] = job["job_time"]
        return times

    def result(self):
        count = self.c["demand_count"]
        served = [job for job in self.jobs if job["served_time"] is not None]
        physical_failures = [x for x in self.failures if x.startswith(PHYSICAL)]
        physical = (
            len(served) == count
            and len(self.accepted) >= count
            and len(self.fired) == count
            and not physical_failures
        )
        judged = c_class.judge_stream(
            self.job_times(),
            self.deadlines(),
            physical,
            termination_reason=self.termination_reason,
        )
        active = (count - 1) * self.c["firing_interval"] + 2.0
        return dict(
            task="C3",
            revision="fixed-service-v1",
            primary_temporal_category="throughput",
            success=judged["joint_success"],
            physical_success=physical,
            temporal_success=judged["temporal_success"],
            failures=self.failures,
            events=self.events,
            accepted=len(self.accepted),
            served=len(self.fired),
            demand=self.demand_index,
            accepted_times=dict(self.accepted_times),
            served_times={job["ball"]: job["served_time"] for job in served},
            demands=[dict(job) for job in self.jobs],
            completed_at=self.last_served if len(served) == count else None,
            termination_reason=self.termination_reason,
            throughput=judged,
            max_occupancy=self.max_occupancy,
            occupancy=self.occupancy,
            active_service_seconds=active,
            served_per_minute=len(self.fired) * 60 / active,
            grasp_sides=self.grasped,
            peak_arm_contact=self.peak_arm_contact,
            controller="Privileged state-feedback scripted baseline; not a trained visual agent",
            outlet="Physical gravity-metering gates and collector; no launch-wheel or flight-aerodynamics model",
        )
