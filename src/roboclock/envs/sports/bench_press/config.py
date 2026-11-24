"""Seeded human fatigue; difficulty changes the response deadline only."""
import numpy as np


# Oracle catch motion (2026-11-04): faster approach, joints near the X5 limit
# (5.5 rad/s for joints 1-3) and jaws pre-closed to 26 mm per side on the way
# down (was 0.85 m/s, 3.4 rad/s, fully open 38 mm).
APPROACH_SPEED = 1.6
BENCH_JOINT_SPEED = 5.0
APPROACH_GRIP = 0.026
CAPTURE_GAP_MIN = 0.045


def configure(c):
    level = ("easy", "medium", "hard").index(c["difficulty"])
    rng = np.random.default_rng(c["seed"] + 409)
    c["task_revision"] = "bench-press-v6-eccentric"  # v6: slow failed-rep descent (2026-11-17)
    c["workflow"].update(
        bench_press=True, bench_mass=1.2, bench_mount=0.53,
        bench_bottom=0.86, bench_top=1.12, bench_x=0.32,
        bench_grip_y=0.36, human_grip_y=0.23,
        rack_x=0.23, rack_height=1.08,
        fatigue_rep=int(rng.integers(1, 4)),
        fatigue_progress=float(rng.uniform(0.35, 0.65)),
        fatigue_stall=float(rng.uniform(0.65, 1.1)),
        # Catch deadline, from the detected bar descent (2026-11-17; was 1.10 / 0.98 / 0.88 s
        # after the 2026-11-16 recalibration, 0.80 / 0.72 / 0.64 s, before 2026-11-04 0.65 /
        # 0.40 / 0.25 s): every policy-facing limit is at least 1 s, so hard is 1.0 s, easy
        # 2.0 s and medium sqrt(easy * hard) = 1.41 s. The reference oracle catches 0.40-0.58 s
        # after the descent (old fall, seeds 0-9). With the slow failed-rep descent below the
        # bar is still more than 6 cm above human_danger_height when the easy deadline expires.
        # Easy since 2026-11-23: no documented time, so 4 x t = 4 x 0.512 = 2.05 s; medium = sqrt(easy * hard).
        catch_deadline=[2.05, 1.43, 1.0][level],
        catch_dwell=0.06, rack_dwell=0.5,
        descent_threshold=0.025, stall_dwell=0.12,
        standby_position=[0.28, 0., 1.18], standby_tolerance=0.015,
        standby_grip=0.038, bench_joint_speed=BENCH_JOINT_SPEED,
        approach_speed=APPROACH_SPEED, approach_grip=APPROACH_GRIP,
        capture_gap_min=CAPTURE_GAP_MIN,
        # Failed rep (2026-11-17; was a fall with the support decayed to 0.97 x weight,
        # 0.29 m/s^2): the exhausted lifter lowers the bar at 5 cm/s after a 0.25 s ramp
        # (``process.eccentric_drop``). The descent is detected (0.025 m/s) ~0.26 s after the
        # stall ends. From the lowest stall height (0.933 m) the bar is at 0.83 m 2 s later
        # (easy deadline) and reaches the 0.76 m danger height 3.2 s after detection; from
        # the highest (1.047 m) 5.3 s (1-D model of this wrench, bar mass 1.2 kg).
        fatigue_descent_speed=0.05, fatigue_descent_ramp=0.25,
        human_danger_height=0.76, bench_timeout=45.0,
    )
    return c
