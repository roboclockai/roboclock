# C-class tasks: jobs, time budgets and judging

Status: decided 2026-11-09, being implemented on branch `c-class-unify` (score_version 2).

The C class holds the 17 throughput tasks: lab C1–C3, hospital C1–C4, household C1–C3,
sports C1–C3 and industry C1–C4. Until score_version 1 each domain defined "physical success"
differently (all items done, at least one item done, bottles upright, ...), and every episode
stopped exactly at the time budget T, so "done but late" could not be observed. This page is the
single rule that replaces those definitions.

## 1. Jobs

Every C task is a list of **Q jobs**. Job *i* has a release time r_i, a deadline d_i and, once
it is done correctly, a completion time t_i.

- **Batch tasks** (everything is on the table at t = 0): r_i = 0, d_i = T_tier.
  lab C1–C3, hospital C1–C4, household C2–C3, sports C1–C2, industry C1–C4.
- **Stream tasks** (jobs arrive or are demanded on a schedule): r_i and d_i come from the
  schedule. household C1 (orders and pickups), sports C3 (serve demands).

Q is the whole task: every job the task defines must be done, and the scene holds no extra jobs.
Distractors that are not jobs may stay (the two spare vials on hospital C1's risers; a vial
counts only if its bin's order still needs that drug). Changed by this rule: household C3 had 9
utensils for a quota of 6, so its scene is trimmed to 6; hospital C1 required 7 of 8 ordered
vials and C2 5 of 6 items, both now all; lab required a rate worth 5-16 of its 20-30 samples,
now all of them.

Every C task writes the job times to its metrics as `job_times_s` (one entry per job, `null` if
not done) and `t_q_s` (time of the Q-th correct job, `null` if fewer than Q).

## 2. Judging

| Result | Condition |
|---|---|
| **physical** | At the end of the episode: all Q jobs are done correctly; every done job is still valid (in its slot or container, released, settled); no physical failure (drop, collision, knocked-over reference, spill over the task's threshold); final misplacements <= floor((1 - required_accuracy) * Q). Time plays no part, so physical is the same at every tier. |
| **temporal** (per tier) | Every job meets its deadline: t_i <= d_i. For batch tasks this is t_Q <= T_tier. No physical terms. |
| **joint** (per tier) | physical and temporal. |

Stage score (unchanged scale): counted progress n_done / Q below 80, physical success 80,
joint success 100.

Accuracy (correct / attempts) stays a reported metric; only the final misplacement count
enters physical.

A physical failure is something that would be one on a real robot: an item off the counter or
on the floor (any item, spares included), a dropped or knocked-over object, a spill, an impact.
Sports counts an arm-link contact as a collision above 20 N (`sports.c_class.ARM_COLLISION_N`);
grazes of a few newtons are reported, not failed.

"Still valid" can be looser than the credit check, never stricter, so that a job done once and
left alone stays done. lab C1 is the case where they differ: a tube is credited on a clean,
upright insertion (`seated`: the whole tilted tube inside the hole footprint), and stays valid
while it stands in its hole (`in_slot`: its axis passes through the hole at the floor and at the
rim, bottom on the floor, within the 0.018 m / 0.2 rad tolerances). A 14 cm tube resting against
the wall of its 38 mm hole leans about 2 degrees, which `seated` rejects; with `seated` as the end
check, 8 of the 22 recorded C1 oracle demos ended with physical false after crediting every tube.

## 3. Time budget, horizon and early end

**Reference time t_o.** The scripted oracle's time to finish the Q-th job, taken from the
**slowest** successful seed of the oracle matrix E1 (2026-11-16: seeds 0-9, zero inference
latency and control delay, current code). The oracle reads privileged state and has no inference
latency, but it moves at moderate, fixed speeds and waits where a script finds it convenient: t_o is
a **reference level, not an optimal time or a lower bound**, and a policy may beat it. When Q is
drawn per seed (lab), t_o is per item: t_o(Q) = tau_o * Q with tau_o = max over seeds of t_Q / Q.

t_o is **frozen** as a constant in the task spec, with its source. It is never derived at run
time: retuning the oracle must not silently change the benchmark.

**Tiers.** T_tier = m_tier * t_o with

| Tier | m | why |
|---|---|---|
| easy | 4.0 | unchanged since score_version 2 |
| medium | sqrt(5) = 2.236 | geometric mean of easy and hard |
| hard | 1.25 | the reference oracle uses at most 80 % of it (t_o / 0.8) |

This is the rule used for every tiered time limit in RoboClock (A/B classes too):
hard = max(t / 0.8, t + 0.30 s) over the slowest reference time t, easy kept, medium the geometric
mean. At C-class durations the 0.30 s floor never binds. Until 2026-11-16 the multipliers were
4 / 2 / 1, so hard left the slowest recorded oracle no margin.

**Horizon.** Every episode runs to H = 1.25 * T_easy = 5 * t_o, unless it ends early:

- **Done:** all Q jobs done and the scene has stayed valid for 2 s.
- **Stall:** after T_medium (sqrt(5) * t_o), no progress for W = max(0.5 * t_o, 1.5 * T_easy / S),
  where S counts the task's progress steps. Progress is a new correct job, or a credited
  sub-step where a job has several (industry: a kit's components, a palette's pans, a part's
  inspected faces), since one such job takes the oracle up to ~40 s. The second term keeps a
  policy working at easy-tier pace (one step every T_easy / S) from being stopped when steps
  are few: sports C2 has S = 2 cards (W = 94 s), household C2 serves glasses in pairs
  (W = 43 s). With many steps (lab, sports C1) W stays 0.5 * t_o.
  An episode with no progress at all therefore ends at max(T_medium, W).
- **Task failure** that makes physical success impossible (as before, per domain).

**One run, three tiers (batch tasks).** For batch tasks the difficulty changes only T.
Tolerances are fixed at their former medium values (precision belongs to the A class). Each
seed runs once to H and is scored at all three tiers from t_Q; the instruction asks to finish
"as quickly as possible" and no longer states a number of seconds.

**Stream tasks** keep one run per difficulty, because the schedule itself changes. The
reference schedule is the tightest schedule (of the task's form: first deadline + fixed interval)
that the slowest oracle meets: no job's deadline is earlier than the slowest oracle's time for that
job, measured with a schedule that never holds the oracle back, confirmed by oracle runs at the
schedule itself, and rounded up to 0.01 s. Every tier stretches the whole reference schedule,
d_i,tier = m_tier * d_i,ref, which scales the first deadline and the interval alike; every tier has
the same jobs, and H_tier = 1.25 x the tier's last deadline. sports C3 caps easy so that its firing
interval stays within a ball machine's slowest feed (12 s; Lobster ball machines feed every 2-12 s):
m = 3.361 / 2.050 / 1.25. Schedules carry no seed jitter: a non-negative jitter would loosen hard on
most seeds, a symmetric one would make it tighter than the oracle on some.

## 4. Reference times and budgets

Source: oracle matrix E1 (2026-11-16, `python -m roboclock eval --policy oracle --oracle
--inference-latency zero --control-delay 0`, seeds 0-9 easy; stream tasks at every tier), slowest
successful seed, rounded up (0.01 s; lab 0.001 s per item). Failed oracle seeds (household C3
seed 9, industry C4 seeds 0 and 2, sports C1 seed 9: all stall stops, not timing) are reported as
oracle failures and not used. lab C1 seeds 8–9 (3.73, 3.55 s per item) were run later and are faster than 3.977 s.
Until 2026-11-16 t_o came from the recorded demos (`$DEMOS_DIR/roboclock_oracle_v1`, `66e69a9`);
the E1 values differ from those by -7 % to +1.6 %.

| Task | Q | t_o (s) | slowest seed | T easy / medium / hard (s) | H (s) |
|---|---|---|---|---|---|
| hospital C1 | 8 | 16.09 | seed 7, all 8 orders | 64.4 / 36.0 / 20.1 | 80.5 |
| hospital C2 | 6 | 18.27 | seed 2, all 6 items | 73.1 / 40.9 / 22.8 | 91.3 |
| hospital C3 | 12 | 28.44 | seed 1, 12 vials | 113.8 / 63.6 / 35.5 | 142.2 |
| hospital C4 | 12 | 26.34 | seed 8, 12 tubes | 105.4 / 58.9 / 32.9 | 131.7 |
| industry C1 | 2 kits | 64.50 | seed 4, 2nd kit | 258.0 / 144.2 / 80.6 | 322.5 |
| industry C2 | 3 cartons | 45.32 | seed 4, 3rd carton | 181.3 / 101.3 / 56.6 | 226.6 |
| industry C3 | 3 parts | 69.36 | seed 4, 3rd part (re-measured 2026-11-19 as a weighing task; was 83.20) | 277.4 / 155.1 / 86.7 | 346.8 |
| industry C4 | 2 palettes | 81.46 | seed 4, 2nd palette | 325.8 / 182.2 / 101.8 | 407.3 |
| lab C1 | N = 20-30 | 3.977 per item | max t_Q/Q, seed 5 | 15.908 N / 8.893 N / 4.971 N | 19.89 N |
| lab C2 | N = 20-30 | 2.807 per item | max t_Q/Q, seed 1 | 11.228 N / 6.277 N / 3.509 N | 14.04 N |
| lab C3 | N = 20-30 | 3.025 per item | max t_Q/Q, seed 1 | 12.100 N / 6.764 N / 3.781 N | 15.12 N |
| household C2 | 6 glasses | 43.21 | seed 9, finish_time | 172.8 / 96.6 / 54.0 | 216.1 |
| household C3 | 6 utensils | 25.97 | seed 7, 6 utensils | 103.9 / 58.1 / 32.5 | 129.8 |
| sports C1 | 15 balls | 127.49 | seed 4, 15th ball | 510.0 / 285.1 / 159.4 | 637.5 |
| sports C2 | 2 cards | 31.41 after the 2 s preparation | seed 9, 2nd card minus 2.0 s preparation | 125.6 / 70.2 / 39.3 | 2 + 157.1 |
| household C1 | 4 orders | pickups 10.53 + 12.96 k (k = 0-3), reference | E1: every order met at every tier | last pickup 197.64 / 110.48 / 61.76 | 247.05 / 138.11 / 77.20 |
| sports C3 | 6 demands | demands 6.44 + 3.57 k (k = 0-5), reference | E1: every demand met at every tier | last demand 81.65 / 49.79 / 30.36 | 102.06 / 62.24 / 37.95 |

The table lives in code as `roboclock.evaluation.throughput.ORACLE_TIMES`.

**Stream reference schedules** (2026-11-09; tier factors 2026-11-16). The demos could not be used, because on their loose
schedules the oracle waits: household C1's board holds one bowl until its pickup, and sports C3's
4-cell hopper is filled during the old 18-20 s preparation. Both were measured with every
deadline moved before the oracle could reach it, so it never waited, on seeds 0-6 and 50-56.

- household C1: 4 orders at every tier (hard used to have 3). The first meal is on the board at
  10.524 s on every seed; the relay then takes 12.5-12.9 s per meal depending on the counter
  slot. The reference schedule picks up at 10.53 + 12.96 k s (hard: 1.25 x). The binding job is seed 2's second order (23.484 s at
  hard): the 12.94 s interval from the free runs alone left it 0.014 s late, because at hard the
  waiter frees the board earlier and moving a pickup shifts later set-downs by up to 0.1 s. At
  the final schedule all 14 seeds served every order by its pickup (smallest slack 0.006 s), as
  did seeds 0, 2 and 54 at medium and easy. Frozen in `c1_x5/task.py` (`HARD_PICKUPS_S`); the
  measurement tool is `scripts/household/c1_oracle_free_run.py`.
  Verified 2026-11-16 at the released tiers (4 / sqrt5 / 1.25 x this schedule): easy and medium
  seeds 0-4 and hard seeds 0-9 serve every order (hard slack >= 2.64 s), except hard seed 3, where
  the oracle stalls placing the third bowl on the mat (insert_Left step timeout at 39.36 s, two
  identical reruns): a manipulation failure that the later hard pickups expose, not a late meal.
- sports C3: every demand owed from the start (`configs/c3_eager.yaml`), so each ball is fired as
  soon as it is accepted. The slowest k-th ball times are 6.440 / 9.144 / 13.564 / 16.068 /
  20.236 / 23.220 s; the first demand is due at 6.44 s and the interval is max_k (a_k - 6.44) / k
  = 3.562 -> 3.57 s (schedules that start later could fire faster only because the hopper
  absorbs a burst). This replaced 18 + U(0, 2) s then 9 / 7 / 5 s. The 4-cell hopper stays; at
  hard the oracle never banks a ball. At the final schedule all 14 seeds served every demand on
  time (seed 2's first ball exactly at its deadline), as did seeds 0, 2 and 6 at medium and easy.
  Frozen in `ball_service/config.py` (`HARD_PREPARATION`, `HARD_INTERVAL`); derived by
  `scripts/sports/c3_hard_schedule.py`.

## 5. Result rows

A C-class row has `score_version: 2` and, from `throughput.judge` / `judge_stream`:
`job_times_s`, `jobs_done`, `jobs_required`, `t_q_s` (= `completion_time`), and for batch tasks
`oracle_time_s`, `time_origin_s`, `budget_s_by_tier`, `horizon_s`,
`temporal_success_by_tier`, `joint_success_by_tier`; the scorer adds `score_by_tier`.
`temporal_success`, `joint_success` and `score` are the requested difficulty's tier, so a sweep
that runs only `easy` still gets medium and hard from the same episodes.
`termination_reason` is `all_done`, `stall`, `failure` or `horizon`.
Stream rows (`judge_stream`) carry `job_deadlines_s` and `late_jobs` instead of the tier fields;
their `t_q_s` is the latest job time (sports C3: the last ball stocked), which can differ from
the domain's own `completion_time` (sports C3: the last outlet passage).

## 6. Migration

- Results with score_version 1 (no field) cannot be rescored for physical: their episodes stopped
  at the old T. C-class results must be rerun.
- Oracle demos stay valid as trajectories, but their instructions state the old budgets, and the
  household C3 scene changes. Resume demo collection only after this lands; re-record household C3.
