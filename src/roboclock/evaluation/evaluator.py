"""Seeded benchmark runner and JSONL episode/event artifacts.

Time model (block-and-replay). Simulation is blocked while the policy computes, and the
inference time is then replayed in simulation. Each cycle:

1. The policy receives the observation sampled at environment time t_obs. Its camera images
   render when the policy first reads them (``roboclock.envs.observation``), at t_obs, so a
   chunked policy costs one render per query and none for the steps its chunk plays out.
2. Simulation is paused while act() runs. The inference latency L is the model time the
   policy reports in ``last_inference_latency`` (e.g. a remote server's own compute time, which
   leaves out network transport), or else the wall-clock time of act() minus the time spent
   rendering the observation it read.
3. The environment advances L + delta under the previous joint targets
   (``env.advance_latency``, rounded up to a physics tick). delta is the plan's fixed
   ``control_delay``.
4. The action computed from the observation at t_obs is applied for one policy period: one
   environment step, or with ``policy_hz`` 25 two steps with the targets ramped linearly from
   the previous action. The next observation is the one at the end of the last step.

``inference_latency="zero"`` charges no inference time (delta still applies), which isolates
policy quality from policy speed; each query's latency is still recorded (``_Calls``). ``inference_latency="ideal"`` does not use the time a query
took on our hardware. It charges the time the policy's model would take at the fastest surveyed
decode speed in ``model_speeds.yaml``: L = ttft_s + output_tokens / max_tokens_per_s. The
policy names its served model in ``inference_model`` and reports each query's output tokens,
thinking included, in ``last_output_tokens``. Nothing else that act() does is charged in this
mode.

A chunked policy sets ``queried`` False on cycles that return
an action from a chunk it already holds; those cycles are charged neither L nor delta. The
built-in hold and oracle diagnostics are always charged zero inference time.

Time model (streaming, ``time_model="streaming"``). The environment emits an observation every
policy period and keeps executing the agent's actions while it computes; the agent and the
simulator advance in lockstep, so simulator cost only adds wall-clock time. See
``_run_streaming``.
"""

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from roboclock.envs.registry import get_environment
from roboclock.envs.start_pose import start_pose
from roboclock.evaluation.metrics import aggregate
from roboclock.evaluation.score import score_fields
from roboclock.policies.base import load_policy
from roboclock.tasks.registry import task_registry

# "measured" charges each query's inference latency to environment time; "zero" does not;
# "ideal" charges the latency at the model's fastest surveyed decode speed.
INFERENCE_LATENCY_MODES = ("measured", "zero", "ideal")
# Fastest surveyed decode speed per served model name, for "ideal".
MODEL_SPEEDS = Path(__file__).with_name("model_speeds.yaml")
# Built-in diagnostics stand in for an instantaneous controller.
DIAGNOSTIC_POLICIES = ("hold", "oracle")
# --time-model value -> manifest/row name.
TIME_MODELS = {"block": "block_and_replay", "streaming": "streaming"}


class RestartEpisode(Exception):
    """Raised by a policy's act() to abandon the attempt and run the same episode again."""


def episode_plan(
    tasks=None,
    difficulties=None,
    seeds=range(10),
    delays=(0.0,),
    *,
    environment="lab",
    interface_delay=0.0,
    inference_budgets=None,
):
    """The episodes to run. ``delays`` are the total delays D charged per model query.

    A latency ablation gives the split as well: ``inference_budgets`` are the budgets L that
    produced ``delays`` (one per delay, in the same order) and ``interface_delay`` is the
    control interface's own delay delta, so that D = L + delta is recorded as its parts and not
    only as its sum. A plot may then say which part is inference time (docs/ablations.md).
    """
    spec = get_environment(environment)
    if inference_budgets is not None and len(inference_budgets) != len(delays):
        raise ValueError("inference_budgets must have one budget per delay")
    budget_of = (
        {}
        if inference_budgets is None
        else {delay: budget for delay, budget in zip(delays, inference_budgets)}
    )
    return [
        {
            "environment": environment,
            "task_id": task,
            "difficulty": difficulty,
            "seed": seed,
            "control_delay": delay,
            "inference_budget_s": budget_of.get(delay),
            "interface_delay_s": float(interface_delay),
        }
        for task, difficulty, seed, delay in itertools.product(
            tasks or task_registry(environment), difficulties or spec.difficulties, seeds, delays
        )
    ]


def latency_mode(policy_name, inference_latency):
    """Inference latency mode in effect for a policy; diagnostics are never charged."""
    if inference_latency not in INFERENCE_LATENCY_MODES:
        raise ValueError(f"inference_latency must be one of {', '.join(INFERENCE_LATENCY_MODES)}")
    return "zero" if policy_name in DIAGNOSTIC_POLICIES else inference_latency


def load_model_speeds(path=None):
    """{served model name: {"max_tokens_per_s", "ttft_s"}} from a model_speeds.yaml file."""
    path = Path(path or MODEL_SPEEDS)
    try:
        entries = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models") or {}
    except (yaml.YAMLError, AttributeError) as error:
        raise ValueError(f"{path}: not a model speeds file ({error})") from error
    speeds = {}
    for model, entry in entries.items():
        try:
            rate, ttft = float(entry["max_tokens_per_s"]), float(entry.get("ttft_s", 0.0))
        except (AttributeError, KeyError, TypeError, ValueError):
            rate = ttft = math.nan
        if not (math.isfinite(rate) and rate > 0 and math.isfinite(ttft) and ttft >= 0):
            raise ValueError(f"{path}: {model} needs max_tokens_per_s > 0 and ttft_s >= 0")
        speeds[model] = {"max_tokens_per_s": rate, "ttft_s": ttft}
    return speeds


def model_speed(policy, speeds):
    """The surveyed speed of the model a policy queries, for ideal latency."""
    model = getattr(policy, "inference_model", None)
    if model is None:
        raise ValueError(
            "Ideal inference latency needs a policy that names its model in inference_model"
        )
    if model not in speeds:
        raise ValueError(f"No surveyed decode speed for {model!r}; add it to {MODEL_SPEEDS.name}")
    return speeds[model]


def output_tokens(policy):
    """Output tokens the query act() just made reports (thinking included), or None."""
    tokens = getattr(policy, "last_output_tokens", None)
    if tokens is None:
        return None
    tokens = float(tokens)
    return tokens if math.isfinite(tokens) and tokens >= 0 else None


def query_latency(policy, wall_time, speeds=None):
    """Inference latency of the query act() just made. With ``speeds`` (ideal mode), its output
    tokens at the model's surveyed speed; otherwise the policy's own report, else wall time."""
    if speeds is not None:
        speed = model_speed(policy, speeds)
        tokens = output_tokens(policy)
        if tokens is None:
            raise ValueError(
                "Ideal inference latency needs each query's output token count "
                "in last_output_tokens"
            )
        return speed["ttft_s"] + tokens / speed["max_tokens_per_s"]
    reported = getattr(policy, "last_inference_latency", None)
    if reported is None:
        return wall_time
    reported = float(reported)
    if not math.isfinite(reported) or reported < 0:
        raise ValueError("Policy reported a negative or non-finite inference latency")
    return reported


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def control_steps_per_action(env, policy_name, policy_hz):
    """Environment steps one policy action spans; None means the control rate."""
    if policy_hz is None:
        return 1
    sim = env.config["sim"]
    control_hz = 1 / (sim["physics_dt"] * sim["control_decimation"])
    steps = control_hz / policy_hz if policy_hz > 0 else 0.0
    if steps < 1 or not math.isclose(steps, round(steps)):
        raise ValueError(f"Policy rate {policy_hz} Hz must divide the {control_hz:g} Hz control rate")
    if policy_name == "oracle" and round(steps) != 1:
        # Oracles integrate their rate limits on every control tick.
        raise ValueError("The diagnostic oracle runs at the control rate")
    return round(steps)


def execute_action(env, obs, action, previous, steps):
    """Run one policy action, ramping the joint targets from the previous action to it."""
    for step in range(1, steps + 1):
        target = action if step == steps else previous + (action - previous) * step / steps
        obs = env.step(target)[0]
        if env.done:
            break
    return obs


class _Calls:
    """Times policy calls for one episode. Rendering an observation the policy reads during a
    call is simulator work (Delta_sim), so it is subtracted from the call's wall time.
    ``speeds`` (ideal mode) makes each query's latency its output tokens at the model's
    surveyed speed.

    Every mode also records, per query, its observation time, its measured latency (the
    policy's report, else its wall time), its output tokens and, when ``survey`` has the
    policy's model, its ideal latency. A zero-latency run then still shows what each query
    took and what it would have been charged; ideal-mode rows show the real time too."""

    def __init__(self, speeds=None, survey=None):
        self.wall_times, self.latencies = [], []
        self.times, self.measured, self.tokens, self.ideal = [], [], [], []
        self.render_time = 0.0
        self.rendered = 0
        self.speeds = speeds
        self.survey = speeds if speeds is not None else survey

    def query(self, policy, wall_time, t_obs=None):
        """Record the query act() just made from the observation at `t_obs`, and return its
        inference latency (ideal in ideal mode, else measured; zero mode does not charge it)."""
        measured = query_latency(policy, wall_time)
        ideal_mode = self.speeds is not None
        latency = query_latency(policy, wall_time, self.speeds) if ideal_mode else measured
        tokens = output_tokens(policy)
        ideal = None
        if self.survey and getattr(policy, "inference_model", None) in self.survey:
            if tokens is not None:
                ideal = query_latency(policy, wall_time, self.survey)
        self.latencies.append(latency)
        self.times.append(None if t_obs is None else round(float(t_obs), 6))
        self.measured.append(measured)
        self.tokens.append(tokens)
        self.ideal.append(ideal)
        return latency

    def call(self, method, obs):
        rendered_before = getattr(obs, "render_seconds", 0.0)
        start = time.perf_counter()
        result = method(obs)
        wall_time = time.perf_counter() - start
        render = getattr(obs, "render_seconds", 0.0) - rendered_before
        wall_time -= render
        self.render_time += render
        self.rendered += bool(getattr(obs, "image_keys", ())) and getattr(obs, "rendered", False)
        self.wall_times.append(wall_time)
        return result, wall_time

    def row(self):
        ideal = [value for value in self.ideal if value is not None]
        return dict(
            queries=len(self.latencies),
            mean_inference_latency=_mean(self.latencies),
            max_inference_latency=max(self.latencies, default=0.0),
            mean_policy_wall_time=_mean(self.wall_times),
            # Observations whose images a policy read (each rendered once, at its t_obs).
            rendered_observations=self.rendered,
            render_wall_time=self.render_time,
            # Per query, in every latency mode (charged or not).
            query_times_s=self.times,
            measured_inference_latencies_s=self.measured,
            mean_measured_inference_latency=_mean(self.measured),
            **(
                {"query_output_tokens": self.tokens}
                if any(t is not None for t in self.tokens)
                else {}
            ),
            **(
                {
                    "ideal_inference_latencies_s": self.ideal,
                    "mean_ideal_inference_latency": _mean(ideal),
                }
                if ideal
                else {}
            ),
        )


def _run_block(env, policy, obs, delay, mode, steps, calls):
    """Block-and-replay: see the module docstring."""
    previous = (
        np.clip(obs["joint_positions"], env.action_space.low, env.action_space.high)
        if steps > 1
        else None
    )
    latency_time = 0.0
    while not env.done:
        t_obs = env.task.time
        action, wall_time = calls.call(policy.act, obs)
        charged = 0.0
        # A chunked policy answers the other cycles from its chunk.
        if getattr(policy, "queried", True):
            latency = calls.query(policy, wall_time, t_obs)
            charged = (0.0 if mode == "zero" else latency) + delay
        if charged:
            before = env.task.time
            env.advance_latency(charged)
            latency_time += env.task.time - before
        # The action was computed from the pre-latency observation, as in deployment.
        if not env.done:
            action = np.asarray(action, dtype=np.float32)
            obs = execute_action(env, obs, action, previous, steps)
            previous = action
    # Environment time spent under held targets: charged latency plus delay.
    return {"latency_sim_time": latency_time}


def _active_action(chunks, tick, decimation):
    """(action, (query, index), next tick it may change) of the stream at `tick`, or Nones
    before the first action. Chunk actions are one control period apart; the last one holds."""
    current = None
    for chunk in chunks:
        if chunk[0] <= tick:
            current = chunk
    later = [chunk[0] for chunk in chunks if chunk[0] > tick]
    change = min(later, default=math.inf)
    if current is None:
        return None, None, change
    start, actions, query = current
    index = min((tick - start) // decimation, len(actions) - 1)
    if index < len(actions) - 1:
        change = min(change, start + (index + 1) * decimation)
    return actions[index], (query, index), change


def _run_streaming(env, policy, obs, delay, mode, steps, calls):
    """Streaming: the environment emits an observation every policy period dt at t = n dt and
    keeps executing the agent's action stream while the agent computes.

    The agent and the simulator advance in lockstep, so agent time equals environment time and
    rendering (or any simulator cost) never counts toward the agent. The lockstep is evaluated
    as the equivalent event sequence: when the agent is idle it takes the newest observation it
    has not consumed (o_n during [n dt, (n+1) dt), else it waits for the next one) and starts a
    query at tau_s; the query's inference latency L (the policy's own report or its wall time
    without rendering, its ideal time in "ideal" mode, 0 in "zero" mode) gives
    tau_out = tau_s + L. Its actions take effect at t_act = tau_out + delta, at the first
    physics tick at or after t_act, one control period apart, replacing whatever the previous
    query emitted from that tick on; the last action holds. Before the first action the targets set at reset stay in effect.

    Policies: ``act_chunk(obs)`` returns a chunk (H x action_dim, at the control rate);
    otherwise ``act(obs)`` returns one action. Optional ``wants_query(remaining)`` lets an idle
    agent skip an observation (remaining: control steps left in its current chunk), and
    ``on_action(query, index)`` is told when an action takes effect.
    """
    sim = env.config["sim"]
    dt = sim["physics_dt"]
    decimation = int(sim["control_decimation"])
    interval = decimation * steps  # physics ticks per observation period
    period = interval * dt
    act = getattr(policy, "act_chunk", None) or (lambda o: np.asarray(policy.act(o))[None])
    wants_query = getattr(policy, "wants_query", None)
    on_action = getattr(policy, "on_action", None)
    chunks = []  # (start tick, actions, query), start ticks non-decreasing
    tick = 0  # physics ticks since reset; some clocks do not start at 0 (industry settles)
    t0 = env.task.time
    idle_from = 0.0  # tau at which the agent is idle
    consumed = -1  # index of the newest observation the agent has taken
    applied = None
    action_delays = []
    while not env.done:
        n = tick // interval
        t_obs = n * period
        if n > consumed and idle_from < t_obs + period - 1e-9:
            consumed = n
            remaining = 0
            if chunks:
                start, actions, _ = chunks[-1]
                remaining = max(0, len(actions) - 1 - max(0, tick - start) // decimation)
            if wants_query is None or wants_query(remaining):
                if n:  # the reset observation was already made
                    obs = env.get_obs()
                actions, wall_time = calls.call(act, obs)
                actions = np.asarray(actions, dtype=np.float32).reshape(-1, env.action_space.shape[0])
                if not len(actions):
                    raise ValueError("Policy returned an empty action chunk")
                latency = calls.query(policy, wall_time, t0 + t_obs)
                tau_out = max(idle_from, t_obs) + (0.0 if mode == "zero" else latency)
                start = max(tick, math.ceil((tau_out + delay) / dt - 1e-9))
                chunks = [c for c in chunks if c[0] < start] + [(start, actions, len(calls.latencies) - 1)]
                action_delays.append(start * dt - t_obs)
                idle_from = tau_out
        end = (n + 1) * interval
        while tick < end and not env.done:
            action, key, change = _active_action(chunks, tick, decimation)
            if key is not None and key != applied:
                env.set_action(action)
                applied = key
                if on_action is not None:
                    on_action(*key)
            stop = min(end, change)
            env.advance_latency((stop - tick) * dt)
            tick = stop
            if not env.done and abs(env.task.time - t0 - tick * dt) > dt / 2:
                raise RuntimeError(
                    f"{env.environment_id} clock at {env.task.time:.6f} s, "
                    f"expected {t0 + tick * dt:.6f} s"
                )
        # Chunks wholly superseded by a later one that is already in effect are done.
        while len(chunks) > 1 and chunks[1][0] <= tick:
            chunks.pop(0)
    return {"stream_interval": period, "mean_action_delay": _mean(action_delays)}


def evaluate(
    env,
    plan,
    policy_name,
    output_dir,
    *,
    inference_latency="measured",
    policy_hz=None,
    time_model="block",
    model_speeds=None,
):
    """Run the plan and write episodes.jsonl, manifest.json and summary.json. ``model_speeds``
    is the speeds file for ideal latency (default: the surveyed model_speeds.yaml)."""
    mode = latency_mode(policy_name, inference_latency)
    # Model policies get their ideal latency recorded in every mode; it is charged only in
    # ideal mode. The built-in diagnostics query no model.
    survey = None if policy_name in DIAGNOSTIC_POLICIES else load_model_speeds(model_speeds)
    speeds = survey if mode == "ideal" else None
    if time_model not in TIME_MODELS:
        raise ValueError(f"time_model must be one of {', '.join(TIME_MODELS)}")
    run = _run_streaming if time_model == "streaming" else _run_block
    if any(item.get("environment", env.environment_id) != env.environment_id for item in plan):
        raise ValueError("Evaluation plan contains episodes for a different environment")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Fail before simulation if this run would overwrite another run's artifacts.
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        with (output / "manifest.json").open("x", encoding="utf-8") as manifest:
            json.dump(
                {
                    "benchmark_version": "0.1.0",
                    **env.benchmark_metadata(),
                    "environment": env.environment_id,
                    "policy": policy_name,
                    "oracle": env.oracle,
                    "plan": plan,
                    "config": env.config,
                    "time_model": TIME_MODELS[time_model],
                    "inference_latency": mode,
                    # The speeds of the ideal latencies (charged only in ideal mode).
                    **({"model_speeds": survey} if survey is not None else {}),
                    "policy_hz": policy_hz,
                },
                manifest,
                indent=2,
            )
        for index, episode in enumerate(plan):
            attempt = 0
            while True:
                attempt += 1
                obs, _ = env.reset(seed=episode["seed"], options=episode)
                policy = load_policy(policy_name, env)
                policy.reset(obs)
                if speeds is not None:
                    model_speed(policy, speeds)  # an unsurveyed model fails before the episode runs
                steps = control_steps_per_action(env, policy_name, policy_hz)
                delay = episode["control_delay"]
                calls = _Calls(speeds, survey)
                try:
                    extra = run(env, policy, obs, delay, mode, steps, calls)
                    restarted = False
                except RestartEpisode:  # an operator abandoned the attempt (teleop)
                    extra, restarted = {}, True
                row = env.task.metrics()
                row.update(env.episode_metadata())
                row.update(
                    environment=env.environment_id,
                    policy=policy_name,
                    oracle=env.oracle,
                    time_model=TIME_MODELS[time_model],
                    start_pose=start_pose(env.config),
                    control_delay=delay,
                    # D = L + delta, as its parts, where the run gave them (a latency ablation).
                    inference_budget_s=episode.get("inference_budget_s"),
                    interface_delay_s=episode.get("interface_delay_s"),
                    inference_latency_mode=mode,
                    control_steps_per_action=steps,
                    **calls.row(),
                    **extra,
                    # C-class tasks (score_version 2) name their own reason: all_done, stall,
                    # failure or horizon.
                    termination_reason=row.get("termination_reason")
                    or ("task_or_drop" if env.task.terminated else "time_budget"),
                )
                row["events"] = env.task.event_log
                row["instruction"] = env.task.gen_instruction()
                # The environments step until time >= max_time, so a horizon that is not a whole number of
                # physics steps (C-class H = 5 t_o) runs to the next step: ceil, not round-half-even.
                row["max_physics_steps"] = math.ceil(
                    env.task.spec.max_time / env.config["sim"]["physics_dt"] - 1e-6
                )
                # Stage score in the manner of RoboDojo (roboclock.evaluation.score).
                row.update(score_fields(env.environment_id, row))
                if attempt > 1 or restarted or hasattr(policy, "episode_finished"):
                    row["attempt"] = attempt
                if restarted:
                    row["termination_reason"] = "restarted"
                else:
                    # Optional: a policy that shows the outcome to someone (the teleop page)
                    # and may ask for the episode again.
                    finished = getattr(policy, "episode_finished", None)
                    restarted = finished is not None and finished(row) == "restart"
                if not restarted:
                    break
                # A redone attempt is kept apart: episodes.jsonl holds one row per plan entry.
                with (output / "restarted_attempts.jsonl").open("a", encoding="utf-8") as redo:
                    redo.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"[{index + 1}/{len(plan)}] {row['task_id']} attempt {attempt} restarted", flush=True)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(
                f"[{index + 1}/{len(plan)}] {row['task_id']} {row['difficulty']} seed={row['seed']} "
                f"physical={row['task_success']} temporal={row['temporal_success']}",
                flush=True,
            )
    summary = aggregate(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
