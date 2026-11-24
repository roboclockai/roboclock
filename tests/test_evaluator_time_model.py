"""Block-and-replay evaluation loop: what it charges to environment time, and in what order."""

import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from roboclock.evaluation import evaluator

PHYSICS_DT = 0.004
DECIMATION = 5  # 50 Hz environment steps


class FakeEnv:
    """Records the evaluator's calls on a 250 Hz physics clock."""

    environment_id = "lab"
    oracle = False
    config = {"sim": {"physics_dt": PHYSICS_DT, "control_decimation": DECIMATION}}

    def __init__(self, max_time=0.2):
        self.max_time = max_time
        self.action_space = gym.spaces.Box(-10.0, 10.0, (1,), dtype=np.float32)
        self.calls = []

    def reset(self, seed, options):
        self.ticks = 0
        self.done = False
        self.target = None  # the targets in effect (first action entry), None before any
        self.timeline = []  # target in effect during each physics tick
        self.task = SimpleNamespace(
            time=0.0,
            terminated=False,
            event_log=[],
            spec=SimpleNamespace(max_time=self.max_time),
            gen_instruction=lambda: "wait",
            metrics=lambda: {
                "task_id": "A1",
                "difficulty": "easy",
                "seed": seed,
                "category": "temporal_window",
                "task_success": False,
                "manipulation_success": False,
                "temporal_success": False,
                "joint_success": False,
                "temporal_violations": [],
            },
        )
        return self.observation(), {}

    def observation(self):
        return {"joint_positions": np.zeros(1, dtype=np.float32), "time": self.task.time}

    def get_obs(self):
        return self.observation()

    def _advance(self, ticks):
        budget = round(self.max_time / PHYSICS_DT)
        self.timeline += [self.target] * (min(self.ticks + ticks, budget) - self.ticks)
        self.ticks = min(self.ticks + ticks, budget)
        self.task.time = self.ticks * PHYSICS_DT
        self.done = self.ticks >= budget

    def advance_latency(self, seconds):
        assert not self.done
        self.calls.append(("latency", round(seconds, 9)))
        self._advance(int(np.ceil(seconds / PHYSICS_DT - 1e-12)))

    def set_action(self, action):
        assert not self.done
        self.target = float(np.asarray(action)[0])
        self.calls.append(("set", self.target))

    def step(self, action):
        assert not self.done
        self.calls.append(("step", float(np.asarray(action)[0])))
        self.target = float(np.asarray(action)[0])
        self._advance(DECIMATION)
        return self.observation(), 0.0, False, self.done, {}

    def benchmark_metadata(self):
        return {}

    def episode_metadata(self):
        return {}

    def latencies(self):
        return [value for kind, value in self.calls if kind == "latency"]


class CountingPolicy:
    """Returns 1, 2, 3, ... and records the environment time of each observation it sees."""

    def __init__(self, reported=None):
        self.reported = reported
        self.seen = []

    def reset(self, observation):
        self.count = 0

    def act(self, observation):
        self.seen.append(observation["time"])
        self.count += 1
        if self.reported is not None:
            self.last_inference_latency = self.reported
        return np.full(1, self.count, dtype=np.float32)


class ChunkedPolicy(CountingPolicy):
    """Queries its model every `chunk` cycles and replays the chunk in between."""

    def __init__(self, chunk):
        super().__init__()
        self.chunk = chunk

    def reset(self, observation):
        super().reset(observation)
        self.left = 0

    def act(self, observation):
        self.queried = self.left == 0
        if self.queried:
            self.left = self.chunk
        self.left -= 1
        return super().act(observation)


@pytest.fixture
def wall_clock(monkeypatch):
    """Make every act() take `seconds` of wall-clock time."""

    def install(seconds):
        # The evaluator reads the clock right before and right after each act().
        state = {"start": True, "now": 0.0}

        def perf_counter():
            if not state["start"]:
                state["now"] += seconds
            state["start"] = not state["start"]
            return state["now"]

        monkeypatch.setattr(evaluator, "time", SimpleNamespace(perf_counter=perf_counter))

    return install


def run(monkeypatch, tmp_path, policy, *, name="module:factory", delay=0.0, **kwargs):
    env = FakeEnv(kwargs.pop("max_time", 0.2))
    monkeypatch.setattr(evaluator, "load_policy", lambda _name, _env: policy)
    plan = [{"task_id": "A1", "difficulty": "easy", "seed": 0, "control_delay": delay}]
    evaluator.evaluate(env, plan, name, tmp_path, **kwargs)
    row = json.loads((tmp_path / "episodes.jsonl").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    return env, row, manifest


def test_measured_latency_passes_under_the_previous_targets_before_the_action(
    monkeypatch, tmp_path, wall_clock
):
    wall_clock(0.030)
    policy = CountingPolicy()
    env, row, manifest = run(monkeypatch, tmp_path, policy, delay=0.010)
    # 30 ms inference + 10 ms delay = 10 physics ticks held, then one 20 ms step: 60 ms cycles.
    assert env.calls[:4] == [("latency", 0.04), ("step", 1.0), ("latency", 0.04), ("step", 2.0)]
    # Each action is computed from the observation at the end of the previous action.
    assert policy.seen[:4] == pytest.approx([0.0, 0.06, 0.12, 0.18])
    assert manifest["time_model"] == "block_and_replay"
    assert manifest["inference_latency"] == "measured"
    assert row["inference_latency_mode"] == "measured"
    assert row["control_delay"] == 0.010
    assert row["queries"] == 4
    assert row["mean_inference_latency"] == pytest.approx(0.030)
    assert row["mean_policy_wall_time"] == pytest.approx(0.030)
    # The fourth cycle's latency runs into the 0.2 s budget, so no fourth action is applied.
    assert env.calls[-1] == ("latency", 0.04)
    assert row["latency_sim_time"] == pytest.approx(0.04 * 3 + 0.02)


def test_latency_rounds_up_to_a_physics_tick(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.0101)
    env, row, _ = run(monkeypatch, tmp_path, CountingPolicy(), max_time=0.032)
    assert env.calls[:2] == [("latency", 0.0101), ("step", 1.0)]
    assert row["latency_sim_time"] == pytest.approx(0.012)


def test_zero_mode_charges_only_the_delay(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.030)
    env, row, manifest = run(
        monkeypatch, tmp_path, CountingPolicy(), delay=0.008, inference_latency="zero"
    )
    assert set(env.latencies()) == {0.008}
    assert manifest["inference_latency"] == row["inference_latency_mode"] == "zero"
    # The latency is still measured and reported, only not charged.
    assert row["mean_inference_latency"] == pytest.approx(0.030)


def test_zero_mode_without_delay_is_the_plain_control_loop(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.030)
    policy = CountingPolicy()
    env, row, _ = run(monkeypatch, tmp_path, policy, inference_latency="zero")
    assert env.latencies() == []
    assert [kind for kind, _ in env.calls] == ["step"] * 10
    assert policy.seen == pytest.approx([0.02 * k for k in range(10)])
    assert row["latency_sim_time"] == 0.0


def test_chunked_policy_is_charged_only_when_it_queries(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.008)
    policy = ChunkedPolicy(chunk=3)
    env, row, _ = run(monkeypatch, tmp_path, policy, delay=0.004)
    kinds = [kind for kind, _ in env.calls]
    # 12 ms charged once per chunk of three 20 ms steps.
    assert kinds[:8] == ["latency", "step", "step", "step", "latency", "step", "step", "step"]
    assert set(env.latencies()) == {0.012}
    assert row["queries"] == 3
    assert row["latency_sim_time"] == pytest.approx(0.036)


def test_reported_model_time_replaces_the_wall_time(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.150)  # e.g. a round trip through a proxy
    env, row, _ = run(monkeypatch, tmp_path, CountingPolicy(reported=0.012))
    assert set(env.latencies()) == {0.012}
    assert row["mean_inference_latency"] == pytest.approx(0.012)
    assert row["mean_policy_wall_time"] == pytest.approx(0.150)


@pytest.mark.parametrize("reported", [-0.01, float("nan")])
def test_invalid_reported_latency_is_rejected(monkeypatch, tmp_path, wall_clock, reported):
    wall_clock(0.01)
    with pytest.raises(ValueError, match="inference latency"):
        run(monkeypatch, tmp_path, CountingPolicy(reported=reported))


@pytest.mark.parametrize("name", ["hold", "oracle"])
def test_diagnostics_are_never_charged_inference_time(monkeypatch, tmp_path, wall_clock, name):
    wall_clock(0.030)
    env, row, manifest = run(monkeypatch, tmp_path, CountingPolicy(), name=name, delay=0.004)
    assert set(env.latencies()) == {0.004}
    assert manifest["inference_latency"] == row["inference_latency_mode"] == "zero"


def test_unknown_latency_mode_is_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match="inference_latency"):
        evaluator.evaluate(FakeEnv(), [], "hold", tmp_path / "run", inference_latency="real")
    assert not (tmp_path / "run").exists()


def test_policy_rate_ramps_after_the_latency(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.004)
    env, row, _ = run(monkeypatch, tmp_path, CountingPolicy(), policy_hz=25, max_time=0.1)
    # One 4 ms tick held, then the 25 Hz action ramps over two steps from the previous one.
    assert env.calls[:3] == [("latency", 0.004), ("step", 0.5), ("step", 1.0)]
    assert row["control_steps_per_action"] == 2


def test_images_render_only_for_queries_and_rendering_is_not_inference(monkeypatch, tmp_path):
    """A chunked policy reads one observation per query: only those render, at their t_obs,
    and the render time inside act() is not charged as inference latency."""
    from roboclock.envs import observation as lazy

    clock = {"now": 0.0}
    fake_time = SimpleNamespace(perf_counter=lambda: clock["now"])
    monkeypatch.setattr(evaluator, "time", fake_time)
    monkeypatch.setattr(lazy, "time", fake_time)

    class CameraEnv(FakeEnv):
        def reset(self, seed, options):
            self.renders = []
            return super().reset(seed, options)

        def render(self):
            self.renders.append(round(self.task.time, 9))
            clock["now"] += 0.300  # rendering is slow
            return {"rgb": np.zeros((2, 2, 3), np.uint8)}

        def observation(self):
            state = super().observation()
            return lazy.Observation(state, ("rgb",), self.render, lambda: self.task.time)

    class ImagePolicy(ChunkedPolicy):
        def act(self, observation):
            action = super().act(observation)
            if self.queried:
                assert observation["rgb"].shape == (2, 2, 3)
                clock["now"] += 0.030  # the model itself
            return action

    env = CameraEnv(max_time=0.5)
    policy = ImagePolicy(chunk=3)
    monkeypatch.setattr(evaluator, "load_policy", lambda _name, _env: policy)
    plan = [{"task_id": "A1", "difficulty": "easy", "seed": 0, "control_delay": 0.0}]
    evaluator.evaluate(env, plan, "module:factory", tmp_path)
    row = json.loads((tmp_path / "episodes.jsonl").read_text())
    # 30 ms of inference rounds up to 8 ticks (32 ms); the 300 ms render is not charged.
    assert env.latencies() == pytest.approx([0.03] * row["queries"])
    assert row["mean_inference_latency"] == pytest.approx(0.030)
    # One render per query, each at the observation the query read; none for chunk steps.
    query_times = [t for t, cycle in zip(policy.seen, range(len(policy.seen))) if cycle % 3 == 0]
    assert env.renders == pytest.approx(query_times)
    assert row["rendered_observations"] == row["queries"] == len(env.renders)
    assert row["render_wall_time"] == pytest.approx(0.3 * len(env.renders))


# ------------------------------------------------------------------------- streaming


class ChunkPolicy(CountingPolicy):
    """Returns chunks [10k, 10k + 1, ...] of `length` actions and records on_action calls."""

    def __init__(self, length, threshold=None):
        super().__init__()
        self.length, self.threshold = length, threshold
        self.applied, self.remaining = [], []

    def act_chunk(self, observation):
        first = super().act(observation)[0] * 10
        return np.arange(first, first + self.length, dtype=np.float32)[:, None]

    def on_action(self, query, index):
        self.applied.append((query, index))

    def wants_query(self, remaining):
        self.remaining.append(remaining)
        return self.threshold is None or remaining <= self.threshold


def ticks_of(timeline):
    """{target: (first tick, last tick)} of a per-tick target timeline."""
    spans = {}
    for tick, target in enumerate(timeline):
        first, _ = spans.get(target, (tick, tick))
        spans[target] = (first, tick)
    return spans


def test_streaming_without_latency_is_the_plain_control_loop(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.030)
    block, _, _ = run(monkeypatch, tmp_path / "block", CountingPolicy(), inference_latency="zero")
    policy = CountingPolicy()
    streaming, row, manifest = run(
        monkeypatch, tmp_path / "streaming", policy, inference_latency="zero", time_model="streaming"
    )
    assert streaming.timeline == block.timeline  # same targets on every physics tick
    assert policy.seen == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18])
    assert manifest["time_model"] == row["time_model"] == "streaming"
    assert row["stream_interval"] == pytest.approx(0.02) and row["mean_action_delay"] == 0.0
    assert row["queries"] == 10 and "latency_sim_time" not in row


def test_streaming_keeps_acting_while_the_agent_computes(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.030)
    policy = CountingPolicy()
    env, row, _ = run(monkeypatch, tmp_path, policy, time_model="streaming")
    # tau_s = 0 on o_0, tau_out = 0.03; then idle at 0.03 inside [0.02, 0.04): o_1 at once;
    # tau_out = 0.06 = a new period: o_3; 0.09 inside [0.08, 0.1): o_4; 0.12: o_6; ...
    assert policy.seen[:6] == pytest.approx([0.0, 0.02, 0.06, 0.08, 0.12, 0.14])
    # Each action takes effect at the first tick at or after tau_out, and the previous one
    # stays in effect meanwhile (no held pause as in block-and-replay).
    spans = ticks_of(env.timeline)
    assert spans[None] == (0, 7)  # ceil(0.03 / 0.004) = 8
    assert spans[1.0] == (8, 14) and spans[2.0] == (15, 22) and spans[3.0] == (23, 29)
    assert not [call for call in env.calls if call[0] == "step"]
    assert row["mean_inference_latency"] == pytest.approx(0.03)
    # t_act - t_obs: 0.032, 0.06 - 0.02, 0.092 - 0.06, ...
    assert row["mean_action_delay"] == pytest.approx(
        np.mean([0.032, 0.04, 0.032, 0.04, 0.032, 0.04, 0.032][: row["queries"]]), abs=1e-9
    )


def test_streaming_chunks_replace_the_rest_of_the_previous_chunk(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.030)
    policy = ChunkPolicy(length=4)
    env, row, _ = run(monkeypatch, tmp_path, policy, time_model="streaming", max_time=0.1)
    spans = ticks_of(env.timeline)
    # Chunk 0 (10, 11, ...) from tick 8, one control period (5 ticks) per action, until
    # chunk 1 (20, 21, ...) takes over at tick 15.
    assert spans[10.0] == (8, 12) and spans[11.0] == (13, 14)
    assert spans[20.0] == (15, 19) and spans[21.0] == (20, 22) and spans[30.0][0] == 23
    assert policy.applied[:4] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_streaming_agents_may_skip_observations(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.0)
    # Zero latency, 3-action chunks; query only once the chunk is on its last action.
    policy = ChunkPolicy(length=3, threshold=0)
    env, row, _ = run(monkeypatch, tmp_path, policy, time_model="streaming")
    spans = ticks_of(env.timeline)
    assert spans[10.0] == (0, 4) and spans[11.0] == (5, 9) and spans[20.0] == (10, 14)
    assert 12.0 not in spans  # replaced as it was about to start
    assert policy.seen == pytest.approx([0.0, 0.04, 0.08, 0.12, 0.16])
    assert policy.remaining == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    assert row["queries"] == 5


def test_streaming_holds_the_last_action_of_a_chunk(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.100)
    policy = ChunkPolicy(length=2)
    env, row, _ = run(monkeypatch, tmp_path, policy, time_model="streaming")
    spans = ticks_of(env.timeline)
    # Chunk 0 arrives at 0.1 s (tick 25); chunk 1, queried on o_5, would arrive at 0.2 s.
    assert spans[None] == (0, 24) and spans[10.0] == (25, 29) and spans[11.0] == (30, 49)
    assert row["queries"] == 2 and policy.applied == [(0, 0), (0, 1)]


def test_streaming_control_delay_shifts_when_actions_take_effect(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.0)
    env, row, _ = run(
        monkeypatch, tmp_path, CountingPolicy(), time_model="streaming", delay=0.010
    )
    spans = ticks_of(env.timeline)
    assert spans[None] == (0, 2) and spans[1.0] == (3, 7) and spans[2.0] == (8, 12)
    assert row["mean_action_delay"] == pytest.approx(0.012)


def test_streaming_rejects_unknown_time_models(tmp_path):
    with pytest.raises(ValueError, match="time_model"):
        evaluator.evaluate(FakeEnv(), [], "hold", tmp_path, time_model="async")


class TokenPolicy(CountingPolicy):
    """A model policy: names its served model and reports each query's output tokens."""

    inference_model = "test/model"

    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens

    def act(self, observation):
        self.last_output_tokens = self.tokens
        return super().act(observation)


@pytest.fixture
def speeds(tmp_path):
    path = tmp_path / "speeds.yaml"
    path.write_text("models:\n  test/model:\n    max_tokens_per_s: 1000\n    ttft_s: 0.002\n")
    return path


def test_ideal_latency_charges_output_tokens_at_the_surveyed_speed(
    monkeypatch, tmp_path, wall_clock, speeds
):
    wall_clock(5.0)  # what the query took on our hardware is not charged
    env, row, manifest = run(
        monkeypatch,
        tmp_path / "run",
        TokenPolicy(tokens=30),
        delay=0.004,
        inference_latency="ideal",
        model_speeds=speeds,
    )
    # 0.002 s to the first token + 30 tokens at 1000 tokens/s, then the 4 ms delay.
    assert env.calls[:2] == [("latency", 0.036), ("step", 1.0)]
    assert row["mean_inference_latency"] == pytest.approx(0.032)
    assert row["mean_policy_wall_time"] == pytest.approx(5.0)
    assert manifest["inference_latency"] == row["inference_latency_mode"] == "ideal"
    assert manifest["model_speeds"] == {"test/model": {"max_tokens_per_s": 1000.0, "ttft_s": 0.002}}


def test_ideal_latency_applies_to_streaming(monkeypatch, tmp_path, wall_clock, speeds):
    speeds.write_text("models:\n  test/model:\n    max_tokens_per_s: 1000\n")
    wall_clock(1.0)
    env, row, _ = run(
        monkeypatch,
        tmp_path / "run",
        TokenPolicy(tokens=30),
        inference_latency="ideal",
        model_speeds=speeds,
        time_model="streaming",
    )
    spans = ticks_of(env.timeline)
    assert spans[None] == (0, 7) and spans[1.0][0] == 8  # tau_out = 30 / 1000 s
    assert row["mean_inference_latency"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    "policy, match",
    [
        (CountingPolicy(), "inference_model"),
        (SimpleNamespace(inference_model="other/model"), "No surveyed decode speed"),
        (TokenPolicy(tokens=None), "output token count"),
    ],
)
def test_ideal_latency_needs_a_surveyed_model_and_token_counts(
    monkeypatch, tmp_path, wall_clock, speeds, policy, match
):
    if isinstance(policy, SimpleNamespace):
        policy.reset = lambda observation: None
    wall_clock(0.01)
    with pytest.raises(ValueError, match=match):
        run(monkeypatch, tmp_path / "run", policy, inference_latency="ideal", model_speeds=speeds)


def test_diagnostics_ignore_ideal_latency(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.030)
    env, row, manifest = run(
        monkeypatch, tmp_path, CountingPolicy(), name="hold", inference_latency="ideal"
    )
    assert env.latencies() == []
    assert manifest["inference_latency"] == "zero" and "model_speeds" not in manifest


@pytest.mark.parametrize(
    "text",
    ["models:\n  m:\n    max_tokens_per_s: 0\n", "models:\n  m:\n    ttft_s: 0.1\n", "[1, 2"],
)
def test_malformed_model_speeds_are_rejected(tmp_path, text):
    path = tmp_path / "speeds.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="max_tokens_per_s > 0|not a model speeds file"):
        evaluator.load_model_speeds(path)


def test_the_surveyed_speeds_cover_the_vision_llm_agents(monkeypatch):
    from roboclock.policies import vision_llm, vision_llm_primitives

    monkeypatch.delenv("VISION_LLM_MODEL", raising=False)
    speeds = evaluator.load_model_speeds()
    for policy in (vision_llm.make(), vision_llm_primitives.make()):
        assert evaluator.model_speed(policy, speeds)["max_tokens_per_s"] > 0


def test_zero_mode_records_each_query_latency_and_its_ideal_time(
    monkeypatch, tmp_path, wall_clock, speeds
):
    wall_clock(0.030)
    env, row, manifest = run(
        monkeypatch,
        tmp_path / "run",
        TokenPolicy(tokens=30),
        delay=0.004,
        inference_latency="zero",
        model_speeds=speeds,
        max_time=0.03,
    )
    # Only the delay passes in simulation; the latencies are recorded, not charged.
    assert env.latencies() == [0.004, 0.004]
    assert row["queries"] == 2
    assert row["query_times_s"] == pytest.approx([0.0, 0.024])
    assert row["measured_inference_latencies_s"] == pytest.approx([0.030, 0.030])
    assert row["mean_measured_inference_latency"] == pytest.approx(0.030)
    assert row["query_output_tokens"] == [30.0, 30.0]
    # 0.002 s to the first token + 30 tokens at 1000 tokens/s.
    assert row["ideal_inference_latencies_s"] == pytest.approx([0.032, 0.032])
    assert row["mean_ideal_inference_latency"] == pytest.approx(0.032)
    assert manifest["inference_latency"] == "zero"
    assert manifest["model_speeds"] == {"test/model": {"max_tokens_per_s": 1000.0, "ttft_s": 0.002}}


def test_ideal_mode_also_records_the_measured_latency(monkeypatch, tmp_path, wall_clock, speeds):
    wall_clock(5.0)
    _, row, _ = run(
        monkeypatch,
        tmp_path / "run",
        TokenPolicy(tokens=30),
        inference_latency="ideal",
        model_speeds=speeds,
    )
    assert row["mean_inference_latency"] == pytest.approx(0.032)
    assert row["mean_ideal_inference_latency"] == pytest.approx(0.032)
    assert row["measured_inference_latencies_s"] == pytest.approx([5.0] * row["queries"])


def test_policies_without_a_surveyed_model_record_only_the_measured_latency(
    monkeypatch, tmp_path, wall_clock
):
    wall_clock(0.010)
    _, row, _ = run(
        monkeypatch, tmp_path, CountingPolicy(reported=0.012), inference_latency="zero"
    )
    assert row["measured_inference_latencies_s"] == pytest.approx([0.012] * row["queries"])
    assert len(row["query_times_s"]) == row["queries"]
    assert "ideal_inference_latencies_s" not in row and "query_output_tokens" not in row


def test_streaming_records_each_query_at_its_observation_time(monkeypatch, tmp_path, wall_clock):
    wall_clock(0.030)
    _, row, _ = run(
        monkeypatch,
        tmp_path,
        CountingPolicy(),
        inference_latency="zero",
        time_model="streaming",
        max_time=0.06,
    )
    assert row["query_times_s"] == pytest.approx([0.0, 0.02, 0.04])
    assert row["measured_inference_latencies_s"] == pytest.approx([0.030] * 3)


def test_restarted_attempts_are_kept_apart(monkeypatch, tmp_path):
    """A policy may abandon an attempt (RestartEpisode from act) or, once it is judged, ask for
    it again (episode_finished -> "restart"): the plan entry then runs again from reset, and
    only the attempt that was kept is an episode row."""

    class Operator(CountingPolicy):
        attempts = []

        def reset(self, observation):
            super().reset(observation)
            self.attempts.append(len(self.attempts) + 1)

        def act(self, observation):
            if len(self.attempts) == 1 and self.count == 3:
                raise evaluator.RestartEpisode()
            return super().act(observation)

        def episode_finished(self, row):
            return "restart" if len(self.attempts) == 2 else "continue"

    monkeypatch.setattr(evaluator, "load_policy", lambda _name, _env: Operator())
    plan = [{"task_id": "A1", "difficulty": "easy", "seed": 0, "control_delay": 0.0}]
    evaluator.evaluate(FakeEnv(), plan, "module:factory", tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    redone = [json.loads(line) for line in (tmp_path / "restarted_attempts.jsonl").read_text().splitlines()]
    assert [row["attempt"] for row in rows] == [3]
    assert rows[0]["termination_reason"] == "time_budget"
    assert [(row["attempt"], row["termination_reason"]) for row in redone] == [
        (1, "restarted"),
        (2, "time_budget"),
    ]
