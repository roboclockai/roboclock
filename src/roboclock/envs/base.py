"""Shared benchmark contract; domains own their robot, spaces and task semantics."""

from abc import ABC, abstractmethod

import gymnasium as gym


class BenchmarkEnv(gym.Env, ABC):
    """Gym API plus latency and artifact hooks consumed by the shared evaluator.

    The evaluator's time models (see roboclock.evaluation.evaluator) need two hooks beyond the
    Gym API: advance_latency() advances physics under the targets in effect (block-and-replay
    uses it to replay inference time), and set_action() changes the targets without advancing
    physics (streaming uses both to switch actions inside a control period).

    Implementations expose config, oracle, done and task. The task supplies time,
    terminated, spec.max_time, metrics(), event_log and gen_instruction(). Metrics
    follow the temporal benchmark schema described in docs/architecture.md.
    """

    environment_id: str

    @abstractmethod
    def advance_latency(self, seconds: float):
        """Advance physics by `seconds` under the joint targets already in effect.

        Rounds up to a whole physics tick and stops at the end of the episode budget.
        """

    def set_action(self, action):
        """Make `action` the joint targets in effect from the current physics tick on,
        without advancing physics. `step(a)` is `set_action(a)` plus one control period.
        The streaming evaluator switches targets with it inside a control period."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming evaluation")

    @abstractmethod
    def benchmark_metadata(self) -> dict:
        """Serializable domain, simulator and asset metadata for the run manifest."""

    @abstractmethod
    def episode_metadata(self) -> dict:
        """Serializable scene/robot/initial-state metadata for an evaluated episode."""
