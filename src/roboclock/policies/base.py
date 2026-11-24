"""Shared policy loading; each domain defines its actions and diagnostic oracle."""

import importlib

from roboclock.envs.registry import get_environment, import_symbol


class HoldPolicy:
    def reset(self, observation):
        self.joints = observation["joint_positions"].copy()

    def act(self, observation):
        return self.joints.copy()


def load_policy(name: str, env):
    if name == "hold":
        return HoldPolicy()
    if name == "oracle":
        if not env.oracle:
            raise ValueError("The diagnostic oracle requires --oracle")
        factory = get_environment(env.environment_id).oracle_factory
        if factory is None:
            raise ValueError(f"Environment {env.environment_id!r} has no diagnostic oracle")
        return import_symbol(factory)(env)
    module, separator, factory = name.partition(":")
    if not separator:
        raise ValueError("Policy must be hold, oracle, or a module:factory import path")
    # Custom factories receive no environment, evaluator, object state, or event schedule.
    policy = getattr(importlib.import_module(module), factory)()
    # The action and observation spaces are the robot's interface, not the task: a policy that
    # wants them (how far a finger may be commanded, how wide the jaws open) defines set_spaces.
    set_spaces = getattr(policy, "set_spaces", None)
    if set_spaces is not None:
        set_spaces(env.action_space, env.observation_space)
    return policy
