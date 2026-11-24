"""Dispatch domain-specific rendering and physics probes through the registry."""

import importlib

from roboclock.envs.registry import get_environment


def _validator(env):
    module = get_environment(env.environment_id).validation_module
    if module is None:
        raise ValueError(f"Environment {env.environment_id!r} has no validation implementation")
    return importlib.import_module(module)


def export_render(env, output, *, usd=False):
    return _validator(env).export_render(env, output, usd=usd)


def validate_physics(env, plan, output, **kwargs):
    if any(item.get("environment", env.environment_id) != env.environment_id for item in plan):
        raise ValueError("Validation plan contains episodes for a different environment")
    return _validator(env).validate_physics(env, plan, output, **kwargs)
