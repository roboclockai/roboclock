"""Environment registration with lazy imports: listing domains never starts Isaac Sim."""

import importlib
import re
from importlib.metadata import entry_points

from roboclock.envs.spec import EnvironmentSpec


def import_symbol(reference: str):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"Expected module:symbol, got {reference!r}")
    return getattr(importlib.import_module(module), name)


_ENVIRONMENTS: dict[str, EnvironmentSpec] = {}
_PLUGINS_LOADED = False


def _load_plugins():
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    for entry in entry_points(group="roboclock.environments"):
        spec = entry.load()
        if not isinstance(spec, EnvironmentSpec) or spec.name != entry.name:
            raise ValueError(f"Environment entry point {entry.name!r} must export a matching spec")
        register_environment(spec)
    _PLUGINS_LOADED = True


def register_environment(spec: EnvironmentSpec):
    """Register before invoking CLI/create_env; reject accidental name replacement."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name):
        raise ValueError("Environment names must use lowercase letters, digits and underscores")
    if not spec.difficulties:
        raise ValueError("An environment must declare at least one difficulty")
    if spec.name in _ENVIRONMENTS:
        raise ValueError(f"Environment already registered: {spec.name}")
    _ENVIRONMENTS[spec.name] = spec


def environment_names() -> tuple[str, ...]:
    _load_plugins()
    return tuple(_ENVIRONMENTS)


def get_environment(name: str = "lab") -> EnvironmentSpec:
    _load_plugins()
    try:
        return _ENVIRONMENTS[name]
    except KeyError:
        raise ValueError(
            f"Unknown environment {name!r}; choose one of {', '.join(environment_names())}"
        ) from None


def create_env(environment="lab", task_id=None, difficulty=None, **kwargs):
    spec = get_environment(environment)
    return import_symbol(spec.factory)(
        task_id or spec.default_task, difficulty or spec.difficulties[0], **kwargs
    )


# Built-ins declare metadata in their own domain; adding one only adds a spec here.
_BUILTIN_SPECS = (
    "roboclock.envs.lab.spec:SPEC",
    "roboclock.envs.household.spec:SPEC",
    "roboclock.envs.hospital.spec:SPEC",
    "roboclock.envs.sports.spec:SPEC",
    "roboclock.envs.industry.spec:SPEC",
)
for _reference in _BUILTIN_SPECS:
    register_environment(import_symbol(_reference))
