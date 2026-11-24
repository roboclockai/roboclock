"""Dispatch configuration loading to the selected domain's schema."""

from roboclock.envs.registry import get_environment, import_symbol
from roboclock.resources import PACKAGE_ROOT  # noqa: F401 - retained public resource root


def load_config(path=None, *, environment="lab") -> dict:
    spec = get_environment(environment)
    return import_symbol(spec.config_loader)(path or spec.default_config)
