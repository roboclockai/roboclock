"""Prepared bedside fixtures: passive mechanics and measured device state."""

from importlib import import_module


def module_for(task_or_layout):
    name = getattr(task_or_layout, "bedside_module", None)
    if name is None:
        name = getattr(task_or_layout, "info", {}).get("bedside_task")
    return import_module(f"{__name__}.{name}") if name else None
