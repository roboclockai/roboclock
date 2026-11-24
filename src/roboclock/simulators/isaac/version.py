"""Inspect installed runtime metadata without importing Kit or starting a GPU app."""

from importlib.metadata import PackageNotFoundError, version


def installed_version():
    try:
        return version("isaacsim")
    except PackageNotFoundError:
        return None


def required_python(sim_version):
    if sim_version and sim_version.startswith("5.1."):
        return (3, 11)
    if sim_version and sim_version.startswith("6."):
        return (3, 12)
    return None


def simulator_label():
    return f"Isaac Sim {installed_version() or 'unavailable'} / PhysX"
