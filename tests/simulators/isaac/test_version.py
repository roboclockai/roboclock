"""Select the Python ABI from runtime metadata, without initializing Kit."""

import pytest

from roboclock.simulators.isaac import runtime, version


@pytest.mark.parametrize(
    "installed,expected",
    [
        ("5.1.0.0", (3, 11)),
        ("6.2.0.0", (3, 12)),
        (None, None),
        ("4.5.0", None),
    ],
)
def test_required_python(installed, expected):
    assert version.required_python(installed) == expected


def test_wrong_python_fails_before_app_import(monkeypatch):
    monkeypatch.setattr(runtime, "installed_version", lambda: "6.2.0.0")
    monkeypatch.setattr("sys.version_info", (3, 11))
    with pytest.raises(RuntimeError, match="requires Python 3.12"):
        runtime.IsaacRuntime()


def test_provenance_uses_actual_installed_version(monkeypatch):
    monkeypatch.setattr(version, "installed_version", lambda: "6.2.0.0")
    assert version.simulator_label() == "Isaac Sim 6.2.0.0 / PhysX"
