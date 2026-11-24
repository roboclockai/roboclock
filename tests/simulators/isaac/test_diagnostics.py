from types import SimpleNamespace

import pytest

from roboclock.simulators.isaac.diagnostics import graphics_status
from roboclock.simulators.isaac.runtime import IsaacRuntime


def test_graphics_check_reports_missing_library(monkeypatch):
    monkeypatch.setattr("ctypes.util.find_library", lambda name: None if name == "Xt" else name)
    monkeypatch.setattr("ctypes.CDLL", lambda path: object())
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = graphics_status()
    assert not result["libraries"]["Xt"]
    assert result["vulkan_gpu_enumeration"] is None


def test_graphics_check_catches_driver_failure(monkeypatch):
    monkeypatch.setattr("ctypes.util.find_library", lambda name: name)
    monkeypatch.setattr("ctypes.CDLL", lambda path: object())
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/vulkaninfo")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr="ERROR_INCOMPATIBLE_DRIVER", stdout=""
        ),
    )
    result = graphics_status()
    assert result["vulkan_gpu_enumeration"] is False
    assert "ERROR_INCOMPATIBLE_DRIVER" in result["detail"]


def test_known_bad_graphics_fails_before_starting_isaac(monkeypatch):
    monkeypatch.setattr("sys.version_info", (3, 11))
    monkeypatch.setattr("roboclock.simulators.isaac.runtime.installed_version", lambda: "5.1.0.0")
    monkeypatch.setattr(
        "roboclock.simulators.isaac.diagnostics.graphics_status",
        lambda: {"libraries": {"vulkan": True}, "vulkan_gpu_enumeration": False},
    )
    with pytest.raises(RuntimeError, match="Vulkan cannot enumerate"):
        IsaacRuntime()


def test_explicit_vulkan_driver_choice_is_preserved(monkeypatch):
    from roboclock.simulators.isaac.diagnostics import select_headless_driver

    monkeypatch.setenv("VK_DRIVER_FILES", "/user/driver.json")
    monkeypatch.setattr(
        "roboclock.simulators.isaac.diagnostics.graphics_status",
        lambda: pytest.fail("must not probe or replace explicit driver"),
    )
    assert select_headless_driver() is None


def test_runtime_reports_exception_before_kit_shutdown(capsys):
    runtime = IsaacRuntime.__new__(IsaacRuntime)
    runtime.exit_code = 0
    closed = []
    runtime.close = lambda: closed.append(runtime.exit_code)
    error = RuntimeError("lifecycle diagnostic")
    runtime.__exit__(RuntimeError, error, None)
    assert closed == [1]
    assert "lifecycle diagnostic" in capsys.readouterr().err
