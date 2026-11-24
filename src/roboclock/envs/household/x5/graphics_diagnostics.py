"""Read-only graphics diagnostics; do not import Isaac Sim during prerequisite checks."""

import ctypes
import ctypes.util
import json
import os
import shutil
import subprocess
import tempfile


def select_headless_driver():
    """Use NVIDIA EGL only when the default Vulkan driver fails, respecting overrides.

    Returns an owned temporary ICD path; the runtime removes it when it closes.
    No system driver configuration is changed.
    """
    if os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES"):
        return None
    if graphics_status()["vulkan_gpu_enumeration"] is not False:
        return None
    library = ctypes.util.find_library("EGL_nvidia")
    if not library:
        return None
    try:
        driver = ctypes.CDLL(library)
        proc = driver.vk_icdGetInstanceProcAddr
        proc.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        proc.restype = ctypes.c_void_p
        if not proc(None, b"vkCreateInstance"):
            return None
    except (OSError, AttributeError):
        return None
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="roboclock-egl-", delete=False
    ) as stream:
        json.dump(
            {
                "file_format_version": "1.0.0",
                "ICD": {"library_path": library, "api_version": "1.3.0"},
            },
            stream,
        )
        path = stream.name
    os.environ["VK_ICD_FILENAMES"] = path
    if graphics_status()["vulkan_gpu_enumeration"] is not True:
        os.environ.pop("VK_ICD_FILENAMES", None)
        os.unlink(path)
        return None
    return path


def graphics_status():
    libraries = {}
    for name in ("vulkan", "GLU", "Xt", "GLX_nvidia"):
        path = ctypes.util.find_library(name)
        try:
            if path is None:
                raise OSError("library not found")
            ctypes.CDLL(path)
            libraries[name] = True
        except OSError:
            libraries[name] = False
    executable = shutil.which("vulkaninfo")
    vulkan_ok, detail = None, "vulkaninfo is not installed; GPU rendering remains unchecked"
    if executable:
        try:
            result = subprocess.run(
                [executable, "--summary"], capture_output=True, text=True, timeout=15
            )
            vulkan_ok = result.returncode == 0
            detail = (
                "GPU enumeration succeeded"
                if vulkan_ok
                else (result.stderr or result.stdout)[-1500:]
            )
        except subprocess.TimeoutExpired:
            vulkan_ok, detail = False, "Vulkan GPU enumeration timed out"
    return {"libraries": libraries, "vulkan_gpu_enumeration": vulkan_ok, "detail": detail}
