"""Task-focused presentation cameras; independent of robot sensor cameras/physics."""

import copy

# Keep task displays, tools and complete interaction envelopes in view. These
# lenses are rendered at native output resolution, never cropped from an MP4.
PROFILES = {
    "B4": dict(eye=[1.9, -2.25, 2.15], target=[0.38, 0.02, 1.02], focal_length=0.045),
    "C3": dict(eye=[1.70, -2.50, 2.00], target=[0.40, 0.0, 1.02], focal_length=0.044),
    "A1": dict(eye=[0.48, -2.20, 2.10], target=[0.48, 0.02, 1.08], focal_length=0.043),
    "A2": dict(eye=[1.65, -1.70, 1.75], target=[0.40, 0.08, 1.12], focal_length=0.039),
    "A3": dict(eye=[0.06, -2.55, 1.88], target=[0.42, 0.0, 1.12], focal_length=0.043),
    "B1": dict(eye=[1.65, -1.70, 1.75], target=[0.43, 0.04, 1.02], focal_length=0.043),
    "B2": dict(eye=[1.85, -2.65, 2.02], target=[0.32, 0.04, 1.08], focal_length=0.048),
    "B3": dict(eye=[2.20, -2.80, 1.90], target=[0.52, 0.0, 1.00], focal_length=0.043),
    "C1": dict(eye=[2.40, -0.50, 2.70], target=[0.68, 0.06, 0.87], focal_length=0.0455),
    "C2": dict(eye=[1.40, -1.70, 1.80], target=[0.38, 0.01, 1.03], focal_length=0.039),
}


def configure_presentation(config, task):
    config["visuals"]["presentation_camera"] = {
        "profile": "x5-task-closeup-v2",
        # Lift the dark arcade cabinets slightly more than the brighter rooms;
        # preserve the authored light balance, materials and camera exposure.
        "light_intensity_multiplier": 1.80 if task in {"A2", "A3", "B1", "C2"} else 1.60,
        **copy.deepcopy(PROFILES[task]),
        "focal_length": PROFILES[task]["focal_length"] * 1.05,
    }


def apply_presentation_camera(backend):
    """Change only camera extrinsics/intrinsics, without advancing simulation."""
    profile = backend.config.get("visuals", {}).get("presentation_camera")
    if not profile:
        return False
    from roboclock.simulators.isaac.embodiments.views import look_pose

    backend.camera.set_world_pose(
        *look_pose(profile["eye"], profile["target"], [0, 0, 1]), camera_axes="usd"
    )
    backend.camera.set_focal_length(profile["focal_length"])
    backend.camera.set_horizontal_aperture(0.036)
    apply_presentation_lighting(backend, profile.get("light_intensity_multiplier", 1.0))
    return True


def apply_presentation_lighting(backend, multiplier):
    """Scale authored light intensities once, preserving material contrast."""
    if (
        getattr(backend, "_presentation_light_multiplier", None) == multiplier
        and getattr(backend, "_presentation_light_stage", None) is backend.stage
    ):
        return True
    backend._presentation_light_multiplier = multiplier
    backend._presentation_light_stage = backend.stage
    # Original intensities make repeated camera updates idempotent.
    for prim in backend.stage.Traverse():
        intensity = prim.GetAttribute("inputs:intensity")
        if intensity and intensity.Get() is not None:
            original = prim.GetCustomDataByKey("benchmark:originalLightIntensity")
            if original is None:
                original = float(intensity.Get())
                prim.SetCustomDataByKey("benchmark:originalLightIntensity", original)
            intensity.Set(original * multiplier)
    return True
