"""Explicit RTX quality settings shared by scene setup and metadata export."""


def configure_renderer(config):
    import carb

    visual = config["visuals"]
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", visual.get("renderer", "RaytracedLighting"))
    # Hollow laboratory glass crosses multiple interfaces before reaching liquid.
    settings.set("/rtx/translucency/enabled", True)
    settings.set("/rtx/translucency/maxRefractionBounces", visual.get("refraction_bounces", 16))
    settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)
    settings.set("/rtx/pathtracing/spp", visual.get("path_tracing_spp", 8))
    settings.set("/rtx/pathtracing/totalSpp", visual.get("path_tracing_samples", 512))
    settings.set("/rtx/pathtracing/clampSpp", visual.get("path_tracing_samples", 512))
    settings.set("/rtx/pathtracing/maxBounces", visual.get("path_tracing_bounces", 8))
    settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    settings.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0)
    settings.set("/rtx/resetPtAccumOnAnimTimeChange", False)
    # Frozen-time multi-subframe capture is driven explicitly at observation/export.
    settings.set("/omni/replicator/RTSubframes", 1)
    if "anti_aliasing" not in visual:
        # Lab, household and hospital renders: DLSS, unchanged.
        settings.set("/rtx/post/aa/op", 3)
    else:
        # Native-resolution DLAA preserves fine text; frozen-time subframes settle changes.
        # Replicator enum: None=0, TAA=1, FXAA=2, DLSS=3, DLAA=4.
        aa = visual["anti_aliasing"]
        from roboclock.simulators.isaac.version import installed_version

        # RTX Real-Time 2.0 needs neural reconstruction independently of the old
        # FXAA workaround. Isaac 6 otherwise exposes raw path-sampling noise.
        if (installed_version() or "").startswith("6.") and aa not in (3, 4):
            aa = 4
        # Isaac Sim 5.1 otherwise restricts AA to DLSS/DLAA and silently resets FXAA.
        # This is the same gate used by Replicator's set_render_rtx_realtime().
        if aa not in (3, 4):
            settings.set("/rtx-transient/post/aa/limitedOps", False)
            settings.set("/rtx-transient/dldenoiser/enabled", False)
        else:
            settings.set("/rtx-transient/dldenoiser/enabled", True)
        settings.set("/rtx/post/aa/op", aa)
        settings.set("/rtx-defaults/post/aa/op", aa)
    # Optional exposure for bright interiors: the RTX tonemapper's film ISO (renderer default 100).
    if "film_iso" in visual:
        settings.set("/rtx/post/tonemap/filmIso", float(visual["film_iso"]))
    return settings.get("/rtx/rendermode")


def capture_camera(subframes=8):
    import omni.replicator.core as rep

    rep.orchestrator.step(delta_time=0.0, rt_subframes=subframes, pause_timeline=False)


def renderer_state():
    """Read effective values after reset/render, rather than report requested values."""
    import carb

    settings = carb.settings.get_settings()
    return {
        key: settings.get(key)
        for key in (
            "/rtx/rendermode",
            "/rtx/post/aa/op",
            "/rtx-transient/post/aa/limitedOps",
            "/rtx-transient/dldenoiser/enabled",
            "/omni/replicator/RTSubframes",
        )
    }
