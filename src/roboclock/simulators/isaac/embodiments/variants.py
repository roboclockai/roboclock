"""Dual ARX X5 asset configuration: physics variant, surfaces, finger collisions."""


def configure_variant(stage, root, model):
    if model == "dual_arx_x5":
        root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        from .arx_appearance import apply_surfaces
        from .arx_collision import configure_finger_collisions

        apply_surfaces(stage, root)
        configure_finger_collisions(stage, root)
        return ("left", "right")
    raise ValueError(f"Unsupported robot {model!r}: roboclock uses only dual_arx_x5")
