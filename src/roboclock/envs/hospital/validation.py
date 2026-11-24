"""Hospital rendering export. Physics probes are not implemented for this domain yet."""

from roboclock.envs.lab.validation import export_render  # noqa: F401


def validate_physics(env, plan, output, **kwargs):
    raise ValueError(
        "The hospital environment has no physics probes yet; use `roboclock render --env hospital` "
        "and `roboclock smoke --env hospital` to inspect the scene."
    )
