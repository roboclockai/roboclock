"""Shared dual-X5 collision monitoring."""


def build_collision_monitor(backend, fixtures=()):
    from isaacsim.core.prims import RigidPrim

    from roboclock.envs.lab.robot import HAND_LINKS, LINKS, ROOTS

    links = list(LINKS)
    all_links = list(dict.fromkeys((*LINKS, *HAND_LINKS)))
    filters = ["/World/Table"] if backend.stage.GetPrimAtPath("/World/Table").IsValid() else []
    scene = getattr(backend, "reactive_scene", None)
    filters += list(getattr(scene, "surfaces", {}).values())
    filters += [body.prim_path for body in getattr(backend, "clutter", {}).values()]
    filters += list(fixtures)
    filters += [
        p
        for p in ("/World/ARXMount0", "/World/ARXMount1")
        if backend.stage.GetPrimAtPath(p).IsValid()
    ]
    filters = list(dict.fromkeys(filters))
    backend.secondary_clutter_filter_indices = [
        filters.index(body.prim_path) for body in getattr(backend, "clutter", {}).values()
    ]
    # Finger contacts with fixtures are part of insertion, just as for the
    # primary arm. Count inter-arm contacts separately, including both grippers.
    backend.inter_arm_contacts = backend.world.scene.add(
        RigidPrim(
            prim_paths_expr=[ROOTS[1] + "/" + name for name in all_links],
            name="inter_arm_collision_monitor",
            track_contact_forces=True,
            reset_xform_properties=False,
            contact_filter_prim_paths_expr=[
                [ROOTS[0] + "/" + name for name in all_links] for _ in all_links
            ],
            max_contact_count=4096,
        )
    )
    return backend.world.scene.add(
        RigidPrim(
            prim_paths_expr=["/World/ARXRight/" + name for name in links],
            name="secondary_collision_monitor",
            track_contact_forces=True,
            reset_xform_properties=False,
            contact_filter_prim_paths_expr=[filters for _ in links],
            max_contact_count=4096,
        )
    )
