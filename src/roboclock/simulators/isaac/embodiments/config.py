"""Migration assignments. Existing mobile tasks remain outside this change."""

MOBILE_TASKS = ()
ASSIGNMENTS = {task: "dual_arx_x5" for task in ("A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1", "C2", "C3")}


def configure(config, task, model=None):
    if task in MOBILE_TASKS:
        raise ValueError(f"{task} retains its existing mobile robot")
    if task == "B4" and not config.get("workflow", {}).get("fixed_freeze"):
        raise ValueError("B4 requires the fixed-freeze configuration; legacy mobile B4 is unchanged")
    if task == "C3" and not config.get("workflow", {}).get("fixed_service"):
        raise ValueError("C3 requires the fixed-service configuration; legacy mobile C3 is unchanged")
    model = model or ASSIGNMENTS[task]
    if model != "dual_arx_x5":
        raise ValueError(f"Unsupported robot {model!r}: roboclock uses only dual_arx_x5")
    config["visuals"]["anti_aliasing"] = 4
    if task in ("B3", "C1"):
        config["visuals"]["focal_length"] = 0.032
    config["embodiment"] = dict(
        model=model,
        fixed_base=True,
        active_side="right",
        position=[0.15, 0, 1.0],
        action_interface="native articulation joint positions",
        asset_cache="manufacturer download; see embodiments/assets.json",
    )
    config["embodiment"]["active_arms"] = (
        ["right"] if task in ("A1", "A3", "B1", "C1") else ["left", "right"]
    )
    if task == "B1":
        config["embodiment"]["active_arms"] = ["left", "right"]
        config["v2"].update(dual_mallets=True, mole_cartesian_speed=0.30,
                            mole_joint_speed=2.5, mole_aim_timeout=1.6,
                            mole_ready_angle=-25.0, mole_hit_angle=0.0, mole_contact_depth=0.008)
    if task == "A2":
        config["v2"].update(
            dual_sticks=True,
            tool_offset=0.225,
            grip_sleeve_size=[0.032, 0.036, 0.080],
            native_gripper_effort=3.5,
            tool_mass=0.075,
            screen_offset=[0.20, 0.16, 0.20],
        )
    config["embodiment"]["asset_cache"] = (
        "Manufacturer X5A CAD; src/roboclock/envs/sports/common/assets/dual_arx_x5/manifest.json"
    )
    if task == "B2":
        config["embodiment"]["position"] = [0.05, 0, 0.92]
    config["embodiment"]["auxiliary_arm_role"] = (
        "fixed observation pose with wrist camera" if task in ("A1", "A3", "C1")
        else "bimanual manipulation"
    )
    if task == "A1":
        config["embodiment"]["finger_force_n"] = 4.0
        config["control"].update(grasp_height_offset=0.022, grasp_open_half_width=0.025)
    if task == "C2":
        config["embodiment"]["position"] = [-0.02, 0, 1.0]
    if task == "B3":
        config["embodiment"]["position"] = [-0.02, 0, 1.08]
        config["workflow"]["standby_position"] = [0.32, 0.0, 1.16]
        config["workflow"]["bench_joint_speed"] = 5.2
        config["workflow"]["capture_prediction"] = 0.18
        config["workflow"]["capture_prediction_limit"] = 0.04
    if task == "A3":
        config["embodiment"]["position"] = [0.22, 0.32, 1.0]
        config.setdefault("basketball", {}).update(
            joint_speed=5.0,
            native_shot=dict(aim_position=[0.59, 0.06, 1.24],
                             follow_through=[0.71, 0.06, 1.34], open_after=0.12),
        )
        config["visuals"]["focal_length"] = 0.030
    if task == "C1":
        config["embodiment"].update(
            position=[0.58, -.42, 1.0],
            yaw=0.0,
        )
        config["robot_mount"] = config["embodiment"]["position"].copy()
        config["embodiment"]["active_arms"] = ["right", "left"]
        config["embodiment"]["secondary_arm_role"] = "Reach-aware shared-cue exchange through a supported rest"
        config["embodiment"]["yaw"] = 1.5707963267948966
        config["pool"] = dict(grasp_height_offset=.008, grasp_yaw_by_side={"right": 180., "left": 0.}, dual_cue_exchange=True, table_scale=.60, cue_grip_x=.09, backswing=.040, hit_height_offset=0., cue_pitch_candidates=[10., 18., 26., 34., 45., 55.], cue_roll_candidates=[0., -20., 20., -40., 40., -70., 70.], table_center=[.58, -.04], joint_speed=2.5, stroke_joint_speed=4.5, stroke_lookahead=.030, finger_force_n=14.0, heading_trim_degrees=0.0, cue_pitch_degrees=10.0)
    from .presentation import configure_presentation

    configure_presentation(config, task)
    if task == "C1":
        config["visuals"]["presentation_camera"].update(eye=[2.10, -.70, 2.70], target=[.47, -.03, .88])
    return config
