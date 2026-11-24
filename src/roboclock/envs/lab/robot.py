"""Dual ARX X5 embodiment shared by every laboratory task.

Each arm takes six arm radians and two finger meters. The env interface (LabEnv.step /
get_obs, inherited by hospital) lists the rig's LEFT arm (+y in the rig frame) first, then
the rig's right arm, as every RoboDojo domain does. Internally (ROOTS, the backend,
`backend.apply_action` / `backend.joint_positions`, the oracles' planners) side 0 is
/World/ARXLeft, which stands on the rig's RIGHT (-y) and does the single-arm work; the prim
names predate the rig. `arms_to_external` / `arms_to_internal` convert between the two.
The exported X5A geometry supplies both fingers; no Panda geometry is reused.
"""

from pathlib import Path

import numpy as np

from roboclock.envs import robodojo_rig

ASSET_DIR = Path(__file__).parent / "assets/arx_x5"
ROBOT_ID = "dual_arx_x5"
DOF = 8
ARM_DOF = 6
JOINT_LOW = np.array([-3.14, -0.05, -0.1, -1.6, -1.57, -2, 0, 0], dtype=np.float32)
JOINT_HIGH = np.array([2.618, 3.50, 3.20, 1.55, 1.57, 2, 0.044, 0.044], dtype=np.float32)
EFFORTS = np.array([30, 40, 30, 15, 10, 10, 10, 10], dtype=float)
VELOCITIES = np.array([5, 5, 5.5, 5.5, 5, 5, 0.1, 0.1], dtype=float)
HOME = np.array([0, 1.6, 1.3, 0.3 - np.pi / 2, 0, 0, 0.04, 0.04], dtype=float)
ROOTS = ("/World/ARXLeft", "/World/ARXRight")
LINKS = tuple(f"link{i}" for i in range(1, 7))
DEFAULT_MOUNTS = [[0.10, -0.10, 1.05], [0.0, 0.38, 1.05]]
HAND_LINKS = ("link7", "link8", "link6")


def link_paths(side=0, hand=False):
    return [f"{ROOTS[side]}/{name}" for name in (HAND_LINKS if hand else LINKS)]


# Common dual-X5 rig (`robot.rig`), RoboDojo geometry from roboclock.envs.robodojo_rig:
# both bases side by side at one height, both facing the rig's forward direction,
# ARM_SEPARATION_M apart center to center; the head camera is rigid at HEAD_IN_RIG_M
# (forward, left, up from the base midpoint). Only the rig origin/yaw/height vary by task.
# These names are kept for importers (hospital); they are not local copies.
RIG_SEPARATION = robodojo_rig.ARM_SEPARATION_M
HEAD_CAMERA_OFFSET = tuple(robodojo_rig.HEAD_IN_RIG_M)
# ROOTS index of the arm on the rig's left (+y in the rig frame). Side 0 (ARXLeft, the
# single-arm worker) is on the rig's right, see rig_mounts; RoboDojo's left wrist camera
# (`left_wrist_rgb`) therefore rides side 1.
RIG_LEFT_ARM, RIG_RIGHT_ARM = 1, 0
# Internal side (ROOTS index) of each arm of the env interface, in interface order: the rig's
# left arm first. INTERNAL_ARMS is the identity order (legacy envs that keep ROOTS order).
EXTERNAL_ARMS = (RIG_LEFT_ARM, RIG_RIGHT_ARM)
INTERNAL_ARMS = (0, 1)
# Interface labels of the 16 joints, in action / joint_positions order.
EXTERNAL_JOINT_ORDER = tuple(
    f"{side}/joint{i}" for side in ("left", "right") for i in range(1, DOF + 1)
)
# Observation entries made of one equal block per arm (8 joints, 7-value pose, 1 width).
PER_ARM_OBSERVATIONS = ("joint_positions", "ee_pose", "gripper_width")


def _reorder_arms(values, order):
    values = np.asarray(values)
    if tuple(order) == INTERNAL_ARMS:
        return values.copy()
    size = values.shape[-1] // len(order)
    if size * len(order) != values.shape[-1]:
        raise ValueError(f"Expected one equal block per arm, got {values.shape[-1]} values")
    return np.concatenate([values[..., size * a : size * (a + 1)] for a in order], axis=-1)


def arms_to_external(values, order=EXTERNAL_ARMS):
    """Per-arm blocks (last axis) from internal side order to the env interface order."""
    return _reorder_arms(values, order)


def arms_to_internal(values, order=EXTERNAL_ARMS):
    """Per-arm blocks (last axis) from the env interface order to internal side order."""
    return _reorder_arms(values, tuple(int(i) for i in np.argsort(order)))


def observation_to_external(observation, order=EXTERNAL_ARMS):
    """Copy of an internal observation dict with its per-arm vectors in interface order."""
    result = dict(observation)
    for key in PER_ARM_OBSERVATIONS:
        if key in result and np.size(result[key]) % len(order) == 0:
            result[key] = arms_to_external(result[key], order)
    return result


def observation_to_internal(observation, order=EXTERNAL_ARMS):
    """Copy of an interface observation dict with its per-arm vectors in internal side order."""
    result = dict(observation)
    for key in PER_ARM_OBSERVATIONS:
        if key in result and np.size(result[key]) % len(order) == 0:
            result[key] = arms_to_internal(result[key], order)
    return result


def rig_pose(config, task_id):
    """(origin, yaw) of the dual-arm rig, or None for configs without `robot.rig`.

    The origin is the midpoint of the two base mounts on the mount plane (the
    X5 base flange height); yaw is the shared forward direction of both arms.
    """
    rig = config.get("robot", {}).get("rig")
    if rig is None:
        return None
    pose = {**rig.get("default", {}), **rig.get("tasks", {}).get(task_id, {})}
    return np.asarray(pose["origin"], dtype=float), float(pose.get("yaw", 0.0))


def rig_axes(yaw):
    """Unit forward and left vectors of a rig with the given yaw (world xy-plane)."""
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return forward, left


def rig_mounts(origin, yaw, separation=RIG_SEPARATION):
    """Base mounts of internal side 0 (rig RIGHT, -y) and side 1 (rig LEFT, +y).

    Side 0 (/World/ARXLeft despite its name) keeps the original lab placement: it is
    the arm on the rig's right (world -y for yaw 0) and does the single-arm work in
    most tasks. The env interface lists side 1 first (EXTERNAL_ARMS).
    """
    _, left = rig_axes(yaw)
    origin = np.asarray(origin, dtype=float)
    return [origin - left * separation / 2, origin + left * separation / 2]


def head_camera_position(config, task_id):
    """Rigid head camera position for a rig config (None without `robot.rig`)."""
    pose = rig_pose(config, task_id)
    if pose is None:
        return None
    origin, yaw = pose
    rig = config["robot"]["rig"]
    # Configs without the key (lab) use the shared RoboDojo offset.
    ahead, lateral, up = rig.get("head_camera_offset", HEAD_CAMERA_OFFSET)
    forward, left = rig_axes(yaw)
    return origin + forward * ahead + left * lateral + np.array([0.0, 0.0, up])


def mount_positions(config, task_id):
    robot = config.get("robot", {})
    pose = rig_pose(config, task_id)
    if pose is not None:
        separation = robot["rig"].get("separation", RIG_SEPARATION)
        return [m.tolist() for m in rig_mounts(*pose, separation)]
    return robot.get("task_mount_positions", {}).get(
        task_id, robot.get("mount_positions", DEFAULT_MOUNTS)
    )


def mount_yaws(config, task_id):
    """Base yaw of each arm: the shared rig yaw, or legacy `task_mount_yaws`."""
    pose = rig_pose(config, task_id)
    if pose is not None:
        return [pose[1], pose[1]]
    return config.get("robot", {}).get("task_mount_yaws", {}).get(task_id, [0.0, 0.0])


def home_joints(config, task_id):
    """Initial 2x8 joint targets in internal side order; `robot.task_home_joints` overrides
    HOME per task (a 2x6 override is in internal side order too).

    Opposed C1/C3 mounts start with the tools retracted and pointing down: with the
    default HOME the two grippers touch at reset and lock both base joints.

    `robot.start_pose: zero` (roboclock.envs.start_pose) puts joints 1-6 of both arms at 0
    for every task, ignoring the overrides; the fingers stay at HOME (open).
    """
    from roboclock.envs.start_pose import zero_start

    joints = np.tile(HOME, (2, 1))
    if zero_start(config):
        joints[:, :ARM_DOF] = 0.0
        return joints
    custom = config.get("robot", {}).get("task_home_joints", {}).get(task_id)
    if custom is not None:
        arm = np.asarray(custom, dtype=float)
        joints[:, :ARM_DOF] = arm if arm.ndim == 2 else np.tile(arm, (2, 1))
    return joints


def filter_self_collisions(backend, root):
    """Drop configured non-adjacent link pairs from the arm's self-collision checks.

    The X5 collision shapes are convex hulls. When joint1 points the arm to its
    side, the link2 hull touches the base_link hull at joint2 ~2.45 rad and
    stops the shoulder well inside its 200-degree range, so low goals far to the
    side (C1/C3 far rows) become unreachable although IK solves them.
    """
    from pxr import UsdPhysics

    robot = backend.config.get("robot", {})
    if backend.task.task_id not in robot.get("self_collision_filter_tasks", []):
        return
    for first, second in robot.get("self_collision_filters", []):
        prim = backend.stage.GetPrimAtPath(f"{root}/{first}")
        other = backend.stage.GetPrimAtPath(f"{root}/{second}")
        if not prim.IsValid() or not other.IsValid():
            raise RuntimeError(f"Missing X5 links for collision filter: {first}, {second}")
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        api.CreateFilteredPairsRel().AddTarget(other.GetPath())


def create_robots(backend):
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot.manipulators import SingleManipulator
    from isaacsim.robot.manipulators.grippers import ParallelGripper

    asset = ASSET_DIR / "arx_x5.usd"
    if not asset.exists():
        raise FileNotFoundError("Build the X5 asset with scripts/lab/prepare_arx_x5.py first")
    mounts = mount_positions(backend.config, backend.task.task_id)
    yaws = mount_yaws(backend.config, backend.task.task_id)
    homes = home_joints(backend.config, backend.task.task_id)
    robots = []
    for side, (root, position) in enumerate(zip(ROOTS, mounts)):
        # A raised mounting column gives the six-axis X5 downward access to
        # the laboratory fixtures without changing its kinematics or joint limits.
        base = backend.layout.table_height
        if rig_pose(backend.config, backend.task.task_id) is not None:
            # A rig column whose footprint leaves the worktop stands on the floor.
            scene = backend.config["scene"]
            low = np.asarray(scene["table_center"][:2]) - np.asarray(scene["table_size"][:2]) / 2
            high = np.asarray(scene["table_center"][:2]) + np.asarray(scene["table_size"][:2]) / 2
            if np.any(np.asarray(position[:2]) - 0.06 < low) or np.any(
                np.asarray(position[:2]) + 0.06 > high
            ):
                base = 0.0
        height = position[2] - base
        if height > 0:
            backend.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/ARXMount{side}",
                    name=f"arx_mount_{side}",
                    position=np.array([position[0], position[1], base + height / 2]),
                    scale=np.array([0.12, 0.12, height]),
                    color=np.array([0.20, 0.22, 0.24]),
                )
            )
        add_reference_to_stage(str(asset), root)
        filter_self_collisions(backend, root)
        gripper = ParallelGripper(
            end_effector_prim_path=root + "/link6",
            joint_prim_names=["joint7", "joint8"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.zeros(2),
            action_deltas=np.array([0.01, 0.01]),
        )
        robot = backend.world.scene.add(
            SingleManipulator(
                prim_path=root,
                # Scene-registry names follow the prim names, not the rig side.
                name=("arx_left", "arx_right")[side],
                position=np.array(position, dtype=float),
                orientation=np.array([np.cos(yaws[side] / 2), 0, 0, np.sin(yaws[side] / 2)]),
                end_effector_prim_path=root + "/link6",
                gripper=gripper,
            )
        )
        robot.set_joints_default_state(positions=homes[side].copy())
        robots.append(robot)
    backend.robot, backend.secondary_robot = robots


def initialize_robots(backend):
    homes = home_joints(backend.config, backend.task.task_id)
    for side, robot in enumerate((backend.robot, backend.secondary_robot)):
        expected = [f"joint{i}" for i in range(1, 9)]
        if robot.dof_names != expected:
            raise RuntimeError(f"Unexpected X5 joint ordering: {robot.dof_names}")
        robot.set_joint_positions(homes[side].copy())
        robot.set_joint_velocities(np.zeros(DOF))
        controller = robot.get_articulation_controller()
        controller.set_gains(
            kps=np.array([1500.0] * ARM_DOF + [1000.0] * 2),
            kds=np.array([60.0] * ARM_DOF + [40.0] * 2),
        )
        controller.set_max_efforts(EFFORTS)
    backend.initial_hold_joints = homes.reshape(-1).copy()
    backend.apply_action(backend.initial_hold_joints)


def tool_pose(robot):
    from roboclock.physics.geometry import rotation_matrix

    position, quat = robot.end_effector.get_world_pose()
    # Native link6 +X points toward the fingertips; task frames use +Z.
    w, x, y, z = quat
    tool_quat = np.array([w - y, x - z, w + y, x + z]) / np.sqrt(2.0)
    return position + rotation_matrix(quat) @ np.array([0.145, 0, 0]), tool_quat


def make_solver(robot):
    from isaacsim.robot_motion.motion_generation import (
        ArticulationKinematicsSolver,
        LulaKinematicsSolver,
    )

    solver = LulaKinematicsSolver(
        robot_description_path=str(ASSET_DIR / "robot_description.yaml"),
        urdf_path=str(ASSET_DIR / "arx_x5.urdf"),
    )
    solver.set_robot_base_pose(*robot.get_world_pose())
    return ArticulationKinematicsSolver(robot, solver, "grasp_center")
