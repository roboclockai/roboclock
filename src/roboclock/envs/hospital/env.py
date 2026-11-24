"""Hospital environment: the dual ARX X5 joint-position interface of `lab` with hospital scenes.

The robot, action space, observation contract and simulation-time latency injection are shared
with the laboratory environment; only the scene backend, task registry and metadata differ.
Actions and per-arm observations list the rig's left arm (+y, `left_wrist_rgb`) first: LabEnv
converts to and from the backend's internal side order (arm 0 = /World/ARXLeft = rig right).
"""

from copy import deepcopy

import gymnasium as gym
import numpy as np

from roboclock.envs import robodojo_rig
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.lab.env import LabEnv
from roboclock.envs.lab.robot import ROBOT_ID


def _template(camera):
    return robodojo_rig.HEAD_CAMERA if camera == "cam_head" else robodojo_rig.WRIST_CAMERA


class HospitalEnv(LabEnv):
    environment_id = "hospital"

    def __init__(self, *args, depth=False, **kwargs):
        super().__init__(*args, depth=depth, **kwargs)
        # RoboDojo policy cameras (robodojo_rig.OBSERVATION_KEYS / DEPTH_KEYS), 640x480 each.
        self.backend.want_depth = depth
        spaces = self.observation_space.spaces
        for camera, key in robodojo_rig.OBSERVATION_KEYS.items():
            width, height = _template(camera)["resolution"]
            spaces[key] = gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
            if depth:
                spaces[robodojo_rig.DEPTH_KEYS[camera]] = gym.spaces.Box(
                    0, np.inf, (height, width), dtype=np.float32
                )

    def _create_backend(self, asset_catalog):
        from roboclock.envs.hospital.backend import HospitalBackend

        return HospitalBackend(self.runtime, self.config, asset_catalog)

    def _make_task(self, task_id, difficulty, seed):
        return make_task(task_id, difficulty, seed, self.config)

    def _ticks(self, count, *, render=True):
        # Physics ticks never render: the policy cameras render in get_obs (25 Hz).
        render = False
        super()._ticks(count, render=False)
        if self.done and self.task.layout.info.get("bedside_task"):
            from roboclock.envs.hospital.mdp.events import GREEN, RED, SceneCommand

            # Evaluation may terminate on this tick, before the next process() call.
            # Refresh the visible outcome without advancing either simulation clock.
            color = GREEN if self.task.joint_success else RED
            # Countdown dials keep their timed ring color when only the hub changes.
            self.backend.command(
                SceneCommand(
                    "timer", "indicator", (1.0, "green" if self.task.joint_success else "red")
                )
            )
            self.backend.command(SceneCommand("indicator", "indicator", color))
            if render:
                self.backend.world.render()

    def episode_metadata(self):
        layout = self.task.layout
        return {
            "scene_id": "hospital_scene02_left_ward_v2"
            if layout.info.get("bedside_task")
            else "hospital_tabletop_v1",
            "bedside_task": layout.info.get("bedside_task"),
            "bedside_state": self._bedside_state(),
            "ward_asset": self.config["scene"].get("bedside_ward_usd")
            if layout.info.get("bedside_task")
            else None,
            "robot": ROBOT_ID,
            "joint_order": self.joint_order(),
            "control_mode": "joint_position",
            "robodojo_rig": robodojo_rig.describe(),
            "rig_check": getattr(self.backend, "rig_check", None),
            "renderer_readback": getattr(self.backend, "render_state", None),
            "camera_intrinsics_reported": getattr(self.backend, "camera_intrinsics", None),
            "camera_intrinsics_rendered": getattr(self.backend, "camera_intrinsics_rendered", None),
            "rates_hz": {
                "physics": round(1 / self.config["sim"]["physics_dt"]),
                "control": round(
                    1 / (self.config["sim"]["physics_dt"] * self.config["sim"]["control_decimation"])
                ),
                "policy_render": robodojo_rig.POLICY_RENDER,
            },
            "clutter_initial_state": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "position": item.position.tolist(),
                    "size": item.size.tolist(),
                    "mass": item.mass,
                    "label": item.label,
                }
                for item in getattr(self.backend, "clutter_specs", [])
            ],
            "clutter_robot_contact_count": getattr(self.backend, "clutter_robot_contact_count", 0),
            "initial_state": {
                name: {
                    "asset": obj.asset,
                    "position": obj.position.tolist(),
                    "orientation": obj.orientation.tolist(),
                    "color": obj.color.tolist(),
                    "mass": obj.mass,
                    "friction": obj.friction,
                    "label": obj.label,
                    "held_until_event": name in layout.held,
                    "target": layout.assignments.get(name),
                }
                for name, obj in layout.objects.items()
            },
            "targets": {
                name: {
                    "position": target.position.tolist(),
                    "half_extents": target.half_extents.tolist(),
                    "kind": target.kind,
                    "label": layout.labels.get(name, ""),
                }
                for name, target in layout.targets.items()
            },
            "final_state": self._final_state(),
            "zones": {
                name: {
                    "center": zone.center.tolist(),
                    "half_extents": None
                    if zone.half_extents is None
                    else zone.half_extents.tolist(),
                    "radius": zone.radius,
                }
                for name, zone in layout.zones.items()
            },
        }

    def _bedside_state(self):
        """Separate measured mechanism state, scored operations and symbolic device state."""
        task = self.task
        mechanism = task.layout.info.get("bedside_task")
        if not mechanism:
            return None
        state = task.previous
        return {
            "mechanism": mechanism,
            "requested_operation": getattr(task, "mode", task.name),
            "attempted_operations": sorted(task.attempted),
            "completed_operations": sorted(task.correct),
            "physical_success": task.physical_success,
            "temporal_success": task.temporal_success,
            "measured_controls": deepcopy(state.controls) if state is not None else {},
            "symbolic_device_state": deepcopy(getattr(task, "device_state", None)),
        }

    def _object_operations(self, name):
        mechanism = self.task.layout.info.get("bedside_task")
        if mechanism == "pump_panel":
            return (name.removeprefix("pump_"),)
        if mechanism == "iv_bag" and name == self.task.bag:
            return ("spike", "hang")
        if mechanism == "ventilator" and name == self.task.elbow:
            return ("elbow", "connector")
        if mechanism == "autoinjector":  # the barrel and its tip are one device
            return (self.task.pen, "site", "hold")
        if mechanism == "tourniquet" and name == self.task.rod:
            return ("windlass", "clip")
        return (name,)

    def _final_state(self):
        """Measured end-of-episode object states, for diagnosing failed episodes."""
        from roboclock.physics.geometry import orientation_error, upright_error

        state = self.task.previous
        if state is None:
            return {}
        result = {}
        bedside = bool(self.task.layout.info.get("bedside_task"))
        for name, obj in state.objects.items():
            operations = self._object_operations(name)
            moved = name in self.task.moved
            if bedside:
                initial = self.task.layout.objects[name]
                moved = moved or bool(
                    np.linalg.norm(obj.position - initial.position) > 0.001
                    or orientation_error(obj.orientation, initial.orientation) > np.deg2rad(3)
                    or any(op in self.task.attempted | self.task.correct for op in operations)
                )
            result[name] = {
                "position": obj.position.tolist(),
                "tilt_rad": float(upright_error(obj.orientation)),
                "grasped": bool(obj.grasped),
                "speed": float(np.linalg.norm(obj.velocity)),
                "contacts": sorted(obj.fixture_contacts),
                "moved": moved,
                "correct": all(op in self.task.correct for op in operations),
            }
            if bedside:
                result[name]["operations"] = list(operations)
        return result
