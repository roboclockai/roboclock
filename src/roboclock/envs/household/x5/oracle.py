"""Diagnostic privileged oracle for the Dual-X5 household tasks (``--policy oracle --oracle``).

This is the controller of the accepted recordings (``<id>_x5/sim.py:Oracle.command``, run by
``scripts/household/run_x5_episode.py``) in policy form. It reads privileged state from the
environment: the probe's measured object poses and contact forces, the hidden process cues and
the simulation clock. Its Cartesian/joint servos (``x5/sim/servo.py``) are unchanged; their
articulation writes are captured into the 16-value action instead of being sent directly, so the
environment applies exactly the targets the recorder applies, at the same control ticks.

Differences from the recorder: an oracle failure (phase timeout, lost grasp, an IK step outside
the X5 bounds) or a finished fixed-horizon schedule does not end the episode; the policy then
holds its last targets and the environment runs to termination or ``max_time``.
"""

import sys

import numpy as np

from roboclock.envs.household.x5.interface import DOF_ORDER, SIDES, action_to_native, native_to_action


class _CapturingArticulation:
    """Stands in for one arm's SingleArticulation inside the oracle's Servo.

    Reads (joint positions, link transforms, Jacobians, limits) go to the real articulation;
    ``apply_action`` records the native joint targets, which persist until replaced, exactly as
    drive targets do in PhysX.
    """

    def __init__(self, robot, targets):
        self._robot = robot
        self.targets = np.asarray(targets, dtype=np.float64).copy()  # native, DOF_ORDER
        self.written = False  # apply_action called during the current command()
        self._slot = {robot.dof_names.index(name): k for k, name in enumerate(DOF_ORDER)}
        self.dof_names = robot.dof_names
        self._articulation_view = robot._articulation_view

    def get_joint_positions(self):
        return self._robot.get_joint_positions()

    def apply_action(self, action):
        self.written = True
        if action.joint_velocities is not None or action.joint_efforts is not None:
            raise ValueError("The household oracle commands joint positions only")
        indices = action.joint_indices
        if indices is None:
            indices = range(len(self._robot.dof_names))
        values = np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)
        for dof, value in zip(np.asarray(indices).reshape(-1).tolist(), values):
            self.targets[self._slot[int(dof)]] = value


class HouseholdX5Oracle:
    """``reset(obs)`` / ``act(obs)`` policy wrapping the task module's privileged controller."""

    def __init__(self, env):
        # roboclock eval only builds it with --oracle (policies/base.py); roboclock validate uses it
        # as the diagnostic controller.
        self.env = env

    def reset(self, observation):
        from roboclock.envs.household.x5.sim.servo import Servo

        episode = self.env.x5
        self.probe = episode.probe
        # Start from the targets the environment applied at reset (ready pose, jaws open).
        active = np.asarray(episode.commanded, dtype=np.float64)
        self.arms = {
            side: _CapturingArticulation(
                episode.robots[side], action_to_native(active[8 * k : 8 * k + 8])
            )
            for k, side in enumerate(SIDES)
        }
        self.servos = {
            side: Servo(
                self.arms[side], side, episode.control_modes[side], control_dt=episode.control_dt
            )
            for side in SIDES
        }
        for servo in self.servos.values():  # as the recorder did after settling
            servo.reset_target()
        self.failures = []
        self.stopped = False

    def act(self, observation):
        probe = self.probe
        if not self.stopped and not getattr(probe, "finished", False):
            measuring = probe.servos
            probe.servos = self.servos  # the controller actuates through the capturing servos
            for arm in self.arms.values():
                arm.written = False
            try:
                probe.command(self.env.x5.clock())
            except Exception as exc:  # the recorder ends the attempt on any controller error
                self.failures.append(f"{type(exc).__name__}: {exc}")
            finally:
                probe.servos = measuring
            # Arms the controller wrote, even with unchanged values: the recorder's PhysX writes
            # (each also wakes the articulation) are then reproduced exactly.
            self.env.x5.rewrite = {side for side, arm in self.arms.items() if arm.written}
            if probe.failures or self.failures:
                self.stopped = True
                print(
                    f"household oracle stopped at t={self.env.task.time:.2f} s: "
                    f"{list(probe.failures) + self.failures}",
                    file=sys.stderr,
                    flush=True,
                )
        return np.concatenate([native_to_action(self.arms[side].targets) for side in SIDES])
