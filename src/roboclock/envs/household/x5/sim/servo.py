"""Measured-joint Cartesian servo for one X5 arm (bounded joint-position drives, no pose writes).

Two modes, both from the completed V1.0_X5 A1 episode:
- ``ik``: per control tick solve a short Cartesian step with bounded least squares on the URDF
  chain, then rate-limit the joint targets;
- ``jacobian``: damped least-squares step on the simulator's Jacobian.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.x5.scene_lib import BASES, H
from roboclock.envs.household.x5.sim.physics import (
    ARM_LOWER,
    ARM_UPPER,
    JAW_OPENING,
    JAW_TRAVEL,
    TCP_OFFSET,
    URDF,
)
from roboclock.envs.rates import CONTROL_DT

LOW, HIGH = np.array(ARM_LOWER), np.array(ARM_UPPER)
# Jaw travel rates (m/s per side): closing until contact, ramping the squeeze on a held part.
JAW_CLOSE_SPEED, JAW_SQUEEZE_SPEED = .024, .036


class Kinematics:
    """URDF forward kinematics to the tool point and a bounded pose solver."""

    def __init__(self, side):
        joints = ET.parse(URDF).getroot().findall('joint')
        self.base = np.eye(4)
        self.base[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
        self.base[:3, 3] = [*BASES[side], H+.012]
        self.specs = []
        for j in joints:
            m = np.eye(4)
            o = j.find('origin')
            m[:3, 3] = np.fromstring(o.get('xyz'), sep=' ')
            m[:3, :3] = R.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if j.get('type') == 'revolute' else None
            self.specs.append((m, axis))

    def fk(self, q):
        """eef_link pose (= link6 + 0.145 m along x): position, Rotation."""
        out = self.base.copy()
        i = 0
        for origin, axis in self.specs:
            out = out@origin
            if axis is not None:
                m = np.eye(4)
                m[:3, :3] = R.from_rotvec(axis*q[i]).as_matrix()
                out = out@m
                i += 1
        return out[:3, 3], R.from_matrix(out[:3, :3])

    def solve(self, pos, rot, seed, extra_seeds=()):
        def error(q):
            p, r = self.fk(q)
            return np.r_[(p-pos)*3., (rot*r.inv()).as_rotvec()*.35]
        best = None
        for q0 in [seed, np.array([-.5, 2., 1.8, -1., 0., 0.]), np.array([1., 2., 1.2, -1., 0., 0.]), *extra_seeds]:
            sol = least_squares(error, np.clip(q0, LOW+.001, HIGH-.001), bounds=(LOW+.0001, HIGH-.0001), max_nfev=160)
            if best is None or np.linalg.norm(sol.fun) < np.linalg.norm(best.fun):
                best = sol
            if np.linalg.norm(sol.fun) < 1e-4:
                break
        p, r = self.fk(best.x)
        return best.x, float(np.linalg.norm(p-pos)), float(np.linalg.norm((rot*r.inv()).as_rotvec()))


class Servo:
    def __init__(self, robot, side, mode='ik', control_dt=CONTROL_DT):
        from isaacsim.core.utils.types import ArticulationAction
        self.action = ArticulationAction
        self.robot, self.side, self.mode, self.dt = robot, side, mode, control_dt
        self.close_step, self.squeeze_step = JAW_CLOSE_SPEED*control_dt, JAW_SQUEEZE_SPEED*control_dt
        self.view = robot._articulation_view
        self.names = list(robot.dof_names)
        self.bodies = list(self.view.body_names)
        self.arm = [self.names.index(f'joint{i}') for i in range(1, 7)]
        self.jaws = [self.names.index(f'{x}_joint') for x in ['jawP', 'jawN']]
        self.eef = self.bodies.index('link6')
        self.target = robot.get_joint_positions().copy()
        self.error = [1., 1.]
        self.kin = Kinematics(side)
        self.limits = np.asarray(self.view.get_dof_limits())[0]

    def reset_target(self):
        self.target = self.robot.get_joint_positions().copy()

    def pose(self):
        """Measured tool point and orientation (link6 frame)."""
        t = np.asarray(self.view._physics_view.get_link_transforms())[0, self.eef]
        r = R.from_quat(t[3:])
        return t[:3]+r.apply([TCP_OFFSET, 0, 0]), r

    def joints(self):
        return self.robot.get_joint_positions()[self.arm]

    def jaw_gap(self):
        q = self.robot.get_joint_positions()[self.jaws]
        return float(JAW_OPENING-abs(q[0])-abs(q[1]))

    def _send(self, target):
        q = self.joints()
        target = np.clip(target, q-.12, q+.12)
        target = np.clip(target, self.limits[self.arm, 0]+.002, self.limits[self.arm, 1]-.002)
        self.target[self.arm] = target
        self.robot.apply_action(self.action(joint_positions=target, joint_indices=np.array(self.arm)))

    def move(self, pos, rot, speed=.24, turn_rate=1.0, tol=(.006, .10)):
        """One bounded Cartesian step toward (pos, rot). Returns True when within tolerance."""
        dt = self.dt
        p, r = self.pose()
        ep = np.asarray(pos)-p
        er = (rot*r.inv()).as_rotvec()
        self.error = [float(np.linalg.norm(ep)), float(np.linalg.norm(er))]
        if self.mode == 'ik':
            step = ep.copy()
            n = np.linalg.norm(step)
            if n > speed*dt:
                step *= speed*dt/n
            turn = er.copy()
            a = np.linalg.norm(turn)
            if a > turn_rate*dt:
                turn *= turn_rate*dt/a
            measured = self.joints()
            target, pe, re = self.kin.solve(p+step, R.from_rotvec(turn)*r, measured)
            if pe > .006 or re > .12:
                raise RuntimeError(f'{self.side}: Cartesian step outside X5 bounds at {p+step} ({pe:.4f} m, {re:.3f} rad)')
            self._send(self.target[self.arm]+np.clip(target-measured, -1.5*dt, 1.5*dt))
        else:
            if np.linalg.norm(ep) > .06:
                ep *= .06/np.linalg.norm(ep)
            jac = np.asarray(self.view.get_jacobians())[0]
            row = self.eef-(jac.shape[0] == len(self.bodies)-1)
            j = jac[row][:, self.arm].copy()
            j[:3] += np.cross(j[3:].T, r.apply([TCP_OFFSET, 0, 0])).T
            j[3:] *= .35
            v = np.r_[ep*4., er*1.4]
            if np.linalg.norm(v[:3]) > speed:
                v[:3] *= speed/np.linalg.norm(v[:3])
            dq = j.T@np.linalg.solve(j@j.T+np.eye(6)*.001, v)
            self._send(self.target[self.arm]+np.clip(dq, -1.5, 1.5)*dt)
        return self.error[0] < tol[0] and self.error[1] < tol[1]

    def move_joints(self, q, rate=1.2):
        """Rate-limited joint-space move (used for large repositioning)."""
        q = np.asarray(q)
        cur = self.target[self.arm]
        self._send(cur+np.clip(q-cur, -rate*self.dt, rate*self.dt))
        return float(np.max(np.abs(self.joints()-q)))

    def grip_each(self, travel_p, travel_n):
        """Per-jaw closing travel (m from the open stop): used for close-until-contact grasps."""
        self.robot.apply_action(self.action(joint_positions=np.array([-travel_p, travel_n]), joint_indices=np.array(self.jaws)))

    def grip(self, close, opening=JAW_OPENING, squeeze=0.):
        """close=True drives both jaws shut (contact limits them); else open to `opening`.

        `squeeze` (m) places the closing target beyond the closed stop, so a thin part is clamped with
        up to the drive force limit instead of stiffness x (half its thickness)."""
        travel = JAW_TRAVEL+squeeze if close else max(0., (JAW_OPENING-opening)/2)
        self.robot.apply_action(self.action(joint_positions=np.array([-travel, travel]), joint_indices=np.array(self.jaws)))
