"""Dual ARX X5 kinematic screen: URDF forward kinematics and bounded numerical IK.

This is a geometry-only reach screen for initial-state design. It ignores
collisions, payload, joint torques and gripper aperture. A pass means the
tool point can reach the target with the requested approach direction inside
the URDF joint limits; it does not establish a feasible trajectory.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

URDF = Path(__file__).resolve().parents[1]/'assets/x5/robot/arx5-sdk/models/X5.urdf'
from roboclock.envs.household.x5.rig import BASE_YAW_DEG, BASES, COUNTER_TOP, MOUNT_THICKNESS  # noqa: F401 (BASES: active layout, rig.use)

BASE_YAW = math.radians(BASE_YAW_DEG)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _axis_angle(axis, theta):
    a = np.asarray(axis, float)/np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3)+math.sin(theta)*k+(1-math.cos(theta))*k@k


class X5:
    def __init__(self, urdf=URDF):
        tree = ET.parse(urdf)
        self.chain = []
        for j in tree.findall('joint'):
            o = j.find('origin')
            T = np.eye(4)
            T[:3, :3] = _rpy(*np.fromstring(o.get('rpy', '0 0 0'), sep=' '))
            T[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
            axis = j.find('axis')
            lim = j.find('limit')
            self.chain.append(dict(name=j.get('name'), type=j.get('type'), T=T,
                                   axis=None if axis is None else np.fromstring(axis.get('xyz'), sep=' '),
                                   lo=None if lim is None else float(lim.get('lower')),
                                   hi=None if lim is None else float(lim.get('upper'))))
        self.revolute = [c for c in self.chain if c['type'] == 'revolute']
        # joint1 is mechanically limited to +-10 rad in the URDF; use one turn.
        self.lower = np.array([max(c['lo'], -math.pi) for c in self.revolute])
        self.upper = np.array([min(c['hi'], math.pi) for c in self.revolute])

    def base(self, side):
        T = np.eye(4)
        T[:3, :3] = _rpy(0, 0, BASE_YAW)
        T[:3, 3] = [*BASES[side], COUNTER_TOP+MOUNT_THICKNESS]
        return T

    def fk(self, q, side):
        T = self.base(side)
        i = 0
        frames = []
        for c in self.chain:
            T = T@c['T']
            if c['type'] == 'revolute':
                R = np.eye(4)
                R[:3, :3] = _axis_angle(c['axis'], q[i])
                T = T@R
                i += 1
            frames.append(T.copy())
        return T, frames

    def ik(self, side, target, approach=None, q0=None, restarts=24, seed=0):
        """Return (q, position_error_m, approach_error_deg)."""
        target = np.asarray(target, float)
        want = None if approach is None else np.asarray(approach, float)/np.linalg.norm(approach)
        rng = np.random.default_rng(seed)

        def residual(q):
            T, _ = self.fk(q, side)
            r = [T[:3, 3]-target]
            if want is not None:
                r.append(.15*(T[:3, 0]-want))
            return np.concatenate(r)

        best = None
        starts = [np.clip(q0, self.lower, self.upper)] if q0 is not None else []
        starts += [rng.uniform(self.lower, self.upper) for _ in range(restarts)]
        for s in starts:
            sol = least_squares(residual, s, bounds=(self.lower, self.upper), xtol=1e-10, ftol=1e-10, max_nfev=400)
            T, _ = self.fk(sol.x, side)
            pe = float(np.linalg.norm(T[:3, 3]-target))
            ae = 0. if want is None else float(math.degrees(math.acos(np.clip(T[:3, 0]@want, -1, 1))))
            score = pe+ae/1000
            if best is None or score < best[3]:
                best = (sol.x, pe, ae, score)
            if pe < .002 and ae < 3:
                break
        return best[0], best[1], best[2]


APPROACHES = {'down': (0, 0, -1), 'forward': (0, 1, 0), 'forward_down': (0, .707, -.707),
              'left': (-1, 0, 0), 'right': (1, 0, 0), 'back_toward_robot': (0, -1, 0),
              'right_down': (.643, 0, -.766), 'left_down': (-.643, 0, -.766),
              # C1 handle grasps: tool pitched 20 deg down at azimuth 0 / 45 / 90 deg
              'right_down20': (.940, 0, -.342), 'forward_right_down20': (.664, .664, -.342), 'forward_down20': (0, .940, -.342),
              'left_down20': (-.940, 0, -.342), 'forward_left_down20': (-.664, .664, -.342), 'left_forward_down20': (-.868, .360, -.342)}


def screen(points, robot=None, pos_tol=.005, ang_tol=15.):
    """points: list of dict(name, side, xyz, approaches=[names]). Returns rows with best result."""
    robot = robot or X5()
    rows = []
    for p in points:
        best = None
        for name in p.get('approaches', ['down']):
            q, pe, ae = robot.ik(p['side'], p['xyz'], APPROACHES[name])
            ok = pe <= pos_tol and ae <= ang_tol
            row = dict(name=p['name'], side=p['side'], target_m=[round(float(v), 4) for v in p['xyz']], approach=name,
                       reachable=bool(ok), position_error_m=round(pe, 4), approach_error_deg=round(ae, 1),
                       joint_solution_rad=[round(float(v), 3) for v in q],
                       horizontal_distance_from_base_m=round(float(np.hypot(p['xyz'][0]-BASES[p['side']][0], p['xyz'][1]-BASES[p['side']][1])), 3))
            if best is None or (row['reachable'] and not best['reachable']) or (row['reachable'] == best['reachable'] and pe+ae/1000 < best['position_error_m']+best['approach_error_deg']/1000):
                best = row
            if ok:
                break
        best['purpose'] = p.get('purpose', '')
        rows.append(best)
    return rows


if __name__ == '__main__':
    r = X5()
    for q in ([0, 1.15, 1.60, 0, .35, 0], [0, .55, 1.55, -.55, 0, 0], [0, .35, 1.25, -.35, 0, 0]):
        T, _ = r.fk(np.array(q), 'Left')
        print(q, np.round(T[:3, 3], 3), np.round(T[:3, 0], 2))
