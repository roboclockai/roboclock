"""Experimental physical white-ball repositioning for a timed pool drill.

The evaluator observes contacts and motion independently of controller phases.
This variant is opt-in; single-target reset probes are not clearance evidence.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .pool import Clearance
from .pool_x5 import X5ClearanceOracle


def reset_zones(b, state, pots):
    """Legal spots derived from measured remaining targets, independently of policy."""
    c = b.config['pool']
    if c.get('placement_rule') != 'behind_next_target':
        return [np.asarray(z) for z in c['placement_zones']]
    remaining = [n for n in b.pool_targets if n not in pots]
    if not remaining:
        return [np.asarray(c['placement_zones'][0])]
    name = remaining[0]
    p = np.asarray(state['objects'][name]['position'])
    pocket = np.asarray(b.pockets[c['assigned_pockets'][name]])
    direction = pocket - p[:2]
    direction /= max(np.linalg.norm(direction), 1e-9)
    lower, upper = b.pool_layout.bed_bounds
    zones = []
    rests = [(b.cue_home + [b.cue_grip_x, 0, -.028], np.array([.10, .08, .032]))]
    if hasattr(b, 'cue_exchange_home'):
        rests.append((b.cue_exchange_home + [b.cue_grip_x, 0, -.028], np.array([.10, .10, .032])))
    for distance in c.get('reset_distances', [.10, .12, .14]):
        z = np.array([*(p[:2] - distance * direction), .773])
        if not (np.all(z[:2] > lower + b.ball_radius + .003)
                and np.all(z[:2] < upper - b.ball_radius - .003)):
            continue
        if any(np.linalg.norm(z - np.asarray(state['objects'][n]['position'])) < 2 * b.ball_radius + .006
               for n in remaining):
            continue
        if any(np.linalg.norm(z - np.clip(z, center - size / 2, center + size / 2)) < b.ball_radius + .003
               for center, size in rests):
            continue
        zones.append(z)
    return zones


class SpotClearance(Clearance):
    def __init__(self, b):
        super().__init__(b)
        self.resets = []
        self.arm_collision_since = None
        self.reset_open = False
        self.reset_grasp_since = None
        self.reset_release_since = None
        self.reset_grasp_offset = None
        self.reset_grasped = self.reset_lifted = False

    @property
    def reset_required(self):
        return len(self.shots) > len(self.resets) and (
            len(self.pots) < len(self.b.pool_targets)
            or self.b.config['pool'].get('reset_probe', False))

    def hand_contact_allowed(self, name, s):
        f = s['objects'][name]['forces']
        return (name == 'cue_ball' and self.reset_open
                and max(f[:2]) <= .03)

    def update(self, s):
        if self.done:
            return
        if s["time"] > self.c["pool_horizon"] + 1e-8:
            super().update(s)
            return
        if s.get("arm_contact_peak", 0.) > 2.:
            if self.arm_collision_since is None:
                self.arm_collision_since = s["time"]
            if s["time"] - self.arm_collision_since > .04:
                self.fail("arm_collision", s["time"])
        else:
            self.arm_collision_since = None
        if self.reset_required and not self.reset_open:
            cue = s['objects']['cue']
            rotation = Rotation.from_quat(np.roll(cue['orientation'], -1))
            lowest = min((rotation.apply([x, 0, 0]) + cue['position'])[2]
                         for x in (-.139, .186)) - .040
            if self.ready and lowest > .83:
                self.reset_open = True
                self.reset_grasped = self.reset_lifted = False
                self.reset_grasp_since = self.reset_release_since = None
                self.events.append(dict(event='reset_allowed', time=s['time']))
        super().update(s)
        if self.reset_open:
            o = s['objects']['cue_ball']
            p = np.asarray(o['position'])
            hp = np.asarray(s['hands']['left']['position'])
            f = o['forces']
            if min(f[2:4]) > .03:
                offset = p - hp
                if self.reset_grasp_since is None or np.linalg.norm(offset - self.reset_grasp_offset) > .004:
                    self.reset_grasp_since = s['time']
                    self.reset_grasp_offset = offset.copy()
                if s['time'] - self.reset_grasp_since >= .15:
                    self.reset_grasped = True
                if self.reset_grasped and p[2] > .825:
                    self.reset_lifted = True
            else:
                self.reset_grasp_since = None
            zones = reset_zones(self.b, s, self.pots)
            in_zone = min((np.linalg.norm(p[:2] - z[:2]) for z in zones), default=np.inf) < .012
            released = max(f[:4]) <= .03
            stable = (np.linalg.norm(o['velocity'][:2]) < .004
                      and abs(o['velocity'][2]) < .03)
            supported = o['support_contacts'].get('PoolBed', 0.) > .05
            accepted = (self.reset_lifted and in_zone and released and stable and supported
                        and np.linalg.norm(hp - p) > .08
                        and abs(p[2] - .773) < .004)
            if accepted:
                if self.reset_release_since is None:
                    self.reset_release_since = s['time']
                if s['time'] - self.reset_release_since >= .25:
                    self.resets.append(dict(time=s['time'], position=p.tolist(), shot=len(self.shots)))
                    self.events.append(dict(event='white_reset_complete', **self.resets[-1]))
                    self.reset_open = False
            else:
                self.reset_release_since = None
            self.ready = self.previous_ready = False
        if self.reset_required:
            self.phase = 'WHITE BALL RESET'

    def display(self):
        activity = 'RESET' if self.reset_open else 'CLEAR' if len(self.pots) == len(self.b.pool_targets) else 'PLAY'
        return (f'{self.valid_count}/{len(self.b.pool_targets)} POTS / {max(0, self.c["pool_horizon"] - self.b.time):04.1f}s LEFT',
                f'SPOT CLEARANCE / {activity}')

    def result(self):
        r = super().result()
        complete = self.valid_count == len(self.b.pool_targets) and not self.failures
        r.update(variant='spot_clearance', reset_events=self.resets,
                 diagnostic_probe=len(self.b.pool_targets) != 3 or self.b.config['pool'].get('reset_probe', False),
                 physical_success=complete,
                 success=complete and r['temporal_success'],
                 complete_clearance=complete)
        if r['diagnostic_probe']:
            r['success'] = False
            r['complete_clearance'] = False
            r['diagnostic_targets_potted'] = self.valid_count == len(self.b.pool_targets)
        return r


class SpotOracle(X5ClearanceOracle):
    def __init__(self, b, task):
        super().__init__(b, task)
        self.reset_rotation = Rotation.from_euler('y', b.config['pool'].get('reset_pitch', 180), degrees=True).as_matrix()
        self.reset_home_p, self.reset_home_q = b.rig.pose("left")
        self.reset_home_rotation = Rotation.from_quat(np.roll(self.reset_home_q, -1)).as_matrix()
        self.reset_hold = None
        self.reset_offset = np.zeros(3)
        self.reset_destination = np.asarray(b.config['pool']['placement_zones'][0])

    def act(self):
        b, t = self.b, self.task
        if min(b.last_state["objects"]["cue"]["forces"][:2]) > .025:
            self.last_grip = b.time
        if self.phase.startswith('reset_'):
            self.act_reset()
            return
        if self.phase == 'wait' and t.reset_open:
            zones = reset_zones(b, b.last_state, t.pots)
            if not zones:
                t.fail('no_legal_reset_spot', b.time)
                self.transition('finished')
                return
            if b.config['pool'].get('placement_rule') == 'behind_next_target':
                self.reset_destination = zones[0]
            else:
                remaining = [n for n in b.pool_targets if n not in t.pots]
                index = b.pool_targets.index(remaining[0]) if remaining else 0
                self.reset_destination = zones[min(index, len(zones) - 1)]
            self.cue_park = np.array(b.config['pool'].get('cue_park', [1., -.18, 1.10]))
            self.cue_park_rotation = self.rotation.copy()
            self.transition('reset_park')
            return
        if self.phase == 'wait' and t.reset_required:
            hold = self.goal.copy()
            hold[2] = .99
            b.move('main', hold, self.rotation, grip=0., speed=.18)
            return
        super().act()

    def act_reset(self):
        b, t = self.b, self.task
        main_goal = (np.array([b.pool_layout.center_x + .10, -.14, 1.03])
                     if self.phase == 'reset_cue_return' else self.cue_park)
        park_error = b.move('main', main_goal, self.cue_park_rotation, grip=0., speed=.15)
        p = np.asarray(b.last_state['objects']['cue_ball']['position'])
        hp = np.asarray(b.last_state['hands']['left']['position'])
        elapsed = b.time - self.start
        target, grip, speed = hp.copy(), .032, .15
        phase = self.phase
        if phase == 'reset_park':
            if park_error < .012:
                self.transition('reset_approach')
            elif elapsed > 8.:
                t.fail('reset_cue_park_timeout', b.time)
                self.transition('finished')
            return
        if phase == 'reset_approach':
            target = p + [0, 0, .15]
            if np.linalg.norm(target - hp) < .008:
                self.transition('reset_descend')
        elif phase == 'reset_descend':
            target = p + [0, 0, b.config['pool'].get('reset_grasp_height', .012)]
            speed = .06
            if np.linalg.norm(target - hp) < .004:
                self.reset_hold = target.copy()
                self.transition('reset_grasp')
        elif phase == 'reset_grasp':
            target, grip = self.reset_hold.copy(), .014
            if t.reset_grasped:
                self.reset_offset = p - hp
                self.transition('reset_lift')
        elif phase == 'reset_lift':
            target, grip = self.reset_hold.copy(), .014
            target[2] = .98
            if np.linalg.norm(target - hp) < .008 and t.reset_lifted:
                self.transition('reset_transfer')
        elif phase == 'reset_transfer':
            target, grip = self.reset_destination - self.reset_offset, .014
            target[2] = .98
            if np.linalg.norm(target - hp) < .008:
                self.transition('reset_lower')
        elif phase == 'reset_lower':
            target, grip, speed = self.reset_destination - self.reset_offset, .014, .06
            if np.linalg.norm(target - hp) < .004:
                self.transition('reset_release')
        elif phase == 'reset_release':
            target = self.reset_destination - self.reset_offset
            grip = .014 + .018 * min(1., elapsed / .6)
            obj = b.last_state['objects']['cue_ball']
            if elapsed > .8 and max(obj['forces'][2:4]) < .03:
                self.transition('reset_retreat')
        elif phase == 'reset_retreat':
            target = self.reset_destination - self.reset_offset + [-.10, 0, .20]
            if np.linalg.norm(target - hp) < .01 and not t.reset_open:
                self.transition('reset_home')
        elif phase == 'reset_home':
            target = self.reset_home_p
            if np.linalg.norm(target - hp) < .01 and elapsed > 1.:
                self.excluded_target = self.excluded_at = None
                self.transition('finished' if b.config['pool'].get('reset_probe') else 'reset_cue_return')
        elif phase == 'reset_cue_return':
            target = self.reset_home_p
            if park_error < .01:
                self.goal = main_goal.copy()
                self.transition('wait')
        rotation = self.reset_home_rotation if phase in ('reset_home', 'reset_cue_return') else self.reset_rotation
        b.move('left', target, rotation, grip=grip, speed=speed)
        if elapsed > 8.:
            t.fail('reset_controller_timeout_' + phase, b.time)
            self.transition('finished')
