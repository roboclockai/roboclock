"""Physical shared-cue exchange on a supported rest between fixed X5 arms."""
import numpy as np
from scipy.spatial.transform import Rotation


def begin(oracle):
    b = oracle.b
    hp, hr, _, cr = oracle.measured_tool()
    cp = np.asarray(b.last_state['objects']['cue']['position'])
    oracle.offset = hr.T @ (cp - hp)
    oracle.tool_rotation = hr.T @ cr
    oracle.exchange_rotation = oracle.tool_rotation.T
    oracle.exchange_goal = b.cue_exchange_home - oracle.exchange_rotation @ oracle.offset
    oracle.exchange_high = hp.copy()
    oracle.exchange_high[2] = 1.04
    oracle.exchange_neutral = np.array([.38, -.08, .99])
    oracle.rotation = hr.copy()
    oracle.exchange_attempted = True
    oracle.transition('exchange_lift')


def act(oracle):
    b = oracle.b
    phase = oracle.phase
    elapsed = b.time - oracle.start
    cue = b.last_state['objects']['cue']
    target = oracle.exchange_high.copy()
    rotation = oracle.rotation.copy()
    grip = 0.
    if phase in ('exchange_move_neutral', 'exchange_turn'):
        target = oracle.exchange_neutral.copy()
        if phase == 'exchange_turn':
            rotation = oracle.exchange_rotation
    elif phase in ('exchange_transfer', 'exchange_lower', 'exchange_release', 'exchange_retreat'):
        target = oracle.exchange_goal.copy()
        rotation = oracle.exchange_rotation
        if phase in ('exchange_transfer', 'exchange_retreat'):
            target[2] = .99
        if phase in ('exchange_release', 'exchange_retreat'):
            grip = .04
    oracle.goal, oracle.rotation = target.copy(), rotation.copy()
    error = b.move('main', target, rotation, grip=grip, speed=.23)
    supported = cue.get('support_contacts', {}).get('CueRest', 0.) > .05
    placed = (np.linalg.norm(np.asarray(cue['position'])[:2] - b.cue_exchange_home[:2]) < .012
              and abs(cue['position'][2] - b.cue_exchange_home[2]) < .012 and supported)
    if phase == 'exchange_lift' and error < .008:
        oracle.transition('exchange_move_neutral')
    elif phase == 'exchange_move_neutral' and error < .008:
        oracle.transition('exchange_turn')
    elif phase == 'exchange_turn' and error < .008 and elapsed > .4:
        oracle.transition('exchange_transfer')
    elif phase == 'exchange_transfer' and error < .008:
        oracle.transition('exchange_lower')
    elif phase == 'exchange_lower' and placed and elapsed > .3:
        oracle.transition('exchange_release')
    elif phase == 'exchange_release' and placed and elapsed > .45:
        oracle.transition('exchange_retreat')
    elif phase == 'exchange_retreat' and error < .008 and elapsed > .4:
        # Only the controller/sensor view changes. The real cue remains on the
        # contact-supported rest, ready for the other native gripper to grasp.
        b.active_side = 'left'
        b.filter_paths[:4] = b.rig.fingers['left'] + b.rig.fingers['right']
        hp, hq = b.rig.pose('left')
        hr = Rotation.from_quat(np.roll(hq, -1)).as_matrix()
        b.hand_targets['main'] = (hp.copy(), hr.copy())
        oracle.goal = hp.copy()
        oracle.rotation = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        oracle.last_grip = b.time
        oracle.excluded_target = oracle.excluded_at = None
        oracle.right_park = np.array([.30, -.30, 1.10])
        oracle.diagnostics.append(dict(time=b.time, phase='cue_exchange_complete',
                                       cue_supported=bool(supported), active_side='left'))
        oracle.transition('approach')
    elif elapsed > 6.:
        oracle.diagnostics.append(dict(time=b.time, phase='cue_exchange_timeout', stage=phase))
        oracle.transition('finished')
