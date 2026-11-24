"""Three genuine carton placements in the fixed workcell's90s budget.

Existing imported carton geometry is reused at declared SKU package sizes.
Products remain free native bodies; return trajectories use bounded original arms.
"""
import hashlib
import json

import numpy as np

from roboclock.envs.rates import CONTROL_HZ

from .putaway_arx_task import (
    PutawayArmOperation as ReviewedArmOperation,
)
from .putaway_arx_task import (
    PutawayARXController as ReviewedPutawayController,
)
from .putaway_arx_task import (
    PutawayARXScene as ReviewedPutawayScene,
)
from .putaway_arx_task import (
    resolve_putaway as resolve_reviewed_putaway,
)


def resolve_putaway_throughput(seed=0, *, layout=False, timing=False, physical=False, sku_order=False):
    """``sku_order`` permutes the SKU pick order (A, A, B) from its own SeedSequence child.
    It used to ride on ``layout``; it is now separate and off in every evaluation, because
    the scripted oracle is built for A, A, B (2026-11-20)."""
    d = resolve_reviewed_putaway(seed, layout=layout, timing=timing, physical=physical)
    d.pop('sha256', None)
    jitter = np.asarray(d['carton_xy_jitter_m'])
    # Two brake-kit packages share the A shelf in separate front/back positions.
    # Distinct filters package retains its blue SKU label and taller proportions.
    templates = {'A':dict(label='BRAKE KITS / A',slot='A-01',size_m=[.28,.10,.14],
                          destination_bounds_xy_m=[[-.25,1.49],[.09,2.01]]),
                 'B':dict(label='FILTERS / B',slot='B-01',size_m=[.28,.18,.26],
                          destination_center_xy_m=[.32,1.66],
                          destination_bounds_xy_m=[[.15,1.49],[.49,2.01]])}
    order = ['A','A','B']
    if sku_order:
        order = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[3]).permutation(order).tolist()
    products=[]
    seen_a=0
    source_y = [1.26, 1.08, .93]
    for index,sku in enumerate(order):
        row=dict(templates[sku])
        floor=.35182544341264272
        if sku=='A':
            row['destination_center_xy_m']=[-.08,1.75 if seen_a==0 else 1.61]
            seen_a+=1
        lift=.10
        products.append(dict(id=index,sku=sku,**row,source_floor_m=floor,
                             source_center_m=[.12+jitter[0],source_y[index]+jitter[1],floor+row['size_m'][2]/2+.002],
                             placement_tool_bias_xy_m=[0., .006] if sku=="B" else [0.,0.],
                             lift_height_m=lift, minimum_source_lift_m=max(.04,lift-.025),
                             source_clear_x_shift_m=0.,source_clear_y_m=1.36,
                             transport_clearance_m=.08 if index==0 else .35182544341264272+.10-d["shelf_support_floor_m"]))
    d.update(mounts_m=dict(left=[-.38,1.36,.36],right=[.62,1.36,.36]),
             transport_clearance_m=.35182544341264272 + .10 - d["shelf_support_floor_m"],
             placement_reference_bias_scope="B lower tool references compensate6mm measured gravity-seating drift in Y; scoring destination and free body pose remain unchanged.",
             transport_strategy="Constant source-lift altitude for front A and B; rear A uses80mm shelf clearance to remain within original reach, then vertical supported lowering.",
             carton_trajectory_speed_m_s=dict(lift=.055, clear_source=.10, transfer_to_shelf=.08, lower=.05),
             bimanual_command_tracking_gate_m=.010, minimum_grasp_hand_N=5.0, target_transport_hand_N=8.0,
             retention_force_upper_N=12.0, retention_outward_limit_m=.006, retention_outward_speed_m_s=.001,
             products=products,source_center_xy_m=[.12,source_y[0]], source_floor_m=products[0]['source_floor_m'],
             destination_center_xy_m=products[0]['destination_center_xy_m'],
             destination_bounds_xy_m=products[0]['destination_bounds_xy_m'],
             operation_required_boxes=3,required_shelved_boxes=3,mobile_base_required=False,
             operating_mode='three_carton_fixed_workcell', component_probe_index=1, component_probe_goal="shelf_arrival", component_preplaced_indices=[],
             grasp_force_profiles=[dict(minimum_N=15., target_N=25., upper_N=None),
                                   dict(minimum_N=5., target_N=8., upper_N=12., carry_target_N=20., carry_upper_N=28.),
                                   dict(minimum_N=8., target_N=12., upper_N=18., carry_target_N=20., carry_upper_N=28.)],
             source_support_raise_m=.14,
             source_support_adaptation='Reuse a second existing pallet underneath the original source pallet/tray; all three separated free cartons rest on the same native flat source floor. This avoids stock interlocking and keeps original ARX reach/caps unchanged.',
             scope='Three free cartons, mixed A/B SKU, native supported correct-shelf placement within90s. Fixed cell; no mobile delivery required per review, October22.',
             scene_adaptation='Same original NVIDIA carton mesh, UVs and native decomposition at declared280x100x140mm A and280x180x260mm B sizes; original shelf intact. A cartons occupy distinct back/front positions in A-01. The taller260mm B/filter package raises its side grasp40mm, providing the original wrist/camera envelope room to withdraw above the140mm A cartons; original product topology, rack and original robot geometry remain intact. Three separated free cartons, picked rear-to-front on the reused original pallet/tray; no source-box nesting or initial product contact.100mm native lift clears the original source tray without crossing unpicked cartons.',
             empty_return_speed_m_s=.16, empty_return_lower_speed_m_s=.12,
             empty_return_waypoint_blend_m=.006, empty_return_inward_shift_m=.05, empty_return_above_z_m=.53, empty_return_raise_z_m=.52,
             mounting_assumption='Original complete fixed X5 arms on360mm stands,1000mm apart atY1.36m; intact pallet/tray and rack; no mobile requirement.',
             return_path='Measured-pose Cartesian raise above original rack front beam, out of shelf, toward stock, then lower. Unchanged joint/effort/velocity caps.')
    d['bounds']['sku_orders']=[['A','B','A'],['A','A','B'],['B','A','A']] if sku_order else [['A','A','B']]
    d['sha256']=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    return d


class PutawayThroughputScene(ReviewedPutawayScene):
    def configure_multisku(self, cell, scale):
        super().configure_multisku(cell,scale)
        # Reset-only stock layout. Avoid physically nesting identical source
        # cartons: all three sit separately on the original resized tray floor.
        for name, height, old_depth, new_depth in [('StagingPallet', .24, .60, .70), ('LoadTray', .36, .50, .60)]:
            body = self.world.scene.get_object(name)
            factor = np.asarray(body.get_local_scale()).copy()
            factor[1] *= new_depth / old_depth
            body.set_local_scale(factor)
            position = np.array([.12, 1.06, height])
            body.set_world_pose(position=position)
            body.set_default_state(position=position)
        from .arx_cell_task import collide
        from .remaining_environments import ROOT
        cell.asset('Pilot/StagingPalletLower', ROOT / 'task48/recovered/pallet.usd', [.12,1.06,.07], [.80,.70,.14])
        self.static_colliders += collide(self.stage, ['/World/Pilot/StagingPalletLower'])
        # Explicit component-only initial conditions, applied before world.reset.
        # Prior stock is placed on the shelf for isolated carry probes; these
        # reset-only placements never receive throughput credit.
        for index in self.resolved.get("component_preplaced_indices", []):
            product = self.resolved["products"][index]
            center = np.r_[product["destination_center_xy_m"], self.target_floor + product["size_m"][2]/2 + .002]
            self.boxes[index].set_world_pose(position=center)
            self.boxes[index].set_default_state(position=center)
        # The reviewer removed evaluator/progress text from RGB. Keep one ID per SKU.
        for index in range(len(self.resolved['products'])):
            self.stage.RemovePrim(f'/World/Signs/Slot{index}')
        for index, (sku, x, color) in enumerate([('A', -.08, (31,94,55)), ('B', .32, (30,65,122))]):
            cell.sign(f'Slot{index}', sku + '-01', [x,1.455,self.target_floor+.012], [.36,.065], background_rgb=color)
        del self.count_display
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asset_manifest['source_stock_layout'] = dict(pallet_center_m=[.12,1.06,.24], pallet_size_m=[.8,.7,.2], tray_center_m=[.12,1.06,.36], tray_size_m=[.7,.6,.04], products='Three separate free cartons; no initial nesting or product-product contact')
    def observe(self):
        observation = super().observe()
        observation["bimanual_control"] = getattr(self, "controller_debug", None)
        return observation
    def final_valid(self, o):
        return bool(len(self.placed)>=3 and all(self.valid_product(i) for i in self.placed)
                    and o['time_s'] <= self.resolved['budget_s']+.02)


class ThroughputArmOperation(ReviewedArmOperation):
    def __init__(self, scene, mode):
        super().__init__(scene, mode)
        self.force_limits = scene.resolved["grasp_force_profiles"][scene.active_index]
        self.previous_retention_press = 0.0
        self.minimum_grasp_hand_N = self.force_limits["minimum_N"]
        self.target_transport_hand_N = self.force_limits["target_N"]
    def step(self, observation):
        # Regulate excessive squeeze as well as lost contact. Excessive opposing
        # pressure can saturate the original6Nm wrist drive; hardware caps and
        # native hold/release predicates remain unchanged.
        carrying = self.phase in ["clear_source", "transfer_to_shelf"]
        self.target_transport_hand_N = self.force_limits.get("carry_target_N", self.force_limits["target_N"]) if carrying else self.force_limits["target_N"]
        upper = self.force_limits.get("carry_upper_N", self.force_limits["upper_N"]) if carrying else self.force_limits["upper_N"]
        if self.phase in ["lift", "hold", "clear_source", "transfer_to_shelf"]:
            if upper is not None and min(observation["hand_forces_N"].values()) > upper:
                self.retention_press = max(-self.s.resolved["retention_outward_limit_m"],
                    self.retention_press - self.s.resolved["retention_outward_speed_m_s"] / CONTROL_HZ)
        super().step(observation)
        self.s.controller_debug = dict(phase=self.phase, force_limits=self.force_limits, retention_press_m=self.retention_press,
            arms={side: dict(command=m.command.tolist(), position_integral=m.correction.tolist(),
                             orientation_integral=m.orientation_correction.tolist()) for side,m in self.m.items()})
    def move_pair(self, goals, speed):
        speed = self.s.resolved["carton_trajectory_speed_m_s"].get(self.phase, speed)
        if self.phase == "transfer_to_shelf":
            clearance = self.s.resolved["products"][self.s.active_index]["transport_clearance_m"]
            dz = clearance - self.s.resolved["transport_clearance_m"]
            goals = {side: np.asarray(goal) + [0, 0, dz] for side, goal in goals.items()}
        if self.phase == "lower":
            bias = self.s.resolved["products"][self.s.active_index]["placement_tool_bias_xy_m"]
            goals = {side: np.asarray(goal) + [bias[0], bias[1], 0] for side, goal in goals.items()}
        loaded = self.phase in ["lift", "hold", "clear_source", "transfer_to_shelf", "lower"]
        if not loaded:
            return super().move_pair(goals, speed)
        # Apply the bounded squeeze increment independently of the long carry
        # path fraction. Otherwise a millimetre grip correction is diluted by
        # the remaining300mm translation and cannot recover real clamp force.
        delta_press = self.retention_press - self.previous_retention_press
        for side, sign in [("left", 1), ("right", -1)]:
            self.m[side].command = self.m[side].command + [sign*delta_press, 0, 0]
        self.previous_retention_press = self.retention_press
        goals = {side: np.asarray(goal) + [self.retention_press if side == "left" else -self.retention_press, 0, 0]
                 for side, goal in goals.items()}
        # One shared path fraction keeps the two clamp references together.
        # ArmMotion's independent native-following gates otherwise permit one
        # hand to lead the other during a loaded turn. These are joint-drive
        # references only; the carton and robot poses are never overwritten.
        lag = max(np.linalg.norm(self.s.dual.tool_world(side)[0] - self.m[side].command)
                  for side in ["left", "right"])
        remaining = {side: goals[side] - self.m[side].command for side in ["left", "right"]}
        fraction = min(1.0, speed / CONTROL_HZ / max(max(np.linalg.norm(v) for v in remaining.values()), 1e-12))
        if lag <= self.s.resolved["bimanual_command_tracking_gate_m"]:
            for side in ["left", "right"]:
                self.m[side].command = self.m[side].command + fraction * remaining[side]
        done = []
        for side in ["left", "right"]:
            m = self.m[side]
            m.move(m.command.copy(), 0.0, self.R[side])
            actual, rotation = self.s.dual.tool_world(side)
            from scipy.spatial.transform import Rotation
            done.append(np.linalg.norm(actual - goals[side]) < .003 and
                        np.linalg.norm(m.command - goals[side]) < 1e-5 and
                        Rotation.from_matrix(self.R[side] @ rotation.T).magnitude() < .03)
        return all(done)


class PutawayThroughputController(ReviewedPutawayController):
    def __init__(self, scene, mode):
        self.s, self.mode = scene, mode
        if mode == "grasp":
            scene.activate(scene.resolved["component_probe_index"])
        self.operation = ThroughputArmOperation(scene, mode)
        self.phase, self.failure, self.complete = "sku0:preposition", None, False
        self.return_phase = self.return_entered = self.return_origins = None
    def step(self, o):
        super().step(o)
        goal = self.s.resolved.get("component_probe_goal")
        suffix = {"shelf_arrival":"actual_shelf_transport_arrived_s", "placement":"released_correct_shelf_retention_s"}.get(goal,"hold_pose_validation")
        probe_key = f"{self.s.active_index}_{suffix}"
        if self.mode == "grasp" and probe_key in self.s.events and not self.failure:
            self.complete = True
            self.phase = "native_component_probe_complete"
            return
        if self.complete and not self.failure:
            self.s.success=self.s.final_valid(o)
            self.s.cue_state='SHELVED'
    def empty_waypoint(self, side, goal):
        op, s = self.operation, self.s
        done = op.m[side].move(goal, s.resolved["empty_return_speed_m_s"], op.R[side])
        actual, rotation = s.dual.tool_world(side)
        from scipy.spatial.transform import Rotation
        tolerance = s.resolved["empty_return_waypoint_blend_m"]
        return done or (np.linalg.norm(actual-goal) < tolerance and
            np.linalg.norm(op.m[side].command-goal) < tolerance and
            Rotation.from_matrix(op.R[side]@rotation.T).magnitude()<.03)
    def step_return(self, observation):
        s,op,t=self.s,self.operation,observation['time_s']
        next_product=s.resolved['products'][s.active_index+1]
        if self.return_phase=='raise':
            goals={side:np.r_[p[:2],s.resolved["empty_return_raise_z_m"]] for side,p in self.return_origins.items()}
            done=[self.empty_waypoint(side, goals[side]) for side in ['left','right']]
            self.phase='empty_return:raise_clear'
            if all(done):
                self.next_prepose={side:np.asarray(next_product['source_center_m'])+[
                    sign*(s.resolved['side_tool_offset_x_m']+.06),0,s.resolved['side_tool_offset_z_m']+.02]
                    for side,sign in [('left',-1),('right',1)]}
                self.return_front={side:np.array([self.next_prepose[side][0] + (1 if side=='left' else -1)*s.resolved['empty_return_inward_shift_m'],1.36,s.resolved["empty_return_raise_z_m"]]) for side in ['left','right']}
                self.return_phase='front'
                self.return_entered=t
        elif self.return_phase=='front':
            done=[self.empty_waypoint(side, self.return_front[side]) for side in ['left','right']]
            self.phase='empty_return:exit_shelf'
            if all(done):
                self.next_prepose={side:np.asarray(next_product['source_center_m'])+[
                    sign*(s.resolved['side_tool_offset_x_m']+.06),0,s.resolved['side_tool_offset_z_m']+.02]
                    for side,sign in [('left',-1),('right',1)]}
                self.return_above={side:np.r_[p[:2],s.resolved['empty_return_above_z_m']] for side,p in self.next_prepose.items()}
                self.return_phase='above'
                self.return_entered=t
        elif self.return_phase=='above':
            done=[self.empty_waypoint(side, self.return_above[side]) for side in ['left','right']]
            self.phase='empty_return:above_stock'
            if all(done):
                self.return_phase='lower'
                self.return_entered=t
        elif self.return_phase=='lower':
            done=[op.m[side].move(self.next_prepose[side],s.resolved["empty_return_lower_speed_m_s"],op.R[side]) for side in ['left','right']]
            self.phase='empty_return:lower_to_stock'
            if all(done):
                s.activate(s.active_index+1)
                self.operation=ThroughputArmOperation(s,self.mode)
                self.return_phase=None
                s.events[f'empty_return_complete_{s.active_index}_s']=t
                self.phase=f'sku{s.active_index}:preposition'
        if self.return_phase is not None and t-self.return_entered>18:
            self.failure='Empty Cartesian return timeout '+self.return_phase
