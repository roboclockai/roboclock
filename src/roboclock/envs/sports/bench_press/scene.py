"""Bench, open J-hooks and a standard shaft without auxiliary robot handles."""
import numpy as np
from roboclock.envs.sports._shared.workflow.scenes import solid,rigid,display
from roboclock.envs.sports._shared.workflow.geometry import rod


def build(b):
    from pxr import UsdGeom,UsdPhysics,PhysxSchema
    from roboclock.envs.sports.arcade.refinement import pbr,panel
    from .character import Athlete
    c=b.config['workflow']
    b.grip_y=c['bench_grip_y']
    b.bar_home=np.array([c['rack_x'],0.,c['rack_height']])
    pbr(b.appearance,'bench_leather',[.12,.16,.17],roughness=.65,texture='leather_white',uv=(3,3))
    pbr(b.appearance,'rack_metal',[.20,.23,.23],roughness=.30)
    solid(b,'/World/Bench/Platform',[.55,0,.025],[2.70,2.0,.05],'rubber')
    solid(b,'/World/Bench/Cushion',[.65,0,.485],[1.25,.29,.075],'bench_leather',.025,True)
    solid(b,'/World/Bench/HeadPad',[.10,0,.49],[.29,.28,.085],'bench_leather',.026,True)
    for i,x in enumerate([.2,1.10]):
        solid(b,f'/World/Bench/Foot{i}',[x,0,.07],[.16,.60,.08],'rack_metal',collision=True)
        solid(b,f'/World/Bench/Leg{i}',[x,0,.27],[.08,.10,.40],'rack_metal',collision=True)
    solid(b,'/World/Bench/Spine',[.65,0,.40],[1.25,.09,.10],'steel',collision=True)
    solid(b,'/World/Bench/RobotPlinth',[0,0,c['bench_mount']/2],[.22,.45,c['bench_mount']],'ink')
    b.support_paths=[]
    for side,sign in [('Left',1),('Right',-1)]:
        path='/World/Rack'+side
        root=UsdGeom.Xform.Define(b.stage,path)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
        y=sign*.52
        solid(b,path+'/Post',[c['rack_x']-.075,y,.66],[.06,.065,1.24],'rack_metal',collision=True)
        solid(b,path+'/Foot',[c['rack_x'],y,.07],[.52,.20,.08],'rack_metal',collision=True)
        solid(b,path+'/Seat',[c['rack_x'],y,c['rack_height']-.020],[.10,.065,.012],'rubber',.003,True)
        solid(b,path+'/FrontLip',[c['rack_x']+.055,y,c['rack_height']-.005],[.012,.065,.048],'steel',.004,True)
        solid(b,path+'/Back',[c['rack_x']-.055,y,c['rack_height']+.025],[.012,.065,.10],'steel',.004,True)
        for z in np.arange(.35,1.22,.08):
            rod(b,path+f'/Bolt{int(z*100)}',[c['rack_x']-.11,y-.034,z],[c['rack_x']-.11,y+.034,z],.008,'steel')
        b.support_paths.append(path)
    root=UsdGeom.Xform.Define(b.stage,'/World/HumanTorso')
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    collider,_=b.collider('/World/HumanTorso/Chest','cube',[.55,0,.645],[.43,.31,.17],[1,1,1])
    UsdGeom.Imageable(collider).MakeInvisible()
    b.support_paths.append('/World/HumanTorso')
    path='/World/Objects/bar'
    UsdGeom.Xform.Define(b.stage,path)
    rod(b,path+'/Shaft',[0,-.72,0],[0,.72,0],.014,'steel',True)
    for i,y in enumerate([-.65,.65]):
        rod(b,path+f'/Plate{i}',[0,y-.025,0],[0,y+.025,0],.13,'plate_blue',True)
        rod(b,path+f'/Hub{i}',[0,y-.032,0],[0,y+.032,0],.032,'steel',True)
        rod(b,path+f'/Collar{i}',[0,y-.050,0],[0,y-.035,0],.025,'brass',True)
    for i,y in enumerate([-.36,-.23,.23,.36]):
        for j in range(8):
            yy=y-.028+j*.008
            rod(b,path+f'/Knurl{i}_{j}',[0,yy,0],[0,yy+.002,0],.0141,'ink')
    # The bar starts on the hooks at its seeded offset (``initial_bar_offset``, applied since
    # 2026-11-20; ``sports/spatial.py`` draws it for roboclock eval, the lifter's unrack target
    # is still the rack centre).
    obj=rigid(b,'bar',b.bar_home+np.asarray(c.get('initial_bar_offset',[0.,0.,0.])), c['bench_mass'])
    api=PhysxSchema.PhysxRigidBodyAPI.Apply(obj.prim)
    api.CreateLinearDampingAttr(.05)
    api.CreateAngularDampingAttr(.4)
    b.athlete=Athlete(b)
    # Keep the accepted studio's HDR, wall palette and sourced room props.
    panel(b,b.appearance,'/World/Bench/Brand',[.65,1.66,2.10],[2.0,.45],
          'HUMAN + ROBOT','BENCH PRESS / REACTIVE ASSIST',style='title',accent=(222,177,102))
    UsdGeom.Imageable(b.stage.GetPrimAtPath('/World/Studio/Brand')).MakeInvisible()
    for i,x in enumerate([-.9,-.6,-.3]):
        rod(b,f'/World/Bench/StoredPlate{i}',[x,1.10,.50],[x,1.16,.50],.15,['plate_blue','plate_red','rubber'][i])
        rod(b,f'/World/Bench/Peg{i}',[x,1.03,.50],[x,1.23,.50],.015,'steel')
    solid(b,'/World/Bench/Storage',[-.60,1.25,.5],[1.0,.09,.09],'rack_metal')
    display(b,[.83,.96,1.55],1.18)
    from roboclock.envs.sports.arcade.scenes import vertical_label
    vertical_label(b,'/World/Display/Title','B3 / BENCH PRESS ASSIST',b.display_center+[0,-.034,.087],[1.18,.046])
    b.view=([2.2,-2.8,1.9],[.55,0.,.9])
