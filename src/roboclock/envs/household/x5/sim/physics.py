"""Physics overlay authored on top of a static R2 initial scene.

``overlay(initial_scene, out_dir)`` writes ``simulation_scene.usda``: a new layer whose only
sublayer is the untouched initial scene. In it we deactivate the visual-only robot, rebuild both
X5 arms as PhysX articulations, add a physics scene and contact material, and let each task add
its own colliders and rigid bodies. Static source assets and the initial scene stay unchanged.

Robot model (inherited from the completed V1.0_X5 A1 episode, see docs/household/x5-simulation.md):
six revolute joints from the pinned arx5-sdk URDF with SDK position/effort bounds and PhysX
position drives; the native link6 CAD split into palm + two fingers with an approximate symmetric
prismatic jaw (about 60.3 mm native opening, 30.14 mm travel per side). Within-arm self collision
is disabled; arm-environment and arm-arm contacts stay active.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import BASES, H

URDF = L.X5/'arx5-sdk/models/X5.urdf'
VISUAL = L.X5/'visual'
ARM_LOWER = [-3.14, -.05, -.1, -1.6, -1.57, -2.]
ARM_UPPER = [2.618, 3.5, 3.2, 1.55, 1.57, 2.]
ARM_EFFORT = [30., 40., 30., 15., 10., 10.]
JAW_TRAVEL = .03014        # per side, native CAD
JAW_OPENING = 2*JAW_TRAVEL  # ~60.3 mm between distal pads when open
# Jaw force per side. Not a published X5 gripper figure: chosen so a parallel grasp can carry the
# arm's 1.5-2 kg rated payload by friction (>= ~12 N/side at mu 0.8). The first A1 episode used 5 N.
JAW_FORCE = 20.
TCP_OFFSET = .145          # tool point along link6 x
ARMS = '/World/Arms'


def quat(r):
    q = r.as_quat()
    return Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3])))


def pose(prim, xyz, r):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
    xf.AddOrientOp().Set(quat(r))


def shift(prim, offset, pivot):
    """Move a prim rigidly in the world by a layout offset (dx, dy, dyaw deg): turn dyaw about the vertical through
    pivot (x, y), then translate (dx, dy) (task_base.place on its points). One transform op is prepended to the
    prim's own ops (its scale, orient and the ops a task looks up at run time are kept). Per-seed placement in
    Oracle.setup(), before the world resets; no-op for a zero offset."""
    dx, dy, yaw = map(float, offset)
    if not (dx or dy or yaw):
        return
    m = (Gf.Matrix4d().SetTranslate(Gf.Vec3d(-pivot[0], -pivot[1], 0.))
         * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0., 0., 1.), yaw))
         * Gf.Matrix4d().SetTranslate(Gf.Vec3d(pivot[0]+dx, pivot[1]+dy, 0.)))  # row vectors: applied left to right
    parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) \
        if prim.GetParent().IsA(UsdGeom.Xformable) else Gf.Matrix4d(1.)
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    op = xf.AddTransformOp(opSuffix='layout')
    op.Set(parent*m*parent.GetInverse())
    xf.SetXformOpOrder([op, *ops], xf.GetResetXformStack())


def collision(prim, mat=None, approximation='convexHull'):
    UsdPhysics.CollisionAPI.Apply(prim)
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr(.0005)
    api.CreateRestOffsetAttr(0.)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        mesh.CreateExtentAttr([points.min(0).tolist(), points.max(0).tolist()])
        mesh.CreateDoubleSidedAttr(True)
    if mat:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose='physics')


def xform_parents(stage, path):
    """Define missing ancestors as Xforms (typeless parents cannot carry rigid-body APIs)."""
    for parent in reversed(Sdf.Path(path).GetParentPath().GetPrefixes()):
        if not stage.GetPrimAtPath(parent).IsValid():
            UsdGeom.Xform.Define(stage, parent)


def box(stage, path, pos, size, mat, angle=0., visible=False):
    """Axis-aligned (optionally z-rotated) box collider, invisible by default."""
    xform_parents(stage, path)
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    xf.AddRotateZOp().Set(float(angle))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if not visible:
        c.CreateVisibilityAttr('invisible')
    collision(c.GetPrim(), mat)
    return c.GetPrim()


def body(prim, mass, kinematic=False, com=None):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if kinematic:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    api.CreateSolverPositionIterationCountAttr(32)
    api.CreateSolverVelocityIterationCountAttr(8)
    api.CreateMaxDepenetrationVelocityAttr(.5)


def contact_material(stage, name='ContactMaterial', static=1., dynamic=.9):
    mat = UsdShade.Material.Define(stage, '/World/PhysicsMaterials/'+name)
    p = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    p.CreateStaticFrictionAttr(static)
    p.CreateDynamicFrictionAttr(dynamic)
    p.CreateRestitutionAttr(0.)
    return mat


def physics_scene(stage):
    scene = UsdPhysics.Scene.Define(stage, '/World/Physics')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.807)
    api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    api.CreateEnableCCDAttr(True)
    api.CreateEnableGPUDynamicsAttr(False)
    api.CreateBroadphaseTypeAttr('MBP')
    return scene


def vessel(stage, name, center, floor_z, rim_z, radii, mat, segments=48, wall=.005):
    """Open round cavity: floor box plus a ring of wall boxes per radius band (fixed)."""
    root = '/World/Fixtures/'+name
    box(stage, root+'/Floor', [*center, floor_z-.002], [radii[0]*1.9, radii[0]*1.9, .004], mat)
    layers = len(radii)-1
    for k in range(layers):
        lo = floor_z+(rim_z-floor_z)*k/layers
        hi = floor_z+(rim_z-floor_z)*(k+1)/layers
        r = (radii[k]+radii[k+1])/2+wall/2
        for i in range(segments):
            a = 2*np.pi*i/segments
            box(stage, f'{root}/Wall{k}_{i}', [center[0]+r*np.cos(a), center[1]+r*np.sin(a), (lo+hi)/2],
                [wall, 2*r*np.tan(np.pi/segments)+.0005, hi-lo], mat, math.degrees(a))
    return root


def _robot(stage, side, mat, steel):
    path = f'{ARMS}/{side}'
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(False)
    urdf = ET.parse(URDF).getroot()
    base = np.eye(4)
    base[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
    base[:3, 3] = [*BASES[side], H+.012]
    frames = {'base_link': base}
    # Links are authored at q = 0; the episode sets the ready pose once at reset.
    for j in urdf.findall('joint'):
        frames[j.find('child').get('link')] = frames[j.find('parent').get('link')]@L._origin(j.find('origin'))
    for link in urdf.findall('link'):
        name = link.get('name')
        if name == 'eef_link':
            continue
        p = UsdGeom.Xform.Define(stage, f'{path}/{name}').GetPrim()
        f = frames[name]
        pose(p, f[:3, 3], R.from_matrix(f[:3, :3]))
        # two derived 30 g fingers are deducted from the palm mass
        body(p, float(link.find('inertial/mass').get('value'))-(.06 if name == 'link6' else 0.))
        mass = UsdPhysics.MassAPI(p)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*np.fromstring(link.find('inertial/origin').get('xyz'), sep=' ')))
        inertia = link.find('inertial/inertia')
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(inertia.get('i'+a+a)) for a in 'xyz']))
        visual = stage.DefinePrim(f'{path}/{name}/Visual')
        visual.GetReferences().AddReference(str(VISUAL/(('link6_palm' if name == 'link6' else name)+'.usdc')))
        mesh = stage.GetPrimAtPath(f'{path}/{name}/Visual/Mesh')
        collision(mesh, mat)
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(steel)
    anchor = UsdPhysics.FixedJoint.Define(stage, path+'/fixed_base')
    anchor.CreateBody1Rel().SetTargets([Sdf.Path(path+'/base_link')])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*base[:3, 3]))
    anchor.CreateLocalRot0Attr(quat(R.from_matrix(base[:3, :3])))
    revolute = [j for j in urdf.findall('joint') if j.get('type') == 'revolute']
    for i, j in enumerate(revolute):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/{j.get('name')}")
        joint.CreateBody0Rel().SetTargets([f"{path}/{j.find('parent').get('link')}"])
        joint.CreateBody1Rel().SetTargets([f"{path}/{j.find('child').get('link')}"])
        f = L._origin(j.find('origin'))
        joint.CreateLocalPos0Attr(Gf.Vec3f(*f[:3, 3]))
        joint.CreateLocalRot0Attr(quat(R.from_matrix(f[:3, :3])))
        axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
        joint.CreateAxisAttr('XYZ'[int(np.argmax(abs(axis)))])
        joint.CreateLowerLimitAttr(math.degrees(ARM_LOWER[i]))
        joint.CreateUpperLimitAttr(math.degrees(ARM_UPPER[i]))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
        drive.CreateStiffnessAttr(3000.)
        drive.CreateDampingAttr(90.)
        drive.CreateMaxForceAttr(ARM_EFFORT[i])
    for name, sign in [('jawP', 1), ('jawN', -1)]:
        p = UsdGeom.Xform.Define(stage, f'{path}/{name}').GetPrim()
        q = frames['link6']@np.array([0, .044*sign, 0, 1])
        pose(p, q[:3], R.from_matrix(frames['link6'][:3, :3]))
        body(p, .03)
        v = stage.DefinePrim(f'{path}/{name}/Visual')
        v.GetReferences().AddReference(str(VISUAL/f'link6_{name}.usdc'))
        # One hull around the bent finger would create a fictitious wedge at the tip:
        # keep a proximal hull plus the measured native flat distal pad.
        vis = UsdGeom.Mesh(stage.GetPrimAtPath(f'{path}/{name}/Visual/Mesh'))
        UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(steel)
        points = np.asarray(vis.GetPointsAttr().Get())
        faces = np.asarray(vis.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        keep = points[:, 0] < .131
        mapping = np.full(len(points), -1)
        mapping[np.flatnonzero(keep)] = np.arange(keep.sum())
        selected = faces[keep[faces].all(axis=1)]
        hull = UsdGeom.Mesh.Define(stage, f'{path}/{name}/ProximalCollision')
        hull.CreatePointsAttr(points[keep].tolist())
        hull.CreateFaceVertexCountsAttr([3]*len(selected))
        hull.CreateFaceVertexIndicesAttr(mapping[selected].reshape(-1).tolist())
        hull.CreateVisibilityAttr('invisible')
        collision(hull.GetPrim(), mat)
        pad = box(stage, f'{path}/{name}/PadCollision', [.13907, -.013107*sign, 0.], [.015, .0015, .010], mat)
        patch = PhysxSchema.PhysxCollisionAPI(pad)
        patch.CreateTorsionalPatchRadiusAttr(.004)
        patch.CreateMinTorsionalPatchRadiusAttr(.002)
        joint = UsdPhysics.PrismaticJoint.Define(stage, f'{path}/{name}_joint')
        joint.CreateBody0Rel().SetTargets([path+'/link6'])
        joint.CreateBody1Rel().SetTargets([f'{path}/{name}'])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, .044*sign, 0))
        joint.CreateAxisAttr('Y')
        joint.CreateLowerLimitAttr(-JAW_TRAVEL if sign == 1 else 0.)
        joint.CreateUpperLimitAttr(0. if sign == 1 else JAW_TRAVEL)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateStiffnessAttr(800.)
        drive.CreateDampingAttr(8.)
        drive.CreateMaxForceAttr(JAW_FORCE)
    return path


def overlay(initial_scene, out_dir, build_task):
    """Write out_dir/simulation_scene.usda = initial scene + shared physics + build_task(ctx) additions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir/'simulation_scene.usda'
    if dest.exists():
        dest.unlink()
    stage = Usd.Stage.CreateNew(str(dest))
    stage.GetRootLayer().subLayerPaths.append(str(Path(initial_scene).resolve()))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    physics_scene(stage)
    mat = contact_material(stage)
    stage.GetPrimAtPath('/World/Robots').SetActive(False)  # visual-only arms of the static scene
    steel = L.material(stage, 'X5RuntimeSteel', [.70, .73, .76], .29, .7)
    arms = {side: _robot(stage, side, mat, steel) for side in BASES}
    # Island counter (the static scene's island mesh is appearance only)
    box(stage, '/World/Fixtures/Island', [.30, 0., H-.025], [1.1496, .7613, .05], mat)
    info = build_task(stage, mat)
    stage.GetRootLayer().Save()
    setup = dict(
        robot='Two X5 six-DOF chains from the pinned arx5-sdk URDF; PhysX position drives (stiffness 3000, '
              'damping 90) bounded by SDK effort/position limits',
        gripper='Native link6 CAD split into palm + two fingers; approximate symmetric prismatic jaws, '
                f'{JAW_OPENING*1000:.1f} mm native opening, {JAW_TRAVEL*1000:.2f} mm travel/side, {JAW_FORCE:.0f} N/side '
                '(assumption consistent with the rated payload). '
                'Not a calibrated hardware linkage.',
        finger_collision='Proximal hull plus measured 15 x 1.5 x 10 mm distal pad; 4 mm torsional patch',
        self_collisions='Disabled within each arm; inter-arm and environment collisions active',
        mass_inertia='URDF masses and diagonal inertia; cross terms omitted',
        arms=arms, initial_scene=str(Path(initial_scene).resolve()), no_runtime_pose_writes=True, task=info)
    return dest, setup
