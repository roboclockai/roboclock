"""Geometric inspection proxy: source-surface rays, frustum and orientation.

This is not a visual defect detector or complete pixel-occlusion certificate.
Finite source-triangle-interior ROI rays can miss unsampled small occluders.
External occlusion is checked separately through the native physics scene.
"""
import numpy as np


def first_triangle_distance(origin,direction,triangles):
    """Two-sided vectorized Moller-Trumbore intersections, in metres."""
    e1=triangles[:,1]-triangles[:,0]
    e2=triangles[:,2]-triangles[:,0]
    cross=np.cross(direction,e2)
    det=np.einsum('ij,ij->i',e1,cross)
    inverse=np.zeros_like(det)
    valid=np.abs(det)>1e-12
    inverse[valid]=1/det[valid]
    offset=origin-triangles[:,0]
    u=np.einsum('ij,ij->i',offset,cross)*inverse
    q=np.cross(offset,e1)
    v=q@direction*inverse
    t=np.einsum('ij,ij->i',e2,q)*inverse
    valid&=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>1e-7)
    return float(t[valid].min()) if valid.any() else None


def visible_region(position,orientation,region,triangles,camera,quality,external_blocked):
    """Use measured object pose and canonical fixed-camera extrinsics."""
    eye=np.asarray(camera['eye_m'])
    samples=np.asarray(region['sample_points_local_m'])@orientation.T+position
    center=samples.mean(axis=0)
    distance=float(np.linalg.norm(eye-center))
    normal=orientation@np.asarray(region['normal_local'])
    normal_cos=float(normal@((eye-center)/max(distance,1e-12)))
    boundary=np.asarray(region.get('boundary_points_local_m',region['sample_points_local_m']))@orientation.T+position
    local=(boundary-eye)@np.asarray(camera['rotation_world'])
    depth=-local[:,2]
    intrinsics=np.asarray(camera['intrinsics'])
    pixels=np.c_[intrinsics[0,0]*local[:,0]/np.maximum(depth,1e-12)+intrinsics[0,2],
        -intrinsics[1,1]*local[:,1]/np.maximum(depth,1e-12)+intrinsics[1,2]]
    margin=quality['image_margin_px']
    width,height=camera['resolution']
    in_image=bool((depth>0).all() and (pixels[:,0]>margin).all() and (pixels[:,0]<width-margin).all()
        and (pixels[:,1]>margin).all() and (pixels[:,1]<height-margin).all())
    aligned=normal_cos>=np.cos(np.deg2rad(quality['normal_error_deg']))
    in_range=quality['distance_m'][0]<=distance<=quality['distance_m'][1]
    unoccluded=[]
    if in_image and aligned and in_range:
        origin_local=orientation.T@(eye-position)
        for point in samples:
            ray=point-eye
            length=float(np.linalg.norm(ray))
            direction=ray/length
            hit=first_triangle_distance(origin_local,orientation.T@direction,triangles)
            clear=hit is not None and abs(hit-length)<.0005 and not external_blocked(eye,direction,length)
            unoccluded.append(bool(clear))
    passed=bool(in_image and aligned and in_range and unoccluded and all(unoccluded))
    return dict(valid=passed,distance_m=distance,normal_cos=normal_cos,in_image=in_image,
        in_range=bool(in_range),normal_aligned=bool(aligned),sample_visibility=unoccluded,
        projected_points_px=pixels.tolist(),scope='Finite actual source surface ROIs; geometric/raycast proxy, not learned image inspection')
