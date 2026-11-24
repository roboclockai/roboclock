"""Depth-based screen visibility diagnostics; no task state or clock mutations."""

import numpy as np


def compare_depth(depth, pixels, expected, tolerance=0.04):
    """Report sampled points in frame and not hidden by nearer scene geometry."""
    pixels = np.rint(pixels).astype(int)
    h, w = depth.shape[:2]
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
        & (expected > 0)
    )
    observed = np.full(len(pixels), np.nan)
    observed[inside] = depth[pixels[inside, 1], pixels[inside, 0]]
    valid = inside & np.isfinite(observed) & (observed > 0)
    occluded = valid & (observed < expected - tolerance)
    missing_surface = valid & (observed > expected + tolerance)
    clear = valid & ~occluded & ~missing_surface
    return {
        "samples": len(pixels),
        "in_frame": int(inside.sum()),
        "occluded": int(occluded.sum()),
        "missing_surface": int(missing_surface.sum()),
        "clear_fraction": float(clear.mean()),
    }


def display_visibility(backend):
    """Sample both timeline lanes and the instruction area at every recorded frame.

    A 4 cm depth tolerance includes notes and shallow screen trim. This tests
    geometric visibility, not text recognition or learned-policy performance.
    """
    camera = backend.camera
    depth = np.asarray(camera.get_depth()).squeeze()
    points = np.array(
        [
            backend.screen_point([x, 0.32, z])
            for x in np.linspace(0.365, 0.720, 36)
            for z in np.r_[np.linspace(1.108, 1.232, 14), np.linspace(1.06, 1.09, 4)]
        ]
    )
    # Use the renderer's matrices: the active RTX projection can differ from
    # the USD attribute-based Camera.get_intrinsics_matrix() at small apertures.
    params = backend.display_camera_params.get_data()
    view = np.asarray(params["cameraViewTransform"]).reshape(4, 4)
    projection = np.asarray(params["cameraProjection"]).reshape(4, 4)
    camera_points = np.c_[points, np.ones(len(points))] @ view
    clip = camera_points @ projection
    ndc = clip[:, :2] / clip[:, 3:4]
    h, w = depth.shape
    pixels = np.c_[(ndc[:, 0] + 1) * w / 2, (1 - ndc[:, 1]) * h / 2]
    expected = -camera_points[:, 2]
    result = compare_depth(depth, pixels, expected)
    result["pixel_bounds"] = [pixels.min(axis=0).tolist(), pixels.max(axis=0).tolist()]
    result["renderer_projection"] = projection.tolist()
    return result
