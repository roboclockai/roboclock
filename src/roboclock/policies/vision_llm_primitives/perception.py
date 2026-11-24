"""What the agent reads from the images: watched colours, SAM3 masks, and openings in the depth.

Nothing here reads simulator state: only the head RGB and depth images and the head camera's
pose, which the agent derives from the arm bases and the rig spec (vision_llm.rig_camera).
"""

import base64
import io
import json
import math
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from roboclock.policies.vision_llm import HEAD_SIZE

PIXEL_MAX = 1000.0  # pixel targets are normalised to 0..PIXEL_MAX on the head image
SAM3_MIN_SCORE = 0.3
SAM3_NEAR_PX = 25  # a mask this close to the clicked pixel (640x480 head image) still counts as under it
RIM_STEP = 0.005  # m a surface must stand above the clicked point to be an opening's rim
RIM_SECTORS, RIM_SECTORS_NEEDED = 8, 6  # the rim must show in this many of the directions around the pixel
RIM_LEVEL = 0.012  # m the rim's height may differ between those directions
RIM_AHEAD = 2  # pixels further out that must not stand higher for a point to be the rim's top
RIM_RISE = 0.001  # m by which those may still be higher


def rgb_at(image, pixel, radius):
    """Mean colour of the patch of half-size `radius` (0..1000 image units) around `pixel`."""
    image = np.asarray(image)
    height, width = image.shape[:2]
    column = int(round(pixel["u"] / PIXEL_MAX * (width - 1)))
    row = int(round(pixel["v"] / PIXEL_MAX * (height - 1)))
    r = max(1, round(radius / PIXEL_MAX * width))
    patch = image[max(0, row - r) : row + r + 1, max(0, column - r) : column + r + 1, :3]
    return patch.reshape(-1, 3).astype(float).mean(axis=0)


def segment_request(url, image, prompts, timeout=60.0):
    """SAM3 text-prompted masks [(bool HxW mask, score)] for each prompt (scripts/xpolicylab/serve_sam3.py)."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(buffer, "PNG")
    body = {"image_png_b64": base64.b64encode(buffer.getvalue()).decode("ascii"), "prompts": prompts,
            "min_score": SAM3_MIN_SCORE, "max_instances": 10}
    request = urllib.request.Request(url.rstrip("/") + "/segment", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"SAM3 server unreachable at {url}: {error}") from error
    if payload.get("error"):
        raise RuntimeError(f"SAM3: {payload['error']}")
    out = []
    for prompt in prompts:
        for item in payload.get("results", {}).get(prompt, []):
            mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(item["mask_png_b64"])))) > 127
            out.append((mask, float(item["score"])))
    return out


def point_request(url, image, u, v, timeout=60.0):
    """SAM3 point-prompted (bool HxW mask, score) of whatever lies under the normalised pixel (u, v),
    the best of the server's masks, or None."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "PNG")
    point = [u / PIXEL_MAX * (width - 1), v / PIXEL_MAX * (height - 1)]
    body = {"image_png_b64": base64.b64encode(buffer.getvalue()).decode("ascii"), "points": [point]}
    request = urllib.request.Request(url.rstrip("/") + "/point", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"SAM3 point prompts: {error.read().decode(errors='replace')[:200]}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"SAM3 server unreachable at {url}: {error}") from error
    masks = payload.get("masks") or []
    if not masks:
        return None
    best = masks[0]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(best["mask_png_b64"])))) > 127, float(best["score"])


def pick_instance(masks, u, v, near=SAM3_NEAR_PX):
    """The mask under the normalised pixel (u, v), else the one whose pixels come nearest, within
    `near` pixels of a 640-wide image."""
    best, best_distance = None, math.inf
    for mask, score in masks:
        height, width = mask.shape
        row, column = round(v / PIXEL_MAX * (height - 1)), round(u / PIXEL_MAX * (width - 1))
        if mask[row, column]:
            distance = 0.0
        else:
            rows, columns = np.nonzero(mask)
            if not rows.size:
                continue
            distance = float(np.min(np.hypot(rows - row, columns - column))) * HEAD_SIZE[0] / width
        if distance < best_distance or (distance == best_distance and best is not None and score > best[1]):
            best, best_distance = (mask, score), distance
    return best if best_distance <= near else None


def mask_geometry(mask, depth, camera):
    """World extent of the surface a head-image mask covers: centre, top, bottom and footprint
    axes (from the depth image), or None when too little of it has depth."""
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape[:2]
    if mask.shape != (height, width):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)) > 127
    rows, columns = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if rows.size < 12:
        return None
    k = np.asarray(camera["intrinsics"], dtype=float).copy()
    k[0] *= width / camera["resolution"][0]
    k[1] *= height / camera["resolution"][1]
    d = depth[rows, columns]
    local = np.stack([(columns + 0.5 - k[0, 2]) * d / k[0, 0], (rows + 0.5 - k[1, 2]) * d / k[1, 1], d], axis=1)
    pose = np.asarray(camera["pose"], dtype=float)
    points = local @ pose[:3, :3].T + pose[:3, 3]
    # Mask edges can bleed onto the background: keep points near the bulk.
    centre = np.median(points, axis=0)
    spread = np.linalg.norm(points - centre, axis=1)
    points = points[spread <= max(0.03, 3.0 * np.median(spread))]
    xy = points[:, :2]
    middle = np.median(xy, axis=0)
    values, vectors = np.linalg.eigh(np.cov((xy - middle).T) + 1e-12 * np.eye(2))
    major, minor = vectors[:, 1], vectors[:, 0]

    def span(axis):
        return float(np.subtract(*np.percentile((xy - middle) @ axis, [95, 5])))

    return {
        "pixels": int(rows.size),
        "centre": [float(middle[0]), float(middle[1])],
        "top": float(np.percentile(points[:, 2], 95)),
        "bottom": float(np.percentile(points[:, 2], 5)),
        "length": span(major),
        "width": span(minor),
        "narrow_yaw": float(math.atan2(minor[1], minor[0])),
    }


def opening_under(observation, camera, ref, point, radius):
    """The opening a head-image pixel looks into: (rim height, where the pixel's ray crosses the
    rim's plane), or None when the pixel does not look into one.

    A pixel aimed at an opening seen from the side (a rack well, a tray slot, a hopper cell)
    lands on the inner wall or the floor, behind the spot it points at; releasing over that
    point puts the object on the far rim. The rim is found in the depth image alone: walking
    outward from the pixel in each of RIM_SECTORS image directions, the surface climbs the
    inner wall (or jumps, where the near rim hides the wall) and the rim is where it first
    stops climbing, RIM_STEP or more above the clicked point. It counts as an opening when
    RIM_SECTORS_NEEDED directions find that within `radius` at one height (within RIM_LEVEL),
    so a spot on a flat surface, or one beside a tall object, is left alone."""
    depth = np.asarray(observation["depth"], dtype=float)
    height, width = depth.shape[:2]
    k = np.asarray(camera["intrinsics"], dtype=float).copy()
    k[0] *= width / camera["resolution"][0]
    k[1] *= height / camera["resolution"][1]
    pose = np.asarray(camera["pose"], dtype=float)
    x, y = ref["u"] / PIXEL_MAX * width, ref["v"] / PIXEL_MAX * height
    reach = max(3, round(radius / PIXEL_MAX * width))
    steps = np.arange(1, reach + 1)
    rims = []
    for index in range(RIM_SECTORS):
        angle = 2 * math.pi * index / RIM_SECTORS
        columns = np.round(x + steps * math.cos(angle)).astype(int)
        rows = np.round(y + steps * math.sin(angle)).astype(int)
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        rows, columns = rows[inside], columns[inside]
        d = depth[rows, columns]
        seen = np.isfinite(d) & (d > 0)
        rows, columns, d = rows[seen], columns[seen], d[seen]
        local = np.stack([(columns - k[0, 2]) * d / k[0, 0], (rows - k[1, 2]) * d / k[1, 1], d], axis=-1)
        z = (local @ pose[:3, :3].T + pose[:3, 3])[:, 2]
        for n in range(len(z) - RIM_AHEAD):
            if z[n] > point[2] + RIM_STEP and np.all(z[n + 1 : n + 1 + RIM_AHEAD] <= z[n] + RIM_RISE):
                rims.append(float(z[n]))
                break
    if len(rims) < RIM_SECTORS_NEEDED:
        return None
    rim = float(np.median(rims))
    if sum(abs(value - rim) <= RIM_LEVEL for value in rims) < RIM_SECTORS_NEEDED:
        return None
    direction = pose[:3, :3] @ np.array([(x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], 1.0])
    if direction[2] > -1e-6:
        return None  # the ray does not come down onto the rim's plane
    return rim, pose[:3, 3] + direction * (rim - pose[2, 3]) / direction[2]

