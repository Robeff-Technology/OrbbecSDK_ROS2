#!/usr/bin/env python3
"""End-to-end check of dock_detector against synthetic but realistic frames.

Renders the actual generated AprilTag target into a camera image at a known pose,
builds the matching registered-depth image, and runs the shipped detect ->
PnP -> plane-fit pipeline. Then sweeps distance to find where detection
actually stops, which is the number that decides the usable approach range.

Rendering uses zero distortion so the ground truth is exact; the undistortion
path in plane_from_depth is exercised but contributes nothing here.
"""
import importlib.util
import pathlib
import sys
import types

import cv2
import numpy as np

SRC = str(pathlib.Path(__file__).resolve().parent.parent
          / "dock_target_detector" / "dock_detector.py")
spec = importlib.util.spec_from_file_location("qdd", SRC)
qdd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qdd)

W, H = 1280, 720
K = np.array([[612.1275, 0.0, 640.5306], [0.0, 612.2515, 357.8101], [0.0, 0.0, 1.0]])
D = np.zeros((1, 5))
MARKER = 0.190          # black area, edge to edge
SHEET_W, SHEET_H = 0.210, 0.297
SS = 2                  # supersampling factor for rendering

det = types.SimpleNamespace(
    k=K, d=D, marker_size=MARKER, depth_scale=0.001, plane_roi_scale=1.2,
    min_plane_points=100, max_plane_points=4000, ransac_threshold=0.01,
    ransac_iterations=100, min_range=0.2, max_range=8.0, rescue_scale=4,
    rescue_pad=0.25, rescue_target_px=240.0, roi_track_scale=4.0, roi_miss_limit=5,
    _roi=None, _roi_misses=0,
    aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11), apriltag_ids=[],
    rng=np.random.default_rng(0))
det.aruco_params = cv2.aruco.DetectorParameters_create()
det.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
for name in ("detect_apriltag", "solve_pnp", "reprojection_error", "plane_from_depth"):
    setattr(det, name, types.MethodType(getattr(qdd.DockDetector, name), det))
angles = qdd.DockDetector.angles


def tag_texture(tag_id=7, px_per_cell=24, quiet=2, surround=255):
    """AprilTag 36h11 image plus quiet zone, and its black-border corners.

    Corners are returned TL, TR, BR, TL-last-BL, matching how objp maps into the
    image under board_pose(): that matrix is a 180 deg rotation about x (it is
    built from the face normal), so objp[0] = (-h, +h, 0) lands TOP-left. Get the
    order wrong and the tag renders MIRRORED, which AprilTag never matches
    because its code is not mirror invariant -- a silent, total detection
    failure, not a degraded one.
    """
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    cells = d.markerSize + 2                      # 6 data bits + 1 border each side
    core = cv2.aruco.drawMarker(d, tag_id, cells * px_per_cell)
    pad = int(round(quiet * px_per_cell))
    side = core.shape[0]
    tex = np.full((side + 2 * pad,) * 2, surround, np.uint8)
    tex[pad:pad + side, pad:pad + side] = core
    corners = np.float32([[pad, pad], [pad + side, pad],
                          [pad + side, pad + side], [pad, pad + side]])
    return tex, corners, cells


TEX, TEX_CORNERS, N_MODULES = tag_texture()


def board_pose(yaw_deg, pitch_deg=0.0):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    n = np.array([np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)])
    n /= np.linalg.norm(n)
    x = np.cross(np.array([0.0, -1.0, 0.0]), n)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(n, x), n))


def render(r, t, noise_px=2.0, depth_noise=0.005, bg_depth=4.0, rng=None, bg=110):
    """Return (bgr, depth_uint16_mm) for the target at pose (r, t)."""
    rng = rng or np.random.default_rng(0)
    h = MARKER / 2
    objp = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    img_pts, _ = cv2.projectPoints(objp, cv2.Rodrigues(r)[0], t, K * SS, D)
    dst = img_pts.reshape(4, 2).astype(np.float32)

    canvas = np.full((H * SS, W * SS), bg, np.uint8)
    m = cv2.getPerspectiveTransform(TEX_CORNERS, dst)
    warped = cv2.warpPerspective(TEX, m, (W * SS, H * SS), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    filled = cv2.warpPerspective(np.full_like(TEX, 255), m, (W * SS, H * SS),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
    canvas[filled > 0] = warped[filled > 0]
    gray = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    gray = np.clip(gray.astype(np.float32) + rng.normal(0, noise_px, gray.shape),
                   0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Depth: the rigid A4 backing is one plane; everything else is a far wall.
    normal, centre = r[:, 2], np.asarray(t, float)
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1],
                     np.ones_like(us, float)], axis=-1)
    denom = rays @ normal
    denom[np.abs(denom) < 1e-9] = 1e-9
    scale = (normal @ centre) / denom
    pts = rays * scale[..., None]
    local = pts - centre
    on_sheet = ((np.abs(local @ r[:, 0]) < SHEET_W / 2) &
                (np.abs(local @ r[:, 1]) < SHEET_H / 2) & (scale > 0))
    z = np.where(on_sheet, pts[..., 2], bg_depth)
    z = z + rng.normal(0, depth_noise, z.shape)
    return bgr, (np.clip(z, 0, 65.0) * 1000).astype(np.uint16)


def estimate(bgr, depth):
    found = det.detect_apriltag(bgr)
    if found is None:
        return None
    payload, quad = found
    pnp = det.solve_pnp(quad)
    plane = det.plane_from_depth(depth, quad)
    return payload, quad, pnp, plane


print(f"detector: cv2.aruco DICT_APRILTAG_36h11   "
      f"{N_MODULES}x{N_MODULES} cells, {MARKER * 1000 / N_MODULES:.2f} mm per cell\n")

print("Pose recovery at 1.5 m (true yaw varied)")
print(f"{'true yaw':>9} {'plane yaw':>10} {'pnp yaw':>8} {'range':>8} "
      f"{'true rng':>9} {'bearing':>8} {'inliers':>8} {'rms mm':>7}")
rng = np.random.default_rng(1)
ok = True
for yaw in (0.0, 8.0, -8.0, 20.0):
    r = board_pose(yaw)
    t = np.array([0.0, 0.0, 1.5])
    bgr, depth = render(r, t, rng=rng)
    out = estimate(bgr, depth)
    if out is None:
        print(f"{yaw:9.1f}   NOT DETECTED")
        ok = False
        continue
    payload, quad, pnp, plane = out
    normal, centroid, inliers, rms = plane
    b, py, _ = angles(normal, centroid)
    n_pnp = cv2.Rodrigues(pnp[0])[0] @ np.array([0.0, 0.0, 1.0])
    _, yy, _ = angles(n_pnp, pnp[1].ravel())
    print(f"{yaw:9.1f} {py:10.2f} {yy:8.2f} {np.linalg.norm(centroid):8.3f} "
          f"{1.5:9.3f} {b:8.2f} {inliers:8d} {rms * 1000:7.1f}")
    if payload != "TAG7":
        print(f"    payload mismatch: {payload!r}")
        ok = False
    if abs(py - yaw) > 1.5:
        print(f"    plane yaw off by {py - yaw:+.2f} deg")
        ok = False
    if abs(np.linalg.norm(centroid) - 1.5) > 0.02:
        print(f"    range off by {(np.linalg.norm(centroid) - 1.5) * 1000:+.0f} mm")
        ok = False

print("\nLateral offset and pitch at 2.0 m")
r = board_pose(0.0, 12.0)
t = np.array([0.45, 0.0, 2.0])
bgr, depth = render(r, t, rng=rng)
out = estimate(bgr, depth)
if out is None:
    print("  NOT DETECTED")
    ok = False
else:
    normal, centroid, inliers, rms = out[3]
    b, y, p = angles(normal, centroid)
    exp_b = np.degrees(np.arctan2(0.45, 2.0))
    print(f"  bearing {b:+.2f} (expect {exp_b:+.2f})   pitch {p:+.2f} (expect +12.00)"
          f"   lateral {centroid[0]:+.3f} (expect +0.450)")
    if abs(b - exp_b) > 1.0 or abs(p - 12.0) > 1.5 or abs(centroid[0] - 0.45) > 0.03:
        print("    FAIL: outside tolerance")
        ok = False

print("\nDecode range sweep at 1280x720 (fx 612), head-on")
print(f"{'dist m':>7} {'px/module':>10} {'decoded':>8} {'plane yaw':>10} "
      f"{'range err mm':>13}")
last_ok = 0.0
for dist in np.arange(1.0, 4.6, 0.25):
    r = board_pose(0.0)
    t = np.array([0.0, 0.0, float(dist)])
    bgr, depth = render(r, t, rng=np.random.default_rng(5))
    out = estimate(bgr, depth)
    ppm = (MARKER / N_MODULES) * K[0, 0] / dist
    if out is None:
        print(f"{dist:7.2f} {ppm:10.2f} {'no':>8}")
        continue
    normal, centroid, inliers, rms = out[3]
    _, y, _ = angles(normal, centroid)
    print(f"{dist:7.2f} {ppm:10.2f} {'YES':>8} {y:10.2f} "
          f"{(np.linalg.norm(centroid) - dist) * 1000:13.1f}")
    last_ok = dist
print(f"\n  last successful decode: {last_ok:.2f} m")

print("\nQuiet zone: a 190 mm tag on A4 leaves 10 mm of paper = 0.4 cells; AprilTag")
print("wants 1. Whatever the sheet is mounted ON supplies the rest.")
print(f"{'mounting':<46} " + " ".join(f"{d:>6.1f}m" for d in (1.0, 2.0, 2.5, 3.0)))
mountings = [
    ("spec 1-cell quiet zone (white backing)", 1.0, 255),
    ("A4 sheet on white backing", 0.42, 255),
    ("A4 sheet taped to a dark panel", 0.42, 40),
]
dark_hits = white_hits = 0
for label, quiet, surround in mountings:
    tex, corners, _ = tag_texture(quiet=quiet, surround=surround)
    TEX, TEX_CORNERS = tex, corners
    row = []
    for dist in (1.0, 2.0, 2.5, 3.0):
        # bg matters as much as the texture padding: what the sheet is mounted
        # ON supplies most of the quiet zone.
        hits = sum(
            estimate(*render(board_pose(0.0), np.array([0.0, 0.0, dist]),
                             rng=np.random.default_rng(s), bg=surround)) is not None
            for s in range(5))
        row.append(f"{hits}/5")
        if surround == 255 and quiet < 1.0:
            white_hits += hits
        if surround == 40:
            dark_hits += hits
    print(f"  {label:<44} " + " ".join(f"{c:>7}" for c in row))

# The sheet on white backing must not be materially worse than the spec zone;
# the dark-panel case is expected to be much worse and is the reason the
# generator prints a backing requirement.
if white_hits < 18:
    print(f"    FAIL: A4 on white backing only {white_hits}/20")
    ok = False
if dark_hits >= white_hits:
    print(f"    NOTE: dark panel {dark_hits}/20 vs white {white_hits}/20 -- "
          "expected dark to be clearly worse")

print("\nAll end-to-end checks passed." if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
