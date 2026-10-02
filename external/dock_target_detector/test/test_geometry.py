#!/usr/bin/env python3
"""Check dock_detector's geometry against synthetic ground truth.

Verifies the sign conventions (bearing/yaw/pitch), the RANSAC plane fit and the
PnP path, and measures the yaw noise of each estimator so the accuracy claim
behind the two-estimator design is an observed number, not an assertion.

Exercises the shipped functions directly (solve_pnp is called unbound against a
stand-in holding just the attributes it reads), so it tests the real code.
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

# Real intrinsics read off the Gemini 335L colour stream at 1280x720.
K = np.array([[612.1275, 0.0, 640.5306], [0.0, 612.2515, 357.8101], [0.0, 0.0, 1.0]])
D = np.array([[-0.033578, 0.036342, -0.000206, 0.000489, -0.012423]])
MARKER = 0.19

# Stand-in exposing only what solve_pnp/reprojection_error read off self.
det = types.SimpleNamespace(k=K, d=D, marker_size=MARKER)
det.reprojection_error = types.MethodType(qdd.DockDetector.reprojection_error, det)
solve_pnp = types.MethodType(qdd.DockDetector.solve_pnp, det)
angles = qdd.DockDetector.angles


def board_pose(yaw_deg, pitch_deg=0.0):
    """Rotation whose columns are the board's right / up / normal-towards-camera.

    Matches the marker frame solve_pnp assumes: x right, y up, z out of the face.
    """
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    n = np.array([np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)])
    n /= np.linalg.norm(n)
    x = np.cross(np.array([0.0, -1.0, 0.0]), n)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(n, x), n))


def synth_plane(centroid, r, half=0.12, n=2000, noise=0.005, rng=None):
    """Noisy planar patch of the board, in camera coordinates.

    Columns 0 and 1 of r span the face; column 2 is its normal.
    """
    rng = rng or np.random.default_rng(0)
    uv = rng.uniform(-half, half, size=(n, 2))
    pts = centroid + uv[:, :1] * r[:, 0] + uv[:, 1:] * r[:, 1]
    # Depth noise acts along the ray, which is near enough the optical axis here.
    return pts + np.column_stack([np.zeros(n), np.zeros(n), rng.normal(0, noise, n)])


def project(r, t):
    h = MARKER / 2
    objp = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    img, _ = cv2.projectPoints(objp, cv2.Rodrigues(r)[0], t, K, D)
    return img.reshape(4, 2)


def fit(pts, seed=2):
    normal, centroid, inliers, rms = qdd.fit_plane_ransac(
        pts, 0.01, 100, np.random.default_rng(seed))
    if normal @ centroid > 0:
        normal = -normal
    return normal, centroid, inliers, rms


ok = True


def run(label, fn):
    global ok
    try:
        fn()
        print(f"  PASS  {label}")
    except AssertionError as e:
        print(f"  FAIL  {label}: {e}")
        ok = False


print("Plane-fit sign conventions (noiseless)")
for yaw, pitch in [(0, 0), (15, 0), (-15, 0), (0, 10), (0, -10), (20, -8)]:
    def check(yaw=yaw, pitch=pitch):
        pts = synth_plane(np.array([0.0, 0.0, 2.0]), board_pose(yaw, pitch),
                          noise=0.0, rng=np.random.default_rng(1))
        normal, centroid, _, _ = fit(pts)
        _, y, p = angles(normal, centroid)
        assert abs(y - yaw) < 0.5, f"yaw {y:+.2f} expected {yaw:+.2f}"
        assert abs(p - pitch) < 0.5, f"pitch {p:+.2f} expected {pitch:+.2f}"
    run(f"yaw={yaw:+3d} pitch={pitch:+3d}", check)

print("\nBearing sign")
def bearing_check():
    pts = synth_plane(np.array([0.5, 0.0, 2.0]), board_pose(0),
                      noise=0.0, rng=np.random.default_rng(1))
    normal, centroid, _, _ = fit(pts)
    b, _, _ = angles(normal, centroid)
    expect = np.degrees(np.arctan2(0.5, 2.0))
    assert abs(b - expect) < 0.5, f"bearing {b:+.2f} expected {expect:+.2f}"
run("target 0.5 m to the right at 2 m reads positive", bearing_check)

print("\nsolve_pnp round trip (noiseless corners)")
def pnp_check():
    for yaw in (0, 10, 25, -18):
        r = board_pose(yaw)
        t = np.array([0.0, 0.0, 2.0])
        rvec, tvec, ambiguity = solve_pnp(project(r, t))
        n = cv2.Rodrigues(rvec)[0] @ np.array([0.0, 0.0, 1.0])
        _, y, _ = angles(n, tvec.ravel())
        assert abs(y - yaw) < 0.5, f"yaw {y:+.2f} expected {yaw:+.2f}"
        err = abs(np.linalg.norm(tvec) - 2.0)
        assert err < 0.005, f"range off by {err * 1000:.1f} mm at yaw {yaw}"
        assert 0.0 <= ambiguity <= 1.0, f"ambiguity {ambiguity} out of range"
run("yaw and range recovered at 0/10/25/-18 deg", pnp_check)

print("\nsolve_pnp reports high ambiguity head-on, low when clearly tilted")
def ambiguity_check():
    rng = np.random.default_rng(3)
    got = {}
    for yaw in (0.0, 35.0):
        vals = []
        q0 = project(board_pose(yaw), np.array([0.0, 0.0, 2.0]))
        for _ in range(50):
            vals.append(solve_pnp(q0 + rng.normal(0, 0.3, (4, 2)))[2])
        got[yaw] = float(np.median(vals))
    assert got[0.0] > 0.8, f"head-on ambiguity only {got[0.0]:.2f}, expected ~1"
    assert got[35.0] < got[0.0], (
        f"tilted ambiguity {got[35.0]:.2f} not below head-on {got[0.0]:.2f}")
    print(f"        median ambiguity: 0 deg -> {got[0.0]:.2f}, "
          f"35 deg -> {got[35.0]:.2f}")
run("ambiguity discriminates the degenerate case", ambiguity_check)

print("\nface_frame / quaternion")
def quat_check():
    for yaw in (0, 20, -35):
        n = board_pose(yaw)[:, 2]
        r = qdd.face_frame(n)
        assert abs(np.linalg.det(r) - 1.0) < 1e-9, f"det {np.linalg.det(r)}"
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-9), "not orthonormal"
        q = np.array(qdd.quat_from_matrix(r))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9, f"|q| = {np.linalg.norm(q)}"
        assert np.allclose(r[:, 0], n, atol=1e-9), "x axis is not the normal"
run("right-handed, orthonormal, unit quaternion, x = normal", quat_check)

print("\n--- Yaw noise: depth plane fit vs PnP corners, 2 m, A4 marker ---")
print("    (5 mm depth noise per pixel, 0.3 px corner noise)")
rng = np.random.default_rng(42)
for true_yaw in (0.0, 10.0):
    r = board_pose(true_yaw)
    t = np.array([0.0, 0.0, 2.0])
    q0 = project(r, t)
    plane_yaw, pnp_yaw = [], []
    for _ in range(200):
        normal, centroid, _, _ = fit(synth_plane(t, r, noise=0.005, rng=rng), seed=None)
        plane_yaw.append(angles(normal, centroid)[1])
        rvec, tvec, _ = solve_pnp(q0 + rng.normal(0, 0.3, (4, 2)))
        n = cv2.Rodrigues(rvec)[0] @ np.array([0.0, 0.0, 1.0])
        pnp_yaw.append(angles(n, tvec.ravel())[1])
    pl, pn = np.array(plane_yaw), np.array(pnp_yaw)
    print(f"  true {true_yaw:+5.1f} deg | plane: mean {pl.mean():+6.2f} sd {pl.std():5.2f} "
          f"range [{pl.min():+6.2f},{pl.max():+6.2f}]")
    print(f"                 | pnp:   mean {pn.mean():+6.2f} sd {pn.std():5.2f} "
          f"range [{pn.min():+6.2f},{pn.max():+6.2f}]")

print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
