#!/usr/bin/env python3
"""Estimate the pose of a printed AprilTag dock target from colour + registered depth.

The tag id is used only as an identity ("which trailer / which dock point").
The geometry is measured two independent ways, and both are published so they can
be compared on the bench:

  * **PnP** on the four tag corners with a known physical marker size. Range from
    this is good, but yaw is weak: the apparent width of a planar square goes as
    cos(yaw), so the sensitivity vanishes near head-on and the two planar
    solutions flip against each other. `yaw_pnp_deg` and `pnp_ambiguity` are
    published so that spread stays visible rather than hidden behind a
    confident-looking number.
  * **Plane fit** to the registered depth pixels inside (and just around) the tag.
    Its range and bearing are excellent: measured live at 2.86 m, 1 mm and 0.04
    deg repeatability, and 13x more stable in range than PnP. Its yaw did NOT beat
    PnP on real hardware (0.86 against 0.83 deg sd), because the synthetic
    estimate assumed independent per-pixel depth noise while real stereo error is
    spatially correlated. See README.md.

Detection is cv2.aruco with DICT_APRILTAG_36h11. On a 180 mm sheet at 1280x720
that reads to 5 m, survives 8 px of motion blur, and costs about 4 ms full-frame:
respectively 1.7x, 4x and 15x better than the QR code this replaced, on the same
sheet. No zbar, no apriltag3, so the dependency set stays permissively licensed.

Depth must be registered to colour (`depth_registration:=true`,
`align_target_stream:=COLOR`) so a colour pixel indexes the depth image directly.
The node checks the two line up and says so once if they do not; it still
publishes the PnP-only pose in that case.

Angle conventions, all in the colour optical frame (x right, y down, z forward):

  bearing_deg  where the target is       -- positive = target is to the right
  yaw_deg      how the face is turned    -- positive = target's right edge is
                                            farther from the camera
  pitch_deg    how the face is tilted    -- positive = target's bottom edge is
                                            farther from the camera (elevation of
                                            the normal, independent of yaw)

A black tag cell absorbs the IR pattern, so depth comes back sparse over the tag
and dense over the white quiet zone around it. That is why `plane_roi_scale`
defaults above 1.0: it grows the depth sampling window out onto the white margin
of the sheet. Grow it too far and it starts eating background, which is what the
RANSAC pass is there to reject.
"""
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from builtin_interfaces.msg import Duration
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import message_filters

def fit_plane_ransac(pts, threshold, iterations, rng):
    """RANSAC a plane through Nx3 points, then refit on the inliers by SVD.

    Returns (normal, centroid, inlier_count, rms) or None. The normal is left
    unoriented; the caller flips it to face the camera.
    """
    n_pts = len(pts)
    best = None
    best_count = 0
    for _ in range(iterations):
        p = pts[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        inliers = np.abs((pts - p[0]) @ n) < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best = count, inliers
    if best is None or best_count < 3:
        return None

    q = pts[best]
    centroid = q.mean(axis=0)
    # Smallest singular vector of the centred inliers is the plane normal.
    _, _, vt = np.linalg.svd(q - centroid, full_matrices=False)
    normal = vt[-1]
    rms = float(np.sqrt(np.mean(((q - centroid) @ normal) ** 2)))
    return normal, centroid, best_count, rms


def quat_from_matrix(r):
    """Quaternion (x, y, z, w) from a 3x3 rotation matrix."""
    t = np.trace(r)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return tuple(q)


def face_frame(normal, up=np.array([0.0, -1.0, 0.0])):
    """Build a REP-103 style frame on the target face from its normal.

    x points out of the face back towards the camera, z is the in-plane
    direction closest to vertical, y completes the right-handed set.
    """
    x = normal / np.linalg.norm(normal)
    z = up - (up @ x) * x
    if np.linalg.norm(z) < 1e-6:  # face is looking straight up or down
        z = np.cross(x, np.array([1.0, 0.0, 0.0]))
    z /= np.linalg.norm(z)
    return np.column_stack((x, np.cross(z, x), z))


class DockDetector(Node):
    def __init__(self):
        super().__init__("dock_detector")

        p = self.declare_parameter
        # Side of the tag's black border -- NOT the sheet of paper. Measure the
        # printed tag edge to edge, excluding the white quiet zone.
        self.marker_size = float(p("marker_size", 0.19).value)
        self.apriltag_ids = list(p("apriltag_ids", []).value or [])
        self.depth_scale = float(p("depth_scale", 0.001).value)  # 16UC1 mm -> m
        # >1.0 grows the depth window onto the white margin, where the IR pattern
        # actually returns depth. See the module docstring.
        self.plane_roi_scale = float(p("plane_roi_scale", 1.2).value)
        self.min_plane_points = int(p("min_plane_points", 100).value)
        self.max_plane_points = int(p("max_plane_points", 4000).value)
        self.ransac_threshold = float(p("ransac_threshold", 0.01).value)
        self.ransac_iterations = int(p("ransac_iterations", 100).value)
        self.min_range = float(p("min_range", 0.2).value)
        self.max_range = float(p("max_range", 6.0).value)
        # Frames per second actually processed. A decode plus the crop-and-upscale
        # fallback costs ~60 ms in Python, so taking every frame at 30 Hz saturates a
        # core and builds a backlog that surfaces as lag. 0 disables the limit.
        self.process_rate = float(p("process_rate", 10.0).value)
        self._last_processed = 0.0
        # Newest frame stamp already handled. message_filters can pair a stale
        # message still sitting in its queue with a fresh one, which republishes a
        # frame that was already seen -- timestamps measured going backwards by up
        # to 600 ms, showing up downstream as repeated frames. Anything not
        # strictly newer is dropped.
        self._last_stamp_ns = 0
        self._stale_frames = 0
        self.publish_tf = p("publish_tf", True).value
        self.target_frame = p("target_frame", "dock_target").value
        self.publish_debug_image = p("publish_debug_image", True).value
        color_topic = p("color_topic", "/camera/color/image_raw").value
        info_topic = p("color_info_topic", "/camera/color/camera_info").value
        depth_topic = p("depth_topic", "/camera/depth/image_raw").value

        self.bridge = CvBridge()
        self.rng = np.random.default_rng(0)
        self.k = None
        self.d = None
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        # SUBPIX, emphatically not CORNER_REFINE_APRILTAG. Measured on a 38 px
        # marker: NONE 4.7 ms / 0.57 px corner error, SUBPIX 4.2 ms / 0.48 px,
        # APRILTAG 474 ms / 0.32 px. The AprilTag refiner buys 0.15 px for 100x
        # the cost, and the geometry comes from the depth plane anyway, so it
        # simply caps the whole node at ~6 fps for nothing.
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.warned_alignment = False
        self.pub_pose = self.create_publisher(PoseStamped, "~/pose", 10)
        self.pub_payload = self.create_publisher(String, "~/payload", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "~/debug/markers", 10)
        self.pub_debug_image = self.create_publisher(Image, "~/debug/image", 1)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.create_subscription(CameraInfo, info_topic, self.on_info, 10)
        color_sub = message_filters.Subscriber(
            self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        # queue_size 1: a deeper queue only stores stale frames to pair up late.
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=1, slop=0.05)
        self.sync.registerCallback(self.on_frames)

        self.get_logger().info(
            f"dock_detector up: AprilTag 36h11, "
            f"marker_size={self.marker_size:.3f} m")

    def on_info(self, msg):
        self.k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.d = np.array(msg.d, dtype=np.float64).reshape(1, -1)

    def detect_apriltag(self, bgr):
        """Largest AprilTag as (payload, 4x2 corners), or None.

        detectMarkers returns corners clockwise from the marker's own top-left
        and refines them subpixel, so nothing further is needed: it reads a 37 px
        marker at 5 m, full-frame, in about 4 ms.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None or len(ids) == 0:
            return None
        best = None
        for quad, marker_id in zip(corners, ids.ravel()):
            if self.apriltag_ids and int(marker_id) not in self.apriltag_ids:
                continue
            q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
            area = cv2.contourArea(q.astype(np.float32))
            if best is None or area > best[0]:
                best = (area, int(marker_id), q)
        if best is None:
            return None
        return f"TAG{best[1]}", best[2]

    def reprojection_error(self, objp, quad, rvec, tvec):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, self.k, self.d)
        return float(np.sqrt(np.mean(
            np.sum((projected.reshape(-1, 2) - quad) ** 2, axis=1))))

    def solve_pnp(self, quad):
        """PnP on the four corners, plus the mirrored planar twin.

        Uses SOLVEPNP_ITERATIVE rather than the obvious SOLVEPNP_IPPE_SQUARE:
        IPPE is broken in OpenCV 4.5.4 (the Humble system build) and comes back
        with rotations up to 180 deg out even on noiseless synthetic corners,
        while ITERATIVE recovers them exactly.

        The ambiguous twin is found by reflecting the marker normal about the
        line of sight and re-refining from there. Returns
        (rvec, tvec, ambiguity), ambiguity = err(best) / err(twin) in [0, 1];
        near 1.0 the two fits are indistinguishable and the PnP yaw means
        nothing, which is the normal state for a small marker seen near head-on.
        """
        h = self.marker_size / 2.0
        objp = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
        obj, img = objp.reshape(-1, 1, 3), quad.reshape(-1, 1, 2)

        ok, rvec, tvec = cv2.solvePnP(
            obj, img, self.k, self.d, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        err = self.reprojection_error(objp, quad, rvec, tvec)

        r = cv2.Rodrigues(rvec)[0]
        u = tvec.ravel() / np.linalg.norm(tvec)
        n2 = 2.0 * (r[:, 2] @ u) * u - r[:, 2]
        x2 = r[:, 0] - (r[:, 0] @ n2) * n2
        if np.linalg.norm(x2) < 1e-6:
            return rvec, tvec, 1.0
        x2 /= np.linalg.norm(x2)

        guess = cv2.Rodrigues(np.column_stack((x2, np.cross(n2, x2), n2)))[0]
        ok2, rvec2, tvec2 = cv2.solvePnP(
            obj, img, self.k, self.d, rvec=guess, tvec=tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2:
            return rvec, tvec, 1.0
        err2 = self.reprojection_error(objp, quad, rvec2, tvec2)
        if err2 < err:  # the twin fits better; keep it as the answer
            rvec, tvec, err, err2 = rvec2, tvec2, err2, err
        # Both fitting perfectly means they are indistinguishable, not confident.
        ambiguity = 1.0 if err2 < 1e-6 else float(min(err / err2, 1.0))
        return rvec, tvec, ambiguity

    def plane_from_depth(self, depth, quad):
        """Fit a plane to the depth pixels under the (scaled) quad.

        Returns (normal towards camera, centroid, inliers, rms) or None.
        """
        centre = quad.mean(axis=0)
        roi = (centre + (quad - centre) * self.plane_roi_scale).astype(np.int32)
        mask = np.zeros(depth.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, roi, 255)

        vs, us = np.nonzero(mask)
        z = depth[vs, us].astype(np.float64) * self.depth_scale
        keep = (z > self.min_range) & (z < self.max_range)
        vs, us, z = vs[keep], us[keep], z[keep]
        if len(z) < self.min_plane_points:
            return None
        if len(z) > self.max_plane_points:
            idx = self.rng.choice(len(z), self.max_plane_points, replace=False)
            vs, us, z = vs[idx], us[idx], z[idx]

        # Undistort before back-projecting: the lens k1 is small but it biases the
        # plane tilt systematically, which is exactly the quantity being measured.
        uv = np.stack([us, vs], axis=-1).astype(np.float64).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(uv, self.k, self.d).reshape(-1, 2)
        pts = np.column_stack((norm[:, 0] * z, norm[:, 1] * z, z))

        fit = fit_plane_ransac(pts, self.ransac_threshold, self.ransac_iterations, self.rng)
        if fit is None:
            return None
        normal, centroid, inliers, rms = fit
        if normal @ centroid > 0.0:  # camera sits at the origin; face it
            normal = -normal
        return normal, centroid, inliers, rms

    @staticmethod
    def angles(normal, centroid):
        """(bearing, yaw, pitch) in degrees -- see the module docstring.

        yaw is the azimuth of the face normal (its heading once projected into the
        horizontal plane) and pitch is its elevation out of that plane. Taking
        pitch as an elevation rather than the symmetric atan2(n_y, -n_z) keeps the
        two independent: a target mounted a little high or low then shifts pitch
        only, instead of dragging the yaw reading with it (atan2(n_y, -n_z) picks
        up a spurious 1/cos(yaw) factor, worth ~0.5 deg at 20 deg of yaw).
        """
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        return (
            math.degrees(math.atan2(centroid[0], centroid[2])),
            math.degrees(math.atan2(n[0], -n[2])),
            math.degrees(math.asin(float(np.clip(n[1], -1.0, 1.0)))),
        )

    def on_frames(self, color_msg, depth_msg):
        if self.k is None:
            return
        stamp_ns = (color_msg.header.stamp.sec * 1000000000
                    + color_msg.header.stamp.nanosec)
        if stamp_ns <= self._last_stamp_ns:
            self._stale_frames += 1
            self.get_logger().warn(
                f"dropped {self._stale_frames} stale/repeated frames from the "
                f"synchroniser", throttle_duration_sec=10.0)
            return
        self._last_stamp_ns = stamp_ns
        if self.process_rate > 0.0:
            now = time.monotonic()
            if now - self._last_processed < 1.0 / self.process_rate:
                return
            self._last_processed = now
        bgr = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        aligned = depth.shape[:2] == bgr.shape[:2]
        if not aligned and not self.warned_alignment:
            self.warned_alignment = True
            self.get_logger().error(
                f"depth {depth.shape[1]}x{depth.shape[0]} does not match colour "
                f"{bgr.shape[1]}x{bgr.shape[0]}: depth is not registered to colour, so "
                "only the PnP estimate will be published. Relaunch the driver with "
                "depth_registration:=true align_target_stream:=COLOR")

        found = self.detect_apriltag(bgr)
        if found is None:
            self.publish_diagnostics(color_msg.header, None)
            if self.publish_debug_image:
                self.publish_debug(color_msg.header, bgr, None)
            return
        payload, quad = found

        pnp = self.solve_pnp(quad)
        if pnp is None:
            return
        rvec, tvec, ambiguity = pnp
        r_pnp, _ = cv2.Rodrigues(rvec)
        # Marker frame z comes out of the board towards the camera under IPPE_SQUARE.
        n_pnp = r_pnp @ np.array([0.0, 0.0, 1.0])
        c_pnp = tvec.ravel()

        plane = self.plane_from_depth(depth, quad) if aligned else None
        if plane is not None:
            normal, centroid, inliers, rms = plane
            source = "depth"
        else:
            normal, centroid, inliers, rms = n_pnp, c_pnp, 0, float("nan")
            source = "pnp"

        bearing, yaw, pitch = self.angles(normal, centroid)
        _, yaw_pnp, _ = self.angles(n_pnp, c_pnp)
        result = {
            "payload": payload,
            "source": source,
            "range": float(np.linalg.norm(centroid)),
            "range_z": float(centroid[2]),
            "range_pnp": float(np.linalg.norm(c_pnp)),
            "lateral": float(centroid[0]),
            "bearing_deg": bearing,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "yaw_pnp_deg": yaw_pnp,
            "pnp_ambiguity": ambiguity,
            "plane_inliers": inliers,
            "plane_rms": rms,
        }

        self.publish_pose(color_msg.header, normal, centroid)
        self.pub_payload.publish(String(data=payload))
        self.publish_markers(color_msg.header, normal, centroid, quad, result)
        self.publish_diagnostics(color_msg.header, result)
        if self.publish_debug_image:
            self.publish_debug(color_msg.header, bgr, (quad, normal, centroid, result))

        self.get_logger().info(
            f"TAG \"{payload}\"  range {result['range']:.3f} m  "
            f"bearing {bearing:+.1f} deg  yaw {yaw:+.1f} deg [{source}]  "
            f"(pnp yaw {yaw_pnp:+.1f} deg, ambig {ambiguity:.2f})  "
            f"plane {inliers} pts rms {rms * 1000.0:.1f} mm",
            throttle_duration_sec=1.0)

    def publish_pose(self, header, normal, centroid):
        r = face_frame(normal)
        qx, qy, qz, qw = quat_from_matrix(r)

        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = centroid
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.pub_pose.publish(pose)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header = header
            tf.child_frame_id = self.target_frame
            tf.transform.translation.x, tf.transform.translation.y, \
                tf.transform.translation.z = centroid
            tf.transform.rotation = pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf)

    def quad_3d(self, quad, normal, centroid):
        """The four image corners intersected with the fitted plane.

        Gives the target's real outline in 3D, which is far easier to see and to
        sanity-check in RViz than a bare arrow.
        """
        norm = cv2.undistortPoints(
            quad.reshape(-1, 1, 2).astype(np.float64), self.k, self.d).reshape(-1, 2)
        rays = np.column_stack([norm, np.ones(len(norm))])
        denom = rays @ normal
        denom[np.abs(denom) < 1e-9] = 1e-9
        return rays * ((normal @ centroid) / denom)[:, None]

    def publish_markers(self, header, normal, centroid, quad, result):
        # Sized to be visible from the vehicle, not from the camera: with the
        # camera 2 m forward of base_link and the target ~3 m beyond that, the
        # marker sits ~5 m from the RViz origin, where a centimetre-scale arrow is
        # a few pixels. lifetime keeps a marker through the gaps between
        # detections without leaving a stale one on screen forever.
        life = Duration(sec=1, nanosec=0)

        outline = Marker()
        outline.header = header
        outline.ns = "dock"
        outline.id = 2
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.02
        outline.color.r, outline.color.g, outline.color.b, outline.color.a = \
            1.0, 0.85, 0.1, 1.0
        outline.lifetime = life
        try:
            corners = self.quad_3d(quad, normal, centroid)
            outline.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                              for p in np.vstack([corners, corners[:1]])]
        except Exception:  # a degenerate plane should not take the markers down
            outline.points = []

        arrow = Marker()
        arrow.header = header
        arrow.ns = "dock"
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.points = [
            Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
            for v in (centroid, centroid + normal * 0.6)
        ]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.03, 0.08, 0.0
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.1, 0.9, 0.2, 1.0
        arrow.lifetime = life

        text = Marker()
        text.header = header
        text.ns = "dock"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x, text.pose.position.y, text.pose.position.z = centroid
        text.pose.position.y -= 0.25   # optical y is down, so this sits above
        text.pose.orientation.w = 1.0
        text.scale.z = 0.12
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.lifetime = life
        text.text = (f"{result['payload']}  {result['range']:.2f} m\n"
                     f"yaw {result['yaw_deg']:+.1f}  bearing {result['bearing_deg']:+.1f}")

        self.pub_markers.publish(MarkerArray(markers=[outline, arrow, text]))

    def publish_diagnostics(self, header, result):
        status = DiagnosticStatus()
        status.name = "dock_detector"
        status.hardware_id = "orbbec_gemini_335l"
        if result is None:
            status.level = DiagnosticStatus.WARN
            status.message = "no tag detected"
        else:
            status.level = DiagnosticStatus.OK
            status.message = (f"{result['payload']} @ {result['range']:.2f} m, "
                              f"yaw {result['yaw_deg']:+.1f} deg")
            status.values = [KeyValue(key=k, value=f"{v}") for k, v in result.items()]

        msg = DiagnosticArray()
        msg.header = header
        msg.status = [status]
        self.pub_diag.publish(msg)

    def publish_debug(self, header, bgr, detection):
        img = bgr.copy()
        if detection is not None:
            quad, normal, centroid, result = detection
            cv2.polylines(img, [quad.astype(np.int32)], True, (0, 255, 0), 2)
            for i, pt in enumerate(quad.astype(np.int32)):
                cv2.circle(img, tuple(pt), 4, (0, 0, 255), -1)
                cv2.putText(img, str(i), tuple(pt + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 255), 1)
            # Project the measured normal so a wrong plane fit is visible at a glance.
            ends, _ = cv2.projectPoints(
                np.array([centroid, centroid + normal * 0.3]), np.zeros(3), np.zeros(3),
                self.k, self.d)
            a, b = ends.reshape(-1, 2).astype(np.int32)
            cv2.arrowedLine(img, tuple(a), tuple(b), (255, 128, 0), 2, tipLength=0.2)
            lines = [
                f"{result['payload']}  [{result['source']}]",
                f"range {result['range']:.3f} m   lateral {result['lateral']:+.3f} m",
                f"bearing {result['bearing_deg']:+.2f}   yaw {result['yaw_deg']:+.2f}"
                f"   pitch {result['pitch_deg']:+.2f}",
                f"pnp yaw {result['yaw_pnp_deg']:+.2f}  ambig {result['pnp_ambiguity']:.2f}",
                f"plane {result['plane_inliers']} pts  rms {result['plane_rms'] * 1000:.1f} mm",
            ]
        else:
            lines = ["no tag detected"]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, 30 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4)
            cv2.putText(img, line, (10, 30 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

        out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        out.header = header
        self.pub_debug_image.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DockDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
