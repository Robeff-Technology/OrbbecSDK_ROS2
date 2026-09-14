#!/usr/bin/env python3
"""Publish the camera frame, positioned from a sensor-kit style calibration file and
optionally oriented live from the camera's on-board IMU.

Frame chain:  parent_frame (base_link) -> child_frame (camera_link) -> optical frames
(the last hop is published by the Orbbec driver itself).

Position always comes from the calibration YAML: an IMU measures specific force and
angular rate, so it cannot observe position at all -- integrating acceleration twice
drifts without bound. Orientation is different: gravity gives an absolute reference for
roll and pitch, so those are estimated live. Yaw has no absolute reference on this
sensor (no magnetometer), so it is integrated from the gyro and WILL drift.
"""
import math
import os
import threading

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformBroadcaster, TransformListener


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotate(q, v):
    """Rotate vector v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


class CameraFramePublisher(Node):
    def __init__(self):
        super().__init__("camera_frame_publisher")

        self.parent_frame = self.declare_parameter("parent_frame", "base_link").value
        self.child_frame = self.declare_parameter("child_frame", "camera_link").value
        self.calibration_file = self.declare_parameter("calibration_file", "").value
        self.use_imu = self.declare_parameter("use_imu", True).value
        self.imu_topic = self.declare_parameter("imu_topic", "/camera/gyro_accel/sample").value
        self.imu_frame = self.declare_parameter(
            "imu_frame", "camera_accel_gyro_optical_frame").value
        self.publish_rate = float(self.declare_parameter("publish_rate", 50.0).value)
        # Complementary filter weight on the gyro. 0.98 keeps the fast gyro response while
        # letting the accelerometer correct drift over ~1 s.
        self.alpha = float(self.declare_parameter("complementary_alpha", 0.98).value)
        self.bias_samples = int(self.declare_parameter("gyro_bias_samples", 200).value)
        self.integrate_yaw = self.declare_parameter("integrate_yaw", False).value

        self.x, self.y, self.z, self.roll0, self.pitch0, self.yaw0 = self._load_calibration()
        self.roll, self.pitch, self.yaw = self.roll0, self.pitch0, self.yaw0

        self.lock = threading.Lock()
        self.have_imu = False
        self.last_stamp = None
        self.bias = [0.0, 0.0, 0.0]
        self.bias_acc = []
        self.q_cam_imu = None

        self.br = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.use_imu:
            self.create_subscription(
                Imu, self.imu_topic, self.on_imu, qos_profile_sensor_data)
            self.get_logger().info(
                f"use_imu=true: orientation from {self.imu_topic}; "
                f"position fixed at ({self.x:.3f}, {self.y:.3f}, {self.z:.3f}) from calibration.")
            if self.integrate_yaw:
                self.get_logger().warn(
                    "integrate_yaw=true: yaw comes from gyro integration and will drift "
                    "(no magnetometer on this sensor).")
        else:
            self.get_logger().info("use_imu=false: static frame from calibration file only.")

        self.create_timer(1.0 / max(1.0, self.publish_rate), self.publish_tf)

    def _load_calibration(self):
        path = self.calibration_file
        if not path or not os.path.isfile(path):
            self.get_logger().warn(
                f"calibration_file '{path}' not found; using zeros with z=0.5.")
            return 0.0, 0.0, 0.5, 0.0, 0.0, 0.0
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        entry = (data.get(self.parent_frame) or {}).get(self.child_frame)
        if entry is None:
            self.get_logger().warn(
                f"'{self.parent_frame}: {self.child_frame}' missing from {path}; using zeros.")
            return 0.0, 0.0, 0.5, 0.0, 0.0, 0.0
        g = lambda k: float(entry.get(k, 0.0))  # noqa: E731
        self.get_logger().info(
            f"calibration {self.parent_frame} -> {self.child_frame}: "
            f"xyz=({g('x'):.3f}, {g('y'):.3f}, {g('z'):.3f}) "
            f"rpy=({g('roll'):.3f}, {g('pitch'):.3f}, {g('yaw'):.3f})")
        return g("x"), g("y"), g("z"), g("roll"), g("pitch"), g("yaw")

    def _imu_to_camera_rotation(self):
        """Rotation taking vectors from the IMU's optical frame into child_frame."""
        if self.q_cam_imu is not None:
            return self.q_cam_imu
        try:
            tf = self.tf_buffer.lookup_transform(
                self.child_frame, self.imu_frame, rclpy.time.Time())
            q = tf.transform.rotation
            self.q_cam_imu = (q.x, q.y, q.z, q.w)
            self.get_logger().info(
                f"resolved {self.child_frame} <- {self.imu_frame} rotation from TF.")
        except Exception:
            return None
        return self.q_cam_imu

    def on_imu(self, msg):
        q = self._imu_to_camera_rotation()
        if q is None:
            return  # driver TF not up yet; skip until it is

        a = rotate(q, (msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z))
        g = rotate(q, (msg.angular_velocity.x,
                       msg.angular_velocity.y,
                       msg.angular_velocity.z))

        # Estimate and remove the gyro bias from the first samples, assumed at rest.
        if len(self.bias_acc) < self.bias_samples:
            self.bias_acc.append(g)
            if len(self.bias_acc) == self.bias_samples:
                n = float(len(self.bias_acc))
                self.bias = [sum(s[i] for s in self.bias_acc) / n for i in range(3)]
                self.get_logger().info(
                    "gyro bias (rad/s): "
                    f"[{self.bias[0]:.5f}, {self.bias[1]:.5f}, {self.bias[2]:.5f}]")
            return

        gx, gy, gz = (g[0] - self.bias[0], g[1] - self.bias[1], g[2] - self.bias[2])

        stamp = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        dt = 0.0 if self.last_stamp is None else stamp - self.last_stamp
        self.last_stamp = stamp
        if dt <= 0.0 or dt > 0.5:
            return

        norm = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        with self.lock:
            if norm > 1e-3:
                # Gravity-referenced roll/pitch. Valid whenever the sensor is not being
                # accelerated hard; the complementary filter keeps brief accelerations
                # from dragging the estimate.
                roll_a = math.atan2(a[1], a[2])
                pitch_a = math.atan2(-a[0], math.sqrt(a[1] ** 2 + a[2] ** 2))
                self.roll = self.alpha * (self.roll + gx * dt) + (1.0 - self.alpha) * roll_a
                self.pitch = self.alpha * (self.pitch + gy * dt) + (1.0 - self.alpha) * pitch_a
            else:
                self.roll += gx * dt
                self.pitch += gy * dt
            if self.integrate_yaw:
                self.yaw += gz * dt
            self.have_imu = True

    def publish_tf(self):
        with self.lock:
            roll, pitch, yaw = self.roll, self.pitch, self.yaw
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z
        qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = CameraFramePublisher()
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
