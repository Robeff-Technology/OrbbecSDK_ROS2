#!/usr/bin/env python3
"""Live terminal readout of the AprilTag dock target's range and pose.

Run it next to `dock_detector` and move the camera around to watch the numbers
track. It redraws a fixed block in place, so the distance stays on one line rather
than scrolling past.

    ros2 run dock_target_detector dock_monitor

`--plain` prints one line per update instead, for logging or a dumb terminal.

It reads `/diagnostics` rather than `~/pose` because the detector publishes every
field there, including the ones that say whether to trust the reading:
`plane_rms`, `plane_inliers` and `source`.
"""
import argparse
import collections
import math
import sys
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node

DETECTOR = "dock_detector"
BAR_W = 34
ESC = "\033["
LINES = 8


def status_level(status):
    """DiagnosticStatus.level as an int.

    It is a `byte` field, so rclpy hands it back as `bytes` on some builds and as
    `int` on others; comparing the bytes form against 0 silently never matches.
    """
    lvl = status.level
    return lvl if isinstance(lvl, int) else int.from_bytes(lvl, "little")


class Monitor(Node):
    def __init__(self, args):
        super().__init__("dock_monitor")
        self.args = args
        self.values = None
        self.last_seen = 0.0
        self.seen_any = False
        self.recent = collections.deque(maxlen=120)  # (stamp, detected)
        self.drawn = False
        self.create_subscription(DiagnosticArray, "/diagnostics", self.on_diag, 50)
        self.create_timer(1.0 / max(args.rate, 1.0), self.draw)

    def on_diag(self, msg):
        for status in msg.status:
            if status.name != DETECTOR:
                continue
            now = time.time()
            ok = status_level(status) == 0
            self.recent.append((now, ok))
            if ok:
                self.values = {kv.key: kv.value for kv in status.values}
                self.last_seen = now
                self.seen_any = True

    def num(self, key, default=float("nan")):
        try:
            return float(self.values[key])
        except (TypeError, KeyError, ValueError):
            return default

    def hit_rate(self):
        if not self.recent:
            return float("nan")
        return 100.0 * sum(1 for _, ok in self.recent if ok) / len(self.recent)

    def hz(self):
        if len(self.recent) < 2:
            return float("nan")
        span = self.recent[-1][0] - self.recent[0][0]
        return (len(self.recent) - 1) / span if span > 1e-6 else float("nan")

    def bar(self, value, lo, hi):
        if math.isnan(value):
            return "?" * BAR_W
        frac = min(max((value - lo) / (hi - lo), 0.0), 1.0)
        n = int(round(frac * BAR_W))
        return "#" * n + "." * (BAR_W - n)

    def render(self):
        age = time.time() - self.last_seen
        live = self.seen_any and age < self.args.hold
        rng = self.num("range") if self.values else float("nan")

        if not self.seen_any:
            head = "  SEARCHING          no tag seen yet"
        elif live:
            head = f"  {self.values.get('payload', '?'):<18} {rng:8.3f} m"
        else:
            head = f"  LOST {age:5.1f} s ago   last {rng:8.3f} m"

        lines = [
            f"  dock_monitor        hits {self.hit_rate():5.1f} %   "
            f"{self.hz():5.1f} Hz",
            head,
            f"  [{self.bar(rng, self.args.min, self.args.max)}]  "
            f"{self.args.min:.1f}{'-' * 8}{self.args.max:.1f} m",
            "",
        ]
        if self.values:
            src = self.values.get("source", "?")
            lines += [
                f"  bearing {self.num('bearing_deg'):7.2f} deg     "
                f"lateral {self.num('lateral'):7.3f} m",
                f"  yaw     {self.num('yaw_deg'):7.2f} deg     "
                f"pitch   {self.num('pitch_deg'):7.2f} deg",
                f"  source  {src:<11}       pnp yaw {self.num('yaw_pnp_deg'):7.2f} deg",
                f"  plane   {self.num('plane_inliers'):7.0f} pts     "
                f"rms {self.num('plane_rms') * 1000.0:5.1f} mm",
            ]
        else:
            lines += ["", "  waiting for dock_detector on /diagnostics ...", "", ""]
        return lines[:LINES] + [""] * max(0, LINES - len(lines))

    def draw(self):
        lines = self.render()
        if self.args.plain:
            if self.values and time.time() - self.last_seen < self.args.hold:
                print(f"{self.values.get('payload','?')}  "
                      f"range {self.num('range'):.3f} m  "
                      f"bearing {self.num('bearing_deg'):+.2f}  "
                      f"yaw {self.num('yaw_deg'):+.2f}  "
                      f"[{self.values.get('source','?')}]", flush=True)
            elif self.seen_any:
                # A bare "no tag" makes a short gap look like a total loss; detection
                # is bursty near the resolution limit, so show how long it has been.
                print(f"no tag ({time.time() - self.last_seen:.1f} s since "
                      f"{self.num('range'):.3f} m)", flush=True)
            else:
                print("no tag (none seen yet)", flush=True)
            return
        out = []
        if self.drawn:
            out.append(f"{ESC}{LINES}A")
        for line in lines:
            out.append(f"{ESC}2K{line}\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self.drawn = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min", type=float, default=0.0, help="bar low end [m]")
    parser.add_argument("--max", type=float, default=4.0, help="bar high end [m]")
    parser.add_argument("--rate", type=float, default=10.0, help="redraws per second")
    parser.add_argument("--hold", type=float, default=1.0,
                        help="seconds a reading stays 'live' before it reads LOST")
    parser.add_argument("--plain", action="store_true",
                        help="one line per update instead of an in-place block")
    # Strip ROS args (--ros-args ...) before argparse sees them.
    argv = rclpy.utilities.remove_ros_args(sys.argv if argv is None else argv)[1:]
    args = parser.parse_args(argv)

    rclpy.init()
    node = Monitor(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.plain:
            sys.stdout.write("\n")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
