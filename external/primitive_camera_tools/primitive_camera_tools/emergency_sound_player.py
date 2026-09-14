#!/usr/bin/env python3
"""Play a sound while the primitive emergency detector reports points in the field.

Playback is *resumed*, not restarted: when the field clears, the position reached in the
track is remembered, and the next detection continues from there. When the track runs to
its end it wraps back to the start and keeps going for as long as the field stays
occupied.

Resuming requires seeking into the file, which paplay/aplay cannot do, so ffplay is used
whenever resume is enabled (it takes -ss <offset>). Set resume:=false to go back to
restart-from-zero behaviour with any player.

A watchdog stops playback if the emergency topic goes silent (detector or camera died),
so a stuck 'true' can never leave the sound running forever.
"""
import os
import shutil
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class EmergencySoundPlayer(Node):
    def __init__(self):
        super().__init__("emergency_sound_player")

        self.sound_file = self.declare_parameter("sound_file", "").value
        self.topic = self.declare_parameter(
            "topic", "/perception/primitive_emergency_detector/emergency").value
        self.player = self.declare_parameter("player", "").value
        self.resume = bool(self.declare_parameter("resume", True).value)
        self.timeout = float(self.declare_parameter("message_timeout", 1.0).value)
        # Audio does not start the instant the process is spawned. Subtracting this from
        # the measured wall time keeps the remembered position from creeping forward
        # every time the sound is stopped and resumed.
        self.start_latency = float(self.declare_parameter("player_start_latency", 0.12).value)

        self.duration = self._probe_duration() if self.resume else 0.0
        if self.resume and self.duration <= 0.0:
            self.get_logger().warn(
                "Could not determine track duration; falling back to restart-from-zero.")
            self.resume = False

        if not self.player:
            self.player = "ffplay" if self.resume else self._detect_player()
        if self.resume and self.player != "ffplay":
            if shutil.which("ffplay"):
                self.get_logger().info(
                    f"resume=true needs seeking, which '{self.player}' cannot do; using ffplay.")
                self.player = "ffplay"
            else:
                self.get_logger().warn(
                    "resume=true needs ffplay, which is not installed; "
                    "falling back to restart-from-zero.")
                self.resume = False
        if not self.player or not shutil.which(self.player):
            self.get_logger().error(
                f"Audio player '{self.player}' not available. Sound disabled.")
            self.player = ""
        if not os.path.isfile(self.sound_file):
            self.get_logger().error(f"Sound file not found: '{self.sound_file}'. Sound disabled.")

        self._proc = None
        self._proc_lock = threading.Lock()
        self._active = threading.Event()
        self._shutdown = threading.Event()
        self._last_msg_time = None
        # Where in the track to pick up from, in seconds. Guarded by _proc_lock together
        # with the offset/start time of the process currently playing, so the position can
        # be settled synchronously the moment playback is stopped.
        self._position = 0.0
        self._play_offset = 0.0
        self._play_started = None

        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

        self.create_subscription(Bool, self.topic, self.on_emergency, 10)
        if self.timeout > 0.0:
            self.create_timer(0.2, self._watchdog)

        mode = (f"resumes from where it stopped (track {self.duration:.1f}s)"
                if self.resume else "restarts from the beginning")
        self.get_logger().info(
            f"Watching {self.topic}; player='{self.player}' file='{self.sound_file}' ({mode})")

    def _probe_duration(self):
        if not os.path.isfile(self.sound_file) or not shutil.which("ffprobe"):
            return 0.0
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", self.sound_file],
                capture_output=True, text=True, timeout=10)
            return float(out.stdout.strip())
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _detect_player():
        for candidate in ("paplay", "aplay", "ffplay"):
            if shutil.which(candidate):
                return candidate
        return ""

    def _command(self, offset):
        if self.player == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if offset > 0.0:
                cmd += ["-ss", f"{offset:.3f}"]
            return cmd + [self.sound_file]
        return [self.player, self.sound_file]

    def _playback_loop(self):
        while not self._shutdown.is_set():
            if not self._active.wait(timeout=0.1):
                continue
            if not self.player or not os.path.isfile(self.sound_file):
                self._active.clear()
                continue

            offset = self._position if self.resume else 0.0
            try:
                proc = subprocess.Popen(
                    self._command(offset),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Failed to start playback: {exc}")
                self._active.clear()
                continue

            with self._proc_lock:
                self._proc = proc
                self._play_offset = offset
                self._play_started = time.monotonic()

            # Poll so a stop request is acted on within ~20 ms instead of waiting out
            # the clip. A stop is handled by _stop_now(), which settles the position
            # itself; this loop only has to deal with the clip ending naturally.
            while proc.poll() is None:
                if not self._active.is_set() or self._shutdown.is_set():
                    break
                time.sleep(0.02)

            with self._proc_lock:
                ran_to_end = proc.poll() is not None and self._play_started is not None
                if ran_to_end and self.resume:
                    # Reached the end: wrap round and keep going from the top.
                    self._position = 0.0
                if self._proc is proc:
                    self._proc = None
                    self._play_started = None

    @staticmethod
    def _kill(proc):
        try:
            proc.kill()          # SIGKILL: stop now, do not drain the audio buffer
            proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass

    def _stop_now(self):
        """Stop playback and settle the resume position before returning, so a caller
        that immediately restarts picks up from the right place."""
        self._active.clear()
        with self._proc_lock:
            proc = self._proc
            offset = self._play_offset
            started = self._play_started
            self._play_started = None
        if proc is not None and proc.poll() is None:
            self._kill(proc)
            if self.resume and started is not None:
                played = max(0.0, time.monotonic() - started - self.start_latency)
                with self._proc_lock:
                    self._position = (offset + played) % self.duration

    def on_emergency(self, msg):
        self._last_msg_time = self.get_clock().now()
        if msg.data and not self._active.is_set():
            where = f" from {self._position:.1f}s" if self.resume else ""
            self.get_logger().info(f"Object inside the primitive field: sound on{where}.")
            self._active.set()
        elif not msg.data and self._active.is_set():
            self._stop_now()
            where = f" at {self._position:.1f}s" if self.resume else ""
            self.get_logger().info(f"Field clear: sound off{where}.")

    def _watchdog(self):
        if not self._active.is_set() or self._last_msg_time is None:
            return
        age = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if age > self.timeout:
            self.get_logger().warn(f"No emergency message for {age:.1f}s; stopping sound.")
            self._stop_now()

    def destroy_node(self):
        self._shutdown.set()
        self._stop_now()
        super().destroy_node()


def main():
    rclpy.init()
    node = EmergencySoundPlayer()
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
