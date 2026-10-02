#!/usr/bin/env python3
"""Reset the Orbbec camera over USB, instead of unplugging and replugging it.

The driver installs a SIGINT/SIGTERM handler and its destructor calls `clean()`
to stop the streams and close the device, but that takes a moment. If the process
is killed before it finishes -- `kill -9`, a `timeout`, or `ros2 launch`
escalating to SIGKILL after its shutdown grace period -- the device is left with
its streams claimed, and the next open fails until the USB port is cycled.

This issues the same `USBDEVFS_RESET` ioctl the kernel performs on a replug, so
it recovers the camera without physical access. It needs write access to the
device node, which the shipped udev rule (`99-obsensor-libusb.rules`) already
grants; otherwise run it with sudo.

    ros2 run dock_target_detector camera_reset

Prevention beats cure: stop the stack with Ctrl-C and let it finish, and give the
launch a generous `sigterm_timeout` so the driver is never SIGKILLed mid-cleanup.
If the driver is still alive and merely wedged, prefer its own service, which is
gentler than a bus reset:

    ros2 service call /camera/reboot_device std_srvs/srv/Empty
"""
import argparse
import fcntl
import glob
import os
import sys

# USBDEVFS_RESET is _IO('U', 20) -> (ord('U') << 8) | 20
USBDEVFS_RESET = 0x5514
ORBBEC_VENDOR = "2bc5"


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def find_devices(vendor):
    """Yield (bus, dev, node, description) for each matching USB device."""
    for entry in sorted(glob.glob("/sys/bus/usb/devices/*")):
        if read(os.path.join(entry, "idVendor")) != vendor:
            continue
        bus, dev = read(os.path.join(entry, "busnum")), read(os.path.join(entry, "devnum"))
        if not bus or not dev:
            continue
        node = f"/dev/bus/usb/{int(bus):03d}/{int(dev):03d}"
        product = read(os.path.join(entry, "product")) or "?"
        serial = read(os.path.join(entry, "serial")) or "?"
        yield int(bus), int(dev), node, f"{product} (serial {serial})"


def reset(node):
    fd = os.open(node, os.O_WRONLY)
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    finally:
        os.close(fd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", default=ORBBEC_VENDOR,
                    help="USB vendor id to match (default: %(default)s, Orbbec)")
    ap.add_argument("--list", action="store_true", help="list matches and exit")
    args = ap.parse_args(argv)

    devices = list(find_devices(args.vendor))
    if not devices:
        print(f"no USB device with vendor {args.vendor} found.", file=sys.stderr)
        print("The camera may already be gone from the bus; replug it.", file=sys.stderr)
        return 1

    for bus, dev, node, desc in devices:
        print(f"  bus {bus:03d} device {dev:03d}  {node}  {desc}")
    if args.list:
        return 0

    failed = 0
    for bus, dev, node, desc in devices:
        try:
            reset(node)
            print(f"reset {node}")
        except PermissionError:
            print(f"permission denied on {node}; run with sudo, or check that "
                  f"99-obsensor-libusb.rules is installed", file=sys.stderr)
            failed += 1
        except OSError as exc:
            print(f"reset failed on {node}: {exc}", file=sys.stderr)
            failed += 1

    if not failed:
        # Verified on a Gemini 335L: this re-initialises the device in place and it
        # keeps the same bus/device number, unlike a physical replug which gives it
        # a new one. Still give the kernel a moment to re-bind before reopening.
        print("\nDevice re-initialised in place (same bus/device number).")
        print("Wait a second or two, then start the stack again.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
