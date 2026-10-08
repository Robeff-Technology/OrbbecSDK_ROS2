# dock_target_detector

QR dock target detection and pose estimation for reversing onto a trailer.

The QR carries **identity** ("which trailer / which dock point"). The **geometry** —
how far away the target is, which way it is off to the side, and how its face is
turned — is measured from the depth image. Those are deliberately two different
jobs, for the reason below.

## Marker choice: AprilTag by default

`marker_type` selects `apriltag` (cv2.aruco `DICT_APRILTAG_36h11`, the default) or
`qr` (zbar). Measured on the **same 180 mm sheet** at 1280x720, 5 renders each:

| | QR (21x21, 8.6 mm/module) | AprilTag 36h11 (8x8, 22.6 mm/cell) |
| --- | --- | --- |
| 3.0 m | 5/5 | 5/5 |
| 3.5 m | **0/5** | 5/5 |
| 5.0 m | 0/5 | **5/5** |
| motion blur tolerated | 2 px | **8 px** |
| full-frame detect cost | ~50 ms | **3.4 ms** |

A QR spends its area on an arbitrary byte string; 36h11 spends 36 bits on one of
587 ids with a large Hamming distance, so its cells are 2.6x bigger on the same
paper. A dock target only needs "which dock", so the payload buys nothing and
costs range, blur tolerance and CPU.

Three consequences worth noting:

- **0-3 m becomes comfortable.** At 3 m a QR on A4 is 1.75 px/module, already under
  zbar's ~2 threshold; the same sheet as an AprilTag is 4.6 px/cell.
- **Vibration stops mattering as much.** 8 px of smear against 2 px is roughly 4x
  the angular rate at the same exposure.
- **The speed work becomes optional.** At 3.4 ms a full-frame scan every frame at
  30 Hz costs about a tenth of a core, so `roi_track_scale`, `rescue_scale` and
  `process_rate` are tuning knobs rather than necessities.

The QR path is kept for when an arbitrary string really is needed.

## Why the yaw does not come from the marker

A single planar square is degenerate for yaw near head-on. Apparent width scales
with `cos(yaw)`, and `d(cos θ)/dθ → 0` at `θ = 0`, so the measurement carries
almost no information about small angles. At 2 m an A4 marker is about 58 px
across; 0.3 px of corner noise is a 0.5 % width error, which is several degrees of
yaw. Measured on synthetic corners with that noise level:

| true yaw | PnP corners | depth plane fit |
| --- | --- | --- |
| 0° | mean −0.05°, **sd 5.05°**, range [−9.6°, +11.7°] | mean +0.01°, **sd 0.18°** |
| +10° | mean +9.20°, **sd 4.80°**, range [−12.5°, +16.2°] | mean +10.04°, **sd 0.18°** |

PnP does not merely get noisy — at +10° of true yaw it reports the wrong *sign*
routinely. Yaw is exactly the quantity reverse docking needs, so the node takes it
from a plane fitted to the thousands of depth pixels on the target's face instead.

**These are synthetic numbers and the yaw advantage did not survive contact with
real hardware** — see [Measured on real hardware](#measured-on-real-hardware)
below, where the two came out equal. The synthetic model assumed independent
per-pixel depth noise; real depth error is spatially correlated. Range and bearing
did hold up.

PnP is still computed and published (`yaw_pnp_deg`, `pnp_ambiguity`) so the two can
be compared on the bench rather than taken on trust. Its range is good; only its
orientation is weak.

## Measured on real hardware

Gemini 335L, 1280x720 colour, SW D2C registration, A4 target (printed at 95 %, so
180.5 mm / 8.6 mm modules) taped flat to a white wall at ~2.86 m, off-axis by 8.3
degrees. 283 detections:

| field | mean | sd | min | max |
| --- | --- | --- | --- | --- |
| `range` (depth) | 2.862 m | **0.001 m** | 2.858 | 2.865 |
| `range_pnp` | 2.899 m | 0.013 m | 2.851 | 2.942 |
| `bearing_deg` | −8.316° | **0.039°** | −8.413 | −8.176 |
| `lateral` | −0.402 m | 0.002 m | −0.406 | −0.395 |
| `yaw_deg` (depth) | −1.996° | **0.855°** | −4.515 | +0.342 |
| `yaw_pnp_deg` | −3.865° | 0.834° | −5.820 | −1.249 |
| `plane_inliers` | 1829 | 45 | 1702 | 1931 |
| `plane_rms` | 5 mm | — | 4 mm | 5 mm |

**Range and bearing are excellent**: 1 mm and 0.04° repeatability, and the depth
range is 13x more stable than PnP's (1 mm vs 13 mm sd). That part of the design
holds up.

**The yaw claim does not.** The synthetic figures above predicted ~0.18° for the
plane fit against ~5° for PnP. On real hardware the two are *indistinguishable*
(0.855° vs 0.834°), and they disagree on the mean by 1.9°, so at least one carries
a bias. Two reasons, both real:

- The synthetic model used **independent** per-pixel depth noise, which averages
  down as 1/sqrt(N). Real stereo depth error is **spatially correlated**, so the
  effective sample count is far below 1829 and the averaging buys much less.
- PnP was flattered here by the target sitting 8.3° off-axis, which partly breaks
  the head-on degeneracy that makes PnP yaw bad in the first place.

So treat 0.855° as the number to plan against, not 0.18°. It is still usable for
docking, but the plane fit has not yet been shown to beat PnP on yaw. The
untested case is the one that matters: a **large, near head-on** target at
**short range**, where PnP should degrade and the plane fit should not. Measure
that before committing to either.

## How it works

1. Decode the QR in the colour image (zbar). The payload is the target's identity.
2. Subpixel-refine the four corners. Three sit on a finder pattern and refine well;
   the fourth is in the data area, so corners that move too far are rejected.
3. `solvePnP` on those corners with the known `marker_size` → range, plus the
   mirrored planar twin, whose relative reprojection error becomes
   `pnp_ambiguity` (near 1.0 = the two fits are indistinguishable, yaw meaningless).
4. Sample the registered depth image under the quad, back-project to 3D
   (undistorting first), and RANSAC a plane → face normal, centroid, fit RMS.
5. Publish range / bearing / yaw / pitch, a `PoseStamped`, a TF frame, RViz markers
   and an annotated debug image.

## Angle conventions

All in the colour optical frame (x right, y down, z forward):

| Field | Meaning | Sign |
| --- | --- | --- |
| `bearing_deg` | where the target is | + = target is to the right |
| `yaw_deg` | how the face is turned | + = target's right edge is farther away |
| `pitch_deg` | how the face is tilted | + = target's bottom edge is farther away |

`yaw` is the azimuth of the face normal and `pitch` its elevation, so the two are
independent: a target mounted a little high or low moves `pitch` only, instead of
dragging `yaw` with it.

## Setup

The Humble system OpenCV 4.5.4 is built **without QUIRC**, so `cv2.QRCodeDetector`
decodes nothing at all — it just prints `Library QUIRC is not linked` once per
frame. The zbar wrapper is therefore required, not optional:

```bash
pip3 install --user pyzbar        # decoder; libzbar0 itself ships with the distro
pip3 install --user segno         # only for make_dock_target
```

## Printing a target

```bash
ros2 run dock_target_detector make_dock_target --payload DOCK01 --size-mm 190
```

Writes a print-ready A4 sheet and reports the `marker_size` to pass to the node.
The `--out` extension picks the format: **`.pdf` (default) or `.svg`**. Prefer PDF
for printing — viewers offer a dependable "Actual size", where a browser printing
SVG applies its own page setup. Ready-made sheets for `DOCK01` are in
[targets/](targets/) in both formats.

The generated PDF was rasterised at 300 dpi and measured to confirm it: page A4,
QR 189.99 mm, calibration bar 99.99 mm, decodes as `DOCK01`.
Print at **100 % / actual size** — any fit-to-page scaling silently invalidates
`marker_size`. The sheet carries a 100 mm bar to check with a ruler.

**If the bar does not measure 100 mm**, the printer scaled the page (shrinking to
the printable area is a common default). This matters less than it sounds. Measured
against a target rendered at a true 190 mm:

| `marker_size` | range (depth) | yaw (depth) | bearing | `range_pnp` | `yaw_pnp` |
| --- | --- | --- | --- | --- | --- |
| 0.1900 (correct) | 1.9992 | 8.23 | 0.00 | 2.0124 | 7.98 |
| 0.1805 (5 % low) | 1.9992 | 8.23 | 0.00 | 1.9118 | 7.98 |
| 0.2100 (10 % high) | 1.9992 | 8.23 | 0.00 | 2.2242 | 7.98 |

Everything the docking logic uses is unchanged; only `range_pnp` moves, and exactly
proportionally. Even the PnP *orientation* is scale-invariant. `marker_size` is used
in one place only, inside `solve_pnp`. It matters if depth registration is broken and
the node falls back to PnP-only; otherwise it is a diagnostic.

So you do not need to reprint: the bar, the symbol and the margins all scaled
together, so one measurement recovers the true size. Feed it back in:

```bash
ros2 run dock_target_detector make_dock_target --measured-bar-mm 95
#   print scale  95.0 %
#   marker_size  0.1805 m   <-- use this, no reprint needed
```

To reprint at true size instead, disable "fit to page" / "shrink to printable
area" and set scaling to 100 %.

Any QR decodes, so the payload content is irrelevant to detection. Three
properties of the sheet are not:

### Quiet zone: barely matters with AprilTag

The spec likes 1 blank cell around the tag, and a 190 mm tag on A4 leaves 0.4.
That turns out not to matter. Measured synthetically at 2 m, 5 renders each:

| quiet zone | white bg | grey bg | dark bg |
| --- | --- | --- | --- |
| 1.0 cell | 5/5 | 5/5 | 5/5 |
| 0.4 cell (the A4 sheet) | 5/5 | 5/5 | 5/5 |
| **none at all** | **5/5** | **5/5** | **5/5** |

AprilTag's detector finds the black border by contour rather than scanning lines
across it, so it does not depend on the surround. **This is a change from the QR
version of this package**, where a sheet taped to a dark panel failed 0/5 at every
distance and white backing was mandatory. Mount the tag on anything flat.

### Cell size sets the range

`px_per_cell = cell_mm * fx / distance`, and AprilTag needs about 1.5. 36h11 is
8x8 cells, so a 190 mm sheet gives 23.75 mm cells and reads a long way off.
There is no payload to choose: the tag id is the identity.

### Mount it flat

On rigid backing. A rippled sheet corrupts the plane fit, which is where the yaw
comes from.

Simulated decode range at 1280x720 (`fx` 612) for that sheet: reliable to **2.75 m**,
intermittent to 3.75 m, gone past 4 m. At 1920x1080 (`fx` 918) expect about +50 %.
Real-world will be shorter — glare, motion blur and exposure all cost range that a
synthetic test does not model.

## Running

```bash
ros2 launch dock_target_detector dock_detector.launch.xml marker_size:=0.1805
```

That is the whole thing in one command: camera, detector and the live readout.

| argument | default | meaning |
| --- | --- | --- |
| `marker_size` | `0.19` | side of the printed black area [m] |
| `launch_camera` | `true` | set false to attach to a camera already running |
| `launch_monitor` | `true` | live readout, in plain mode (see below) |
| `monitor_rate` | `4.0` | monitor updates per second |
| `color_width` / `color_height` | `1280` / `720` | colour resolution; more pixels = more range |
| `align_mode` | `SW` | leave it; `HW` is refused at these resolutions (see below) |
| `point_cloud_decimation` | `8` | matches the emergency detector; cloud only |
| `depth_crop_left` | `150` | matches the emergency detector |
| `process_rate` | `10.0` | frames per second the detector processes |
| `launch_rviz` | `true` | open RViz with [config/dock_detector.rviz](config/dock_detector.rviz) |
| `publish_camera_tf` | `true` | static `base_link` -> `camera_link`; false if a URDF already does |
| `base_frame` | `base_link` | frame the camera is mounted on |
| `camera_x/y/z`, `camera_roll/pitch/yaw` | from `camera_mount_calibration.yaml` | camera mount pose [m, rad] |

### RViz

Marker sizes are chosen for the **vehicle's** viewpoint, not the camera's. With the
camera 2 m forward of `base_link` and the target ~3 m beyond it, the markers sit
~5 m from the RViz origin, where a centimetre-scale arrow is a few pixels. The
displays are:

- a **yellow outline** of the target's real 3D corners, obtained by intersecting
  the corner rays with the fitted plane (validated to 2 % of the true 190 mm side,
  with every corner on the plane)
- a **green arrow** 0.6 m along the face normal
- a **text label** with payload, range, yaw and bearing

Markers carry a 1 s lifetime, so they ride through the gaps between detections
without leaving a stale pose on screen once the target is really gone.


[config/dock_detector.rviz](config/dock_detector.rviz) is the emergency detector's config with
the QR displays swapped in: the depth cloud, `~/debug/markers`, `~/pose` as axes
and `~/debug/image`.

Its Fixed Frame is `base_link`, and the driver only publishes
`camera_link` -> `*_optical_frame`. **Something must supply `base_link` ->
`camera_link` or RViz displays nothing at all** — no cloud, no markers, no TF.
The launch publishes that hop as a static transform (`publish_camera_tf`), using the
mount from `orbbec_camera/config/camera_mount_calibration.yaml`. It is fixed: there
is no live IMU roll/pitch correction. If you see an empty RViz, check that hop first:

```bash
ros2 run tf2_ros tf2_echo base_link camera_link
```

`ros2 launch` prefixes every line with `[node-N]`, which breaks the monitor's
in-place block redraw, so the launch file runs it in plain mode (one line per
update). For the full block view, launch with `launch_monitor:=false` and run the
monitor in its own terminal.

If a camera is **already running**, you do not have to restart it to get depth
registration — the driver can switch at runtime:

```bash
ros2 service call /camera/set_image_registration_mode \
  orbbec_camera_msgs/srv/SetString "{data: 'SW_D2C'}"     # 'OFF' to revert
ros2 launch dock_target_detector dock_detector.launch.xml launch_camera:=false
```

The launch brings the camera up with **depth registered to colour**
(`depth_registration:=true`, `align_target_stream:=COLOR`), which the estimator
requires: it indexes the depth image with colour pixel coordinates. If the two
streams do not line up the node says so once and falls back to publishing the
PnP-only estimate.

Against an already-running camera:

```bash
ros2 run dock_target_detector dock_detector --ros-args \
  --params-file $(ros2 pkg prefix dock_target_detector)/share/dock_target_detector/config/params.yaml
```

## Watching it live

`dock_monitor` redraws a fixed block in place, so the distance stays on one
line while you move the camera instead of scrolling past. Run it in its own
terminal next to the detector:

```bash
ros2 run dock_target_detector dock_monitor
```
```
  dock_monitor        hits  40.0 %    15.9 Hz
  DOCK01                2.856 m
  [########################..........]  0.0--------4.0 m

  bearing   -8.30 deg     lateral  -0.400 m
  yaw       -0.83 deg     pitch     -3.12 deg
  source  depth             pnp yaw   -5.20 deg   [rescued]
  plane      2110 pts     rms   4.0 mm
```

`--min` / `--max` set the bar range (default 0-4 m), `--rate` the redraw rate, and
`--plain` prints one line per update instead, for logging or over SSH:

```bash
ros2 run dock_target_detector dock_monitor --plain --max 3
```

What to watch besides the distance:

- **`hits`** — percentage of recent frames that decoded. Below ~30 % you are near
  the resolution cliff; move closer or print bigger.
- **`[rescued]`** — the symbol was too small for zbar at native resolution and only
  decoded through the crop-and-upscale fallback. Expect this at long range.
- **`source`** — `depth` means the range came from the plane fit; `pnp` means depth
  was unusable and it fell back to the corners, which is 13x noisier in range.
- **`rms`** — scatter about the fitted plane. If it climbs, the fit is eating
  background or the target is not flat, and the yaw should not be trusted.

For a visual check, the debug image has the same numbers drawn on the video:

```bash
ros2 topic echo /dock_detector/pose
ros2 run rqt_image_view rqt_image_view /dock_detector/debug/image
```

## Topics

### Subscribed

- `<color_topic>` (`sensor_msgs/Image`, `rgb8`)
- `<color_info_topic>` (`sensor_msgs/CameraInfo`)
- `<depth_topic>` (`sensor_msgs/Image`, `16UC1`, **registered to colour**)

Colour and depth are paired with an approximate time synchroniser (50 ms slop).

### Published

- `~/pose` (`geometry_msgs/PoseStamped`) — target face in the colour optical
  frame; x out of the face towards the camera, z the in-plane direction nearest
  vertical
- `~/detections` (`rbf_tag_msgs/TagDetectionArray`) — the same pose with the tag
  id, for rbf_docking; stamped at acquisition, empty when no tag is in view
- `~/payload` (`std_msgs/String`) — decoded QR contents
- `~/debug/image` (`sensor_msgs/Image`) — annotated colour frame
- `~/debug/markers` (`visualization_msgs/MarkerArray`) — normal arrow + readout
- `/diagnostics` (`diagnostic_msgs/DiagnosticArray`) — every field below
- TF `<colour optical frame>` → `<target_frame>`

### Services

- `~/enable` (`std_srvs/SetBool`) — start / stop detecting. While disabled the
  colour and depth streams are not subscribed at all. `start_enabled` sets the
  state at launch.

### Docking integration

On the vehicle the detector runs idle (`start_enabled:=false`) and rbf_docking
drives it: when docking starts it calls `/dock_detector/enable` with `true`, reads
the tag from `/dock_detector/detections`, and disables the detector again once
docking ends (docked, failed or cancelled). The wiring is in
`ozismak_vehicle_launch/launch/vehicle_interface.launch.xml`.

The tag frame in `~/detections` is the face frame above: **x out of the face
towards the vehicle, y to the camera's right, z up**. rbf_docking's
`tag_to_coupling` must be given in that frame. Only the largest tag in view is
reported; restrict `apriltag_ids` if several can be seen at once.

Diagnostic keys: `payload`, `source` (`depth` or `pnp`), `range`, `range_z`,
`range_pnp`, `lateral`, `bearing_deg`, `yaw_deg`, `pitch_deg`, `yaw_pnp_deg`,
`pnp_ambiguity`, `plane_inliers`, `plane_rms`.

`plane_rms` is the quality number to watch: it is the scatter of the depth points
about the fitted plane. If it climbs, the fit is picking up background or the
target is not flat, and the yaw should not be trusted.

## Parameters

See [config/params.yaml](config/params.yaml), which documents each one inline.

## Performance

Measured with AprilTag, one clean stack, target at ~2.9 m:

| | |
| --- | --- |
| detection (`detectMarkers`, SUBPIX) | ~4 ms |
| end-to-end throughput | ~10 frames/s |
| hit rate | 100 % |

10 Hz is comfortable for docking: a vehicle reversing at 0.5 m/s travels ~5 cm
between updates. `process_rate` caps it; raising it above ~10 did not increase
measured throughput, and the remaining bottleneck has **not** been isolated --
detection itself is only 4 ms of the budget. Worth revisiting if you ever need
faster, but do it with nothing else running (see below).

### Corner refinement: never use CORNER_REFINE_APRILTAG here

`cv2.aruco` offers four corner refinements. Measured on a 38 px marker:

| refinement | time | corner error |
| --- | --- | --- |
| NONE | 4.7 ms | 0.573 px |
| **SUBPIX** (what we use) | **4.2 ms** | **0.475 px** |
| CONTOUR | 3.9 ms | 0.722 px |
| APRILTAG | **474 ms** | 0.322 px |

`CORNER_REFINE_APRILTAG` buys 0.15 px for **100x the cost**, and the geometry
comes from the depth plane fit rather than the corners, so it is worth nothing
here. Setting it capped the whole node at ~6 fps.

### The point cloud, and why it is decimated

`point_cloud_decimation: 8` matches the emergency detector. Undecimated at
1280x720 the cloud is ~740k points per message (11.8 MB, ~355 MB/s at 30 Hz),
which on its own makes RViz and the driver lag. It costs the detector nothing:
the detector reads `/camera/depth/image_raw`, not the cloud.

`align_mode` must stay `SW`. The 335L rejects HW D2C for 1280x720 colour with
1280x800 depth (`Current stream profile is not support hardware d2c`) and the
pipeline never starts.

### Watch for leftover stacks

A launch that does not shut down cleanly leaves its nodes running, and they are
easy to miss. Several stacks at once means several camera containers competing
for the device and several transform publishers fighting over the same
`base_link -> camera_link` transform, which makes everything jitter in RViz and
quietly corrupts any measurement. Check before trusting a number:

```bash
ros2 node list          # any duplicate = a leftover stack
```

## Stopping the stack, and recovering a wedged camera

The Orbbec driver installs a SIGINT/SIGTERM handler and its destructor calls
`clean()` to stop the streams and close the device. That works, but it takes a
moment. Kill the process before it finishes -- `kill -9`, a `timeout`, or the
launch system escalating to SIGKILL -- and the device is left with its streams
claimed. It then refuses to open until the USB port is cycled, which is the
"I have to unplug and replug the camera" symptom.

**Stop it with Ctrl-C and let it finish.** Measured on this stack: SIGINT to
`ros2 launch` exits in **2 s** with all three processes reporting "finished
cleanly", and the camera reopens immediately afterwards at full rate. The launch
file also sets generous `sigterm_timeout` / `sigkill_timeout` so the launch system
never SIGKILLs the driver part way through cleanup.

If the camera is already wedged, you do not need to unplug it:

```bash
ros2 run dock_target_detector camera_reset          # or --list to just look
```

It issues the same `USBDEVFS_RESET` ioctl the kernel performs on a replug. The
shipped udev rule (`99-obsensor-libusb.rules`) makes the device node
world-writable, so this needs no sudo. Verified on the 335L: the device
re-initialises **in place**, keeping its bus/device number (a physical replug
gives it a new one), and the driver opens it again a second later.

If the driver is still running but the camera has stopped delivering, prefer its
own service first -- it is gentler than a bus reset:

```bash
ros2 service call /camera/reboot_device std_srvs/srv/Empty
```

## Known limits

- **Yaw still needs a sanity check on real hardware.** The 0.18° figure is from a
  synthetic depth image with 5 mm per-pixel noise. Bench-measure it against a
  protractor before anyone relies on it.
- **A single A4 target is small.** If the real yaw noise disappoints, the cheapest
  fix is two or three targets spread ~1 m apart across the trailer face: yaw then
  becomes `atan2(z_left − z_right, baseline)`, which turns 1 cm of range accuracy
  into well under a degree without depending on a plane fit at all.
- **Colour is rolling shutter** on the Gemini 335L. Fine at backing speeds, but the
  left IR stream is global shutter and natively in the depth frame (no registration
  needed). The IR laser speckle would have to be interleaved away
  (`interleave_frame_enable`, `interleave_ae_mode:=laser`) at the cost of halving
  depth rate.
- **Backing into a dark trailer interior** will fight colour auto-exposure. Expect
  to pin the exposure for the detection stream.
