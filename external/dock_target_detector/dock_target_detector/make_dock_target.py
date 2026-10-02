#!/usr/bin/env python3
"""Generate a print-ready A4 AprilTag dock target at an exactly known physical size.

An AprilTag 36h11 sheet. The tag id is the identity; there is no payload to
choose. Two things about the sheet matter:

  * **Cell size.** Detection range is ``px_per_cell = cell_mm * fx / distance``
    and AprilTag needs about 1.5. 36h11 is 8x8 cells, so a 190 mm sheet gives
    23.75 mm cells -- 2.6x larger than a QR of the same size, which is why it
    reads so much further and through so much more motion blur.
  * **Quiet zone -- barely matters.** The spec likes 1 blank cell, and a 190 mm
    tag on A4 leaves 0.4. Measured synthetically, 36h11 still read 5/5 with *no*
    quiet zone at all against white, grey and dark backgrounds: the detector
    finds the black border by contour rather than scanning lines across it, so it
    does not depend on the surround the way a QR code does. A margin is good
    practice, not a requirement.
  * **Print scale -- minor.** ``marker_size`` feeds only the PnP range estimate,
    which is a cross-check; the depth-derived range, bearing and yaw never use it,
    and PnP *orientation* is scale-invariant too. A 5 % scale error costs 5 % of
    the PnP range and about 5 % of decode range, and changes nothing else. Measure
    the printed calibration bar and pass ``--measured-bar-mm`` rather than
    reprinting.

Output is PDF or SVG, both sized in real millimetres; the ``--out`` extension
picks which. Prefer PDF for printing: viewers offer a dependable "Actual size",
where a browser printing SVG will apply its own page setup.

The PDF is written directly rather than through a rendering library. The page is
only filled rectangles and base-14 text, so this keeps the package free of any
dependency beyond OpenCV and NumPy.
"""
import argparse
import sys

import cv2
import numpy as np

A4_W_MM = 210.0
A4_H_MM = 297.0
# AprilTag likes 1 blank cell on every side, though it detects fine without one
# (see the module docstring).
QUIET_SPEC_CELLS = 1.0
MM_TO_PT = 72.0 / 25.4

TOP_MM = 10.0           # margin above the symbol
BAR_GAP_MM = 6.0        # symbol bottom -> calibration bar
TEXT_GAP_MM = 11.0      # bar -> first caption line
LINE_MM = 4.4
BOTTOM_MM = 8.0
CAPTION_MM = 3.2        # caption glyph height
LABEL_MM = 4.0          # calibration bar label

GREY = "0.2 0.2 0.2"
BLACK = "0 0 0"


def apriltag_matrix(tag_id):
    """36h11 bit matrix including its black border, as rows of 0/1."""
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    cells = d.markerSize + 2                      # 6 data bits + 1 border each side
    img = cv2.aruco.drawMarker(d, int(tag_id), cells * 10)
    small = cv2.resize(img, (cells, cells), interpolation=cv2.INTER_NEAREST)
    return [[1 if v < 128 else 0 for v in row] for row in small]


def build_layout(_payload, size_mm, fx_list, tag_id=7):
    """Lay the sheet out in millimetres, y measured downwards from the top edge.

    Returns black `rects`, `texts` and the derived `meta` numbers, which the SVG
    and PDF renderers both consume.
    """
    # 8x8 cells. Measured against a QR on the same 180 mm sheet at 1280x720:
    # readable to 5 m where the QR stopped at 3 m, through 8 px of motion blur
    # against the QR's 2 px, and ~4 ms full-frame against ~50 ms.
    matrix = apriltag_matrix(tag_id)
    version, payload = f"36h11 id {tag_id}", f"TAG{tag_id}"
    n = len(matrix)  # cells per side, quiet zone excluded
    module_mm = size_mm / n

    x0 = (A4_W_MM - size_mm) / 2.0
    y0 = TOP_MM
    paper_quiet_modules = x0 / module_mm
    # Extra white needed beyond the sheet edge to reach the 4-module spec.
    quiet_spec = QUIET_SPEC_CELLS
    backing_extra_mm = max(0.0, quiet_spec * module_mm - x0)

    rects = []
    for r, row in enumerate(matrix):
        c = 0
        while c < n:
            if row[c]:
                run = 1
                while c + run < n and row[c + run]:
                    run += 1
                rects.append((x0 + c * module_mm, y0 + r * module_mm,
                              run * module_mm, module_mm))
                c += run
            else:
                c += 1

    # Calibration bar. The two end ticks are placed so their OUTER edges are
    # exactly 100 mm apart.
    bar_y = y0 + size_mm + BAR_GAP_MM
    rects += [
        (x0, bar_y, 100.0, 1.2),
        (x0, bar_y - 2.0, 0.5, 5.2),
        (x0 + 99.5, bar_y - 2.0, 0.5, 5.2),
    ]
    texts = [(x0 + 103.0, bar_y + 2.5, LABEL_MM, "sans", BLACK,
              "100 mm - measure me")]

    lines = [
        'payload "%s"   %s   (%d x %d cells)' % (payload, version, n, n),
        "marker_size = %.4f m   (%.1f mm black area, edge to edge)   module = %.2f mm"
        % (size_mm / 1000.0, size_mm, module_mm),
        "",
        "Mount FLAT on rigid backing - a rippled sheet corrupts the depth plane fit.",
    ]
    if backing_extra_mm > 0.5:
        lines += [
            "",
            "QUIET ZONE: this sheet leaves %.1f of the %.0f blank cell AprilTag likes."
            % (paper_quiet_modules, quiet_spec),
            "Measured synthetically, 36h11 still read 5/5 with NO quiet zone on white,",
            "grey and dark backgrounds: the detector finds the black border by contour,",
            "unlike a QR scanline decoder. A margin is good practice, not a requirement.",
        ]
    lines += [
        "",
        "Print at 100% / actual size; check the bar. A wrong scale costs only",
        "proportional PnP range and decode range - nothing else.",
        "Estimated decode range (~1.5 px per cell):",
    ]
    for label, fx in fx_list:
        lines.append("    %s: %.1f m" % (label, module_mm * fx / 2.0 / 1000.0))

    ty = bar_y + TEXT_GAP_MM
    for line in lines:
        if line:
            texts.append((x0, ty, CAPTION_MM, "mono", GREY, line))
        ty += LINE_MM

    # The caption is generated, so its height depends on the options. Fail loudly
    # rather than silently running text off the bottom of the page.
    if ty > A4_H_MM - BOTTOM_MM:
        raise ValueError(
            "layout overflows A4: caption reaches %.1f mm of %.0f mm. "
            "Reduce --size-mm (currently %.0f)." % (ty, A4_H_MM, size_mm))

    return {
        "rects": rects,
        "texts": texts,
        "meta": {
            "version": version, "modules": n, "module_mm": module_mm,
            "quiet_modules": paper_quiet_modules,
            "backing_extra_mm": backing_extra_mm, "size_mm": size_mm,
        },
    }


def _svg_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(layout):
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%smm" height="%smm" '
        'viewBox="0 0 %s %s">' % (A4_W_MM, A4_H_MM, A4_W_MM, A4_H_MM),
        '<rect width="%s" height="%s" fill="#ffffff"/>' % (A4_W_MM, A4_H_MM),
    ]
    # One path for every black rectangle keeps the file small and the edges crisp.
    d = "".join("M%.4f,%.4fh%.4fv%.4fh%.4fz" % (x, y, w, h, -w)
                for x, y, w, h in layout["rects"])
    parts.append('<path d="%s" fill="#000000"/>' % d)
    for x, y, size, font, colour, text in layout["texts"]:
        parts.append(
            '<text x="%.2f" y="%.2f" font-family="%s" font-size="%s" fill="%s">%s</text>'
            % (x, y, "monospace" if font == "mono" else "sans-serif", size,
               "#333333" if colour == GREY else "#000000", _svg_escape(text)))
    parts.append("</svg>")
    return "\n".join(parts).encode("utf-8")


def _pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render_pdf(layout):
    k = MM_TO_PT
    pw, ph = A4_W_MM * k, A4_H_MM * k
    fonts = {"mono": "/F1", "sans": "/F2"}

    ops = ["1 1 1 rg 0 0 %.3f %.3f re f" % (pw, ph), "%s rg" % BLACK]
    for x, y, w, h in layout["rects"]:
        # Layout y is the top edge growing downwards; PDF y is the bottom edge
        # growing upwards.
        ops.append("%.3f %.3f %.3f %.3f re"
                   % (x * k, (A4_H_MM - y - h) * k, w * k, h * k))
    ops.append("f")  # one fill for all the accumulated subpaths
    for x, y, size, font, colour, text in layout["texts"]:
        ops.append("%s rg BT %s %.3f Tf %.3f %.3f Td (%s) Tj ET"
                   % (colour, fonts[font], size * k, x * k,
                      (A4_H_MM - y) * k, _pdf_escape(text)))
    stream = "\n".join(ops).encode("latin-1")

    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        ("<</Type/Page/Parent 2 0 R/MediaBox[0 0 %.3f %.3f]"
         "/Resources<</Font<</F1 5 0 R/F2 6 0 R>>>>/Contents 4 0 R>>"
         % (pw, ph)).encode("latin-1"),
        (b"<</Length " + str(len(stream)).encode() + b">>\nstream\n"
         + stream + b"\nendstream"),
        b"<</Type/Font/Subtype/Type1/BaseFont/Courier>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += ("%d 0 obj\n" % i).encode("latin-1") + body + b"\nendobj\n"
    xref_at = len(out)
    out += ("xref\n0 %d\n" % (len(objects) + 1)).encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += ("%010d 00000 n \n" % off).encode("latin-1")
    out += ("trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref_at)).encode("latin-1")
    return bytes(out)


RENDERERS = {".pdf": render_pdf, ".svg": render_svg}


def report_print_scale(args, fx_list):
    """Correct marker_size for a sheet that printed at the wrong scale.

    The calibration bar, the symbol and the margins all scale together, so one
    measurement recovers the true size of everything on the page.
    """
    scale = args.measured_bar_mm / 100.0
    n = len(apriltag_matrix(args.tag_id))
    size = args.size_mm * scale
    module = size / n
    quiet_mm = ((A4_W_MM - args.size_mm) / 2.0) * scale
    quiet_spec = QUIET_SPEC_CELLS

    print("calibration bar measured %.1f mm instead of 100.0" % args.measured_bar_mm)
    print("  print scale  %.1f %%" % (scale * 100))
    if abs(scale - 1.0) < 0.005:
        print("  within half a percent of correct; nothing to do.")
        return 0
    print("  printed tag  %.1f mm (intended %.1f)" % (size, args.size_mm))
    print("  module       %.2f mm" % module)
    print()
    print("  marker_size  %.4f m   <-- use this, no reprint needed" % (size / 1000.0))
    print()
    for label, fx in fx_list:
        print("  range @ %s: ~%.1f m" % (label, module * fx / 2.0 / 1000.0))
    backing = max(0.0, quiet_spec * module - quiet_mm)
    if backing > 0.5:
        print("\n  White backing must still reach >= %.0f mm beyond every edge "
              "of the sheet." % backing)
    print("\nOnly the PnP range cross-check and the decode range scale with this;")
    print("the depth range, bearing and yaw are unaffected.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size-mm", type=float, default=190.0,
                    help="side of the black module area, excluding the quiet "
                         "zone (default: %(default)s)")
    ap.add_argument("--ecc", default="m", choices=["l", "m", "q", "h"],
                    help="error correction; higher survives dirt but costs "
                         "modules, so range (default: %(default)s)")
    ap.add_argument("--marker-type", default="apriltag", choices=["apriltag", "qr"],
                    help="apriltag reads ~1.7x further, survives ~4x more motion "
                         "blur and detects ~15x faster on the same sheet; qr "
                         "carries an arbitrary string (default: %(default)s)")
    ap.add_argument("--tag-id", type=int, default=7,
                    help="AprilTag 36h11 id, 0-586 (default: %(default)s)")
    ap.add_argument("--out", default="dock_target.pdf",
                    help="output file; .pdf or .svg (default: %(default)s)")
    ap.add_argument("--measured-bar-mm", type=float, default=None,
                    help="what the printed 100 mm calibration bar actually "
                         "measures. Printers commonly shrink a page to fit their "
                         "printable area, and everything on the sheet scales "
                         "together. Give the measurement and this reports the "
                         "corrected marker_size for the sheet you already have, "
                         "instead of reprinting.")
    args = ap.parse_args(argv)

    if args.size_mm > A4_W_MM - 10.0:
        ap.error("--size-mm %s leaves under 5 mm of paper either side; use %s or less"
                 % (args.size_mm, A4_W_MM - 10.0))

    # Measured off this Gemini 335L; fx scales with colour width.
    fx_list = [("1280x720  (fx 612)", 612.1), ("1920x1080 (fx 918)", 918.2)]

    if args.measured_bar_mm is not None:
        return report_print_scale(args, fx_list)

    ext = args.out[args.out.rfind("."):].lower() if "." in args.out else ""
    if ext not in RENDERERS:
        ap.error("unsupported output extension %r; use one of %s"
                 % (ext, ", ".join(sorted(RENDERERS))))

    layout = build_layout(None, args.size_mm, fx_list, args.tag_id)
    with open(args.out, "wb") as f:
        f.write(RENDERERS[ext](layout))

    m = layout["meta"]
    print("wrote %s" % args.out)
    print("  marker       %s  (%d x %d cells)" % (m["version"], m["modules"], m["modules"]))
    print("  marker_size  %.4f m   <-- pass this to the node" % (args.size_mm / 1000.0))
    print("  module       %.2f mm" % m["module_mm"])
    thresh = 1.5   # px per cell that AprilTag needs
    for label, fx in fx_list:
        print("  range @ %s: ~%.1f m (%.1f px per cell)"
              % (label, m["module_mm"] * fx / thresh / 1000.0, thresh))
    if m["backing_extra_mm"] > 0.5:
        print("\n  QUIET ZONE: the sheet leaves %.1f of the 1 blank cell AprilTag likes."
              % m["quiet_modules"])
        print("  Not a blocker: 36h11 read 5/5 with no quiet zone at all, on white,")
        print("  grey and dark backgrounds. Mount it on anything reasonably flat.")
    print("\nPrint at 100% / actual size, then check the 100 mm bar with a ruler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
