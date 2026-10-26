#!/usr/bin/env python3
"""Rotating 3D wireframe cube rendered with Unicode Braille.

Standard library only.  The cube's 8 vertices are rotated about two axes,
perspective-projected, and the 12 edges are rasterized onto a Braille canvas
that fills the whole terminal (minus the final row).  Each Braille
character carries a 2x4 dot grid, so W characters and H character rows
give a 2*W by 4*H dot grid.  Standard dot bits by row are
[(1,8),(2,16),(4,32),(64,128)].  Aspect ratio is preserved and the cube
is centered with margins.  Animation runs at ~25 FPS and the terminal and
cursor are restored on Ctrl-C.
"""

import math
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Terminal sizing
# ---------------------------------------------------------------------------

def terminal_size(default_cols=80, default_rows=24):
    """Return (cols, rows) of the controlling terminal."""
    try:
        size = shutil.get_terminal_size(
            fallback=(default_cols, default_rows)
        )
        cols, rows = size.columns, size.lines
        if cols < 4:
            cols = default_cols
        if rows < 2:
            rows = default_rows
        return cols, rows
    except Exception:
        return default_cols, default_rows


# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------

# The 8 vertices of a unit cube, in model space.
CUBE_VERTICES = (
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5,  0.5),
    (0.5, -0.5,  0.5),
    (0.5, 0.5,  0.5),
    (-0.5, 0.5,  0.5),
)

# The 12 edges as index pairs into CUBE_VERTICES.
CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),      # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),      # top face
    (0, 4), (1, 5), (2, 6), (3, 7),      # verticals
)


def rotate(point, ax, ay, az):
    """Rotate ``point`` about the X, Y and Z axes (ZYX Euler order).

    ``ax``, ``ay``, ``az`` are rotation angles in radians.
    """
    x, y, z = point

    # Rotation about Z
    x, y = (
        x * math.cos(az) - y * math.sin(az),
        x * math.sin(az) + y * math.cos(az),
    )
    # Rotation about Y
    x, z = (
        x * math.cos(ay) + z * math.sin(ay),
        -x * math.sin(ay) + z * math.cos(ay),
    )
    # Rotation about X
    y, z = (
        y * math.cos(ax) - z * math.sin(ax),
        y * math.sin(ax) + z * math.cos(ax),
    )
    return (x, y, z)


def project(point, focal, distance):
    """Perspective-project a rotated vertex to screen space.

    Returns (col, row) in character-cell units relative to the canvas
    center; ``col`` grows right and ``row`` grows down.
    """
    x, y, z = point
    # Camera looks down -Z; the vertex must sit in front of the eye.
    depth = distance + z
    if depth <= 0.15:
        depth = 0.15
    scale = focal / depth
    return (x * scale, -y * scale)


# ---------------------------------------------------------------------------
# Braille rasterization
# ---------------------------------------------------------------------------

# Standard Braille dot bits by row: each row holds two dots.
BRAILLE_ROW_BITS = ((1, 8), (2, 16), (4, 32), (64, 128))


class BrailleCanvas:
    """A dot grid rendered as Braille characters.

    The canvas holds ``dot_rows`` dot rows and ``cols`` character columns,
    i.e. a ``2*cols`` by ``dot_rows`` dot grid.  ``to_text`` groups every
    4 dot rows into one character line, because a Braille cell is 2x4
    dots wide/tall.
    """

    def __init__(self, cols, dot_rows):
        self.cols = cols
        self.dot_rows = dot_rows
        # bit grid[y][x]: y in [0, dot_rows), x in [0, 2*cols)
        self.grid = [[0] * (2 * cols) for _ in range(dot_rows)]

    def clear(self):
        for row in self.grid:
            for cell in row:
                cell = 0

    def set_dot(self, x, y):
        """Set dot (x, y); both in dot units."""
        if y < 0 or y >= self.dot_rows:
            return
        max_x = 2 * self.cols - 1
        if x < 0 or x > max_x:
            return
        bit_row, bit_side = BRAILLE_ROW_BITS[y % 4]
        bit = bit_row if x % 2 == 0 else bit_side
        self.grid[y][x] |= bit

    def to_text(self):
        """Collapse the dot grid into Braille characters.

        Every group of 4 dot rows becomes one character line; a final
        incomplete group is zero-padded so the output stays rectangular.
        """
        char_rows = (self.dot_rows + 3) // 4
        lines = []
        for gy in range(char_rows):
            line = []
            for x in range(self.cols):
                bits = 0
                for dy in range(4):
                    yy = gy * 4 + dy
                    if yy >= self.dot_rows:
                        break
                    for dx in range(2):
                        xx = x * 2 + dx
                        if self.grid[yy][xx]:
                            bit_row, bit_side = BRAILLE_ROW_BITS[dy]
                            bits |= bit_row if dx == 0 else bit_side
                line.append(chr(0x2800 + bits) if bits else " ")
            lines.append("".join(line))
        return "\n".join(lines)


def rasterize_edge(canvas, p0, p1, thickness=1.0):
    """Rasterize a projected edge onto the canvas.

    ``p0``/``p1`` are in character-cell units; they are converted to dot
    units (x*2, y*2) before stepping along the line.
    """
    x0, y0 = p0
    x1, y1 = p1

    dx = (x1 - x0) * 2
    dy = (y1 - y0) * 2
    steps = max(1, int(math.hypot(dx, dy) / (thickness * 0.5)))

    for i in range(steps + 1):
        t = i / steps
        x = x0 * 2 + (x1 - x0) * 2 * t
        y = y0 * 2 + (y1 - y0) * 2 * t
        # Snap to the nearest dot.
        canvas.set_dot(int(round(x)), int(round(y)))


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------

def render_frame(vertices_rotated, terminal_cols, terminal_rows,
                 margin=1, focal=10.0, distance=4.0):
    """Build one Braille frame for the current rotation.

    The canvas fills the terminal minus the last row (and a small margin).
    Aspect ratio is preserved and the cube is centered.
    """
    # Reserve the last terminal row.
    usable_rows = max(1, terminal_rows - 1)

    # Character grid that fits the terminal box, minus margins.
    char_rows = max(1, usable_rows - 2 * margin)
    char_cols = max(4, terminal_cols - 2 * margin)

    # Braille characters are ~1/2 as wide as they are tall.  Work in dot
    # units: the horizontal budget is reduced by that factor so the true
    # aspect ratio is kept.
    dot_cols = 2 * char_cols
    dot_cols_corrected = dot_cols * (2.0 / 3.0)
    dot_rows = 4 * char_rows

    canvas = BrailleCanvas(char_cols, dot_rows)

    projected = []
    for v in vertices_rotated:
        col, row = project(v, focal, distance)
        projected.append((col, row))

    # Fit the projected bounding box into the (corrected) dot grid,
    # preserving aspect and centering it.
    min_col = min(p[0] for p in projected)
    max_col = max(p[0] for p in projected)
    min_row = min(p[1] for p in projected)
    max_row = max(p[1] for p in projected)

    dot_w = max_col - min_col
    dot_h = max_row - min_row
    if dot_w <= 0.0 or dot_h <= 0.0:
        dot_w = dot_h = 1.0

    target_w = dot_cols_corrected
    target_h = float(dot_rows)

    scale = min(target_w / dot_w, target_h / dot_h)

    # Center offsets in character-cell units (rasterize_edge converts
    # character cells to dot units internally).
    offset_x = (target_w - dot_w * scale) / 2.0
    offset_y = (target_h - dot_h * scale) / 2.0

    for a, b in CUBE_EDGES:
        p0 = projected[a]
        p1 = projected[b]
        rasterize_edge(
            canvas,
            (
                (p0[0] - min_col) * scale / 2.0 + offset_x / 2.0,
                (p0[1] - min_row) * scale / 2.0 + offset_y / 2.0,
            ),
            (
                (p1[0] - min_col) * scale / 2.0 + offset_x / 2.0,
                (p1[1] - min_row) * scale / 2.0 + offset_y / 2.0,
            ),
        )

    return canvas.to_text()


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def _set_cursor(visible):
    """Hide/show the terminal cursor via stty, best effort."""
    if shutil.which("stty") is None:
        return
    try:
        subprocess.run(
            ["stty", "echo", "vsUSP" if not visible else "vsON"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    cols, rows = terminal_size()

    # ~25 FPS animation; optional knobs, no extra features.
    fps = 25.0
    if "-fps" in argv:
        fps = float([a for a in argv if a != "-fps"][0])

    margin = 1
    if "-margin" in argv:
        margin = int([a for a in argv if a != "-margin"][0])

    _set_cursor(False)

    canvas_cols = cols
    canvas_rows = max(1, rows - 1)

    angle_x = 0.0
    angle_y = 0.0
    last_frame_time = time.monotonic()
    frame_count = 0

    try:
        while True:
            now = time.monotonic()
            if now - last_frame_time > 0.5:
                last_frame_time = now

            # Rotate about two axes.
            angle_x += 0.035
            angle_y += 0.022

            rotated = [
                rotate(v, angle_x, angle_y, 0.0) for v in CUBE_VERTICES
            ]

            text = render_frame(rotated, cols, rows, margin=margin)

            # Reflow the frame into exactly canvas_rows lines so the
            # reserved bottom row stays empty.
            lines = text.split("\n")
            if len(lines) > canvas_rows:
                lines = lines[:canvas_rows]
            elif len(lines) < canvas_rows:
                lines.extend([""] * (canvas_rows - len(lines)))

            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            frame_count += 1
            period = 1.0 / fps
            remaining = period - (time.monotonic() - now)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        # Restore the terminal and cursor on Ctrl-C.
        _set_cursor(True)
        sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
