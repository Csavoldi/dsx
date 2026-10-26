#!/usr/bin/env python3
"""Rotating 3D wireframe cube rendered in Unicode Braille.

Standard library only. All 8 vertices are rotated about two axes,
perspective-projected, and all 12 edges are rasterized onto a Braille
canvas that fills the actual terminal dimensions (last row reserved).

Canvas layout
-------------
For W character columns and H character rows we use a 2W x 4H dot grid.
Each Unicode Braille character cell holds 2 columns x 4 rows of dots,
with dot bits by row given as [(1,8), (2,16), (4,32), (64,128)].

Aspect ratio
------------
Terminal character cells are roughly twice as tall as wide, so a dot
grid scaled 2x wide by 4x tall already approximates square dots. The
projection uses a uniform world scale and the cube is centered with
margins, keeping the aspect ratio correct.

Animation
---------
Roughly 25 FPS. On Ctrl-C the terminal is restored and the cursor
returned to the top-left before exiting.
"""

import math
import shutil
import sys
import time

# ---------------------------------------------------------------------------
# Terminal handling
# ---------------------------------------------------------------------------


def get_terminal_size(default_cols=80, default_rows=24):
    """Return (cols, rows) of the controlling terminal."""
    size = shutil.get_terminal_size(
        (default_cols, default_rows)
    )
    cols = max(1, int(size.columns))
    rows = max(1, int(size.lines))
    return cols, rows


def raw_mode(fd):
    """Enable non-echoing, non-canonical input mode on a file descriptor.

    ISIG stays enabled (we only modify local flags in index 3) so Ctrl-C
    still raises KeyboardInterrupt and we can restore the terminal cleanly.
    """
    old = list(termios.tcgetattr(fd))
    new = list(old)
    # index 3 = local mode flags: disable echo, keep ISIG on
    new[3] = old[3] & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new)
    return old


def restore_terminal(fd, old):
    """Restore a terminal's previous attribute state."""
    if old is not None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def hide_cursor(stream):
    stream.write("\x1b[?25l")
    stream.flush()


def show_cursor(stream):
    stream.write("\x1b[?25h")
    stream.flush()


def clear_screen(stream):
    stream.write("\x1b[2J")
    stream.flush()


# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------


class Cube:
    """A centered unit cube with its 8 vertices and 12 edges."""

    def __init__(self, half_size=1.0):
        self.half = half_size
        half = half_size
        # 8 vertices, corners of the cube in [-half, half]^3
        self.vertices = []
        for x in (-half, half):
            for y in (-half, half):
                for z in (-half, half):
                    self.vertices.append((x, y, z))
        # 12 edges connect vertices that differ in exactly one coordinate
        self.edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                diff = sum(
                    1
                    for k in range(3)
                    if self.vertices[i][k] != self.vertices[j][k]
                )
                if diff == 1:
                    self.edges.append((i, j))

        assert len(self.vertices) == 8
        assert len(self.edges) == 12


def rotate(point, angle_x, angle_y):
    """Rotate a 3D point about the X and Y axes by given radians."""
    x, y, z = point

    # Rotation about X axis
    cy = y * math.cos(angle_x) - z * math.sin(angle_x)
    cz = y * math.sin(angle_x) + z * math.cos(angle_x)
    x, y, z = x, cy, cz

    # Rotation about Y axis
    cx = x * math.cos(angle_y) + z * math.sin(angle_y)
    cz2 = -x * math.sin(angle_y) + z * math.cos(angle_y)
    return (cx, y, cz2)




def project(point, focal_length=5.0, distance=5.0):
    """Perspective-project a 3D point to screen space.

    Returns (col, row) in a coordinate system whose origin is at the
    center of the available dot grid and units are dots.
    """
    x, y, z = point
    # Camera looks down the negative Z axis; translate so cube is centered
    z_cam = distance + z
    if z_cam <= 0.1:
        z_cam = 0.1
    scale = focal_length / z_cam
    return (x * scale, y * scale, z_cam)


def line_points(p0, p1, step=0.05):
    """Yield interpolated points along the segment p0 -> p1 (any dimension)."""
    deltas = [p1[i] - p0[i] for i in range(len(p0))]
    length = (sum(d * d for d in deltas)) ** 0.5
    if length == 0:
        yield p0
        return
    count = max(2, int(length / step))
    for i in range(count + 1):
        t = i / count
        yield tuple(p0[k] + deltas[k] * t for k in range(len(p0)))


# ---------------------------------------------------------------------------
# Braille rasterizer
# ---------------------------------------------------------------------------

# Standard dot bits by row: each row contributes two bits.
DOT_BITS = [(1, 8), (2, 16), (4, 32), (64, 128)]


class BrailleCanvas:
    """A 2W x 4H dot grid rasterized into WxH Braille characters."""

    def __init__(self, cols, rows):
        self.cols = cols  # character columns
        self.rows = rows  # character rows
        self.w = 2 * cols  # dot columns
        self.h = 4 * rows  # dot rows
        # bit grid indexed [dot_row][dot_col]
        self.grid = [0] * (self.h * self.w)

    def clear(self):
        self.grid = [0] * (self.h * self.w)

    def set_dot(self, r, c):
        if 0 <= r < self.h and 0 <= c < self.w:
            self.grid[r * self.w + c] |= 1

    def pixel_on(self, r, c):
        return bool(self.grid[r * self.w + c])

    def to_characters(self):
        """Collapse dot grid into WxH Braille characters."""
        out = []
        for char_row in range(self.rows):
            line = []
            for char_col in range(self.cols):
                bits = 0
                base = char_row * 4
                base_c = char_col * 2
                for r_off in range(4):
                    row_bits = DOT_BITS[r_off]
                    for c_off, bit in enumerate(row_bits):
                        if self.grid[(base + r_off) * self.w + (base_c + c_off)]:
                            bits |= bit
                line.append(chr(0x2800 | bits))
            out.append("".join(line))
        return "\n".join(out)


def rasterize_segment(canvas, p0_screen, p1_screen):
    """Rasterize a screen-space segment onto the Braille canvas.

    Screen points are (x, y) in dot units where x grows right and y grows
    down; the canvas grid is indexed [row][col] = [y][x], so the order is
    swapped when writing a dot.
    """
    for p in line_points(p0_screen, p1_screen):
        canvas.set_dot(int(round(p[1])), int(round(p[0])))


# ---------------------------------------------------------------------------
# Main animation
# ---------------------------------------------------------------------------

FPS = 25
ROTATION_SPEED_X = 0.6  # radians per second about X
ROTATION_SPEED_Y = 0.9  # radians per second about Y
FOCAL_LENGTH = 5.0
CAMERA_DISTANCE = 5.0


def frame_string(cube, angle_x, angle_y, cols, rows, center_x, center_y,
                 scale):
    """Build one frame of Braille text for the rotating cube."""
    canvas = BrailleCanvas(cols, rows)

    rotated = [
        rotate(v, angle_x, angle_y) for v in cube.vertices
    ]
    projected = [
        project(p, FOCAL_LENGTH, CAMERA_DISTANCE) for p in rotated
    ]

    # Convert projected units to dot offsets around the canvas center
    for idx, (i, j) in enumerate(cube.edges):
        a = (
            center_x + projected[i][0] * scale,
            center_y + projected[i][1] * scale,
        )
        b = (
            center_x + projected[j][0] * scale,
            center_y + projected[j][1] * scale,
        )
        rasterize_segment(canvas, a, b)

    return canvas.to_characters()


def main():
    # --- terminal setup ---------------------------------------------------
    fd = sys.stdout.fileno()
    old_attrs = raw_mode(fd)
    hide_cursor(sys.stdout)

    try:
        # Reserve the last row: the animation area has (rows - 1) character
        # rows, filled edge-to-edge across all `cols` columns.
        cols, rows = get_terminal_size()
        char_rows = max(1, rows - 1)

        # The canvas is 2 cols wide and 4 rows tall in dots, which already
        # matches the terminal's roughly-2:1 (tall) character aspect.
        # Keep the cube centered with a margin by scaling to fit inside the
        # dot grid with a safety factor.
        margin = 0.08
        max_dot_radius = min(
            (cols * 2 - 2 * margin) / 2.0,
            (char_rows * 4 - 2 * margin) / 2.0,
        )
        # Pick scale so the whole cube fits with the margin across all
        # orientations. A corner of the unit cube has radius sqrt(3);
        # with focal 5 and distance 5 its projected extent stays under
        # ~2.6 screen units, so dividing by 3.0 keeps a safe margin.
        scale = max_dot_radius / 3.0

        center_x = cols  # dot column center (0-based between cells)
        center_y = char_rows * 2  # dot row center

        cube = Cube(half_size=1.0)
        angle_x = 0.0
        angle_y = 0.0
        last_time = time.time()

        while True:
            now = time.time()
            dt = min(now - last_time, 0.1)
            last_time = now

            angle_x += ROTATION_SPEED_X * dt
            angle_y += ROTATION_SPEED_Y * dt

            text = frame_string(
                cube, angle_x, angle_y, cols, char_rows,
                center_x, center_y, scale,
            )

            # Redraw: home cursor, clear, then print the frame. The raw
            # terminal mode has ONLCR off, so rows must carry an explicit
            # carriage return to start each line at column 0.
            sys.stdout.write("\x1b[H\x1b[2J")
            sys.stdout.write(text.replace("\n", "\r\n"))
            sys.stdout.flush()

            # Frame pacing: aim for ~25 FPS.
            elapsed = time.time() - now
            sleep_time = max(0.0, (1.0 / FPS) - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        restore_terminal(fd, old_attrs)
        show_cursor(sys.stdout)
        sys.stdout.write("\x1b[H")
        sys.stdout.flush()


if __name__ == "__main__":
    import math
    import termios

    main()
