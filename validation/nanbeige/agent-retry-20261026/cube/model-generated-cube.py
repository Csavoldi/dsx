#!/usr/bin/env python3
"""Compact rotating 3D cube in the terminal (Python stdlib only).

Rotates vertices around two axes, perspective-projects them, and draws all
12 edges into a dot grid. The grid is packed into Unicode Braille characters
(2x4 dots per character), filling the preview terminal.

Raster pipeline (W character columns x H character rows):
  1. dot grid width  2*W, height  4*H
  2. project rotated vertices, scale/center so the cube occupies most of
     the height with margins
  3. draw every edge as a line of dots across the dot grid
  4. pack each 2x4 dot block into one Braille character
"""
import math
import shutil
import sys
import time

VERTICES = [
    (x, y, z) for x in (-1, 1)
    for y in (-1, 1)
    for z in (-1, 1)
]

# Edge list (12 edges), each stored as a pair of vertex indices.
EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]

# Standard Braille bits by row/column within a character.
BRAILLE_BIT = (1, 8, 2, 16, 4, 32, 64, 128)


def rotate_vertex(v, angle_x, angle_y):
    """Rotate a vertex around the X and Y axes (two axes)."""
    x, y, z = v
    # Y rotation first
    cr = math.cos(angle_y)
    sr = math.sin(angle_y)
    x2 = x * cr + z * sr
    z2 = -x * sr + z * cr
    x, z = x2, z2
    # X rotation
    cx = math.cos(angle_x)
    sx = math.sin(angle_x)
    y2 = y * cx - z * sx
    z3 = y * sx + z * cx
    return (x, y2, z3)


def project(v, distance):
    """Perspective projection to screen coordinates."""
    z_cam = v[2] + distance
    if z_cam <= 0.1:
        return None
    scale = 1.0 / z_cam
    return (v[0] * scale, v[1] * scale)


def rasterize_edges(angle_x, angle_y, cols, rows, distance=2.5):
    """Render one frame into a dot grid (width 2*cols, height 4*rows).

    Project vertices, scale/center them so the rotating cube occupies most
    of the height with margins, then draw all 12 edges as lines of dots.
    """
    grid_w = 2 * cols
    grid_h = 4 * rows

    rotated = [rotate_vertex(v, angle_x, angle_y) for v in VERTICES]
    projected = []
    for rv in rotated:
        p = project(rv, distance=distance)
        if p is not None:
            projected.append(p)

    # Scale/center the projected rotating cube so it occupies most of the
    # height with margins. Use the projected bounds over all vertices.
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Vertical extent: leave margins on top/bottom (reserve last row too).
    height_span = (y_max - y_min)
    vertical_scale = (grid_h - 2 * 10) / max(height_span, 1e-6)
    # Horizontal extent: fill width, with modest margins.
    horizontal_scale = (grid_w - 2 * 10) / max(x_max - x_min, 1e-6)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    def to_dot(p):
        """Map projected coords -> dot grid (x horizontal, y vertical)."""
        dx = (p[0] - cx) * horizontal_scale
        dy = (p[1] - cy) * vertical_scale
        # invert y so up is +dots, and clamp inside the grid.
        gx = int(round(grid_w / 2.0 + dx))
        gy = int(round(grid_h / 2.0 + dy))
        return gy, gx

    grid = [[0] * grid_w for _ in range(grid_h)]

    for a, b in EDGES:
        pa = projected[a]
        pb = projected[b]
        ra, ca = to_dot(pa)
        rb, cb = to_dot(pb)
        # Draw the line between the two dot cells using Bresenham.
        steps = max(abs(rb - ra), abs(cb - ca)) or 1
        for t in range(steps + 1):
            r = ra + (rb - ra) * t // steps
            c = ca + (cb - ca) * t // steps
            if 0 <= r < grid_h and 0 <= c < grid_w:
                grid[r][c] |= 1

    return grid


def dot_grid_to_braille(grid):
    """Pack every 2x4 dot block into one Unicode Braille character."""
    lines = []
    for r in range(0, len(grid), 4):
        line = []
        for c in range(0, len(grid[0]), 2):
            bits = 0
            for dr in range(4):
                for dc in range(2):
                    if grid[r + dr][c + dc]:
                        bits |= BRAILLE_BIT[dr * 2 + dc]
            line.append(chr(0x2800 + bits))
        lines.append("".join(line))
    return "\n".join(lines)


def frame_to_unicode(grid):
    """Convert a dot grid into Unicode Braille characters."""
    return dot_grid_to_braille(grid)


def render_frame(angle_x, angle_y, cols, rows, distance=2.5):
    """Render one frame into a cols x rows Braille buffer (dot grid)."""
    grid = rasterize_edges(angle_x, angle_y, cols, rows, distance)
    return frame_to_unicode(grid)


def get_canvas_size():
    """Terminal size in Braille characters, reserving the last row."""
    size = shutil.get_terminal_size()
    cols = size.columns
    rows = max(size.lines - 1, 4)  # reserve the last row
    return cols, rows


def main():
    """Main loop: animate at ~25 FPS filling the terminal."""
    angle_x = 0.0
    angle_y = 0.0
    delay = 1.0 / 25.0  # ~25 FPS

    # Hide cursor, allow alternate screen, reset on exit
    sys.stdout.write("\033[?25l\033[?1049h\033[2J")
    sys.stdout.flush()

    cols, rows = get_canvas_size()

    try:
        while True:
            angle_x += 0.02
            angle_y += 0.03
            frame = render_frame(angle_x, angle_y, cols, rows)
            sys.stdout.write("\033[H" + frame + "\033[0m")
            sys.stdout.flush()
            time.sleep(delay)
    except KeyboardInterrupt:
        # Restore cursor and restore previous screen on Ctrl-C
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    main()
