#!/usr/bin/env python3
"""A compact Doom-inspired first-person terminal shooter.

The scene is ray-cast at Braille dot resolution: every character carries
2x4 real dots, so walls, floors, ceilings, sprites, the weapon and its
flash all share one continuous 108x60 (for a 54x15 view) dot image.
"""
import math
import os
import random
import select
import shutil
import signal
import sys
import termios
import time
import tty

# ----------------------------------------------------------------------------
# Fixed map.  Every cell is either '#' (wall) or '.' (walkable floor).
# ----------------------------------------------------------------------------
MAP = [
    "################",
    "#..............#",
    "#.####.###.###.#",
    "#....#...#.....#",
    "#.##.#.#.#.###.#",
    "#.#.........#..#",
    "#.#.#####.###..#",
    "#...#...#......#",
    "###.#.#.#.####.#",
    "#...#...#....#.#",
    "#.#####.####.#.#",
    "#.......#....#.#",
    "#.#####.#.##.#.#",
    "#.#.....#....#.#",
    "#.#.#####.####.#",
    "################",
]
MH = len(MAP)
MW = len(MAP[0])

WAYPOINTS = [(1, 1), (13, 1), (13, 3), (1, 3), (1, 5), (13, 5),
             (13, 7), (1, 9), (14, 9), (1, 11), (11, 11), (9, 14),
             (14, 13), (5, 13), (1, 13), (7, 7), (5, 9)]
ENEMY_SPAWNS = [(13, 1), (1, 5), (7, 9), (11, 11), (5, 13), (11, 3)]

# 256-colour palettes, dark -> bright.
WALL = [52, 88, 94, 130, 136, 172, 178, 214, 220, 226]
FLOOR = [234, 235, 236, 237, 238, 239, 240, 241, 242, 243]
CEIL = [17, 18, 19, 23, 24, 25, 30, 31, 37, 44]
DEMON = [88, 124, 130, 160, 166, 172, 196, 202, 208, 214]
GUN = [236, 238, 240, 242, 244, 246, 248, 250, 252, 255]

# Braille dot bit layout: (column, row) -> bit.
BRAILLE_BITS = ((0x01, 0x02, 0x04, 0x40),
                (0x08, 0x10, 0x20, 0x80))

# Sprite masks are drawn in the same dot grid as the world.
DEMON_ART = [
    "..H#....#H..",
    ".HH##..##HH.",
    "..########..",
    ".##########.",
    ".##########.",
    ".##E####E##.",
    ".##########.",
    "..########..",
    "...######...",
    "..##.##.##..",
    ".##..##..##.",
    ".##..##..##.",
    "..#..##..#..",
    "..#..##..#..",
    ".##.....##..",
    "##.......##.",
]
GUN_ART = [
    "......##......",
    "......##......",
    ".....####.....",
    ".....####.....",
    "....######....",
    "...########...",
    "..##########..",
    ".############.",
    "##############",
    "##############",
    ".############.",
    "..##########..",
]
FLASH_ART = [
    "...###...",
    ".#######.",
    "#########",
    ".#######.",
    "...###...",
]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def ang_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


class Game:
    def __init__(self, autoplay):
        self.autoplay = autoplay
        self.term_w, self.term_h = 54, 17
        self.running = True
        self.resized = True
        self.keys = ""
        self.round = 1
        self.health = 100
        self.ammo = 24
        self.kills = 0
        self.fire_cd = 0.0
        self.flash = 0.0
        self.reload = 0.0
        self.angle = 0.0
        self.px, self.py = 1.5, 1.5
        self.enemies = []
        self.wp_index = 0
        self.path = []
        self.spawn_round()
        self.set_size(self.term_w, self.term_h)

    def spawn_round(self):
        self.health = 100
        self.ammo = 24
        self.kills = 0
        self.fire_cd = 0.0
        self.flash = 0.0
        self.reload = 0.0
        self.px, self.py = 1.5, 1.5
        self.angle = 0.1
        self.enemies = []
        for ex, ey in ENEMY_SPAWNS:
            self.enemies.append({
                "x": ex + 0.5, "y": ey + 0.5, "hp": 100,
                "alive": True, "attack_cd": 0.4 + random.random() * 1.2,
            })
        self.path = []
        self.wp_index = 0

    def set_size(self, w, h):
        self.term_w = max(20, w)
        self.term_h = max(8, h)
        self.hud_rows = 2
        self.view_rows = max(4, self.term_h - self.hud_rows)
        self.dot_w = self.term_w * 2
        self.dot_h = self.view_rows * 4
        self.horizon = self.dot_h // 2
        self.pixels = [None] * (self.dot_w * self.dot_h)
        self.wall_dist = [40.0] * self.dot_w
        self.wall_tex = [0.0] * self.dot_w
        self.wall_side = [0] * self.dot_w
        self.resize_clear = True

    # ------------------------------------------------------------------ world
    def los(self, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        steps = max(2, int(max(abs(dx), abs(dy)) * 4))
        for i in range(1, steps):
            t = i / steps
            mx, my = int(x0 + dx * t), int(y0 + dy * t)
            if 0 <= mx < MW and 0 <= my < MH and MAP[my][mx] == '#':
                return False
        return True

    def blocked(self, x, y):
        r = 0.17
        for ox, oy in ((r, r), (r, -r), (-r, r), (-r, -r)):
            mx, my = int(x + ox), int(y + oy)
            if not (0 <= mx < MW and 0 <= my < MH) or MAP[my][mx] == '#':
                return True
        return False

    def move(self, dx, dy):
        if not self.blocked(self.px + dx, self.py):
            self.px += dx
        if not self.blocked(self.px, self.py + dy):
            self.py += dy

    def bfs(self, start, target):
        if start == target:
            return []
        q = [start]
        prev = {start: None}
        while q:
            cur = q.pop(0)
            if cur == target:
                break
            cx, cy = cur
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (0 <= nx < MW and 0 <= ny < MH and MAP[ny][nx] == '.'
                        and (nx, ny) not in prev):
                    prev[(nx, ny)] = cur
                    q.append((nx, ny))
        if target not in prev:
            return []
        out = []
        cur = target
        while cur != start and prev[cur] is not None:
            out.append(cur)
            cur = prev[cur]
        out.reverse()
        return out

    def nearest_visible_enemy(self):
        best, bd = None, 1e9
        for e in self.enemies:
            if not e["alive"]:
                continue
            d = math.hypot(e["x"] - self.px, e["y"] - self.py)
            if d < bd and self.los(self.px, self.py, e["x"], e["y"]):
                best, bd = e, d
        return best, bd

    def shoot(self):
        if self.fire_cd > 0.0 or self.ammo <= 0 or self.reload > 0.0:
            return
        self.fire_cd = 0.19
        self.flash = 0.075
        self.ammo -= 1
        target, dist = self.nearest_visible_enemy()
        if target is None:
            return
        aim = math.atan2(target["y"] - self.py, target["x"] - self.px)
        if abs(ang_diff(aim, self.angle)) < 0.20:
            target["hp"] -= 34
            if target["hp"] <= 0:
                target["alive"] = False
                self.kills += 1
                self.ammo = min(24, self.ammo + 4)
        if self.ammo <= 0:
            self.reload = 1.15

    # ----------------------------------------------------------------- update
    def update(self, dt):
        self.fire_cd = max(0.0, self.fire_cd - dt)
        self.flash = max(0.0, self.flash - dt)
        if self.reload > 0.0:
            self.reload -= dt
            if self.reload <= 0.0:
                self.reload = 0.0
                self.ammo = 24

        if self.autoplay:
            self.update_auto(dt)
        else:
            self.update_manual(dt, self.keys)
        self.keys = ""

        # Enemies punish exposed players; the player can die and restart.
        for e in self.enemies:
            if not e["alive"]:
                continue
            e["attack_cd"] -= dt
            d = math.hypot(e["x"] - self.px, e["y"] - self.py)
            if e["attack_cd"] <= 0.0 and d < 10.0 and self.los(
                    e["x"], e["y"], self.px, self.py):
                e["attack_cd"] = 1.25 + random.random() * 0.8
                self.health -= 4
        if self.health <= 0 or self.kills >= len(self.enemies):
            self.round += 1
            self.spawn_round()

    def update_auto(self, dt):
        # Always follow a BFS path to the next waypoint: no parking.
        if not self.path:
            target = WAYPOINTS[self.wp_index % len(WAYPOINTS)]
            self.wp_index += 1
            start = (int(self.px), int(self.py))
            self.path = self.bfs(start, target)
            if not self.path and target != start:
                self.path = self.bfs(start, WAYPOINTS[self.wp_index % len(WAYPOINTS)])
                self.wp_index += 1

        move_angle = self.angle
        if self.path:
            tx = self.path[0][0] + 0.5
            ty = self.path[0][1] + 0.5
            vx, vy = tx - self.px, ty - self.py
            md = math.hypot(vx, vy)
            if md < 0.09:
                self.path.pop(0)
            else:
                move_angle = math.atan2(vy, vx)
                speed = 2.35
                self.move(vx / md * speed * dt, vy / md * speed * dt)
                if md > 0.5 and self.blocked(self.px + vx / md * 0.12,
                                             self.py + vy / md * 0.12):
                    self.path = []

        enemy, dist = self.nearest_visible_enemy()
        if enemy is not None:
            want = math.atan2(enemy["y"] - self.py, enemy["x"] - self.px)
            turn = clamp(ang_diff(want, self.angle), -4.8 * dt, 4.8 * dt)
            self.angle += turn
            if abs(ang_diff(want, self.angle)) < 0.15 and dist < 12.0:
                self.shoot()
        else:
            turn = clamp(ang_diff(move_angle, self.angle), -3.2 * dt, 3.2 * dt)
            self.angle += turn

    def update_manual(self, dt, keys):
        turn = 0.0
        if "\x1b[D" in keys or "a" in keys:
            turn -= 2.4 * dt
        if "\x1b[C" in keys or "d" in keys:
            turn += 2.4 * dt
        self.angle += turn
        dx, dy = math.cos(self.angle), math.sin(self.angle)
        step = 2.4 * dt
        if "w" in keys:
            self.move(dx * step, dy * step)
        if "s" in keys:
            self.move(-dx * step, -dy * step)
        if " " in keys:
            self.shoot()

    # ---------------------------------------------------------------- renderer
    def set_dot(self, x, y, color, priority):
        if 0 <= x < self.dot_w and 0 <= y < self.dot_h:
            self.pixels[y * self.dot_w + x] = (color, priority)

    def draw_scene(self):
        px, py = self.px, self.py
        dir_x, dir_y = math.cos(self.angle), math.sin(self.angle)
        fov = 0.66
        plane_x, plane_y = -dir_y * fov, dir_x * fov
        dw, dh, horizon = self.dot_w, self.dot_h, self.horizon

        for x in range(dw):
            cam = 2.0 * x / dw - 1.0
            rdx = dir_x + plane_x * cam
            rdy = dir_y + plane_y * cam
            map_x, map_y = int(px), int(py)
            dx = abs(1.0 / rdx) if rdx != 0.0 else 1e30
            dy = abs(1.0 / rdy) if rdy != 0.0 else 1e30
            if rdx < 0:
                sx = (px - map_x - 1.0) * -dx
            else:
                sx = (px - map_x) * dx
            if rdy < 0:
                sy = (py - map_y - 1.0) * -dy
            else:
                sy = (py - map_y) * dy
            step_x = -1 if rdx < 0 else 1
            step_y = -1 if rdy < 0 else 1
            side = 0
            hit = False
            for _ in range(40):
                if sx < sy:
                    sx += dx
                    map_x += step_x
                    side = 0
                else:
                    sy += dy
                    map_y += step_y
                    side = 1
                if 0 <= map_x < MW and 0 <= map_y < MH and MAP[map_y][map_x] == '#':
                    hit = True
                    break
            if hit:
                if side == 0:
                    perp = (map_x - px + (1 - step_x) / 2.0) / rdx
                    tex = py + perp * rdy
                else:
                    perp = (map_y - py + (1 - step_y) / 2.0) / rdy
                    tex = px + perp * rdx
                perp = max(0.08, perp)
                tex -= math.floor(tex)
                self.wall_dist[x] = perp
                self.wall_tex[x] = tex
                self.wall_side[x] = side
            else:
                perp = 40.0
                tex = 0.0
                self.wall_dist[x] = 40.0
                self.wall_tex[x] = 0.0
                self.wall_side[x] = side

            line_h = int(dh * 1.22 / perp)
            if not hit:
                line_h = 2
            top = horizon - line_h // 2
            bot = horizon + line_h // 2
            shade = clamp(int(9 - perp * 0.42 - (0.5 if side else 0)), 1, 9)
            for y in range(dh):
                if y < top:
                    # ceiling panel geometry
                    rd = (dh * 0.5) / max(0.75, horizon - y)
                    wx = px + rd * rdx
                    wy = py + rd * rdy
                    gx = (wx * 2.0) % 1.0
                    gy = (wy * 2.0) % 1.0
                    if gx < 0.10 or gy < 0.10:
                        continue
                    cs = clamp(int(7 - rd * 0.23), 1, 9)
                    self.set_dot(x, y, CEIL[cs], 5 + cs)
                elif y >= bot:
                    # floor tile geometry
                    rd = (dh * 0.5) / max(0.75, y - horizon)
                    wx = px + rd * rdx
                    wy = py + rd * rdy
                    fx, fy = wx * 2.0, wy * 2.0
                    gx, gy = fx % 1.0, fy % 1.0
                    if gx < 0.08 or gy < 0.08:
                        continue
                    checker = (int(fx) + int(fy)) & 1
                    fs = clamp(int(8 - rd * 0.34) + checker, 1, 9)
                    self.set_dot(x, y, FLOOR[fs], 15 + fs)
                else:
                    # brick wall texture, with mortar gaps and highlights
                    v = (y - top) / max(1.0, line_h)
                    bv = v * 8.0
                    row = int(bv)
                    fv = bv - row
                    bx = tex * 3.0 + (0.5 if row & 1 else 0.0)
                    fx = bx - math.floor(bx)
                    if fv < 0.14 or fx < 0.09:
                        continue
                    ws = shade
                    if 0.40 < fx < 0.60 and 0.40 < fv < 0.72:
                        ws = min(9, ws + 1)
                    self.set_dot(x, y, WALL[ws], 30 + ws)

    def draw_sprites(self):
        px, py = self.px, self.py
        dir_x, dir_y = math.cos(self.angle), math.sin(self.angle)
        fov = 0.66
        plane_x, plane_y = -dir_y * fov, dir_x * fov
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y)
        dw, dh, horizon = self.dot_w, self.dot_h, self.horizon
        for e in self.enemies:
            if not e["alive"]:
                continue
            sx, sy = e["x"] - px, e["y"] - py
            tx = inv_det * (dir_y * sx - dir_x * sy)
            ty = inv_det * (-plane_y * sx + plane_x * sy)
            if ty <= 0.12:
                continue
            screen_x = int((dw / 2.0) * (1.0 + tx / ty))
            sh = int(dh * 0.90 / ty)
            sw = int(sh * 12 / 16)
            if sh < 2 or sw < 2:
                continue
            x0 = screen_x - sw // 2
            y0 = horizon - sh // 2
            body_shade = clamp(int(8 - ty * 0.45), 1, 9)
            for yy in range(sh):
                my = int(yy * 16 / sh)
                row = DEMON_ART[my]
                for xx in range(sw):
                    mx = int(xx * 12 / sw)
                    ch = row[mx]
                    if ch == '.':
                        continue
                    gx, gy = x0 + xx, y0 + yy
                    if not (0 <= gx < dw and 0 <= gy < dh):
                        continue
                    if ty >= self.wall_dist[gx]:
                        continue
                    if ch == 'E':
                        self.set_dot(gx, gy, 226, 300)
                    elif ch == 'H':
                        self.set_dot(gx, gy, 220, 250)
                    else:
                        self.set_dot(gx, gy, DEMON[body_shade], 80 + body_shade)

    def draw_weapon(self):
        dw, dh = self.dot_w, self.dot_h
        # Crosshair at the aim point.
        cx, cy = dw // 2, self.horizon
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            self.set_dot(cx + dx, cy + dy, 46, 350)
        scale = max(1.0, self.view_rows / 15.0)
        sw = int(14 * 1.55 * scale)
        sh = int(12 * 1.55 * scale)
        x0 = dw // 2 - sw // 2
        y0 = dh - sh - 1
        for yy in range(sh):
            my = int(yy * 12 / sh)
            row = GUN_ART[my]
            for xx in range(sw):
                mx = int(xx * 14 / sw)
                if row[mx] == '.':
                    continue
                gx, gy = x0 + xx, y0 + yy
                gs = clamp(int(9 - yy * 4 / max(1, sh)), 3, 9)
                self.set_dot(gx, gy, GUN[gs], 100 + gs)
        if self.flash > 0.0:
            fw, fh = 9, 5
            fx0 = dw // 2 - fw // 2
            fy0 = y0 - fh + 2
            for yy in range(fh):
                row = FLASH_ART[yy]
                for xx in range(fw):
                    if row[xx] != '.':
                        self.set_dot(fx0 + xx, fy0 + yy, 228 if (xx + yy) % 3 else 220, 400)

    def pack_cells(self):
        """Pack 2x4 dot samples into Braille characters while retaining the
        dominant dot colour (so eye/flash dots stay distinct from body dots)."""
        dw, dh = self.dot_w, self.dot_h
        rows = []
        for cy in range(self.view_rows):
            parts = []
            last_col = None
            for cx in range(self.term_w):
                bits = 0
                best_col, best_pri = None, -1
                for dy in range(4):
                    y = cy * 4 + dy
                    if y >= dh:
                        continue
                    for dx in range(2):
                        x = cx * 2 + dx
                        if x >= dw:
                            continue
                        pix = self.pixels[y * dw + x]
                        if pix is not None:
                            bits |= BRAILLE_BITS[dx][dy]
                            col, pri = pix
                            if pri > best_pri:
                                best_col, best_pri = col, pri
                if bits == 0:
                    parts.append(" ")
                else:
                    if best_col != last_col:
                        parts.append("\x1b[38;5;%dm" % best_col)
                        last_col = best_col
                    parts.append(chr(0x2800 + bits))
            parts.append("\x1b[0m")
            rows.append("".join(parts))
        return rows

    def hud(self):
        w = self.term_w
        hp_col = 46 if self.health > 45 else 196
        ammo = "RLD" if self.reload > 0 else "%2d" % self.ammo
        line1 = ("\x1b[48;5;16m\x1b[38;5;%dmHP %3d  "
                 "\x1b[38;5;51mAMMO %s  "
                 "\x1b[38;5;226mKILLS %d/%d  "
                 "\x1b[38;5;201mMAP %d"
                 % (hp_col, self.health, ammo, self.kills,
                    len(self.enemies), self.round))
        mode = "AUTOPLAY PATROL + AUTO-AIM" if self.autoplay else "WASD MOVE  SPACE FIRE"
        line2 = "\x1b[48;5;16m\x1b[38;5;244m%s   CTRL-C QUIT" % mode
        return [line1, line2]

    def render(self):
        self.pixels = [None] * (self.dot_w * self.dot_h)
        self.draw_scene()
        self.draw_sprites()
        self.draw_weapon()
        lines = self.pack_cells() + self.hud()
        return "\x1b[H" + "\x1b[48;5;16m" + "\n".join(lines)


def main():
    autoplay = "--autoplay" in sys.argv
    game = Game(autoplay)
    try:
        w, h = shutil.get_terminal_size((54, 17))
        game.set_size(w, h)
    except Exception:
        pass

    old_term = None
    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    if fd is not None:
        try:
            old_term = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            old_term = None

    def stop(_signum=None, _frame=None):
        game.running = False

    def resize(_signum=None, _frame=None):
        game.resized = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGWINCH, resize)

    out = sys.stdout
    out.write("\x1b[?25l\x1b[2J")
    out.flush()
    last = time.monotonic()
    try:
        while game.running:
            now = time.monotonic()
            dt = min(0.12, max(0.001, now - last))
            last = now
            if game.resized:
                game.resized = False
                try:
                    w, h = shutil.get_terminal_size((game.term_w, game.term_h))
                except Exception:
                    w, h = game.term_w, game.term_h
                if w != game.term_w or h != game.term_h:
                    game.set_size(w, h)
                    out.write("\x1b[2J")
            if fd is not None:
                r, _, _ = select.select([fd], [], [], 0.0)
                if r:
                    try:
                        data = os.read(fd, 64)
                        game.keys += data.decode("utf-8", "ignore")
                    except OSError:
                        pass
            game.update(dt)
            out.write(game.render())
            out.flush()
            elapsed = time.monotonic() - now
            time.sleep(max(0.0, 1.0 / 30.0 - elapsed))
    finally:
        out.write("\x1b[0m\x1b[?25h\x1b[2J\x1b[H")
        out.flush()
        if old_term is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            except Exception:
                pass


if __name__ == "__main__":
    main()
