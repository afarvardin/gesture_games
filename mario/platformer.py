"""Gesture-controlled Super Mario Bros.-style platformer: four worlds, four levels
each, in the shape of the original Worlds 1 to 4.

  1-1 overworld   1-2 underground   1-3 athletic    1-4 castle
  2-1 overworld   2-2 underwater    2-3 bridges     2-4 castle
  3-1 night       3-2 night         3-3 lifts       3-4 firebars
  4-1 buzzies     4-2 beanstalk     4-3 athletic    4-4 maze castle

Level data and geometry validation live in levels.py; this file is the engine.

Steering maps a fixed band of the camera frame straight onto speed: the band's
right edge is full speed forward, its left edge is full speed back, its middle is
stopped. The band is drawn on the camera preview. Nothing is calibrated, so it is
in the same place every session. --steer-zone LO HI moves it.

Your LEFT hand steers and your RIGHT hand throws, decided by hand detection rather
than by which side of the frame you hold them. If reversed, press H.

  Steer hand   lean left/right to walk, lean further to run,
               flick UP to jump (height scales with flick speed),
               flick up again in mid-air to double jump,
               drop the hand low to duck -- and to enter a pipe.
  Fire hand    sweep the hand quickly to throw a fireball.

Grab a mushroom to grow: big Mario takes a hit before dying and smashes bricks from
below. Green pipes with a dark mouth can be entered by ducking on them.

Deliberately forgiving, because ~50-80ms of photon-to-pixel latency is unavoidable:
coyote time, jump buffering, ledge assist, generous stomp windows.

Keyboard fallback is always live: arrows, shift=run, space=jump, X=fire, down=duck.

  python platformer.py [--world 1-4] [--level 1-4] [--no-camera] [--camera N]
                       [--steer-zone LO HI] [--swap-hands] [--log [PATH]]
"""

import argparse
import collections
import csv
import math
import os
import sys
import time

import pygame

import levels as LV
import two_hand_spike as tracker_mod
from two_hand_spike import (
    LOG_FIELDS, RUN_THRESHOLD, STEER_DEADZONE, STEER_ZONE, TwoHandTracker, clamp,
    tuning_meta,
)

STEER_ZONE_DEFAULT = STEER_ZONE

# ==========================================
# WINDOW / WORLD
# ==========================================
WIDTH, HEIGHT = 1280, 720
FPS = 60
TILE = 40
HUD_H = 64
CAM_PREVIEW = (240, 180)

# Physics, in pixels and frames at 60fps. Unchanged from the tuned version.
GRAVITY = 0.58
MAX_FALL = 14.0
WALK_SPEED = 3.1
RUN_SPEED = 6.2
DUCK_FACTOR = 0.35
AIR_CONTROL = 0.85
JUMP_MIN, JUMP_MAX = 11.2, 16.4
DOUBLE_JUMP_FACTOR = 0.88
LEDGE_SNAP = 14
COYOTE_TIME = 0.15
JUMP_BUFFER = 0.15
STOMP_BOUNCE = 8.5

# Water levels: you sink slowly and every flick is a swim stroke.
WATER_GRAVITY = 0.16
WATER_MAX_FALL = 3.2
WATER_STROKE = 5.6
WATER_DRAG = 0.72

FIREBALL_SPEED = 8.0
FIREBALL_GRAVITY = 0.30
FIREBALL_BOUNCE = -5.2
FIREBALL_LIFE = 2.6
MAX_FIREBALLS = 4
ENEMY_SPEED = 1.15
KOOPA_SPEED = 1.45
SHELL_SPEED = 7.0
CHEEP_SPEED = 1.7
LEAP_INTERVAL = 2.4
LEAP_SPEED = -13.0
PODOBOO_INTERVAL = 3.6            # long enough that the clear window
                                  # is wider than a crossing takes
PODOBOO_SPEED = -13.5

LIFT_SPEED = 1.5                  # px/frame for moving platforms
PARA_AMPLITUDE = 34               # px a paratroopa rises and falls
PARA_PERIOD = 1.9                 # s of one flap cycle
PARA_RANGE = 6 * TILE             # px either side of home a flyer patrols. Left
                                  # unbounded they drift the whole level and clip
                                  # you mid-jump 30 tiles from where they started,
                                  # which makes a level unpredictable to design for.
HAMMER_INTERVAL = 2.8             # s between thrown hammers. 1.9 gave a walking
                                  # player no window to pass underneath at all.
HAMMER_VX, HAMMER_VY = 4.6, -8.0
HAMMER_LIFE = 3.0
BULLET_SPEED = 3.4                # px/frame a Bullet Bill travels
BULLET_INTERVAL = 3.1             # s between shots from one cannon
BULLET_RANGE = 22 * TILE          # px in front of the cannon it bothers to fire
BULLET_LIFE = 6.0                 # s before a bullet gives up
LAKITU_INTERVAL = 3.4             # s between spinies dropped
SPINY_LEAD = 3.5 * TILE           # px ahead of the player a spiny is thrown
BOWSER_HP = 5                     # fireballs to finish him
BOWSER_HOP = -9.5                 # he hops rather than walks
BOWSER_HOP_EVERY = 2.2            # s
BOWSER_BREATH_EVERY = 2.7         # s
BREATH_SPEED = 5.2                # px/frame; his fire flies flat and does not drop
BREATH_LIFE = 0.85                # short range on purpose: at 2.2s his fire flew
                                  # 17 tiles and hit players who could not yet see
                                  # him, which is not a fight, just an ambush
FIREBAR_RATE = 1.15               # radians/second
FIREBAR_SEG = 22                  # px between fire segments

MUSHROOM_SPEED = 1.9
GROW_PAUSE = 0.35                 # s of freeze while growing/shrinking
HURT_INVULN = 1.6                 # s of invulnerability after being shrunk
BUMP_TIME = 0.18                  # s of block bump animation
RESPAWN_PAUSE = 1.1
START_LIVES = 5
PIPE_TRAVEL = 0.7                 # s spent sliding into a pipe

# ==========================================
# THEMES -- each level looks like where it is
# ==========================================
THEMES = {
    LV.THEME_OVERWORLD: dict(
        sky_top=(92, 148, 252), sky_bottom=(150, 196, 255),
        ground=(150, 90, 44), ground_top=(86, 172, 62),
        brick=(196, 92, 44), brick_line=(140, 58, 26),
        stone=(180, 180, 190), block=(232, 178, 62), block_edge=(150, 104, 20),
        hills=True, clouds=True, stars=False, night=False,
    ),
    LV.THEME_NIGHT: dict(
        sky_top=(4, 6, 30), sky_bottom=(16, 22, 62),
        ground=(92, 62, 112), ground_top=(152, 92, 172),
        brick=(120, 80, 140), brick_line=(70, 44, 90),
        stone=(150, 150, 170), block=(232, 178, 62), block_edge=(150, 104, 20),
        hills=True, clouds=False, stars=True, night=True,
    ),
    LV.THEME_UNDERGROUND: dict(
        sky_top=(0, 0, 0), sky_bottom=(6, 8, 24),
        ground=(0, 118, 160), ground_top=(0, 168, 200),
        brick=(0, 128, 168), brick_line=(0, 78, 110),
        stone=(120, 160, 180), block=(232, 178, 62), block_edge=(150, 104, 20),
        hills=False, clouds=False, stars=False, night=False,
    ),
    LV.THEME_CASTLE: dict(
        sky_top=(0, 0, 0), sky_bottom=(24, 10, 10),
        ground=(120, 120, 128), ground_top=(160, 160, 170),
        brick=(148, 148, 156), brick_line=(96, 96, 104),
        stone=(96, 96, 104), block=(200, 150, 60), block_edge=(120, 84, 16),
        hills=False, clouds=False, stars=False, night=False,
    ),
    LV.THEME_WATER: dict(
        sky_top=(0, 60, 140), sky_bottom=(0, 108, 190),
        ground=(190, 160, 90), ground_top=(210, 190, 120),
        brick=(0, 128, 168), brick_line=(0, 78, 110),
        stone=(150, 170, 190), block=(232, 178, 62), block_edge=(150, 104, 20),
        hills=False, clouds=False, stars=True, night=False,
    ),
}

C_PIPE = (26, 168, 74)
C_PIPE_DARK = (14, 108, 48)
C_PIPE_MOUTH = (10, 60, 28)
C_COIN = (252, 206, 70)
C_COIN_EDGE = (188, 138, 20)
C_FIRE_A = (255, 156, 40)
C_FIRE_B = (255, 232, 130)
C_FLAG = (240, 240, 240)
C_FLAG_POLE = (72, 72, 84)
C_LAVA_A = (232, 70, 20)
C_LAVA_B = (255, 150, 40)
C_WATER = (40, 120, 210)
C_BRIDGE = (170, 110, 50)
C_TEXT = (255, 255, 255)
C_HUD_BG = (18, 20, 30)

# ==========================================
# SPRITES
# ==========================================
HERO_SMALL = [
    "..RRRR..",
    ".RRRRRR.",
    ".SSKSK..",
    ".SSSSSS.",
    "..RRRR..",
    ".RTRRTR.",
    ".TT..TT.",
    ".bb..bb.",
]
HERO_BIG = [
    "...RRRR...",
    "..RRRRRR..",
    "..SSKSK...",
    "..SSSSSS..",
    "...SSSS...",
    "..RRRRRR..",
    ".RRTRRTRR.",
    "RR.TTTT.RR",
    "...TTTT...",
    "...TTTT...",
    "...T..T...",
    "..bb..bb..",
    ".bbb..bbb.",
]
GOOMBA_PIXELS = [
    "..EEEE..",
    ".EEEEEE.",
    "EEWWWWEE",
    "EEWKWKEE",
    "EEEEEEEE",
    ".EEEEEE.",
    ".FF..FF.",
    "FFF..FFF",
]
KOOPA_PIXELS = [
    "...GGG..",
    "..GWKG..",
    "...GGG..",
    "..YYYY..",
    ".YGGGGY.",
    ".YGGGGY.",
    "..YYYY..",
    "..F..F..",
]
SHELL_PIXELS = [
    "........",
    "..YYYY..",
    ".YGGGGY.",
    "YGGGGGGY",
    "YGGGGGGY",
    ".YGGGGY.",
    "..YYYY..",
    "........",
]
CHEEP_PIXELS = [
    "........",
    "..RRRR..",
    ".RRWKRR.",
    "RRRRRRRO",
    "RRRRRRRO",
    ".RRRRRR.",
    "..RRRR..",
    "........",
]
PARA_PIXELS = [
    "..GGG.WW",
    ".GWKG.WW",
    "..GGG.WW",
    ".YYYYWW.",
    "YGGGGY..",
    "YGGGGY..",
    ".YYYY...",
    ".F..F...",
]
BUZZY_PIXELS = [
    "........",
    "..KKKK..",
    ".KBBBBK.",
    "KBBBBBBK",
    "KBBWKWBK",
    ".KBBBBK.",
    "..KKKK..",
    ".F....F.",
]
HAMMER_BRO_PIXELS = [
    "..GGGG..",
    ".GWKWKG.",
    "..GGGG..",
    ".YHHHHY.",
    "HYGGGGYH",
    ".YGGGGY.",
    "..YYYY..",
    ".FF..FF.",
]
SPINY_PIXELS = [
    "K.K.K.K.",
    ".KKKKKK.",
    "KOOOOOOK",
    "KOWKWKOK",
    ".OOOOOO.",
    "..OOOO..",
    ".F....F.",
    "........",
]
LAKITU_PIXELS = [
    "..WWWW..",
    ".WWWWWW.",
    "WW.GG.WW",
    "WWGWKGWW",
    ".WGGGGW.",
    "..WWWW..",
    "...GG...",
    "..G..G..",
]
MUSHROOM_PIXELS = [
    "..MMMM..",
    ".MWWMWM.",
    "MWWMMWWM",
    "MMMMMMMM",
    ".SSSSSS.",
    ".SKSSKS.",
    ".SSSSSS.",
    "..SSSS..",
]
PIXEL_COLORS = {
    "R": (224, 52, 48), "S": (250, 200, 156), "T": (46, 82, 200),
    "K": (20, 20, 30), "E": (168, 84, 36),
    "F": (72, 40, 20), "W": (250, 250, 250), "G": (60, 180, 70),
    "Y": (240, 210, 90), "M": (228, 60, 50), "O": (250, 170, 60),
    "B": (60, 60, 90),          # buzzy beetle shell
    "H": (190, 190, 200),       # hammer
    "b": (86, 44, 20),          # boots
}


def build_sprite(pixels, scale, flip=False):
    w, h = len(pixels[0]), len(pixels)
    surf = pygame.Surface((w * scale, h * scale), pygame.SRCALPHA)
    for row, line in enumerate(pixels):
        for col, ch in enumerate(line):
            if ch == ".":
                continue
            c = col if not flip else (w - 1 - col)
            surf.fill(PIXEL_COLORS[ch], (c * scale, row * scale, scale, scale))
    return surf


# ==========================================
# ENTITIES
# ==========================================
class Fireball:
    R = 7

    def __init__(self, x, y, vx, born):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = vx, 2.0
        self.born = born
        self.alive = True

    def update(self, level, now):
        if now - self.born > FIREBALL_LIFE:
            self.alive = False
            return
        self.vy = min(self.vy + FIREBALL_GRAVITY, MAX_FALL)
        self.x += self.vx
        self.y += self.vy
        if level.solid_at(self.x, self.y + self.R):
            self.y = math.floor((self.y + self.R) / TILE) * TILE - self.R
            self.vy = FIREBALL_BOUNCE
        if level.solid_at(self.x + math.copysign(self.R, self.vx), self.y):
            self.alive = False
        if not (0 <= self.x <= level.pw) or self.y > level.ph + 100:
            self.alive = False

    def rect(self):
        return pygame.Rect(int(self.x - self.R), int(self.y - self.R),
                           self.R * 2, self.R * 2)


class Hammer:
    """A thrown hammer, or -- flat=True -- a jet of Bowser's fire."""
    R = 9

    def __init__(self, x, y, vx, born, flat=False):
        self.x, self.y = float(x), float(y)
        self.flat = flat
        self.vx, self.vy = vx, 0.0 if flat else HAMMER_VY
        self.born = born
        self.alive = True
        self.spin = 0.0

    def rect(self):
        return pygame.Rect(int(self.x - self.R), int(self.y - self.R),
                           self.R * 2, self.R * 2)

    def update(self, level, now):
        if now - self.born > (BREATH_LIFE if self.flat else HAMMER_LIFE):
            self.alive = False
            return
        if not self.flat:
            self.vy = min(self.vy + GRAVITY, MAX_FALL)
        self.x += self.vx
        self.y += self.vy
        self.spin += 0.5
        if self.y > level.ph + 120 or not (-40 <= self.x <= level.pw + 40):
            self.alive = False


class Mushroom:
    W = H = 30

    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = MUSHROOM_SPEED, 0.0
        self.alive = True
        self.emerging = 0.45          # s rising out of the block

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y - self.H), self.W, self.H)

    def update(self, level, dt):
        if self.emerging > 0:
            self.emerging -= dt
            self.y -= 34 * dt         # slide up out of the block
            return
        self.vy = min(self.vy + GRAVITY, MAX_FALL)
        self.x += self.vx
        r = self.rect()
        if level.solid_at(r.right if self.vx > 0 else r.left, r.centery):
            self.vx = -self.vx
            self.x += self.vx * 2
        self.y += self.vy
        r = self.rect()
        if level.solid_at(r.centerx, r.bottom):
            self.y = (r.bottom // TILE) * TILE
            self.vy = 0.0
        if self.y > level.ph + 120:
            self.alive = False


class Enemy:
    """Goomba, koopa, shell, cheep-cheep, leaper or podoboo.

    One class with a kind rather than six: they differ only in how they move and
    what a stomp does, and keeping the collision handling in one place is what
    stops "stomp works on goombas but not koopas" bugs.
    """
    W, H = 32, 32

    def __init__(self, x, y, kind, now=0.0):
        self.x, self.y = float(x), float(y)
        # Lifts and firebars carry parameters in their kind string.
        self.span, self.width, self.length, self.face = 4, 3, 4, -1
        if ":" in kind:
            parts = kind.split(":")
            kind = parts[0]
            if kind == "firebar":
                self.length = int(parts[1])
            elif kind == "cannon":
                self.face = int(parts[1])
            else:
                self.span, self.width = int(parts[1]), int(parts[2])
        self.kind = kind
        self.angle = 0.0
        self.dx = self.dy = 0.0          # movement this frame, for riders
        self.alive = True
        self.dying = 0.0
        self.shell = False
        self.throw_now = False
        self.target_x = float(x)
        self.hp = BOWSER_HP if kind == "bowser" else 1
        self.vy = 0.0
        # A podoboo at rest sits BELOW the lava surface. Resting it at floor level
        # left a permanent hazard parked on the lip of the pit you have to land on.
        self.home_y = float(y) + (TILE if kind == "podoboo" else 0)
        self.timer = now + (x % 17) * 0.11        # stagger, without randomness
        self.breath_at = self.timer + 1.1        # offset, so hop and breath differ
        speed = {"goomba": ENEMY_SPEED, "koopa": KOOPA_SPEED,
                 "buzzy": ENEMY_SPEED, "cheep": CHEEP_SPEED,
                 "spiny": ENEMY_SPEED * 1.2}.get(kind, 0.0)
        self.vx = -speed
        self.home_x = float(x)
        self.lift_dir = 1

    @property
    def submerged(self):
        """A podoboo is harmless (and invisible) while down in the lava."""
        return self.kind == "podoboo" and self.y >= self.home_y - 4

    @property
    def stompable(self):
        # A spiny has spikes on its back: fire is the only way past it, which is the
        # mirror image of the buzzy beetle shrugging off fire.
        return self.kind in ("goomba", "koopa", "cheep", "buzzy", "para", "hammer",
                             "bullet")   # not spiny (spikes), not bowser (armoured)

    @property
    def fireproof(self):
        """Buzzy beetles shrug off fireballs -- that is the whole point of them."""
        return self.kind in ("buzzy", "podoboo", "firebar", "lift_v", "lift_h",
                             "cannon", "lakitu")

    @property
    def armoured(self):
        """Bowser takes several fireballs, and cannot be stomped at all."""
        return self.kind == "bowser"

    @property
    def is_platform(self):
        return self.kind in ("lift_v", "lift_h")

    @property
    def harmless(self):
        # A stomped shell just sits there. In the original you kick it, and it
        # comes back and kills you -- at 7px/frame it overtakes a walking player
        # from behind with no warning, which is a miserable way to lose a life.
        # Here a shell is scenery until you stomp it again.
        return self.is_platform or self.submerged or self.shell

    def platform_rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.width * TILE, 12)

    def firebar_points(self):
        """Centres of the fire segments, from the pivot outward."""
        cx, cy = self.x + TILE * 0.5, self.y + TILE * 0.5
        return [(cx + math.cos(self.angle) * FIREBAR_SEG * i,
                 cy + math.sin(self.angle) * FIREBAR_SEG * i)
                for i in range(1, self.length + 1)]

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y - self.H), self.W, self.H)

    def update(self, level, dt, now):
        if self.dying > 0:
            self.dying -= dt
            if self.dying <= 0:
                self.alive = False
            return

        if self.is_platform:
            # Lifts shuttle between two points; riders are moved by the game using
            # dx/dy, so the platform stays the single source of truth.
            before_x, before_y = self.x, self.y
            reach = self.span * TILE
            if self.kind == "lift_v":
                self.y += LIFT_SPEED * self.lift_dir
                if abs(self.y - self.home_y) > reach:
                    self.lift_dir = -self.lift_dir
                    self.y = clamp(self.y, self.home_y - reach, self.home_y + reach)
            else:
                self.x += LIFT_SPEED * self.lift_dir
                if abs(self.x - self.home_x) > reach:
                    self.lift_dir = -self.lift_dir
                    self.x = clamp(self.x, self.home_x - reach, self.home_x + reach)
            self.dx, self.dy = self.x - before_x, self.y - before_y
            return

        if self.kind == "firebar":
            self.angle += FIREBAR_RATE * dt
            return

        if self.kind == "cannon":
            if now >= self.timer:
                self.timer = now + BULLET_INTERVAL
                self.throw_now = True      # the game spawns the bullet
            return

        if self.kind == "bullet":
            # Bullet Bills ignore gravity and terrain; they just keep coming.
            self.x += self.vx
            if now - self.home_y > BULLET_LIFE or not (-60 <= self.x <= level.pw + 60):
                self.alive = False
            return

        if self.kind == "spiny":
            # Fall to the ground first. Without this a dropped spiny hovered at the
            # height Lakitu released it, which is exactly where a jumping player's
            # head goes -- an invisible mine at head height.
            if not level.solid_at(self.x + self.W * 0.5, self.y + 2):
                self.vy = min(self.vy + GRAVITY, MAX_FALL)
                self.y += self.vy
                if self.y > level.ph + 120:
                    self.alive = False
                return
            if self.vy > 0:
                self.y = float(int(self.y // TILE) * TILE)   # sit on the tile, not in it
            self.vy = 0.0
            self.x += self.vx
            ahead = self.x + (self.W if self.vx > 0 else 0) + math.copysign(2, self.vx)
            if level.solid_at(ahead, self.y - self.H * 0.5) or \
                    not level.solid_at(ahead, self.y + 4):
                self.vx = -self.vx
                self.x += self.vx * 2
            return

        if self.kind == "bowser":
            self.vy = min(self.vy + GRAVITY, MAX_FALL)
            self.y += self.vy
            if self.y >= self.home_y:
                self.y, self.vy = self.home_y, 0.0
                if now >= self.timer:
                    self.timer = now + BOWSER_HOP_EVERY
                    self.vy = BOWSER_HOP
            if now >= self.breath_at:
                self.breath_at = now + BOWSER_BREATH_EVERY
                self.throw_now = True
            return

        if self.kind == "lakitu":
            # Hovers over the player and lobs spinies down at them.
            target = self.target_x
            self.x += clamp((target - self.x) * 0.02, -2.6, 2.6)
            self.y = self.home_y + math.sin(now * 1.4) * 12
            if now >= self.timer:
                self.timer = now + LAKITU_INTERVAL
                self.throw_now = True
            return

        if self.kind == "para":
            # Flies a slow up-and-down path; a stomp knocks the wings off.
            self.x += self.vx if self.vx else -ENEMY_SPEED
            self.y = self.home_y + math.sin(
                (now - self.timer) * (2 * math.pi / PARA_PERIOD)) * PARA_AMPLITUDE
            if (abs(self.x - self.home_x) > PARA_RANGE
                    or self.x < 8 or self.x > level.pw - 40):
                self.vx = -(self.vx or -ENEMY_SPEED)
                self.x = clamp(self.x, self.home_x - PARA_RANGE,
                               self.home_x + PARA_RANGE)
            return

        if self.kind == "hammer":
            # Hops on the spot and lobs hammers. Standing still and throwing is
            # what makes it a wall you have to time rather than just run past.
            if now >= self.timer:
                self.timer = now + HAMMER_INTERVAL
                self.throw_now = True
            self.y = self.home_y - abs(math.sin(now * 2.2)) * 16
            return

        if self.kind == "podoboo":
            # Rises out of the lava and falls back, forever.
            if now >= self.timer:
                self.vy = PODOBOO_SPEED
                self.timer = now + PODOBOO_INTERVAL
            self.vy = min(self.vy + GRAVITY, MAX_FALL)
            self.y += self.vy
            if self.y > self.home_y:
                self.y, self.vy = self.home_y, 0.0
            return

        if self.kind == "leaper":
            # Sits under the bridge and jumps up through it.
            if self.vy == 0.0 and now >= self.timer:
                self.vy = LEAP_SPEED
                self.timer = now + LEAP_INTERVAL
            if self.vy != 0.0:
                self.vy = min(self.vy + GRAVITY, MAX_FALL)
                self.y += self.vy
                if self.y > self.home_y:
                    self.y, self.vy = self.home_y, 0.0
            return

        if self.kind == "cheep":
            self.x += self.vx
            self.y += math.sin((now + self.x * 0.02) * 1.6) * 0.9
            if (abs(self.x - self.home_x) > PARA_RANGE
                    or self.x < 8 or self.x > level.pw - 40):
                self.vx = -self.vx
                self.x = clamp(self.x, self.home_x - PARA_RANGE,
                               self.home_x + PARA_RANGE)
            return

        # Walkers: goomba, koopa, and a kicked shell.
        speed = SHELL_SPEED if self.shell else abs(self.vx)
        step = math.copysign(speed, self.vx) if self.vx else 0.0
        self.x += step
        if level.water_level:
            return
        ahead = self.x + (self.W if step > 0 else 0) + math.copysign(2, step or 1)
        blocked = level.solid_at(ahead, self.y - self.H * 0.5)
        if blocked and self.shell and abs(self.vx) > 0.1:
            self.dying = 0.2          # a kicked shell spends itself on the wall
            return                    # rather than rebounding into the player
        if blocked or not level.solid_at(ahead, self.y + 4):
            self.vx = -self.vx
            self.x += self.vx * 2

    def stomped(self):
        """Returns True if the stomp killed it outright."""
        if self.kind == "para":
            self.kind = "koopa"           # wings off, now an ordinary koopa
            self.vx = -KOOPA_SPEED
            self.home_y = self.y
            return False
        if self.kind in ("koopa", "buzzy") and not self.shell:
            self.shell = True                 # first stomp: into the shell
            self.vx = 0.0
            return False
        self.dying = 0.25
        return True

    def kick(self, direction):
        """Kept for completeness; the game no longer kicks shells at the player."""
        self.vx = math.copysign(SHELL_SPEED, direction)


class Hero:
    W = 30
    H_SMALL, H_BIG = 38, 72
    DUCK_SMALL, DUCK_BIG = 26, 40

    def __init__(self, x, y):
        self.spawn = (float(x), float(y))
        self.big = False
        self.reset()

    def reset(self, keep_power=False):
        self.x, self.y = self.spawn
        self.vx = self.vy = 0.0
        self.facing = 1
        self.grounded = True
        self.jumps_used = 0
        self.ducking = False
        self.last_ground_t = -99.0
        self.buffered_jump = None
        self.walk_phase = 0.0
        self.was_falling = False     # descending BEFORE this frame's landing
        self.was_airborne = False    # and airborne before it, too
        self.grow_until = 0.0
        self.invuln_until = 0.0
        self.pipe_until = 0.0
        if not keep_power:
            self.big = False

    @property
    def height(self):
        if self.ducking:
            return self.DUCK_BIG if self.big else self.DUCK_SMALL
        return self.H_BIG if self.big else self.H_SMALL

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y - self.height), self.W,
                           self.height)

    def head_rect(self):
        r = self.rect()
        return pygame.Rect(r.x + 4, r.y - 2, r.w - 8, 6)

    def request_jump(self, strength, now, water=False):
        """Apply a jump flick. In water every stroke works, mid-stroke or not."""
        if now < self.grow_until or now < self.pipe_until:
            return "BUSY"
        if water:
            self.vy = -WATER_STROKE
            self.grounded = False
            return "STROKE"
        coyote = self.jumps_used == 0 and (now - self.last_ground_t) <= COYOTE_TIME
        if self.grounded or coyote:
            self.vy = -(JUMP_MIN + (JUMP_MAX - JUMP_MIN) * strength)
            self.grounded = False
            self.jumps_used = 1
            return "JUMP"
        if self.jumps_used < 2:
            self.vy = -(JUMP_MIN + (JUMP_MAX - JUMP_MIN) * strength) * DOUBLE_JUMP_FACTOR
            self.jumps_used = 2
            return "DOUBLE"
        self.buffered_jump = (now, strength)
        return "BUFFERED"

    def grow(self, now):
        if self.big:
            return False
        self.big = True
        self.y += 0                       # feet stay put; the rect grows upward
        self.grow_until = now + GROW_PAUSE
        return True

    def shrink(self, now):
        self.big = False
        self.grow_until = now + GROW_PAUSE
        self.invuln_until = now + HURT_INVULN


# ==========================================
# LEVEL
# ==========================================
class Level:
    """A built level from levels.py, plus the mutable state of its blocks."""

    def __init__(self, data):
        self.data = data
        self.name = data["name"]
        self.theme = data["theme"]
        self.rows = [list(r) for r in data["rows"]]
        self.cols = len(self.rows[0])
        self.pw, self.ph = self.cols * TILE, len(self.rows) * TILE
        self.solids = set(LV.SOLID)
        self.pipe_links = dict(data["pipe_links"])
        self.end_kind, self.end_col = data["end"]
        self.water_level = data["theme"] == LV.THEME_WATER
        self.ceiling = data["ceiling"]
        self.loops = list(data.get("loops", []))
        self.bumps = {}                    # (col,row) -> seconds left of bump
        self.start = (data["start_col"] * TILE, LV.FLOOR * TILE)

    def tile_at(self, col, row):
        if 0 <= row < len(self.rows) and 0 <= col < self.cols:
            return self.rows[row][col]
        return LV.EMPTY

    def solid_at(self, px, py):
        if px < 0 or px > self.pw:
            return True                    # the level edges are walls
        return self.tile_at(int(px // TILE), int(py // TILE)) in self.solids

    def deadly_at(self, px, py):
        return self.tile_at(int(px // TILE), int(py // TILE)) in LV.DEADLY

    def water_at(self, px, py):
        return self.tile_at(int(px // TILE), int(py // TILE)) == LV.WATER

    def take_coin(self, col, row):
        if self.tile_at(col, row) == LV.COIN:
            self.rows[row][col] = LV.EMPTY
            return True
        return False

    def hit_block(self, col, row, big):
        """Bump a block from below. Returns what came out."""
        ch = self.tile_at(col, row)
        if ch not in (LV.QCOIN, LV.QMUSH, LV.BRICK):
            return None
        self.bumps[(col, row)] = BUMP_TIME
        if ch == LV.QCOIN:
            self.rows[row][col] = LV.USED
            return "coin"
        if ch == LV.QMUSH:
            self.rows[row][col] = LV.USED
            return "mushroom"
        if big:                            # only big Mario breaks bricks
            self.rows[row][col] = LV.EMPTY
            return "break"
        return "bump"

    def pipe_dest(self, col):
        """Where the pipe mouth at this column leads, if anywhere."""
        for c in (col, col - 1):
            if c in self.pipe_links:
                return self.pipe_links[c]
        return None

    def enterable_at(self, px, py):
        """The column of a pipe mouth under these coordinates, or None."""
        col, row = int(px // TILE), int(py // TILE)
        if self.tile_at(col, row) in (LV.PIPE_IN, LV.EXIT):
            return col
        return None

    def vine_at(self, rect):
        """The column of a beanstalk this rect is touching, or None."""
        for col in range(rect.left // TILE, rect.right // TILE + 1):
            for row in range(rect.top // TILE, rect.bottom // TILE + 1):
                if self.tile_at(col, row) == LV.VINE:
                    return col
        return None


# ==========================================
# GAME
# ==========================================
class Game:
    def __init__(self, world=1, level=1, use_camera=True, camera_id=0,
                 log_path=None):
        pygame.init()
        pygame.display.set_caption("Gesture Platformer")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = self._mono(20, bold=True)
        self.small = self._mono(15)
        self.big_font = self._mono(44, bold=True)

        self.spr = {
            "hero_s": build_sprite(HERO_SMALL, 5),
            "hero_s_l": build_sprite(HERO_SMALL, 5, flip=True),
            "hero_b": build_sprite(HERO_BIG, 6),
            "hero_b_l": build_sprite(HERO_BIG, 6, flip=True),
            "goomba": build_sprite(GOOMBA_PIXELS, 4),
            "koopa": build_sprite(KOOPA_PIXELS, 4),
            "shell": build_sprite(SHELL_PIXELS, 4),
            "cheep": build_sprite(CHEEP_PIXELS, 4),
            "leaper": build_sprite(CHEEP_PIXELS, 4),
            "mushroom": build_sprite(MUSHROOM_PIXELS, 4),
            "para": build_sprite(PARA_PIXELS, 4),
            "buzzy": build_sprite(BUZZY_PIXELS, 4),
            "hammer": build_sprite(HAMMER_BRO_PIXELS, 4),
            "spiny": build_sprite(SPINY_PIXELS, 4),
            "lakitu": build_sprite(LAKITU_PIXELS, 4),
        }
        self.sky_cache = {}

        self.log_path = log_path
        self._csv_file = self._csv = None
        if log_path:
            self._csv_file = open(log_path, "w", newline="")
            self._csv = csv.DictWriter(self._csv_file, fieldnames=LOG_FIELDS,
                                       restval="", extrasaction="ignore")
            self._csv.writeheader()
            self._csv.writerow({"kind": "meta", "t": 0, "note": tuning_meta()})
            print(f"[Log] writing {log_path}")
        self.tracker = (TwoHandTracker(camera_id=camera_id,
                                       logging_on=bool(log_path))
                        if use_camera else None)

        self.lives = START_LIVES
        self.score = 0
        self.coins_taken = 0
        self.world, self.level_no = world, level
        self.load_level(world, level)
        self.state = "play"
        self.state_until = 0.0
        self.toast = ""
        self.toast_until = 0.0
        self.log = collections.deque(maxlen=6)

    @staticmethod
    def _mono(size, bold=False):
        for name in ("menlo", "dejavusansmono", "consolas", "couriernew"):
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f:
                    return f
            except Exception:
                continue
        return pygame.font.Font(None, size + 4)

    # ---------- level plumbing ----------
    def load_level(self, world, level, keep_power=False):
        self.world, self.level_no = world, level
        self.main = Level(LV.load(world, level))
        self.bonus = None
        self.in_bonus = False
        self.level = self.main
        self.hero = Hero(*self.main.start)
        self.hero.big = keep_power and getattr(self, "hero", None) is not None \
            and self.hero.big
        self.spawn_entities()
        self.cam_x = 0.0
        self.level_start = time.perf_counter()
        self.return_col = None

    def spawn_entities(self, now=0.0):
        self.enemies = [Enemy(c * TILE + 4, (r + 1) * TILE, kind, now)
                        for c, r, kind in self.level.data["enemies"]]
        self.fireballs = []
        self.items = []
        self.hammers = []

    def enter_pipe(self, col, now):
        """Duck on a pipe mouth to travel."""
        dest = self.level.pipe_dest(col)
        if dest is None:
            return False
        self.hero.pipe_until = now + PIPE_TRAVEL
        self.state_until = now + PIPE_TRAVEL
        self.state = "piping"
        self._pipe_target = dest
        self._toast("down the pipe...", now, PIPE_TRAVEL)
        return True

    def finish_pipe(self, now):
        target = self._pipe_target
        if target in ("bonus", "sky"):
            data = LV.load_bonus(self.main.name)
            if data is None:
                self.state = "play"
                return
            self.bonus = Level(data)
            self.level = self.bonus
            self.in_bonus = True
            self.hero.spawn = self.bonus.start
        else:
            # Leaving a bonus room: rejoin the main level further along.
            self.level = self.main
            self.in_bonus = False
            self.hero.spawn = (target * TILE, LV.FLOOR * TILE)
        keep = self.hero.big
        self.hero.reset(keep_power=True)
        self.hero.big = keep
        self.spawn_entities(now)
        self.cam_x = clamp(self.hero.x - WIDTH * 0.42, 0,
                           max(0, self.level.pw - WIDTH))
        self.state = "play"

    def next_level(self, now):
        w, l = self.world, self.level_no + 1
        if l > len(LV.WORLDS[w - 1]):
            w, l = w + 1, 1
        if w > len(LV.WORLDS):
            self.state = "complete"
            self.state_until = now + 8.0
            return
        self.load_level(w, l, keep_power=True)
        self.state = "play"

    # ---------- main loop ----------
    def run(self):
        print(__doc__)
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            now = time.perf_counter()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_h and self.tracker:
                        sw = self.tracker.toggle_hand_polarity()
                        self._toast("hand detection swapped" if sw
                                    else "hand detection normal", now)
                    elif event.key == pygame.K_r:
                        self.restart(now)
                    elif event.key == pygame.K_n:
                        self.next_level(now)
                    elif event.key == pygame.K_SPACE and self.state == "play":
                        self._do_jump(self.hero.request_jump(
                            1.0, now, self.in_water()))
                    elif event.key == pygame.K_x and self.state == "play":
                        self.throw(now)

            analog, run_flag, duck = 0.0, False, False
            snap = None
            if self.tracker:
                snap, events = self.tracker.read()
                analog, run_flag, duck = snap["analog"], snap["run"], snap["duck"]
                if self._csv:
                    self._csv.writerows(self.tracker.drain_log())
                if self.state == "play":
                    for ev in events:
                        if ev["kind"] == "flick":
                            label = self._do_jump(self.hero.request_jump(
                                ev["strength"], now, self.in_water()))
                            self._csv_event(ev, now, label)
                        else:
                            self.throw(now)
                            self._csv_event(ev, now, "FIREBALL")

            keys = pygame.key.get_pressed()
            kdir = (1.0 if keys[pygame.K_RIGHT] else 0.0) - \
                   (1.0 if keys[pygame.K_LEFT] else 0.0)
            if kdir:
                analog = kdir
                run_flag = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            if keys[pygame.K_DOWN]:
                duck = True

            self.update(analog, run_flag, duck, dt, now)
            self.draw(snap, now)

        if self.tracker:
            if self._csv:
                self._csv.writerows(self.tracker.drain_log())
            self.tracker.close()
        if self._csv:
            self._csv_file.close()
            print(f"[Log] wrote {self.log_path}")
        pygame.quit()

    def in_water(self):
        h = self.hero
        return self.level.water_level and self.level.water_at(
            h.x + h.W * 0.5, h.y - h.height * 0.5)

    def _toast(self, text, now, secs=1.6):
        self.toast, self.toast_until = text, now + secs

    def _do_jump(self, label):
        self.log.append(label)
        return label

    def _csv_event(self, ev, now, outcome):
        if not self._csv:
            return
        self._csv.writerow({
            "kind": ev["kind"],
            "t": round(ev["t_capture"] - self.tracker.t0, 4),
            "lat_ms": round((now - ev["t_capture"]) * 1000, 2),
            "vel": round(ev.get("vel", 0.0), 3),
            "strength": round(ev["strength"], 3) if "strength" in ev else "",
            "outcome": outcome,
        })

    def throw(self, now):
        if self.state != "play":
            return
        if len([f for f in self.fireballs if f.alive]) >= MAX_FIREBALLS:
            return
        h = self.hero
        x = h.x + (h.W + 4 if h.facing > 0 else -4)
        self.fireballs.append(Fireball(x, h.y - h.height * 0.55,
                                       FIREBALL_SPEED * h.facing, now))

    def restart(self, now):
        self.lives = START_LIVES
        self.score = 0
        self.load_level(self.world, self.level_no)
        self.state, self.state_until = "play", 0.0

    # ---------- simulation ----------
    def update(self, analog, run_flag, duck, dt, now):
        if self.state == "piping":
            if now >= self.state_until:
                self.finish_pipe(now)
            return
        if self.state == "dead":
            if now >= self.state_until:
                if self.lives <= 0:
                    self.state = "gameover"
                else:
                    self.level = self.main
                    self.in_bonus = False
                    self.hero.spawn = self.main.start
                    self.hero.reset()
                    self.spawn_entities(now)
                    self.state = "play"
            return
        if self.state in ("won", "gameover", "complete"):
            if self.state == "won" and now >= self.state_until:
                self.next_level(now)
            return

        h = self.hero
        water = self.in_water()
        if now < h.grow_until or now < h.pipe_until:
            self.step_world(dt, now)          # the world keeps moving; Mario waits
            return

        h.ducking = duck and h.grounded and not water

        # Ducking on a pipe mouth enters it.
        if duck and h.grounded:
            col = self.level.enterable_at(h.x + h.W * 0.5, h.y + 4)
            if col is not None and self.enter_pipe(col, now):
                return

        target = analog * (RUN_SPEED if run_flag else WALK_SPEED)
        if h.ducking:
            target *= DUCK_FACTOR
        if not h.grounded:
            target *= AIR_CONTROL
        if water:
            target *= WATER_DRAG
        h.vx = target
        if abs(h.vx) > 0.1:
            h.facing = 1 if h.vx > 0 else -1
            h.walk_phase += abs(h.vx) * dt * 6
        self.move_x(h)

        # Standing on a lift overrides gravity for this frame.
        rode = self.carry_on_lifts(dt)

        grav = WATER_GRAVITY if water else GRAVITY
        cap = WATER_MAX_FALL if water else MAX_FALL
        h.vy = min(h.vy + grav, cap)
        # Remember that we were descending BEFORE move_y lands us and zeroes vy.
        # Collisions run after movement, so asking "are you falling?" there always
        # answered no -- which meant landing squarely on an enemy could never count
        # as a stomp and always killed you instead.
        h.was_falling = h.vy > 0
        if rode:
            h.vy = min(h.vy, 0.0)
        was_grounded = h.grounded
        # Standing on the floor still leaves vy positive (gravity is applied then
        # cancelled by the landing), so "descending" alone cannot separate a stomp
        # from a side bump. Coming out of the AIR while descending can.
        h.was_airborne = not was_grounded
        self.move_y(h, now)
        if h.grounded and not was_grounded and h.buffered_jump:
            bt, bs = h.buffered_jump
            if now - bt <= JUMP_BUFFER:
                h.request_jump(bs, now, water)
            h.buffered_jump = None
        if h.grounded:
            h.last_ground_t = now

        # Leaving a bonus area: walk off the far end, or fall out of it.
        if self.in_bonus and (h.x > self.level.pw - TILE * 1.5
                              or h.y > self.level.ph + 60):
            dest = None
            for col, target in self.level.pipe_links.items():
                dest = target
            self._pipe_target = dest if dest is not None else self.main.start[0] // TILE
            self.finish_pipe(now)
            self._toast("back to the level", now)
            return

        # Pits, lava.
        if h.y > self.level.ph + 80:
            self.kill(now, "fell")
            return
        r = h.rect()
        if self.level.deadly_at(r.centerx, r.bottom - 2):
            self.kill(now, "lava", force=True)
            return

        # A beanstalk carries you up to a sky area, the way a pipe takes you down.
        vine = self.level.vine_at(h.rect())
        if vine is not None:
            dest = self.level.pipe_dest(vine)
            if dest is not None:
                self._toast("up the beanstalk!", now, PIPE_TRAVEL)
                self.hero.pipe_until = now + PIPE_TRAVEL
                self.state_until = now + PIPE_TRAVEL
                self.state = "piping"
                self._pipe_target = dest
                return

        # 4-4's maze: crossing the trigger low sends you back to try the top route.
        for col, dest, above_row in self.level.loops:
            if abs(h.x - col * TILE) < 12 and h.y > above_row * TILE + TILE * 0.5:
                h.x = float(dest * TILE)
                self.cam_x = clamp(h.x - WIDTH * 0.42, 0,
                                   max(0, self.level.pw - WIDTH))
                self._toast("wrong way -- try the high road", now, 2.0)
                break

        self.step_world(dt, now)
        if self.state != "play":
            return
        if self.hazard_hits(now):
            return
        self.collide_entities(now)
        if self.state != "play":
            return
        self.collect_coins()
        self.check_end(now)

        target_cam = h.x - WIDTH * 0.42
        self.cam_x += (target_cam - self.cam_x) * min(1.0, dt * 7.0)
        self.cam_x = clamp(self.cam_x, 0, max(0, self.level.pw - WIDTH))

    def carry_on_lifts(self, dt):
        """Move the hero with whatever platform it is standing on."""
        h = self.hero
        hr = h.rect()
        for e in self.enemies:
            if not e.is_platform:
                continue
            top = e.platform_rect()
            landing = pygame.Rect(hr.x, hr.bottom - 4, hr.w, 10)
            if landing.colliderect(top) and h.vy >= 0:
                h.y = float(top.top)
                h.vy = 0.0
                h.grounded = True
                h.jumps_used = 0
                h.x += e.dx
                h.y += e.dy
                return True
        return False

    def hazard_hits(self, now):
        """Firebars and hammers: contact hurts, and no mushroom saves you from
        touching a firebar the way it does from a goomba -- it just shrinks you."""
        hr = self.hero.rect()
        for e in self.enemies:
            if e.kind != "firebar":
                continue
            for px, py in e.firebar_points():
                if hr.collidepoint(px, py):
                    self.hurt(now)
                    return True
        for hm in self.hammers:
            if hm.alive and hm.rect().colliderect(hr):
                hm.alive = False
                self.hurt(now)
                return True
        return False

    def step_world(self, dt, now):
        for hm in self.hammers:
            if hm.alive:
                hm.update(self.level, now)
        self.hammers = [hm for hm in self.hammers if hm.alive]
        for f in self.fireballs:
            if f.alive:
                f.update(self.level, now)
        self.fireballs = [f for f in self.fireballs if f.alive]
        for it in self.items:
            it.update(self.level, dt)
        self.items = [it for it in self.items if it.alive]
        spawned = []
        for e in self.enemies:
            if e.kind == "lakitu":
                e.target_x = self.hero.x
            e.update(self.level, dt, now)
            if not getattr(e, "throw_now", False):
                continue
            e.throw_now = False
            if e.kind == "hammer":
                direction = -1 if self.hero.x < e.x else 1
                self.hammers.append(Hammer(e.x + 16, e.y - 30,
                                           HAMMER_VX * direction, now))
            elif e.kind == "bowser":
                direction = -1 if self.hero.x < e.x else 1
                # Leg height, so a jump clears it. Spawned at head height, the only
                # "dodge" available was to duck -- and jumping, the obvious reflex
                # and the one the original teaches, moved you INTO it.
                self.hammers.append(Hammer(e.x + 16 * direction, e.y - 18,
                                           BREATH_SPEED * direction, now,
                                           flat=True))
            elif e.kind == "cannon":
                # Only fire when the player is in front of it and near enough to
                # see the shot coming; a cannon firing off-screen is just noise.
                ahead = (self.hero.x - e.x) * e.face
                if -BULLET_RANGE < ahead < 0 or (e.face < 0 and
                                                 0 < e.x - self.hero.x < BULLET_RANGE):
                    bullet = Enemy(e.x + (20 if e.face > 0 else -20),
                                   LV.FLOOR * TILE - 8, "bullet", now)
                    bullet.vx = BULLET_SPEED * e.face
                    bullet.home_y = now          # bullets use home_y as a birth stamp
                    spawned.append(bullet)
            elif e.kind == "lakitu":
                # Thrown AHEAD of the player, not dropped on their head. A
                # stomp-proof enemy landing straight down on you cannot be dodged
                # or answered; landing in front, it is an obstacle you can see and
                # burn. Two at a time is plenty.
                live = sum(1 for x in self.enemies
                           if x.alive and x.kind == "spiny" and x.dying <= 0)
                if live < 2:
                    lead = SPINY_LEAD * (1 if self.hero.facing >= 0 else -1)
                    spawned.append(Enemy(e.x + lead, e.y + TILE, "spiny", now))
        self.enemies = [e for e in self.enemies if e.alive] + spawned
        for key in list(self.level.bumps):
            self.level.bumps[key] -= dt
            if self.level.bumps[key] <= 0:
                del self.level.bumps[key]

    def collide_entities(self, now):
        h = self.hero
        hr = h.rect()
        for it in self.items:
            if it.emerging <= 0 and hr.colliderect(it.rect()):
                it.alive = False
                if h.grow(now):
                    self.score += 1000
                    self._toast("MUSHROOM! +1000", now)
                else:
                    self.score += 200

        for e in self.enemies:
            if not e.alive or e.dying > 0 or e.harmless:
                continue
            er = e.rect()
            for f in self.fireballs:
                if f.alive and f.rect().colliderect(er):
                    f.alive = False
                    if e.fireproof:
                        continue          # buzzy beetles are made for this
                    if e.armoured:
                        e.hp -= 1
                        if e.hp > 0:
                            self._toast(f"BOWSER: {e.hp} more!", now, 0.9)
                            continue
                        self.score += 5000
                        self._toast("BOWSER DOWN!", now, 2.0)
                    e.dying = 0.25
                    self.score += 100
                    self._toast("+100", now, 0.7)
            if e.dying > 0:
                continue
            if not hr.colliderect(er):
                continue

            # Generous stomp: either the feet are above the enemy's middle, or we
            # came down out of the air onto it at all. Walkers patrol, so an enemy
            # will always wander into the spot a jump lands -- no placement rule can
            # prevent that, and "I came down on it and died" is exactly the
            # unfairness this latency budget cannot afford. Hitting one from below
            # or walking into its side still hurts.
            came_down = h.was_airborne and h.was_falling
            if (came_down or hr.bottom <= er.centery + 8) and e.stompable \
                    and hr.top < er.top:
                killed = e.stomped()
                h.vy = -STOMP_BOUNCE
                h.jumps_used = min(h.jumps_used, 1)
                self.score += 100
                self._toast("+100", now, 0.7)
                if not killed:
                    self._toast("shell!", now, 0.7)
            else:
                self.hurt(now)
                return

    def hurt(self, now):
        """Big Mario shrinks; small Mario dies."""
        h = self.hero
        if now < h.invuln_until:
            return
        if h.big:
            h.shrink(now)
            self._toast("ouch -- shrunk", now)
        else:
            self.kill(now, "hit")

    def collect_coins(self):
        hr = self.hero.rect()
        for col in range(hr.left // TILE, hr.right // TILE + 1):
            for row in range(hr.top // TILE, hr.bottom // TILE + 1):
                if self.level.take_coin(col, row):
                    self.score += 50
                    self.coins_taken += 1

    def check_end(self, now):
        lv = self.level
        h = self.hero
        if lv.end_kind == "exit":
            return                          # bonus rooms end by their pipe
        ex = lv.end_col * TILE
        # Reaching the column is enough, at any height. Requiring a vertical band
        # too meant a hero swimming high in 2-2 crossed the axe without finishing.
        if h.x + h.W >= ex:
            self.state = "won"
            self.score += 2000 if lv.end_kind == "axe" else 1000
            self.state_until = now + 3.0

    def move_x(self, h):
        h.x += h.vx
        r = h.rect()
        if h.vx > 0:
            if (self.level.solid_at(r.right, r.top + 4)
                    or self.level.solid_at(r.right, r.bottom - 4)):
                if not self.ledge_assist(h, r.right):
                    h.x = (r.right // TILE) * TILE - h.W - 1
        elif h.vx < 0:
            if (self.level.solid_at(r.left, r.top + 4)
                    or self.level.solid_at(r.left, r.bottom - 4)):
                if not self.ledge_assist(h, r.left):
                    h.x = (r.left // TILE + 1) * TILE + 1

    def ledge_assist(self, h, edge_x):
        """Turn 'clipped the lip of the ledge' into 'just made it'.

        A jump that lands a few pixels low hits the side of the far platform, gets
        pushed back, and drops into the pit -- which reads as the game cheating you,
        and at 50-80ms of input latency it happens constantly. If the hero is
        falling and the blocking tile's surface is only a little above its feet,
        stand it on top instead.
        """
        if h.vy <= 0:
            return False
        col = int(edge_x // TILE)
        for row in range(int(h.y // TILE), int(h.y // TILE) + 2):
            top = row * TILE
            if 0 <= h.y - top <= LEDGE_SNAP and \
                    self.level.tile_at(col, row) in self.level.solids:
                h.y = float(top)
                h.vy = 0.0
                if not h.grounded:
                    h.grounded = True
                    h.jumps_used = 0
                return True
        return False

    def move_y(self, h, now):
        h.y += h.vy
        r = h.rect()
        if h.vy >= 0:
            if (self.level.solid_at(r.left + 4, r.bottom)
                    or self.level.solid_at(r.right - 4, r.bottom)):
                h.y = (r.bottom // TILE) * TILE
                h.vy = 0.0
                if not h.grounded:
                    h.grounded = True
                    h.jumps_used = 0
            else:
                h.grounded = False
        else:
            if (self.level.solid_at(r.left + 4, r.top)
                    or self.level.solid_at(r.right - 4, r.top)):
                self.bump_head(h, r, now)
                h.y = (r.top // TILE + 1) * TILE + h.height
                h.vy = 0.0
            h.grounded = False

    def bump_head(self, h, r, now):
        """Hitting a block from below: coins, mushrooms, broken bricks."""
        row = r.top // TILE
        for col in (int((r.left + 4) // TILE), int((r.right - 4) // TILE)):
            what = self.level.hit_block(col, row, h.big)
            if what == "coin":
                self.score += 50
                self.coins_taken += 1
                self._toast("+50", now, 0.6)
            elif what == "mushroom":
                self.items.append(Mushroom(col * TILE + 5, row * TILE + TILE))
                self._toast("a mushroom!", now)
            elif what == "break":
                self.score += 50
            if what:
                break

    def kill(self, now, why, force=False):
        h = self.hero
        if not force and now < h.invuln_until:
            return
        self.lives -= 1
        self.state = "dead"
        self.state_until = now + RESPAWN_PAUSE
        h.vy = -8.0
        self._toast({"hit": "ouch", "lava": "too hot!",
                     "fell": "watch the gap"}.get(why, "ouch"), now)

    # ---------- drawing ----------
    def draw(self, snap, now):
        th = THEMES[self.level.theme]
        self.draw_sky(th)
        ox = int(self.cam_x)
        self.draw_background(th, ox, now)
        self.draw_tiles(th, ox, now)
        self.draw_entities(ox, now)
        self.draw_hud(snap, now)
        self.draw_overlays(now)
        pygame.display.flip()

    def draw_sky(self, th):
        key = (th["sky_top"], th["sky_bottom"])
        surf = self.sky_cache.get(key)
        if surf is None:
            surf = pygame.Surface((1, HEIGHT))
            for i in range(HEIGHT):
                t = i / HEIGHT
                surf.set_at((0, i), tuple(
                    int(th["sky_top"][k] + (th["sky_bottom"][k] - th["sky_top"][k]) * t)
                    for k in range(3)))
            surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
            self.sky_cache[key] = surf
        self.screen.blit(surf, (0, 0))

    def draw_background(self, th, ox, now):
        """Parallax hills and clouds, so the levels do not all look the same."""
        if th["hills"]:
            for i in range(-1, WIDTH // 320 + 3):
                bx = i * 320 - int(ox * 0.35) % 320
                base = HUD_H + LV.FLOOR * TILE
                pygame.draw.polygon(self.screen, (60, 150, 60), [
                    (bx, base), (bx + 90, base - 78), (bx + 180, base)])
                pygame.draw.polygon(self.screen, (78, 168, 74), [
                    (bx + 150, base), (bx + 215, base - 48), (bx + 280, base)])
        if th["clouds"]:
            for i in range(-1, WIDTH // 260 + 3):
                cx = i * 260 - int(ox * 0.18) % 260
                cy = 110 + (i % 3) * 46
                for dx, dy, rr in ((0, 0, 22), (24, 6, 18), (-22, 6, 16)):
                    pygame.draw.circle(self.screen, (250, 250, 255),
                                       (cx + dx, cy + dy), rr)
        if th["stars"]:
            for i in range(60):
                sx = (i * 197 - int(ox * 0.1)) % WIDTH
                sy = HUD_H + (i * 83) % 260
                pygame.draw.circle(self.screen, (200, 230, 255), (sx, sy), 1)

    def draw_tiles(self, th, ox, now):
        first = max(0, ox // TILE)
        last = min(self.level.cols, (ox + WIDTH) // TILE + 2)
        for r, row in enumerate(self.level.rows):
            y = r * TILE + HUD_H
            for c in range(first, last):
                ch = row[c]
                if ch in (LV.EMPTY, LV.START):
                    continue
                x = c * TILE - ox
                bump = self.level.bumps.get((c, r), 0.0)
                if bump:
                    y_off = -int(10 * math.sin(math.pi * bump / BUMP_TIME))
                else:
                    y_off = 0
                self.draw_tile(th, ch, x, y + y_off, c, r, now)

    def draw_tile(self, th, ch, x, y, c, r, now):
        s = self.screen
        if ch == LV.GROUND:
            pygame.draw.rect(s, th["ground"], (x, y, TILE, TILE))
            if self.level.tile_at(c, r - 1) not in self.level.solids:
                pygame.draw.rect(s, th["ground_top"], (x, y, TILE, 9))
            pygame.draw.line(s, th["brick_line"], (x, y), (x, y + TILE))
        elif ch == LV.BRICK:
            pygame.draw.rect(s, th["brick"], (x, y, TILE, TILE))
            for i in range(0, TILE, 10):
                pygame.draw.line(s, th["brick_line"], (x, y + i), (x + TILE, y + i))
            pygame.draw.line(s, th["brick_line"], (x + 20, y), (x + 20, y + 10))
        elif ch == LV.STONE:
            pygame.draw.rect(s, th["stone"], (x, y, TILE, TILE))
            pygame.draw.rect(s, th["brick_line"], (x, y, TILE, TILE), 2)
        elif ch in (LV.QCOIN, LV.QMUSH):
            pulse = 0.5 + 0.5 * math.sin(now * 5 + c)
            col = tuple(int(v * (0.8 + 0.2 * pulse)) for v in th["block"])
            pygame.draw.rect(s, col, (x, y, TILE, TILE))
            pygame.draw.rect(s, th["block_edge"], (x, y, TILE, TILE), 3)
            q = self.small.render("?", True, th["block_edge"])
            s.blit(q, (x + TILE // 2 - q.get_width() // 2,
                       y + TILE // 2 - q.get_height() // 2))
        elif ch == LV.USED:
            pygame.draw.rect(s, th["block_edge"], (x, y, TILE, TILE))
            pygame.draw.rect(s, (90, 62, 10), (x, y, TILE, TILE), 3)
        elif ch in (LV.PIPE, LV.PIPE_IN, LV.EXIT):
            pygame.draw.rect(s, C_PIPE, (x, y, TILE, TILE))
            pygame.draw.rect(s, C_PIPE_DARK, (x, y, 7, TILE))
            top = self.level.tile_at(c, r - 1) not in (LV.PIPE, LV.PIPE_IN, LV.EXIT)
            if top:
                pygame.draw.rect(s, C_PIPE, (x - 4, y, TILE + 8, 13))
                pygame.draw.rect(s, C_PIPE_DARK, (x - 4, y, TILE + 8, 13), 2)
                if ch in (LV.PIPE_IN, LV.EXIT):
                    # A dark mouth is the only cue that this one can be entered.
                    pygame.draw.rect(s, C_PIPE_MOUTH, (x, y + 4, TILE, 9))
        elif ch == LV.COIN:
            cx, cy = x + TILE // 2, y + TILE // 2
            bob = int(math.sin(now * 4 + c) * 3)
            pygame.draw.ellipse(s, C_COIN, (cx - 8, cy - 11 + bob, 16, 22))
            pygame.draw.ellipse(s, C_COIN_EDGE, (cx - 8, cy - 11 + bob, 16, 22), 2)
        elif ch == LV.LAVA:
            wob = int(math.sin(now * 3 + c * 0.6) * 3)
            pygame.draw.rect(s, C_LAVA_A, (x, y, TILE, TILE))
            pygame.draw.rect(s, C_LAVA_B, (x, y + 4 + wob, TILE, 7))
        elif ch == LV.WATER:
            pygame.draw.rect(s, C_WATER, (x, y, TILE, TILE))
            if self.level.tile_at(c, r - 1) != LV.WATER:
                wob = int(math.sin(now * 2 + c * 0.5) * 2)
                pygame.draw.line(s, (140, 200, 255), (x, y + wob),
                                 (x + TILE, y + wob), 2)
        elif ch == LV.BRIDGE:
            pygame.draw.rect(s, C_BRIDGE, (x, y, TILE, 12))
            pygame.draw.line(s, (110, 70, 30), (x, y + 12), (x + TILE, y + 12), 2)
            pygame.draw.line(s, (110, 70, 30), (x + TILE // 2, y + 12),
                             (x + TILE // 2, y + TILE), 2)
        elif ch == LV.POLE:
            pygame.draw.rect(s, C_FLAG_POLE, (x + TILE // 2 - 3, y, 6, TILE))
            if self.level.tile_at(c, r - 1) != LV.POLE:
                wave = int(math.sin(now * 3) * 4)
                pygame.draw.polygon(s, C_FLAG, [
                    (x + TILE // 2 + 3, y + 2),
                    (x + TILE // 2 + 40 + wave, y + 16),
                    (x + TILE // 2 + 3, y + 30)])
        elif ch == LV.CASTLE:
            pygame.draw.rect(s, (170, 80, 50), (x - 30, y - TILE, TILE + 60, TILE * 2))
            for i in range(4):
                pygame.draw.rect(s, (170, 80, 50),
                                 (x - 30 + i * 26, y - TILE - 14, 16, 16))
            pygame.draw.rect(s, (30, 20, 20), (x + 6, y + 8, 28, 32))
        elif ch == LV.CANNON:
            pygame.draw.rect(s, (40, 40, 50), (x, y, TILE, TILE))
            pygame.draw.rect(s, (90, 90, 105), (x, y, TILE, TILE), 2)
            if self.level.tile_at(c, r - 1) != LV.CANNON:
                pygame.draw.rect(s, (20, 20, 28), (x + 2, y + 10, TILE - 4, 16))
        elif ch == LV.VINE:
            for i in range(6):
                yy = y - i * 22
                pygame.draw.line(s, (60, 170, 70), (x + 20, yy), (x + 20, yy - 22), 5)
                side = -1 if i % 2 else 1
                pygame.draw.ellipse(s, (80, 200, 90),
                                    (x + 20 + side * 12 - 6, yy - 16, 14, 10))
        elif ch == LV.AXE:
            pygame.draw.rect(s, (150, 110, 60), (x + 16, y + 10, 6, 28))
            pygame.draw.polygon(s, (220, 230, 240), [
                (x + 8, y + 6), (x + 32, y + 12), (x + 8, y + 18)])

    def draw_entities(self, ox, now):
        s = self.screen
        for it in self.items:
            r = it.rect()
            s.blit(self.spr["mushroom"], (r.x - ox, r.y + HUD_H))
        for e in self.enemies:
            r = e.rect()
            if r.right - ox < -40 or r.left - ox > WIDTH + 40:
                continue
            if e.dying > 0:
                pygame.draw.ellipse(s, (150, 80, 40),
                                    (r.x - ox, r.bottom + HUD_H - 10, r.w, 10))
                continue
            if e.is_platform:
                pr = e.platform_rect()
                pygame.draw.rect(s, (120, 110, 90),
                                 (pr.x - ox, pr.y + HUD_H, pr.w, pr.h))
                pygame.draw.rect(s, (60, 55, 45),
                                 (pr.x - ox, pr.y + HUD_H, pr.w, pr.h), 2)
                for i in range(0, pr.w, TILE):
                    pygame.draw.line(s, (200, 190, 150),
                                     (pr.x - ox + i + 6, pr.y + HUD_H),
                                     (pr.x - ox + i + 6, pr.y + HUD_H + pr.h))
                continue
            if e.kind == "firebar":
                for i, (px, py) in enumerate(e.firebar_points()):
                    rr = 9 if i < e.length - 1 else 7
                    pygame.draw.circle(s, C_FIRE_A, (int(px - ox), int(py + HUD_H)), rr)
                    pygame.draw.circle(s, C_FIRE_B,
                                       (int(px - ox), int(py + HUD_H)), max(2, rr - 4))
                continue
            if e.kind == "podoboo":
                if e.submerged:
                    continue
                cx, cy = int(e.x - ox), int(e.y + HUD_H - 16)
                pygame.draw.circle(s, C_LAVA_B, (cx, cy), 13)
                pygame.draw.circle(s, C_LAVA_A, (cx, cy), 8)
                continue
            if e.kind == "cannon":
                continue                        # drawn as a tile, not a sprite
            if e.kind == "bowser":
                bx, by = int(e.x - ox), int(e.y + HUD_H)
                pygame.draw.rect(s, (70, 160, 60), (bx - 6, by - 58, 56, 44))
                pygame.draw.rect(s, (240, 230, 170), (bx + 4, by - 46, 34, 22))
                for i in range(4):                       # shell spikes
                    pygame.draw.polygon(s, (240, 240, 230), [
                        (bx - 4 + i * 16, by - 58), (bx + 3 + i * 16, by - 70),
                        (bx + 10 + i * 16, by - 58)])
                pygame.draw.rect(s, (70, 160, 60), (bx + 30, by - 84, 26, 28))
                pygame.draw.rect(s, (250, 250, 250), (bx + 36, by - 74, 16, 7))
                for i in range(2):                       # horns
                    pygame.draw.polygon(s, (240, 240, 230), [
                        (bx + 32 + i * 16, by - 84), (bx + 36 + i * 16, by - 96),
                        (bx + 40 + i * 16, by - 84)])
                pygame.draw.rect(s, (200, 90, 40), (bx, by - 14, 18, 14))
                pygame.draw.rect(s, (200, 90, 40), (bx + 30, by - 14, 18, 14))
                bar = max(0, e.hp) / BOWSER_HP
                pygame.draw.rect(s, (60, 20, 20), (bx - 6, by - 106, 56, 6))
                pygame.draw.rect(s, (230, 60, 50), (bx - 6, by - 106, int(56 * bar), 6))
                continue
            if e.kind == "bullet":
                bx, by = int(e.x - ox), int(e.y + HUD_H - 16)
                pygame.draw.ellipse(s, (30, 30, 38), (bx - 14, by - 8, 28, 18))
                pygame.draw.circle(s, (250, 250, 250), (bx + (6 if e.vx > 0 else -6),
                                                        by - 2), 3)
                continue
            key = "shell" if (e.kind in ("koopa", "buzzy") and e.shell) else e.kind
            spr = self.spr.get(key, self.spr["goomba"])
            s.blit(spr, (r.x - ox, r.y + HUD_H))

        for hm in self.hammers:
            hx, hy = int(hm.x - ox), int(hm.y + HUD_H)
            if hm.flat:
                pygame.draw.ellipse(s, C_FIRE_A, (hx - 16, hy - 7, 32, 14))
                pygame.draw.ellipse(s, C_FIRE_B, (hx - 9, hy - 4, 18, 8))
                continue
            ang = hm.spin
            pygame.draw.line(s, (150, 110, 60),
                             (hx - int(8 * math.cos(ang)), hy - int(8 * math.sin(ang))),
                             (hx + int(8 * math.cos(ang)), hy + int(8 * math.sin(ang))), 4)
            pygame.draw.circle(s, (200, 200, 210),
                               (hx + int(9 * math.cos(ang)), hy + int(9 * math.sin(ang))), 5)

        for f in self.fireballs:
            fx, fy = int(f.x - ox), int(f.y + HUD_H)
            pygame.draw.circle(s, C_FIRE_A, (fx, fy), f.R)
            pygame.draw.circle(s, C_FIRE_B, (fx, fy), max(2, f.R - 4))

        h = self.hero
        if now < h.invuln_until and int(now * 20) % 2 == 0:
            return                          # flicker while invulnerable
        r = h.rect()
        pygame.draw.ellipse(s, (0, 0, 0, 60),
                            (r.x - ox + 2, r.bottom + HUD_H - 4, r.w - 4, 6))
        key = ("hero_b" if h.big else "hero_s") + ("" if h.facing > 0 else "_l")
        spr = self.spr[key]
        if h.ducking:
            spr = pygame.transform.scale(
                spr, (spr.get_width(), max(8, int(spr.get_height() * 0.62))))
        bob = int(math.sin(h.walk_phase) * 2) if (h.grounded and abs(h.vx) > 0.2) else 0
        s.blit(spr, (r.x - ox - 5, r.bottom + HUD_H - spr.get_height() + bob))

    def draw_steer_bar(self, snap, x, y, w=240, h=18):
        bar = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (12, 14, 22), bar, border_radius=3)
        cx = bar.centerx
        dz = int(w * 0.5 * STEER_DEADZONE)
        pygame.draw.rect(self.screen, (30, 36, 52), (cx - dz, bar.top, dz * 2, h))
        for sign in (-1, 1):
            rx = cx + sign * int(w * 0.5 * RUN_THRESHOLD)
            pygame.draw.line(self.screen, (150, 130, 60), (rx, bar.top), (rx, bar.bottom))
        pygame.draw.line(self.screen, (140, 150, 170), (cx, bar.top), (cx, bar.bottom))
        val = snap["analog"] if snap else 0.0
        if abs(val) > 0.001:
            fw = int(abs(val) * w * 0.5)
            pygame.draw.rect(self.screen, (0, 235, 255),
                             (cx if val > 0 else cx - fw, bar.top + 3, fw, h - 6),
                             border_radius=2)
        pygame.draw.rect(self.screen, (70, 80, 100), bar, 1, border_radius=3)

    def draw_hud(self, snap, now):
        pygame.draw.rect(self.screen, C_HUD_BG, (0, 0, WIDTH, HUD_H))
        elapsed = int(now - self.level_start)
        label = self.level.name if self.in_bonus else \
            f"{self.world}-{self.level_no}"
        x = 18
        for txt in (f"SCORE {self.score:06d}", f"COINS {self.coins_taken:02d}",
                    f"LIVES {max(0, self.lives)}", f"WORLD {label}",
                    f"TIME {elapsed:3d}"):
            surf = self.font.render(txt, True, C_TEXT)
            self.screen.blit(surf, (x, 12))
            x += surf.get_width() + 26

        if self.tracker and snap:
            bx = WIDTH - CAM_PREVIEW[0] - 18
            for i, (role, seen, side) in enumerate(
                    (("STEER", snap["steer_seen"], snap.get("steer_side")),
                     ("FIRE", snap["fire_seen"], snap.get("fire_side")))):
                col = (60, 230, 130) if seen else (255, 70, 90)
                pygame.draw.circle(self.screen, col, (bx - 100, 20 + i * 22), 6)
                text = f"{role} {side or '?'}" if seen else role
                self.screen.blit(self.small.render(text, True, col),
                                 (bx - 88, 12 + i * 22))
            self.draw_steer_bar(snap, bx - 100, 44)
            frame = snap["frame"]
            if frame is not None:
                surf = self.tracker_surface(frame)
                self.screen.blit(surf, (bx, HEIGHT - CAM_PREVIEW[1] - 12))
                pygame.draw.rect(self.screen, (40, 48, 68),
                                 (bx, HEIGHT - CAM_PREVIEW[1] - 12,
                                  CAM_PREVIEW[0], CAM_PREVIEW[1]), 1)
        elif not self.tracker:
            s = self.small.render(
                "keyboard: arrows, shift=run, space=jump, X=fire, down=duck/pipe",
                True, (150, 160, 180))
            self.screen.blit(s, (WIDTH - s.get_width() - 18, 22))

    def tracker_surface(self, frame):
        import cv2
        import numpy as np
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(np.ascontiguousarray(rgb).tobytes(),
                                       (frame.shape[1], frame.shape[0]), "RGB")
        return pygame.transform.smoothscale(surf, CAM_PREVIEW)

    def draw_overlays(self, now):
        if self.toast and now < self.toast_until:
            s = self.font.render(self.toast, True, (255, 255, 255))
            self.screen.blit(s, (WIDTH // 2 - s.get_width() // 2, HUD_H + 20))
        banner = None
        if self.state == "won":
            nxt = "next level..." if (self.world, self.level_no) != (2, 4) else ""
            banner = (f"{self.level.name} CLEAR!", nxt or "R to restart")
        elif self.state == "complete":
            banner = ("BOTH WORLDS CLEAR!", f"final score {self.score}   R to replay")
        elif self.state == "gameover":
            banner = ("GAME OVER", "R to try again")
        if banner:
            box = pygame.Rect(WIDTH // 2 - 320, HEIGHT // 2 - 90, 640, 160)
            shade = pygame.Surface(box.size, pygame.SRCALPHA)
            shade.fill((10, 12, 20, 220))
            self.screen.blit(shade, box.topleft)
            pygame.draw.rect(self.screen, (90, 110, 150), box, 2)
            t = self.big_font.render(banner[0], True, C_COIN)
            self.screen.blit(t, (box.centerx - t.get_width() // 2, box.y + 28))
            s = self.small.render(banner[1], True, C_TEXT)
            self.screen.blit(s, (box.centerx - s.get_width() // 2, box.y + 96))


def parse_args(argv):
    p = argparse.ArgumentParser(description="Gesture-controlled platformer.")
    p.add_argument("--world", type=int, default=1, help="world number (1-4)")
    p.add_argument("--level", type=int, default=1, help="level number (1-4)")
    p.add_argument("--no-camera", action="store_true",
                   help="keyboard only; useful for testing the game itself")
    p.add_argument("--camera", type=int, default=0, metavar="N",
                   help="camera device index (default 0)")
    p.add_argument("--steer-zone", nargs=2, type=float, metavar=("LO", "HI"),
                   help="steering band as fractions of frame width "
                        "(default %.2f %.2f)" % STEER_ZONE_DEFAULT)
    p.add_argument("--swap-hands", action="store_true",
                   help="flip left/right hand detection (or press H while playing)")
    p.add_argument("--log", nargs="?", const="auto", default=None, metavar="PATH",
                   help="write a CSV of poses and events for analyse_log.py")
    return p.parse_args(argv)


if __name__ == "__main__":
    opts = parse_args(sys.argv[1:])
    path = opts.log
    if path == "auto":
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            time.strftime("play-log-%Y%m%d-%H%M%S.csv"))
    if opts.steer_zone:
        tracker_mod.STEER_ZONE = tuple(opts.steer_zone)
    game = Game(world=opts.world, level=opts.level,
                use_camera=not opts.no_camera, camera_id=opts.camera,
                log_path=path)
    if opts.swap_hands and game.tracker:
        game.tracker.swap_handedness = True
    game.run()
