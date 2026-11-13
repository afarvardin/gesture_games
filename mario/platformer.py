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
F toggles full screen; Esc leaves full screen, or quits from a window.

  python platformer.py [--world 1-4] [--level 1-4] [--no-camera] [--camera N]
                       [--steer-zone LO HI] [--swap-hands] [--log [PATH]]
                       [--fullscreen] [--cpu-tracking]
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
# The other mushrooms (World 5 on). Rot is slow on purpose: it is a trap you are
# meant to see coming out of the block and then jump or burn, not a coin flip.
ITEM_SPEED = {"mushroom": MUSHROOM_SPEED, "life": 2.3, "spring": MUSHROOM_SPEED,
              "gold": 2.1, "rot": 1.25}
GOLD_BOUNCE = 7.0                 # px/frame; the gold one hops as it goes
SPRING_TIME = 15.0                # s of higher jumps
SPRING_BOOST = 1.18               # jump speed multiplier: ~1.4x the height
GOLD_TIME = 8.0                   # s of smashing through enemies untouched
POWER_WARN = 2.0                  # s before a power runs out that it flickers

SKELETON_SPEED = 1.0              # px/frame; a shade slower than a goomba
SKELETON_REST = 3.4               # s a stomped skeleton lies in a heap
SKELETON_RATTLE = 0.8             # s of rattling before it stands: the warning
SKELETON_CLEAR = 1.6 * TILE       # it will not stand up with you this close --
                                  # rising into the player is a hit they could not
                                  # see coming, so it waits, still rattling
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
    # ---- Worlds 5-8: a look per world (see levels.py) ----
    LV.THEME_AUTUMN: dict(
        sky_top=(236, 132, 78), sky_bottom=(255, 214, 150),
        ground=(132, 78, 40), ground_top=(214, 130, 40),
        brick=(176, 86, 44), brick_line=(120, 52, 24),
        stone=(176, 150, 124), block=(232, 178, 62), block_edge=(150, 104, 20),
        hills=True, clouds=True, stars=False, night=False,
        hill_cols=((178, 86, 40), (206, 122, 50)), cloud_col=(255, 232, 210),
        weather="leaves",
    ),
    LV.THEME_HARVEST: dict(
        sky_top=(16, 10, 40), sky_bottom=(70, 40, 70),
        ground=(96, 58, 40), ground_top=(160, 100, 40),
        brick=(140, 80, 50), brick_line=(90, 50, 30),
        stone=(150, 130, 120), block=(232, 178, 62), block_edge=(150, 104, 20),
        hills=True, clouds=False, stars=True, night=True,
        hill_cols=((44, 30, 48), (60, 40, 60)), backdrop="moon",
        weather="fireflies",
    ),
    LV.THEME_HIGHLANDS: dict(
        sky_top=(110, 150, 214), sky_bottom=(214, 230, 248),
        ground=(104, 92, 88), ground_top=(236, 244, 252),
        brick=(150, 108, 92), brick_line=(100, 66, 56),
        stone=(186, 194, 210), block=(232, 178, 62), block_edge=(150, 104, 20),
        hills=False, clouds=True, stars=False, night=False,
        backdrop="mountains", weather="snow",
    ),
    LV.THEME_AURORA: dict(
        sky_top=(4, 8, 26), sky_bottom=(18, 36, 66),
        ground=(70, 66, 80), ground_top=(210, 222, 240),
        brick=(110, 90, 120), brick_line=(70, 54, 80),
        stone=(160, 168, 190), block=(232, 178, 62), block_edge=(150, 104, 20),
        hills=False, clouds=False, stars=True, night=True,
        backdrop="aurora", weather="snow",
    ),
    LV.THEME_STORM: dict(
        sky_top=(36, 42, 56), sky_bottom=(88, 98, 116),
        ground=(84, 76, 62), ground_top=(64, 116, 70),
        brick=(126, 88, 66), brick_line=(84, 56, 40),
        stone=(140, 146, 156), block=(214, 164, 60), block_edge=(130, 90, 20),
        hills=True, clouds=True, stars=False, night=False,
        hill_cols=((48, 82, 58), (58, 96, 66)), cloud_col=(74, 80, 96),
        weather="rain",
    ),
    LV.THEME_ASHEN: dict(
        # The glow is on the horizon, not the bottom of the screen: pits show the
        # sky, and a red-orange pit reads as lava.
        sky_top=(30, 6, 10), sky_bottom=(84, 30, 24),
        ground=(66, 52, 54), ground_top=(104, 84, 64),
        brick=(116, 62, 50), brick_line=(70, 34, 28),
        stone=(124, 112, 112), block=(214, 150, 60), block_edge=(130, 80, 20),
        hills=False, clouds=False, stars=False, night=False,
        backdrop="volcanoes", weather="ash",
    ),
    LV.THEME_KEEP: dict(
        sky_top=(8, 0, 2), sky_bottom=(42, 12, 10),
        ground=(108, 98, 104), ground_top=(146, 136, 144),
        brick=(132, 118, 124), brick_line=(84, 70, 76),
        stone=(90, 80, 88), block=(200, 150, 60), block_edge=(120, 84, 16),
        hills=False, clouds=False, stars=False, night=False,
        backdrop="torches", weather="embers",
    ),
}

# The name of each world from 5 on, shown on a title card as a level starts.
WORLD_NAMES = {5: "THE AMBER WILDS", 6: "THE FROSTPEAK HIGHLANDS",
               7: "THE STORM COAST", 8: "THE ASHEN KINGDOM"}
TITLE_CARD_SECS = 2.6

# One colour per role, everywhere: blue is the LEFT hand and moves you, orange is
# the RIGHT hand and throws fire. The HUD used to colour these by whether the hand
# was currently tracked, which meant the colours said nothing about which hand did
# what -- the thing a player actually needs to know at a glance.
C_ROLE_MOVE = (60, 170, 255)
C_ROLE_FIRE = (255, 140, 50)

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
# The other mushrooms. Same stalk and face as the plain one so they read as
# mushrooms at a glance; the cap is what tells you which one it is.
LIFE_MUSH_PIXELS = [                 # a heart for a cap: an extra life
    ".QQ..QQ.",
    "QWQQQQQQ",
    "QQQQQQQQ",
    ".QQQQQQ.",
    ".SSQQSS.",
    ".SKSSKS.",
    ".SSSSSS.",
    "..SSSS..",
]
SPRING_MUSH_PIXELS = [               # an up-chevron on a teal cap: big jumps
    "..JJJJ..",
    ".JJWWJJ.",
    "JJWJJWJJ",
    "JWJJJJWJ",
    ".SSSSSS.",
    ".SKSSKS.",
    ".SSSSSS.",
    "..SSSS..",
]
GOLD_MUSH_PIXELS = [                 # a shining gold cap: untouchable
    "..ZZZZ..",
    ".ZWZZzZ.",
    "ZWZZZzZZ",
    "zZZZZZZz",
    ".SSSSSS.",
    ".SKSSKS.",
    ".SSSSSS.",
    "..SSSS..",
]
ROT_MUSH_PIXELS = [                  # a sickly spotted cap and a scowl: poison
    "..VVVV..",
    ".VyVVyV.",
    "VVVVyVVV",
    "VyVVVVVV",
    ".ssssss.",
    ".sKssKs.",
    ".ssKKss.",
    "..ssss..",
]
# Skeletons, World 8. Ten rows rather than eight: they stand a head taller than a
# goomba, which is most of what makes them read as walking bones. Face left.
SKELETON_A = [
    "..DDDD..",
    ".DDDDDD.",
    ".DeDDeD.",
    ".DDDDDd.",
    "..DKDK..",
    "...dD...",
    ".DdDdDD.",
    "D.dDdD.D",
    "...dD...",
    "..D..D..",
]
SKELETON_B = SKELETON_A[:8] + [
    "...dD...",
    "...DD...",
]
BONES_PIXELS = [                     # stomped: a heap that pulls itself back up
    "........",
    "........",
    "........",
    "........",
    "........",
    "........",
    "...DDD..",
    "..DeDd..",
    ".dDDdDD.",
    "DDdDDdDD",
]
PIXEL_COLORS = {
    "R": (224, 52, 48), "S": (250, 200, 156), "T": (46, 82, 200),
    "Q": (236, 84, 140), "J": (40, 170, 190), "Z": (250, 196, 40),
    "z": (196, 128, 20), "V": (104, 112, 52), "y": (206, 214, 80),
    "s": (196, 198, 176), "D": (232, 226, 206), "d": (160, 150, 130),
    "e": (255, 110, 40),
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


# ---------- the Magma King, the World 8 boss ----------
# Too big and too animated for an 8x8 pixel map: he is drawn from outlined shapes
# instead, facing right, on a canvas in "design px", at 2x and then smoothed down
# so the ink lines stay clean. Low-slung on purpose: his fire leaves at leg height
# (so a jump clears it), so his mouth sits at leg height too and the jet visibly
# comes out of it.
MK_W, MK_H = 138, 112             # canvas, design px
MK_ANCHOR_X = 80                  # canvas x that sits over the middle of his hitbox
MK_FOOT_Y = 108                   # canvas y of the soles of his feet
MK_SS = 2                         # supersampling
MK_SCALE = 1.25                   # on screen: a final boss should tower a bit
MK_HINGE = (98, 86)               # jaw hinge: the head pivots round this
MK_INK = (24, 12, 16)
MK_SKIN, MK_SKIN_DK, MK_SKIN_LT = (104, 92, 110), (72, 62, 82), (140, 128, 146)
MK_BELLY, MK_BELLY_DK = (188, 168, 140), (146, 126, 104)
MK_SHELL, MK_SHELL_LT, MK_SHELL_RIM = (36, 28, 44), (78, 66, 96), (60, 50, 70)
MK_BONE, MK_BONE_DK = (236, 224, 196), (184, 168, 138)
MK_IRON, MK_IRON_LT = (98, 98, 112), (160, 160, 176)
MK_CRACK = [                      # glowing seams in the shell
    [(30, 62), (38, 53), (36, 45), (44, 38)],
    [(38, 53), (48, 59), (58, 52), (66, 57), (75, 49)],
    [(58, 52), (56, 43), (62, 35)],
    [(48, 59), (46, 70), (54, 78), (52, 88)],
    [(66, 57), (74, 67), (84, 66), (89, 74)],
    [(74, 67), (70, 79), (76, 88)],
    [(22, 76), (34, 73), (46, 70)],
]
MK_SHARDS = [((28, 42), (19, 26)), ((43, 32), (37, 13)),   # (base, tip) of the
             ((60, 28), (59, 9)), ((76, 34), (82, 17))]     # obsidian ridge


def _lerp_col(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3))


def _bez(p0, p1, p2, p3, n=12):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u * u * u * p0[0] + 3 * u * u * t * p1[0]
                    + 3 * u * t * t * p2[0] + t * t * t * p3[0],
                    u * u * u * p0[1] + 3 * u * u * t * p1[1]
                    + 3 * u * t * t * p2[1] + t * t * t * p3[1]))
    return out


def _taper(line, w0, w1):
    """A polygon around a centreline, w0 wide at the root and w1 at the tip."""
    left, right = [], []
    n = len(line)
    for i, (x, y) in enumerate(line):
        ax, ay = line[max(0, i - 1)]
        bx, by = line[min(n - 1, i + 1)]
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy) or 1.0
        w = (w0 + (w1 - w0) * i / (n - 1)) / 2
        left.append((x - dy / d * w, y + dx / d * w))
        right.append((x + dy / d * w, y - dx / d * w))
    return left + right[::-1]


def _rot(pts, angle, pivot=MK_HINGE):
    c, s = math.cos(angle), math.sin(angle)
    px, py = pivot
    return [(px + (x - px) * c - (y - py) * s, py + (x - px) * s + (y - py) * c)
            for x, y in pts]


def _ink(surf, pts, fill, width=1.6):
    """Fill a polygon and give it a round-jointed ink outline."""
    p = [(x * MK_SS, y * MK_SS) for x, y in pts]
    pygame.draw.polygon(surf, fill, p)
    w = max(1, round(width * MK_SS))
    pygame.draw.lines(surf, MK_INK, True, p, w)
    for x, y in p:
        pygame.draw.circle(surf, MK_INK, (round(x), round(y)), w // 2)


def _stroke(surf, pts, col, width):
    p = [(x * MK_SS, y * MK_SS) for x, y in pts]
    w = max(1, round(width * MK_SS))
    pygame.draw.lines(surf, col, False, p, w)
    for x, y in p:
        pygame.draw.circle(surf, col, (round(x), round(y)), w // 2)


def _dot(surf, centre, r, col):
    pygame.draw.circle(surf, col, (round(centre[0] * MK_SS),
                                   round(centre[1] * MK_SS)), max(1, round(r * MK_SS)))


def _leg(x0, x1, top, ground):
    """A thick pillar of a leg with a splayed foot, toes forward (to the right)."""
    return [(x0 + 3, top), (x1 - 3, top), (x1, ground - 9), (x1 + 4, ground - 4),
            (x1 + 4, ground), (x0 - 2, ground), (x0 - 1, ground - 6)]


def _claws(surf, x1, ground):
    for k in range(3):
        tx = x1 - 9 + k * 5
        _ink(surf, [(tx, ground - 5), (tx + 7, ground - 1), (tx, ground)],
             MK_BONE, 1.2)


def draw_magma_king(mouth, glow):
    """One frame of the boss, facing right.

    mouth: 0 shut .. 1 roaring wide.  glow: 0 embers .. 1 white-hot, for the cracks
    in his shell, his eye and his throat -- it climbs as he winds up to breathe.
    """
    S = MK_SS
    surf = pygame.Surface((MK_W * S, MK_H * S), pygame.SRCALPHA)
    ground = MK_FOOT_Y
    hot = _lerp_col((255, 120, 30), (255, 236, 150), glow)
    warm = _lerp_col((150, 40, 12), (255, 120, 30), glow)

    # Tail, sweeping back, with a row of small obsidian spurs.
    _ink(surf, _bez((26, 82), (14, 86), (8, 94), (1, 102))
         + _bez((1, 102), (10, 105), (22, 106), (34, 104)), MK_SKIN)
    for bx, by, tx, ty in ((9, 94, 4, 88), (15, 89, 11, 82)):
        _ink(surf, [(bx - 3, by + 1), (tx, ty), (bx + 3, by - 1)], MK_SHELL, 1.2)

    # Far legs, in shadow, a step up the screen so they read as further away.
    _ink(surf, _leg(42, 62, 82, ground - 3), MK_SKIN_DK)
    _ink(surf, _leg(62, 84, 82, ground - 3), MK_SKIN_DK)

    # Body, and the ash-pale plates of his belly under the shell.
    _ink(surf, _bez((22, 90), (22, 70), (96, 70), (98, 90))
         + _bez((98, 90), (96, 106), (24, 106), (22, 90)), MK_SKIN)
    _ink(surf, [(34, 93), (86, 93), (84, 102), (36, 102)], MK_BELLY, 1.2)
    for x in range(42, 84, 8):
        _stroke(surf, [(x, 94), (x - 1, 101)], MK_BELLY_DK, 1.0)

    # Near legs.
    for x0, x1 in ((28, 54), (72, 98)):
        _ink(surf, _leg(x0, x1, 84, ground), MK_SKIN)
        _stroke(surf, [(x0 + 4, 92), (x0 + 9, 90)], MK_SKIN_DK, 1.2)
        _stroke(surf, [(x0 + 5, 98), (x0 + 11, 96)], MK_SKIN_DK, 1.2)
        _claws(surf, x1, ground)

    # Neck.
    _ink(surf, [(80, 60), (98, 64), (102, 90), (82, 96)], MK_SKIN)

    # The shell: a cracked obsidian dome with a riveted rim.
    top = _bez((14, 94), (8, 58), (24, 26), (52, 26)) \
        + _bez((52, 26), (80, 26), (98, 56), (94, 90))
    bottom = _bez((94, 90), (70, 99), (36, 101), (14, 94))
    _ink(surf, top + bottom, MK_SHELL, 2.0)
    _stroke(surf, _bez((19, 76), (14, 52), (28, 36), (47, 32)), MK_SHELL_LT, 2.2)
    rim_up = [(x, y - 7) for x, y in bottom[::-1]]
    _ink(surf, bottom + rim_up, MK_SHELL_RIM, 1.4)
    for i in range(2, len(bottom) - 1, 2):
        x, y = bottom[i]
        _stroke(surf, [(x, y - 1), (x + 1, y - 6)], MK_INK, 1.0)
        _dot(surf, (x + 3, y - 4), 0.9, MK_SHELL_LT)

    # Heat bleeding out round the cracks, added on top so it only lights the shell.
    halo = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    for crack in MK_CRACK:
        for (ax, ay), (bx, by) in zip(crack, crack[1:]):
            for t in (0.0, 0.5):
                _dot(halo, (ax + (bx - ax) * t, ay + (by - ay) * t), 4.5,
                     _lerp_col((40, 10, 0), (120, 44, 6), glow))
    surf.blit(halo, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    for crack in MK_CRACK:
        _stroke(surf, crack, (90, 20, 8), 3.2)
        _stroke(surf, crack, warm, 2.0)
        _stroke(surf, crack, hot, 0.9)

    # Obsidian shards along the ridge, each with a lit facet.
    for (bx, by), (tx, ty) in MK_SHARDS:
        dx, dy = tx - bx, ty - by
        d = math.hypot(dx, dy)
        nx, ny = -dy / d * 6, dx / d * 6
        mid = (bx + dx * 0.55, by + dy * 0.55)
        shard = [(bx - nx, by - ny), (mid[0] - nx * 0.8, mid[1] - ny * 0.8),
                 (tx, ty), (mid[0] + nx * 0.8, mid[1] + ny * 0.8), (bx + nx, by + ny)]
        _ink(surf, shard, MK_SHELL, 1.4)
        pygame.draw.polygon(surf, MK_SHELL_LT, [
            (p[0] * S, p[1] * S) for p in (shard[0], shard[1], shard[2], (bx, by))])

    # Head, drawn a size up round the hinge so it holds its own against the shell.
    # The jaw drops round the hinge and the skull tips back a little as it does,
    # which is what turns "mouth open" into a roar.
    hs = 1.22
    lift = -0.16 * mouth
    drop = 0.55 * mouth

    def head(pts):
        hx, hy = MK_HINGE
        return _rot([(hx + (x - hx) * hs, hy + (y - hy) * hs) for x, y in pts], lift)

    def jawp(pts):
        hx, hy = MK_HINGE
        return _rot([(hx + (x - hx) * hs, hy + (y - hy) * hs) for x, y in pts], drop)

    skull = head(_bez((92, 72), (94, 58), (108, 54), (116, 64))
                 + _bez((116, 64), (121, 68), (126, 71), (127, 79))
                 + [(125, 85), (100, 86), (92, 80)])
    jaw = jawp([(95, 84), (122, 85), (120, 90), (111, 95), (99, 95), (93, 90)])

    # Throat, then the jaw with its lower teeth, so the skull covers the hinge.
    _ink(surf, [head([(94, 84)])[0], head([(124, 85)])[0], jawp([(121, 86)])[0],
                jawp([(100, 92)])[0]],
         _lerp_col((96, 18, 20), (255, 150, 40), glow * mouth), 1.0)
    if mouth > 0.2:
        _dot(surf, _rot(head([(108, 87)]), drop * 0.4)[0], (3 + 3 * mouth) * hs, hot)
    for x in (105, 111, 117):
        _ink(surf, jawp([(x - 2, 85), (x, 81), (x + 2, 85)]), MK_BONE, 1.0)
    _ink(surf, jaw, MK_SKIN)
    _stroke(surf, jawp([(101, 91), (116, 90)]), MK_SKIN_DK, 1.0)

    horn_far = head(_taper(_bez((106, 62), (103, 51), (98, 45), (91, 42)), 6, 0.6))
    _ink(surf, horn_far, MK_BONE_DK, 1.4)

    _ink(surf, skull, MK_SKIN)
    _ink(surf, head([(97, 70), (106, 62), (103, 68)]), MK_SKIN_LT, 0.0)
    for x in (103, 109, 115, 121):                       # upper teeth
        size = 5 if x == 115 else 3.4
        _ink(surf, head([(x - 2, 85.5), (x, 85.5 + size), (x + 2, 85.5)]),
             MK_BONE, 1.0)
    _stroke(surf, head([(97, 78), (103, 80)]), MK_SKIN_DK, 1.2)   # cheek
    _stroke(surf, head([(96, 82), (101, 83)]), MK_SKIN_DK, 1.2)
    ex, ey = head([(110, 73)])[0]
    rx, ry = 5.0 * hs, 3.9 * hs
    pygame.draw.ellipse(surf, MK_INK, ((ex - rx) * S, (ey - ry) * S, 2 * rx * S,
                                       2 * ry * S))
    pygame.draw.ellipse(surf, hot, ((ex - rx + 1.2) * S, (ey - ry + 1.2) * S,
                                    (2 * rx - 2.4) * S, (2 * ry - 2.4) * S))
    pygame.draw.ellipse(surf, MK_INK, ((ex - 0.8) * S, (ey - ry + 1.2) * S,
                                       1.8 * S, (2 * ry - 2.4) * S))
    _ink(surf, head([(101, 64), (113, 66), (120, 74), (104, 68.5)]),  # brow,
         MK_SKIN_DK, 1.3)                               # slanting down: a scowl
    nose = head([(122.5, 75.5)])[0]
    pygame.draw.ellipse(surf, MK_INK, ((nose[0] - 2) * S, (nose[1] - 1.2) * S,
                                       4 * S, 2.4 * S))
    _dot(surf, nose, 0.7, warm)

    horn = head(_taper(_bez((100, 65), (95, 53), (89, 47), (81, 45)), 8, 0.8))
    _ink(surf, horn, MK_BONE, 1.6)
    for i in (3, 6):                                     # growth rings
        _stroke(surf, [horn[i], horn[-1 - i]], MK_BONE_DK, 1.0)

    # An iron crown, a size too small, with an ember set in it.
    _ink(surf, head([(99, 61), (113, 58.5), (116, 49), (111.5, 54), (107.5, 45.5),
                     (104, 54.5), (99.5, 50.5)]), MK_IRON, 1.4)
    _stroke(surf, head([(100.5, 59), (112, 57)]), MK_IRON_LT, 1.0)
    gem = head([(106.5, 57.5)])[0]
    _dot(surf, gem, 2.2, MK_INK)
    _dot(surf, gem, 1.4, hot)

    return pygame.transform.smoothscale(
        surf, (round(MK_W * MK_SCALE), round(MK_H * MK_SCALE)))


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
    """Any mushroom: kind is "mushroom" (grow), "life", "spring", "gold" or "rot"."""
    W = H = 30

    def __init__(self, x, y, kind="mushroom"):
        self.x, self.y = float(x), float(y)
        self.kind = kind
        self.vx, self.vy = ITEM_SPEED.get(kind, MUSHROOM_SPEED), 0.0
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
            self.vy = -GOLD_BOUNCE if self.kind == "gold" else 0.0
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
        self.landed_at = -9.0                    # for the boss's landing squash
        self.hurt_until = 0.0                    # and his hit flash
        self.collapsed = False                   # a skeleton, knocked into a heap
        self.rise_at = 0.0                       # and when it gets back up
        if kind == "skeleton":
            self.H = 40                          # a head taller than a goomba
        speed = {"goomba": ENEMY_SPEED, "koopa": KOOPA_SPEED,
                 "buzzy": ENEMY_SPEED, "cheep": CHEEP_SPEED,
                 "spiny": ENEMY_SPEED * 1.2,
                 "skeleton": SKELETON_SPEED}.get(kind, 0.0)
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
                             "bullet", "skeleton")   # not spiny (spikes), not
                                                     # bowser (armoured)

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
        # Here a shell is scenery until you stomp it again. A heap of bones is
        # scenery too, until it stands back up.
        return self.is_platform or self.submerged or self.shell or self.collapsed

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

        if self.kind == "skeleton" and self.collapsed:
            if now < self.rise_at:
                return
            if abs(self.target_x - self.x) < SKELETON_CLEAR:
                self.rise_at = now + 0.2          # wait for room, still rattling
                return
            self.collapsed = False
            # Up, and after whoever knocked it down.
            self.vx = math.copysign(SKELETON_SPEED, self.target_x - self.x)
            return

        if self.kind == "bowser":
            self.vy = min(self.vy + GRAVITY, MAX_FALL)
            self.y += self.vy
            if self.y >= self.home_y:
                if self.vy > 0:
                    self.landed_at = now
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

    def stomped(self, now=0.0):
        """Returns True if the stomp killed it outright."""
        if self.kind == "skeleton":
            # Bones do not stay down. Only fire finishes one.
            self.collapsed = True
            self.rise_at = now + SKELETON_REST
            return False
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
            self.spring_until = 0.0       # a pipe trip keeps these, a death not
            self.gold_until = 0.0

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
        boost = SPRING_BOOST if now < self.spring_until else 1.0
        if self.grounded or coyote:
            self.vy = -(JUMP_MIN + (JUMP_MAX - JUMP_MIN) * strength) * boost
            self.grounded = False
            self.jumps_used = 1
            return "JUMP"
        if self.jumps_used < 2:
            self.vy = -(JUMP_MIN + (JUMP_MAX - JUMP_MIN) * strength) * \
                DOUBLE_JUMP_FACTOR * boost
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
        self.gates = list(data.get("gates", []))
        self.checkpoints = list(data.get("checkpoints", []))
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
        if ch not in LV.QBLOCKS and ch != LV.BRICK:
            return None
        self.bumps[(col, row)] = BUMP_TIME
        if ch in LV.QBLOCKS:
            self.rows[row][col] = LV.USED
            return LV.QBLOCKS[ch]
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
                 log_path=None, fullscreen=False):
        pygame.init()
        pygame.display.set_caption("Gesture Platformer")
        # SCALED: the game always draws at 1280x720 and the GPU stretches that to
        # the screen, so full screen costs no extra drawing and every layout stays
        # put. The aspect ratio is kept, so a 16:10 Mac screen gets thin bars.
        # One window per process: a SCALED window cannot always be created twice
        # (the headless tests build a fresh Game per level), so reuse it.
        self.fullscreen = fullscreen
        self.screen = pygame.display.get_surface() or pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.SCALED | (pygame.FULLSCREEN if fullscreen else 0))
        pygame.mouse.set_visible(not fullscreen)
        self._preview_seq, self._preview = None, None
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
            "life": build_sprite(LIFE_MUSH_PIXELS, 4),
            "spring": build_sprite(SPRING_MUSH_PIXELS, 4),
            "gold": build_sprite(GOLD_MUSH_PIXELS, 4),
            "rot": build_sprite(ROT_MUSH_PIXELS, 4),
            "skel_a": build_sprite(SKELETON_A, 4),
            "skel_b": build_sprite(SKELETON_B, 4),
            "skel_a_r": build_sprite(SKELETON_A, 4, flip=True),
            "skel_b_r": build_sprite(SKELETON_B, 4, flip=True),
            "bones": build_sprite(BONES_PIXELS, 4),
        }
        self.sky_cache = {}
        self.mk_cache = {}               # boss frames, by (mouth, glow) step

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
        self.run_start = time.perf_counter()
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
        self.title_at = None             # set on the first frame of play
        self.checkpoint = None           # column of the last banner passed
        # Gems are counted across the level AND its bonus room, which is rebuilt
        # every time you go in -- so the ones taken are remembered by position.
        self.gems_found = set()
        bonus = LV.load_bonus(self.main.name)
        self.gems_total = sum("".join(d["rows"]).count(LV.GEM)
                              for d in (self.main.data, bonus) if d)

    def spawn_entities(self, now=0.0):
        self.enemies = [Enemy(c * TILE + 4, (r + 1) * TILE, kind, now)
                        for c, r, kind in self.level.data["enemies"]]
        self.fireballs = []
        self.items = []
        self.hammers = []

    def respawn_point(self):
        """The last checkpoint banner passed, or the start."""
        if self.checkpoint is not None:
            return (self.checkpoint * TILE, LV.FLOOR * TILE)
        return self.main.start

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
            for name, c, r in self.gems_found:
                if name == self.bonus.name:
                    self.bonus.rows[r][c] = LV.EMPTY
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
                    if event.key == pygame.K_f or (
                            event.key == pygame.K_ESCAPE and self.fullscreen):
                        self.toggle_fullscreen()
                    elif event.key in (pygame.K_ESCAPE, pygame.K_q):
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

    def toggle_fullscreen(self):
        pygame.display.toggle_fullscreen()
        self.fullscreen = not self.fullscreen
        pygame.mouse.set_visible(not self.fullscreen)

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
                    self.hero.spawn = self.respawn_point()
                    self.hero.reset()
                    self.spawn_entities(now)
                    self.cam_x = clamp(self.hero.x - WIDTH * 0.42, 0,
                                       max(0, self.level.pw - WIDTH))
                    self.state = "play"
            return
        if self.state in ("won", "gameover", "complete"):
            if self.state == "won" and now >= self.state_until:
                self.next_level(now)
            return

        if self.title_at is None:
            self.title_at = now
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

        # Checkpoint banners: pass one, and that is where a death brings you back.
        if not self.in_bonus:
            for col in self.level.checkpoints:
                if (self.checkpoint is None or col > self.checkpoint) and \
                        h.x + h.W * 0.5 >= (col + 0.5) * TILE:
                    self.checkpoint = col
                    self.score += 200
                    self._toast("CHECKPOINT!", now, 1.2)

        self.step_world(dt, now)
        if self.state != "play":
            return
        if self.hazard_hits(now):
            return
        self.collide_entities(now)
        if self.state != "play":
            return
        self.collect_coins(now)
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
            if e.kind in ("lakitu", "skeleton"):
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

    def collect(self, item, now):
        """What each mushroom does when you touch it."""
        h = self.hero
        kind = item.kind
        if kind == "life":
            self.lives += 1
            self.score += 1000
            self._toast("HEART! +1 LIFE", now)
        elif kind == "spring":
            h.spring_until = now + SPRING_TIME
            self.score += 500
            self._toast(f"SPRING! big jumps for {SPRING_TIME:.0f}s", now)
        elif kind == "gold":
            h.gold_until = now + GOLD_TIME
            self.score += 500
            self._toast(f"GOLD! smash through enemies for {GOLD_TIME:.0f}s", now)
        elif kind == "rot":
            if now < h.gold_until:
                self.score += 100             # gold shrugs it off like anything else
                self._toast("+100", now, 0.7)
                return
            self.hurt(now)
            if self.state == "play":
                self._toast("ROTTEN! -- burn those, or jump them", now)
        elif h.grow(now):
            self.score += 1000
            self._toast("MUSHROOM! +1000", now)
        else:
            self.score += 200

    def collide_entities(self, now):
        h = self.hero
        hr = h.rect()
        for it in self.items:
            if not it.alive or it.emerging > 0:
                continue
            if it.kind == "rot":
                # Fire burns a rot mushroom: the always-available answer to it.
                for f in self.fireballs:
                    if f.alive and f.rect().colliderect(it.rect()):
                        f.alive = it.alive = False
                        self.score += 100
                        self._toast("+100", now, 0.7)
                        break
            if it.alive and hr.colliderect(it.rect()):
                it.alive = False
                self.collect(it, now)
                if self.state != "play":
                    return

        gold = now < h.gold_until
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
                        e.hurt_until = now + 0.2
                        if e.hp > 0:
                            self._toast(f"MAGMA KING: {e.hp} more!", now, 0.9)
                            continue
                        self.score += 5000
                        self._toast("MAGMA KING DOWN!", now, 2.0)
                    e.dying = 0.25
                    self.score += 100
                    self._toast("+100", now, 0.7)
            if e.dying > 0:
                continue
            if not hr.colliderect(er):
                continue
            if gold:
                # Untouchable: anything you can beat, you beat by walking into it.
                # Hazards (firebars, podoboos, cannons) and the boss just do not
                # hurt you -- gold is not a way to skip the last fight.
                if not e.armoured and e.kind not in ("firebar", "podoboo", "cannon"):
                    e.dying = 0.25
                    self.score += 200
                    self._toast("+200", now, 0.7)
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
                killed = e.stomped(now)
                h.vy = -STOMP_BOUNCE
                h.jumps_used = min(h.jumps_used, 1)
                self.score += 100
                self._toast("+100", now, 0.7)
                if not killed:
                    self._toast("bones! burn it, or it gets back up" if
                                e.kind == "skeleton" else "shell!", now,
                                1.4 if e.kind == "skeleton" else 0.7)
            else:
                self.hurt(now)
                return

    def hurt(self, now):
        """Big Mario shrinks; small Mario dies."""
        h = self.hero
        if now < h.invuln_until or now < h.gold_until:
            return
        if h.big:
            h.shrink(now)
            self._toast("ouch -- shrunk", now)
        else:
            self.kill(now, "hit")

    def collect_coins(self, now=0.0):
        hr = self.hero.rect()
        for col in range(hr.left // TILE, hr.right // TILE + 1):
            for row in range(hr.top // TILE, hr.bottom // TILE + 1):
                if self.level.take_coin(col, row):
                    self.score += 50
                    self.coins_taken += 1
                elif self.level.tile_at(col, row) == LV.GEM:
                    self.level.rows[row][col] = LV.EMPTY
                    self.take_gem(col, row, now)

    def take_gem(self, col, row, now):
        """Ember gems: three to a level from World 5. All three is a life."""
        self.gems_found.add((self.level.name, col, row))
        got = len(self.gems_found)
        self.score += 1000
        if got >= self.gems_total:
            self.lives += 1
            self._toast(f"ALL {self.gems_total} EMBER GEMS! +1 LIFE", now, 2.2)
        else:
            self._toast(f"EMBER GEM {got}/{self.gems_total}", now, 1.4)

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
            elif what in ITEM_SPEED:
                self.items.append(Mushroom(col * TILE + 5, row * TILE + TILE, what))
                self._toast({"mushroom": "a mushroom!",
                             "life": "a heart mushroom!",
                             "spring": "a spring mushroom!",
                             "gold": "a GOLD mushroom!",
                             "rot": "uh oh -- a rotten one. burn it!"}[what], now)
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
        self.draw_loop_gates(ox, now)
        self.draw_entities(ox, now)
        self.draw_weather(th, ox, now)
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
        if th["stars"]:
            for i in range(60):
                sx = (i * 197 - int(ox * 0.1)) % WIDTH
                sy = HUD_H + (i * 83) % 260
                pygame.draw.circle(self.screen, (200, 230, 255), (sx, sy), 1)
        backdrop = th.get("backdrop")
        if backdrop:
            getattr(self, f"draw_backdrop_{backdrop}")(ox, now)
        if th["hills"]:
            near, far = th.get("hill_cols", ((60, 150, 60), (78, 168, 74)))
            for i in range(-1, WIDTH // 320 + 3):
                bx = i * 320 - int(ox * 0.35) % 320
                base = HUD_H + LV.FLOOR * TILE
                pygame.draw.polygon(self.screen, near, [
                    (bx, base), (bx + 90, base - 78), (bx + 180, base)])
                pygame.draw.polygon(self.screen, far, [
                    (bx + 150, base), (bx + 215, base - 48), (bx + 280, base)])
        if th["clouds"]:
            col = th.get("cloud_col", (250, 250, 255))
            for i in range(-1, WIDTH // 260 + 3):
                cx = i * 260 - int(ox * 0.18) % 260
                cy = 110 + (i % 3) * 46
                for dx, dy, rr in ((0, 0, 22), (24, 6, 18), (-22, 6, 16)):
                    pygame.draw.circle(self.screen, col, (cx + dx, cy + dy), rr)

    # ---- backdrops for Worlds 5-8. All positions come from an index and the
    # clock, never random numbers, so a frame is reproducible for the tests. ----
    def draw_backdrop_moon(self, ox, now):
        """A low harvest moon over 5-3."""
        cx, cy = WIDTH - 260 - int(ox * 0.04) % 80, HUD_H + 120
        glow = pygame.Surface((260, 260), pygame.SRCALPHA)
        for r, a in ((128, 18), (100, 26), (80, 40)):
            pygame.draw.circle(glow, (255, 190, 110, a), (130, 130), r)
        self.screen.blit(glow, (cx - 130, cy - 130))
        pygame.draw.circle(self.screen, (252, 196, 120), (cx, cy), 62)
        for dx, dy, r in ((-18, -14, 11), (20, 8, 8), (-6, 24, 6), (26, -22, 5)):
            pygame.draw.circle(self.screen, (232, 170, 100), (cx + dx, cy + dy), r)

    def _ridge(self, ox, rate, spacing, height, base, col, cap=None, seed=0):
        """A repeating band of peaks; cap colours the snow on top."""
        for i in range(-1, WIDTH // spacing + 3):
            k = (i + int(ox * rate) // spacing + seed) % 5
            bx = i * spacing - int(ox * rate) % spacing
            h = height * (0.65 + 0.35 * ((k * 37) % 10) / 10)
            peak = (bx + spacing // 2, base - h)
            pygame.draw.polygon(self.screen, col,
                                [(bx - spacing // 3, base), peak,
                                 (bx + spacing + spacing // 3, base)])
            if cap:
                t = 0.28
                pygame.draw.polygon(self.screen, cap, [
                    peak,
                    (peak[0] - (spacing // 2 + spacing // 3) * t, peak[1] + h * t),
                    (peak[0] - 6, peak[1] + h * t * 0.7),
                    (peak[0] + 4, peak[1] + h * t),
                    (peak[0] + (spacing // 2 + spacing // 3) * t, peak[1] + h * t)])

    def draw_backdrop_mountains(self, ox, now):
        base = HUD_H + LV.FLOOR * TILE
        self._ridge(ox, 0.12, 300, 300, base, (168, 182, 210), (246, 250, 255), 0)
        self._ridge(ox, 0.28, 220, 190, base, (118, 134, 162), (232, 240, 250), 2)

    def draw_backdrop_aurora(self, ox, now):
        """Northern lights over 6-3, then the peaks in silhouette."""
        sky = pygame.Surface((WIDTH, 300), pygame.SRCALPHA)
        for band, (col, y0, amp) in enumerate((((60, 255, 170), 90, 30),
                                               ((80, 200, 255), 130, 24),
                                               ((170, 120, 255), 70, 18))):
            pts = []
            for i in range(0, WIDTH + 40, 40):
                x = i
                y = y0 + amp * math.sin(i * 0.006 + now * 0.5 + band * 1.7) \
                    + 10 * math.sin(i * 0.02 - now * 0.9 + band)
                pts.append((x, y))
            for w, a in ((26, 22), (14, 40), (5, 90)):
                pygame.draw.lines(sky, col + (a,), False, pts, w)
        self.screen.blit(sky, (0, HUD_H))
        base = HUD_H + LV.FLOOR * TILE
        self._ridge(ox, 0.12, 300, 300, base, (26, 34, 58), (90, 104, 136), 0)
        self._ridge(ox, 0.28, 220, 190, base, (16, 22, 40), (70, 84, 112), 2)

    def draw_backdrop_volcanoes(self, ox, now):
        """World 8's skyline: volcanoes with glowing mouths and smoke."""
        base = HUD_H + LV.FLOOR * TILE
        spacing = 520
        for i in range(-1, WIDTH // spacing + 3):
            bx = i * spacing - int(ox * 0.15) % spacing
            top = base - 250 - (i % 2) * 40
            pygame.draw.polygon(self.screen, (26, 8, 10), [
                (bx - 140, base), (bx + 120, top), (bx + 170, top),
                (bx + 430, base)])
            pulse = 0.6 + 0.4 * math.sin(now * 2 + i)
            pygame.draw.polygon(self.screen, (int(255 * pulse), int(110 * pulse), 30), [
                (bx + 120, top), (bx + 170, top), (bx + 160, top + 12),
                (bx + 130, top + 12)])
            for k in range(6):                       # a smoke plume, drifting
                t = (now * 0.25 + k / 6) % 1.0
                r = int(14 + 30 * t)
                sm = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(sm, (70, 50, 52, int(120 * (1 - t))), (r, r), r)
                self.screen.blit(sm, (bx + 145 - r + int(60 * t), top - 10 - int(170 * t) - r))
        self._ridge(ox, 0.3, 260, 120, base, (18, 6, 8), None, 1)

    def draw_backdrop_torches(self, ox, now):
        """Torches along the castle wall, lit and flickering."""
        for c in range(max(0, ox // TILE - 1), (ox + WIDTH) // TILE + 2):
            if c % 9 != 4:
                continue
            x = c * TILE - ox + TILE // 2
            y = HUD_H + (LV.FLOOR - 6) * TILE
            pygame.draw.rect(self.screen, (60, 46, 40), (x - 3, y, 6, 22))
            pygame.draw.rect(self.screen, (90, 70, 60), (x - 8, y + 2, 16, 4))
            f = 0.75 + 0.25 * math.sin(now * 13 + c * 1.7) * math.sin(now * 7 + c)
            glow = pygame.Surface((120, 120), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 140, 50, int(40 * f)), (60, 60), 56)
            self.screen.blit(glow, (x - 60, y - 66))
            pygame.draw.polygon(self.screen, (255, 140, 40), [
                (x - 7, y), (x, y - int(22 * f)), (x + 7, y)])
            pygame.draw.polygon(self.screen, (255, 230, 140), [
                (x - 3, y), (x, y - int(12 * f)), (x + 3, y)])

    def draw_weather(self, th, ox, now):
        """Foreground weather for Worlds 5-8: a level's own, or its theme's."""
        kind = self.level.data.get("weather") or th.get("weather")
        if not kind:
            return
        s = self.screen
        top, span = HUD_H, HEIGHT - HUD_H
        if kind == "leaves":
            cols = ((230, 110, 40), (200, 70, 40), (240, 180, 60))
            for i in range(36):
                x = (i * 211 + now * 40 - ox * 0.6) % (WIDTH + 40) - 20
                x += 18 * math.sin(now * 1.5 + i)
                y = top + (i * 97 + now * (34 + i % 5 * 6)) % span
                a = now * 3 + i
                dx, dy = 5 * math.cos(a), 2.5 * math.sin(a)
                pygame.draw.polygon(s, cols[i % 3], [(x - dx, y - dy), (x + dy, y - dx),
                                                     (x + dx, y + dy), (x - dy, y + dx)])
        elif kind == "snow":
            for i in range(70):
                x = (i * 137 - ox * (0.3 + (i % 3) * 0.2)) % WIDTH
                x += 10 * math.sin(now * 0.9 + i)
                y = top + (i * 61 + now * (24 + i % 4 * 10)) % span
                pygame.draw.circle(s, (250, 252, 255), (int(x), int(y)), 1 + i % 2)
        elif kind == "rain":
            for i in range(110):
                x = (i * 157 - ox * 0.9 - now * 120) % (WIDTH + 60)
                y = top + (i * 71 + now * 640) % span
                pygame.draw.line(s, (170, 186, 214), (x, y), (x - 5, y + 16), 1)
            phase = now % 7.3                        # lightning, now and then
            if phase < 0.07 or 0.15 < phase < 0.21:
                flash = pygame.Surface((WIDTH, HEIGHT - HUD_H), pygame.SRCALPHA)
                flash.fill((230, 236, 255, 90))
                s.blit(flash, (0, HUD_H))
        elif kind == "fireflies":
            for i in range(28):
                x = (i * 173 - ox * 0.5 + 30 * math.sin(now * 0.6 + i * 2)) % WIDTH
                y = top + 120 + (i * 53) % (span - 200) + 20 * math.sin(now * 0.8 + i)
                k = 0.5 + 0.5 * math.sin(now * 3 + i * 1.3)
                if k < 0.25:
                    continue
                glow = pygame.Surface((16, 16), pygame.SRCALPHA)
                pygame.draw.circle(glow, (220, 255, 120, int(90 * k)), (8, 8), 7)
                s.blit(glow, (int(x) - 8, int(y) - 8))
                pygame.draw.circle(s, (240, 255, 170), (int(x), int(y)), 1)
        elif kind in ("ash", "embers"):
            if kind == "ash":
                for i in range(50):
                    x = (i * 149 - ox * 0.5 + 14 * math.sin(now * 0.7 + i)) % WIDTH
                    y = top + (i * 67 + now * (20 + i % 4 * 7)) % span
                    pygame.draw.circle(s, (120, 110, 110), (int(x), int(y)), 1 + i % 2)
            for i in range(26 if kind == "ash" else 40):
                x = (i * 191 - ox * 0.7 + 12 * math.sin(now * 1.3 + i)) % WIDTH
                t = (now * (0.10 + (i % 5) * 0.03) + i * 0.137) % 1.0
                y = HEIGHT - t * span
                k = 1 - t
                pygame.draw.circle(s, (255, int(120 + 100 * k), 40), (int(x), int(y)),
                                   1 + (i % 3 == 0))
        elif kind == "bubbles":
            for i in range(30):
                x = (i * 167 - ox * 0.8 + 8 * math.sin(now * 2 + i)) % WIDTH
                y = HEIGHT - (i * 59 + now * (40 + i % 4 * 12)) % span
                pygame.draw.circle(s, (190, 230, 255), (int(x), int(y)), 2 + i % 3, 1)

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
        elif ch in LV.QBLOCKS:
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
        elif ch == LV.GEM:
            bob = math.sin(now * 3 + c) * 3
            cx, cy = x + TILE // 2, y + TILE // 2 + bob
            glow = pygame.Surface((48, 48), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 90, 40, int(50 + 30 * math.sin(now * 4 + c))),
                               (24, 24), 22)
            s.blit(glow, (cx - 24, cy - 24))
            gem = [(cx, cy - 13), (cx + 10, cy - 4), (cx, cy + 13), (cx - 10, cy - 4)]
            pygame.draw.polygon(s, (220, 40, 40), gem)
            pygame.draw.polygon(s, (255, 120, 80), [(cx, cy - 13), (cx + 10, cy - 4),
                                                     (cx, cy - 1), (cx - 10, cy - 4)])
            pygame.draw.polygon(s, (120, 10, 20), gem, 2)
            if int(now * 2 + c) % 3 == 0:                      # a glint
                pygame.draw.line(s, (255, 255, 230), (cx - 4, cy - 8), (cx - 1, cy - 8))
        elif ch == LV.CHECK:
            taken = (not self.in_bonus and self.checkpoint is not None
                     and c <= self.checkpoint)
            pygame.draw.rect(s, (70, 70, 84), (x + 10, y + TILE - 6, 20, 6))
            pygame.draw.rect(s, (190, 190, 205), (x + 18, y - TILE * 2 + 8, 4,
                                                   TILE * 3 - 14))
            pygame.draw.circle(s, (230, 230, 240), (x + 20, y - TILE * 2 + 8), 4)
            wave = math.sin(now * 4 + c) * 3
            cloth = (255, 150, 50) if taken else (110, 124, 150)
            top = y - TILE * 2 + 12
            if taken:
                glow = pygame.Surface((90, 70), pygame.SRCALPHA)
                pygame.draw.ellipse(glow, (255, 160, 60, 50), (0, 0, 90, 70))
                s.blit(glow, (x - 4, top - 16))
            pygame.draw.polygon(s, cloth, [(x + 22, top), (x + 52 + wave, top + 8),
                                           (x + 46, top + 15), (x + 52 - wave, top + 24),
                                           (x + 22, top + 26)])
            pygame.draw.polygon(s, tuple(v // 2 for v in cloth),
                                [(x + 22, top), (x + 52 + wave, top + 8),
                                 (x + 46, top + 15), (x + 52 - wave, top + 24),
                                 (x + 22, top + 26)], 2)
        elif ch == LV.AXE:
            pygame.draw.rect(s, (150, 110, 60), (x + 16, y + 10, 6, 28))
            pygame.draw.polygon(s, (220, 230, 240), [
                (x + 8, y + 6), (x + 32, y + 12), (x + 8, y + 18)])

    def draw_loop_gates(self, ox, now):
        """Light up the barrier at a maze fork, so the way on reads at a glance."""
        for col, above_row in getattr(self.level, "gates", []):
            x = col * TILE - ox
            if x < -TILE or x > WIDTH + TILE:
                continue
            top = (above_row + 1) * TILE + HUD_H
            bottom = LV.FLOOR * TILE + HUD_H
            glow = 0.5 + 0.5 * math.sin(now * 3)
            pygame.draw.rect(self.screen, (int(150 + 80 * glow), 60, 50),
                             (x + 2, top, TILE - 4, bottom - top), 3)
            arrow = self.small.render("", True, (255, 210, 140))
            self.screen.blit(arrow, (x - 22, top - 20))

    def draw_magma_king(self, e, ox, now):
        """The World 8 boss: winds up, roars, squashes on landing, flashes when hit."""
        # Mouth and glow are read off his breath timer, so the wind-up always
        # matches the jet that follows it -- a tell the player can learn.
        charge = clamp(1 - (e.breath_at - now) / 0.6, 0.0, 1.0)
        since = now - (e.breath_at - BOWSER_BREATH_EVERY)
        roar = clamp(1 - since / 0.4, 0.0, 1.0) if since >= 0 else 0.0
        mouth = max(charge * 0.4, roar)
        glow = clamp(0.35 + 0.12 * math.sin(now * 4) + 0.55 * max(charge, roar),
                     0.0, 1.0)
        key = (round(mouth * 8), round(glow * 8))
        spr = self.mk_cache.get(key)
        if spr is None:
            spr = self.mk_cache[key] = draw_magma_king(key[0] / 8, key[1] / 8)

        # Squash and stretch: crouch before a hop, stretch rising, flatten landing.
        sx = sy = 1.0
        if e.y < e.home_y - 0.5:
            sx, sy = (0.94, 1.07) if e.vy < 0 else (0.97, 1.03)
        elif now - e.landed_at < 0.2:
            k = 1 - (now - e.landed_at) / 0.2
            sx, sy = 1 + 0.12 * k, 1 - 0.12 * k
        elif e.timer - now < 0.18:
            k = 1 - max(0.0, e.timer - now) / 0.18
            sx, sy = 1 + 0.07 * k, 1 - 0.07 * k
        else:
            sy = 1 + 0.012 * math.sin(now * 3.2)          # breathing

        face = -1 if self.hero.x < e.x else 1             # same rule as his fire
        if face < 0:
            spr = pygame.transform.flip(spr, True, False)
        if (sx, sy) != (1.0, 1.0):
            w, h = spr.get_size()
            spr = pygame.transform.smoothscale(spr, (round(w * sx), round(h * sy)))
        if (now < e.hurt_until or e.dying > 0) and int(now * 30) % 2 == 0:
            spr = spr.copy()
            spr.fill((190, 190, 190), special_flags=pygame.BLEND_RGB_ADD)

        # The sprite canvas is in design px; MK_SCALE takes it to screen px.
        cx = e.x + e.W / 2 - ox
        anchor = (MK_ANCHOR_X if face > 0 else MK_W - MK_ANCHOR_X) * MK_SCALE * sx
        body_cx = cx - face * 16 * MK_SCALE              # middle of the shell
        foot = e.home_y + HUD_H
        shrink = clamp(1 - (e.home_y - e.y) / 160, 0.4, 1.0)
        sw, sh = round(120 * MK_SCALE), 16
        shadow = pygame.Surface((sw, sh), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), (sw / 2 * (1 - shrink),
                                                    sh / 2 * (1 - shrink),
                                                    sw * shrink, sh * shrink))
        self.screen.blit(shadow, (round(body_cx - sw / 2), round(foot - sh / 2)))
        self.screen.blit(spr, (round(cx - anchor),
                               round(e.y + HUD_H - MK_FOOT_Y * MK_SCALE * sy)))

        if e.dying <= 0:
            bar = max(0, e.hp) / BOWSER_HP
            top = round(e.y + HUD_H - 99 * MK_SCALE * sy - 14)
            bw = 84
            pygame.draw.rect(self.screen, (60, 20, 20), (body_cx - bw / 2, top, bw, 7))
            pygame.draw.rect(self.screen, (230, 60, 50),
                             (body_cx - bw / 2, top, round(bw * bar), 7))
            pygame.draw.rect(self.screen, MK_INK, (body_cx - bw / 2, top, bw, 7), 1)

    def draw_skeleton(self, e, r, ox, now):
        """Walking, a heap of bones, or a heap about to stand up (it rattles)."""
        if e.dying > 0 or e.collapsed:
            spr = self.spr["bones"]
            shake = 0
            if e.collapsed and e.rise_at - now < SKELETON_RATTLE:
                shake = (-2, 2)[int(now * 24) % 2]
            if e.dying > 0:                      # burned: the heap sinks away
                spr = spr.copy()
                spr.set_alpha(int(255 * e.dying / 0.25))
            self.screen.blit(spr, (r.x - ox + shake, r.bottom + HUD_H - spr.get_height()))
            return
        step = "a" if int(e.x / 12) % 2 == 0 else "b"
        key = f"skel_{step}" + ("_r" if e.vx > 0 else "")
        self.screen.blit(self.spr[key], (r.x - ox, r.y + HUD_H))

    def draw_entities(self, ox, now):
        s = self.screen
        for it in self.items:
            r = it.rect()
            s.blit(self.spr.get(it.kind, self.spr["mushroom"]), (r.x - ox, r.y + HUD_H))
            if it.kind == "gold" and int(now * 6) % 3 == 0:     # a glint
                gx, gy = r.x - ox + 6 + int(now * 40) % 18, r.y + HUD_H + 4
                pygame.draw.line(s, (255, 255, 230), (gx - 3, gy), (gx + 3, gy))
                pygame.draw.line(s, (255, 255, 230), (gx, gy - 3), (gx, gy + 3))
        for e in self.enemies:
            r = e.rect()
            # The boss is drawn well past his hitbox, so he gets a wider margin
            # or he pops in and out at the screen edge.
            margin = MK_W * MK_SCALE if e.kind == "bowser" else 40
            if r.right - ox < -margin or r.left - ox > WIDTH + margin:
                continue
            if e.kind == "bowser":
                self.draw_magma_king(e, ox, now)     # he flashes out, not flattens
                continue
            if e.kind == "skeleton":
                self.draw_skeleton(e, r, ox, now)
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
        # Powers show on the hero: a gold shimmer, or a teal tint for spring. Both
        # flicker for their last couple of seconds, so running out is never a
        # surprise.
        for until, tint, shimmer in ((h.gold_until, (150, 110, 0), True),
                                     (h.spring_until, (0, 60, 70), False)):
            left = until - now
            if left > 0 and (left > POWER_WARN or int(now * 8) % 2 == 0):
                if shimmer:
                    k = 0.6 + 0.4 * math.sin(now * 14)
                    tint = tuple(int(v * k) for v in tint)
                spr = spr.copy()
                spr.fill(tint, special_flags=pygame.BLEND_RGB_ADD)
                break
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
            pygame.draw.rect(self.screen, C_ROLE_MOVE,
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
        # Gems found, then running powers, on a second line under the score.
        x = 18
        if self.gems_total:
            label = self.small.render("GEMS", True, (255, 150, 120))
            self.screen.blit(label, (x, 40))
            x += label.get_width() + 8
            for i in range(self.gems_total):
                cx, cy = x + 6, 49
                pts = [(cx, cy - 7), (cx + 6, cy), (cx, cy + 7), (cx - 6, cy)]
                if i < len(self.gems_found):
                    pygame.draw.polygon(self.screen, (230, 50, 50), pts)
                pygame.draw.polygon(self.screen, (255, 150, 120), pts, 1)
                x += 16
            x += 18
        h = self.hero
        for name, until, col in (("SPRING", h.spring_until, (90, 220, 235)),
                                 ("GOLD", h.gold_until, (255, 205, 60))):
            left = until - now
            if left <= 0:
                continue
            surf = self.small.render(f"{name} {left:4.1f}s", True, col)
            self.screen.blit(surf, (x, 40))
            x += surf.get_width() + 22

        if self.tracker and snap:
            bx = WIDTH - CAM_PREVIEW[0] - 18
            roles = (("MOVE", "left", snap["steer_seen"], snap.get("steer_side"),
                      C_ROLE_MOVE),
                     ("FIRE", "right", snap["fire_seen"], snap.get("fire_side"),
                      C_ROLE_FIRE))
            for i, (label, want, seen, side, col) in enumerate(roles):
                y = 12 + i * 22
                # The role's own colour always; dimmed when that hand is not in
                # frame, so presence reads without the colour changing meaning.
                shown = col if seen else tuple(c // 3 for c in col)
                pygame.draw.circle(self.screen, shown, (bx - 104, y + 8), 6)
                text = f"{label} {want} hand" if seen else f"{label} -- no hand"
                self.screen.blit(self.small.render(text, True, shown),
                                 (bx - 92, y))
                # If detection disagrees with the hand this role is meant to be,
                # say so plainly and name the fix.
                if seen and side and side != want:
                    warn = self.small.render("press H", True, (255, 90, 90))
                    self.screen.blit(warn, (bx - 92 + 132, y))
            self.draw_steer_bar(snap, bx - 100, 44)
            if snap["frame"] is not None:
                surf = self.tracker_surface(snap)
                self.screen.blit(surf, (bx, HEIGHT - CAM_PREVIEW[1] - 12))
                pygame.draw.rect(self.screen, (40, 48, 68),
                                 (bx, HEIGHT - CAM_PREVIEW[1] - 12,
                                  CAM_PREVIEW[0], CAM_PREVIEW[1]), 1)
        elif not self.tracker:
            s = self.small.render(
                "keyboard: arrows, shift=run, space=jump, X=fire, down=duck/pipe",
                True, (150, 160, 180))
            self.screen.blit(s, (WIDTH - s.get_width() - 18, 22))

    def tracker_surface(self, snap):
        """The camera preview, rebuilt only when the camera has a new frame.

        The camera delivers 30 frames a second and the game draws 60, so
        converting on every draw did each frame's work twice. Shrinking in OpenCV
        first and handing pygame BGR directly is also several times cheaper than
        converting the full frame and smoothscaling it down.
        """
        if snap["seq"] != self._preview_seq:
            import cv2
            small = cv2.resize(snap["frame"], CAM_PREVIEW, interpolation=cv2.INTER_AREA)
            self._preview = pygame.image.frombuffer(small.tobytes(), CAM_PREVIEW, "BGR")
            self._preview_seq = snap["seq"]
        return self._preview

    def draw_victory(self, now):
        """The end of the game deserves more than a line of text.

        Sparks, a trophy, a rainbow title and the run's numbers. What was here
        before named the wrong number of worlds -- left over from when there were
        only two, and by the end simply untrue.
        """
        t = now - self.state_until + 8.0                # seconds since the win
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((6, 4, 26, 232))
        self.screen.blit(veil, (0, 0))

        if not hasattr(self, "_sparks"):
            self._sparks = [_Spark(i) for i in range(VICTORY_SPARKS)]
        for sp in self._sparks:
            sp.draw(self.screen, t)

        cx = WIDTH // 2
        # Sunburst behind the trophy.
        for k in range(16):
            a = t * 0.6 + k * math.pi / 8
            x2 = cx + math.cos(a) * 300
            y2 = 250 + math.sin(a) * 300
            pygame.draw.line(self.screen, (44, 34, 90), (cx, 250), (int(x2), int(y2)), 12)

        # Trophy.
        bob = int(math.sin(t * 2) * 6)
        base = 300 + bob
        pygame.draw.rect(self.screen, (196, 150, 40), (cx - 46, base - 70, 92, 60),
                         border_radius=10)
        pygame.draw.rect(self.screen, (250, 214, 90), (cx - 38, base - 64, 76, 46),
                         border_radius=8)
        for side in (-1, 1):
            pygame.draw.arc(self.screen, (196, 150, 40),
                            (cx + side * 60 - 22, base - 66, 44, 44),
                            -1.4 if side > 0 else 1.7, 1.7 if side > 0 else 4.5, 7)
        pygame.draw.rect(self.screen, (196, 150, 40), (cx - 12, base - 12, 24, 22))
        pygame.draw.rect(self.screen, (150, 110, 26), (cx - 40, base + 8, 80, 16),
                         border_radius=5)
        star = 0.6 + 0.4 * math.sin(t * 5)
        pygame.draw.circle(self.screen, (255, int(240 * star), 160), (cx, base - 42), 10)

        # Title, letter by letter in shifting colours.
        title = "YOU SAVED THE KINGDOM!"
        widths = [self.big_font.size(ch)[0] for ch in title]
        x = cx - sum(widths) // 2
        for i, ch in enumerate(title):
            wob = int(math.sin(t * 4 + i * 0.4) * 7)
            col = (int(180 + 75 * math.sin(t * 2 + i * 0.5)),
                   int(180 + 75 * math.sin(t * 2 + i * 0.5 + 2.1)),
                   int(180 + 75 * math.sin(t * 2 + i * 0.5 + 4.2)))
            self.screen.blit(self.big_font.render(ch, True, col),
                             (x, 390 + wob))
            x += widths[i]

        sub = self.font.render("all 8 worlds, 32 levels", True, (200, 210, 240))
        self.screen.blit(sub, (cx - sub.get_width() // 2, 452))

        elapsed = int(now - self.run_start) if hasattr(self, "run_start") else 0
        stats = [f"SCORE  {self.score:06d}", f"COINS  {self.coins_taken}",
                 f"LIVES LEFT  {max(0, self.lives)}",
                 f"TIME  {elapsed // 60}:{elapsed % 60:02d}"]
        y = 500
        for line in stats:
            surf = self.font.render(line, True, C_COIN)
            self.screen.blit(surf, (cx - surf.get_width() // 2, y))
            y += 30
        hint = self.small.render("R to play again    Esc to quit", True, (170, 180, 210))
        self.screen.blit(hint, (cx - hint.get_width() // 2, y + 12))

    def draw_title_card(self, now):
        """'WORLD 6-2 / THE FROSTPEAK HIGHLANDS', fading out as the level starts."""
        name = WORLD_NAMES.get(self.world)
        if not name or self.in_bonus or self.title_at is None or self.state != "play":
            return
        t = now - self.title_at
        if t >= TITLE_CARD_SECS:
            return
        fade = clamp(min(t / 0.25, (TITLE_CARD_SECS - t) / 0.6), 0.0, 1.0)
        box = pygame.Rect(0, HUD_H + 150, WIDTH, 130)
        band = pygame.Surface(box.size, pygame.SRCALPHA)
        band.fill((8, 8, 14, int(170 * fade)))
        self.screen.blit(band, box.topleft)
        for text, font, col, dy in ((f"WORLD {self.world}-{self.level_no}",
                                     self.big_font, C_COIN, 22),
                                    (name, self.font, C_TEXT, 84)):
            surf = font.render(text, True, col)
            surf.set_alpha(int(255 * fade))
            self.screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, box.y + dy))

    def draw_overlays(self, now):
        self.draw_title_card(now)
        if self.toast and now < self.toast_until:
            s = self.font.render(self.toast, True, (255, 255, 255))
            self.screen.blit(s, (WIDTH // 2 - s.get_width() // 2, HUD_H + 20))
        banner = None
        if self.state == "won":
            last = (self.world, self.level_no) == (len(LV.WORLDS),
                                                   len(LV.WORLDS[-1]))
            nxt = "" if last else "next level..."
            sub = nxt or "R to restart"
            if self.gems_total:
                sub = f"ember gems {len(self.gems_found)}/{self.gems_total}     {sub}"
            banner = (f"{self.level.name} CLEAR!", sub)
        elif self.state == "complete":
            self.draw_victory(now)
            return
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


VICTORY_SPARKS = 90


class _Spark:
    """One drifting spark on the victory screen. Positions are derived from an
    index rather than random numbers, so the celebration is identical every time
    and the whole file stays reproducible for tests."""

    def __init__(self, i):
        self.i = i
        self.x = (i * 137) % WIDTH
        self.base_y = (i * 71) % HEIGHT
        self.speed = 18 + (i % 7) * 9
        self.size = 2 + (i % 4)
        self.hue = i % 4

    COLORS = ((255, 226, 120), (255, 255, 255), (255, 170, 90), (170, 230, 255))

    def draw(self, screen, t):
        y = int((self.base_y - t * self.speed) % (HEIGHT + 40)) - 20
        # Clamped: 0.45 + 0.55*sin dips just below zero, and a negative colour
        # component is a hard ValueError out of pygame rather than a dim spark.
        twinkle = clamp(0.45 + 0.55 * math.sin(t * 3 + self.i), 0.05, 1.0)
        col = tuple(int(clamp(c * twinkle, 0, 255)) for c in self.COLORS[self.hue])
        r = max(1, int(self.size * (0.6 + 0.4 * twinkle)))
        pygame.draw.circle(screen, col, (self.x, y), r)
        if self.size >= 4:                       # the big ones get a cross-glint
            pygame.draw.line(screen, col, (self.x - r * 3, y), (self.x + r * 3, y))
            pygame.draw.line(screen, col, (self.x, y - r * 3), (self.x, y + r * 3))


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
    p.add_argument("--fullscreen", action="store_true",
                   help="start in full screen (or press F while playing)")
    p.add_argument("--cpu-tracking", action="store_true",
                   help="run hand tracking on the CPU instead of the GPU")
    return p.parse_args(argv)


if __name__ == "__main__":
    opts = parse_args(sys.argv[1:])
    path = opts.log
    if path == "auto":
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            time.strftime("play-log-%Y%m%d-%H%M%S.csv"))
    if opts.steer_zone:
        tracker_mod.STEER_ZONE = tuple(opts.steer_zone)
    if opts.cpu_tracking:
        tracker_mod.USE_GPU = False
    game = Game(world=opts.world, level=opts.level,
                use_camera=not opts.no_camera, camera_id=opts.camera,
                log_path=path, fullscreen=opts.fullscreen)
    if opts.swap_hands and game.tracker:
        game.tracker.swap_handedness = True
    game.run()
