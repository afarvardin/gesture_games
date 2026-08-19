"""World and level data for the gesture platformer, in the shape of Super Mario
Bros. World 1 and World 2: eight levels across four themes.

  1-1 overworld    goombas, ? blocks, pipes, a pipe you can enter, end stairs
  1-2 underground  ceilinged corridor, brick runs, coin rooms, exit pipe
  1-3 athletic     mushroom platforms and koopas over a gappy floor
  1-4 castle       lava pits, rising podoboos, an axe at the end
  2-1 overworld    longer, more pipes, a bonus room in the sky
  2-2 underwater   swim physics, cheep-cheeps, no jumping out
  2-3 bridge       long bridges over water with cheep-cheeps leaping through
  2-4 castle       longer castle with more lava and podoboos

Levels are BUILT, not typed. Hand-authoring 14x120 ASCII maps is how the first
draft ended up with pipes inside pits and platforms roofing every jump; a builder
plus a validator makes those mistakes impossible to commit. Each motif here
(pipe, stair, brick run, mushroom platform) knows how to place itself legally.
"""

ROWS = 14                  # 14 tiles tall; ground occupies the bottom two
FLOOR = 12                 # first ground row
SKY_TOP = 0

# Tiles. Anything in SOLID stops the player.
EMPTY = "."
GROUND = "#"               # earth / floor
STONE = "X"               # unbreakable block
BRICK = "B"               # breakable when big
QCOIN = "?"               # ? block holding a coin
QMUSH = "M"               # ? block holding a mushroom
USED = "U"                # a block that has been hit
PIPE = "P"                # pipe body (solid)
PIPE_IN = "D"             # pipe mouth you can enter by ducking on it
COIN = "o"
LAVA = "L"
WATER = "W"
BRIDGE = "="
POLE = "F"                # flagpole
CASTLE = "C"
AXE = "A"
START = "S"
EXIT = "T"                # pipe mouth that leaves a bonus room
STALK = "|"               # decorative mushroom-platform stalk, NOT solid --
                          # a solid stalk 8 tiles tall is an unjumpable wall
CANNON = "c"              # Bullet Bill cannon: solid, and it fires
VINE = "v"                # beanstalk: touch it to be carried to a sky area
LOOP = "@"                # 4-4's maze trigger: cross it low and you go back

SOLID = frozenset({GROUND, STONE, BRICK, QCOIN, QMUSH, USED, PIPE, PIPE_IN,
                   BRIDGE, EXIT, CANNON})
DEADLY = frozenset({LAVA})

# Themes drive palette and background; the engine reads these names.
THEME_OVERWORLD = "overworld"
THEME_UNDERGROUND = "underground"
THEME_CASTLE = "castle"
THEME_WATER = "water"
THEME_NIGHT = "night"      # World 3's overworld levels are after dark

# Jump geometry, mirrored from the engine so the validator can reason about it.
# Weakest flick the gesture layer can make carries ~3.6 tiles walking, ~7 running,
# and clears ~3.9 tiles of height.
MAX_GAP = 2                # tiles. 3 left only a 1.2x margin on the
                           # weakest walking jump and the bot died in
                           # pits it had every right to clear.
MAX_JUMP_UP = 5            # tiles of height the weakest flick clears (232px / 40)
MAX_PIPE_H = 3             # tiles; a 4-tall pipe cannot be jumped
HIGH_TIER = 6              # blocks this far up need a stepping stone below
STEP_REACH = 4             # columns to look sideways for that stepping stone
CLEAR_OF_GAP = 3           # tiles that pipes/platforms/enemies keep from a pit
# How far a walk-off carries you, per shelf height, at running speed:
#   sqrt(2 * height_px / GRAVITY) frames * RUN_SPEED / TILE
# A pit inside that distance means the shelf funnels you into it.
GRAVITY_PX = 0.58
RUN_PX = 6.2


def fall_reach_tiles(row):
    import math as _math
    height_px = max(1, (FLOOR - row)) * 40
    return _math.sqrt(2 * height_px / GRAVITY_PX) * RUN_PX / 40.0
LANDING_CLEAR = 8          # tiles PAST a pit that stay free of enemies. A running
                           # jump covers ~7 tiles, so an enemy just beyond a pit is
                           # standing exactly where the player comes down -- and
                           # landing flat-footed beside one is a death, not a stomp.


class Builder:
    """Places level motifs and refuses to place them illegally."""

    def __init__(self, width, theme, name, rows=ROWS):
        self.w, self.rows_n, self.theme, self.name = width, rows, theme, name
        self.g = [[EMPTY] * width for _ in range(rows)]
        self.enemies = []                  # (col, row, kind)
        self.gaps = set()                  # columns with no floor
        self.pipe_links = {}               # entrance col -> ("bonus"|"main", dest col)
        self.start_col = 2
        self.end = None                    # ("flag", col) | ("axe", col)
        self.water_cols = set()
        self.ceiling = False
        self.problems = []          # collected, so one run reports every mistake
        self.loops = []             # (trigger col, destination col, must-be-above row)
        self.no_reach = set()       # blocks exempt from the reachability rule

    # ---- terrain ----
    def floor(self, a=None, b=None, row=FLOOR):
        a = 0 if a is None else a
        b = self.w - 1 if b is None else b
        for c in range(a, b + 1):
            for r in range(row, self.rows_n):
                self.g[r][c] = GROUND
        return self

    def gap(self, a, b):
        if b - a + 1 > MAX_GAP:
            self.problems.append(f"gap {a}-{b} is wider than {MAX_GAP}")
        for c in range(a, b + 1):
            for r in range(FLOOR, self.rows_n):
                self.g[r][c] = EMPTY
            self.gaps.add(c)
        return self

    def lava(self, a, b):
        for c in range(a, b + 1):
            for r in range(FLOOR, self.rows_n):
                self.g[r][c] = LAVA
            self.gaps.add(c)               # deadly counts as a hole for placement
        return self

    def water(self, a=None, b=None, top=2):
        a = 0 if a is None else a
        b = self.w - 1 if b is None else b
        for c in range(a, b + 1):
            self.water_cols.add(c)
            for r in range(top, FLOOR):
                if self.g[r][c] == EMPTY:
                    self.g[r][c] = WATER
        return self

    def roof(self, a=None, b=None, rows=2):
        """Underground ceiling."""
        a = 0 if a is None else a
        b = self.w - 1 if b is None else b
        for c in range(a, b + 1):
            for r in range(rows):
                self.g[r][c] = GROUND
        self.ceiling = True
        return self

    # ---- blocks ----
    def run(self, row, col, pattern):
        """A horizontal run of blocks, e.g. 'BB?BB'."""
        for i, ch in enumerate(pattern):
            if ch != EMPTY:
                self.g[row][col + i] = ch
        return self

    def platform(self, row, col, width, kind=BRICK, coins_above=True):
        self._assert_clear_of_gaps(range(col, col + width), "platform")
        for c in range(col, col + width):
            self.g[row][c] = kind
        if coins_above and row - 1 >= 0:
            for c in range(col, col + width):
                if self.g[row - 1][c] in (EMPTY, WATER):
                    self.g[row - 1][c] = COIN
        return self

    def mushroom_platform(self, row, col, width):
        """The stalked platforms from the athletic levels.

        The stalk is decorative. Made solid it becomes a wall as tall as the
        platform, and anything above three tiles cannot be jumped -- the level
        stops being completable rather than becoming harder.
        """
        self.platform(row, col, width, kind=STONE, coins_above=True)
        stalk = col + width // 2
        for r in range(row + 1, FLOOR):
            if self.g[r][stalk] == EMPTY:
                self.g[r][stalk] = STALK
        return self

    def pipe(self, col, height=2, enterable=False, dest=None):
        """A two-tile-wide pipe standing on the floor.

        Height is capped at 3: four tiles is 160px, and the weakest flick only
        clears 158px, so a 4-tall pipe is a wall the player cannot pass.
        """
        if height > MAX_PIPE_H:
            self.problems.append(f"pipe at {col} is {height} tiles tall; the weakest "
                                 f"jump only clears {MAX_PIPE_H}")
        self._assert_clear_of_gaps((col, col + 1), "pipe")
        top = FLOOR - height
        for c in (col, col + 1):
            for r in range(top, FLOOR):
                self.g[r][c] = PIPE
        if enterable:
            self.g[top][col] = PIPE_IN
            self.g[top][col + 1] = PIPE_IN
            self.pipe_links[col] = dest
        return self

    def stairs(self, col, height, descend=False):
        """A staircase of stone blocks, as at the end of an overworld level."""
        cols = range(col, col + height)
        self._assert_clear_of_gaps(cols, "stairs")
        for i, c in enumerate(cols):
            h = (height - i) if descend else (i + 1)
            for r in range(FLOOR - h, FLOOR):
                self.g[r][c] = STONE
        return self

    def bridge(self, row, a, b):
        for c in range(a, b + 1):
            self.g[row][c] = BRIDGE
        return self

    def coins(self, row, col, n):
        # Water counts as empty here, or an underwater level ends up with no coins
        # at all -- water() runs first and fills every free cell.
        for c in range(col, col + n):
            if self.g[row][c] in (EMPTY, WATER):
                self.g[row][c] = COIN
        return self

    # ---- entities ----
    def enemy(self, col, kind="goomba", row=FLOOR - 1):
        for g in self.gaps:
            if col > g and col - g <= LANDING_CLEAR:
                self.problems.append(
                    f"{kind} at {col} is {col - g} tile(s) past the pit at {g}, "
                    f"inside the landing corridor")
                break
            if col < g and g - col <= CLEAR_OF_GAP:
                self.problems.append(
                    f"{kind} at {col} is only {g - col} tile(s) before the pit at {g}")
                break
        for up in range(1, 4):
            r = row - up
            if r >= 0 and self.g[r][col] in SOLID:
                self.problems.append(f"{kind} at {col} is trapped under "
                                     f"'{self.g[r][col]}' {up} row(s) above")
                break
        self.enemies.append((col, row, kind))
        return self

    def swimmer(self, col, row, kind="cheep"):
        self.enemies.append((col, row, kind))
        return self

    def flyer(self, col, row, kind="para"):
        """A paratroopa or a hammer bro, placed off the floor."""
        self.enemies.append((col, row, kind))
        return self

    def lift(self, col, row, kind="lift_v", span=4, width=3):
        """A moving platform. Vertical ones ride up and down, horizontal ones
        shuttle sideways. Recorded as an entity, not a tile, so it can move."""
        self.enemies.append((col, row, f"{kind}:{span}:{width}"))
        return self

    def cannon(self, col, height=2, face=-1):
        """A Bullet Bill cannon. Stands on the floor and fires horizontally.

        The barrel is solid, so it doubles as a small step; the bullets are
        entities, spawned by the engine on the cannon's own clock.
        """
        self._assert_clear_of_gaps((col,), "cannon")
        top = FLOOR - height
        for r in range(top, FLOOR):
            self.g[r][col] = CANNON
        self.no_reach.add((col, top))
        self.enemies.append((col, top, f"cannon:{face}"))
        return self

    def lakitu(self, col):
        """A cloud that shadows the player and drops spinies."""
        self.enemies.append((col, 3, "lakitu"))
        return self

    def bowser(self, col):
        """The final boss: hops, breathes fire, and throws hammers."""
        self.enemies.append((col, FLOOR - 1, "bowser"))
        return self

    def firebar(self, col, row, length=4):
        """A bar of fire rotating around a block, as in the castles."""
        self.g[row][col] = STONE
        self.no_reach.add((col, row))
        self.enemies.append((col, row, f"firebar:{length}"))
        return self

    def vine(self, col, dest):
        """A beanstalk that carries you to a sky area."""
        self.g[FLOOR - 1][col] = VINE
        self.pipe_links[col] = dest
        return self

    def loop(self, col, dest_col, above_row):
        """4-4's maze: crossing this column below `above_row` sends you back."""
        self.loops.append((col, dest_col, above_row))
        return self

    def podoboo(self, col):
        self.enemies.append((col, FLOOR - 1, "podoboo"))
        return self

    def start(self, col):
        self.start_col = col
        self.g[FLOOR - 1][col] = START
        return self

    def flag(self, col):
        if col + 5 >= self.w:
            self.problems.append(f"flag at {col} leaves no room for the castle in a "
                                 f"{self.w}-tile level (needs {col + 5})")
            return self
        self.g[FLOOR - 1][col] = POLE
        for r in range(FLOOR - 6, FLOOR - 1):
            if self.g[r][col] == EMPTY:
                self.g[r][col] = POLE
        self.g[FLOOR - 1][col + 4] = CASTLE
        self.end = ("flag", col)
        return self

    def axe(self, col):
        self.g[FLOOR - 1][col] = AXE
        self.end = ("axe", col)
        return self

    def exit_pipe(self, col, dest_col):
        """The mouth that returns you from a bonus room -- that is its 'end'."""
        self.g[FLOOR - 1][col] = EXIT
        self.pipe_links[col] = dest_col
        self.end = ("exit", col)
        return self

    # ---- checks ----
    def _assert_clear_of_gaps(self, cols, what):
        for c in cols:
            for g in self.gaps:
                if abs(c - g) <= CLEAR_OF_GAP:
                    self.problems.append(
                        f"{what} at column {c} is only {abs(c-g)} tile(s) from the "
                        f"pit at {g}")
                    return

    def validate(self):
        errs = list(self.problems)
        # Nothing may float over a hole.
        for c in range(self.w):
            for r in range(self.rows_n):
                ch = self.g[r][c]
                if ch in (PIPE, PIPE_IN, START, AXE, CASTLE, EXIT, VINE):
                    below = self.g[r + 1][c] if r + 1 < self.rows_n else GROUND
                    if below not in SOLID and below != LAVA:
                        errs.append(f"'{ch}' at ({c},{r}) floats over '{below}'")
                elif ch == POLE:
                    # A flagpole is decorative and stacks on itself; only its base
                    # has to be standing on something.
                    below = self.g[r + 1][c] if r + 1 < self.rows_n else GROUND
                    if below not in SOLID and below != POLE:
                        errs.append(f"flagpole at ({c},{r}) floats over '{below}'")
        # Gaps must be jumpable, and reachable from a standing start.
        runs, run_start = [], None
        for c in range(self.w + 1):
            hole = c in self.gaps or c >= self.w
            if hole and run_start is None:
                run_start = c
            elif not hole and run_start is not None:
                runs.append(c - run_start)
                run_start = None
        for width in runs:
            if width > MAX_GAP:
                errs.append(f"a pit {width} tiles wide is not walk-clearable")
        # Anything high enough to need a boost must have one within reach.
        for c in range(self.w):
            for r in range(self.rows_n):
                if self.g[r][c] not in (BRICK, QCOIN, QMUSH, STONE):
                    continue
                if FLOOR - r <= MAX_JUMP_UP or (c, r) in self.no_reach:
                    continue               # reachable off the floor, or a hazard mount
                found = False
                for cc in range(max(0, c - STEP_REACH),
                                min(self.w, c + STEP_REACH + 1)):
                    for rr in range(r + 2, min(self.rows_n, r + MAX_JUMP_UP + 1)):
                        if self.g[rr][cc] in SOLID:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    errs.append(f"block at ({c},{r}) is {FLOOR - r} tiles up with "
                                f"nothing to jump from")
        # A raised shelf must not tip you into a pit when you walk off its end.
        for c in range(self.w - 1):
            for r in range(self.rows_n):
                if FLOOR - r < 3 or self.g[r][c] not in SOLID:
                    continue
                if self.g[r][c + 1] in SOLID or (c, r) in self.no_reach:
                    continue                     # not an edge, or a hazard mount
                reach = int(fall_reach_tiles(r)) + 1
                for g in sorted(self.gaps):
                    if 0 < g - c <= reach:
                        errs.append(f"shelf ending at ({c},{r}) is {g - c} tile(s) "
                                    f"before the pit at {g}; a walk-off carries "
                                    f"{fall_reach_tiles(r):.1f} tiles")
                        break
        if self.end is None:
            errs.append("no level end")
        if not any(GROUND in row or BRIDGE in row for row in self.g):
            errs.append("no floor at all")
        return errs

    def build(self):
        errs = self.validate()
        if errs:
            raise AssertionError(f"{self.name}: " + "; ".join(errs))
        return {
            "name": self.name,
            "theme": self.theme,
            "rows": ["".join(r) for r in self.g],
            "enemies": list(self.enemies),
            "pipe_links": dict(self.pipe_links),
            "start_col": self.start_col,
            "end": self.end,
            "water": sorted(self.water_cols),
            "ceiling": self.ceiling,
            "loops": list(self.loops),
        }


# ===========================================================================
# WORLD 1
# Pit spacing is deliberate: every pit is 3 tiles, and every pipe, platform,
# staircase and enemy keeps CLEAR_OF_GAP tiles away from one, so no jump across a
# pit is ever roofed and no enemy can shove you into one. The builder enforces it.
# ===========================================================================
def w1_1():
    """Overworld. The opening beat: a lone goomba, the ? block cluster, pipes of
    rising height, a pit, then the end staircase."""
    b = Builder(120, THEME_OVERWORLD, "1-1")
    b.floor()
    b.gap(70, 71)
    b.start(3)
    b.enemy(16)
    b.run(FLOOR - 4, 16, QCOIN)
    b.run(FLOOR - 4, 20, "BB?BB")
    b.run(FLOOR - 8, 22, QCOIN)          # high tier, reachable off the row below
    b.coins(FLOOR - 9, 21, 3)
    b.pipe(28, height=2)
    b.pipe(38, height=3)
    b.enemy(33)
    b.enemy(34)
    b.pipe(48, height=3, enterable=True, dest="bonus")
    b.enemy(58)
    b.run(FLOOR - 4, 60, "BMB")
    b.coins(FLOOR - 5, 60, 3)
    b.platform(FLOOR - 8, 58, 4)
    b.enemy(64)
    b.run(FLOOR - 4, 78, "BB?BB")
    b.enemy(84)
    b.enemy(85)
    b.platform(FLOOR - 5, 86, 5)
    b.run(FLOOR - 4, 92, "?B?")
    b.enemy(96)
    b.stairs(100, 4)
    b.pipe(108, height=2)
    b.flag(112)
    return b.build()


def w1_1_bonus():
    """The coin room under the pipe."""
    b = Builder(24, THEME_UNDERGROUND, "1-1 bonus")
    b.floor()
    b.roof()
    b.start(2)
    for row in (FLOOR - 3, FLOOR - 5):
        b.coins(row, 4, 14)
    b.run(FLOOR - 4, 8, "BB?BB")
    b.exit_pipe(21, 52)
    return b.build()


def w1_2():
    """Underground. Ceilinged the whole way, brick shelves, coin seams, and the
    pipe out at the end."""
    b = Builder(120, THEME_UNDERGROUND, "1-2")
    b.floor()
    b.roof()
    b.start(3)
    b.enemy(15)
    b.run(FLOOR - 4, 10, "BBBBB")
    b.coins(FLOOR - 5, 10, 5)
    b.enemy(20)
    b.enemy(21)
    b.run(FLOOR - 4, 24, "BB?BB")
    b.run(FLOOR - 8, 26, "BBB")
    b.coins(FLOOR - 9, 26, 3)
    b.gap(36, 37)
    b.enemy(46)
    b.run(FLOOR - 4, 43, "BMB")
    b.platform(FLOOR - 5, 50, 6)
    b.enemy(58)
    b.enemy(59)
    b.run(FLOOR - 4, 62, "BBBB")
    b.coins(FLOOR - 5, 62, 4)
    b.gap(72, 73)
    b.run(FLOOR - 4, 79, "BB?BB")
    b.enemy(84)
    b.platform(FLOOR - 5, 88, 5)
    b.enemy(96)
    b.enemy(97)
    b.run(FLOOR - 4, 100, "BBBBBB")
    b.coins(FLOOR - 5, 100, 6)
    b.pipe(108, height=3)
    b.axe(114)
    return b.build()


def w1_3():
    """Athletic. Stalked mushroom platforms and koopas over a gappy floor.

    Pits at 20/38/56/74/92 leave nine clear columns between each pair, which is
    what a platform plus an enemy needs while staying CLEAR_OF_GAP from both.
    """
    b = Builder(120, THEME_OVERWORLD, "1-3")
    b.floor()
    for a in (20, 38, 56, 74, 92):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(10, "koopa")
    b.mushroom_platform(FLOOR - 5, 8, 4)
    b.mushroom_platform(FLOOR - 5, 27, 5)
    b.enemy(33, "koopa")
    b.mushroom_platform(FLOOR - 4, 45, 5)
    b.enemy(51, "goomba")
    b.mushroom_platform(FLOOR - 5, 63, 5)
    b.enemy(69, "koopa")
    b.mushroom_platform(FLOOR - 4, 81, 5)
    b.run(FLOOR - 4, 84, "BMB")
    b.mushroom_platform(FLOOR - 5, 99, 4)
    b.enemy(105, "koopa")
    b.stairs(108, 3)
    b.flag(114)
    return b.build()


def w1_4():
    """Castle. Lava pits, podoboos rising out of them, an axe at the end."""
    b = Builder(110, THEME_CASTLE, "1-4")
    b.floor()
    b.start(3)
    for a in (18, 34, 50, 66, 84):
        b.lava(a, a + 1)
    b.run(FLOOR - 4, 10, "XXX")
    b.podoboo(19)
    b.run(FLOOR - 5, 24, "XX")
    b.enemy(30)
    b.podoboo(35)
    b.platform(FLOOR - 5, 41, 4, kind=STONE)
    b.enemy(46)
    b.podoboo(51)
    b.run(FLOOR - 4, 58, "XXXX")
    b.enemy(62)
    b.podoboo(67)
    b.platform(FLOOR - 5, 74, 5, kind=STONE)
    b.enemy(80)
    b.podoboo(85)
    b.run(FLOOR - 5, 92, "XXX")
    b.enemy(97)
    b.stairs(100, 3)
    b.axe(106)
    return b.build()


# ===========================================================================
# WORLD 2
# ===========================================================================
def w2_1():
    """Overworld, busier: more pipes, wider brick work, a bonus room in a pipe."""
    b = Builder(136, THEME_OVERWORLD, "2-1")
    b.floor()
    for a in (28, 48, 70, 92, 112):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(12)
    b.enemy(13)
    b.run(FLOOR - 4, 8, "BB?BB")
    b.pipe(18, height=2)
    b.run(FLOOR - 6, 20, "BBB")          # reachable off the pipe top beside it
    b.coins(FLOOR - 7, 20, 3)
    b.enemy(40)
    b.run(FLOOR - 4, 34, "BMB")
    b.pipe(41, height=3, enterable=True, dest="bonus")
    b.enemy(60)
    b.enemy(61)
    b.platform(FLOOR - 5, 54, 6)
    b.run(FLOOR - 8, 58, "?B?")
    b.pipe(64, height=3)
    b.enemy(82)
    b.stairs(80, 3)
    b.run(FLOOR - 4, 81, "BB?BB")
    b.enemy(107)
    b.enemy(108)
    b.mushroom_platform(FLOOR - 5, 98, 5)
    b.run(FLOOR - 4, 104, "BBB")
    b.enemy(125)
    b.stairs(121, 3)
    b.flag(128)
    return b.build()


def w2_1_bonus():
    b = Builder(26, THEME_UNDERGROUND, "2-1 bonus")
    b.floor()
    b.roof()
    b.start(2)
    for row in (FLOOR - 3, FLOOR - 5, FLOOR - 7):
        b.coins(row, 4, 16)
    b.run(FLOOR - 5, 10, "BMB")
    b.exit_pipe(23, 45)
    return b.build()


def w2_2():
    """Underwater. Swim the whole way; cheep-cheeps cross at several depths."""
    b = Builder(120, THEME_WATER, "2-2")
    b.floor()
    b.roof(rows=1)
    b.water(top=1)
    b.start(3)
    for col, row in ((14, 5), (24, 8), (36, 3), (46, 7), (58, 5), (70, 9),
                     (82, 4), (94, 7), (104, 6)):
        b.swimmer(col, row)
    b.platform(FLOOR - 3, 18, 4, kind=STONE, coins_above=False)
    b.coins(FLOOR - 4, 18, 4)
    b.platform(FLOOR - 5, 40, 5, kind=STONE, coins_above=False)
    b.coins(FLOOR - 6, 40, 5)
    b.platform(FLOOR - 4, 64, 4, kind=STONE, coins_above=False)
    b.coins(FLOOR - 5, 64, 4)
    b.platform(FLOOR - 5, 88, 5, kind=STONE, coins_above=False)
    b.coins(FLOOR - 6, 88, 5)
    b.pipe(110, height=3)
    b.axe(116)
    return b.build()


def w2_3():
    """Bridges over water, with cheep-cheeps leaping up through them."""
    b = Builder(120, THEME_WATER, "2-3")
    b.bridge(FLOOR, 0, 119)              # the bridge is the only footing
    b.water(top=FLOOR + 1)
    b.start(3)
    for col in (14, 26, 40, 54, 68, 82, 96, 108):
        b.swimmer(col, ROWS - 1, kind="leaper")
    b.run(FLOOR - 5, 18, "BB?BB")
    b.coins(FLOOR - 6, 18, 5)
    b.run(FLOOR - 5, 44, "BMB")
    b.platform(FLOOR - 5, 62, 5)
    b.run(FLOOR - 5, 86, "BB?BB")
    b.coins(FLOOR - 6, 86, 5)
    b.flag(112)
    return b.build()


def w2_4():
    """Castle, longer: more lava, more podoboos, a maze of stone shelves."""
    b = Builder(124, THEME_CASTLE, "2-4")
    b.floor()
    b.start(3)
    for a in (16, 30, 46, 62, 78, 94, 108):
        b.lava(a, a + 1)
    b.run(FLOOR - 4, 8, "XXX")
    b.podoboo(17)
    b.run(FLOOR - 5, 20, "XXX")
    b.enemy(26)
    b.podoboo(31)
    b.platform(FLOOR - 4, 36, 4, kind=STONE)
    b.enemy(41)
    b.podoboo(47)
    b.run(FLOOR - 5, 52, "XXXX")
    b.enemy(57)
    b.podoboo(63)
    b.platform(FLOOR - 4, 68, 4, kind=STONE)
    b.enemy(73)
    b.podoboo(79)
    b.run(FLOOR - 5, 84, "XXXXX")
    b.enemy(89)
    b.podoboo(95)
    b.platform(FLOOR - 5, 99, 4, kind=STONE)
    b.enemy(104)
    b.podoboo(109)
    b.run(FLOOR - 4, 114, "XXX")
    b.stairs(117, 3)
    b.axe(122)
    return b.build()


# ===========================================================================
# WORLD 3 -- after dark. Koopas and paratroopas, moving lifts, hammer bros.
# ===========================================================================
def w3_1():
    """Night overworld. Koopas and goombas together, and a pair of hammer bros on
    brick platforms guarding the run to the flag."""
    b = Builder(126, THEME_NIGHT, "3-1")
    b.floor()
    for a in (30, 52, 76, 98):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(12, "koopa")
    b.run(FLOOR - 4, 8, "BB?BB")
    b.pipe(18, height=2)
    b.enemy(22, "goomba")
    b.run(FLOOR - 5, 21, "BBB")
    b.coins(FLOOR - 6, 21, 3)
    b.enemy(42, "koopa")
    b.run(FLOOR - 4, 38, "BMB")
    b.pipe(46, height=3)
    b.enemy(64, "koopa")
    b.enemy(65, "goomba")
    b.platform(FLOOR - 5, 62, 6)
    b.run(FLOOR - 4, 66, "?B?")
    b.enemy(88, "koopa")
    b.stairs(84, 3)
    b.platform(FLOOR - 3, 106, 5, kind=BRICK)
    b.flyer(107, FLOOR - 4, "hammer")
    b.platform(FLOOR - 5, 112, 4, kind=BRICK)
    b.flyer(113, FLOOR - 6, "hammer")
    b.flag(120)
    return b.build()


def w3_2():
    """Night overworld, longer and meaner: staircases, a crowd of koopas, and a
    pipe down to a coin room."""
    b = Builder(132, THEME_NIGHT, "3-2")
    b.floor()
    for a in (26, 46, 68, 90, 112):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(10, "koopa")
    b.enemy(11, "koopa")
    b.run(FLOOR - 4, 15, "BB?BB")
    b.pipe(20, height=2, enterable=True, dest="bonus")
    b.enemy(38, "koopa")
    b.stairs(34, 3)
    b.run(FLOOR - 5, 36, "BBBB")
    b.coins(FLOOR - 6, 36, 4)
    b.enemy(58, "goomba")
    b.enemy(59, "koopa")
    b.run(FLOOR - 4, 54, "BMB")
    b.stairs(59, 4, descend=True)
    b.enemy(80, "koopa")
    b.platform(FLOOR - 5, 78, 6)
    b.run(FLOOR - 8, 78, "?B?")
    b.enemy(102, "koopa")
    b.enemy(103, "goomba")
    b.stairs(98, 3)
    b.run(FLOOR - 4, 101, "BB?BB")
    b.stairs(120, 4)
    b.flag(126)
    return b.build()


def w3_2_bonus():
    b = Builder(26, THEME_UNDERGROUND, "3-2 bonus")
    b.floor()
    b.roof()
    b.start(2)
    for row in (FLOOR - 3, FLOOR - 5, FLOOR - 7):
        b.coins(row, 4, 16)
    b.run(FLOOR - 5, 20, "BMB")
    b.exit_pipe(23, 24)
    return b.build()


def w3_3():
    """Night athletic: platforms that move. Vertical lifts ride up and down,
    horizontal ones shuttle across. The pits under them are still jumpable, so a
    missed lift costs coins rather than the level."""
    b = Builder(120, THEME_NIGHT, "3-3")
    b.floor()
    for a in (22, 40, 58, 76, 94):
        b.gap(a, a + 1)
    b.start(3)
    b.flyer(12, FLOOR - 4, "para")
    b.lift(16, FLOOR - 6, "lift_v", span=4, width=3)
    b.coins(FLOOR - 9, 16, 3)
    b.flyer(32, FLOOR - 5, "para")
    b.lift(34, FLOOR - 5, "lift_h", span=5, width=3)
    b.coins(FLOOR - 8, 34, 3)
    b.flyer(50, FLOOR - 4, "para")
    b.lift(52, FLOOR - 7, "lift_v", span=5, width=3)
    b.coins(FLOOR - 10, 52, 3)
    b.enemy(68, "koopa")
    b.lift(70, FLOOR - 5, "lift_h", span=6, width=3)
    b.coins(FLOOR - 8, 70, 3)
    b.flyer(86, FLOOR - 5, "para")
    b.lift(88, FLOOR - 6, "lift_v", span=4, width=3)
    b.coins(FLOOR - 9, 88, 3)
    b.enemy(104, "koopa")
    b.stairs(108, 3)
    b.flag(114)
    return b.build()


def w3_4():
    """Castle with firebars: rotating spokes of fire over lava."""
    b = Builder(116, THEME_CASTLE, "3-4")
    b.floor()
    b.start(3)
    for a in (18, 34, 52, 70, 88):
        b.lava(a, a + 1)
    b.run(FLOOR - 4, 10, "XXX")
    b.firebar(14, FLOOR - 5, length=4)
    b.podoboo(19)
    b.enemy(28)
    b.firebar(30, FLOOR - 6, length=5)
    b.podoboo(35)
    b.platform(FLOOR - 5, 42, 4, kind=STONE)
    b.firebar(46, FLOOR - 5, length=4)
    b.podoboo(53)
    b.run(FLOOR - 4, 60, "XXXX")
    b.firebar(64, FLOOR - 6, length=5)
    b.podoboo(71)
    b.platform(FLOOR - 5, 78, 5, kind=STONE)
    b.firebar(82, FLOOR - 5, length=4)
    b.podoboo(89)
    b.enemy(100)
    b.firebar(98, FLOOR - 6, length=4)
    b.stairs(104, 3)
    b.axe(111)
    return b.build()


# ===========================================================================
# WORLD 4 -- buzzy beetles that shrug off fire, a beanstalk, and a maze castle.
# ===========================================================================
def w4_1():
    """Overworld. Buzzy beetles walk where goombas did, and fire will not touch
    them -- they have to be jumped or stomped."""
    b = Builder(128, THEME_OVERWORLD, "4-1")
    b.floor()
    for a in (28, 50, 72, 96):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(12, "buzzy")
    b.run(FLOOR - 4, 8, "BB?BB")
    b.pipe(18, height=3)
    b.flyer(24, FLOOR - 6, "para")
    b.run(FLOOR - 5, 12, "BBB")
    b.coins(FLOOR - 6, 12, 3)
    b.enemy(40, "buzzy")
    b.enemy(41, "goomba")
    b.run(FLOOR - 4, 36, "BMB")
    b.platform(FLOOR - 5, 38, 5)
    b.flyer(62, FLOOR - 6, "para")
    b.run(FLOOR - 4, 60, "BB?BB")
    b.enemy(66, "buzzy")
    b.pipe(84, height=2)
    b.enemy(88, "buzzy")
    b.stairs(88, 3)
    b.run(FLOOR - 5, 106, "?B?")
    b.enemy(110, "koopa")
    b.stairs(114, 4)
    b.flag(122)
    return b.build()


def w4_2():
    """Underground, with a beanstalk hidden in a block: touch the vine and it
    carries you up to a sky full of coins."""
    b = Builder(126, THEME_UNDERGROUND, "4-2")
    b.floor()
    b.roof()
    b.start(3)
    b.enemy(14, "buzzy")
    b.run(FLOOR - 4, 10, "BBBBB")
    b.coins(FLOOR - 5, 10, 5)
    b.vine(20, "sky")
    b.enemy(30, "koopa")
    b.run(FLOOR - 4, 26, "BB?BB")
    b.run(FLOOR - 8, 28, "BBB")
    b.coins(FLOOR - 9, 28, 3)
    b.gap(40, 41)
    b.enemy(52, "buzzy")
    b.run(FLOOR - 4, 48, "BMB")
    b.platform(FLOOR - 5, 58, 6)
    b.enemy(70, "koopa")
    b.gap(78, 79)
    b.run(FLOOR - 4, 86, "BBBB")
    b.coins(FLOOR - 5, 86, 4)
    b.enemy(94, "buzzy")
    b.run(FLOOR - 5, 98, "BBBBBB")
    b.coins(FLOOR - 6, 98, 6)
    b.enemy(108, "koopa")
    b.pipe(114, height=3)
    b.axe(120)
    return b.build()


def w4_2_sky():
    """Coin heaven at the top of the beanstalk."""
    b = Builder(30, THEME_NIGHT, "4-2 sky")
    b.floor()
    b.start(2)
    for row in (FLOOR - 3, FLOOR - 5, FLOOR - 7):
        b.coins(row, 4, 20)
    b.run(FLOOR - 5, 12, "BMB")
    b.exit_pipe(27, 26)
    return b.build()


def w4_3():
    """Athletic: tall mushroom platforms, paratroopas between them, and a hammer
    bro waiting on the last one."""
    b = Builder(124, THEME_OVERWORLD, "4-3")
    b.floor()
    for a in (24, 44, 64, 84, 104):
        b.gap(a, a + 1)
    b.start(3)
    b.mushroom_platform(FLOOR - 5, 10, 5)
    b.flyer(18, FLOOR - 6, "para")
    b.mushroom_platform(FLOOR - 4, 30, 5)
    b.enemy(37, "koopa")
    b.flyer(38, FLOOR - 7, "para")
    b.mushroom_platform(FLOOR - 5, 50, 5)
    b.flyer(58, FLOOR - 6, "para")
    b.mushroom_platform(FLOOR - 4, 70, 5)
    b.run(FLOOR - 4, 76, "BMB")
    b.mushroom_platform(FLOOR - 5, 90, 5)
    b.flyer(98, FLOOR - 6, "para")
    b.platform(FLOOR - 5, 110, 5, kind=BRICK)
    b.flyer(111, FLOOR - 6, "hammer")
    b.flag(118)
    return b.build()


def w4_4():
    """The maze castle. Take the low road at the fork and it puts you back at the
    start of the section; the way on is over the top."""
    b = Builder(120, THEME_CASTLE, "4-4")
    b.floor()
    b.start(3)
    for a in (16, 32, 62, 84):
        b.lava(a, a + 1)
    b.run(FLOOR - 4, 8, "XXX")
    b.firebar(12, FLOOR - 5, length=4)
    b.podoboo(17)
    b.enemy(26)
    b.firebar(28, FLOOR - 6, length=5)
    b.podoboo(33)
    # The fork: the upper shelf carries you past the trigger at 52.
    b.platform(FLOOR - 3, 40, 15, kind=STONE)
    b.coins(FLOOR - 4, 40, 15)
    b.enemy(58)
    b.loop(52, 34, FLOOR - 3)
    b.firebar(58, FLOOR - 5, length=4)
    b.podoboo(63)
    b.run(FLOOR - 4, 70, "XXXX")
    b.enemy(76)
    b.firebar(78, FLOOR - 6, length=5)
    b.podoboo(85)
    b.platform(FLOOR - 5, 92, 5, kind=STONE)
    b.enemy(98)
    b.firebar(102, FLOOR - 5, length=4)
    b.stairs(108, 3)
    b.axe(115)
    return b.build()


# ===========================================================================
# WORLD 5 -- Bullet Bills arrive. Cannons fire along the ground, so a jump is no
# longer only about where the floor is.
# ===========================================================================
def w5_1():
    """Overworld. Cannons on plateaus, and hammer bros guarding the flag."""
    b = Builder(128, THEME_OVERWORLD, "5-1")
    b.floor()
    for a in (30, 52, 76, 100):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(12, "koopa")
    b.run(FLOOR - 4, 8, "BB?BB")
    b.cannon(20, height=2)
    b.enemy(24, "goomba")
    b.run(FLOOR - 5, 22, "BBB")
    b.coins(FLOOR - 6, 22, 3)
    b.cannon(38, height=3)
    b.enemy(42, "koopa")
    b.run(FLOOR - 4, 44, "BMB")
    b.cannon(60, height=2)
    b.enemy(64, "buzzy")
    b.platform(FLOOR - 5, 62, 5)
    b.cannon(84, height=3)
    b.enemy(88, "koopa")
    b.run(FLOOR - 4, 90, "?B?")
    b.platform(FLOOR - 3, 108, 5, kind=BRICK)
    b.flyer(109, FLOOR - 4, "hammer")
    b.platform(FLOOR - 5, 114, 4, kind=BRICK)
    b.flyer(115, FLOOR - 6, "hammer")
    b.flag(122)
    return b.build()


def w5_2():
    """Overworld with a pipe down to a coin room and lifts over the gaps."""
    b = Builder(132, THEME_OVERWORLD, "5-2")
    b.floor()
    for a in (26, 48, 70, 92, 114):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(11, "goomba")
    b.enemy(12, "koopa")
    b.run(FLOOR - 4, 8, "BB?BB")
    b.pipe(18, height=2, enterable=True, dest="bonus")
    b.cannon(34, height=2)
    b.run(FLOOR - 5, 36, "BMB")
    b.lift(40, FLOOR - 5, "lift_h", span=5, width=3)
    b.coins(FLOOR - 8, 40, 3)
    b.enemy(58, "koopa")
    b.platform(FLOOR - 5, 56, 6)
    b.cannon(64, height=3)
    b.lift(80, FLOOR - 6, "lift_v", span=4, width=3)
    b.coins(FLOOR - 9, 80, 3)
    b.enemy(84, "buzzy")
    b.run(FLOOR - 4, 100, "BB?BB")
    b.enemy(106, "koopa")
    b.cannon(108, height=2)
    b.stairs(120, 4)
    b.flag(126)
    return b.build()


def w5_2_bonus():
    b = Builder(26, THEME_UNDERGROUND, "5-2 bonus")
    b.floor()
    b.roof()
    b.start(2)
    for row in (FLOOR - 3, FLOOR - 5, FLOOR - 7):
        b.coins(row, 4, 16)
    b.run(FLOOR - 5, 20, "BMB")
    b.exit_pipe(23, 22)
    return b.build()


def w5_3():
    """Night athletic: mushroom platforms and paratroopas over a gappy floor."""
    b = Builder(120, THEME_NIGHT, "5-3")
    b.floor()
    for a in (20, 38, 56, 74, 92):
        b.gap(a, a + 1)
    b.start(3)
    b.mushroom_platform(FLOOR - 5, 8, 4)
    b.flyer(14, FLOOR - 6, "para")
    b.mushroom_platform(FLOOR - 4, 26, 5)
    b.enemy(32, "koopa")
    b.flyer(33, FLOOR - 7, "para")
    b.mushroom_platform(FLOOR - 5, 44, 5)
    b.flyer(51, FLOOR - 6, "para")
    b.mushroom_platform(FLOOR - 4, 62, 5)
    b.run(FLOOR - 4, 65, "BMB")
    b.mushroom_platform(FLOOR - 5, 80, 5)
    b.flyer(87, FLOOR - 6, "para")
    b.mushroom_platform(FLOOR - 4, 98, 4)
    b.enemy(104, "koopa")
    b.stairs(108, 3)
    b.flag(114)
    return b.build()


def w5_4():
    """Castle: firebars over a long run of lava, podoboos between them."""
    b = Builder(118, THEME_CASTLE, "5-4")
    b.floor()
    b.start(3)
    for a in (18, 32, 48, 64, 80, 96):
        b.lava(a, a + 1)
    b.run(FLOOR - 4, 10, "XXX")
    b.firebar(14, FLOOR - 5, length=4)
    b.podoboo(19)
    b.firebar(28, FLOOR - 6, length=5)
    b.podoboo(33)
    b.platform(FLOOR - 4, 38, 4, kind=STONE)
    b.firebar(44, FLOOR - 5, length=4)
    b.podoboo(49)
    b.run(FLOOR - 5, 54, "XXXX")
    b.enemy(59)
    b.firebar(60, FLOOR - 6, length=5)
    b.podoboo(65)
    b.platform(FLOOR - 4, 70, 4, kind=STONE)
    b.firebar(76, FLOOR - 5, length=4)
    b.podoboo(81)
    b.run(FLOOR - 5, 86, "XXX")
    b.enemy(91)
    b.firebar(92, FLOOR - 6, length=4)
    b.podoboo(97)
    b.enemy(106)
    b.stairs(108, 3)
    b.axe(113)
    return b.build()


# ===========================================================================
# WORLD 6 -- plateaus and height. The overworlds stack their ground in tiers and
# 6-3 is the high athletic level, all platform-to-platform.
# ===========================================================================
def w6_1():
    """Overworld built in tiers: stone plateaus you climb rather than run past."""
    b = Builder(126, THEME_OVERWORLD, "6-1")
    b.floor()
    for a in (28, 50, 74, 98):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(10, "buzzy")
    b.stairs(14, 3)
    b.run(FLOOR - 5, 18, "BB?BB")
    b.enemy(38, "koopa")
    b.stairs(34, 3, descend=True)
    b.platform(FLOOR - 3, 40, 6, kind=STONE)
    b.coins(FLOOR - 4, 40, 6)
    b.cannon(60, height=2)
    b.enemy(64, "buzzy")
    b.stairs(66, 3)
    b.run(FLOOR - 5, 62, "BMB")
    b.enemy(86, "koopa")
    b.platform(FLOOR - 3, 84, 6, kind=STONE)
    b.coins(FLOOR - 4, 84, 6)
    b.cannon(92, height=3)
    b.enemy(110, "buzzy")
    b.run(FLOOR - 4, 106, "?B?")
    b.stairs(114, 4)
    b.flag(120)
    return b.build()


def w6_2():
    """Overworld, long and pipe-heavy, with a beanstalk hidden mid-level."""
    b = Builder(134, THEME_OVERWORLD, "6-2")
    b.floor()
    for a in (30, 54, 78, 104):
        b.gap(a, a + 1)
    b.start(3)
    b.enemy(11, "goomba")
    b.enemy(12, "koopa")
    b.run(FLOOR - 4, 8, "BB?BB")
    b.pipe(18, height=2)
    b.vine(24, "sky")
    b.enemy(42, "koopa")
    b.run(FLOOR - 4, 38, "BMB")
    b.pipe(46, height=3)
    b.enemy(64, "buzzy")
    b.platform(FLOOR - 5, 62, 6)
    b.run(FLOOR - 8, 66, "?B?")
    b.pipe(70, height=2)
    b.enemy(88, "koopa")
    b.enemy(89, "goomba")
    b.run(FLOOR - 4, 84, "BB?BB")
    b.pipe(94, height=3)
    b.cannon(98, height=2)
    b.enemy(114, "buzzy")
    b.run(FLOOR - 5, 110, "BBB")
    b.coins(FLOOR - 6, 110, 3)
    b.pipe(120, height=2)
    b.stairs(122, 4)
    b.flag(128)
    return b.build()


def w6_2_sky():
    """Coin heaven above 6-2."""
    b = Builder(30, THEME_NIGHT, "6-2 sky")
    b.floor()
    b.start(2)
    for row in (FLOOR - 3, FLOOR - 5, FLOOR - 7):
        b.coins(row, 4, 20)
    b.run(FLOOR - 5, 12, "BMB")
    b.exit_pipe(27, 30)
    return b.build()


def w6_3():
    """The high athletic level: everything happens on platforms, with paratroopas
    patrolling the gaps between them."""
    b = Builder(126, THEME_NIGHT, "6-3")
    b.floor()
    for a in (18, 34, 50, 66, 82, 98):
        b.gap(a, a + 1)
    b.start(3)
    b.mushroom_platform(FLOOR - 3, 8, 5)
    b.flyer(13, FLOOR - 5, "para")
    b.mushroom_platform(FLOOR - 5, 24, 5)
    b.flyer(29, FLOOR - 7, "para")
    b.mushroom_platform(FLOOR - 3, 40, 5)
    b.enemy(46, "koopa")
    b.mushroom_platform(FLOOR - 5, 56, 5)
    b.flyer(61, FLOOR - 7, "para")
    b.mushroom_platform(FLOOR - 3, 72, 5)
    b.run(FLOOR - 5, 74, "BMB")
    b.mushroom_platform(FLOOR - 5, 88, 5)
    b.flyer(93, FLOOR - 7, "para")
    b.mushroom_platform(FLOOR - 3, 104, 5)
    b.enemy(110, "koopa")
    b.stairs(112, 3)
    b.flag(118)
    return b.build()


def w6_4():
    """Castle: firebars over lava, and lifts to carry you between the shelves."""
    b = Builder(120, THEME_CASTLE, "6-4")
    b.floor()
    b.start(3)
    for a in (18, 34, 52, 70, 88):
        b.lava(a, a + 1)
    b.run(FLOOR - 4, 10, "XXX")
    b.firebar(14, FLOOR - 5, length=4)
    b.podoboo(19)
    b.lift(24, FLOOR - 6, "lift_v", span=3, width=3)
    b.enemy(28)
    b.firebar(30, FLOOR - 6, length=5)
    b.podoboo(35)
    b.platform(FLOOR - 4, 40, 4, kind=STONE)
    b.firebar(46, FLOOR - 5, length=4)
    b.podoboo(53)
    b.lift(58, FLOOR - 5, "lift_h", span=4, width=3)
    b.enemy(62)
    b.firebar(64, FLOOR - 6, length=5)
    b.podoboo(71)
    b.platform(FLOOR - 4, 76, 4, kind=STONE)
    b.firebar(82, FLOOR - 5, length=4)
    b.podoboo(89)
    b.enemy(98)
    b.firebar(100, FLOOR - 6, length=4)
    b.stairs(108, 3)
    b.axe(115)
    return b.build()


WORLDS = [
    [w1_1, w1_2, w1_3, w1_4],
    [w2_1, w2_2, w2_3, w2_4],
    [w3_1, w3_2, w3_3, w3_4],
    [w4_1, w4_2, w4_3, w4_4],
    [w5_1, w5_2, w5_3, w5_4],
    [w6_1, w6_2, w6_3, w6_4],
]
BONUS = {"1-1": w1_1_bonus, "2-1": w2_1_bonus, "3-2": w3_2_bonus,
         "4-2": w4_2_sky, "5-2": w5_2_bonus, "6-2": w6_2_sky}


def load(world, level):
    """1-based world and level -> built level dict."""
    return WORLDS[world - 1][level - 1]()


def load_bonus(name):
    fn = BONUS.get(name)
    return fn() if fn else None


def all_levels():
    for wi, w in enumerate(WORLDS, 1):
        for li in range(1, len(w) + 1):
            yield wi, li, load(wi, li)


if __name__ == "__main__":
    for wi, li, lv in all_levels():
        w = len(lv["rows"][0])
        print(f"{lv['name']:>4}  {lv['theme']:<12} {w:3d} tiles  "
              f"{len(lv['enemies']):2d} enemies  "
              f"{sum(r.count(COIN) for r in lv['rows']):3d} coins  "
              f"end={lv['end'][0]}"
              + (f"  bonus->{list(lv['pipe_links'])}" if lv["pipe_links"] else ""))
    for name in BONUS:
        lv = load_bonus(name)
        print(f"{lv['name']:>10}  {len(lv['rows'][0])} tiles, "
              f"{sum(r.count(COIN) for r in lv['rows'])} coins")
