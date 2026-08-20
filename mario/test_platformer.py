"""Headless playability tests for every level in every world.

No camera, no display: SDL runs on the dummy driver and the game is driven by a
bot. The point is to prove every level is completable at the *weakest* jump the
gesture layer can produce, because a gap that needs a full-strength flick is a gap
that will feel broken at 50-80ms of input latency.
"""
import os, sys, math
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pygame
import levels as LV
import platformer as P

FAIL = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)

DT = 1.0 / P.FPS
ALL = [(w, l) for w in range(1, len(LV.WORLDS) + 1) for l in (1, 2, 3, 4)]


def new_game(world=1, level=1):
    return P.Game(world=world, level=level, use_camera=False)


def footing_below(level, x, from_y, depth_tiles=5):
    """Is there anything to land on under x, within a survivable fall?

    Probing only at the hero's own foot height is wrong whenever it is standing on
    something raised: the row it is on reads empty while the real floor below is
    solid, so the bot sees a phantom hole and jumps far too early.
    """
    for d in range(0, depth_tiles * P.TILE, 8):
        if level.solid_at(x, from_y + 6 + d):
            return True
        if level.deadly_at(x, from_y + 6 + d):
            return False          # lava is not footing
    return False


def autoplay(g, strength, run_flag, lookahead=14, max_seconds=140, t0=1000.0):
    """Run right; jump holes, walls and enemies; burn what is in the way."""
    now = t0
    g.level_start = now
    g.hero.last_ground_t = now
    airtime = 2 * (P.JUMP_MIN + (P.JUMP_MAX - P.JUMP_MIN) * strength) / P.GRAVITY
    deaths = 0
    # Once a jump is committed at walking speed it has to STAY at walking speed.
    # Choosing walk for the take-off and then reverting to run on the airborne
    # frames overshoots the landing by three tiles, straight into the pit the slow
    # jump was chosen to avoid.
    commit_walk = False
    for _ in range(int(max_seconds * P.FPS)):
        now += DT
        h = g.hero
        if h.grounded:
            commit_walk = False
        want_run = run_flag and not commit_walk
        water = g.in_water()
        # Bowser cannot be stomped and takes five fireballs. Charging him is running
        # into a wall that hits back, so hold at range and shoot -- and do it whether
        # or not we happen to be grounded, since arriving mid-jump skipped the check
        # entirely and walked straight into him.
        if g.state == "play" and any(
                e.alive and e.dying <= 0 and e.armoured and 0 < e.x - h.x < 6 * P.TILE
                for e in g.enemies):
            lives_before = g.lives
            g.throw(now)
            if any(hm.alive and hm.flat and -0.5 * P.TILE < hm.x - h.x < 5 * P.TILE
                   for hm in g.hammers):
                h.request_jump(strength, now)
            g.update(0.0, False, False, DT, now)
            if g.lives < lives_before:
                deaths += 1
            if g.state == "dead":
                now = g.state_until
                g.update(0.0, False, False, DT, now)
            elif g.state in ("won", "complete"):
                return {"outcome": "won", "t": now - t0, "deaths": deaths,
                        "score": g.score, "x": g.hero.x}
            elif g.state == "gameover":
                return {"outcome": "gameover", "t": now - t0, "deaths": deaths,
                        "x": g.hero.x}
            continue
        if g.state == "play" and (h.grounded or water):
            probe = h.x + h.W + lookahead
            hole = not footing_below(g.level, probe, h.y)
            wall = g.level.solid_at(h.x + h.W + 2, h.y - h.height * 0.5)
            ahead = [e for e in g.enemies
                     if e.alive and e.dying <= 0 and not e.harmless
                     and 0 < e.x - h.x < 220 and abs(e.y - h.y) < 70]
            if ahead:
                h.facing = 1
                g.throw(now)
            # A stomp-proof enemy (spiny) must be jumped OVER, not onto, so commit
            # to the jump from further out. A fire-proof one (buzzy) can only be
            # stomped, so close the distance first. Treating both the same is how
            # the bot kept landing on spines.
            # Only actual spinies: "anything not stompable" also caught airborne
            # podoboos and firebars, which made the bot commit early jumps straight
            # into lava and broke four levels that had been passing.
            spiky = [e for e in ahead if e.kind == "spiny"]
            # Bowser's fire flies flat at body height, so waiting for it just means
            # being hit standing still. It is jumped, like everything else he does.
            if any(hm.alive and hm.flat and 0 < hm.x - h.x < 4 * P.TILE
                   for hm in g.hammers):
                h.request_jump(strength, now)
            # Bowser cannot be stomped and takes five fireballs. Charging him is
            # just running into a wall that hits back: hold at range and shoot.
            # Not from `ahead`: he hops, so half the time he is outside the vertical
            # window that filters ordinary walkers, and the standoff never fired.
            boss = [e for e in g.enemies
                    if e.alive and e.dying <= 0 and e.armoured
                    and 0 < e.x - h.x < 6 * P.TILE]
            if boss:
                lives_before = g.lives
                g.throw(now)
                # Still have to dodge while shooting: standing still at range is how
                # the bot got torched by the fire it was supposed to be jumping.
                if any(hm.alive and hm.flat and -0.5 * P.TILE < hm.x - h.x < 5 * P.TILE
                       for hm in g.hammers):
                    h.request_jump(strength, now)
                g.update(0.0, False, False, DT, now)
                if g.lives < lives_before:
                    deaths += 1
                if g.state == "dead":
                    now = g.state_until
                    g.update(0.0, False, False, DT, now)
                elif g.state == "gameover":
                    return {"outcome": "gameover", "t": now - t0,
                            "deaths": deaths, "x": g.hero.x}
                continue
            close = [e for e in ahead if e.x - h.x < (150 if spiky else 80)]
            # About to step off a RAISED ledge: jump instead of walking off. A fall
            # from a shelf carries you four tiles at running speed, which is how the
            # bot kept sailing off a brick row straight into the pit past its end.
            on_ledge = (h.y < LV.FLOOR * P.TILE - 4
                        and not g.level.solid_at(h.x + h.W + 6, h.y + 6))
            if on_ledge:
                drop = max(P.TILE, LV.FLOOR * P.TILE - h.y)
                fall_frames = math.sqrt(2 * drop / P.GRAVITY)
                speed = P.RUN_SPEED if run_flag else P.WALK_SPEED
                if not footing_below(g.level, h.x + fall_frames * speed, h.y,
                                     depth_tiles=8):
                    h.request_jump(strength, now)
            # Timed hazards are dodged by waiting, which is what a player does and
            # what the bot has to model, or a castle level looks uncompletable when
            # it is merely a matter of rhythm.
            # Wait if a timed hazard ahead is airborne OR about to launch: a
            # podoboo that fires while you are mid-jump is not dodgeable, so a
            # player waits for the window and so must the bot.
            # Podoboos are dodged by waiting for the window. Leapers are NOT:
            # they come up through the bridge you are standing on, so stopping is
            # the one thing that guarantees getting hit. Keep moving past those.
            hazard = any(e.kind == "podoboo"
                         and 0 < e.x - h.x < 4 * P.TILE
                         and (e.vy != 0.0 or e.timer - now < 1.0)
                         for e in g.enemies)
            # A thrown hammer is dodged the same way: let it land, then move.
            hazard = hazard or any(hm.alive and not hm.flat
                                   and -1.5 * P.TILE < hm.x - h.x < 5 * P.TILE
                                   for hm in g.hammers)
            # A firebar is a timed hazard too: it sweeps a circle, so wait until the
            # arc is not across the way ahead rather than walking into a spoke.
            front = h.x + h.W + 8
            for e in g.enemies:
                if e.kind != "firebar":
                    continue
                for px, py in e.firebar_points():
                    if abs(px - front) < 1.6 * P.TILE and \
                            h.y - h.height - 12 < py < h.y + 12:
                        hazard = True
                        break
                if hazard:
                    break
            if hazard and not water:
                before_wait = g.lives
                g.update(0.0, False, False, DT, now)
                if g.lives < before_wait:
                    deaths += 1
                if g.state == "dead":
                    now = g.state_until
                    g.update(0.0, False, False, DT, now)
                elif g.state == "gameover":
                    return {"outcome": "gameover", "t": now - t0,
                            "deaths": deaths, "x": g.hero.x}
                continue
            # A maze fork has a high road: climb onto the shelf rather than walking
            # into the barrier that blocks the low one.
            for col, above_row in getattr(g.level, "gates", []):
                gap_tiles = (col * P.TILE - h.x) / P.TILE
                if 0 < gap_tiles < 16 and h.y > above_row * P.TILE:
                    h.request_jump(strength, now)
                    break
            if water:
                # Swimming: stroke often enough to stay off the floor hazards.
                if int(now * 6) % 2 == 0:
                    h.request_jump(strength, now, True)
            elif hole or wall or close:
                safe = [flag for speed, flag in ((P.RUN_SPEED, True),
                                                 (P.WALK_SPEED, False))
                        if footing_below(g.level, h.x + airtime * speed, h.y)]
                if safe:
                    want_run = (run_flag and True in safe) or (safe == [True])
                    commit_walk = not want_run
                    h.request_jump(strength, now)
                elif close:
                    want_run = False
        before = g.lives
        g.update(1.0, want_run, False, DT, now)
        if g.lives < before:
            deaths += 1
        if g.state in ("won", "complete"):
            return {"outcome": "won", "t": now - t0, "deaths": deaths,
                    "score": g.score, "x": g.hero.x}
        if g.state == "gameover":
            return {"outcome": "gameover", "t": now - t0, "deaths": deaths,
                    "x": g.hero.x}
        if g.state == "dead":
            now = g.state_until
            g.update(0.0, False, False, DT, now)
    return {"outcome": "timeout", "t": now - t0, "deaths": deaths, "x": g.hero.x}


print(f"\n[1] all {len(ALL)} levels build and validate")
for w, l in ALL:
    try:
        data = LV.load(w, l)
        ok = bool(data["rows"]) and data["end"] is not None
        check(f"{w}-{l} builds ({data['theme']}, {len(data['rows'][0])} tiles)", ok)
    except AssertionError as e:
        check(f"{w}-{l} builds", False, str(e)[:160])
for name in LV.BONUS:
    check(f"bonus room for {name} builds", LV.load_bonus(name) is not None)

print("\n[2] the worlds are actually different from each other")
sigs, themes, widths = {}, {}, {}
for w, l in ALL:
    d = LV.load(w, l)
    sigs[(w, l)] = "".join(d["rows"])
    themes[(w, l)] = d["theme"]
    widths[(w, l)] = len(d["rows"][0])
check("no two levels have identical layouts", len(set(sigs.values())) == len(sigs),
      f"{len(set(sigs.values()))} distinct of {len(sigs)}")
check("at least five themes are used", len(set(themes.values())) >= 5,
      str(sorted(set(themes.values()))))
check("levels differ in length", len(set(widths.values())) >= 3,
      str(sorted(set(widths.values()))))
tiles_used = set("".join(sigs.values()))
for want, label in ((LV.PIPE, "pipes"), (LV.QMUSH, "mushroom blocks"),
                    (LV.LAVA, "lava"), (LV.WATER, "water"),
                    (LV.BRIDGE, "bridges"), (LV.PIPE_IN, "enterable pipes"),
                    (LV.POLE, "flagpoles"), (LV.AXE, "axes"),
                    (LV.STONE, "stone"), (LV.COIN, "coins"),
                    (LV.VINE, "beanstalks")):
    check(f"the worlds contain {label}", want in tiles_used)

print("\n[3] jump arithmetic vs the widest gap in any level")
widest = 0
for w, l in ALL:
    d = LV.load(w, l)
    row = d["rows"][LV.FLOOR]
    run = None
    for c, ch in enumerate(row + "#"):
        if ch not in LV.SOLID and run is None:
            run = c
        elif ch in LV.SOLID and run is not None:
            widest = max(widest, c - run)
            run = None
weakest = P.JUMP_MIN + (P.JUMP_MAX - P.JUMP_MIN) * 0.45
airtime = 2 * weakest / P.GRAVITY
check("weakest jump clears the widest gap at a WALK",
      airtime * P.WALK_SPEED > widest * P.TILE * 1.15,
      f"reach {airtime*P.WALK_SPEED:.0f}px vs gap {widest*P.TILE}px "
      f"({airtime*P.WALK_SPEED/max(widest*P.TILE,1):.2f}x)")

print("\n[3b] the new worlds bring new things with them")
kinds = set()
for w, l in ALL:
    for _, _, k in LV.load(w, l)["enemies"]:
        kinds.add(k.split(":")[0])
for kind, label in (("para", "paratroopas"), ("buzzy", "buzzy beetles"),
                    ("hammer", "hammer bros"), ("firebar", "firebars"),
                    ("lift_v", "vertical lifts"), ("lift_h", "horizontal lifts"),
                    ("podoboo", "podoboos"), ("koopa", "koopas"),
                    ("cheep", "cheep-cheeps"), ("leaper", "leaping cheeps")):
    check(f"the game contains {label}", kind in kinds)
gates = sum(len(LV.load(w, l).get("gates", [])) for w, l in ALL)
check("the castles have maze forks", gates >= 3, f"{gates} fork(s)")
night = [f"{w}-{l}" for w, l in ALL if LV.load(w, l)["theme"] == LV.THEME_NIGHT]
check("World 3 is after dark", len(night) >= 3, str(night))

print("\n[4] a bot completes every level on the WEAKEST flick")
# Bullet Bills are dodged by jumping, so the bot has to treat one like a wall it
# cannot fight: fire does not stop them.
for w, l in ALL:
    for run_flag, label in ((False, "walking"), (True, "running")):
        g = new_game(w, l)
        res = autoplay(g, strength=0.45, run_flag=run_flag)
        check(f"{w}-{l} completed while {label}", res["outcome"] == "won",
              f"{res['outcome']} at x={res['x']:.0f}/{g.level.pw} "
              f"after {res['t']:.0f}s, deaths={res['deaths']}")

print("\n[4b] maze forks block the low road; nothing rewinds the player")
# The reported 8-4 bug: an invisible trigger teleported you backwards, and the high
# road was a shelf you walked UNDERNEATH -- so being sent back just meant walking
# under it again, for ever. A fork must be a wall you can see, never a rewind.
forked = [(w, l) for w, l in ALL if LV.load(w, l).get("gates")]
check("some levels have forks at all", forked, str(forked))
for w, l in forked:
    g = new_game(w, l)
    for col, above_row in g.level.gates:
        blocked = g.level.solid_at(col * P.TILE + P.TILE / 2,
                                   (LV.FLOOR - 1) * P.TILE + 4)
        open_above = not g.level.solid_at(col * P.TILE + P.TILE / 2,
                                          (above_row - 1) * P.TILE)
        check(f"{w}-{l}: the low road at col {col} is walled off", blocked)
        check(f"{w}-{l}: the high road over col {col} is clear", open_above)
    # And walking right must never yank the hero backwards.
    g.hero.x = float(g.level.gates[0][0] - 14) * P.TILE
    g.hero.y = float(LV.FLOOR * P.TILE)
    g.hero.grounded = True
    now, worst_back = 5000.0, 0.0
    for _ in range(14 * P.FPS):
        now += DT
        before = g.hero.x
        g.update(1.0, False, False, DT, now)
        worst_back = max(worst_back, before - g.hero.x)
        if g.state != "play":
            break
    check(f"{w}-{l}: walking into the fork never rewinds you",
          worst_back < P.TILE, f"largest backwards jump {worst_back:.0f}px")

print("\n[5] mushrooms, growing and taking a hit")
g = new_game(1, 1)
now = 2000.0
h = g.hero
h.big = False
g.items.append(P.Mushroom(h.x + 10, h.y))
g.items[0].emerging = 0.0
for _ in range(6):
    now += DT
    g.update(0.0, False, False, DT, now)
check("touching a mushroom grows you", h.big)
now += P.GROW_PAUSE + 0.1
check("big Mario is taller", h.height > P.Hero.H_SMALL,
      f"{h.height}px vs {P.Hero.H_SMALL}px")
e = g.enemies[0]
h.x, h.y = e.x - 6, e.y
h.vy = 0.0
h.grounded = True
lives = g.lives
g.update(1.0, False, False, DT, now)
check("big Mario shrinks instead of dying", not h.big and g.lives == lives,
      f"big={h.big} lives={g.lives}")
now += 0.05
g.update(1.0, False, False, DT, now)
check("and is briefly invulnerable", g.lives == lives, f"lives={g.lives}")

g = new_game(1, 1)
now = 2100.0
h = g.hero
e = g.enemies[0]
h.x, h.y = e.x - 6, e.y
h.grounded = True
lives = g.lives
g.update(1.0, False, False, DT, now)
check("small Mario dies on contact", g.lives == lives - 1)

print("\n[6] ? blocks, bricks and bumping")
g = new_game(1, 1)
now = 2200.0
found = None
for r, row in enumerate(g.level.rows):
    for c, ch in enumerate(row):
        if ch == LV.QMUSH:
            found = (c, r)
            break
    if found:
        break
check("a mushroom block exists in 1-1", found is not None, str(found))
if found:
    c, r = found
    before = len(g.items)
    g.level.hit_block(c, r, big=False)
    out = g.level.tile_at(c, r)
    check("hitting it uses it up", out == LV.USED, f"tile now '{out}'")
g = new_game(1, 1)
qc = next(((c, r) for r, row in enumerate(g.level.rows)
           for c, ch in enumerate(row) if ch == LV.QCOIN), None)
check("a coin block exists", qc is not None)
if qc:
    check("it yields a coin", g.level.hit_block(qc[0], qc[1], big=False) == "coin")
brick = next(((c, r) for r, row in enumerate(g.level.rows)
              for c, ch in enumerate(row) if ch == LV.BRICK), None)
if brick:
    g2 = new_game(1, 1)
    check("small Mario only bumps a brick",
          g2.level.hit_block(brick[0], brick[1], big=False) == "bump")
    g3 = new_game(1, 1)
    check("big Mario breaks it",
          g3.level.hit_block(brick[0], brick[1], big=True) == "break")

print("\n[7] pipes lead somewhere and come back")
g = new_game(1, 1)
now = 2300.0
col = next(iter(g.main.pipe_links))
check("1-1 has an enterable pipe", g.main.pipe_dest(col) == "bonus", f"col {col}")
g.hero.x = col * P.TILE
g.hero.y = (LV.FLOOR - 4) * P.TILE
g.hero.grounded = True
entered = g.enter_pipe(col, now)
check("ducking on it starts the trip", entered and g.state == "piping")
g.update(0.0, False, True, DT, now + P.PIPE_TRAVEL + 0.05)
check("you arrive in the bonus room", g.in_bonus and g.level.name.endswith("bonus"),
      g.level.name)
check("the bonus room is full of coins",
      sum(r.count(LV.COIN) for r in g.level.rows) > 10)
ex = next(iter(g.level.pipe_links))
g.hero.x = ex * P.TILE
g.hero.y = (LV.FLOOR - 1) * P.TILE
g.hero.grounded = True
g.enter_pipe(ex, now + 5)
g.update(0.0, False, True, DT, now + 5 + P.PIPE_TRAVEL + 0.05)
check("and the exit pipe puts you back in the level",
      not g.in_bonus and g.level is g.main, g.level.name)

print("\n[8] hazards: lava kills, water swims")
g = new_game(1, 4)
now = 2400.0
lava = next(((c, r) for r, row in enumerate(g.level.rows)
             for c, ch in enumerate(row) if ch == LV.LAVA), None)
check("1-4 has lava", lava is not None)
if lava:
    lives = g.lives
    g.hero.x = lava[0] * P.TILE
    g.hero.y = lava[1] * P.TILE + 4
    g.hero.big = True                    # lava ignores the mushroom
    g.update(0.0, False, False, DT, now)
    check("lava kills even when big", g.lives == lives - 1 and g.state == "dead")
g = new_game(2, 2)
check("2-2 is a water level", g.level.water_level)
g.hero.x, g.hero.y = 20 * P.TILE, 6 * P.TILE
check("the hero is in water there", g.in_water())
now = 2500.0
g.hero.vy = 0.0
for _ in range(30):
    now += DT
    g.update(0.0, False, False, DT, now)
check("sinking in water is slow", g.hero.vy <= P.WATER_MAX_FALL + 0.01,
      f"vy={g.hero.vy:.2f} (cap {P.WATER_MAX_FALL})")
g.hero.request_jump(0.5, now, water=True)
check("a stroke lifts you", g.hero.vy < 0, f"vy={g.hero.vy:.2f}")

print("\n[9] koopas shell before they die")
g = new_game(1, 3)
k = next((e for e in g.enemies if e.kind == "koopa"), None)
check("1-3 has koopas", k is not None)
if k:
    check("first stomp shells it", k.stomped() is False and k.shell)
    check("second stomp kills it", k.stomped() is True)

print("\n[10] progression walks through all four worlds")
g = new_game(1, 1)
seen = []
for _ in range(len(ALL)):
    seen.append((g.world, g.level_no))
    g.next_level(3000.0)
    if g.state == "complete":
        break
check(f"{len(ALL)} levels in order", seen == ALL, str(seen))
check("finishing the last level completes the game",
      g.state == "complete", g.state)

print("\n[10b] the ending is a celebration, not a stale line of text")
g = new_game(1, 1)
g.state, g.state_until = "complete", 9000.0
g.score, g.coins_taken, g.lives = 123456, 214, 3
ok = True
try:
    for i in range(180):                  # three seconds, every frame
        g.draw(None, 9000.0 + i / P.FPS)
except Exception as exc:
    import traceback; traceback.print_exc()
    ok = False
check("the victory screen animates without erroring", ok)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "platformer.py")).read()
check("no stale 'both worlds' wording survives", "BOTH WORLDS" not in src.upper())
check("the ending names the real size of the game",
      "8 worlds" in src and "32 levels" in src)
for state in ("won", "gameover", "dead", "play"):
    g.state = state
    try:
        g.draw(None, 9100.0)
        fine = True
    except Exception:
        fine = False
    check(f"the '{state}' screen still draws", fine)

print("\n[11] one frame of every level renders")
for w, l in ALL:
    try:
        gg = new_game(w, l)
        gg.draw(None, 4000.0)
        ok = True
    except Exception as exc:
        import traceback; traceback.print_exc()
        ok = False
    check(f"{w}-{l} renders", ok)

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
