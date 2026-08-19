# Gesture Platformer + Two-Hand Control Spike

Two things live here: **[`platformer.py`](platformer.py)**, the game, and
**[`two_hand_spike.py`](two_hand_spike.py)**, the measurement tool the game's
control layer was derived from.

This is the single source of truth for the platformer: what it is, how to run it,
and *why* every threshold is the value it is. The [project README](../README.md)
covers the other games and links here.

## The game

Four worlds of four levels, shaped after the original Worlds 1 to 4:

| | World 1 | World 2 | World 3 (night) | World 4 |
| --- | --- | --- | --- | --- |
| **-1** | overworld: goombas, ? blocks, rising pipes | overworld, busier | koopas + **hammer bros** on brick platforms | **buzzy beetles** that shrug off fire |
| **-2** | underground: brick shelves, coin seams | **underwater**: swim physics, cheep-cheeps | long night level, pipe to a coin room | underground with a **beanstalk** to coin heaven |
| **-3** | athletic: mushroom platforms, koopas | bridges over water, leaping cheeps | **moving lifts**, vertical and horizontal, **paratroopas** | tall mushroom platforms, paratroopas, a hammer bro |
| **-4** | castle: lava, podoboos, axe | castle, longer | castle with **rotating firebars** | **maze castle**: take the low road and it sends you back |

Five visual themes with their own palettes and parallax backgrounds — hills and
clouds by day, a starfield and purple ground at night, black-and-blue underground,
grey stone in the castles, deep blue underwater.

Along the way: **mushrooms** from ? blocks (big Mario takes a hit before dying and
smashes bricks from below), **pipes you can enter** by ducking on a dark-mouthed one,
a **beanstalk** you walk into to be carried up to a sky full of coins (walk off the
end to drop back), koopas and buzzy beetles that shell before they die, paratroopas
that lose their wings to a stomp, flagpoles, axes, and lava that ignores your
mushroom.

## Running and options

```bash
./run.sh mario                          # from the repo root (also: platformer, plat)
./run.sh mario --world 3 --level 3      # jump straight to the moving lifts
./run.sh mario --no-camera              # keyboard only, no webcam needed
```

Anything after the game name is passed straight through to `platformer.py`:

| Option | What it does |
| --- | --- |
| `--world N` | Starting world, 1–4 (default 1) |
| `--level N` | Starting level, 1–4 (default 1) |
| `--no-camera` | Skip hand tracking entirely and play on the keyboard |
| `--camera N` | Camera device index, if the default is not the right one |
| `--steer-zone LO HI` | The steering band, as fractions of frame width (default `0.02 0.60`). Lower `HI` for a shorter reach to full speed |
| `--swap-hands` | Flip which physical hand steers, for a camera that reports handedness reversed. `H` does the same mid-game |
| `--log [PATH]` | Record every hand pose and gesture event to CSV, for `analyse_log.py` |

### Controls

| Hand | Gesture |
| --- | --- |
| Left (steer) | lean left/right to walk, lean further to run; flick **up** to jump (harder = higher); flick again mid-air to double jump; drop the hand low to duck — and to enter a pipe |
| Right (fire) | sweep the hand quickly to throw a fireball |

| Key | Action |
| --- | --- |
| `H` | Swap left/right hand detection, if the HUD shows them the wrong way round |
| `N` | Skip to the next level |
| `R` | Restart from the first life |
| `Esc` / `Q` | Quit |
| Arrows, `Shift`, `Space`, `X`, `Down` | Keyboard fallback: move, run, jump, fire, duck — always live alongside the camera |

There is deliberately **no calibration key**: the steering band is fixed, so there
is nothing to calibrate. The HUD names the hand holding each role (`STEER left`,
`FIRE right`), so a reversed camera is obvious at a glance — press `H` once if so.

### Recording and analysing a session

```bash
./run.sh mario --log                                  # writes mario/play-log-<time>.csv
../.venv/bin/python analyse_log.py play-log-*.csv     # turn it into a tuning report
```

The report covers latency, how much of the steering range you actually used, whether
jump height varied or saturated, hand-tracking dropouts, and what to change. Every
threshold in this file was set from one of these, which is why they are worth taking:
see [what a real play session changed](#what-an-87-second-real-play-session-changed).

### Tests

Headless — no camera, no display. They drive the game with a bot and feed the
tracker synthetic hand landmarks:

```bash
cd mario
for t in test_spike.py test_roles.py test_neutral.py test_zone.py test_platformer.py; do
    ../.venv/bin/python $t
done
```

250 checks. The important one is `test_platformer.py`: an autoplay bot has to finish
**all 16 levels, walking and running, using the weakest jump the gesture layer can
produce**. That is the guarantee that no gap needs a hard flick or a run-up.

### Levels are built, not typed

Hand-authoring 14x130 ASCII maps is exactly how the first draft got pipes standing
inside pits and platforms roofing every jump. [`levels.py`](levels.py) is a builder
instead: `pipe`, `stairs`, `platform`, `mushroom_platform`, `lift`, `firebar`,
`lava`, `vine` and `enemy` each refuse an illegal placement, and every level is
validated before it loads. Nine rules, each one earned by a bug that shipped:

1. Nothing floats over a pit.
2. No pit wider than 2 tiles — 3 left only a 1.2x margin on the weakest walking
   jump, and the bot died in pits it should have cleared.
3. No pipe taller than 3 tiles: four is 160px against a 158px weakest jump, which is
   a wall, not an obstacle.
4. No platform or pipe within 3 tiles of a pit, or the jump across bonks a ceiling.
5. **No raised shelf whose right end sits within a walk-off of a pit.** Stepping off
   a shelf carries you up to 5 tiles at running speed, so a shelf ending 3 tiles
   short of a pit funnels you straight in. This one was invisible until traced, and
   it existed **14 times** across the levels — height-aware now, since the distance
   depends on how high the shelf is.
6. No enemy under a low platform, or it cannot be jumped and fire is the only out.
7. No enemy within 8 tiles *past* a pit: a running jump covers seven, so an enemy
   just beyond one stands exactly where you come down.
8. No block higher than 5 tiles without a stepping stone within 4 columns — and
   anything a *required* route depends on must be within 3, because that is all the
   weakest flick clears. 4-4's high road was at 5 and the maze was a dead end.
9. Mushroom-platform stalks and firebar mounts are exempt from the reachability
   rules: one is decorative, the other is a hazard mount, and neither is a floor.

Then an autoplay bot must finish **all sixteen levels, walking and running, on the
weakest flick the gesture layer can produce** — 32 runs, currently zero deaths on 30
of them. The bot waits out podoboos and thrown hammers (timed hazards are rhythm,
not geometry), keeps moving past leapers, which come up through the bridge you stand
on, jumps off raised ledges rather than walking off them, and climbs to 4-4's high
road instead of looping for ever.

### Engine bugs the bot found

None of these would have been fixed by tuning levels:

- **Landing on an enemy could never be a stomp.** Movement runs before collisions,
  so `move_y` had already zeroed `vy` when the stomp test asked "are you falling?"
  Every landing on a goomba was a death.
- **Coming down onto a walker has to count as a stomp.** Patrolling enemies wander
  into wherever a jump lands, so no placement rule can keep them clear.
- **A kicked shell overtook the player from behind at 7px/frame.** Being killed by a
  shell you kicked is a miserable way to lose a life, so a stomped shell is now
  scenery until you stomp it again.
- **Flyers roamed the whole level.** A paratroopa drifted 27 tiles from home and
  clipped the player mid-jump; they patrol a bounded area now, which is also the
  only way a level stays predictable enough to design.
- **You could get stuck in coin heaven.** The beanstalk carried you up to an area
  whose only exit was a pipe you had to know to duck on. Walking off the end now
  drops you back, as it does in the original.

### Which hand does what

**Your left hand steers; your right hand throws.** That comes from MediaPipe's
handedness label, not from which side of the frame you happen to hold each hand —
so reaching across your body no longer hands your controls to the other hand.

Two things make this less trivial than it sounds, and both were the actual bug:

- **A single frame's label flickers.** Snapping roles to it means the controls
  swap for a frame or two mid-jump, which is exactly the "hands sometimes get
  mistakenly recognized" symptom. A role only moves once a clear majority of the
  last ~0.4s agrees (`HANDEDNESS_SWAP_VOTES` of `HANDEDNESS_VOTES`), and labels
  below `HANDEDNESS_MIN_SCORE` are ignored entirely. When both hands are
  confidently labelled at the start, it settles immediately rather than waiting for
  a vote.
- **Polarity depends on the mirror.** MediaPipe labels handedness as if the image
  were a selfie view; this code flips the frame before inference, which produces
  exactly that, so the labels already name your real hands. If a camera turns out
  reversed, **press `H`** (or start with `--swap-hands`) — no code change needed.
  The HUD names the hand holding each role (`STEER left`, `FIRE right`), so a
  reversed setup is obvious at a glance.

Position prediction still does the frame-to-frame tracking underneath, because
that is what survives two hands crossing over each other. Handedness only decides
which track owns which role, so crossing your hands keeps working.

### Steering: a fixed band of the frame

Speed comes from where the steering hand sits inside a **fixed band of the camera
frame** — by default from 2% to 60% of the frame width:

```
 |<---------------- band ---------------->|
 2%              31%                     60%          100%
full left       STOP                 full right    (still full right)
```

The band's right edge is full speed forward, its left edge is full speed back, its
middle is stopped, and anything past an edge stays pinned at full. It is drawn
directly on the camera preview with `FULL LEFT` / `STOP` / `FULL RIGHT` marked, so
full speed is somewhere you can see rather than somewhere you have to hunt for.
`--steer-zone LO HI` moves it.

Nothing is calibrated and nothing drifts, which is the entire point. The previous
scheme measured the player's own centre and reach from a sweep, and it kept parking
the band wherever the hand happened to be resting. One logged session ended up with
a band spanning **61%–91%** of the frame: full speed sat at the very edge of shot,
a hand at 83% produced only +0.47 — a walk — and the player reported that moving
forward was painful. Three separate attempts at inferring the centre automatically
each failed on their own horn:

- **First pose as centre.** The origin landed wherever the hand entered, usually a
  corner. A hand at frame centre then read `RUN RIGHT +0.90`.
- **Per-side gain.** Scaling each direction to the room on that side keeps both
  reachable, but a hand resting near its own edge of frame has little room outward
  and lots inward: hair-trigger one way, sluggish the other.
- **Drift correction.** Gate it on the axis reading small and a wrong origin can
  never heal — it reads large, which switches the correction off. Gate it on the
  hand being still and a deliberately held lean gets absorbed, and holding a lean
  is exactly how you run.

The sweep scheme is still in the file as `STEER_MODE = "relative"` and still fully
tested ([`test_neutral.py`](test_neutral.py)), because per-player range is a real
thing to want. It just should not be the default. Duck is a fixed band too: hand
below `DUCK_ZONE_Y` of the frame height.

### What an 87-second real play session changed

The first log taken from actual play (rather than from demonstrating gestures)
settled several things and exposed two bugs in this tooling.

**The origin is right.** For 41% of the session the steering hand was genuinely
still, and at rest the axis read a median of **+0.00** (mean +0.079). Symmetric
gain fixed the sensitivity mismatch.

**Full deflection one way was unreachable.** With the centre 212px into a 640px
frame and the edge treated as the literal pixel boundary, "full left" sat at
**x=8px** — the hand is lost long before it gets there. The best the session ever
reached was −0.73. Each edge now reserves 10% of the frame width, putting full
left at x=64 against a logged leftmost reach of 62px.

**The sweep never happened.** Zero direction reversals in the first six seconds:
the player never swept, so calibration silently fell back to the steady-hand
method. An on-screen prompt during play is too easy to miss, so calibration is now
a **blocking start screen** with the camera feed shown large. A feature that only
works when noticed does not work.

**Gestures made while playing are far gentler than gestures made while
demonstrating.** In-game flicks peaked at 2.6–5.7 span/s against 4–14 in the test
sessions, so no jump reached even 0.7 strength. `FLICK_VEL_MAX` went 22 → 10 → 6
across three logs, and `FLICK_VEL_MIN` 4.0 → 3.0 → 2.5 → 1.8. Twelve of 22 throws
only just cleared `THROW_VEL_MIN`, so that dropped 3.0 → 2.2. **Tune against a log
of someone playing, never against a log of someone showing you the gesture.**

Two bugs in the analysis tooling, both of which made its own advice wrong:

- The game's log had no `meta` row, so the analyser fell back to its defaults and
  confidently recommended lowering `FLICK_VEL_MAX` "from 22" when it was 10. The
  metadata is now written by one shared function used by both programs.
- It reported **0 fireballs in a session with 22 throws**, because it only knew the
  old `pinch` event kind. Silence from a tool is not evidence of absence.

### Why it is tuned to be forgiving

Photon-to-pixel latency is ~50–80ms and cannot be removed, so the game absorbs it
instead of punishing it: coyote time (150ms after a ledge), jump buffering (150ms
before landing), a generous stomp window, and **ledge assist** — if a jump lands
a few pixels low and clips the *side* of the far platform, the hero is stood on
top of it rather than wedged and dropped. That last one is the single biggest
fairness win in the file; without it, near misses read as the game cheating.

Level geometry is machine-checked by [`test_platformer.py`](test_platformer.py)
against four rules, every one of which came from a bug that actually shipped into
the first draft:

1. Nothing hovers over a gap. The first draft put the pipes *in* the gaps, making
   them mid-air walls exactly where you need to jump.
2. No platform or pipe within 3 tiles of a gap. Otherwise the jump across bonks a
   ceiling, loses its upward velocity and drops you in — this one was the hardest
   to find, because the jump just ended early with no visible cause.
3. No enemy under a low platform, or it cannot be jumped and fire is the only out.
4. No enemy within 3 tiles of a gap, or the jump you take to clear it lands you in
   the pit.

An autoplay bot then proves both levels are completable **at the weakest flick the
gesture layer can produce (strength 0.45), walking and running, with zero deaths.**
That is the real fairness guarantee: no gap needs a hard flick or a run-up.

---

# The Spike

Not a game. This is the measurement tool that decides whether a hand-controlled
platformer is worth building, by answering the three things that could kill it:

1. **Can two hands hold distinct roles reliably?** One hand steers, the other fires.
2. **Does a velocity-triggered flick work as a jump?** A platformer needs variable
   jump height and precise timing, which a "hand is up" posture can't express.
3. **How much end-to-end latency is there, really?** Pac-Man tolerates ~200ms.
   A platformer does not.

```bash
./run.sh spike          # from the repo root
```

## What you see

- **Left** — the camera feed with each hand skeleton labelled `STEER` or `FIRE`,
  the steering neutral crosshair, and the pinch indicator.
- **Right** — live analog steer bar (with the deadzone and run threshold marked),
  flick velocity, pinch gap, measured latency, and a timestamped event log where
  every jump and fireball shows the latency of the event that caused it.
- **Bottom** — a minimal sandbox: one square, one gap, gravity. It exists so you
  can *feel* the lag instead of reading it. The gap is what makes coyote time and
  jump buffering measurable.

## Spike controls

| Key | Action |
| --- | --- |
| `C` | Re-learn the steering centre — only does anything with `STEER_MODE = "relative"`, since the shipped zone mode has no centre to learn |
| `T` | Toggle the latency assists (coyote time + jump buffer) — A/B the difference |
| `K` | Toggle keyboard control, to compare the feel against arrow keys + space |
| `R` | Reset the sandbox player |
| `Esc` / `Q` | Quit, printing a latency summary |

## The gestures

**Steer hand** — horizontal wrist offset from a calibrated neutral, as an analog
axis rather than four discrete zones. Small lean walks, deep lean runs, so
"fast run" is continuous instead of a separate gesture. Jump is an upward wrist
*flick*, with jump height scaled by flick speed. Duck is a sustained low hand.

**Fire hand** — thumb-to-index pinch, edge-triggered with a cooldown.

## Three design decisions worth keeping

**Everything is measured in hand-spans, not pixels.** The unit is the distance
from wrist to middle-finger MCP, so thresholds hold for a kid's hand at 40cm and
an adult's at 80cm without recalibration. The test suite verifies that a given
physical lean produces the same analog value at two different hand sizes.

**Jump and fire are edge-triggered, never debounced.** Pac-Man smooths direction
with a 4-frame majority vote ([`../pacman/pacman.py`](../pacman/pacman.py) around
the `direction_buffer`), which costs ~130ms at 30fps. Direction can afford that;
a jump cannot. Here the flick fires on the frame the velocity threshold is
crossed, gated by a refractory window and a re-arm threshold (the hand must slow
down before it can fire again, or one long upward sweep fires twice).

**Roles come from predicted position, not MediaPipe handedness.** The frame is
mirrored, which inverts handedness labels, and hardcoding a hand would break for
left-handed players. Each role carries a smoothed velocity and is matched against
where it was *heading* — matching on last-known position alone fails in exactly
the case that matters, two hands crossing over each other at similar speed, where
each ends up where the other was and the roles swap mid-jump.

## Measured latency

On an M4 Pro at 640x480, 240 samples, no hands in frame (worst case — palm
detection re-runs every frame instead of taking the tracking shortcut):

| capture → | mean | p95 |
| --- | --- | --- |
| landmarks | 14.5 ms | 19.9 ms |
| game loop | 24.7 ms | 34.9 ms |

Tracker held a full 30 fps with zero dropped frames.

**These figures exclude the sensor and USB time before `read()` returns**, which
is typically another 20–50ms on a webcam and is not measurable from inside the
process. Treat the honest photon-to-pixel number as roughly 50–80ms. That is well
inside what a forgiving platformer absorbs, and far better than the ~200ms a
debounced single-hand pipeline would cost.

## Logging and tuning from real motion

```bash
./run.sh spike --log                    # auto-names mario/spike-log-<timestamp>.csv
./run.sh spike --log /tmp/session.csv   # or pick the path
python analyse_log.py spike-log-*.csv
```

The log is one flat CSV: a `pose` row per tracked frame (~30/s) and an `event`
row per flick and pinch, plus a `meta` row stamping the thresholds in force and
`marker` rows for key presses. Events carry the *pose* timestamp they came from,
not the time the render loop got round to them, so the event timeline lines up
against the pose stream without inheriting render jitter.

[`analyse_log.py`](analyse_log.py) turns that into a tuning report: latency,
hand-presence dropouts, how much of the steering range you actually used, whether
jump strength varied or saturated, suspected role swaps, and a recommendations
block.

The most useful part is the **near-miss counts**. A gesture that fires reliably
only tells you a threshold isn't too high — it's the motions that *tried and
failed* that reveal one that is. Those show up two ways: local peaks in wrist
velocity that fell just short of `FLICK_VEL_MIN`, and pinch gaps that dipped into
the hysteresis band without ever crossing `PINCH_ON`. Neither is visible while
playing; you just feel like the game ignored you.

## What a 129-second session actually taught us

The first real two-hand session (3,874 poses) changed five constants and found
three bugs. Worth reading before tuning anything, because most of it was
invisible while playing.

**Roles traded places, two different ways.** The player noticed it; the log
explained it.

- *Partial matching was missing.* Role matching required the whole two-hand
  pairing to pass the radius check. At t=97.3s one hand appeared 350px from any
  prediction, so no pairing was valid — and the code threw away *both*
  identities, including a match that was 1px off, then re-seeded by screen side.
  Matching is now greedy over individually valid pairs, so a good match is never
  discarded because its partner failed.
- *Tracks expired too fast.* Nine dropouts outlasted `ROLE_MEMORY=0.5s` (worst
  5.1s), and on return roles were guessed purely from screen side. `ROLE_MEMORY`
  is now 1.2s, and re-seeding prefers each role's **last known side** over
  "leftmost steers" — rest your hands for a second and they come back as
  themselves.

**Three of four "swaps" my own analyser reported were false.** It asked whether
the cross-assignment explained the new positions better, which is exactly what
happens after a crossing that was handled *correctly* — each role really is
nearer to where the other one was. It now flags a role only when its position
moves faster than an arm can (>2500 px/s). Of 11 crossings in the session, the
prediction matching held through 10; one was a genuine teleport at 5282 px/s.

**A hand span of 25px manufactured a 300 span/s jump.** Span is the denominator
for every threshold here, and the first pose after a hand is re-acquired carries
a bad one (25px against a normal 65px). The next frame's ratio then swings
enough to look like a violent upward flick from a hand that barely moved. Fixed
with a 15-pose median plus a 5-pose warm-up gate that refuses to jump until the
span settles. Replaying the session confirms it: peak flick velocity drops from
300 span/s to 55, with no genuine flick lost.

**Jump height was not actually variable.** `FLICK_VEL_MAX=22` was set from
synthetic motion; real flicks ran 4–8 span/s, so strength sat at the floor
(p10–p90 of 0.45–0.58). The two jumps that did reach full height were the span
artefacts above, not hard flicks — which is why the recommendation logic now
judges spread on percentiles instead of min/max, where a single outlier made a
stuck distribution look healthy. At `FLICK_VEL_MAX=10` the same session's motion
yields a 0.45–0.71 spread.

**Thresholds were too demanding across the board.** 34 flick attempts peaked at
3.3–3.9 span/s and never fired (`FLICK_VEL_MIN` 4.0 → 3.0). 49 pinches dipped to
~0.45 and never fired (`PINCH_ON` 0.38 → 0.50). Steering sat at full deflection
39% of the time, behaving like an on/off switch (`STEER_FULL_LEAN` 2.2 → 3.4).

**Latency held up.** 18.7ms mean to landmarks, 27.3ms to the game loop, 30fps
with zero dropped frames across 129 seconds of real two-hand tracking — barely
worse than the empty-frame measurement, so tracking two hands is not the
bottleneck.

**Framing is the real weak point.** The steer hand was missing for 8% of poses
and the fire hand 7.6%, with one 5.1-second dropout. No threshold fixes that;
it needs a wider field of view or standing further back.

## Session two: the role fix held, and the pinch gesture didn't

A second 170-second session (5,116 poses) confirmed the role work and exposed a
deeper problem.

**Roles are fixed.** Zero teleports, and all 3 hand crossings survived. Steering
saturation dropped from 39% to 20%, and jump height finally varied (p10–p90 of
0.46–0.68). Latency unchanged at 18.5ms to landmarks, 27.2ms to the game loop.

**The velocity maths was wrong, and no amount of filtering would have fixed it.**
Flicks were still reporting impossible speeds — up to 195 span/s — and the cause
was not noise. The code stored `y/span` and differentiated *that*:

```
d(y/s)/dt = (y'·s − y·s') / s²
```

When the span changes — which it does violently while a returning hand's
landmarks settle, ramping 6px → 19px over eight frames — the `s'` term swamps the
real motion. At t=171.4s a hand moving 21px **downwards** reported 195 span/s
**upwards**. The fix is to store raw pixels, differentiate position, and divide by
the span once at the end. Peak velocity across both sessions dropped from 122 to
7.8 span/s, with zero impossible values and no genuine flick lost. *Differentiate
the position, then scale — never differentiate the ratio.*

**Jump now also requires displacement, not just speed.** Landmark jitter clears a
speed threshold over one or two frames without the hand going anywhere, which is
what kept `FLICK_VEL_MIN` artificially high. Requiring 0.20 hand-spans of actual
upward travel made it safe to drop the speed floor to 2.5, and the resulting flick
rate is stable across both sessions (19 and 21 per minute).

**Pinch is the wrong gesture, and this is the main finding.** A fixed threshold on
thumb-index gap could not work: the same person's median resting gap was 0.75 in
one session and 0.50 in the other. Raising `PINCH_ON` to suit the second session
made the hand read as pinched 47% of the time. Switching to a threshold relative
to the player's own rolling open-hand baseline helped — it lands in the valley of
what turns out to be a bimodal distribution, and fire rates match across sessions
(11.6 vs 13.0 per minute) — but it does not rescue the gesture. In session two the
fire hand's gap **never opened past 0.60 for a continuous 34 seconds**, and 53% of
all poses were below it. The player's relaxed hand *is* a loose fist, so "closed"
carries no intent.

That is a posture problem, not a threshold problem, and it matches the principle
this spike started from: **velocity edges generalise, absolute postures don't.**
The flick works across both sessions with one set of numbers; the pinch needed
different numbers each time and still failed. The recommendation is to make fire a
*forward flick* of the fire hand — the same edge-triggered velocity detector that
already works for jumping — rather than to keep tuning a posture test.

Two smaller things the tests caught, worth recording because both fixes were
initially wrong:

- Learning the pinch baseline only from open frames sounds obviously right and is
  actively harmful: a frozen-high reference puts the release threshold above
  anything the hand reaches, latching it shut for up to 40s in replay. Letting
  the baseline sag while pinched lowers the release bar and self-corrects.
- Gating *release* on the same trustworthiness test as *firing* is a latch by
  construction. There must always be a release path — hence `PINCH_RELEASE_ABS`.

## Spike tests

```bash
cd mario && ../.venv/bin/python test_spike.py && ../.venv/bin/python test_roles.py
```

60 headless checks, no camera needed — synthetic landmarks for the gesture logic,
and for the role bugs, the actual coordinate sequences replayed out of the session
logs. Between them they caught six real defects, so they are worth keeping.

## Tuning

Every threshold is a named constant at the top of
[`two_hand_spike.py`](two_hand_spike.py), grouped under `GESTURE TUNING`, in
hand-spans and hand-spans/second. The ones most likely to need adjusting per
player:

- `FLICK_VEL_MIN` / `FLICK_VEL_MAX` — jump sensitivity and the range that maps to
  full jump height. If every jump comes out at full height, `FLICK_VEL_MAX` is
  too low.
- `STEER_FULL_LEAN` — how far the hand travels for full deflection.
- `PINCH_ON` / `PINCH_OFF` — pinch sensitivity, with hysteresis between them.

## Known caveats

- A 10-second run with no hands in frame produced one spurious flick, so false
  detections do happen. Raise `min_hand_detection_confidence` in `_make_detector`
  if it becomes noticeable during play.
- These caveats were written against the sweep-calibrated `"relative"` steering
  mode. The shipped default is `"zone"`, which has no centre to drift.
- `cv2` and `pygame` ship separate copies of SDL2, so macOS prints a wall of
  `objc[...] Class SDL... implemented in both` warnings on startup. Harmless, and
  pre-existing in the other two games.
