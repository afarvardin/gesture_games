"""Regression tests for the steering-origin bug seen in the session screenshots.

The failure: neutral was taken from the first pose the steer hand appeared in,
which was the bottom-left corner. Neutral landed at x~75 in a 640px frame, so a
hand at frame centre read RUN RIGHT +0.90 and left was physically unreachable.
"""
import os, sys, math, statistics
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import two_hand_spike as S

# This suite covers the sweep-calibrated "relative" scheme. The shipped default is
# "zone" (see test_zone.py); relative is still selectable and still tested.
S.STEER_MODE = "relative"

FAIL = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)

class LM:
    __slots__ = ("x", "y", "z")
    def __init__(s, x, y, z=0.0): s.x, s.y, s.z = x, y, z

def hand(cx, cy, span=0.125):
    l = [LM(cx, cy) for _ in range(21)]
    l[S.WRIST] = LM(cx, cy)
    l[S.MIDDLE_MCP] = LM(cx, cy - span)
    l[S.INDEX_TIP] = LM(cx, cy - span * 2)
    l[S.THUMB_TIP] = LM(cx + 0.9 * span * 0.75, cy - span * 2)
    return l

class Bare(S.TwoHandTracker):
    def _open_camera(self, preferred_id=0):
        self.cap, self.camera_ok, self.error_lines = None, False, []
    def _make_detector(self): return None

W, H = 640, 480
SPAN_PX = 0.125 * H                    # 60px, matching the session logs
FULL = S.STEER_FULL_LEAN * SPAN_PX     # px from neutral to full deflection

def hold(tr, x_px, y_px, n, t0=100.0, dt=1/30):
    """Hold the hand still at a pixel position for n poses."""
    out = None
    for i in range(n):
        out = tr._process_steer(hand(x_px / W, y_px / H), W, H, t0 + i * dt)
    return out, t0 + n * dt

STEADY = int((S.SWEEP_TIME + 0.6) * 30) + S.NEUTRAL_STABLE_N   # poses needed to
                                                               # reach the fallback

def sweep(tr, lo_px, hi_px, y_px=240, t0=100.0, dt=1/30, cycles=2):
    """Sweep between two x positions, as the calibration prompt asks."""
    now = t0
    steps = int(S.SWEEP_TIME / dt) + 4
    for i in range(steps):
        phase = math.sin(2 * math.pi * cycles * i / steps)
        x = (lo_px + hi_px) * 0.5 + (hi_px - lo_px) * 0.5 * phase
        tr._process_steer(hand(x / W, y_px / H), W, H, now)
        now += dt
    return now

print("\n[1] the bug: a hand entering at the bottom-left must not become neutral")
t = Bare()
# Exactly the screenshot geometry: first sighting low and far left.
out, now = hold(t, 75, 406, 1)
check("no neutral from a single corner pose", t._neutral is None)
check("no steering output before calibration", abs(out["analog"]) < 1e-9,
      f"analog={out['analog']:+.2f}")
out, now = hold(t, 75, 406, 20, t0=now)
check("a hand parked in the corner never becomes neutral", t._neutral is None,
      "edge margin rejects it")
# Now the player brings the hand into frame properly.
out, now = hold(t, 320, 240, STEADY, t0=now)
check("a steady hand in open frame does become neutral", t._neutral is not None,
      f"neutral_x={t._neutral[0]:.0f}" if t._neutral else "")
check("neutral lands where the hand actually was", abs(t._neutral[0] - 320) < 3)
check("hand at its own neutral reads centred", abs(out["analog"]) < 1e-9,
      f"analog={out['analog']:+.2f}")

# Entering the frame is not a sweep.
t_enter = Bare()
now_e = 500.0
for i in range(int(S.SWEEP_TIME * 30) + 10):        # slide in from the left, one way
    x = min(60 + i * 4, 330)
    t_enter._process_steer(hand(x / W, 240 / H), W, H, now_e); now_e += 1/30
check("a hand entering the frame is not read as a sweep",
      t_enter._sweep_lean is None,
      "one-way travel has no reversals, so it cannot set the gain")

print("\n[2] both directions reachable from the learned neutral")
t2 = Bare()
_, now = hold(t2, 320, 240, STEADY)
res = {}
for label, x in (("full left", 320 - FULL), ("full right", 320 + FULL)):
    t2._y_hist.clear()
    o, now = hold(t2, x, 240, 1, t0=now + 1.0)
    res[label] = o["analog"]
check("full left reaches -1.00", res["full left"] < -0.95, f"analog={res['full left']:+.2f}")  # noqa
check("full right reaches +1.00", res["full right"] > 0.95, f"analog={res['full right']:+.2f}")
check("the axis is symmetric", abs(res["full left"] + res["full right"]) < 0.05,
      f"{res['full left']:+.2f} vs {res['full right']:+.2f}")

print("\n[3] an off-centre rest pose keeps rest at zero AND stays reachable")
# Player rests left of centre -- legal (just inside the edge margin), but with
# less room to the left than a nominal full lean needs.
t3 = Bare()
rest_x = S.NEUTRAL_EDGE_MARGIN * W + 6         # ~166px
o, now = hold(t3, rest_x, 240, STEADY)
check("neutral was learned", t3._neutral is not None)
nx = t3._neutral[0]
check("neutral stays exactly at the rest pose", abs(nx - rest_x) < 3,
      f"rest={rest_x:.0f}px neutral={nx:.0f}px")
check("the hand reads centred while resting", abs(o["analog"]) < 1e-9,
      f"analog={o['analog']:+.2f}  <- this is what clamping the origin broke")
lean_l, lean_r = t3._side_leans(nx, SPAN_PX, W)
check("both directions get the SAME travel", abs(lean_l - lean_r) < 1e-9,
      f"left {lean_l:.0f}px right {lean_r:.0f}px -- asymmetric gain is what made one "
      f"direction hair-trigger and the other sluggish")
check("travel is sized to the narrower side", lean_l <= min(nx, W - nx) + 1,
      f"{lean_l:.0f}px against {min(nx, W-nx):.0f}px of room")
check("but never small enough to be twitchy", lean_l >= S.STEER_SIDE_MIN * FULL - 1,
      f"{lean_l/FULL:.2f}x nominal (floor {S.STEER_SIDE_MIN})")
t3._y_hist.clear()
o, now = hold(t3, nx - lean_l, 240, 1, t0=now + 1.0)
check("full left is reachable and saturates", o["analog"] < -0.95,
      f"analog={o['analog']:+.2f}")
t3._y_hist.clear()
o, now = hold(t3, nx + lean_r, 240, 1, t0=now + 2.0)
check("full right is reachable and saturates", o["analog"] > 0.95,
      f"analog={o['analog']:+.2f}")

print("\n[4] the screenshot regression, end to end")
# Reproduce it: corner first sighting, then hand at frame centre.
t4 = Bare()
_, now = hold(t4, 75, 406, 3)                       # corner appearance
_, now = hold(t4, 320, 240, STEADY, t0=now)
t4._y_hist.clear()
o, now = hold(t4, 320, 240, 1, t0=now + 1.0)
check("hand at frame centre no longer reads RUN RIGHT", abs(o["analog"]) < 0.15,
      f"analog={o['analog']:+.2f}  (was +0.90 in the session)")

print("\n[5] slow drift correction, without cancelling a held lean")
t5 = Bare()
_, now = hold(t5, 320, 240, STEADY)
n0 = t5._neutral[0]
# Player shifts 25px right and stays there: neutral should follow.
_, now = hold(t5, 345, 240, 300, t0=now)            # 10s
check("neutral follows a small posture shift", t5._neutral[0] > n0 + 12,
      f"{n0:.0f}px -> {t5._neutral[0]:.0f}px (hand moved to 345)")
# A deliberate full lean held for 10s must NOT be absorbed.
t6 = Bare()
_, now = hold(t6, 320, 240, STEADY)
n0 = t6._neutral[0]
o, now = hold(t6, 320 + FULL, 240, 300, t0=now)     # 10s of hard right
check("a held full lean is not absorbed by recentring", o["analog"] > 0.95,
      f"analog after 10s = {o['analog']:+.2f}, neutral moved "
      f"{t6._neutral[0]-n0:+.1f}px")

print("\n[6] origin is re-learned after a long absence")
t7 = Bare()
_, now = hold(t7, 250, 240, STEADY)
before = t7._neutral[0]
t7._steer_gone_since = now
gone = now + S.NEUTRAL_RECAL_AFTER + 1.0
# Simulate what _on_result does when the hand comes back after a long gap.
if gone - t7._steer_gone_since > S.NEUTRAL_RECAL_AFTER:
    t7._neutral = None
    t7._neutral_buf.clear()
check("long absence clears the stale origin", t7._neutral is None)
_, now = hold(t7, 420, 240, STEADY, t0=gone)
check("new origin learned at the new rest pose", t7._neutral is not None
      and abs(t7._neutral[0] - 420) < 5,
      f"{before:.0f}px -> {t7._neutral[0]:.0f}px")

print("\n[7] C key re-learns rather than snapping to the current instant")
t8 = Bare()
_, now = hold(t8, 320, 240, STEADY)
t8.calibrate()
check("calibrate() requests a re-learn", t8._request_calibrate)

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
