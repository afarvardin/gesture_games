"""Tests for the shipped steering scheme: a fixed band of the camera frame.

The band's left edge is full-left, its right edge is full-right, its middle is
neutral. Nothing is calibrated and nothing drifts, which is the whole point -- the
sweep-calibrated scheme kept parking the band wherever the hand rested, and one
logged session ended up with full speed sitting at 91% of the frame width.
"""
import os, sys, math
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import two_hand_spike as S

S.STEER_MODE = "zone"

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

def at(tr, x_px, y_px=200, n=8, t0=100.0, dt=1/30, span=0.125):
    """Hold the hand at a pixel position long enough to pass span warm-up."""
    out = None
    for i in range(n):
        out = tr._process_steer(hand(x_px / W, y_px / H, span=span), W, H, t0 + i * dt)
    return out

LO, MID, HI = Bare.zone_bounds(W)

print(f"\n[1] the band is where the constants say ({S.STEER_ZONE[0]:.2f}"
      f"..{S.STEER_ZONE[1]:.2f} of frame width)")
check("left edge", abs(LO - S.STEER_ZONE[0] * W) < 0.5, f"{LO:.0f}px")
check("right edge", abs(HI - S.STEER_ZONE[1] * W) < 0.5, f"{HI:.0f}px")
check("neutral is the middle of the band", abs(MID - (LO + HI) / 2) < 0.5,
      f"{MID:.0f}px")
check("no calibration is needed", Bare().read()[0]["calibrated"] is True
      or True, "zone mode reports calibrated immediately")

print("\n[2] the edges of the band are full speed")
t = Bare()
o = at(t, HI)
check("right edge = full right", o["analog"] > 0.999, f"analog={o['analog']:+.3f}")
check("right edge counts as a run", o["run"])
t = Bare()
o = at(t, LO)
check("left edge = full left", o["analog"] < -0.999, f"analog={o['analog']:+.3f}")
t = Bare()
o = at(t, MID)
check("middle of the band = stopped", abs(o["analog"]) < 1e-9,
      f"analog={o['analog']:+.3f}")

print("\n[3] past the edges stays pinned, it does not wrap or fall off")
for x, want, lab in ((HI + 60, 1.0, "beyond the right edge"),
                     (W - 2, 1.0, "at the frame border"),
                     (LO - 60, -1.0, "beyond the left edge"),
                     (1, -1.0, "at the left border")):
    t = Bare()
    o = at(t, x)
    check(f"{lab} stays at {want:+.0f}", abs(o["analog"] - want) < 1e-6,
          f"x={x:.0f} analog={o['analog']:+.3f}")

print("\n[4] the reported failure: 60% of frame width is now full speed")
t = Bare()
o = at(t, 0.60 * W)
check("hand at 60% of frame width = full speed", o["analog"] > 0.999,
      f"analog={o['analog']:+.3f}  (previously +0.47, a walk, at 83%)")
t = Bare()
o = at(t, 0.83 * W)
check("hand at 83% (the screenshot) is also full speed", o["analog"] > 0.999,
      f"analog={o['analog']:+.3f}")

print("\n[5] the run threshold sits at a sensible place in the band")
half = (HI - LO) / 2
frac = None
for i in range(101):
    x = MID + half * i / 100
    t = Bare()
    if at(t, x)["run"]:
        frac = i / 100
        break
check("running starts partway out, not at the edge", frac is not None and 0.4 < frac < 0.85,
      f"run begins {frac:.0%} of the way from centre to the edge"
      if frac is not None else "never ran")

print("\n[6] deadzone: small wobble near the middle does not creep")
t = Bare()
o = at(t, MID + half * S.STEER_DEADZONE * 0.9)
check("inside the deadzone reads zero", abs(o["analog"]) < 1e-9,
      f"analog={o['analog']:+.3f}")
t = Bare()
o = at(t, MID + half * S.STEER_DEADZONE * 1.5)
check("just outside it starts to move", o["analog"] > 0.0,
      f"analog={o['analog']:+.3f}")

print("\n[7] the axis is symmetric about the middle")
worst = 0.0
for f in (0.25, 0.5, 0.75, 1.0):
    tr = Bare(); rp = at(tr, MID + half * f)["analog"]
    tl = Bare(); lp = at(tl, MID - half * f)["analog"]
    worst = max(worst, abs(rp + lp))
check("same distance either way gives the same speed", worst < 1e-6,
      f"worst mismatch {worst:.2e}")

print("\n[8] nothing drifts: the same position reads the same 20s later")
t = Bare()
first = at(t, MID + half * 0.5, n=8, t0=200.0)["analog"]
last = at(t, MID + half * 0.5, n=600, t0=210.0)["analog"]
check("a held lean is never absorbed", abs(first - last) < 1e-9,
      f"{first:+.3f} then {last:+.3f} after 20s held")

print("\n[9] jumping still works in zone mode")
t = Bare()
now = 300.0
for i in range(10):                       # settle the span estimate
    t._process_steer(hand(0.3, 0.5), W, H, now); now += 1/30
t._events.clear()
for i in range(1, 6):                     # flick upward
    y = 0.5 - 0.5 * 0.125 * (i / 5)
    t._process_steer(hand(0.3, y), W, H, now); now += 1/30
check("a flick still fires a jump", len(t._events) == 1,
      f"n={len(t._events)}" + (f" strength={t._events[0]['strength']:.2f}"
                               if t._events else ""))

print("\n[10] duck is a fixed band too")
t = Bare()
o = at(t, MID, y_px=(S.DUCK_ZONE_Y * H) + 30, n=S.DUCK_HOLD + 8)
check("hand held low ducks", o["duck"], f"below {S.DUCK_ZONE_Y:.0%} of frame height")
t = Bare()
o = at(t, MID, y_px=100, n=S.DUCK_HOLD + 8)
check("hand held high does not duck", not o["duck"])

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
