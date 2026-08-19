"""Headless validation of two_hand_spike.py logic + the MediaPipe Tasks API."""
import os, sys, math, time, types, threading
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import two_hand_spike as S

FAIL = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)

class LM:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z

def hand(cx, cy, span=0.12, pinch=0.9):
    """21 landmarks; wrist at (cx,cy), middle-MCP `span` above it (normalized)."""
    lms = [LM(cx, cy) for _ in range(21)]
    lms[S.WRIST] = LM(cx, cy)
    lms[S.MIDDLE_MCP] = LM(cx, cy - span)
    # thumb/index tips separated by pinch * span_px  (same axis scale as span)
    lms[S.INDEX_TIP] = LM(cx, cy - span * 2)
    lms[S.THUMB_TIP] = LM(cx + pinch * span * (480.0 / 640.0), cy - span * 2)
    return lms

class Bare(S.TwoHandTracker):
    """Tracker with no camera and no detector, for pure logic tests."""
    def _open_camera(self, preferred_id=0):
        self.cap, self.camera_ok, self.error_lines = None, False, []
    def _make_detector(self):
        return None

W, H = 640, 480

print("\n[1] module compiles / imports")
check("import two_hand_spike", True)

print("\n[2] role assignment")
t = Bare()
# Two hands, no memory -> leftmost steers.
a = t._assign_roles([hand(0.7, 0.5), hand(0.3, 0.5)], [], W, H, 1.0)
check("leftmost hand gets STEER", a[S.STEER][S.WRIST].x == 0.3 and a[S.FIRE][S.WRIST].x == 0.7)

# Hands drift toward each other and cross over: roles must follow the tracks.
t2 = Bare()
xs = [(0.30, 0.70), (0.40, 0.60), (0.48, 0.52), (0.52, 0.48), (0.60, 0.40), (0.70, 0.30)]
now = 10.0
swapped = False
for i, (ls, rs) in enumerate(xs):
    now += 1 / 30
    a = t2._assign_roles([hand(ls, 0.5), hand(rs, 0.5)], [], W, H, now)
    if abs(a[S.STEER][S.WRIST].x - ls) > 1e-9:
        swapped = True
check("roles survive hands crossing over", not swapped,
      f"steer ended at x={a[S.STEER][S.WRIST].x:.2f} (started 0.30)")

# Stale memory -> re-seed by screen side.
a = t2._assign_roles([hand(0.2, 0.5)], [], W, H, now + 5.0)
check("single hand, stale memory, left half -> STEER", S.STEER in a and S.FIRE not in a)
a = t2._assign_roles([hand(0.8, 0.5)], [], W, H, now + 11.0)
check("single hand, stale memory, right half -> FIRE", S.FIRE in a and S.STEER not in a)

# One hand disappears mid-session: the other keeps its role.
t3 = Bare()
now = 50.0
t3._assign_roles([hand(0.3, 0.5), hand(0.7, 0.5)], [], W, H, now)
now += 1 / 30
a = t3._assign_roles([hand(0.71, 0.5)], [], W, H, now)
check("surviving hand keeps FIRE when STEER leaves", S.FIRE in a and S.STEER not in a)

print("\n[3] analog steer (relative mode; see test_zone.py for the default)")
S.STEER_MODE = "relative"

def settle_neutral(tr, x_norm=0.5, y_norm=0.5, span=0.12, t0=100.0, dt=1/30):
    """Hold the hand still until the steering origin is learned.

    A still hand is not a sweep, so calibration waits SWEEP_TIME before falling
    back to the steady-hand method. Hold for longer than that.
    """
    out, now = None, t0
    steps = int((S.SWEEP_TIME + 0.6) / dt) + S.NEUTRAL_STABLE_N
    for i in range(steps):
        out = tr._process_steer(hand(x_norm, y_norm, span=span), W, H, now)
        now += dt
    return out, now

t4 = Bare()
one = t4._process_steer(hand(0.5, 0.5), W, H, 100.0)
check("a single pose does NOT set neutral", t4._neutral is None,
      "an origin from one frame is how it ended up in a corner")
check("no steering before the origin is known", abs(one["analog"]) < 1e-9)
out, now = settle_neutral(t4)
check("holding still sets neutral", t4._neutral is not None)
check("neutral pose reads 0.00", abs(out["analog"]) < 1e-9, f"analog={out['analog']:+.3f}")

span_px = 0.12 * H                       # hand span in pixels
full = S.STEER_FULL_LEAN * span_px       # px from neutral to full lean
vals = {}
for frac, label in ((0.10, "inside deadzone"), (0.40, "walk zone"), (0.95, "run zone")):
    now += 1.0
    t4._y_hist.clear()
    o = t4._process_steer(hand(0.5 + (frac * full) / W, 0.5), W, H, now)
    vals[label] = o
check("deadzone suppresses small lean", abs(vals["inside deadzone"]["analog"]) < 1e-9)
check("mid lean = WALK, not run",
      0.2 < vals["walk zone"]["analog"] < S.RUN_THRESHOLD and not vals["walk zone"]["run"],
      f"analog={vals['walk zone']['analog']:+.2f}")
check("deep lean = RUN", vals["run zone"]["run"] and vals["run zone"]["analog"] > S.RUN_THRESHOLD,
      f"analog={vals['run zone']['analog']:+.2f}")
now += 1.0
t4._y_hist.clear()
o = t4._process_steer(hand(0.5 - (0.95 * full) / W, 0.5), W, H, now)
check("leaning left is negative", o["analog"] < -S.RUN_THRESHOLD, f"analog={o['analog']:+.2f}")

# Distance invariance: the same physical lean at a different hand size gives the
# same analog -- as long as the frame can actually contain the required travel.
t5 = Bare()
big = 0.15            # 25% larger hand (a bit closer to the camera)
_, now = settle_neutral(t5, span=big, t0=200.0)
now += 1.0
t5._y_hist.clear()
o_big = t5._process_steer(hand(0.5 + (0.95 * S.STEER_FULL_LEAN * big * H) / W, 0.5, span=big),
                          W, H, now)
check("analog is hand-size invariant while the range fits the frame",
      abs(o_big["analog"] - vals["run zone"]["analog"]) < 0.02,
      f"larger hand={o_big['analog']:+.3f} vs baseline={vals['run zone']['analog']:+.3f}")

# ...and when the hand is so close that a full lean cannot fit, the range
# compresses instead of silently clipping one side.
t5b = Bare()
huge = 0.24           # span 115px -> a nominal lean needs 392px EACH way of 640
_, now = settle_neutral(t5b, span=huge, t0=300.0)
lean_l, lean_r = t5b._side_leans(t5b._neutral[0], huge * H, W)
nominal = S.STEER_FULL_LEAN * huge * H
check("a very close hand has its range compressed, not clipped",
      lean_l < nominal and lean_r < nominal and min(lean_l, lean_r) > 0,
      f"nominal {nominal:.0f}px, available {lean_l:.0f}/{lean_r:.0f}px "
      f"in a {W}px frame")
t5b._y_hist.clear()
o = t5b._process_steer(hand((t5b._neutral[0] - lean_l) / W, 0.5, span=huge), W, H, now + 1.0)
check("full left still saturates for a very close hand", o["analog"] < -0.95,
      f"analog={o['analog']:+.2f}")

print("\n[4] flick jump")
def flick(tracker, travel_spans, duration, start=0.5, dt=1/30, t0=300.0):
    """Move the wrist up `travel_spans` hand-spans over `duration` seconds."""
    span_norm = 0.12
    steps = max(2, int(round(duration / dt)))
    now, events = t0, []
    tracker._process_steer(hand(0.5, start), W, H, now)     # calibrate
    tracker._events.clear()
    for i in range(1, steps + 1):
        now += dt
        y = start - (travel_spans * span_norm) * (i / steps)
        tracker._process_steer(hand(0.5, y), W, H, now)
    return list(tracker._events), now

t6 = Bare()
evs, _ = flick(t6, travel_spans=2.0, duration=0.15)
check("fast flick fires exactly one jump", len(evs) == 1,
      f"n={len(evs)}" + (f" v={evs[0]['vel']:.1f} s={evs[0]['strength']:.2f}" if evs else ""))
if evs:
    check("strength within [0.45, 1.0]", 0.45 <= evs[0]["strength"] <= 1.0,
          f"s={evs[0]['strength']:.2f}")

t7 = Bare()
evs, _ = flick(t7, travel_spans=1.0, duration=0.7)
check("slow hand raise fires nothing", len(evs) == 0, f"n={len(evs)}")

# Both flicks must sit inside the real in-game range (FLICK_VEL_MIN..MAX), or they
# both saturate and the test proves nothing.
t8 = Bare()
evs_gentle, _ = flick(t8, travel_spans=0.45, duration=0.15)    # ~3 span/s
t9 = Bare()
evs_fast, _ = flick(t9, travel_spans=1.5, duration=0.15)       # ~10 span/s, saturates
if evs_gentle and evs_fast:
    check("faster flick -> higher jump",
          evs_fast[0]["strength"] > evs_gentle[0]["strength"] + 0.15,
          f"{evs_gentle[0]['strength']:.2f} (v{evs_gentle[0]['vel']:.0f}) -> "
          f"{evs_fast[0]['strength']:.2f} (v{evs_fast[0]['vel']:.0f})")
else:
    check("faster flick -> higher jump", False,
          f"gentle={len(evs_gentle)} fast={len(evs_fast)}")

# Refractory: a single continuous flick must not retrigger.
t10 = Bare()
evs, _ = flick(t10, travel_spans=6.0, duration=0.30)
check("one continuous flick != multiple jumps", len(evs) == 1, f"n={len(evs)}")

print("\n[5] duck vs flick recoil")
t11 = Bare()
_, now = settle_neutral(t11, t0=400.0)
low_y = 0.5 + (S.DUCK_OFFSET + 0.3) * 0.12
for i in range(S.DUCK_HOLD + 3):          # hold the hand low, barely moving
    now += 1 / 30
    o = t11._process_steer(hand(0.5, low_y), W, H, now)
check("sustained low hand = DUCK", o["duck"])
t12 = Bare()
evs, now2 = flick(t12, travel_spans=2.5, duration=0.15)      # jump up...
o = None
for i in range(3):                                            # ...then snap back down
    now2 += 1 / 30
    o = t12._process_steer(hand(0.5, 0.5 + 0.02 * i), W, H, now2)
check("flick recoil does not read as DUCK", o is not None and not o["duck"])

S.STEER_MODE = "zone"

print("\n[6] pinch fire (legacy posture mode, FIRE_MODE='pinch')")
_saved_mode = S.FIRE_MODE
S.FIRE_MODE = "pinch"
def settle(tr, gap, n, t0=500.0, dt=1/30):
    """Hold a resting gap for n poses so a baseline establishes."""
    for i in range(n):
        tr._process_fire(hand(0.5, 0.5, pinch=gap), W, H, t0 + i * dt)
    return t0 + n * dt

t13 = Bare()
now = settle(t13, 0.95, 20)
check("open hand not pinched", not t13._pinched)
t13._events.clear()
now += 1/30
t13._process_fire(hand(0.5, 0.5, pinch=0.2), W, H, now)
check("closing fires one shot", len(t13._events) == 1 and t13._pinched)
for i in range(10):
    now += 1/30
    t13._process_fire(hand(0.5, 0.5, pinch=0.2), W, H, now)
check("held pinch does not repeat", len(t13._events) == 1, f"n={len(t13._events)}")
now += 1/30
t13._process_fire(hand(0.5, 0.5, pinch=0.95), W, H, now)
check("release clears pinch state", not t13._pinched)
now += S.FIRE_COOLDOWN + 0.05
t13._process_fire(hand(0.5, 0.5, pinch=0.2), W, H, now)
check("re-pinch after cooldown fires again", len(t13._events) == 2)

t14 = Bare()
now = settle(t14, 0.95, 20)
t14._events.clear()
for i, gap in enumerate((0.2, 0.95, 0.2)):        # pinch, release, pinch within 50ms
    t14._process_fire(hand(0.5, 0.5, pinch=gap), W, H, now + 0.02 * (i + 1))
check("cooldown blocks sub-250ms double fire", len(t14._events) == 1,
      f"n={len(t14._events)}")

# The latch regression: a hand held closed long enough drags the baseline below
# PINCH_BASELINE_MIN. Releasing must still work, or the fire hand sticks shut.
t15 = Bare()
now = settle(t15, 0.95, 20)
now += 1/30
t15._process_fire(hand(0.5, 0.5, pinch=0.15), W, H, now)
check("pinch fires before the latch test", t15._pinched)
for i in range(120):                               # 4s closed -- baseline sags away
    now += 1/30
    t15._process_fire(hand(0.5, 0.5, pinch=0.15), W, H, now)
base = S.pctile(t15._gap_hist, S.PINCH_BASELINE_PCT)
check("baseline has sagged below the trust floor", base < S.PINCH_BASELINE_MIN,
      f"baseline={base:.2f}")
now += 1/30
t15._process_fire(hand(0.5, 0.5, pinch=0.75), W, H, now)
check("hand still releases with an untrusted baseline", not t15._pinched,
      "otherwise the fire hand latches shut (observed 40s in a replay)")

# Posture independence: the same gesture fires at very different absolute gaps.
tA = Bare(); nowA = settle(tA, 0.55, 90)           # rests semi-closed
tA._events.clear()
tA._process_fire(hand(0.5, 0.5, pinch=0.22), W, H, nowA + 1/30)
tB = Bare(); nowB = settle(tB, 1.10, 90)           # rests wide open
tB._events.clear()
tB._process_fire(hand(0.5, 0.5, pinch=0.45), W, H, nowB + 1/30)
check("fires for a hand that rests semi-closed", len(tA._events) == 1)
check("fires for a hand that rests wide open", len(tB._events) == 1)
check("same gesture, different absolute gaps (0.22 vs 0.45)",
      len(tA._events) == 1 and len(tB._events) == 1)

S.FIRE_MODE = _saved_mode

print("\n[6b] throw fire (default mode: a fast wrist move, like the jump flick)")
def throw(tr, travel_spans, duration, dt=1/30, t0=700.0, span=0.125, x0=0.5, y0=0.5):
    """Sweep the fire hand `travel_spans` hand-spans over `duration` seconds."""
    steps = max(2, int(round(duration / dt)))
    now = t0
    for i in range(8):                       # settle the span estimate first
        tr._process_fire(hand(x0, y0, span=span), W, H, now); now += dt
    tr._events.clear()
    for i in range(1, steps + 1):
        x = x0 + (travel_spans * span * (H / W)) * (i / steps)
        tr._process_fire(hand(x, y0, span=span), W, H, now); now += dt
    return list(tr._events), now

t20 = Bare()
evs, now = throw(t20, travel_spans=1.5, duration=0.15)
check("a fast wrist sweep throws exactly one fireball", len(evs) == 1,
      f"n={len(evs)}" + (f" v={evs[0]['vel']:.1f}" if evs else ""))
check("the event is a throw, not a pinch", evs and evs[0]["kind"] == "throw")

t21 = Bare()
evs, _ = throw(t21, travel_spans=0.8, duration=1.2)       # slow drift
check("slowly moving the hand throws nothing", len(evs) == 0, f"n={len(evs)}")

t22 = Bare()
evs, _ = throw(t22, travel_spans=4.0, duration=0.40)      # one long continuous sweep
check("one continuous sweep is not a burst of fireballs", len(evs) == 1, f"n={len(evs)}")

# A resting closed fist must not throw -- the exact failure that killed the pinch.
t23 = Bare()
now = 800.0
for i in range(90):
    t23._process_fire(hand(0.5, 0.5), W, H, now); now += 1/30
check("a still hand never throws", len(t23._events) == 0, f"n={len(t23._events)}")

# Throw is posture-independent: same sweep, hand open or closed.
res = {}
for label, pinch_amt in (("open hand", 0.95), ("closed fist", 0.12)):
    tr = Bare()
    def h2(cx, cy, span=0.125, _p=pinch_amt):
        l = hand(cx, cy, span=span)
        l[S.THUMB_TIP] = LM(cx + _p * span * 0.75, cy - span * 2)
        return l
    now = 900.0
    for i in range(8):
        tr._process_fire(h2(0.5, 0.5), W, H, now); now += 1/30
    tr._events.clear()
    for i in range(1, 6):
        tr._process_fire(h2(0.5 + (1.5 * 0.125 * (H/W)) * (i/5), 0.5), W, H, now)
        now += 1/30
    res[label] = len(tr._events)
check("throws with an open hand", res["open hand"] == 1, f"n={res['open hand']}")
check("throws with a closed fist too", res["closed fist"] == 1, f"n={res['closed fist']}",)
check("posture no longer matters", res["open hand"] == res["closed fist"] == 1)

print("\n[7] sandbox physics")
import pygame
pygame.init()
sb = S.Sandbox()
now = 700.0
sb.reset(now)                      # align the sandbox clock with the fake one
check("starts grounded", sb.grounded)
check("first flick = JUMP", sb.request_jump(0.8, now) == "JUMP")
sb.update(0, False, False, 0, now + 0.016)
check("airborne after jump", not sb.grounded)
check("second flick = DOUBLE JUMP", sb.request_jump(0.8, now + 0.02) == "DOUBLE JUMP")
check("third flick is not a triple jump",
      sb.request_jump(0.8, now + 0.04) in ("buffered", "ignored"))
# Land again -> jump budget resets.
for i in range(200):
    now += 1 / 60
    sb.update(0, False, False, 0, now)
    if sb.grounded:
        break
check("lands back on the floor", sb.grounded, f"y={sb.y:.0f} ground={S.GROUND_Y}")
check("jump budget resets on landing", sb.request_jump(0.8, now) == "JUMP")

# Walked off a ledge, flick arrives 50ms late. With assists it is a ground jump
# and the air jump is still in the bank; without, it silently eats the air jump.
sb2 = S.Sandbox(); sb2.reset(now); sb2.assists = True
sb2.grounded = False
sb2.jumps_used = 0
sb2.last_ground_t = now - 0.05
check("coyote time forgives a late flick",
      sb2.request_jump(0.9, now) == "JUMP" and sb2.jumps_used == 1)
sb3 = S.Sandbox(); sb3.reset(now); sb3.assists = False
sb3.grounded = False
sb3.jumps_used = 0
sb3.last_ground_t = now - 0.05
check("assists off: the same flick costs the air jump",
      sb3.request_jump(0.9, now) == "DOUBLE JUMP" and sb3.jumps_used == 2)

# A ground jump must close the coyote window, or repeat flicks fly forever.
sb2b = S.Sandbox(); sb2b.reset(now); sb2b.assists = True
sb2b.request_jump(1.0, now)
labels = [sb2b.request_jump(1.0, now + 0.03 * i) for i in range(1, 5)]
check("coyote cannot be milked into infinite jumps",
      labels.count("JUMP") == 0 and labels.count("DOUBLE JUMP") == 1, str(labels))

# Jump buffer: flick just before landing still jumps.
sb4 = S.Sandbox()
sb4.reset(now)
sb4.request_jump(1.0, now)
sb4.jumps_used = 2
t_air = now
for i in range(300):
    t_air += 1 / 60
    if sb4.vy > 0 and sb4.y > S.GROUND_Y - 20 and sb4.buffered_jump is None:
        sb4.request_jump(1.0, t_air)          # early flick, mid-fall
    was = sb4.grounded
    sb4.update(0, False, False, 0, t_air)
    if sb4.grounded and not was:
        break
check("buffered flick converts to a jump on landing", sb4.vy < 0, f"vy={sb4.vy:.1f}")

# Run is faster than walk; the gap is fatal.
sb5 = S.Sandbox()
x0 = sb5.x
for i in range(10):
    sb5.update(1.0, False, False, 0, now + i / 60)
walked = sb5.x - x0
sb6 = S.Sandbox()
x0 = sb6.x
for i in range(10):
    sb6.update(1.0, True, False, 0, now + i / 60)
check("run outruns walk", sb6.x - x0 > walked * 1.5, f"walk={walked:.0f}px run={sb6.x-x0:.0f}px")
sb7 = S.Sandbox()
sb7.reset(now)
sb7.x = float(S.GAP[0] + 40)
sb7.y = float(S.GROUND_Y)
falls0 = sb7.falls
tt = now
for i in range(400):
    tt += 1 / 60
    sb7.update(0, False, False, 0, tt)
    if sb7.falls > falls0:
        break
check("falling in the gap respawns", sb7.falls > falls0)
sb8 = S.Sandbox()
sb8.facing = -1
sb8.fire()
check("fireball inherits facing", sb8.fireballs[0][2] < 0)

print("\n[8] MediaPipe Tasks LIVE_STREAM API (real model, synthetic frames)")
model = next((p for p in S.MODEL_CANDIDATES if os.path.exists(p)), None)
check("hand_landmarker.task found", model is not None, str(model))
if model:
    got = {"n": 0, "err": None}
    done = threading.Event()
    class Live(S.TwoHandTracker):
        def _open_camera(self, preferred_id=0):
            self.cap, self.camera_ok, self.error_lines = None, False, []
    lt = Live()
    check("detector created with num_hands=2 + LIVE_STREAM", lt._detector is not None)
    if lt._detector is not None:
        import cv2, mediapipe as mp
        real_cb = lt._on_result
        def spy(result, out_img, ts):
            try:
                real_cb(result, out_img, ts)
                got["n"] += 1
            except Exception as e:
                got["err"] = e
            done.set()
        lt._on_result = spy
        lt._detector.close()
        lt._detector = lt._make_detector()   # rebuild so the callback is the spy
        frame = np.zeros((H, W, 3), np.uint8)
        cv2.circle(frame, (320, 240), 90, (180, 200, 210), -1)
        t_cap = time.perf_counter()
        with lt._lock:
            lt._pending[1] = (t_cap, frame)
        lt._detector.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))), 1)
        fired = done.wait(10.0)
        check("result callback fired", fired)
        check("callback ran without exception", got["err"] is None, repr(got["err"]))
        check("snapshot published", lt.snapshot["seq"] >= 1)
        check("latency sample recorded", len(lt.lat_infer) >= 1,
              f"{(lt.lat_infer[0]*1000):.1f}ms inference on a synthetic frame"
              if lt.lat_infer else "")
        lt.close()

print("\n[9] pygame UI construction (dummy video driver)")
try:
    sp = S.Spike()
    snap, evs = sp.tracker.read()
    sp._log("test line")
    sp.draw(snap, False)
    sp.summary()
    sp.tracker.close()
    check("Spike builds and renders one frame", True)
except Exception as e:
    import traceback; traceback.print_exc()
    check("Spike builds and renders one frame", False, repr(e))

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
