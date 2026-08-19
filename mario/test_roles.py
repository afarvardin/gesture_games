"""Regression tests for the two role-identity bugs found in the 129s session."""
import os, sys, math
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import two_hand_spike as S

FAIL=[]
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ")+name+(f"   {extra}" if extra else ""))
    if not cond: FAIL.append(name)

class LM:
    __slots__=("x","y","z")
    def __init__(s,x,y,z=0.0): s.x,s.y,s.z=x,y,z
def hand(cx,cy,span=0.12):
    l=[LM(cx,cy) for _ in range(21)]
    l[S.WRIST]=LM(cx,cy); l[S.MIDDLE_MCP]=LM(cx,cy-span)
    l[S.INDEX_TIP]=LM(cx,cy-span*2); l[S.THUMB_TIP]=LM(cx+0.9*span*0.75,cy-span*2)
    return l
class Bare(S.TwoHandTracker):
    def _open_camera(self,preferred_id=0): self.cap,self.camera_ok,self.error_lines=None,False,[]
    def _make_detector(self): return None
W,H=640,480
px=lambda x: x/W        # px -> normalised x

print("\n[A] the t=97.3s failure: one hand teleports, the other must keep its role")
t=Bare(); now=100.0
# Replay the real positions from the log, converted back to normalised coords.
t._assign_roles([hand(px(262.9),0.5), hand(px(189.4),0.5)], [], W, H, now)      # steer L? no:
# leftmost seeds STEER, so force the log's arrangement: steer=262.9, fire=189.4
t._role_pos={S.STEER:(262.9,240.0,now,-190.0,0.0), S.FIRE:(189.4,240.0,now,40.0,0.0)}
t._role_last={S.STEER:(262.9,240.0), S.FIRE:(189.4,240.0)}
now+=1/30
a=t._assign_roles([hand(px(191.1),0.5)], [], W, H, now)
check("lone hand near FIRE's track stays FIRE", S.FIRE in a and S.STEER not in a)
now+=1/30
a=t._assign_roles([hand(px(192.5),0.5), hand(px(541.2),0.5)], [], W, H, now)
fire_x=a[S.FIRE][S.WRIST].x*W if S.FIRE in a else -1
steer_x=a[S.STEER][S.WRIST].x*W if S.STEER in a else -1
check("FIRE keeps the hand that never moved", abs(fire_x-192.5)<1.0,
      f"fire_x={fire_x:.1f} (was 191.1)")
check("the far new hand becomes STEER", abs(steer_x-541.2)<1.0, f"steer_x={steer_x:.1f}")

print("\n[B] roles survive a dropout longer than ROLE_MEMORY")
t2=Bare(); now=200.0
# Player has hands crossed: STEER on the right, FIRE on the left.
t2._role_pos={S.STEER:(450.0,240.0,now,0.0,0.0), S.FIRE:(200.0,240.0,now,0.0,0.0)}
t2._role_last={S.STEER:(450.0,240.0), S.FIRE:(200.0,240.0)}
now += S.ROLE_MEMORY + 1.5                      # both hands rest out of frame
a=t2._assign_roles([hand(px(210.0),0.5), hand(px(440.0),0.5)], [], W, H, now)
sx=a[S.STEER][S.WRIST].x*W
check("returning hands keep their own roles, not screen side", abs(sx-440.0)<1.0,
      f"steer_x={sx:.1f} (last seen at 450; leftmost hand is at 210)")

print("\n[C] a collapsed hand span cannot manufacture a flick")
t3=Bare(); now=300.0
for i in range(12):                              # steady hand, healthy span
    now+=1/30
    t3._process_steer(hand(0.5,0.5,span=0.135), W, H, now)
t3._events.clear()
now+=1/30
o=t3._process_steer(hand(0.5,0.5,span=0.052), W, H, now)   # span collapses to 25px
check("span collapse produces no flick", len(t3._events)==0,
      f"n={len(t3._events)} vel={o['vel_up']:.1f} span={o['span']:.0f}px")
check("filtered span ignores the outlier", abs(o["span"]-0.135*H)<2.0,
      f"span={o['span']:.1f}px, raw was {0.052*H:.0f}px")

print("\n[D] genuine depth change still tracked")
t4=Bare(); now=400.0
o=None
for i in range(30):                              # hand moves steadily closer
    now+=1/30
    o=t4._process_steer(hand(0.5,0.5,span=0.10+0.004*i), W, H, now)
check("median follows a real depth change", o["span"] > 0.13*H,
      f"span={o['span']:.0f}px after ramping 48->104px")


# ---------------------------------------------------------------------------
# Handedness: the LEFT hand steers, the RIGHT hand throws.
# ---------------------------------------------------------------------------
class Cat:
    """Stand-in for a MediaPipe handedness Category."""
    def __init__(self, name, score=0.95):
        self.category_name, self.score = name, score

def hd(*names):
    """Handedness argument for _assign_roles, one entry per hand."""
    return [[Cat(n)] for n in names]

FAIL2 = FAIL

print("\n[E] left hand steers, right hand throws")
t = Bare()
now = 900.0
# Right hand on screen-left, left hand on screen-right: screen side would get this
# exactly backwards, which is the bug being fixed.
a = t._assign_roles([hand(px(180.0), 0.5), hand(px(460.0), 0.5)],
                    hd("Right", "Left"), W, H, now)
check("STEER goes to the left hand, not the left side of frame",
      abs(a[S.STEER][S.WRIST].x * W - 460.0) < 1.0,
      f"steer_x={a[S.STEER][S.WRIST].x*W:.0f} (left hand was at 460, right at 180)")
check("FIRE goes to the right hand", abs(a[S.FIRE][S.WRIST].x * W - 180.0) < 1.0)

print("\n[F] a lone hand takes the role its handedness says")
t = Bare()
a = t._assign_roles([hand(px(120.0), 0.5)], hd("Right"), W, H, 1000.0)
check("a right hand alone on the left of frame is FIRE, not STEER",
      S.FIRE in a and S.STEER not in a)
t = Bare()
a = t._assign_roles([hand(px(520.0), 0.5)], hd("Left"), W, H, 1100.0)
check("a left hand alone on the right of frame is STEER",
      S.STEER in a and S.FIRE not in a)

print("\n[G] one flickered label does not swap the roles")
t = Bare()
now = 1200.0
for i in range(20):                      # settle correctly: left hand steers
    now += 1/30
    a = t._assign_roles([hand(px(200.0), 0.5), hand(px(440.0), 0.5)],
                        hd("Left", "Right"), W, H, now)
steer_x = a[S.STEER][S.WRIST].x * W
check("settles with the left hand steering", abs(steer_x - 200.0) < 1.0)
for i in range(2):                       # two bad frames
    now += 1/30
    a = t._assign_roles([hand(px(200.0), 0.5), hand(px(440.0), 0.5)],
                        hd("Right", "Left"), W, H, now)
check("two mislabelled frames are ignored",
      abs(a[S.STEER][S.WRIST].x * W - 200.0) < 1.0,
      f"steer_x={a[S.STEER][S.WRIST].x*W:.0f} -- acting on one frame is the bug")

print("\n[H] a sustained disagreement DOES swap the roles")
for i in range(S.HANDEDNESS_SWAP_VOTES + 3):
    now += 1/30
    a = t._assign_roles([hand(px(200.0), 0.5), hand(px(440.0), 0.5)],
                        hd("Right", "Left"), W, H, now)
check("a persistent relabel moves the roles across",
      abs(a[S.STEER][S.WRIST].x * W - 440.0) < 1.0,
      f"steer_x={a[S.STEER][S.WRIST].x*W:.0f} (now the left hand is the one at 440)")

print("\n[I] low-confidence labels are ignored")
t = Bare()
now = 1400.0
for i in range(20):
    now += 1/30
    a = t._assign_roles([hand(px(200.0), 0.5), hand(px(440.0), 0.5)],
                        hd("Left", "Right"), W, H, now)
for i in range(20):
    now += 1/30
    unsure = [[Cat("Right", 0.35)], [Cat("Left", 0.35)]]
    a = t._assign_roles([hand(px(200.0), 0.5), hand(px(440.0), 0.5)],
                        unsure, W, H, now)
check("a label below HANDEDNESS_MIN_SCORE cannot move a role",
      abs(a[S.STEER][S.WRIST].x * W - 200.0) < 1.0,
      f"steer_x={a[S.STEER][S.WRIST].x*W:.0f}")

print("\n[J] polarity can be flipped for a reversed camera")
t = Bare()
now = 1500.0
for i in range(20):
    now += 1/30
    a = t._assign_roles([hand(px(200.0), 0.5), hand(px(440.0), 0.5)],
                        hd("Left", "Right"), W, H, now)
before = a[S.STEER][S.WRIST].x * W
t.toggle_hand_polarity()
for i in range(S.HANDEDNESS_SWAP_VOTES + 3):
    now += 1/30
    a = t._assign_roles([hand(px(200.0), 0.5), hand(px(440.0), 0.5)],
                        hd("Left", "Right"), W, H, now)
check("toggling polarity swaps which hand steers",
      abs(a[S.STEER][S.WRIST].x * W - 440.0) < 1.0,
      f"{before:.0f} -> {a[S.STEER][S.WRIST].x*W:.0f}")

print("\n[K] crossing hands still tracked, roles still correct")
t = Bare()
now = 1600.0
xs = [(200, 440), (260, 380), (310, 330), (330, 310), (380, 260), (440, 200)]
ok = True
for lx, rx in xs:                        # the left hand sweeps right, right sweeps left
    now += 1/30
    a = t._assign_roles([hand(px(lx), 0.5), hand(px(rx), 0.5)],
                        hd("Left", "Right"), W, H, now)
    if abs(a[S.STEER][S.WRIST].x * W - lx) > 1.0:
        ok = False
check("roles hold through a crossing", ok,
      "handedness decides the role, prediction does the tracking")

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
