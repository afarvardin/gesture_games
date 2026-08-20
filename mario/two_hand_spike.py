"""Two-hand control spike for a gesture platformer.

This is a measurement tool, not a game. It answers the three questions that
decide whether a hand-controlled Mario is worth building:

  1. Can two hands hold distinct roles reliably? (role assignment + hysteresis)
  2. Does a velocity-triggered flick beat posture classification for jumps?
  3. How much end-to-end latency is there, really?

Design notes that matter if you tune this:

* Everything is measured in HAND SPANS (wrist -> middle-finger MCP), not pixels,
  so the thresholds hold whether it's a kid's hand at 40cm or an adult's at 80cm.
* The LEFT hand steers and the RIGHT hand throws, decided by MediaPipe handedness
  and held steady by a majority vote over ~0.4s (a single frame's label flickers,
  and snapping to it is what made the roles trade places mid-game). Position
  prediction still does the frame-to-frame tracking underneath, because it is what
  survives two hands crossing over each other; handedness only decides which
  track owns which role.
* Jump and fire are EDGE-triggered on velocity / pinch transitions with a
  refractory window. There is deliberately no majority-vote debounce on them --
  that debounce is what costs Pac-Man ~130ms, and a platformer cannot pay it.
* Gesture detection runs in the MediaPipe callback (pose rate, has timestamps)
  and pushes timestamped events to a deque. The game loop drains it at 60fps.
  That keeps event timing honest and the render loop cheap.

Controls:
  C      recalibrate steering neutral to where the steer hand is right now
  H      swap left/right hand detection, if your camera reports it reversed
  T      toggle the latency-forgiveness assists (coyote time + jump buffer)
  K      toggle keyboard control, to A/B the feel against arrow keys + space
  R      reset the sandbox player
  Esc/Q  quit (prints a latency summary)

Usage:
  python two_hand_spike.py [--log [PATH]] [--camera N]

  --steer-zone LO HI sets the steering band, as fractions of the frame width.

  --swap-hands flips left/right hand detection for a camera that reports it
  reversed; the H key does the same thing while running.

  --log writes one CSV row per pose and per event, so the thresholds can be
  tuned against real motion instead of recollection. Analyse it with
  analyse_log.py.
"""

import argparse
import collections
import csv
import itertools
import math
import os
import statistics
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ==========================================
# LAYOUT
# ==========================================
WIDTH, HEIGHT = 1280, 720
FPS = 60
CAM_W, CAM_H = 640, 480

PREVIEW_RECT = pygame.Rect(20, 20, 640, 480)
PANEL_RECT = pygame.Rect(676, 20, 584, 480)
SANDBOX_RECT = pygame.Rect(20, 516, 1240, 184)

COLOR_BG = (12, 14, 22)
COLOR_PANEL = (20, 24, 36)
COLOR_TEXT = (226, 230, 240)
COLOR_DIM = (120, 130, 150)
COLOR_STEER = (0, 235, 255)
COLOR_FIRE = (255, 120, 40)
COLOR_OK = (60, 230, 130)
COLOR_WARN = (255, 210, 60)
COLOR_BAD = (255, 70, 90)

# ==========================================
# GESTURE TUNING
# Distances are in hand-spans; velocities in hand-spans/second.
#
# These are no longer guesses: every value marked [log] was set from the
# distributions in a 129s two-hand session (see mario/README.md). The originals
# were derived from synthetic motion and were mostly too demanding.
# ==========================================
SPAN_MEDIAN_N = 15          # poses in the hand-span median filter (~0.5s at 30fps)
SPAN_MIN_PX = 22            # [log] a span this small is a failed detection, not a
                            # distant hand (observed: 6px against a 58px median)
SPAN_OUTLIER = 0.55         # [log] reject a pose whose span is below this fraction
                            # of the established median
SPAN_REJECT_LIMIT = 12      # consecutive rejects before re-learning the span
SPAN_WARMUP = 5             # [log] poses to distrust after a hand is re-acquired
                            # The first detection of a returning hand carries a
                            # bad span (25px vs a normal 65px), and since span is
                            # the velocity denominator, the second pose then shows
                            # a 300 span/s "flick" from a hand that barely moved.
                            # No jump fires until the span estimate settles.

# How the steering hand's position becomes an axis.
#
# "zone" maps a FIXED band of the camera frame onto the full range: the left edge
# of the band is full-left, the right edge is full-right, and the middle is
# neutral. Nothing is calibrated and nothing drifts, so the controls are in the
# same place every session and the player can see exactly where they are.
#
# "relative" is the older sweep-calibrated scheme: it measures the player's own
# centre and reach. In principle nicer, in practice it kept parking the band
# wherever the hand happened to be resting -- one logged session ended up with the
# band spanning 61%..91% of the frame, so full speed sat at the very edge of shot
# and everything short of that was a slow walk. Fixed zones do not have that
# failure mode.
STEER_MODE = "zone"         # "zone" | "relative"
STEER_ZONE = (0.02, 0.60)   # fractions of frame width: full-left .. full-right
DUCK_ZONE_Y = 0.80          # hand below this fraction of frame height = duck

# Steering origin ("neutral"). Only used by STEER_MODE == "relative". This is the
# single most important thing to get
# right: a badly placed origin makes one direction unreachable and the axis read
# hard-over at rest, which is far more damaging than any threshold being off.
NEUTRAL_STABLE_N = 8        # [log] steady poses required before neutral is set
NEUTRAL_STABLE_SPREAD = 0.8 # hand-spans of allowed jitter across those poses
NEUTRAL_EDGE_MARGIN = 0.25  # [log] fraction of the frame at each edge where a hand
                            # is NOT a valid origin -- a hand entering from a
                            # corner is exactly what used to anchor neutral there.
                            # Also keeps the two sides within ~1.3x gain of each
                            # other for any origin that is accepted.
STEER_SIDE_MIN = 0.60       # [log] the symmetric lean is never smaller than this
                            # fraction of nominal, so a hand resting close to an
                            # edge cannot make the whole axis twitchy
STEER_EDGE_FRAC = 0.10      # [log] fraction of the frame at each edge treated as
                            # unusable. It was 8px, i.e. the literal pixel edge --
                            # so with a centre 212px into a 640px frame, "full
                            # left" sat at x=8 and the hand was lost before it ever
                            # got there (a logged session reached -0.73 at best).
                            # Deflection has to max out somewhere a hand can
                            # actually be tracked.
# Preferred calibration: a short side-to-side sweep gives the centre AND the gain.
SWEEP_TIME = 3.0            # s of sweeping collected before deciding
SWEEP_MIN_WIDTH = 2.0       # hand-spans of travel before it counts as a sweep
SWEEP_MIN_REVERSALS = 2     # direction changes required -- a hand entering the
                            # frame travels one way, a real sweep goes back and forth
SWEEP_LEAN_MIN = 1.2        # clamp on the learned half-reach, in hand-spans
SWEEP_LEAN_MAX = 4.5
NEUTRAL_RECENTRE_TAU = 4.0  # s time constant of the gentle drift correction
NEUTRAL_RECENTRE_MAX_ANALOG = 0.35   # only drift while near centre, so a held
                            # lean is never absorbed. A wrong origin no longer
                            # needs to self-heal, because the sweep measures it.
STILL_TIME = 1.5            # s the hand must hold position to count as at rest
STILL_RADIUS = 0.40         # hand-spans of jitter allowed while "still"
NEUTRAL_RECAL_AFTER = 3.0   # s of absence after which the origin is re-learned,
                            # since the player has probably repositioned

STEER_FULL_LEAN = 3.4       # [log] hand-spans from neutral to full deflection
                            # was 2.2, which saturated 39% of the time -- the
                            # analog axis was behaving like an on/off switch
STEER_DEADZONE = 0.15       # fraction of full lean treated as centered
RUN_THRESHOLD = 0.60        # |analog| above this is a run, below is a walk

FLICK_VEL_MIN = 1.8         # [log] upward wrist speed that counts as a jump flick
                            # Walked down 4.0 -> 3.0 -> 2.5 -> 1.8 as each log
                            # showed attempts falling just short. Flicks made while
                            # actually playing are far gentler than flicks made
                            # while demonstrating: an 87s play session peaked at
                            # 2.6-5.7 span/s, against 4-14 in the test sessions.
                            # Safe this low only because FLICK_MIN_TRAVEL guards
                            # against jitter.
FLICK_VEL_MAX = 6.0         # [log] speed that maps to a full-height jump
                            # 22 -> 10 -> 6. In real play no jump reached even 0.7
                            # strength, because in-game flicks top out near
                            # 5.7 span/s. At 6.0 the same motion spans the range.
FLICK_MIN_TRAVEL = 0.20     # [log] hand-spans of upward travel a flick must cover,
                            # on top of the speed test -- jitter clears a speed
                            # threshold without the hand actually going anywhere.
                            # 0.30 sat at the median observed peak travel and cut
                            # real flicks; 0.20 keeps the rate stable across both
                            # sessions (19 and 21 flicks/min) while still
                            # excluding peaks that go nowhere.
FLICK_REARM = 2.0           # speed the hand must drop below before it can fire again
FLICK_REFRACTORY = 0.22     # s between accepted flicks
FLICK_WINDOW = 0.12         # s of history used for the velocity estimate

DUCK_OFFSET = 0.90          # hand-spans below neutral to read as a duck
DUCK_HOLD = 4               # consecutive poses required (rejects flick recoil)

# How the fire hand triggers. "throw" is the default and the recommendation from
# session two: a fast wrist movement, detected exactly like the jump flick.
# "pinch" keeps the older posture test available for comparison.
FIRE_MODE = "throw"         # "throw" | "pinch"
THROW_VEL_MIN = 2.2         # [log] wrist speed (spans/s, any direction) to throw
                            # 12 of 22 logged throws only just cleared 3.0, so
                            # gentler intended throws were being swallowed
THROW_MIN_TRAVEL = 0.25     # hand-spans the throw must actually cover
THROW_REARM = 2.0           # speed the hand must drop below before re-arming

# Pinch is measured RELATIVE to the player's own open-hand baseline, because the
# absolute gap/span of a resting hand varies far too much between sessions.
PINCH_BASELINE_N = 90       # poses (~3s) used to estimate the open-hand gap
PINCH_BASELINE_PCT = 0.80   # percentile of recent gaps taken as "open"
PINCH_BASELINE_MIN = 0.35   # below this the hand is just closed; do not fire
PINCH_DROP = 0.45           # [log] fire when gap falls below this fraction of open
PINCH_RELEASE = 0.80        # [log] hysteresis, as a fraction of open
PINCH_RELEASE_ABS = 0.60    # absolute release used when the baseline is not
                            # trustworthy -- a release path must always exist,
                            # or the fire hand can latch shut (observed: 40s)
FIRE_COOLDOWN = 0.25        # s between fireballs

# Which physical hand gets which role. The left hand steers, the right hand
# throws, taken from MediaPipe's handedness label rather than from screen side.
#
# Two things make this less trivial than it sounds. MediaPipe reports handedness
# as if the input were a mirror (selfie view); this file flips the frame before
# inference, which produces exactly that, so the labels are already the player's
# real hands -- but if a setup turns out reversed, SWAP_HANDEDNESS (or the H key)
# flips it without touching anything else. And a single frame's label does flicker,
# so a role only moves once a clear majority of recent frames agrees. Snapping on
# one frame is what makes hands "sometimes swap" mid-game.
ROLE_BY_HANDEDNESS = True   # False falls back to pure position tracking
SWAP_HANDEDNESS = False     # flip if left/right come out backwards on your camera
HANDEDNESS_MIN_SCORE = 0.6  # ignore labels the model is not confident about
HANDEDNESS_VOTES = 12       # frames of history kept per role (~0.4s at 30fps)
HANDEDNESS_SWAP_VOTES = 4   # agreeing votes needed before a role changes hands.
                            # Only reached when a label is missing or unsure -- when
                            # both hands are clearly labelled the roles are set
                            # directly, so this no longer needs to be slow.

ROLE_MEMORY = 1.2           # [log] s a role's live track stays valid for matching
                            # was 0.5, and 9 dropouts in one session outlasted it
ROLE_MATCH_RADIUS = 0.30    # fraction of frame width allowed between prediction and hand
ROLE_VEL_BLEND = 0.6        # smoothing on per-role velocity used to predict positions
ROLE_PREDICT_MAX = 0.10     # s -- cap on extrapolation, so a stale track can't fly off

# Landmarks
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (9, 10), (10, 11), (11, 12),             # middle
    (13, 14), (14, 15), (15, 16),            # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
)

STEER, FIRE = "STEER", "FIRE"

# One flat CSV, one row per pose and per event, distinguished by `kind`. Sparse
# columns beat two files here -- the whole point is to line events up against the
# pose stream they came out of.
LOG_FIELDS = (
    "kind", "t", "seq", "lat_ms", "steer_seen", "fire_seen", "analog", "run",
    "duck", "vel_up", "pinch_gap", "pinched", "span", "steer_x", "steer_y",
    "fire_x", "fire_y", "neutral_x", "reach_left", "reach_right",
    "vel", "strength", "outcome", "note",
)

MODEL_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fruit_ninja",
                 "hand_landmarker.task"),
)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def pctile(values, q):
    """Percentile of an unsorted iterable; returns 0.0 when empty."""
    vals = sorted(values)
    if not vals:
        return 0.0
    return vals[clamp(int(round(q * (len(vals) - 1))), 0, len(vals) - 1)]


# ==========================================
# TRACKER
# ==========================================
def tuning_meta():
    """The thresholds in force, stamped into every log.

    Shared so the spike and the game write identical metadata. Without it the
    analyser silently falls back to its own defaults and then gives advice
    about constants that are not the ones being used.
    """
    return ";".join(f"{k}={v}" for k, v in (
                ("STEER_MODE", STEER_MODE),
                ("STEER_ZONE", f"{STEER_ZONE[0]}-{STEER_ZONE[1]}"),
                ("DUCK_ZONE_Y", DUCK_ZONE_Y),
                ("STEER_FULL_LEAN", STEER_FULL_LEAN), ("STEER_DEADZONE", STEER_DEADZONE),
                ("STEER_SIDE_MIN", STEER_SIDE_MIN), ("STILL_TIME", STILL_TIME),
                ("SWEEP_TIME", SWEEP_TIME), ("SWEEP_MIN_WIDTH", SWEEP_MIN_WIDTH),
                ("NEUTRAL_EDGE_MARGIN", NEUTRAL_EDGE_MARGIN),
                ("RUN_THRESHOLD", RUN_THRESHOLD), ("FLICK_VEL_MIN", FLICK_VEL_MIN),
                ("FLICK_VEL_MAX", FLICK_VEL_MAX), ("FLICK_REARM", FLICK_REARM),
                ("FLICK_REFRACTORY", FLICK_REFRACTORY), ("FLICK_WINDOW", FLICK_WINDOW),
                ("DUCK_OFFSET", DUCK_OFFSET), ("DUCK_HOLD", DUCK_HOLD),
                ("FIRE_MODE", FIRE_MODE), ("THROW_VEL_MIN", THROW_VEL_MIN),
                ("THROW_MIN_TRAVEL", THROW_MIN_TRAVEL),
                ("PINCH_DROP", PINCH_DROP), ("PINCH_RELEASE", PINCH_RELEASE),
                ("PINCH_BASELINE_N", PINCH_BASELINE_N),
                ("PINCH_BASELINE_PCT", PINCH_BASELINE_PCT),
                ("PINCH_BASELINE_MIN", PINCH_BASELINE_MIN),
                ("FLICK_MIN_TRAVEL", FLICK_MIN_TRAVEL),
                ("SPAN_MIN_PX", SPAN_MIN_PX), ("SPAN_OUTLIER", SPAN_OUTLIER),
                ("STEER_EDGE_FRAC", STEER_EDGE_FRAC),
                ("ROLE_BY_HANDEDNESS", ROLE_BY_HANDEDNESS),
                ("SWAP_HANDEDNESS", SWAP_HANDEDNESS),
                ("FIRE_COOLDOWN", FIRE_COOLDOWN),
                ("ROLE_MEMORY", ROLE_MEMORY),
                ("ROLE_MATCH_RADIUS", ROLE_MATCH_RADIUS),
                ("ROLE_VEL_BLEND", ROLE_VEL_BLEND),
                ("SPAN_MEDIAN_N", SPAN_MEDIAN_N), ("SPAN_WARMUP", SPAN_WARMUP),
                ("CAM", f"{CAM_W}x{CAM_H}"),
            ))


class TwoHandTracker:
    """Threaded camera + MediaPipe Tasks HandLandmarker in LIVE_STREAM mode.

    The capture thread reads frames and fires them at the detector
    asynchronously; the detector's own thread runs the gesture state machine and
    publishes a snapshot plus an event queue. Nothing blocks the render loop.
    """

    def __init__(self, camera_id=0, logging_on=False):
        self.cap = None
        self.camera_ok = False
        self.error_lines = []

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending = {}          # detect timestamp_ms -> (t_capture, bgr frame)
        self._events = collections.deque(maxlen=128)
        self._seq = 0
        self._t0 = time.perf_counter()
        self.t0 = self._t0          # shared clock origin for log timestamps
        self._last_ts_ms = -1

        # Pose rows queue here and are written by the render loop, so all file
        # I/O stays on one thread. Unbounded: dropping rows would bias the stats.
        self.logging_on = logging_on
        self._log_rows = collections.deque()

        # Published snapshot, read by the render loop.
        self.snapshot = {
            "seq": 0,
            "frame": None,
            "t_capture": None,
            "steer_seen": False,
            "fire_seen": False,
            "analog": 0.0,
            "run": False,
            "duck": False,
            "pinched": False,
            "pinch_gap": None,
            "vel_up": 0.0,
            "calibrated": False,
        }

        # Gesture state (touched only by the detector callback).
        self._neutral = None        # (x, y) in px, steering origin
        self._neutral_buf = collections.deque(maxlen=NEUTRAL_STABLE_N)
        self._neutral_clamped = False
        self._last_steer_t = None
        self._steer_gone_since = None
        self._still_hist = collections.deque()
        self._at_rest = False
        self._sweep_start = None
        self._sweep_pts = []
        self._sweep_lean = None      # px of half-reach measured from the sweep
        self._sweep_gave_up = False
        self._y_hist = collections.deque(maxlen=16)   # (t, wrist_y / span)
        self._last_flick_t = -1.0
        self._flick_armed = True
        self._duck_run = 0
        self._pinched = False
        self._last_fire_t = -1.0
        self._gap_hist = collections.deque(maxlen=PINCH_BASELINE_N)
        self._fire_hist = collections.deque(maxlen=16)   # (t, wx, wy) of the fire hand
        self._throw_armed = True
        self._fire_vel = 0.0
        self.swap_handedness = SWAP_HANDEDNESS
        self._hand_votes = {STEER: collections.deque(maxlen=HANDEDNESS_VOTES),
                            FIRE: collections.deque(maxlen=HANDEDNESS_VOTES)}
        self._role_side = {}        # role -> "left"/"right" last seen, for the UI
        self._role_pos = {}         # role -> (x_px, y_px, t, vx, vy), expires
        self._role_last = {}        # role -> (x_px, y_px), never expires
        self._span_hist = {STEER: collections.deque(maxlen=SPAN_MEDIAN_N),
                           FIRE: collections.deque(maxlen=SPAN_MEDIAN_N)}
        self._span_rejects = {STEER: 0, FIRE: 0}
        self._request_calibrate = False

        # Latency instrumentation.
        self.lat_infer = collections.deque(maxlen=240)   # capture -> landmarks
        self.tracker_frame_times = collections.deque(maxlen=120)
        self.dropped = 0

        self._open_camera(camera_id)
        self._detector = self._make_detector()
        self._thread = None
        if self.camera_ok and self._detector is not None:
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    # ---------- setup ----------
    def _open_camera(self, preferred_id=0):
        """Open the webcam. On macOS, Camera access is granted to the app that
        launches Python (Terminal / iTerm / VS Code), not to Python itself; when
        it is missing, VideoCapture opens fine and read() just returns False
        forever with no prompt. So we confirm frames actually flow.
        """
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin" else [cv2.CAP_ANY]
        candidates = [preferred_id] + [i for i in (0, 1, 2) if i != preferred_id]

        for backend in backends:
            for cam_id in candidates:
                cap = cv2.VideoCapture(cam_id, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                # A 1-frame buffer keeps stale frames out of the latency path.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(30):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.camera_ok = True
                        print(f"[Camera] Opened device {cam_id} (backend={backend}).")
                        return
                cap.release()

        self.error_lines = [
            "CAMERA BLOCKED",
            "Grant Camera access to your terminal/IDE:",
            "System Settings > Privacy & Security > Camera",
            "then fully QUIT & reopen that app and relaunch.",
        ]
        print("[Camera] ERROR: no device delivered frames (usually a macOS permission block).")

    def _make_detector(self):
        model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
        if model_path is None:
            self.error_lines = [
                "MODEL NOT FOUND",
                "hand_landmarker.task is missing.",
                "Expected next to this file or in ../fruit_ninja/.",
            ]
            print("[Model] ERROR: hand_landmarker.task not found in", MODEL_CANDIDATES)
            return None

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print(f"[Model] {model_path} (LIVE_STREAM, num_hands=2)")
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def calibrate(self):
        self._request_calibrate = True

    # ---------- capture thread ----------
    def _capture_loop(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            t_capture = time.perf_counter()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)   # mirror, so moving right moves right
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ts_ms = int((t_capture - self._t0) * 1000)
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms

            with self._lock:
                self._pending[ts_ms] = (t_capture, frame)
                # Bound the map in case the detector ever drops a callback.
                if len(self._pending) > 8:
                    for stale in sorted(self._pending)[:-4]:
                        del self._pending[stale]
                        self.dropped += 1

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(rgb))
            self._detector.detect_async(mp_image, ts_ms)

    # ---------- detector thread ----------
    def _on_result(self, result, output_image, timestamp_ms):
        t_result = time.perf_counter()
        with self._lock:
            entry = self._pending.pop(timestamp_ms, None)
        if entry is None:
            return
        t_capture, frame = entry
        h, w = frame.shape[:2]

        hands = self._assign_roles(result.hand_landmarks or [],
                                   result.handedness or [], w, h, t_capture)

        if self._request_calibrate:
            # Re-learn through the same stability + clamp path as a cold start,
            # rather than snapping to wherever the hand is this instant.
            self._neutral = None
            self._neutral_buf.clear()
            self._sweep_start = None
            self._sweep_pts = []
            self._sweep_lean = None
            self._sweep_gave_up = False
            self._request_calibrate = False

        steer_out = {"analog": 0.0, "run": False, "duck": False, "vel_up": 0.0,
                     "span": None, "reach_left": None, "reach_right": None}
        if STEER in hands:
            if (self._steer_gone_since is not None
                    and t_capture - self._steer_gone_since > NEUTRAL_RECAL_AFTER):
                self._neutral = None      # gone long enough that the rest pose moved
                self._neutral_buf.clear()
            self._steer_gone_since = None
            steer_out = self._process_steer(hands[STEER], w, h, t_capture)
        else:
            if self._steer_gone_since is None:
                self._steer_gone_since = t_capture
            self._y_hist.clear()
            self._duck_run = 0
            self._flick_armed = True
            self._span_hist[STEER].clear()   # depth may differ when it returns
            self._span_rejects[STEER] = 0
            self._still_hist.clear()
            self._at_rest = False
            self._sweep_start = None
            self._sweep_pts = []
            self._sweep_gave_up = False

        pinch_gap = None
        if FIRE in hands:
            pinch_gap = self._process_fire(hands[FIRE], w, h, t_capture)
        else:
            self._pinched = False
            self._span_hist[FIRE].clear()
            self._span_rejects[FIRE] = 0
            self._fire_hist.clear()
            self._throw_armed = True
            self._fire_vel = 0.0

        self._draw_overlay(frame, hands, steer_out, pinch_gap)

        self.lat_infer.append(t_result - t_capture)
        self.tracker_frame_times.append(t_result)

        if self.logging_on:
            def rp(role, axis):
                p = self._role_pos.get(role)
                return round(p[axis], 1) if p is not None and role in hands else ""
            self._log_rows.append({
                "kind": "pose",
                "t": round(t_capture - self._t0, 4),
                "seq": self._seq + 1,
                "lat_ms": round((t_result - t_capture) * 1000, 2),
                "steer_seen": int(STEER in hands),
                "fire_seen": int(FIRE in hands),
                "analog": round(steer_out["analog"], 4),
                "run": int(steer_out["run"]),
                "duck": int(steer_out["duck"]),
                "vel_up": round(steer_out["vel_up"], 3),
                "pinch_gap": round(pinch_gap, 4) if pinch_gap is not None else "",
                "pinched": int(self._pinched),
                "vel": round(self._fire_vel, 3) if FIRE_MODE == "throw" else "",
                "span": round(steer_out["span"], 1) if steer_out["span"] else "",
                "steer_x": rp(STEER, 0), "steer_y": rp(STEER, 1),
                "fire_x": rp(FIRE, 0), "fire_y": rp(FIRE, 1),
                "neutral_x": round(self._neutral[0], 1) if self._neutral else "",
                "reach_left": (round(steer_out["reach_left"], 3)
                               if steer_out.get("reach_left") is not None else ""),
                "reach_right": (round(steer_out["reach_right"], 3)
                                if steer_out.get("reach_right") is not None else ""),
            })

        with self._lock:
            self._seq += 1
            self.snapshot = {
                "seq": self._seq,
                "frame": frame,
                "t_capture": t_capture,
                "steer_seen": STEER in hands,
                "fire_seen": FIRE in hands,
                "analog": steer_out["analog"],
                "run": steer_out["run"],
                "duck": steer_out["duck"],
                "pinched": self._pinched,
                "pinch_gap": pinch_gap,
                "vel_up": steer_out["vel_up"],
                "calibrated": STEER_MODE == "zone" or self._neutral is not None,
                "neutral_clamped": self._neutral_clamped,
                "neutral_x": self._neutral[0] if self._neutral else None,
                "reach_left": steer_out.get("reach_left"),
                "reach_right": steer_out.get("reach_right"),
                "at_rest": self._at_rest,
                "steer_side": self._role_side.get(STEER),
                "fire_side": self._role_side.get(FIRE),
                "swap_handedness": self.swap_handedness,
            }

    def _real_side(self, cat):
        """Which physical hand a MediaPipe handedness category refers to.

        Returns ("left"|"right"|None, score). MediaPipe labels as if the image
        were a mirror, and this file flips the frame before inference, so the label
        already names the player's real hand. swap_handedness inverts it for setups
        where that does not hold.
        """
        if not cat:
            return None, 0.0
        c = cat[0]
        side = (getattr(c, "category_name", None) or "").strip().lower()
        score = float(getattr(c, "score", 0.0) or 0.0)
        if side not in ("left", "right"):
            return None, 0.0
        if self.swap_handedness:
            side = "right" if side == "left" else "left"
        return side, score

    def _vote_sides(self, assigned, points, sides):
        """Record which physical hand currently holds each role."""
        for role in (STEER, FIRE):
            lm = assigned.get(role)
            if lm is None:
                self._hand_votes[role].clear()   # stale votes must not outlive a hand
                self._role_side.pop(role, None)
                continue
            idx = next((i for i, p in enumerate(points) if p[2] is lm), None)
            if idx is None:
                continue
            side, score = sides[idx]
            if side and score >= HANDEDNESS_MIN_SCORE:
                self._hand_votes[role].append(side)
                self._role_side[role] = side

    def _roles_are_swapped(self):
        """Has a clear majority of recent frames said the roles are the wrong way
        round? One frame is never enough -- labels flicker, and acting on a single
        frame is what made the hands trade places mid-game.
        """
        steer_votes = self._hand_votes[STEER]
        fire_votes = self._hand_votes[FIRE]
        steer_is_right = sum(1 for v in steer_votes if v == "right")
        fire_is_left = sum(1 for v in fire_votes if v == "left")
        if steer_is_right >= HANDEDNESS_SWAP_VOTES:
            # If the other hand is also tracked, it has to agree before we act.
            if len(fire_votes) >= HANDEDNESS_SWAP_VOTES:
                return fire_is_left >= HANDEDNESS_SWAP_VOTES
            return True
        # Only the fire hand is visible and it is really the left hand.
        return (fire_is_left >= HANDEDNESS_SWAP_VOTES and not steer_votes)

    def toggle_hand_polarity(self):
        """Flip which label means which hand, for a camera that reports reversed."""
        self.swap_handedness = not self.swap_handedness
        for role in (STEER, FIRE):
            self._hand_votes[role].clear()
        self._role_side.clear()
        return self.swap_handedness

    def _assign_roles(self, hand_landmarks, handedness, w, h, now):
        """Match detected hands to roles by predicted position.

        Matching on last-known position alone is ambiguous in exactly the case
        that matters: two hands crossing over each other at similar speed end up
        where the other one was, and the roles swap mid-jump. So each role
        carries a smoothed velocity and we match against where it was *heading*.
        A hand moving right keeps its role through the crossing.

        Roles are never taken from MediaPipe handedness -- the frame is mirrored,
        which inverts those labels, and hardcoding a hand would break for
        left-handed players. When a role's track goes stale we re-seed by screen
        side instead: the left hand steers.
        """
        if not hand_landmarks:
            return {}

        points = [(lm[WRIST].x * w, lm[WRIST].y * h, lm) for lm in hand_landmarks[:2]]
        sides = [self._real_side(handedness[i]) if i < len(handedness) else (None, 0.0)
                 for i in range(len(points))]
        fresh = {r: p for r, p in self._role_pos.items() if now - p[2] <= ROLE_MEMORY}
        radius = ROLE_MATCH_RADIUS * w

        assigned = {}        # role -> landmarks
        taken = set()        # point indices already claimed

        # Handedness is authoritative: the left hand moves, the right hand throws.
        # It is applied through the vote below rather than from the raw label,
        # because a single flickered frame would otherwise flip the controls
        # mid-jump -- which is the bug this whole mechanism exists to prevent. With
        # HANDEDNESS_SWAP_VOTES low, a genuine mismatch is corrected in about an
        # eighth of a second, while one or two bad frames are still ignored.
        if fresh:
            # Greedy nearest-prediction matching over the pairs that pass the
            # radius check. Match PARTIALLY: a role whose own match is 1px away
            # must keep it even when the other hand is nowhere near its
            # prediction. Requiring the whole pairing to be valid meant one hand
            # appearing far from where it was (or a third hand blinking in)
            # discarded *both* identities and re-seeded by screen side, which is
            # how the roles visibly traded places mid-session.
            predicted = {}
            for role, (rx, ry, rt, rvx, rvy) in fresh.items():
                dt = min(now - rt, ROLE_PREDICT_MAX)
                predicted[role] = (rx + rvx * dt, ry + rvy * dt)

            candidates = []
            for role, (px, py) in predicted.items():
                for pi, (x, y, _) in enumerate(points):
                    d = math.hypot(x - px, y - py)
                    if d <= radius:
                        candidates.append((d, role, pi))
            for d, role, pi in sorted(candidates):
                if role in assigned or pi in taken:
                    continue
                assigned[role] = points[pi][2]
                taken.add(pi)

        leftover = sorted((i for i in range(len(points)) if i not in taken),
                          key=lambda i: points[i][0])
        free_roles = [r for r in (STEER, FIRE) if r not in assigned]

        def labelled(i):
            side, score = sides[i]
            return side if (ROLE_BY_HANDEDNESS and side
                            and score >= HANDEDNESS_MIN_SCORE) else None

        if len(free_roles) == 2 and len(leftover) == 2:
            a, b = leftover
            sa, sb = labelled(a), labelled(b)
            if sa and sb and sa != sb:
                # Both hands confidently labelled: settle it now rather than
                # seeding by screen side and letting the vote correct it over the
                # next third of a second.
                assigned[STEER] = points[a if sa == "left" else b][2]
                assigned[FIRE] = points[a if sa == "right" else b][2]
                taken.update((a, b))
                free_roles, leftover = [], []
            elif len(self._role_last) == 2:
                # No usable labels, but both roles have been seen before: prefer
                # the arrangement matching where each one last was, so resting your
                # hands for a second does not bring them back swapped.
                keep = (abs(points[a][0] - self._role_last[STEER][0])
                        + abs(points[b][0] - self._role_last[FIRE][0]))
                flip = (abs(points[b][0] - self._role_last[STEER][0])
                        + abs(points[a][0] - self._role_last[FIRE][0]))
                free_roles = [STEER, FIRE] if keep <= flip else [FIRE, STEER]
        elif len(free_roles) == 2 and len(leftover) == 1:
            # One unclaimed hand and no live track. Its handedness says which role
            # it owns; screen side is only the fallback, and a poor one -- a player
            # reaching across the body lands in the wrong half.
            side = labelled(leftover[0])
            if side:
                free_roles = [STEER if side == "left" else FIRE]
            else:
                free_roles = [STEER if points[leftover[0]][0] < w * 0.5 else FIRE]

        for role, pi in zip(free_roles, leftover):
            assigned[role] = points[pi][2]
            taken.add(pi)

        # Position tracking has decided which track is which; handedness now
        # decides which track owns which role. Acting only on a sustained majority
        # keeps a flickered label from swapping the roles for a frame or two, which
        # in play reads as the controls randomly changing hands.
        if ROLE_BY_HANDEDNESS:
            self._vote_sides(assigned, points, sides)
            if self._roles_are_swapped():
                assigned = {r: lm for r, lm in
                            ((STEER, assigned.get(FIRE)), (FIRE, assigned.get(STEER)))
                            if lm is not None}
                for role in (STEER, FIRE):
                    self._hand_votes[role].clear()
                    self._role_pos.pop(role, None)   # velocities belong to the old hand
                self._vote_sides(assigned, points, sides)

        for role, lm in assigned.items():
            x, y = lm[WRIST].x * w, lm[WRIST].y * h
            prev = self._role_pos.get(role)
            vx = vy = 0.0
            if prev is not None:
                dt = now - prev[2]
                if 1e-3 < dt <= ROLE_MEMORY:
                    vx = ROLE_VEL_BLEND * ((x - prev[0]) / dt) + (1 - ROLE_VEL_BLEND) * prev[3]
                    vy = ROLE_VEL_BLEND * ((y - prev[1]) / dt) + (1 - ROLE_VEL_BLEND) * prev[4]
            self._role_pos[role] = (x, y, now, vx, vy)
            self._role_last[role] = (x, y)      # never expires; identity anchor

        return assigned

    @staticmethod
    def _hand_span(lm, w, h):
        dx = (lm[MIDDLE_MCP].x - lm[WRIST].x) * w
        dy = (lm[MIDDLE_MCP].y - lm[WRIST].y) * h
        return max(math.hypot(dx, dy), 1e-3)

    def _span(self, role, lm, w, h):
        """Median-filtered hand span, plus whether this pose is trustworthy.

        Span is the scale for every threshold in this file, so noise in it is
        noise in everything. Degenerate detections do occur -- a 6px span means
        the wrist and middle-finger MCP landmarks came back nearly coincident,
        which is a failed detection, not a distant hand. Those are rejected
        outright rather than fed to the median, because they arrive in ramping
        runs that a median follows instead of rejecting.

        Returns (span_px, ok). A rejected pose still gets a usable span (the last
        established median) so steering keeps working; callers should refuse to
        fire discrete events on it.
        """
        raw = self._hand_span(lm, w, h)
        hist = self._span_hist[role]
        established = statistics.median(hist) if len(hist) >= SPAN_WARMUP else None

        degenerate = raw < SPAN_MIN_PX or (established is not None
                                           and raw < SPAN_OUTLIER * established)
        if degenerate:
            self._span_rejects[role] += 1
            # If it keeps happening, the hand really has moved or changed shape:
            # stop rejecting reality and re-learn from scratch.
            if self._span_rejects[role] > SPAN_REJECT_LIMIT:
                hist.clear()
                self._span_rejects[role] = 0
            return (established or raw), False

        self._span_rejects[role] = 0
        hist.append(raw)
        return statistics.median(hist), True

    def _acquire_by_sweep(self, wx, wy, span, w, h, now):
        """Learn the origin AND the gain from a short side-to-side sweep.

        This replaces guessing. Every automatic scheme tried here failed on one
        horn of the same dilemma: gate drift correction on the axis reading small
        and a wrong origin can never heal (it reads large, which switches the
        correction off); gate it on the hand being still and a deliberately held
        lean gets absorbed -- and holding a lean is exactly how you run.

        A sweep sidesteps it. The midpoint of what the player can comfortably
        reach is their true centre, and half the width of that reach is the right
        gain for them, at their distance, in their seating position. Three seconds,
        once, and both numbers are measured rather than assumed.
        """
        # Once the sweep window has been given its chance and declined, stop
        # restarting it: every pose from then on feeds the steady-hand path. The
        # first version reset the window instead, which handed the fallback a
        # single pose per 3s cycle -- so a player who simply held their hand up
        # waited ~24s for an origin.
        if self._sweep_gave_up:
            self._acquire_neutral(wx, wy, span, w, h)
            return
        if self._sweep_start is None:
            self._sweep_start = now
            self._sweep_pts = []
        # Ignore poses near the frame edges while calibrating, or the hand simply
        # entering the shot gets measured as part of the player's reach.
        margin = NEUTRAL_EDGE_MARGIN * w
        if margin <= wx <= w - margin:
            self._sweep_pts.append((wx, wy))
        if now - self._sweep_start < SWEEP_TIME:
            return
        xs = [p[0] for p in self._sweep_pts]
        ys = [p[1] for p in self._sweep_pts]
        width = (max(xs) - min(xs)) if xs else 0.0

        # A sweep goes out AND back. A hand arriving in frame travels one way only,
        # and without this test that arrival was read as the player's full reach --
        # producing an origin halfway to wherever they came from.
        reversals, last_dir = 0, 0
        for a, b in zip(xs, xs[1:]):
            if abs(b - a) < 0.05 * span:
                continue
            d = 1 if b > a else -1
            if last_dir and d != last_dir:
                reversals += 1
            last_dir = d

        if width >= SWEEP_MIN_WIDTH * span and reversals >= SWEEP_MIN_REVERSALS:
            mid = (max(xs) + min(xs)) * 0.5
            self._sweep_lean = clamp(width * 0.5,
                                     SWEEP_LEAN_MIN * span, SWEEP_LEAN_MAX * span)
            self._set_neutral(mid, statistics.median(ys), span, w)
            self._sweep_start = None
            self._sweep_pts = []
            print(f"[Steer] calibrated by sweep: centre x={mid:.0f}px, "
                  f"full lean {self._sweep_lean:.0f}px "
                  f"({self._sweep_lean / span:.1f} hand-spans, "
                  f"{reversals} reversals)")
            return
        # Not enough movement to be a sweep. Fall back to the steady-hand method,
        # so a player who just holds their hand up still gets a usable origin.
        self._sweep_gave_up = True
        self._sweep_pts = []
        self._acquire_neutral(wx, wy, span, w, h)

    def _acquire_neutral(self, wx, wy, span, w, h):
        """Learn the steering origin from a hand that is steady and in frame.

        Taking the first pose as neutral -- which is what this used to do -- puts
        the origin wherever the hand happened to enter, typically a corner. A
        neutral at x=75 in a 640px frame leaves only 75px of leftward travel
        against the ~204px a full lean needs, so the axis reads hard-right at rest
        and left is physically unreachable. Instead: ignore poses near the edges,
        wait for the hand to hold still, and take the median.
        """
        margin = NEUTRAL_EDGE_MARGIN * w
        if not (margin <= wx <= w - margin) or wy > h * (1.0 - NEUTRAL_EDGE_MARGIN):
            self._neutral_buf.clear()
            return
        self._neutral_buf.append((wx, wy))
        if len(self._neutral_buf) < NEUTRAL_STABLE_N:
            return
        xs = [p[0] for p in self._neutral_buf]
        ys = [p[1] for p in self._neutral_buf]
        if max(xs) - min(xs) > span * NEUTRAL_STABLE_SPREAD:
            return                       # still moving; the window slides on
        if max(ys) - min(ys) > span * NEUTRAL_STABLE_SPREAD:
            return
        self._set_neutral(statistics.median(xs), statistics.median(ys), span, w)

    def _set_neutral(self, nx, ny, span, w):
        """Store neutral exactly where the hand rests.

        Do NOT pull it toward the frame centre to make room for a full lean. That
        was the first attempt here and it recreates the original bug from the
        other direction: a hand resting at x=87 with neutral forced to x=163 sits
        76px from its own origin, so the axis reads WALK LEFT while the hand is
        motionless. Rest must always read zero. Reachability is handled instead by
        scaling each side to the room actually available -- see _side_leans.
        """
        self._neutral = (nx, ny)

    def _side_leans(self, nx, span, w):
        """Travel (px) that maps to full deflection. SYMMETRIC, sized to the
        narrower side.

        Scaling each side to its own available room was a mistake. It keeps full
        deflection reachable both ways, but the gains differ -- and the steering
        hand naturally rests near its own edge of the frame, so it has little room
        outward and lots inward. The result played as hair-trigger one way and
        sluggish the other, which is worse than losing a bit of range.

        So both sides get the SAME travel: the smaller of the two. The roomier
        direction simply doesn't use all its space, which costs nothing, and the
        axis feels the same in both directions. The floor keeps a hand resting
        very close to an edge from making the whole axis twitchy; when the room
        available is below it, the UI asks the player to move to the middle.
        """
        nominal = self._sweep_lean if self._sweep_lean else STEER_FULL_LEAN * span
        floor = STEER_SIDE_MIN * nominal
        pad = STEER_EDGE_FRAC * w
        room = min(nx - pad, w - pad - nx)
        lean = clamp(room, floor, nominal)
        return lean, lean

    @staticmethod
    def zone_bounds(w):
        """(left, centre, right) in pixels of the steering band."""
        lo, hi = STEER_ZONE[0] * w, STEER_ZONE[1] * w
        return lo, (lo + hi) * 0.5, hi

    def _steer_by_zone(self, wx, wy, span, w, h, now):
        """Map a fixed band of the frame straight onto the axis.

        Reaching the right edge of the band is full speed, whatever the hand did
        before. No origin to learn, nothing to drift, and the band is drawn on the
        preview so the player can see where full speed lives instead of hunting for
        it.
        """
        lo, mid, hi = self.zone_bounds(w)
        half = max((hi - lo) * 0.5, 1.0)
        raw = (wx - mid) / half
        mag = (abs(raw) - STEER_DEADZONE) / (1.0 - STEER_DEADZONE)
        analog = math.copysign(clamp(mag, 0.0, 1.0), raw) if mag > 0 else 0.0

        vel_up, travel = self._vertical(wy, span, now)
        warming = len(self._span_hist[STEER]) < SPAN_WARMUP
        if warming:
            vel_up, travel = 0.0, 0.0
        self._maybe_flick(vel_up, travel, now)

        # Duck is a fixed band too: hand low in frame, held, and not mid-flick.
        low = wy > DUCK_ZONE_Y * h and abs(vel_up) < 3.0
        self._duck_run = self._duck_run + 1 if low else 0

        self._at_rest = False
        self._last_steer_t = now
        return {
            "analog": analog,
            "run": abs(analog) > RUN_THRESHOLD,
            "duck": self._duck_run >= DUCK_HOLD,
            "vel_up": vel_up,
            "span": span,
            "reach_left": 1.0,
            "reach_right": 1.0,
        }

    def _vertical(self, wy, span, now):
        """Upward wrist velocity and travel, in hand-spans (and spans/second).

        Differentiate the POSITION and divide by span once, never the ratio -- see
        the long note in the relative path for what that bug cost.
        """
        self._y_hist.append((now, wy))
        for t_old, y_old in self._y_hist:
            if now - t_old >= FLICK_WINDOW:
                continue
            dt = now - t_old
            if dt > 1e-3:
                travel = (y_old - self._y_hist[-1][1]) / span
                return travel / dt, travel
            break
        return 0.0, 0.0

    def _maybe_flick(self, vel_up, travel, now):
        """Shared jump trigger: re-arm on slowing, fire on speed AND travel."""
        if vel_up < FLICK_REARM:
            self._flick_armed = True
        if (self._flick_armed and vel_up >= FLICK_VEL_MIN
                and travel >= FLICK_MIN_TRAVEL
                and now - self._last_flick_t > FLICK_REFRACTORY):
            self._flick_armed = False
            self._last_flick_t = now
            span_range = FLICK_VEL_MAX - FLICK_VEL_MIN
            strength = 0.45 + 0.55 * clamp((vel_up - FLICK_VEL_MIN) / span_range,
                                           0.0, 1.0)
            self._events.append({"kind": "flick", "t_capture": now,
                                 "strength": strength, "vel": vel_up})
            self._duck_run = 0

    def _process_steer(self, lm, w, h, now):
        span, span_ok = self._span(STEER, lm, w, h)
        wx, wy = lm[WRIST].x * w, lm[WRIST].y * h

        if STEER_MODE == "zone":
            return self._steer_by_zone(wx, wy, span, w, h, now)

        if self._neutral is None:
            self._acquire_by_sweep(wx, wy, span, w, h, now)
        nx, ny = self._neutral if self._neutral else (wx, wy)

        # Analog deflection, scaled by the room available on the side being used,
        # then deadzone removed and rescaled so it starts at 0.
        lean_l, lean_r = self._side_leans(nx, span, w)
        raw = (wx - nx) / (lean_r if wx >= nx else lean_l)
        mag = (abs(raw) - STEER_DEADZONE) / (1.0 - STEER_DEADZONE)
        analog = math.copysign(clamp(mag, 0.0, 1.0), raw) if mag > 0 else 0.0
        if self._neutral is None:
            analog = 0.0                 # don't steer on a guessed origin

        # Drift correction, gated on the hand being STILL -- not on the analog
        # value being small.
        #
        # Gating on |analog| was a bug that made the system unable to heal: a
        # misplaced origin reads as a large deflection, which switched the
        # correction off, so exactly the situation that needed recentring was the
        # one that prevented it. The origin stayed wrong for the whole session.
        #
        # Stillness is the right signal. A hand that has not moved for a couple of
        # seconds is at rest by definition, whatever the axis currently reports, so
        # that position is the truth and the origin should move to it.
        self._still_hist.append((now, wx, wy))
        while self._still_hist and now - self._still_hist[0][0] > STILL_TIME:
            self._still_hist.popleft()
        at_rest = False
        if len(self._still_hist) >= 8 and now - self._still_hist[0][0] >= STILL_TIME * 0.9:
            xs = [p[1] for p in self._still_hist]
            ys = [p[2] for p in self._still_hist]
            at_rest = (max(xs) - min(xs) < STILL_RADIUS * span
                       and max(ys) - min(ys) < STILL_RADIUS * span)
        if (self._neutral is not None and span_ok and at_rest
                and abs(analog) < NEUTRAL_RECENTRE_MAX_ANALOG):
            dt = now - self._last_steer_t if self._last_steer_t else 0.0
            if 0.0 < dt < 0.5:
                k = 1.0 - math.exp(-dt / NEUTRAL_RECENTRE_TAU)
                self._set_neutral(nx + (statistics.median(xs) - nx) * k,
                                  ny + (statistics.median(ys) - ny) * k, span, w)
                nx, ny = self._neutral
                # Recompute against the corrected origin so the player sees the
                # effect this frame rather than the next.
                lean_l, lean_r = self._side_leans(nx, span, w)
                raw = (wx - nx) / (lean_r if wx >= nx else lean_l)
                mag = (abs(raw) - STEER_DEADZONE) / (1.0 - STEER_DEADZONE)
                analog = math.copysign(clamp(mag, 0.0, 1.0), raw) if mag > 0 else 0.0
        self._at_rest = at_rest
        self._last_steer_t = now

        # Vertical velocity in hand-spans/sec, upward positive.
        #
        # Store RAW pixels and divide by span once, at the end. Storing wy/span
        # and differentiating that is wrong: d(y/s)/dt = (y's - ys')/s², so when
        # the span is changing -- which it does violently while a returning hand's
        # landmarks settle -- the s' term swamps the real motion. That is how a
        # hand moving 21px *downwards* reported 195 span/s upwards. Differentiate
        # the position, then scale; never differentiate the ratio.
        self._y_hist.append((now, wy))
        vel_up, travel = 0.0, 0.0
        for t_old, y_old in self._y_hist:
            if now - t_old >= FLICK_WINDOW:
                continue
            dt = now - t_old
            if dt > 1e-3:
                travel = (y_old - self._y_hist[-1][1]) / span
                vel_up = travel / dt
                break

        # The trigger re-arms only after the hand slows down. Without that, one
        # long continuous sweep upward keeps re-crossing the threshold and fires
        # a second jump the moment the refractory window expires.
        if vel_up < FLICK_REARM:
            self._flick_armed = True

        # A just-re-acquired hand has an unreliable span, and span scales vel_up.
        # Refuse to jump until it settles.
        warming = not span_ok or len(self._span_hist[STEER]) < SPAN_WARMUP
        if warming:
            vel_up, travel = 0.0, 0.0

        # Velocity alone is not enough: landmark jitter clears a speed threshold
        # over one or two frames without the hand going anywhere. Requiring real
        # upward displacement as well is what makes it safe to lower the speed
        # threshold far enough to catch the gentle flicks players actually make.
        if (self._flick_armed and not warming and vel_up >= FLICK_VEL_MIN
                and travel >= FLICK_MIN_TRAVEL
                and now - self._last_flick_t > FLICK_REFRACTORY):
            self._flick_armed = False
            self._last_flick_t = now
            span_range = FLICK_VEL_MAX - FLICK_VEL_MIN
            strength = 0.45 + 0.55 * clamp((vel_up - FLICK_VEL_MIN) / span_range, 0.0, 1.0)
            self._events.append({
                "kind": "flick", "t_capture": now,
                "strength": strength, "vel": vel_up,
            })
            self._duck_run = 0

        # Duck needs a sustained low hand, so flick recoil doesn't read as one.
        low = (wy - ny) / span > DUCK_OFFSET and abs(vel_up) < 3.0
        self._duck_run = self._duck_run + 1 if low else 0

        nominal = STEER_FULL_LEAN * span
        return {
            "analog": analog,
            "run": abs(analog) > RUN_THRESHOLD,
            "duck": self._duck_run >= DUCK_HOLD,
            "vel_up": vel_up,
            "span": span,
            "reach_left": lean_l / nominal,
            "reach_right": lean_r / nominal,
        }

    def _process_fire(self, lm, w, h, now):
        """Pinch detection against the player's own open-hand baseline.

        A fixed threshold on gap/span does not survive across sessions. Two logs
        of the same person gave median gaps of 0.75 and 0.50 -- the resting
        posture of the fire hand simply differs day to day, and a threshold that
        sat comfortably above resting in one session sat right on the median in
        the other, reading as "pinched" 47% of the time. So track a slow estimate
        of what open looks like *for this hand right now*, and fire on a relative
        drop from it. Same principle as the flick: detect the gesture, not an
        absolute posture.
        """
        span, span_ok = self._span(FIRE, lm, w, h)
        wx, wy = lm[WRIST].x * w, lm[WRIST].y * h
        dx = (lm[THUMB_TIP].x - lm[INDEX_TIP].x) * w
        dy = (lm[THUMB_TIP].y - lm[INDEX_TIP].y) * h
        gap = math.hypot(dx, dy) / span

        # THROW mode: fire on a fast wrist movement in any direction -- the same
        # edge-triggered velocity detector that works for the jump. Session two
        # showed why: the pinch is a posture test, and this player's relaxed hand
        # is a loose fist, so "closed" carried no intent (the gap never opened past
        # 0.60 for a continuous 34s). Velocity edges generalise; postures don't.
        self._fire_hist.append((now, wx, wy))
        f_vel, f_travel = 0.0, 0.0
        for t_old, x_old, y_old in self._fire_hist:
            if now - t_old >= FLICK_WINDOW:
                continue
            dt = now - t_old
            if dt > 1e-3:
                _, x_now, y_now = self._fire_hist[-1]
                f_travel = math.hypot(x_now - x_old, y_now - y_old) / span
                f_vel = f_travel / dt
                break

        warming = not span_ok or len(self._span_hist[FIRE]) < SPAN_WARMUP
        if warming:
            f_vel, f_travel = 0.0, 0.0
        if f_vel < THROW_REARM:
            self._throw_armed = True

        if FIRE_MODE == "throw":
            if (self._throw_armed and not warming and f_vel >= THROW_VEL_MIN
                    and f_travel >= THROW_MIN_TRAVEL
                    and now - self._last_fire_t > FIRE_COOLDOWN):
                self._throw_armed = False
                self._last_fire_t = now
                self._events.append({"kind": "throw", "t_capture": now,
                                     "vel": f_vel, "gap": gap})
            self._fire_vel = f_vel
            return gap

        # Learn the baseline from every frame, including pinched ones. It is
        # tempting to learn only from open frames -- the reference then stays
        # "true" during a pinch -- but replaying both sessions shows that latches
        # the hand shut for up to 40s: a frozen-high baseline puts the release
        # threshold above anything the hand actually opens to. Letting the
        # baseline sag while pinched lowers the release bar and self-corrects.
        self._gap_hist.append(gap)
        baseline = pctile(self._gap_hist, PINCH_BASELINE_PCT)
        # A hand closed the whole time has no meaningful open reference, and
        # firing off wobbles around it would be chatter.
        trustworthy = span_ok and baseline >= PINCH_BASELINE_MIN

        if not self._pinched:
            if trustworthy and gap < PINCH_DROP * baseline:
                self._pinched = True
                if now - self._last_fire_t > FIRE_COOLDOWN:
                    self._last_fire_t = now
                    self._events.append({"kind": "pinch", "t_capture": now,
                                         "gap": gap, "baseline": baseline})
        else:
            # Releasing must ALWAYS be possible. Gating it on the same
            # trustworthiness test as firing is what allows a latch: an untrusted
            # baseline would freeze the hand in the pinched state indefinitely.
            release_at = (PINCH_RELEASE * baseline if trustworthy
                          else PINCH_RELEASE_ABS)
            if gap > release_at:
                self._pinched = False
        return gap

    # ---------- overlay (drawn on the detector thread) ----------
    def _draw_overlay(self, frame, hands, steer_out, pinch_gap):
        h, w = frame.shape[:2]
        # BGR: blue for the moving hand, orange for the firing hand, matching the
        # game's HUD exactly so the two never disagree about what a colour means.
        role_colors = {STEER: (255, 170, 60), FIRE: (50, 140, 255)}   # BGR

        for role, lm in hands.items():
            color = role_colors[role]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], color, 2)
            for p in pts:
                cv2.circle(frame, p, 3, (255, 255, 255), -1)
            side = self._role_side.get(role)
            verb = "MOVE" if role == STEER else "FIRE"
            label = f"{verb} ({side} hand)" if side else verb
            cv2.putText(frame, label, (pts[WRIST][0] - 45, pts[WRIST][1] + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            if role == FIRE and pinch_gap is not None:
                mid = ((pts[THUMB_TIP][0] + pts[INDEX_TIP][0]) // 2,
                       (pts[THUMB_TIP][1] + pts[INDEX_TIP][1]) // 2)
                hot = (0, 80, 255) if self._pinched else (200, 200, 200)
                cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], hot, 2)
                cv2.circle(frame, mid, 14 if self._pinched else 8, hot, 2)

        if STEER_MODE == "zone":
            lo, mid, hi = self.zone_bounds(w)
            dead = STEER_DEADZONE * (hi - lo) * 0.5
            # The band, drawn where the player can see it. Full speed lives at the
            # right line; there is nothing to hunt for and nothing that moves.
            overlay = frame.copy()
            cv2.rectangle(overlay, (int(lo), 0), (int(hi), h), (70, 55, 25), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            for x, col, label in ((lo, (120, 220, 120), "FULL LEFT"),
                                  (hi, (120, 220, 120), "FULL RIGHT")):
                xd = clamp(int(x), 1, w - 2)
                cv2.line(frame, (xd, 0), (xd, h), col, 2)
                tx = xd - 100 if label == "FULL RIGHT" else xd + 6
                cv2.putText(frame, label, (max(2, tx), 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.line(frame, (int(mid), 0), (int(mid), h), (170, 170, 170), 1)
            cv2.putText(frame, "STOP", (int(mid) - 22, h - 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            for sign in (-1, 1):
                xd = int(mid + sign * dead)
                cv2.line(frame, (xd, h // 2 - 10), (xd, h // 2 + 10), (150, 150, 150), 1)
            cv2.line(frame, (0, int(DUCK_ZONE_Y * h)), (w, int(DUCK_ZONE_Y * h)),
                     (90, 140, 200), 1)
            cv2.putText(frame, "duck below", (8, int(DUCK_ZONE_Y * h) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 140, 200), 1, cv2.LINE_AA)
        elif self._neutral is not None:
            nx, ny = int(self._neutral[0]), int(self._neutral[1])
            cv2.line(frame, (nx, ny - 18), (nx, ny + 18), (180, 180, 180), 1)
            cv2.line(frame, (nx - 18, ny), (nx + 18, ny), (180, 180, 180), 1)
            cv2.putText(frame, "neutral", (nx - 28, ny - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

            # Draw the steering range itself. Without this, an origin parked in a
            # corner looks like a harmless crosshair rather than the reason one
            # direction is unreachable.
            span = steer_out.get("span")
            if span:
                lean_l, lean_r = self._side_leans(nx, span, w)
                nominal = STEER_FULL_LEAN * span
                cramped = False
                for x, frac, name in ((nx - lean_l, lean_l / nominal, "FULL LEFT"),
                                      (nx + lean_r, lean_r / nominal, "FULL RIGHT")):
                    tight = frac <= STEER_SIDE_MIN + 1e-6
                    cramped = cramped or tight
                    color = (60, 80, 255) if tight else (120, 220, 120)
                    xd = clamp(int(x), 2, w - 3)
                    cv2.line(frame, (xd, 0), (xd, h), color, 2)
                    tx = xd - 110 if "RIGHT" in name else xd + 6
                    cv2.putText(frame, name, (max(2, tx), 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                dead = int(STEER_DEADZONE * min(lean_l, lean_r))
                cv2.line(frame, (nx - dead, ny - 8), (nx - dead, ny + 8), (110, 110, 110), 1)
                cv2.line(frame, (nx + dead, ny - 8), (nx + dead, ny + 8), (110, 110, 110), 1)
                if cramped:
                    cv2.putText(frame, "not much room that way -- move to the middle "
                                "and press C", (12, h - 66),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1,
                                cv2.LINE_AA)
        else:
            cv2.putText(frame, "SWEEP YOUR STEERING HAND LEFT AND RIGHT",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, "as far as is comfortable -- sets your centre and range",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 255), 1,
                        cv2.LINE_AA)
            if self._sweep_pts:
                xs = [p[0] for p in self._sweep_pts]
                cv2.line(frame, (int(min(xs)), h - 30), (int(max(xs)), h - 30),
                         (0, 210, 255), 4)
                for xx in (min(xs), max(xs)):
                    cv2.line(frame, (int(xx), h - 44), (int(xx), h - 16),
                             (0, 210, 255), 2)

        if STEER not in hands:
            cv2.putText(frame, "steer hand lost", (12, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA)
        if FIRE not in hands:
            cv2.putText(frame, "fire hand lost", (12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA)

    # ---------- render-loop API ----------
    def read(self):
        with self._lock:
            snap = dict(self.snapshot)
            events = list(self._events)
            self._events.clear()
        return snap, events

    def drain_log(self):
        rows = []
        while self._log_rows:
            rows.append(self._log_rows.popleft())
        return rows

    def tracker_fps(self):
        times = list(self.tracker_frame_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._detector is not None:
            self._detector.close()
        if self.cap is not None:
            self.cap.release()


# ==========================================
# SANDBOX -- just enough platformer to feel the latency
# ==========================================
GRAVITY = 0.9
WALK_SPEED = 3.2
RUN_SPEED = 6.6
COYOTE_TIME = 0.15
JUMP_BUFFER = 0.15

GROUND_Y = SANDBOX_RECT.bottom - 26
GAP = (SANDBOX_RECT.left + 520, SANDBOX_RECT.left + 660)


class Sandbox:
    """One square, one gap, gravity. The point is to feel input lag, not to be
    a game -- the gap exists so coyote time and jump buffering are measurable.
    """

    def __init__(self):
        self.assists = True
        self.falls = 0
        self.reset()

    def reset(self, now=None):
        self.x = float(SANDBOX_RECT.left + 60)
        self.y = float(GROUND_Y)
        self.vy = 0.0
        self.w, self.h = 24, 32
        self.grounded = True
        self.jumps_used = 0
        self.facing = 1
        self.last_ground_t = time.perf_counter() if now is None else now
        self.buffered_jump = None      # (t, strength)
        self.fireballs = []

    def on_ground_at(self, x):
        return not (GAP[0] < x < GAP[1])

    def request_jump(self, strength, now):
        """Returns a label for the log: what the flick actually did."""
        # Coyote time forgives a flick that arrives just after the ledge -- but
        # only if no jump has been spent yet. Otherwise the window would re-open
        # a ground jump right after a real one, and a flick every ~140ms would
        # let the player climb forever.
        coyote_ok = (self.assists and self.jumps_used == 0
                     and (now - self.last_ground_t) <= COYOTE_TIME)
        if self.grounded or coyote_ok:
            self._launch(strength)
            self.jumps_used = 1
            return "JUMP"
        if self.jumps_used < 2:
            self._launch(strength)
            self.jumps_used = 2
            return "DOUBLE JUMP"
        if self.assists:
            self.buffered_jump = (now, strength)
            return "buffered"
        return "ignored"

    def _launch(self, strength):
        self.vy = -(9.0 + 8.0 * strength)
        self.grounded = False

    def fire(self):
        cx = self.x + (self.w if self.facing > 0 else 0)
        self.fireballs.append([cx, self.y - self.h * 0.6, 11.0 * self.facing])

    def update(self, analog, run, duck, keys_dir, now):
        move = keys_dir if keys_dir else analog
        speed = (RUN_SPEED if run or abs(keys_dir) > 0.9 else WALK_SPEED)
        if duck:
            speed *= 0.4
        self.x += move * speed
        if move:
            self.facing = 1 if move > 0 else -1
        self.x = clamp(self.x, SANDBOX_RECT.left + 4, SANDBOX_RECT.right - self.w - 4)

        self.vy += GRAVITY
        self.y += self.vy

        foot_x = self.x + self.w * 0.5
        if self.y >= GROUND_Y and self.on_ground_at(foot_x):
            self.y = GROUND_Y
            self.vy = 0.0
            if not self.grounded:
                self.grounded = True
                self.jumps_used = 0
                if self.assists and self.buffered_jump:
                    bt, bs = self.buffered_jump
                    if now - bt <= JUMP_BUFFER:
                        self._launch(bs)
                        self.jumps_used = 1
                    self.buffered_jump = None
            self.last_ground_t = now
        else:
            self.grounded = False

        if self.y > SANDBOX_RECT.bottom + 80:
            x_keep = self.x
            self.reset(now)
            self.falls += 1
            self.x = clamp(x_keep - 120, SANDBOX_RECT.left + 40, GAP[0] - 60)

        for fb in self.fireballs:
            fb[0] += fb[2]
        self.fireballs = [fb for fb in self.fireballs
                          if SANDBOX_RECT.left - 20 < fb[0] < SANDBOX_RECT.right + 20]

    def draw(self, surf, duck):
        pygame.draw.rect(surf, (16, 20, 30), SANDBOX_RECT, border_radius=8)
        pygame.draw.rect(surf, (40, 48, 68), SANDBOX_RECT, 1, border_radius=8)

        for x0, x1 in ((SANDBOX_RECT.left + 1, GAP[0]), (GAP[1], SANDBOX_RECT.right - 1)):
            pygame.draw.rect(surf, (90, 70, 50), (x0, GROUND_Y, x1 - x0, SANDBOX_RECT.bottom - GROUND_Y - 1))
            pygame.draw.line(surf, (150, 120, 80), (x0, GROUND_Y), (x1, GROUND_Y), 2)

        h = self.h * (0.6 if duck else 1.0)
        body = pygame.Rect(int(self.x), int(self.y - h), self.w, int(h))
        pygame.draw.rect(surf, (230, 60, 60), body, border_radius=4)
        eye_x = body.right - 7 if self.facing > 0 else body.left + 3
        pygame.draw.rect(surf, (255, 255, 255), (eye_x, body.top + 5, 4, 4))
        for fb in self.fireballs:
            pygame.draw.circle(surf, COLOR_FIRE, (int(fb[0]), int(fb[1])), 6)
            pygame.draw.circle(surf, (255, 230, 120), (int(fb[0]), int(fb[1])), 3)


# ==========================================
# RENDER HELPERS
# ==========================================
def pct(values, q):
    vals = sorted(values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = clamp(int(round(q * (len(vals) - 1))), 0, len(vals) - 1)
    return vals[idx]


def lat_color(ms):
    return COLOR_OK if ms < 80 else COLOR_WARN if ms < 140 else COLOR_BAD


class Spike:
    def __init__(self, log_path=None, camera_id=0):
        pygame.init()
        pygame.display.set_caption("Two-Hand Control Spike")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = self._mono(16)
        self.small = self._mono(13)
        self.big = self._mono(22, bold=True)

        self.log_path = log_path
        self._csv_file = None
        self._csv = None
        if log_path:
            self._csv_file = open(log_path, "w", newline="")
            self._csv = csv.DictWriter(self._csv_file, fieldnames=LOG_FIELDS,
                                       restval="", extrasaction="ignore")
            self._csv.writeheader()
            # Stamp the thresholds in force, so a log is interpretable later even
            # after the constants have been retuned.
            tuning = tuning_meta()
            self._csv.writerow({"kind": "meta", "t": 0, "note": tuning})
            print(f"[Log] writing {log_path}")

        self.tracker = TwoHandTracker(camera_id=camera_id, logging_on=bool(log_path))
        self.sandbox = Sandbox()
        self.log = collections.deque(maxlen=13)
        self.lat_e2e = collections.deque(maxlen=240)
        self.last_seq = -1
        self.keyboard_mode = False
        self.counts = {"flick": 0, "pinch": 0, "jump": 0, "double": 0, "dropped_jump": 0}
        self.t_start = time.perf_counter()

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

    # ---------- main loop ----------
    def run(self):
        print(__doc__)
        running = True
        while running:
            now = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_c:
                        self.tracker.calibrate()
                        self._log("recalibrating neutral", COLOR_DIM)
                        self._csv_marker("calibrate", now)
                    elif event.key == pygame.K_t:
                        self.sandbox.assists = not self.sandbox.assists
                        state = "ON" if self.sandbox.assists else "OFF"
                        self._log(f"assists {state} (coyote + jump buffer)", COLOR_WARN)
                        self._csv_marker(f"assists={state}", now)
                    elif event.key == pygame.K_k:
                        self.keyboard_mode = not self.keyboard_mode
                        mode = "KEYBOARD" if self.keyboard_mode else "GESTURE"
                        self._log(f"control: {mode}", COLOR_WARN)
                        self._csv_marker(f"control={mode}", now)
                    elif event.key == pygame.K_r:
                        self.sandbox.reset()
                        self._csv_marker("reset", now)
                    elif event.key == pygame.K_h and self.tracker.camera_ok:
                        sw = self.tracker.toggle_hand_polarity()
                        self._log(f"hand polarity {'SWAPPED' if sw else 'normal'}",
                                  COLOR_WARN)
                        self._csv_marker(f"swap_handedness={sw}", now)
                    elif event.key == pygame.K_SPACE and self.keyboard_mode:
                        label = self.sandbox.request_jump(1.0, now)
                        self._log(f"SPACE -> {label}", COLOR_TEXT)

            snap, events = self.tracker.read()

            # Sample end-to-end latency once per new pose, not once per frame.
            if snap["seq"] != self.last_seq and snap["t_capture"] is not None:
                self.last_seq = snap["seq"]
                self.lat_e2e.append(now - snap["t_capture"])

            # Pose rows first, so events land after the poses they came from.
            if self._csv:
                self._csv.writerows(self.tracker.drain_log())

            for ev in events:
                lat_ms = (now - ev["t_capture"]) * 1000.0
                if ev["kind"] == "flick":
                    self.counts["flick"] += 1
                    label = self.sandbox.request_jump(ev["strength"], now)
                    if label == "JUMP":
                        self.counts["jump"] += 1
                    elif label == "DOUBLE JUMP":
                        self.counts["double"] += 1
                    else:
                        self.counts["dropped_jump"] += 1
                    color = COLOR_STEER if "JUMP" in label else COLOR_DIM
                    self._log(f"FLICK v{ev['vel']:4.1f} s{ev['strength']:.2f} "
                              f"-> {label:<12} {lat_ms:5.0f}ms", color)
                    self._csv_event("flick", ev, lat_ms, label,
                                    vel=round(ev["vel"], 3),
                                    strength=round(ev["strength"], 3))
                elif ev["kind"] == "throw":
                    self.counts["pinch"] += 1
                    self.sandbox.fire()
                    self._log(f"THROW v{ev['vel']:4.1f}       -> FIREBALL     "
                              f"{lat_ms:5.0f}ms", COLOR_FIRE)
                    self._csv_event("throw", ev, lat_ms, "FIREBALL",
                                    vel=round(ev["vel"], 3))
                else:
                    self.counts["pinch"] += 1
                    self.sandbox.fire()
                    self._log(f"PINCH gap{ev['gap']:.2f}      -> FIREBALL     "
                              f"{lat_ms:5.0f}ms", COLOR_FIRE)
                    self._csv_event("pinch", ev, lat_ms, "FIREBALL",
                                    pinch_gap=round(ev["gap"], 4))

            keys = pygame.key.get_pressed()
            keys_dir = 0.0
            if self.keyboard_mode:
                keys_dir = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (1.0 if keys[pygame.K_LEFT] else 0.0)
            duck = snap["duck"] or (self.keyboard_mode and keys[pygame.K_DOWN])
            analog = 0.0 if self.keyboard_mode else snap["analog"]
            self.sandbox.update(analog, snap["run"], duck, keys_dir, now)

            self.draw(snap, duck)
            self.clock.tick(FPS)

        if self._csv:
            self._csv.writerows(self.tracker.drain_log())   # don't lose the tail
        self.tracker.close()
        self.summary()
        self.close_log()
        pygame.quit()

    def _log(self, text, color=COLOR_TEXT):
        self.log.append((f"{time.perf_counter() - self.t_start:6.1f}s  {text}", color))

    def _csv_marker(self, note, now):
        if self._csv:
            self._csv.writerow({"kind": "marker", "t": round(now - self.tracker.t0, 4),
                                "note": note})

    def _csv_event(self, kind, ev, lat_ms, outcome, **extra):
        """Log an event stamped with the POSE time it came from, not the time the
        render loop got round to it -- otherwise the event timeline inherits
        render jitter and can't be lined up against the pose stream.
        """
        if not self._csv:
            return
        row = {
            "kind": kind,
            "t": round(ev["t_capture"] - self.tracker.t0, 4),
            "lat_ms": round(lat_ms, 2),
            "outcome": outcome,
            "note": "keyboard" if self.keyboard_mode else "",
        }
        row.update(extra)
        self._csv.writerow(row)

    # ---------- drawing ----------
    def draw(self, snap, duck):
        self.screen.fill(COLOR_BG)
        self.draw_preview(snap)
        self.draw_panel(snap, duck)
        self.sandbox.draw(self.screen, duck)
        self.draw_sandbox_hud()
        pygame.display.flip()

    def draw_preview(self, snap):
        frame = snap["frame"]
        if frame is None:
            pygame.draw.rect(self.screen, COLOR_PANEL, PREVIEW_RECT, border_radius=8)
            lines = self.tracker.error_lines or ["Waiting for camera..."]
            for i, line in enumerate(lines):
                font = self.big if i == 0 else self.small
                color = COLOR_BAD if i == 0 else COLOR_TEXT
                surf = font.render(line, True, color)
                self.screen.blit(surf, (PREVIEW_RECT.centerx - surf.get_width() // 2,
                                        PREVIEW_RECT.centery - 40 + i * 26))
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(np.ascontiguousarray(rgb).tobytes(),
                                       (frame.shape[1], frame.shape[0]), "RGB")
        if surf.get_size() != PREVIEW_RECT.size:
            surf = pygame.transform.smoothscale(surf, PREVIEW_RECT.size)
        self.screen.blit(surf, PREVIEW_RECT.topleft)
        pygame.draw.rect(self.screen, (40, 48, 68), PREVIEW_RECT, 1, border_radius=2)

    def draw_panel(self, snap, duck):
        pygame.draw.rect(self.screen, COLOR_PANEL, PANEL_RECT, border_radius=8)
        x = PANEL_RECT.left + 16
        y = PANEL_RECT.top + 14
        w = PANEL_RECT.width - 32

        # --- roles ---
        for role, seen, color in ((STEER, snap["steer_seen"], COLOR_STEER),
                                  (FIRE, snap["fire_seen"], COLOR_FIRE)):
            dot = COLOR_OK if seen else COLOR_BAD
            pygame.draw.circle(self.screen, dot,
                               (x + 6, y + 8), 5)
            label = f"{role} hand  {'locked' if seen else 'lost'}"
            self.screen.blit(self.font.render(label, True, color), (x + 20, y))
            y += 24

        if not snap["calibrated"]:
            self.screen.blit(self.small.render(
                "hold the steer hand still, away from the edges", True, COLOR_WARN), (x, y))
        else:
            rl, rr = snap.get("reach_left"), snap.get("reach_right")
            if rl is not None and rr is not None:
                worst = min(rl, rr)
                col = COLOR_OK if worst >= 1.0 else COLOR_WARN if worst >= 0.8 else COLOR_BAD
                note = "" if worst >= 1.0 else "  <- move to the middle of frame"
                rest = "  [at rest, centring]" if snap.get("at_rest") else ""
                self.screen.blit(self.small.render(
                    f"lean range {worst:.2f}x nominal, both ways{note}{rest}",
                    True, col), (x, y))
        y += 22

        # --- analog steer bar ---
        self.screen.blit(self.small.render("ANALOG STEER", True, COLOR_DIM), (x, y))
        y += 18
        bar = pygame.Rect(x, y, w, 26)
        pygame.draw.rect(self.screen, (14, 17, 26), bar, border_radius=4)
        cx = bar.centerx
        dz = int(w * 0.5 * STEER_DEADZONE)
        pygame.draw.rect(self.screen, (30, 36, 52), (cx - dz, bar.top, dz * 2, bar.height))
        for sign in (-1, 1):
            rx = cx + sign * int(w * 0.5 * RUN_THRESHOLD)
            pygame.draw.line(self.screen, COLOR_WARN, (rx, bar.top), (rx, bar.bottom), 1)
        pygame.draw.line(self.screen, COLOR_DIM, (cx, bar.top), (cx, bar.bottom), 1)
        val = snap["analog"]
        if abs(val) > 0.001:
            fill_w = int(abs(val) * w * 0.5)
            fx = cx if val > 0 else cx - fill_w
            pygame.draw.rect(self.screen, COLOR_STEER, (fx, bar.top + 4, fill_w, bar.height - 8),
                             border_radius=3)
        pygame.draw.rect(self.screen, (60, 70, 92), bar, 1, border_radius=4)
        y += 32

        if duck:
            state = "DUCK"
        elif abs(val) < 0.001:
            state = "IDLE"
        else:
            arrow = "RIGHT" if val > 0 else "LEFT"
            state = f"{'RUN' if snap['run'] else 'WALK'} {arrow}"
        self.screen.blit(self.big.render(f"{state:<12}{val:+.2f}", True, COLOR_TEXT), (x, y))
        y += 32

        if FIRE_MODE == "throw":
            fire_txt = f"throw vel {snap.get('fire_vel', 0.0):5.1f} span/s"
        else:
            gap_txt = f"{snap['pinch_gap']:.2f}" if snap["pinch_gap"] is not None else "--"
            fire_txt = f"pinch {gap_txt} {'PINCHED' if snap['pinched'] else 'open'}"
        self.screen.blit(self.font.render(
            f"jump vel {snap['vel_up']:5.1f} span/s   {fire_txt}",
            True, COLOR_DIM), (x, y))
        y += 30

        # --- latency ---
        infer = [v * 1000 for v in self.tracker.lat_infer]
        e2e = [v * 1000 for v in self.lat_e2e]
        self.screen.blit(self.small.render("LATENCY (capture -> ...)", True, COLOR_DIM), (x, y))
        y += 18
        for name, vals in (("landmarks", infer), ("game loop", e2e)):
            if vals:
                mean, p95 = statistics.fmean(vals), pct(vals, 0.95)
                txt = f"{name:<10} mean {mean:5.0f}ms   p95 {p95:5.0f}ms"
                self.screen.blit(self.font.render(txt, True, lat_color(p95)), (x, y))
            else:
                self.screen.blit(self.font.render(f"{name:<10} --", True, COLOR_DIM), (x, y))
            y += 21
        self.screen.blit(self.small.render(
            f"tracker {self.tracker.tracker_fps():4.1f} fps    "
            f"render {self.clock.get_fps():4.1f} fps    dropped {self.tracker.dropped}",
            True, COLOR_DIM), (x, y))
        y += 16
        self.screen.blit(self.small.render(
            "excludes sensor+USB time before read() returns", True, COLOR_DIM), (x, y))
        y += 24

        # --- event log ---
        pygame.draw.line(self.screen, (40, 48, 68), (x, y), (x + w, y), 1)
        y += 10
        self.screen.blit(self.small.render("EVENT LOG", True, COLOR_DIM), (x, y))
        y += 16
        for text, color in self.log:
            self.screen.blit(self.small.render(text, True, color), (x, y))
            y += 15

    def draw_sandbox_hud(self):
        x = SANDBOX_RECT.left + 12
        y = SANDBOX_RECT.top + 10
        assists = "ON" if self.sandbox.assists else "OFF"
        mode = "KEYBOARD" if self.keyboard_mode else "GESTURE"
        lines = [
            (f"control {mode}   assists {assists}   falls {self.sandbox.falls}",
             COLOR_WARN if self.keyboard_mode else COLOR_TEXT),
            (f"flicks {self.counts['flick']}  jump {self.counts['jump']}  "
             f"double {self.counts['double']}  dropped {self.counts['dropped_jump']}  "
             f"fire {self.counts['pinch']}", COLOR_DIM),
            ("[C]alibrate [T]assists [K]eyboard [H]and-swap [R]eset [Esc]quit",
             COLOR_DIM),
        ]
        for text, color in lines:
            self.screen.blit(self.small.render(text, True, color), (x, y))
            y += 16

    def summary(self):
        infer = [v * 1000 for v in self.tracker.lat_infer]
        e2e = [v * 1000 for v in self.lat_e2e]
        print("\n=== spike summary ===")
        for name, vals in (("capture -> landmarks", infer), ("capture -> game loop", e2e)):
            if vals:
                print(f"{name:<22} mean {statistics.fmean(vals):6.1f}ms  "
                      f"p50 {pct(vals, 0.5):6.1f}ms  p95 {pct(vals, 0.95):6.1f}ms  "
                      f"n={len(vals)}")
            else:
                print(f"{name:<22} no samples")
        print(f"tracker fps            {self.tracker.tracker_fps():.1f}")
        print(f"events                 flicks={self.counts['flick']} "
              f"jump={self.counts['jump']} double={self.counts['double']} "
              f"dropped={self.counts['dropped_jump']} fire={self.counts['pinch']}")
        print(f"frames dropped         {self.tracker.dropped}")

    def close_log(self):
        if not self._csv:
            return
        self._csv.writerow({
            "kind": "summary",
            "t": round(time.perf_counter() - self.tracker.t0, 4),
            "note": (f"tracker_fps={self.tracker.tracker_fps():.2f};"
                     f"render_fps={self.clock.get_fps():.2f};"
                     f"dropped={self.tracker.dropped};"
                     f"flicks={self.counts['flick']};jump={self.counts['jump']};"
                     f"double={self.counts['double']};"
                     f"dropped_jump={self.counts['dropped_jump']};"
                     f"fire={self.counts['pinch']};falls={self.sandbox.falls}"),
        })
        self._csv_file.close()
        self._csv = None
        print(f"[Log] wrote {self.log_path}")


STEER_ZONE_DEFAULT = STEER_ZONE


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Two-hand control spike: a latency and gesture measurement tool.")
    p.add_argument("--log", nargs="?", const="auto", default=None, metavar="PATH",
                   help="write a CSV of every pose and event; omit PATH to "
                        "auto-name it spike-log-<timestamp>.csv")
    p.add_argument("--camera", type=int, default=0, metavar="N",
                   help="camera device index (default 0)")
    p.add_argument("--steer-zone", nargs=2, type=float, metavar=("LO", "HI"),
                   help="steering band as fractions of frame width "
                        "(default %.2f %.2f)" % STEER_ZONE_DEFAULT)
    p.add_argument("--swap-hands", action="store_true",
                   help="flip left/right hand detection (or press H while running)")
    return p.parse_args(argv)


if __name__ == "__main__":
    opts = parse_args(sys.argv[1:])
    path = opts.log
    if path == "auto":
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            time.strftime("spike-log-%Y%m%d-%H%M%S.csv"))
    if opts.steer_zone:
        STEER_ZONE = tuple(opts.steer_zone)      # noqa: F841 -- module-level rebind
        globals()["STEER_ZONE"] = STEER_ZONE
    spike = Spike(log_path=path, camera_id=opts.camera)
    if opts.swap_hands:
        spike.tracker.swap_handedness = True
    spike.run()
