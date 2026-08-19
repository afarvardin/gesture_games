"""Analyse a spike log and recommend threshold changes.

    python analyse_log.py spike-log-20260812-181500.csv

Reads the CSV written by `two_hand_spike.py --log` and reports what actually
happened: latency, hand-presence dropouts, how much of the steering range was
used, whether jump strength varied or saturated, near-miss flicks and pinches
that the thresholds swallowed, and suspected role swaps.

The near-miss counts are the most useful part. A gesture that fires reliably
tells you a threshold is not too high; only the motions that *tried* and failed
tell you it is. Those show up as local peaks in wrist velocity that fell just
short of FLICK_VEL_MIN, and as pinch gaps that dipped into the hysteresis band
without crossing PINCH_ON.
"""

import csv
import statistics
import sys
from collections import Counter

# A local peak in wrist velocity must be this far from the previous one to count
# as a separate attempt -- vel_up is windowed, so one flick makes a plateau.
CAM_W_ASSUMED = 640         # frame width assumed when reporting neutral position
PEAK_SEPARATION = 0.20      # s
PEAK_FLOOR = 1.5            # span/s, below this it's just hand tremor
TELEPORT_PX_S = 2500.0      # px/s a role cannot exceed without changing hands
                            # (a fast hand sweep measures ~1500 px/s)
SPAN_COLLAPSE = 0.65        # fraction of median span below which landmarks are suspect


def pct(vals, q):
    vals = sorted(vals)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = min(max(int(round(q * (len(vals) - 1))), 0), len(vals) - 1)
    return vals[idx]


def num(row, key, default=None):
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def stat_line(label, vals, unit="ms", width=22, dp=1):
    if not vals:
        return f"  {label:<{width}} no samples"
    return (f"  {label:<{width}} mean {statistics.fmean(vals):6.{dp}f}  "
            f"p50 {pct(vals, .5):6.{dp}f}  p95 {pct(vals, .95):6.{dp}f}  "
            f"max {max(vals):6.{dp}f}  {unit}  n={len(vals)}")


def histogram(vals, edges, width=34):
    """Text histogram over explicit bucket edges."""
    if not vals:
        return ["    (no samples)"]
    counts = [0] * (len(edges) + 1)
    for v in vals:
        placed = False
        for i, e in enumerate(edges):
            if v < e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    peak = max(counts) or 1
    labels = ([f"< {edges[0]:g}"]
              + [f"{edges[i]:g}-{edges[i+1]:g}" for i in range(len(edges) - 1)]
              + [f">= {edges[-1]:g}"])
    out = []
    for label, c in zip(labels, counts):
        bar = "#" * int(round(width * c / peak))
        share = 100.0 * c / len(vals)
        out.append(f"    {label:>12}  {bar:<{width}} {c:5d}  {share:5.1f}%")
    return out


def find_peaks(samples, floor=PEAK_FLOOR, separation=PEAK_SEPARATION):
    """Local maxima in a (t, value) series, thinned by minimum separation."""
    peaks = []
    for i in range(1, len(samples) - 1):
        t, v = samples[i]
        if v < floor:
            continue
        if v >= samples[i - 1][1] and v > samples[i + 1][1]:
            if peaks and t - peaks[-1][0] < separation:
                if v > peaks[-1][1]:
                    peaks[-1] = (t, v)      # keep the taller of the pair
                continue
            peaks.append((t, v))
    return peaks


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    kinds = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r)
    return rows, kinds


def parse_tuning(kinds):
    meta = kinds.get("meta", [])
    if not meta:
        return {}
    out = {}
    for part in (meta[0].get("note") or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def report(path):
    rows, kinds = load(path)
    poses = kinds.get("pose", [])
    flicks = kinds.get("flick", [])
    pinches = kinds.get("pinch", [])
    throws = kinds.get("throw", [])
    markers = kinds.get("marker", [])
    tuning = parse_tuning(kinds)

    if not poses:
        print(f"{path}: no pose rows -- was the camera working?")
        return

    print(f"\n=== {path} ===")

    # ---------- session ----------
    times = [num(r, "t", 0.0) for r in poses]
    dur = times[-1] - times[0]
    gaps = [b - a for a, b in zip(times, times[1:])]
    print("\nSESSION")
    print(f"  duration            {dur:6.1f} s")
    print(f"  poses               {len(poses)}  ({len(poses)/dur if dur else 0:.1f}/s)")
    if gaps:
        stalls = [g for g in gaps if g > 0.1]
        print(f"  pose interval       p50 {pct(gaps,.5)*1000:5.1f}ms  "
              f"p95 {pct(gaps,.95)*1000:5.1f}ms  max {max(gaps)*1000:6.1f}ms")
        if stalls:
            print(f"  stalls >100ms       {len(stalls)}  (worst {max(stalls)*1000:.0f}ms)")
    for m in kinds.get("summary", []):
        print(f"  reported            {m.get('note','')}")
    for m in markers:
        print(f"  marker  t={num(m,'t',0):6.1f}s  {m.get('note','')}")

    # ---------- latency ----------
    print("\nLATENCY")
    print(stat_line("capture -> landmarks", [num(r, "lat_ms", 0.0) for r in poses]))
    ev_lat = [num(r, "lat_ms", 0.0) for r in flicks + pinches + throws]
    print(stat_line("event -> game loop", ev_lat))
    print("  (excludes sensor+USB time before read() returns, ~20-50ms more)")

    # ---------- hand presence ----------
    steer = [int(num(r, "steer_seen", 0)) for r in poses]
    fire = [int(num(r, "fire_seen", 0)) for r in poses]
    both = sum(1 for s, f in zip(steer, fire) if s and f)
    print("\nHAND PRESENCE")
    print(f"  steer hand          {100.0*sum(steer)/len(poses):5.1f}% of poses")
    print(f"  fire hand           {100.0*sum(fire)/len(poses):5.1f}% of poses")
    print(f"  both together       {100.0*both/len(poses):5.1f}% of poses")
    for name, seen in (("steer", steer), ("fire", fire)):
        episodes, run_start = [], None
        for t, v in zip(times, seen):
            if not v and run_start is None:
                run_start = t
            elif v and run_start is not None:
                episodes.append(t - run_start)
                run_start = None
        if run_start is not None:
            episodes.append(times[-1] - run_start)
        real = [e for e in episodes if e > 0.1]
        if real:
            print(f"  {name:<5} dropouts      {len(real)}  "
                  f"total {sum(real):.1f}s  longest {max(real):.2f}s")
        else:
            print(f"  {name:<5} dropouts      none over 100ms")

    # ---------- steering ----------
    analog = [num(r, "analog", 0.0) for r in poses if int(num(r, "steer_seen", 0))]
    print("\nSTEERING")
    if analog:
        mags = [abs(a) for a in analog]
        idle = sum(1 for a in mags if a == 0)
        run_th = tuning.get("RUN_THRESHOLD", 0.60)
        walking = sum(1 for a in mags if 0 < a <= run_th)
        running = sum(1 for a in mags if a > run_th)
        sat = sum(1 for a in mags if a >= 0.99)
        n = len(mags)
        print(f"  idle / walk / run   {100.0*idle/n:4.1f}% / {100.0*walking/n:4.1f}% / "
              f"{100.0*running/n:4.1f}%")
        print(f"  saturated (|a|>=1)  {100.0*sat/n:4.1f}%   max reached {max(mags):.2f}")
        print(f"  left / right        {100.0*sum(1 for a in analog if a<0)/n:4.1f}% / "
              f"{100.0*sum(1 for a in analog if a>0)/n:4.1f}%")
        print("  |analog| distribution:")
        for line in histogram(mags, [0.001, 0.2, 0.4, 0.6, 0.8, 0.99]):
            print(line)
        # Steering origin health. A badly placed origin is the most damaging
        # failure in this whole pipeline and it is invisible in the analog
        # distribution alone -- it looks like a player who just likes going right.
        nx = [num(r, "neutral_x") for r in poses if num(r, "neutral_x") is not None]
        rl = [num(r, "reach_left") for r in poses if num(r, "reach_left") is not None]
        rr = [num(r, "reach_right") for r in poses if num(r, "reach_right") is not None]
        if nx:
            print(f"  neutral x           p50 {pct(nx,.5):5.0f}px  "
                  f"range {min(nx):.0f}-{max(nx):.0f}px of a {CAM_W_ASSUMED}px frame")
        if rl and rr:
            print(f"  reach (x full lean) left p50 {pct(rl,.5):.2f}   "
                  f"right p50 {pct(rr,.5):.2f}")
        if analog:
            bias = statistics.fmean(analog)
            print(f"  mean signed analog  {bias:+.3f}  "
                  f"(a persistent offset means the origin is off, not the player)")
        duck = sum(1 for r in poses if int(num(r, "duck", 0)))
        print(f"  duck                {100.0*duck/len(poses):4.1f}% of poses")
        spans = [num(r, "span", 0.0) for r in poses if num(r, "span")]
        if spans:
            print(f"  hand span           p50 {pct(spans,.5):5.0f}px  "
                  f"range {min(spans):.0f}-{max(spans):.0f}px "
                  f"(depth change {max(spans)/max(min(spans),1):.1f}x)")
    else:
        print("  steer hand never seen")

    # ---------- flicks ----------
    v_min = tuning.get("FLICK_VEL_MIN", 4.0)
    v_max = tuning.get("FLICK_VEL_MAX", 22.0)
    print("\nJUMP FLICKS")
    print(f"  fired               {len(flicks)}")
    if flicks:
        vels = [num(r, "vel", 0.0) for r in flicks]
        strengths = [num(r, "strength", 0.0) for r in flicks]
        print(stat_line("  flick velocity", vels, "span/s"))
        print(stat_line("  jump strength", strengths, "     ", dp=2))
        sat = sum(1 for s in strengths if s >= 0.995)
        floor = sum(1 for s in strengths if s <= 0.455)
        print(f"  at full height      {sat}/{len(strengths)}  "
              f"({100.0*sat/len(strengths):.0f}%)")
        print(f"  at minimum height   {floor}/{len(strengths)}")
        print(f"  outcomes            {dict(Counter(r.get('outcome','') for r in flicks))}")
        print("  velocity distribution:")
        for line in histogram(vels, [v_min, v_min * 2, v_max * .5, v_max, v_max * 1.5]):
            print(line)

    # Attempts that fell short: local velocity peaks below the trigger.
    vel_series = [(num(r, "t", 0.0), num(r, "vel_up", 0.0)) for r in poses
                  if int(num(r, "steer_seen", 0))]
    peaks = find_peaks(vel_series)
    near = [(t, v) for t, v in peaks if 0.5 * v_min <= v < v_min]
    fired = [(t, v) for t, v in peaks if v >= v_min]
    print(f"  velocity peaks       {len(fired)} above threshold, "
          f"{len(near)} near-misses in [{0.5*v_min:.1f}, {v_min:.1f})")
    if near:
        print(f"  near-miss peaks     p50 {pct([v for _, v in near], .5):.1f}  "
              f"max {max(v for _, v in near):.1f} span/s")

    # ---------- pinches ----------
    # Pinch thresholds are relative to a rolling open-hand baseline, so the
    # near-miss band has to be computed per-pose rather than as fixed numbers.
    drop = tuning.get("PINCH_DROP")
    release = tuning.get("PINCH_RELEASE", 0.80)
    b_pct = tuning.get("PINCH_BASELINE_PCT", 0.80)
    b_min = tuning.get("PINCH_BASELINE_MIN", 0.35)
    relative = drop is not None
    p_on = tuning.get("PINCH_ON", 0.38)          # legacy absolute logs
    p_off = tuning.get("PINCH_OFF", 0.55)
    gaps_v = [num(r, "pinch_gap") for r in poses if num(r, "pinch_gap") is not None]
    if throws:
        # Throw mode: the fire hand fires on a fast sweep, so the useful stats are
        # velocity and rate, not posture.
        print("\nFIRE THROWS")
        tv = [num(r, "vel", 0.0) for r in throws]
        print(f"  fired               {len(throws)}  "
              f"({len(throws) / max(dur, 1e-9) * 60:.1f}/min)")
        print(stat_line("  throw velocity", tv, "span/s", dp=2))
        tmin = tuning.get("THROW_VEL_MIN", 3.0)
        print("  velocity distribution:")
        for line in histogram(tv, [tmin, tmin * 1.5, tmin * 2.5, tmin * 5]):
            print(line)
        if len(throws) > 1:
            ft = [num(r, "t", 0.0) for r in throws]
            iv = [b - a for a, b in zip(ft, ft[1:])]
            print(f"  inter-throw gap     p50 {pct(iv,.5):.2f}s  min {min(iv):.2f}s")
        # Near-misses need the fire hand's own velocity, which poses do not carry
        # in throw mode, so say so rather than implying zero.
        low = [v for v in tv if v < tmin * 1.25]
        if low:
            print(f"  barely cleared      {len(low)}/{len(throws)} throws were within "
                  f"25% of THROW_VEL_MIN={tmin:g} -- gentler intended throws would "
                  f"have been swallowed")

    print("\nFIRE PINCHES" if not throws else "\nPINCH (unused in throw mode)")
    print(f"  fired               {len(pinches)}")
    if len(pinches) > 1:
        ft = [num(r, "t", 0.0) for r in pinches]
        iv = [b - a for a, b in zip(ft, ft[1:])]
        print(f"  inter-fire interval p50 {pct(iv,.5):.2f}s  min {min(iv):.2f}s")
    if gaps_v:
        held = sum(1 for r in poses if int(num(r, "pinched", 0)))
        print(f"  pinched             {100.0*held/len(poses):4.1f}% of poses")
        print(f"  gap                 p05 {pct(gaps_v,.05):.2f}  p50 {pct(gaps_v,.5):.2f}  "
              f"p80 {pct(gaps_v,.8):.2f}  min {min(gaps_v):.2f}")
        print("  gap distribution:")
        for line in histogram(gaps_v, [0.2, 0.4, 0.6, 0.8, 1.1]):
            print(line)
        # A dip into the hysteresis band that never crossed the fire threshold is
        # a pinch the player probably meant. With a relative threshold the band
        # moves, so walk the rolling baseline alongside the gaps.
        if relative:
            from collections import deque as _dq
            win = _dq(maxlen=int(tuning.get("PINCH_BASELINE_N", 90)))
            bands = []
            for g in gaps_v:
                win.append(g)
                b = pct(list(win), b_pct)
                bands.append((drop * b, release * b) if b >= b_min else (None, None))
            dips, in_band, band_min = 0, False, None
            for g, (on, off) in zip(gaps_v, bands):
                if on is None:
                    in_band, band_min = False, None
                elif g < on:
                    in_band, band_min = False, None
                elif g < off:
                    if not in_band:
                        in_band, band_min = True, g
                    band_min = min(band_min, g)
                else:
                    if in_band and band_min is not None:
                        dips += 1
                    in_band, band_min = False, None
            base_now = pct(gaps_v, b_pct)
            print(f"  open baseline       p{b_pct*100:.0f} of gaps = {base_now:.2f}  "
                  f"-> fires below {drop*base_now:.2f}, releases above "
                  f"{release*base_now:.2f}")
            print(f"  near-miss pinches   {dips}  (dipped into the hysteresis band "
                  f"without firing)")
            dips_done = True
        else:
            dips_done = False
        dips2, in_band, band_min = 0, False, None
        for g in ([] if relative else gaps_v):
            if g < p_on:
                in_band = False
                band_min = None
            elif g < p_off:
                if not in_band:
                    in_band, band_min = True, g
                band_min = min(band_min, g)
            else:
                if in_band and band_min is not None:
                    dips2 += 1
                in_band, band_min = False, None
        if not relative:
            print(f"  near-miss pinches   {dips2}  (dipped into [{p_on:g},{p_off:g}) "
                  f"without firing)")
    else:
        print("  fire hand never seen")

    # ---------- role stability ----------
    #
    # A role is only proven to have jumped bodies when its position moves faster
    # than an arm can. Comparing "did the cross-assignment explain the new
    # positions better" does NOT work: after a genuine crossing that was handled
    # correctly, each role IS nearer to where the other one was, so that test
    # flags every successful crossing as a failure. Implied speed doesn't.
    print("\nROLE STABILITY")
    tracked = [(num(r, "t", 0.0), num(r, "steer_x"), num(r, "fire_x"))
               for r in poses
               if num(r, "steer_x") is not None and num(r, "fire_x") is not None]
    crossings, teleports = 0, []
    for (t0, s0, f0), (t1, s1, f1) in zip(tracked, tracked[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > 0.25:
            continue                       # tracking gap, not a frame-to-frame move
        if (s0 - f0) * (s1 - f1) < 0:
            crossings += 1
        for role, before, after in (("STEER", s0, s1), ("FIRE", f0, f1)):
            speed = abs(after - before) / dt
            if speed > TELEPORT_PX_S:
                teleports.append((t1, role, speed))
    print(f"  frames with both    {len(tracked)}")
    print(f"  hands crossed over  {crossings} time(s)")
    print(f"  role teleports      {len(teleports)}  "
          f"(a role moving faster than {TELEPORT_PX_S:.0f} px/s -- it changed hands)")
    for t, role, speed in teleports[:8]:
        print(f"      t={t:7.2f}s  {role} jumped at {speed:.0f} px/s")

    # Long dropouts are the other way roles change hands: once a track expires,
    # identity has to be guessed from scratch.
    memory = tuning.get("ROLE_MEMORY")
    assumed = memory is None
    if assumed:
        memory = 0.5        # logs written before ROLE_MEMORY was stamped in meta
    expiries = []
    for name in ("steer", "fire"):
        start = None
        for r in poses:
            t, seen = num(r, "t", 0.0), int(num(r, f"{name}_seen", 0))
            if not seen and start is None:
                start = t
            elif seen and start is not None:
                expiries.append((name, start, t - start))
                start = None
    long_gaps = [e for e in expiries if memory is None or e[2] > memory]
    if memory is not None:
        print(f"  tracks expired      {len(long_gaps)} dropout(s) longer than "
              f"ROLE_MEMORY={memory:g}s{' (assumed)' if assumed else ''}"
              f" -- identity re-guessed on return")
        for name, t, dur in sorted(long_gaps, key=lambda e: -e[2])[:6]:
            print(f"      t={t:7.2f}s  {name} gone {dur:.1f}s")
    if crossings and not teleports:
        print("  -> crossings survived without a teleport: prediction matching held")

    # ---------- recommendations ----------
    print("\nRECOMMENDATIONS")
    recs = []
    if flicks:
        strengths = [num(r, "strength", 0.0) for r in flicks]
        vels = [num(r, "vel", 0.0) for r in flicks]
        # Judge the spread on percentiles. min/max is worthless here: one flick
        # off a collapsed hand span reads as 300 span/s and makes a stuck-at-the-
        # floor strength distribution look like it covered the whole range.
        spread = pct(strengths, .9) - pct(strengths, .1)
        sat_share = sum(1 for s in strengths if s >= 0.995) / len(strengths)
        if spread < 0.2:
            recs.append(f"jump height barely varied (p10-p90 = {pct(strengths,.1):.2f}-"
                        f"{pct(strengths,.9):.2f} of {spread:.2f} spread) -- flicks ran "
                        f"{pct(vels,.1):.1f}-{pct(vels,.9):.1f} span/s, so lower "
                        f"FLICK_VEL_MAX from {v_max:g} to about {pct(vels,.9):.0f}.")
        elif sat_share > 0.5:
            recs.append(f"{100*sat_share:.0f}% of jumps hit full height -- raise "
                        f"FLICK_VEL_MAX from {v_max:g} toward {pct(vels,.9):.0f} "
                        f"(p90 of observed flick speed) to restore variable height.")
        elif sat_share < 0.05 and pct(strengths, .9) < 0.85:
            recs.append(f"no jump reached full height (max {max(strengths):.2f}) -- lower "
                        f"FLICK_VEL_MAX from {v_max:g} toward {max(vels):.0f}.")
        else:
            recs.append(f"jump height varied properly (p10-p90 {pct(strengths,.1):.2f}-"
                        f"{pct(strengths,.9):.2f}); FLICK_VEL_MAX={v_max:g} looks right.")
        implausible = [v for v in vels if v > 3 * pct(vels, .9)]
        if implausible:
            recs.append(f"{len(implausible)} flick(s) reported an impossible speed (up to "
                        f"{max(implausible):.0f} span/s). Not motion -- the hand span is "
                        f"the velocity denominator, and it is unreliable both on the first "
                        f"poses after a hand is re-acquired and when landmarks briefly "
                        f"degrade. Median-filter the span AND gate jumps during warm-up.")
    if throws:
        tv = [num(r, "vel", 0.0) for r in throws]
        tmin = tuning.get("THROW_VEL_MIN", 3.0)
        if tv and pct(tv, .5) < tmin * 1.3:
            recs.append(f"throws clustered just above THROW_VEL_MIN={tmin:g} "
                        f"(median {pct(tv,.5):.1f} span/s) -- gentler throws the player "
                        f"meant were probably swallowed; try {tmin*0.75:.1f}.")
    if len(near) >= max(2, 0.3 * max(len(flicks), 1)):
        recs.append(f"{len(near)} flick attempts fell short of FLICK_VEL_MIN={v_min:g} "
                    f"(peaks up to {max(v for _, v in near):.1f}) -- consider lowering it "
                    f"to ~{max(2.5, pct([v for _, v in near], .5)):.1f}.")
    dropped = [r for r in flicks if r.get("outcome") in ("buffered", "ignored")]
    if flicks and len(dropped) > 0.25 * len(flicks):
        recs.append(f"{len(dropped)}/{len(flicks)} flicks landed with no jump available "
                    f"(mid-air, budget spent) -- either a third jump is wanted, or "
                    f"flicks are firing unintentionally.")
    if gaps_v and not relative and min(gaps_v) > p_on:
        recs.append(f"the pinch never closed below PINCH_ON={p_on:g} "
                    f"(closest {min(gaps_v):.2f}) -- raise PINCH_ON.")
    if gaps_v and relative and not throws:
        held = sum(1 for r in poses if int(num(r, "pinched", 0))) / len(poses)
        if held > 0.45:
            recs.append(f"the fire hand read as pinched {100*held:.0f}% of the time -- it "
                        f"rests too closed for the release to trigger. Lower PINCH_DROP or "
                        f"ask the player to relax the hand open between shots.")
    if analog:
        mags = [abs(a) for a in analog]
        if max(mags) < 0.75:
            recs.append(f"steering never passed {max(mags):.2f} of full deflection -- "
                        f"lower STEER_FULL_LEAN from "
                        f"{tuning.get('STEER_FULL_LEAN', 2.2):g} so running is reachable.")
        elif sum(1 for a in mags if a >= 0.99) > 0.35 * len(mags):
            recs.append("steering sat at full deflection over a third of the time -- "
                        "raise STEER_FULL_LEAN for finer control.")
    if teleports:
        recs.append(f"{len(teleports)} role teleport(s): a role changed hands mid-frame. "
                    f"Match roles PARTIALLY -- never discard a good match because the "
                    f"other hand failed its radius check.")
    if memory is not None and long_gaps:
        recs.append(f"{len(long_gaps)} dropout(s) outlasted ROLE_MEMORY={memory:g}s "
                    f"(worst {max(e[2] for e in long_gaps):.1f}s), so roles were re-guessed "
                    f"on return. Raise ROLE_MEMORY and fall back to each role's last known "
                    f"side rather than screen side.")
    spans_all = [num(r, "span") for r in poses if num(r, "span") is not None]
    if spans_all:
        med = pct(spans_all, .5)
        collapsed = [v for v in spans_all if v < SPAN_COLLAPSE * med]
        if len(collapsed) > 0.02 * len(spans_all):
            recs.append(f"hand span collapsed below {SPAN_COLLAPSE:g}x its median on "
                        f"{100.0*len(collapsed)/len(spans_all):.0f}% of poses "
                        f"(min {min(spans_all):.0f}px vs median {med:.0f}px). Span is the "
                        f"denominator for every threshold, so this leaks into both steering "
                        f"and flicks -- median-filter it over ~15 poses.")
    # Origin health checks -- these would have caught the hard-right bias.
    nx_all = [num(r, "neutral_x") for r in poses if num(r, "neutral_x") is not None]
    rl_all = [num(r, "reach_left") for r in poses if num(r, "reach_left") is not None]
    rr_all = [num(r, "reach_right") for r in poses if num(r, "reach_right") is not None]
    if rl_all and rr_all:
        worst = min(pct(rl_all, .5), pct(rr_all, .5))
        if worst < 0.7:
            side = "left" if pct(rl_all, .5) < pct(rr_all, .5) else "right"
            recs.append(f"only {worst:.2f}x of a full lean was available to the {side} "
                        f"(neutral sat at x={pct(nx_all,.5):.0f}px). Sit more centred in "
                        f"frame, or press C to re-learn the origin while centred.")
    if analog:
        bias = statistics.fmean(analog)
        if abs(bias) > 0.25:
            recs.append(f"mean signed analog is {bias:+.2f} -- the axis rested "
                        f"{'right' if bias > 0 else 'left'} of centre for the whole "
                        f"session. That is a misplaced steering origin, not a "
                        f"preference; the origin is now learned from a steady hand "
                        f"away from the frame edges.")
    lat = [num(r, "lat_ms", 0.0) for r in poses]
    if lat and pct(lat, .95) > 40:
        recs.append(f"landmark p95 is {pct(lat,.95):.0f}ms -- drop the capture "
                    f"resolution or set model_complexity=0.")
    if not recs:
        recs.append("nothing stands out; thresholds look reasonable for this session.")
    for r in recs:
        print(f"  - {r}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for p in sys.argv[1:]:
        report(p)
