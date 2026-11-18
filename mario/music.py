"""Original chiptune music for the platformer, synthesized at startup.

Nothing here is a recording and nothing is borrowed: every tune is composed below
as data (a key, a chord progression and a few melodic shapes) and rendered by a
tiny synthesizer -- square, pulse and triangle voices plus noise drums, the same
palette an 8-bit console had. So there are no audio files to ship, and a new
tune is a few lines.

To use your own music instead, put `<track>.ogg` / `.wav` / `.mp3` in a `music/`
folder next to this file (names are in TRACK_FOR_THEME / TRACKS below, plus the
jingles `dead`, `clear`, `gameover` and `win`). A file there wins over the
built-in tune of the same name.

How a tune is built
-------------------
* A tune is 8 bars of 4/4 in eighth notes, played twice with different melodic
  shapes the second time, so a level's loop is 16 bars rather than 8.
* The key and mode set the mood (major is bright, phrygian is menacing...).
* A chord progression gives one chord per two bars. The bass follows the chord
  roots, and the melody is written *relative to the chord*: a shape is a list of
  scale steps from the chord's root (0 = root, 2 = third, 4 = fifth, 7 = octave),
  so the strong beats always land on chord tones and the tune stays consonant
  whatever the mode. Variety comes from which shapes, mode, tempo, timbre and
  drums a tune uses.
"""
import os
import threading

import numpy as np

SR = 22050
HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.path.join(HERE, "music")

MODES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "harmonic": [0, 2, 3, 5, 7, 8, 11],
}

# Melodic shapes: two bars (16 eighth-note slots) of scale steps from the chord
# root. "h" holds the previous note, "r" is a rest, "m2" is two steps below.
SHAPES = {
    "bounce": "4 r 4 7 4 r 2 r  0 r 2 4 2 r 0 r",
    "run":    "0 2 4 7 4 2 4 7  9 7 4 2 4 2 0 r",
    "arch":   "0 h 2 h 4 h 7 h  9 h 7 h 4 h 2 h",
    "sync":   "4 h r 4 7 h 4 2  r 2 4 h 2 0 h r",
    "sigh":   "7 h h 4 h 2 h h  0 h h r 2 h 4 h",
    "arp":    "0 4 7 4 2 4 9 4  0 4 7 4 2 4 7 4",
    "stab":   "4 r r 4 r r 4 r  7 r r 7 r 4 r 2",
    "climb":  "0 1 2 3 4 5 6 7  7 6 5 4 3 2 1 0",
    "dotted": "7 h h 4 2 h h 4  7 h h 9 7 h 4 h",
}

# name -> tune. `lead`/`lead2` are the four shapes (one per chord) for the first
# and second pass; `bass` is oct | walk | pulse | long; `drums` is drive | soft |
# none; `arp` adds an arpeggio under the lead.
TRACKS = {
    # Worlds 1-4 and the general-purpose levels
    "meadow": dict(mode="major", root=60, bpm=150, prog=[0, 4, 5, 3], duty=0.5,
                   lead=["bounce", "run", "bounce", "arch"],
                   lead2=["run", "sync", "run", "bounce"], bass="oct",
                   drums="drive", arp=False),
    "cave": dict(mode="dorian", root=55, bpm=108, prog=[0, 3, 0, 4], duty=0.125,
                 lead=["bounce", "sigh", "bounce", "dotted"],
                 lead2=["sigh", "bounce", "dotted", "sigh"], bass="walk",
                 drums="none", arp=False),
    "tide": dict(mode="lydian", root=60, bpm=96, prog=[0, 3, 4, 0], duty=0.5,
                 lead=["arp", "arch", "arp", "sigh"],
                 lead2=["arch", "arp", "sigh", "arp"], bass="long",
                 drums="none", arp=True),
    "night": dict(mode="dorian", root=53, bpm=118, prog=[0, 3, 6, 4], duty=0.5,
                  lead=["sync", "arch", "dotted", "sigh"],
                  lead2=["dotted", "sync", "arch", "sync"], bass="walk",
                  drums="soft", arp=False),
    "castle": dict(mode="minor", root=50, bpm=126, prog=[0, 3, 5, 4], duty=0.25,
                   lead=["stab", "sigh", "sync", "stab"],
                   lead2=["sync", "stab", "stab", "sigh"], bass="pulse",
                   drums="soft", arp=False),
    # World 5, the Amber Wilds
    "amber": dict(mode="mixolydian", root=57, bpm=132, prog=[0, 6, 3, 0],
                  duty=0.25, lead=["sync", "arch", "sync", "sigh"],
                  lead2=["bounce", "sync", "arch", "sync"], bass="oct",
                  drums="soft", arp=True),
    "harvest": dict(mode="dorian", root=62, bpm=112, prog=[0, 3, 0, 4], duty=0.5,
                    lead=["dotted", "sigh", "dotted", "arch"],
                    lead2=["sigh", "dotted", "arch", "dotted"], bass="long",
                    drums="none", arp=True),
    # World 6, the Frostpeak Highlands
    "frost": dict(mode="lydian", root=59, bpm=138, prog=[0, 1, 0, 4], duty=0.125,
                  lead=["run", "arp", "run", "arch"],
                  lead2=["arp", "run", "arch", "run"], bass="oct",
                  drums="soft", arp=False),
    "aurora": dict(mode="major", root=55, bpm=92, prog=[0, 5, 3, 4], duty=0.5,
                   lead=["sigh", "arch", "sigh", "dotted"],
                   lead2=["arch", "sigh", "dotted", "sigh"], bass="long",
                   drums="none", arp=True),
    # World 7, the Storm Coast
    "storm": dict(mode="minor", root=57, bpm=156, prog=[0, 5, 6, 4], duty=0.25,
                  lead=["stab", "run", "stab", "climb"],
                  lead2=["run", "stab", "climb", "stab"], bass="oct",
                  drums="drive", arp=False),
    # World 8, the Ashen Kingdom
    "ashen": dict(mode="phrygian", root=52, bpm=124, prog=[0, 1, 0, 6], duty=0.5,
                  lead=["stab", "sigh", "stab", "sync"],
                  lead2=["sync", "stab", "sigh", "stab"], bass="pulse",
                  drums="drive", arp=True),
    # The World 5-8 castles
    "keep": dict(mode="harmonic", root=52, bpm=144, prog=[0, 5, 3, 4], duty=0.25,
                 lead=["climb", "stab", "run", "sync"],
                 lead2=["stab", "climb", "sync", "run"], bass="oct",
                 drums="drive", arp=False),
    # Boss arenas, and the bonus rooms
    "boss": dict(mode="phrygian", root=50, bpm=172, prog=[0, 1, 0, 6], duty=0.25,
                 lead=["stab", "climb", "stab", "run"],
                 lead2=["climb", "stab", "run", "stab"], bass="oct",
                 drums="drive", arp=True),
    "bonus": dict(mode="major", root=64, bpm=168, prog=[0, 3, 4, 0], duty=0.25,
                  lead=["arp", "bounce", "arp", "run"],
                  lead2=["bounce", "arp", "run", "arp"], bass="oct",
                  drums="none", arp=False),
}

# Short pieces that play once: (midi note, beats) lists, a bass line, and a tempo.
JINGLES = {
    "dead": dict(bpm=130, duty=0.5,
                 lead=[(72, .5), (71, .5), (70, .5), (69, .5), (67, .75), (60, .25),
                       (62, .5), (55, 1.5)],
                 bass=[(48, 2), (43, 2.5)]),
    "clear": dict(bpm=150, duty=0.5,
                  lead=[(60, .25), (64, .25), (67, .25), (72, .5), (71, .25),
                        (72, .25), (76, .5), (74, .25), (76, .25), (79, 1.75)],
                  bass=[(48, 1), (55, 1), (52, 1), (55, 2.5)]),
    "gameover": dict(bpm=80, duty=0.25,
                     lead=[(69, 1), (65, 1), (62, 1), (57, .5), (58, .5), (57, 2.5)],
                     bass=[(45, 2), (41, 2), (38, 3)]),
    "win": dict(bpm=140, duty=0.5,
                lead=[(60, .5), (64, .5), (67, .5), (72, 1), (71, .5), (72, .5),
                      (76, 1), (74, .5), (72, .5), (74, 1), (67, 1.5),
                      (65, .5), (69, .5), (72, .5), (77, 1), (76, .5), (77, .5),
                      (81, 1), (79, .5), (77, .5), (79, 1), (72, 1.5),
                      (72, .5), (76, .5), (79, .5), (84, 3)],
                bass=[(48, 2), (55, 2), (53, 2), (48, 2), (53, 2), (60, 2),
                      (55, 2), (48, 4)]),
}

# Which tune a theme plays. Themes come from levels.py.
TRACK_FOR_THEME = {
    "overworld": "meadow", "underground": "cave", "castle": "castle",
    "water": "tide", "night": "night", "autumn": "amber", "harvest": "harvest",
    "highlands": "frost", "aurora": "aurora", "storm": "storm", "ashen": "ashen",
    "keep": "keep",
}
ALL_LOOPS = tuple(TRACKS)


# ---------------------------------------------------------------- synthesis
def _freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def _semis(mode, degree):
    octave, d = divmod(degree, 7)
    return 12 * octave + MODES[mode][d]


def _wave(kind, freq, n, duty=0.5):
    phase = (np.arange(n) * (freq / SR)) % 1.0
    if kind == "tri":
        return 4.0 * np.abs(phase - 0.5) - 1.0
    return np.where(phase < duty, 1.0, -1.0)


def _env(n, attack=0.004, release=0.012, decay=7.0, sustain=0.55):
    t = np.arange(n) / SR
    e = (sustain + (1 - sustain) * np.exp(-t * decay)) * np.minimum(1.0, t / attack)
    rel = max(1, int(release * SR))
    if n > rel:
        e[-rel:] *= np.linspace(1.0, 0.0, rel)
    else:
        e *= np.linspace(1.0, 0.0, n)
    return e


def _add(buf, start, sig):
    end = min(len(buf), start + len(sig))
    if start < len(buf) and end > start:
        buf[start:end] += sig[:end - start]


def _kind_note(buf, start, n, midi, kind, vol, duty=0.5, **env):
    _add(buf, start, vol * _wave(kind, _freq(midi), n, duty) * _env(n, **env))


def _kick(buf, start, vol):
    n = int(0.13 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 38)
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 26)
    _add(buf, start, vol * sig)


def _snare(buf, start, vol, rng):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    sig = (rng.uniform(-1, 1, n) * np.exp(-t * 30)
           + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 40))
    _add(buf, start, vol * sig)


def _hat(buf, start, vol, rng):
    n = int(0.035 * SR)
    t = np.arange(n) / SR
    noise = np.diff(rng.uniform(-1, 1, n + 1))          # cheap high-pass
    _add(buf, start, vol * noise * np.exp(-t * 110))


def _parse(shape):
    toks = shape.split()
    out = []
    for t in toks:
        if t in ("h", "r"):
            out.append(t)
        else:
            out.append(int(t.replace("m", "-")))
    return out


def _to_int16(mono, peak=0.82):
    top = float(np.max(np.abs(mono))) or 1.0
    return (mono * (peak / top) * 32767).astype(np.int16)


def render_track(name):
    """One tune as a mono int16 array: 16 bars that loop with no click."""
    spec = TRACKS[name]
    mode, root, bpm = spec["mode"], spec["root"], spec["bpm"]
    slot = int(SR * 60.0 / bpm / 2)                 # one eighth note
    bars = 16
    total = bars * 8 * slot
    lead = np.zeros(total)
    bass = np.zeros(total)
    arp = np.zeros(total)
    drums = np.zeros(total)
    rng = np.random.default_rng(sum(map(ord, name)))
    prog = spec["prog"]

    for bar in range(bars):
        p = bar // 8                                   # pass: 0 or 1
        chord_idx = (bar % 8) // 2
        root_deg = prog[chord_idx]
        shapes = spec["lead2"] if p else spec["lead"]
        tokens = _parse(SHAPES[shapes[chord_idx]])
        half = (bar % 2) * 8                           # which bar of the shape
        bar0 = bar * 8 * slot

        # --- lead: walk the eighth-note tokens, merging holds into the note
        i = 0
        while i < 8:
            tok = tokens[half + i]
            if tok in ("h", "r"):
                i += 1
                continue
            length = 1
            while i + length < 8 and tokens[half + i + length] == "h":
                length += 1
            midi = root + _semis(mode, root_deg + tok)
            _kind_note(lead, bar0 + i * slot, length * slot, midi, "sq", 1.0,
                       duty=spec["duty"], decay=6.0)
            i += length

        # --- bass
        broot = root - 12 + _semis(mode, root_deg)
        pat = spec["bass"]
        if pat == "oct":
            for k in range(8):
                _kind_note(bass, bar0 + k * slot, slot, broot + (12 if k % 2 else 0),
                           "tri", 1.0, decay=4.0, sustain=0.7)
        elif pat == "walk":
            for k, step in enumerate((0, 4, 0, 2)):      # root, 5th, root, 3rd
                m = root - 12 + _semis(mode, root_deg + step)
                _kind_note(bass, bar0 + k * 2 * slot, 2 * slot, m, "tri", 1.0,
                           decay=3.0, sustain=0.7)
        elif pat == "pulse":
            for k in range(4):
                _kind_note(bass, bar0 + k * 2 * slot, 2 * slot - slot // 3, broot,
                           "tri", 1.0, decay=5.0, sustain=0.6)
        else:                                          # long: a whole bar
            _kind_note(bass, bar0, 8 * slot, broot, "tri", 1.0, decay=1.0,
                       sustain=0.85, release=0.05)

        # --- arpeggio: chord tones in eighths, quietly, in the lead's octave
        if spec["arp"]:
            for k in range(8):
                step = (0, 2, 4, 7, 4, 2, 4, 2)[k]
                midi = root + _semis(mode, root_deg + step)
                _kind_note(arp, bar0 + k * slot, slot, midi, "sq", 1.0, duty=0.125,
                           decay=14.0, sustain=0.2, release=0.006)

        # --- drums
        d = spec["drums"]
        if d == "drive":
            for k in (0, 4):
                _kick(drums, bar0 + k * slot, 1.0)
            if bar % 4 == 3:
                _kick(drums, bar0 + 7 * slot, 0.8)
            for k in (2, 6):
                _snare(drums, bar0 + k * slot, 0.55, rng)
            for k in range(8):
                _hat(drums, bar0 + k * slot, 0.30 if k % 2 else 0.18, rng)
        elif d == "soft":
            _kick(drums, bar0, 0.7)
            _kick(drums, bar0 + 4 * slot, 0.5)
            for k in (2, 6):
                _hat(drums, bar0 + k * slot, 0.30, rng)

    mix = 0.34 * lead + 0.42 * bass + 0.17 * arp + 0.30 * drums
    return _to_int16(mix)


def render_jingle(name):
    spec = JINGLES[name]
    beat = SR * 60.0 / spec["bpm"]
    total = int(max(sum(b for _, b in spec["lead"]), sum(b for _, b in spec["bass"]))
                * beat) + int(0.4 * SR)
    lead, bass = np.zeros(total), np.zeros(total)
    pos = 0.0
    for midi, beats in spec["lead"]:
        n = int(beats * beat)
        _kind_note(lead, int(pos), n, midi, "sq", 1.0, duty=spec["duty"], decay=4.0,
                   sustain=0.7, release=0.03)
        pos += beats * beat
    pos = 0.0
    for midi, beats in spec["bass"]:
        n = int(beats * beat)
        _kind_note(bass, int(pos), n, midi, "tri", 1.0, decay=2.0, sustain=0.8,
                   release=0.05)
        pos += beats * beat
    return _to_int16(0.5 * lead + 0.5 * bass)


def render(name):
    return render_track(name) if name in TRACKS else render_jingle(name)


def user_file(name):
    """A recording the player dropped into music/, if there is one."""
    for ext in (".ogg", ".wav", ".mp3"):
        path = os.path.join(MUSIC_DIR, name + ext)
        if os.path.exists(path):
            return path
    return None


# ------------------------------------------------------------------- player
class Music:
    """Plays one piece at a time. Safe with no audio device: everything still
    tracks what *would* be playing, so the game logic and tests behave the same."""

    VOLUME = 0.30
    _arrays = {}                    # name -> int16 mono, shared by every Music
    _arrays_lock = threading.Lock()

    def __init__(self, enabled=True):
        self.ok = False
        self.muted = False
        self.current = None         # the piece that is (or would be) playing
        self.looping = True
        self._sounds = {}
        self._channel = None
        if enabled and not os.environ.get("GESTURE_NO_AUDIO"):
            self._init_mixer()

    def _init_mixer(self):
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(SR, -16, 2, 512)
            pygame.mixer.set_num_channels(max(8, pygame.mixer.get_num_channels()))
            self._channel = pygame.mixer.Channel(0)
            self.ok = True
        except Exception as e:                       # no device, headless, ...
            print(f"[Music] audio unavailable ({e}); playing silently")

    # -- rendering is slow-ish, so it happens on a thread; sounds are made on the
    #    main thread, where pygame wants them.
    @classmethod
    def get_array(cls, name):
        with cls._arrays_lock:
            arr = cls._arrays.get(name)
        if arr is None:
            arr = render(name)
            with cls._arrays_lock:
                cls._arrays[name] = arr
        return arr

    _queued = set()

    @classmethod
    def prewarm(cls, names):
        """Render these in the background, in order, each only once per run."""
        with cls._arrays_lock:
            todo = [n for n in names if n not in cls._arrays and n not in cls._queued]
            cls._queued.update(todo)
        if not todo:
            return

        def work():
            for n in todo:
                try:
                    cls.get_array(n)
                except Exception as e:
                    print(f"[Music] could not render {n}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _sound(self, name):
        snd = self._sounds.get(name)
        if snd is None:
            import pygame
            path = user_file(name)
            if path:
                snd = pygame.mixer.Sound(path)
            else:
                mono = self.get_array(name)
                snd = pygame.sndarray.make_sound(np.ascontiguousarray(
                    np.stack([mono, mono], axis=1)))
            snd.set_volume(self.VOLUME)
            self._sounds[name] = snd
        return snd

    def _start(self, name):
        if not self.ok or self.muted or name is None:
            return
        try:
            self._channel.play(self._sound(name), loops=-1 if name in TRACKS else 0,
                               fade_ms=150)
        except Exception as e:
            print(f"[Music] could not play {name}: {e}")
            self.ok = False

    def play(self, name):
        """Switch to a piece. Asking for the one already playing does nothing."""
        if name == self.current:
            return
        self.current = name
        if self.ok:
            self._channel.stop()
        self._start(name)

    def stop(self):
        self.current = None
        if self.ok:
            self._channel.stop()

    def toggle_mute(self):
        self.muted = not self.muted
        if self.ok:
            if self.muted:
                self._channel.stop()
            else:
                self._start(self.current)
        return self.muted
