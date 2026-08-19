# Gesture Games

Simple webcam-controlled games built for one reason: to get kids up on their feet, waving their hands around, instead of hunched over a touchscreen or keyboard. There's no controller and no touch input — your hand *is* the controller. A webcam watches your hand through [MediaPipe](https://developers.google.com/mediapipe) hand tracking, and that motion drives the game in real time via [OpenCV](https://opencv.org/) and [Pygame](https://www.pygame.org/).

## Games

### 🍄 Mario (`mario/`)
A two-handed Super Mario Bros.-style platformer: **four worlds of four levels**,
shaped after the original Worlds 1–4, with mushrooms, pipes you can enter, koopas,
firebars, moving lifts and a maze castle. Your **left hand steers and jumps**, your
**right hand throws fire** — and which hand is which comes from hand *detection*, not
from which side of the frame you hold them.

Speed comes from where your steering hand sits inside a fixed band of the camera
frame, jumps are a quick upward flick (flick harder, jump higher), and fire is a
quick sweep of the other hand. Nothing needs calibrating. It is deliberately
forgiving, because ~50–80 ms of camera-to-screen latency cannot be removed.

**→ [`mario/README.md`](mario/README.md)** is the source of truth: the level list,
every command-line option, the controls, and how each threshold was measured from
logged play sessions. It also documents the two-hand latency spike (`./run.sh spike`)
the control layer came from.

### 🍉 Fruit Ninja (`fruit_ninja/`)
Fruit gets launched up the screen — swipe your hand through it to slice, and avoid the bombs. Your index fingertip is tracked frame-to-frame; the line between consecutive positions is the "blade," and anything it crosses gets sliced. Score is tracked, bombs cost you a life, and losing all lives ends the game (`R` to restart).

### 👻 Pac-Man (`pacman/`)
Classic Pac-Man maze chase, steered entirely by pointing your hand up/down/left/right in front of the camera. A live camera preview with an on-screen gesture indicator sits next to the maze so kids can see exactly how their hand is being read. Falls back to arrow keys if needed.

All three games mirror the camera feed so movement feels natural (move your hand right, things move right on screen) and require a working webcam.

## Requirements

- Python 3.12
- A webcam
- macOS users: grant Camera access to whichever app launches the game (Terminal/iTerm/VS Code) via **System Settings → Privacy & Security → Camera**, then fully quit and reopen that app.

Dependencies (see `requirements.txt`): `pygame`, `opencv-python-headless`, `mediapipe`, `numpy`, plus their transitive requirements.

## Running

The easiest way is the launcher script, which creates a virtual environment on first run and installs what it needs:

```bash
./run.sh mario          # Start the Gesture Platformer (also: platformer, plat)
./run.sh pacman         # Start Gesture Pac-Man
./run.sh fruit_ninja    # Start Gesture Fruit Ninja
./run.sh spike          # Two-hand control spike — a measuring tool, not a game
```

Mario takes options — starting world and level, keyboard-only mode, camera index,
steering band, session recording. They are listed in
[`mario/README.md`](mario/README.md#running-and-options); anything after the game
name is passed straight through:

```bash
./run.sh mario --world 3 --level 3   # jump to a specific level
./run.sh mario --no-camera           # keyboard only, no webcam needed
```

Alternatively, set up the environment yourself and run a game directly:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd mario && python platformer.py --world 1 --level 1
# or
cd pacman && python pacman.py
# or
cd fruit_ninja && python fruit_ninja.py
```

## Controls

- **Hand gestures** — the primary control for all three games (point, swipe or flick in front of the webcam)
- **R** — restart after game over
- **Esc** — quit

- **Arrow keys** — fallback for Pac-Man

Mario uses two hands and has its own gesture and key list —
see [`mario/README.md`](mario/README.md#controls).

## Project Layout

```
mario/         Gesture platformer
  platformer.py      The game: physics, entities, rendering, HUD
  levels.py          16 levels across 4 worlds, built by a validating builder
  two_hand_spike.py  Two-hand tracking and gesture detection (also runs standalone)
  analyse_log.py     Turns a recorded session into a tuning report
  test_*.py          250 headless checks — no camera or display needed
  README.md          How every threshold was measured, and what broke on the way
fruit_ninja/   Fruit-slicing game (fruit_ninja.py, image assets, MediaPipe hand model)
pacman/        Gesture-controlled Pac-Man (pacman.py)
requirements.txt
run.sh         Unified launcher (sets up venv, starts any of them)
```

The platformer loads its hand-tracking model from `fruit_ninja/hand_landmarker.task`,
so that file is needed even if you only play Mario.

## Tests

Mario and its control layer have headless test suites that need neither a camera nor
a display — including an autoplay bot that must finish all 16 levels on the weakest
jump the gesture layer can produce. See
[`mario/README.md`](mario/README.md#tests).
