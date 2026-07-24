# Gesture Games

Simple webcam-controlled games built for one reason: to get kids up on their feet, waving their hands around, instead of hunched over a touchscreen or keyboard. There's no controller and no touch input — your hand *is* the controller. A webcam watches your hand through [MediaPipe](https://developers.google.com/mediapipe) hand tracking, and that motion drives the game in real time via [OpenCV](https://opencv.org/) and [Pygame](https://www.pygame.org/).

## Games

### 🍉 Fruit Ninja (`fruit_ninja/`)
Fruit gets launched up the screen — swipe your hand through it to slice, and avoid the bombs. Your index fingertip is tracked frame-to-frame; the line between consecutive positions is the "blade," and anything it crosses gets sliced. Score is tracked, bombs cost you a life, and losing all lives ends the game (`R` to restart).

### 👻 Pac-Man (`pacman/`)
Classic Pac-Man maze chase, steered entirely by pointing your hand up/down/left/right in front of the camera. A live camera preview with an on-screen gesture indicator sits next to the maze so kids can see exactly how their hand is being read. Falls back to arrow keys if needed.

Both games mirror the camera feed so movement feels natural (move your hand right, things move right on screen) and require a working webcam.

## Requirements

- Python 3.12
- A webcam
- macOS users: grant Camera access to whichever app launches the game (Terminal/iTerm/VS Code) via **System Settings → Privacy & Security → Camera**, then fully quit and reopen that app.

Dependencies (see `requirements.txt`): `pygame`, `opencv-python-headless`, `mediapipe`, `numpy`, plus their transitive requirements.

## Running

The easiest way is the launcher script, which creates a virtual environment on first run and installs what it needs:

```bash
./run.sh pacman        # Start Gesture Pac-Man
./run.sh fruit_ninja    # Start Gesture Fruit Ninja
```

Alternatively, set up the environment yourself and run a game directly:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd pacman && python pacman.py
# or
cd fruit_ninja && python fruit_ninja.py
```

## Controls

- **Hand gestures** — primary control for both games (point/swipe in front of the webcam)
- **Arrow keys** — fallback for Pac-Man
- **R** — restart after game over
- **Esc** — quit

## Project Layout

```
fruit_ninja/   Fruit-slicing game (fruit_ninja.py, image assets, MediaPipe hand model)
pacman/        Gesture-controlled Pac-Man (pacman.py)
requirements.txt
run.sh         Unified launcher (sets up venv, starts either game)
```
