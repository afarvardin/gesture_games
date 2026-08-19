#!/bin/bash
# Unified Launcher for Gesture Games (Pac-Man & Fruit Ninja)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

# Ensure virtual environment exists and has required packages
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating Python virtual environment in .venv..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    "$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip
    "$SCRIPT_DIR/.venv/bin/pip" install pygame opencv-python-headless "mediapipe<=0.10.14" numpy
fi

# Convert first parameter to lowercase for case-insensitive matching
GAME=$(echo "$1" | tr '[:upper:]' '[:lower:]')

case "$GAME" in
    pacman|pac-man|packman)
        echo "🎮 Starting Pac-Man with Camera Gesture Control..."
        cd "$SCRIPT_DIR/pacman" || exit 1
        exec "$SCRIPT_DIR/.venv/bin/python3" pacman.py
        ;;
    fruit_ninja|fruit-ninja|fruitninja|fruit)
        echo "🍉 Starting Fruit Ninja with Camera Gesture Control..."
        cd "$SCRIPT_DIR/fruit_ninja" || exit 1
        exec "$SCRIPT_DIR/.venv/bin/python3" fruit_ninja.py
        ;;
    platformer|mario|plat)
        echo "🍄 Starting the Gesture Platformer..."
        shift
        cd "$SCRIPT_DIR/mario" || exit 1
        exec "$SCRIPT_DIR/.venv/bin/python3" platformer.py "$@"
        ;;
    spike|two_hand)
        echo "🖐️🖐️ Starting the two-hand control spike (latency measurement tool)..."
        shift                       # pass any remaining flags (e.g. --log) through
        cd "$SCRIPT_DIR/mario" || exit 1
        exec "$SCRIPT_DIR/.venv/bin/python3" two_hand_spike.py "$@"
        ;;
    *)
        echo "Usage: ./run.sh [pacman | fruit_ninja | platformer | spike]"
        echo ""
        echo "Available games:"
        echo "  ./run.sh pacman       - Start Gesture Pac-Man"
        echo "  ./run.sh fruit_ninja  - Start Gesture Fruit Ninja"
        echo "  ./run.sh platformer   - Start the Gesture Platformer (two hands)"
        echo "  ./run.sh spike        - Two-hand control spike (not a game; see mario/README.md)"
        exit 1
        ;;
esac
