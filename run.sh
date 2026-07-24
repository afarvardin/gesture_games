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
    *)
        echo "Usage: ./run.sh [pacman | fruit_ninja]"
        echo ""
        echo "Available games:"
        echo "  ./run.sh pacman       - Start Gesture Pac-Man"
        echo "  ./run.sh fruit_ninja  - Start Gesture Fruit Ninja"
        exit 1
        ;;
esac
