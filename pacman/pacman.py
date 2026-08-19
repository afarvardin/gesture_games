import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import os

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
FPS = 60
TILE_SIZE = 20
MAP_COLS = 28
MAP_ROWS = 31

GAME_WIDTH = MAP_COLS * TILE_SIZE  # 560 px
GAME_HEIGHT = MAP_ROWS * TILE_SIZE # 620 px
HEADER_HEIGHT = 60
FOOTER_HEIGHT = 40

TOTAL_GAME_HEIGHT = GAME_HEIGHT + HEADER_HEIGHT + FOOTER_HEIGHT # 720 px
HUD_WIDTH = 440
TOTAL_WIDTH = GAME_WIDTH + HUD_WIDTH # 1000 px

# Color Palette (Neon Arcade Style)
COLOR_BG = (10, 10, 20)
COLOR_MAZE_WALL = (25, 40, 180)
COLOR_MAZE_WALL_INNER = (40, 90, 255)
COLOR_PELLET = (255, 200, 150)
COLOR_ENERGIZER = (255, 255, 255)

COLOR_PACMAN = (255, 238, 0)
COLOR_BLINKY = (255, 0, 0)     # Red
COLOR_PINKY = (255, 184, 255)  # Pink
COLOR_INKY = (0, 255, 255)     # Cyan
COLOR_CLYDE = (255, 184, 82)   # Orange
COLOR_FRIGHTENED = (33, 33, 255) # Blue
COLOR_FRIGHTENED_FLASH = (255, 255, 255)
COLOR_EYES = (220, 220, 255)

COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (0, 255, 150)
COLOR_HUD_BG = (18, 22, 35)

# Directions
DIR_NONE = (0, 0)
DIR_UP = (0, -1)
DIR_DOWN = (0, 1)
DIR_LEFT = (-1, 0)
DIR_RIGHT = (1, 0)

DIRECTION_NAMES = {
    DIR_NONE: "HOLD",
    DIR_UP: "UP",
    DIR_DOWN: "DOWN",
    DIR_LEFT: "LEFT",
    DIR_RIGHT: "RIGHT"
}

# ==========================================
# RETRO SOUND SYNTHESIZER
# ==========================================
class SoundSynth:
    """Generates retro 8-bit sound effects directly without external audio files."""
    def __init__(self):
        self.enabled = False
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.enabled = True
            self.sounds = {
                'waka': self._generate_tone(440, 0.05, waveform='square'),
                'energizer': self._generate_tone(600, 0.15, waveform='sine'),
                'eat_ghost': self._generate_sweep(400, 1200, 0.25),
                'death': self._generate_sweep(800, 150, 0.5),
                'start': self._generate_melody([(261, 0.1), (329, 0.1), (392, 0.1), (523, 0.2)])
            }
        except Exception as e:
            print(f"[SoundSynth Warning] Pygame mixer init skipped: {e}")

    def _generate_tone(self, freq, duration, sample_rate=22050, waveform='square'):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        if waveform == 'square':
            wave = np.sign(np.sin(2 * np.pi * freq * t)) * 0.2
        else:
            wave = np.sin(2 * np.pi * freq * t) * 0.2
        audio = (wave * 32767).astype(np.int16)
        return pygame.mixer.Sound(buffer=audio.tobytes())

    def _generate_sweep(self, start_freq, end_freq, duration, sample_rate=22050):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        freqs = np.linspace(start_freq, end_freq, len(t))
        phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
        wave = np.sin(phase) * 0.25
        audio = (wave * 32767).astype(np.int16)
        return pygame.mixer.Sound(buffer=audio.tobytes())

    def _generate_melody(self, notes, sample_rate=22050):
        chunks = []
        for freq, duration in notes:
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sign(np.sin(2 * np.pi * freq * t)) * 0.2
            chunks.append((wave * 32767).astype(np.int16))
        full_wave = np.concatenate(chunks)
        return pygame.mixer.Sound(buffer=full_wave.tobytes())

    def play(self, sound_name):
        if self.enabled and sound_name in self.sounds:
            try:
                self.sounds[sound_name].play()
            except Exception:
                pass


# ==========================================
# CAMERA & HAND GESTURE TRACKER
# ==========================================
class HandGestureTracker:
    """Tracks hand gestures using MediaPipe & OpenCV to output directional commands."""
    def __init__(self, camera_id=0):
        self.cap = None
        self.camera_ok = False
        self.error_lines = []
        self._open_camera(camera_id)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.current_direction = DIR_NONE
        self.direction_buffer = [DIR_NONE] * 4 # Debounce window
        self.hand_detected = False
        self.raw_frame = None
        self.processed_surface = None

    def _open_camera(self, preferred_id=0):
        """Open the webcam robustly.

        On macOS the camera is gated by TCC: access is granted to the *app that
        launches the process* (Terminal / iTerm / VS Code), not to Python. If
        that app lacks Camera permission, VideoCapture opens but read() returns
        False forever with no prompt -- which showed up as 'No Camera Stream
        Found'. We try the native AVFoundation backend and a few device indices,
        then confirm frames actually flow, and record a clear reason on failure.
        """
        # Prefer the platform-native backend; it warms up more reliably.
        if sys.platform == 'darwin':
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        candidate_ids = [preferred_id] + [i for i in (0, 1, 2) if i != preferred_id]

        for backend in backends:
            for cam_id in candidate_ids:
                cap = cv2.VideoCapture(cam_id, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Warm-up: the first frames after opening are often dropped.
                got_frame = False
                for _ in range(30):
                    ret, _frame = cap.read()
                    if ret and _frame is not None:
                        got_frame = True
                        break
                if got_frame:
                    self.cap = cap
                    self.camera_ok = True
                    print(f"[Camera] Opened device {cam_id} (backend={backend}).")
                    return
                # Opened but delivered no frames -> almost always a permission block.
                cap.release()

        self.camera_ok = False
        self.error_lines = [
            "CAMERA BLOCKED",
            "Grant Camera access to your terminal/IDE:",
            "System Settings > Privacy & Security > Camera",
            "then fully QUIT & reopen that app and relaunch.",
        ]
        print("[Camera] ERROR: could not read frames from any device.")
        print("[Camera] On macOS this is almost always a permission issue.")
        print("[Camera] Enable Camera for the app you launch from in:")
        print("[Camera]   System Settings > Privacy & Security > Camera")
        print("[Camera] Then fully quit that app (Cmd+Q) and relaunch the game.")

    def update(self):
        ret, frame = (False, None)
        if self.cap is not None:
            ret, frame = self.cap.read()
        if not ret or frame is None:
            # Fallback frame with a helpful, specific reason.
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            lines = self.error_lines or ["No Camera Stream Found"]
            y = 240 - (len(lines) - 1) * 16
            for i, line in enumerate(lines):
                scale = 0.8 if i == 0 else 0.5
                color = (0, 0, 255) if i == 0 else (255, 255, 255)
                (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                cv2.putText(frame, line, (max(10, 320 - tw // 2), y + i * 32),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            self.hand_detected = False
            self.current_direction = DIR_NONE
            self._frame_to_surface(frame)
            return

        # Mirror frame for natural self-reflection
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        detected_dir = DIR_NONE
        self.hand_detected = False

        # Draw Guide Overlay & Zones on Camera Frame
        cx, cy = w // 2, h // 2
        deadzone_w, deadzone_h = int(w * 0.28), int(h * 0.28)
        
        # Region rectangles
        cv2.rectangle(frame, (cx - deadzone_w, cy - deadzone_h),
                      (cx + deadzone_w, cy + deadzone_h), (255, 255, 255), 1)

        # Zone Labels & Direction Indicators
        cv2.putText(frame, "UP", (cx - 20, cy - deadzone_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "DOWN", (cx - 35, cy + deadzone_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "LEFT", (cx - deadzone_w - 70, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "RIGHT", (cx + deadzone_w + 15, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if results.multi_hand_landmarks:
            self.hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw Skeleton
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )

            # Extract Index Fingertip (8) & Wrist (0)
            ix = hand_landmarks.landmark[8].x * w
            iy = hand_landmarks.landmark[8].y * h
            wx = hand_landmarks.landmark[0].x * w
            wy = hand_landmarks.landmark[0].y * h

            # Highlight fingertip
            cv2.circle(frame, (int(ix), int(iy)), 12, (0, 255, 0), -1)
            cv2.circle(frame, (int(ix), int(iy)), 16, (255, 255, 255), 2)

            # Zone-Based Classification
            if iy < cy - deadzone_h:
                detected_dir = DIR_UP
            elif iy > cy + deadzone_h:
                detected_dir = DIR_DOWN
            elif ix < cx - deadzone_w:
                detected_dir = DIR_LEFT
            elif ix > cx + deadzone_w:
                detected_dir = DIR_RIGHT
            else:
                # Pointing Vector Classification (for inside center zone)
                vec_x = ix - wx
                vec_y = iy - wy
                dist = math.hypot(vec_x, vec_y)
                if dist > 35:
                    if abs(vec_x) > abs(vec_y):
                        detected_dir = DIR_RIGHT if vec_x > 0 else DIR_LEFT
                    else:
                        detected_dir = DIR_DOWN if vec_y > 0 else DIR_UP

        # Update smooth gesture buffer
        self.direction_buffer.pop(0)
        self.direction_buffer.append(detected_dir)

        # Most frequent direction in buffer wins
        dir_counts = {}
        for d in self.direction_buffer:
            dir_counts[d] = dir_counts.get(d, 0) + 1
        
        best_dir = max(dir_counts, key=dir_counts.get)
        if dir_counts[best_dir] >= 2 and best_dir != DIR_NONE:
            self.current_direction = best_dir

        # Draw Active Highlight Arrow Overlay on Frame
        self._draw_active_direction_indicator(frame, cx, cy, deadzone_w, deadzone_h)
        self._frame_to_surface(frame)

    def _draw_active_direction_indicator(self, frame, cx, cy, dw, dh):
        color = (0, 255, 120)
        thick = 4
        if self.current_direction == DIR_UP:
            cv2.arrowedLine(frame, (cx, cy), (cx, cy - dh - 30), color, thick, tipLength=0.4)
        elif self.current_direction == DIR_DOWN:
            cv2.arrowedLine(frame, (cx, cy), (cx, cy + dh + 30), color, thick, tipLength=0.4)
        elif self.current_direction == DIR_LEFT:
            cv2.arrowedLine(frame, (cx, cy), (cx - dw - 30, cy), color, thick, tipLength=0.4)
        elif self.current_direction == DIR_RIGHT:
            cv2.arrowedLine(frame, (cx, cy), (cx + dw + 30, cy), color, thick, tipLength=0.4)

    def _frame_to_surface(self, frame):
        # Convert BGR OpenCV image to Pygame Surface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        frame_rgb = cv2.flip(frame_rgb, 0)
        self.processed_surface = pygame.surfarray.make_surface(frame_rgb)

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


# ==========================================
# CLASSIC MAZE DATA
# ==========================================
GRID_MAP = [
    "1111111111111111111111111111",
    "1222222222222112222222222221",
    "1211112111112112111112111121",
    "1311112111112112111112111131",
    "1211112111112112111112111121",
    "1222222222222222222222222221",
    "1211112112111111112112111121",
    "1211112112111111112112111121",
    "1222222112222112222112222221",
    "1111112111110110111112111111",
    "0000012111110110111112100000",
    "0000012110000000000112100000",
    "0000012110111441110112100000",
    "1111112110100000010112111111",
    "0000002000100000010002000000",
    "1111112110100000010112111111",
    "0000012110111111110112100000",
    "0000012110000000000112100000",
    "0000012110111111110112100000",
    "1111112112111111112112111111",
    "1222222222222112222222222221",
    "1211112111112112111112111121",
    "1211112111112112111112111121",
    "1322112222222002222222112231",
    "1112112112111111112112112111",
    "1112112112111111112112112111",
    "1222222112222112222112222221",
    "1211111111112112111111111121",
    "1211111111112112111111111121",
    "1222222222222222222222222221",
    "1111111111111111111111111111"
]

# Level 2: a more open "comb" layout -- same tunnel/ghost-house core as Level 1
# (rows 9-19 below are identical across all levels so the hardcoded ghost-house
# gate/home logic keeps working), with a fresh top/bottom maze around it.
GRID_MAP_LEVEL2 = [
    "1111111111111111111111111111",
    "1321122221122222211222211231",
    "1222222222222222222222222221",
    "1222221122222112222211222221",
    "1222222222222222222222222221",
    "1221122221122222211222211221",
    "1222222222222222222222222221",
    "1222221122222112222211222221",
    "1222222222222222222222222221",
    "1111112111110110111112111111",
    "0000012111110110111112100000",
    "0000012110000000000112100000",
    "0000012110111441110112100000",
    "1111112110100000010112111111",
    "0000002000100000010002000000",
    "1111112110100000010112111111",
    "0000012110111111110112100000",
    "0000012110000000000112100000",
    "0000012110111111110112100000",
    "1111112112111111112112111111",
    "1222222222222222222222222221",
    "1221122221122222211222211221",
    "1222222222222222222222222221",
    "1222222222222022222222222221",
    "1222221122222222222211222221",
    "1222222222222222222222222221",
    "1221122221122222211222211221",
    "1222222222222222222222222221",
    "1322221122222222222211222231",
    "1222222222222222222222222221",
    "1111111111111111111111111111",
]

# Level 3: denser pillar pattern for a tighter, trickier maze -- same core again.
GRID_MAP_LEVEL3 = [
    "1111111111111111111111111111",
    "1311122211221111221122211131",
    "1222222222222222222222222221",
    "1222211222112222112221122221",
    "1222222222222222222222222221",
    "1211122211221111221122211121",
    "1222222222222222222222222221",
    "1222211222112222112221122221",
    "1222222222222222222222222221",
    "1111112111110110111112111111",
    "0000012111110110111112100000",
    "0000012110000000000112100000",
    "0000012110111441110112100000",
    "1111112110100000010112111111",
    "0000002000100000010002000000",
    "1111112110100000010112111111",
    "0000012110111111110112100000",
    "0000012110000000000112100000",
    "0000012110111111110112100000",
    "1111112112111111112112111111",
    "1222222222222222222222222221",
    "1211122211221221221122211121",
    "1222222222222222222222222221",
    "1222222222222022222222222221",
    "1222211222112222112221122221",
    "1222222222222222222222222221",
    "1211122211222222221122211121",
    "1222222222222222222222222221",
    "1322211222112222112221122231",
    "1222222222222222222222222221",
    "1111111111111111111111111111",
]

LEVEL_MAPS = [GRID_MAP, GRID_MAP_LEVEL2, GRID_MAP_LEVEL3]

# Wall (outer, inner) colors per level, purely cosmetic so each level reads as new.
LEVEL_THEMES = {
    1: ((25, 40, 180), (40, 90, 255)),    # Blue (original)
    2: ((150, 30, 180), (215, 70, 255)),  # Magenta
    3: ((20, 150, 90), (60, 230, 150)),   # Teal/green
}

# ==========================================
# MAZE MANAGEMENT
# ==========================================
class Maze:
    def __init__(self):
        self.reset(1)

    def reset(self, level=1):
        self.level = level
        grid_map = LEVEL_MAPS[(level - 1) % len(LEVEL_MAPS)]
        self.wall_color, self.wall_inner_color = LEVEL_THEMES.get(level, LEVEL_THEMES[1])
        self.grid = [list(row) for row in grid_map]
        self.pellets_left = 0
        for r in range(MAP_ROWS):
            for c in range(MAP_COLS):
                if self.grid[r][c] in ('2', '3'):
                    self.pellets_left += 1

    def is_wall(self, col, row):
        if row < 0 or row >= MAP_ROWS or col < 0 or col >= MAP_COLS:
            return False # Side tunnels
        return self.grid[row][col] == '1'

    def is_ghost_gate(self, col, row):
        if 0 <= row < MAP_ROWS and 0 <= col < MAP_COLS:
            return self.grid[row][col] == '4'
        return False

    def draw(self, surface):
        for r in range(MAP_ROWS):
            for c in range(MAP_COLS):
                cell = self.grid[r][c]
                x = c * TILE_SIZE
                y = r * TILE_SIZE + HEADER_HEIGHT

                if cell == '1':
                    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(surface, self.wall_color, rect, border_radius=4)
                    inner_rect = pygame.Rect(x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4)
                    pygame.draw.rect(surface, self.wall_inner_color, inner_rect, 1, border_radius=3)
                elif cell == '2':
                    pygame.draw.circle(surface, COLOR_PELLET, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 3)
                elif cell == '3':
                    # Blinking Energizer
                    if (pygame.time.get_ticks() // 250) % 2 == 0:
                        pygame.draw.circle(surface, COLOR_ENERGIZER, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 7)
                elif cell == '4':
                    # Ghost Gate Line
                    pygame.draw.line(surface, (255, 184, 255), (x, y + TILE_SIZE // 2), (x + TILE_SIZE, y + TILE_SIZE // 2), 3)


# ==========================================
# ENTITIES: PAC-MAN
# ==========================================
class PacMan:
    def __init__(self):
        # Spawn on an integer tile column: the movement engine only lets an
        # entity turn when it is grid-aligned (abs(x-center) < speed), and a
        # half-tile column (13.5) sits 10px off-center and never aligns, which
        # blocked up/down turns from the start. Speed 2.0 divides TILE_SIZE (20)
        # so Pac-Man lands exactly on tile centers for crisp turning.
        self.spawn_col = 13
        self.spawn_row = 23
        self.speed = 2.0
        self.reset()

    def reset(self):
        self.x = self.spawn_col * TILE_SIZE
        self.y = self.spawn_row * TILE_SIZE
        self.dir = DIR_NONE
        self.next_dir = DIR_NONE
        self.mouth_angle = 45
        self.mouth_speed = 4
        self.mouth_opening = True

    def update(self, maze):
        # Handle Portal Tunnel wrap
        if self.x < -TILE_SIZE / 2:
            self.x = GAME_WIDTH - TILE_SIZE / 2
        elif self.x > GAME_WIDTH - TILE_SIZE / 2:
            self.x = -TILE_SIZE / 2

        # Current tile coordinates
        curr_col = int(round(self.x / TILE_SIZE))
        curr_row = int(round(self.y / TILE_SIZE))

        center_x = curr_col * TILE_SIZE
        center_y = curr_row * TILE_SIZE

        is_aligned_x = abs(self.x - center_x) < self.speed
        is_aligned_y = abs(self.y - center_y) < self.speed

        # 1. Determine open directions from current cell
        open_dirs = []
        for d in (DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT):
            tc = curr_col + d[0]
            tr = curr_row + d[1]
            if not maze.is_wall(tc, tr):
                open_dirs.append(d)

        # 2. Process direction change request
        if self.next_dir != DIR_NONE:
            if self.dir == DIR_NONE:
                # Stopped: start moving if next_dir is open
                if self.next_dir in open_dirs:
                    self.dir = self.next_dir
                elif len(open_dirs) == 1:
                    # Dead end corner: auto-turn along sole open path
                    self.dir = open_dirs[0]
            elif self.next_dir[0] == -self.dir[0] and self.next_dir[1] == -self.dir[1]:
                # Instant 180-degree reversal anywhere in corridor
                self.dir = self.next_dir
            else:
                # 90-degree turn: check alignment at intersection
                can_turn = False
                if self.dir in (DIR_LEFT, DIR_RIGHT) and is_aligned_x:
                    can_turn = True
                elif self.dir in (DIR_UP, DIR_DOWN) and is_aligned_y:
                    can_turn = True

                if can_turn:
                    tc = curr_col + self.next_dir[0]
                    tr = curr_row + self.next_dir[1]
                    if not maze.is_wall(tc, tr):
                        self.dir = self.next_dir
                        self.x = center_x
                        self.y = center_y

        # 3. Move in active direction
        if self.dir != DIR_NONE:
            next_x = self.x + self.dir[0] * self.speed
            next_y = self.y + self.dir[1] * self.speed

            # Collision check based on direction
            if self.dir == DIR_LEFT:
                check_col = int(math.floor(next_x / TILE_SIZE))
                check_row = curr_row
            elif self.dir == DIR_RIGHT:
                check_col = int(math.floor((next_x + TILE_SIZE - 1) / TILE_SIZE))
                check_row = curr_row
            elif self.dir == DIR_UP:
                check_col = curr_col
                check_row = int(math.floor(next_y / TILE_SIZE))
            elif self.dir == DIR_DOWN:
                check_col = curr_col
                check_row = int(math.floor((next_y + TILE_SIZE - 1) / TILE_SIZE))
            else:
                check_col, check_row = curr_col, curr_row

            if not maze.is_wall(check_col, check_row):
                self.x = next_x
                self.y = next_y

                # Animate Mouth
                if self.mouth_opening:
                    self.mouth_angle += self.mouth_speed
                    if self.mouth_angle >= 45:
                        self.mouth_opening = False
                else:
                    self.mouth_angle -= self.mouth_speed
                    if self.mouth_angle <= 5:
                        self.mouth_opening = True
            else:
                # Stopped by wall: snap to grid center
                if self.dir in (DIR_LEFT, DIR_RIGHT):
                    self.x = center_x
                else:
                    self.y = center_y

                # If next_dir is open at this corner, turn into it immediately
                if self.next_dir in open_dirs:
                    self.dir = self.next_dir
                elif len(open_dirs) == 1:
                    # Dead-end corner: auto-turn into sole open direction
                    self.dir = open_dirs[0]
                else:
                    self.dir = DIR_NONE

    def draw(self, surface):
        center_x = int(self.x + TILE_SIZE / 2)
        center_y = int(self.y + TILE_SIZE / 2) + HEADER_HEIGHT
        radius = TILE_SIZE // 2 + 1

        # Calculate rotation angle based on direction
        rotation = 0
        if self.dir == DIR_RIGHT: rotation = 0
        elif self.dir == DIR_DOWN: rotation = 270
        elif self.dir == DIR_LEFT: rotation = 180
        elif self.dir == DIR_UP: rotation = 90

        start_angle = math.radians(rotation + self.mouth_angle)
        end_angle = math.radians(rotation + 360 - self.mouth_angle)

        # Draw Pac-Man pie wedge arc
        points = [(center_x, center_y)]
        num_segments = 24
        for i in range(num_segments + 1):
            ang = start_angle + (end_angle - start_angle) * (i / num_segments)
            px = center_x + radius * math.cos(ang)
            py = center_y - radius * math.sin(ang)
            points.append((px, py))

        if len(points) > 2:
            pygame.draw.polygon(surface, COLOR_PACMAN, points)


# ==========================================
# ENTITIES: GHOSTS
# ==========================================
class Ghost:
    MODE_SCATTER = 'SCATTER'
    MODE_CHASE = 'CHASE'
    MODE_FRIGHTENED = 'FRIGHTENED'
    MODE_EATEN = 'EATEN'

    def __init__(self, name, color, spawn_col, spawn_row, scatter_target):
        self.name = name
        self.color = color
        self.spawn_col = spawn_col
        self.spawn_row = spawn_row
        self.scatter_target = scatter_target
        self.speed_scale = 1.0  # bumped up per level for increasing difficulty
        self.reset()

    def reset(self):
        self.x = self.spawn_col * TILE_SIZE
        self.y = self.spawn_row * TILE_SIZE
        self.dir = DIR_UP
        self.mode = Ghost.MODE_SCATTER
        self.speed = 2.0
        self.frightened_timer = 0

    def set_frightened(self, duration_frames=8 * FPS):
        if self.mode != Ghost.MODE_EATEN:
            self.mode = Ghost.MODE_FRIGHTENED
            self.frightened_timer = duration_frames
            # Reverse direction on scare
            self.dir = (-self.dir[0], -self.dir[1])

    def update(self, maze, pacman, blinky_pos):
        if self.mode == Ghost.MODE_FRIGHTENED:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.mode = Ghost.MODE_CHASE
            self.speed = 1.2 * self.speed_scale
        elif self.mode == Ghost.MODE_EATEN:
            self.speed = 4.0 * self.speed_scale
        else:
            self.speed = 2.0 * self.speed_scale

        # Handle Tunnel Wrap
        if self.x < -TILE_SIZE / 2: self.x = GAME_WIDTH - TILE_SIZE / 2
        elif self.x > GAME_WIDTH - TILE_SIZE / 2: self.x = -TILE_SIZE / 2

        curr_col = int(round(self.x / TILE_SIZE))
        curr_row = int(round(self.y / TILE_SIZE))
        center_x = curr_col * TILE_SIZE
        center_y = curr_row * TILE_SIZE

        # When aligned to tile center, pick next direction
        if abs(self.x - center_x) < self.speed and abs(self.y - center_y) < self.speed:
            self.x = center_x
            self.y = center_y

            # Check for return home when eaten
            if self.mode == Ghost.MODE_EATEN and curr_col in (13, 14) and curr_row in (11, 12):
                self.mode = Ghost.MODE_CHASE

            valid_dirs = []
            for d in [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]:
                # Ghosts cannot reverse 180 degrees unless forced
                if d == (-self.dir[0], -self.dir[1]): continue
                nc, nr = curr_col + d[0], curr_row + d[1]
                if not maze.is_wall(nc, nr):
                    if maze.is_ghost_gate(nc, nr) and self.mode != Ghost.MODE_EATEN:
                        continue
                    valid_dirs.append(d)

            if not valid_dirs:
                valid_dirs = [(-self.dir[0], -self.dir[1])]

            # Target Selection
            if self.mode == Ghost.MODE_FRIGHTENED:
                # Random choice
                idx = (pygame.time.get_ticks() + id(self)) % len(valid_dirs)
                self.dir = valid_dirs[idx]
            else:
                target = self._get_target(pacman, blinky_pos)
                best_dir = valid_dirs[0]
                best_dist = float('inf')
                for d in valid_dirs:
                    nc, nr = curr_col + d[0], curr_row + d[1]
                    dist = (nc - target[0]) ** 2 + (nr - target[1]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_dir = d
                self.dir = best_dir

        self.x += self.dir[0] * self.speed
        self.y += self.dir[1] * self.speed

    def _get_target(self, pacman, blinky_pos):
        if self.mode == Ghost.MODE_EATEN:
            return (13, 11) # Ghost House Home
        if self.mode == Ghost.MODE_SCATTER:
            return self.scatter_target

        pac_c = int(pacman.x // TILE_SIZE)
        pac_r = int(pacman.y // TILE_SIZE)

        if self.name == 'Blinky':
            return (pac_c, pac_r)
        elif self.name == 'Pinky':
            return (pac_c + pacman.dir[0] * 4, pac_r + pacman.dir[1] * 4)
        elif self.name == 'Inky':
            ahead_c = pac_c + pacman.dir[0] * 2
            ahead_r = pac_r + pacman.dir[1] * 2
            vec_c = ahead_c - blinky_pos[0]
            vec_r = ahead_r - blinky_pos[1]
            return (blinky_pos[0] + vec_c * 2, blinky_pos[1] + vec_r * 2)
        elif self.name == 'Clyde':
            dist = math.hypot(pac_c - (self.x // TILE_SIZE), pac_r - (self.y // TILE_SIZE))
            return (pac_c, pac_r) if dist > 8 else self.scatter_target
        return (pac_c, pac_r)

    def draw(self, surface):
        cx = int(self.x + TILE_SIZE / 2)
        cy = int(self.y + TILE_SIZE / 2) + HEADER_HEIGHT
        r = TILE_SIZE // 2

        if self.mode == Ghost.MODE_FRIGHTENED:
            body_color = COLOR_FRIGHTENED
            if self.frightened_timer < 2 * FPS and (self.frightened_timer // 15) % 2 == 0:
                body_color = COLOR_FRIGHTENED_FLASH
        elif self.mode == Ghost.MODE_EATEN:
            body_color = None # Eyes only
        else:
            body_color = self.color

        if body_color:
            # Ghost Dome + Skirt
            pygame.draw.circle(surface, body_color, (cx, cy - 2), r)
            pygame.draw.rect(surface, body_color, (cx - r, cy - 2, r * 2, r))
            
            # Wavy Skirt Bottom
            for i in range(3):
                wave_x = cx - r + (i * (r * 2 // 3)) + (r // 3)
                pygame.draw.circle(surface, COLOR_BG, (wave_x, cy + r - 2), 3)

        # Eyes
        eye_color = COLOR_EYES
        pupil_color = (0, 0, 180)
        
        # Offsets based on direction
        off_x = self.dir[0] * 3
        off_y = self.dir[1] * 3

        if body_color != COLOR_FRIGHTENED:
            # Left Eye
            pygame.draw.circle(surface, eye_color, (cx - 4 + off_x, cy - 4 + off_y), 4)
            pygame.draw.circle(surface, pupil_color, (cx - 4 + off_x * 1.5, cy - 4 + off_y * 1.5), 2)
            # Right Eye
            pygame.draw.circle(surface, eye_color, (cx + 4 + off_x, cy - 4 + off_y), 4)
            pygame.draw.circle(surface, pupil_color, (cx + 4 + off_x * 1.5, cy - 4 + off_y * 1.5), 2)
        else:
            # Scared Face Eyes & Mouth
            pygame.draw.circle(surface, (255, 200, 150), (cx - 3, cy - 3), 2)
            pygame.draw.circle(surface, (255, 200, 150), (cx + 3, cy - 3), 2)


# ==========================================
# MAIN PAC-MAN GAME ENGINE & HUD
# ==========================================
class PacmanGame:
    STATE_COUNTDOWN = 'COUNTDOWN'
    STATE_PLAYING = 'PLAYING'
    STATE_DIED = 'DIED'
    STATE_GAME_OVER = 'GAME_OVER'
    STATE_LEVEL_COMPLETE = 'LEVEL_COMPLETE'
    STATE_VICTORY = 'VICTORY'

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pac-Man: Camera Hand Gesture Control")
        
        self.fullscreen = True
        self.screen = pygame.display.set_mode(
            (TOTAL_WIDTH, TOTAL_GAME_HEIGHT), pygame.FULLSCREEN | pygame.SCALED
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Courier", 20, bold=True)
        self.large_font = pygame.font.SysFont("Arial", 36, bold=True)

        self.synth = SoundSynth()
        self.tracker = HandGestureTracker(camera_id=0)

        self.score = 0
        self.high_score = 1000
        self.lives = 42
        self.level = 1

        self.maze = Maze()
        self.pacman = PacMan()
        # Ghosts spawn on integer tile columns in the open corridor just above
        # the gate. The house interior (where they used to spawn) is reachable
        # ONLY through the '4' gate, which non-eaten ghosts are never allowed to
        # cross -- so they were trapped and never appeared in the maze. These
        # tiles are all in Pac-Man's reachable region, so the ghosts roam freely.
        self.ghosts = [
            Ghost('Blinky', COLOR_BLINKY, 13, 11, (27, 0)),
            Ghost('Pinky', COLOR_PINKY, 14, 11, (0, 0)),
            Ghost('Inky', COLOR_INKY, 12, 11, (27, 30)),
            Ghost('Clyde', COLOR_CLYDE, 15, 11, (0, 30))
        ]

        self.state = PacmanGame.STATE_COUNTDOWN
        self.state_timer = 3 * FPS
        # Frames to ignore gestures on an end screen, so the gesture that was
        # active at the moment of death/victory doesn't instantly restart.
        self.restart_cooldown = 0
        self.synth.play('start')

    def reset_positions(self):
        self.pacman.reset()
        for g in self.ghosts:
            g.reset()

    def _load_level(self, level):
        self.maze.reset(level)
        self.reset_positions()
        for g in self.ghosts:
            g.speed_scale = 1.0 + (level - 1) * 0.15

    def _frightened_duration(self):
        # Power pellets stay effective for less time on later, harder levels.
        return max(3 * FPS, int(8 * FPS - (self.level - 1) * 2 * FPS))

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        flags = (pygame.FULLSCREEN | pygame.SCALED) if self.fullscreen else pygame.SCALED
        self.screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_GAME_HEIGHT), flags)

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            
            # Process Pygame Window Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    # Fallback keyboard controls for manual testing
                    elif event.key == pygame.K_UP: self.pacman.next_dir = DIR_UP
                    elif event.key == pygame.K_DOWN: self.pacman.next_dir = DIR_DOWN
                    elif event.key == pygame.K_LEFT: self.pacman.next_dir = DIR_LEFT
                    elif event.key == pygame.K_RIGHT: self.pacman.next_dir = DIR_RIGHT
                    elif event.key == pygame.K_r and self.state in (PacmanGame.STATE_GAME_OVER, PacmanGame.STATE_VICTORY):
                        self._restart_game()

            # Update Camera Hand Tracker
            self.tracker.update()

            if self.state in (PacmanGame.STATE_GAME_OVER, PacmanGame.STATE_VICTORY):
                # Restart with a hand gesture: after a short cooldown, any
                # detected steering direction restarts the game.
                if self.restart_cooldown > 0:
                    self.restart_cooldown -= 1
                elif self.tracker.hand_detected and self.tracker.current_direction != DIR_NONE:
                    self._restart_game()
            else:
                # Direct Pac-Man using Hand Gesture Output
                if self.tracker.current_direction != DIR_NONE:
                    self.pacman.next_dir = self.tracker.current_direction

            # Update Game State Logic
            self._update_game_logic()

            # Render Screen
            self._render()
            pygame.display.flip()

        self.tracker.release()
        pygame.quit()
        sys.exit()

    def _update_game_logic(self):
        if self.state == PacmanGame.STATE_COUNTDOWN:
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.state = PacmanGame.STATE_PLAYING

        elif self.state == PacmanGame.STATE_PLAYING:
            self.pacman.update(self.maze)
            
            # Check Pellet / Energizer Eating
            pc = int(round(self.pacman.x / TILE_SIZE))
            pr = int(round(self.pacman.y / TILE_SIZE))
            if 0 <= pr < MAP_ROWS and 0 <= pc < MAP_COLS:
                cell = self.maze.grid[pr][pc]
                if cell == '2': # Normal Pellet
                    self.maze.grid[pr][pc] = '0'
                    self.score += 10
                    self.maze.pellets_left -= 1
                    self.synth.play('waka')
                elif cell == '3': # Energizer
                    self.maze.grid[pr][pc] = '0'
                    self.score += 50
                    self.maze.pellets_left -= 1
                    self.synth.play('energizer')
                    for g in self.ghosts:
                        g.set_frightened(self._frightened_duration())

            if self.score > self.high_score:
                self.high_score = self.score

            # Check Level Complete / Final Victory
            if self.maze.pellets_left <= 0:
                if self.level < len(LEVEL_MAPS):
                    self.state = PacmanGame.STATE_LEVEL_COMPLETE
                    self.state_timer = int(2.5 * FPS)
                else:
                    self.state = PacmanGame.STATE_VICTORY
                    self._arm_restart()

            # Update Ghosts
            blinky_pos = (int(self.ghosts[0].x // TILE_SIZE), int(self.ghosts[0].y // TILE_SIZE))
            for g in self.ghosts:
                g.update(self.maze, self.pacman, blinky_pos)

                # Check Collision with Pac-Man
                dist = math.hypot(self.pacman.x - g.x, self.pacman.y - g.y)
                if dist < TILE_SIZE * 0.75:
                    if g.mode == Ghost.MODE_FRIGHTENED:
                        g.mode = Ghost.MODE_EATEN
                        self.score += 200
                        self.synth.play('eat_ghost')
                    elif g.mode not in (Ghost.MODE_EATEN, Ghost.MODE_FRIGHTENED):
                        # Pac-Man dies
                        self.synth.play('death')
                        self.lives -= 1
                        self.state = PacmanGame.STATE_DIED
                        self.state_timer = 2 * FPS

        elif self.state == PacmanGame.STATE_DIED:
            self.state_timer -= 1
            if self.state_timer <= 0:
                if self.lives > 0:
                    self.reset_positions()
                    self.state = PacmanGame.STATE_COUNTDOWN
                    self.state_timer = 2 * FPS
                else:
                    self.state = PacmanGame.STATE_GAME_OVER
                    self._arm_restart()

        elif self.state == PacmanGame.STATE_LEVEL_COMPLETE:
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.level += 1
                self._load_level(self.level)
                self.state = PacmanGame.STATE_COUNTDOWN
                self.state_timer = 3 * FPS

    def _arm_restart(self):
        # Ignore gestures briefly, and clear the sticky gesture, so the player
        # must make a fresh hand movement to restart from an end screen.
        self.restart_cooldown = int(1.2 * FPS)
        self.tracker.current_direction = DIR_NONE
        self.tracker.direction_buffer = [DIR_NONE] * len(self.tracker.direction_buffer)

    def _restart_game(self):
        self.score = 0
        self.lives = 42
        self.level = 1
        self._load_level(self.level)
        self.state = PacmanGame.STATE_COUNTDOWN
        self.state_timer = 2 * FPS

    def _render(self):
        self.screen.fill(COLOR_BG)

        # 1. Render Left Panel (Pac-Man Game Canvas)
        game_surface = pygame.Surface((GAME_WIDTH, TOTAL_GAME_HEIGHT))
        game_surface.fill(COLOR_BG)
        
        # Header (Scores & Lives)
        score_txt = self.font.render(f"SCORE: {self.score:05d}", True, COLOR_TEXT)
        high_txt = self.font.render(f"HIGH: {self.high_score:05d}", True, COLOR_ACCENT)
        level_txt = self.font.render(f"LEVEL {self.level}/{len(LEVEL_MAPS)}", True, COLOR_PACMAN)
        game_surface.blit(score_txt, (15, 15))
        game_surface.blit(high_txt, (GAME_WIDTH - high_txt.get_width() - 15, 15))
        game_surface.blit(level_txt, (GAME_WIDTH // 2 - level_txt.get_width() // 2, 15))

        # Maze & Entities
        self.maze.draw(game_surface)
        self.pacman.draw(game_surface)
        for g in self.ghosts:
            g.draw(game_surface)

        # Footer (Lives Pacman Icons)
        for i in range(self.lives):
            lx = 20 + i * 26
            ly = TOTAL_GAME_HEIGHT - 25
            pygame.draw.circle(game_surface, COLOR_PACMAN, (lx, ly), 9)
            pygame.draw.polygon(game_surface, COLOR_BG, [(lx, ly), (lx + 9, ly - 4), (lx + 9, ly + 4)])

        # Overlay Banners for States
        if self.state == PacmanGame.STATE_COUNTDOWN:
            secs = math.ceil(self.state_timer / FPS)
            txt = self.large_font.render(f"READY! {secs}", True, COLOR_PACMAN)
            game_surface.blit(txt, (GAME_WIDTH // 2 - txt.get_width() // 2, GAME_HEIGHT // 2 + 30))
        elif self.state == PacmanGame.STATE_LEVEL_COMPLETE:
            txt = self.large_font.render(f"LEVEL {self.level} COMPLETE!", True, COLOR_ACCENT)
            sub = self.font.render(f"Get ready for Level {self.level + 1}...", True, COLOR_TEXT)
            game_surface.blit(txt, (GAME_WIDTH // 2 - txt.get_width() // 2, GAME_HEIGHT // 2))
            game_surface.blit(sub, (GAME_WIDTH // 2 - sub.get_width() // 2, GAME_HEIGHT // 2 + 45))
        elif self.state in (PacmanGame.STATE_GAME_OVER, PacmanGame.STATE_VICTORY):
            if self.state == PacmanGame.STATE_GAME_OVER:
                txt = self.large_font.render("GAME OVER", True, (255, 50, 50))
            else:
                txt = self.large_font.render("YOU WIN! ALL LEVELS CLEARED!", True, COLOR_ACCENT)
            game_surface.blit(txt, (GAME_WIDTH // 2 - txt.get_width() // 2, GAME_HEIGHT // 2))

            if self.restart_cooldown > 0:
                sub_str = "Get ready..."
            else:
                sub_str = "Point your hand any direction (or press R) to restart"
            sub = self.font.render(sub_str, True, COLOR_TEXT)
            game_surface.blit(sub, (GAME_WIDTH // 2 - sub.get_width() // 2, GAME_HEIGHT // 2 + 45))

        self.screen.blit(game_surface, (0, 0))

        # 2. Render Divider Bar
        pygame.draw.line(self.screen, (40, 50, 80), (GAME_WIDTH, 0), (GAME_WIDTH, TOTAL_GAME_HEIGHT), 4)

        # 3. Render Right Panel (Camera HUD & Gesture Feedback)
        hud_surface = pygame.Surface((HUD_WIDTH, TOTAL_GAME_HEIGHT))
        hud_surface.fill(COLOR_HUD_BG)

        # Title
        title_txt = self.large_font.render("CAMERA CONTROLLER", True, COLOR_ACCENT)
        hud_surface.blit(title_txt, (HUD_WIDTH // 2 - title_txt.get_width() // 2, 15))

        # Camera Preview Window
        if self.tracker.processed_surface:
            cam_surf = pygame.transform.smoothscale(self.tracker.processed_surface, (400, 300))
            hud_surface.blit(cam_surf, (20, 65))
            pygame.draw.rect(hud_surface, COLOR_ACCENT, (20, 65, 400, 300), 2, border_radius=4)

        # Active Direction Cards / Visual Feedback
        card_y = 385
        pygame.draw.rect(hud_surface, (28, 35, 55), (20, card_y, 400, 240), border_radius=8)

        status_str = "HAND DETECTED" if self.tracker.hand_detected else "NO HAND IN CAMERA"
        status_color = COLOR_ACCENT if self.tracker.hand_detected else (255, 100, 100)
        st_txt = self.font.render(f"STATUS: {status_str}", True, status_color)
        hud_surface.blit(st_txt, (35, card_y + 15))

        active_dir_name = DIRECTION_NAMES.get(self.tracker.current_direction, "HOLD")
        dir_txt = self.large_font.render(f"GESTURE: {active_dir_name}", True, COLOR_PACMAN)
        hud_surface.blit(dir_txt, (35, card_y + 50))

        # Visual Directional Buttons Matrix
        box_center_x, box_center_y = 220, card_y + 160
        b_size = 42
        
        btns = [
            (DIR_UP, "^ UP", box_center_x - b_size//2, box_center_y - b_size - 10),
            (DIR_DOWN, "v DN", box_center_x - b_size//2, box_center_y + b_size + 10),
            (DIR_LEFT, "< L", box_center_x - b_size*2 + 10, box_center_y - b_size//2),
            (DIR_RIGHT, "R >", box_center_x + b_size + 10, box_center_y - b_size//2),
        ]

        for d_code, label, bx, by in btns:
            is_active = (self.tracker.current_direction == d_code)
            bg_c = (0, 220, 120) if is_active else (40, 50, 75)
            fg_c = (10, 20, 30) if is_active else (200, 210, 230)
            pygame.draw.rect(hud_surface, bg_c, (bx, by, b_size + 10, b_size), border_radius=6)
            lbl = self.font.render(label, True, fg_c)
            hud_surface.blit(lbl, (bx + (b_size + 10)//2 - lbl.get_width()//2, by + b_size//2 - lbl.get_height()//2))

        # Footer Instruction
        instr_txt = self.font.render("Move hand to steering zones to play!", True, (160, 180, 210))
        hud_surface.blit(instr_txt, (HUD_WIDTH // 2 - instr_txt.get_width() // 2, TOTAL_GAME_HEIGHT - 35))

        self.screen.blit(hud_surface, (GAME_WIDTH + 4, 0))


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == '__main__':
    game = PacmanGame()
    game.run()
