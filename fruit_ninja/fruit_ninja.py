import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import random
import numpy as np
import os

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
FPS = 60
GRAVITY = 0.5
FRUIT_RADIUS = 60
BOMB_RADIUS = 60
SPAWN_RATE = 25 # Frames between spawns; lower is faster

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 20, 20)       # Apple Splash
GREEN = (40, 220, 40)     # Watermelon Splash
ORANGE = (255, 120, 0)    # Orange Splash
DARK_GREY = (50, 50, 50)  # Bomb Particle
TRAIL_COLOR = (200, 200, 255)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global dictionary for images
IMAGES = {}

def load_and_scale_image(path, radius):
    """Loads an image, sets its top-left pixel as the transparent colorkey, and scales it."""
    try:
        img = pygame.image.load(path).convert()
        colorkey = img.get_at((0, 0))
        img.set_colorkey(colorkey, pygame.RLEACCEL)
        size = int(radius * 2)
        img = pygame.transform.smoothscale(img, (size, size))
        return img
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        # Return a fallback surface
        surf = pygame.Surface((radius*2, radius*2))
        surf.fill((255, 0, 255))
        return surf

class GameObject:
    def __init__(self, is_bomb=False):
        self.is_bomb = is_bomb
        self.radius = BOMB_RADIUS if is_bomb else FRUIT_RADIUS
        self.x = random.randint(100, WIDTH - 100)
        self.y = HEIGHT + 50
        
        # Give a slight angle towards the center
        if self.x < WIDTH / 2:
            self.vx = random.uniform(2, 6)
        else:
            self.vx = random.uniform(-6, -2)
            
        # Initial upward velocity
        self.vy = random.uniform(-25, -18)
        
        if self.is_bomb:
            self.type = 'bomb'
            self.splash_color = DARK_GREY
        else:
            choices = [('apple', RED), ('watermelon', GREEN), ('orange', ORANGE)]
            self.type, self.splash_color = random.choice(choices)
            
        self.image = IMAGES[self.type]
        self.active = True

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += GRAVITY
        
        # Deactivate if it falls off the bottom of the screen
        if self.y > HEIGHT + 100:
            self.active = False

    def draw(self, surface):
        # Draw the image centered at self.x, self.y
        top_left = (int(self.x - self.radius), int(self.y - self.radius))
        surface.blit(self.image, top_left)


class Particle:
    """Dynamic water splash particle"""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        
        # Explosion physics
        self.vx = random.uniform(-15, 15)
        self.vy = random.uniform(-20, 5)
        self.radius = random.randint(4, 15)
        self.life = 255 # alpha
        self.decay = random.randint(5, 15)
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += GRAVITY
        self.life -= self.decay
        
    def draw(self, surface):
        if self.life > 0:
            # Draw particle with alpha
            s = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, max(0, self.life)), (self.radius, self.radius), self.radius)
            surface.blit(s, (int(self.x) - self.radius, int(self.y) - self.radius))


def point_line_distance(pt, line_start, line_end):
    """Calculate minimum distance from a point pt to a line segment."""
    pt = np.array(pt, dtype=float)
    line_start = np.array(line_start, dtype=float)
    line_end = np.array(line_end, dtype=float)
    
    line_vec = line_end - line_start
    p_vec = pt - line_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    
    # If the line segment is a point
    if line_len_sq == 0.0:
        return np.linalg.norm(pt - line_start)
        
    # Project point onto the line segment (clamped to [0, 1])
    t = np.dot(p_vec, line_vec) / line_len_sq
    t = max(0.0, min(1.0, t))
    
    projection = line_start + t * line_vec
    return np.linalg.norm(pt - projection)


def draw_star(surface, x, y, size, color, alpha=255, angle=0):
    """Draws a glowing 5-pointed star with custom color, size, alpha, and angle."""
    outer_radius = size
    inner_radius = size * 0.4
    points = []
    for i in range(10):
        r = outer_radius if i % 2 == 0 else inner_radius
        a = angle + i * np.pi / 5 - np.pi / 2
        px = r * np.cos(a)
        py = r * np.sin(a)
        points.append((px, py))
    
    surf_size = max(10, int(outer_radius * 2.8))
    s = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
    cx, cy = surf_size // 2, surf_size // 2
    
    # Outer glow
    glow_points = [(cx + px * 1.3, cy + py * 1.3) for px, py in points]
    glow_alpha = max(0, min(255, int(alpha * 0.45)))
    pygame.draw.polygon(s, (255, 255, 120, glow_alpha), glow_points)
    
    # Main star body
    r, g, b = color
    shifted_points = [(cx + px, cy + py) for px, py in points]
    body_alpha = max(0, min(255, int(alpha)))
    pygame.draw.polygon(s, (r, g, b, body_alpha), shifted_points)
    
    # Inner highlight for depth
    inner_points = [(cx + px * 0.45, cy + py * 0.45) for px, py in points]
    pygame.draw.polygon(s, (255, 255, 240, body_alpha), inner_points)
    
    surface.blit(s, (int(x - cx), int(y - cy)))


class StarReward:
    """Animated blinking star visual reward shown on top of the screen when fruit is destroyed."""
    def __init__(self, x=None, y=45, is_center_burst=False):
        if x is None:
            self.x = WIDTH // 2 + random.randint(-180, 180)
        else:
            self.x = x
        self.y = y
        self.is_center_burst = is_center_burst
        self.base_size = random.uniform(30, 50) if is_center_burst else random.uniform(18, 35)
        self.size = self.base_size
        self.life = 45 # frames (~0.75s)
        self.max_life = 45
        self.angle = random.uniform(0, 6.28)
        self.rot_speed = random.uniform(-0.15, 0.15)
        self.vx = random.uniform(-3, 3) if is_center_burst else random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-2, 1) if is_center_burst else random.uniform(-0.6, 0.2)
        self.blink_speed = random.uniform(0.3, 0.6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.angle += self.rot_speed
        self.life -= 1
        
        # Rapid scale pulsation (blinking pulse)
        phase = (self.max_life - self.life) * self.blink_speed
        self.size = self.base_size * (0.75 + 0.45 * np.abs(np.sin(phase)))

    def draw(self, surface):
        if self.life > 0:
            life_ratio = self.life / self.max_life
            # Blinking opacity oscillation
            phase = (self.max_life - self.life) * self.blink_speed * 1.5
            blink_mult = 0.5 + 0.5 * np.sin(phase)
            alpha = min(255, int(255 * life_ratio * blink_mult))
            
            color = (255, 215, 0) if (self.life // 3) % 2 == 0 else (255, 245, 120) # Blinking gold/yellow alternate
            draw_star(surface, self.x, self.y, self.size, color, alpha, self.angle)


class FloatingRewardText:
    """Floating reward banner text at top of screen."""
    def __init__(self, text, x=WIDTH // 2, y=65, font=None):
        self.text = text
        self.x = x
        self.y = y
        self.life = 50
        self.max_life = 50
        self.font = font

    def update(self):
        self.y -= 0.6
        self.life -= 1

    def draw(self, surface):
        if self.life > 0 and self.font:
            alpha = min(255, int((self.life / self.max_life) * 300))
            # Create blinking color flash
            color = (255, 220, 0) if (self.life // 4) % 2 == 0 else (255, 255, 200)
            txt_surf = self.font.render(self.text, True, color)
            shadow_surf = self.font.render(self.text, True, (0, 0, 0))
            
            # Apply alpha blending
            txt_s = pygame.Surface(txt_surf.get_size(), pygame.SRCALPHA)
            shadow_s = pygame.Surface(shadow_surf.get_size(), pygame.SRCALPHA)
            
            txt_s.blit(txt_surf, (0, 0))
            shadow_s.blit(shadow_surf, (0, 0))
            
            txt_s.set_alpha(alpha)
            shadow_s.set_alpha(alpha)
            
            rx = int(self.x - txt_surf.get_width() // 2)
            ry = int(self.y - txt_surf.get_height() // 2)
            surface.blit(shadow_s, (rx + 2, ry + 2))
            surface.blit(txt_s, (rx, ry))



def main():
    # 1. Initialization
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Body Movement Fruit Ninja 3D")
    clock = pygame.time.Clock()
    
    # Load 3D Asset Images
    global IMAGES
    asset_dir = os.path.join(os.path.dirname(__file__), 'assets')
    IMAGES['apple'] = load_and_scale_image(os.path.join(asset_dir, 'apple.jpg'), FRUIT_RADIUS)
    IMAGES['watermelon'] = load_and_scale_image(os.path.join(asset_dir, 'watermelon.jpg'), FRUIT_RADIUS)
    IMAGES['orange'] = load_and_scale_image(os.path.join(asset_dir, 'orange.jpg'), FRUIT_RADIUS)
    IMAGES['bomb'] = load_and_scale_image(os.path.join(asset_dir, 'bomb.jpg'), BOMB_RADIUS)

    # Initialize OpenCV
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # Initialize MediaPipe Tasks HandLandmarker
    base_options = BaseOptions(model_asset_path='hand_landmarker.task')
    options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=VisionRunningMode.IMAGE
    )
    detector = HandLandmarker.create_from_options(options)
    
    # Game State
    objects = []
    particles = []
    star_rewards = []
    reward_texts = []
    score = 0
    lives = 5
    frame_count = 0
    
    font = pygame.font.SysFont('Arial', 64, bold=True)
    small_font = pygame.font.SysFont('Arial', 48, bold=True)
    
    # Track previous finger positions to create slicing lines
    prev_finger_pos = {} # Maps hand index to (x, y)
    
    game_over = False

    try:
        while True:
            # Handle Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_r and game_over:
                        # Reset game
                        game_over = False
                        score = 0
                        lives = 5
                        objects.clear()
                        particles.clear()
                        star_rewards.clear()
                        reward_texts.clear()
                        prev_finger_pos.clear()

            if not game_over:
                # Capture frame from webcam
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                # Mirror frame horizontally (CRITICAL step for intuitive control)
                frame = cv2.flip(frame, 1)
                
                # Convert BGR (OpenCV) to RGB (MediaPipe & Pygame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe Tasks API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)
                
                # Current finger positions
                curr_finger_pos = {}
                
                if detection_result.hand_landmarks:
                    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        # Landmark 8 is the tip of the index finger
                        index_finger = hand_landmarks[8]
                        x = int(index_finger.x * WIDTH)
                        y = int(index_finger.y * HEIGHT)
                        curr_finger_pos[idx] = (x, y)
                
                # Convert frame to Pygame Surface
                # The shape is (height, width, 3), pygame needs (width, height, 3)
                surf = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                
                # Update Game Logic
                frame_count += 1
                if frame_count % SPAWN_RATE == 0:
                    # Spawn objects
                    is_bomb = random.random() < 0.15 # 15% chance for a bomb
                    objects.append(GameObject(is_bomb))
                
                # Draw trails and handle slicing
                for idx, pos in curr_finger_pos.items():
                    if idx in prev_finger_pos:
                        start_pos = prev_finger_pos[idx]
                        end_pos = pos
                        
                        # Draw trail
                        pygame.draw.line(screen, TRAIL_COLOR, start_pos, end_pos, 15)
                        pygame.draw.circle(screen, TRAIL_COLOR, end_pos, 8)
                        
                        # Check collisions with objects
                        for obj in objects[:]: # iterate over copy to allow removal
                            if not obj.active:
                                continue
                            
                            dist = point_line_distance((obj.x, obj.y), start_pos, end_pos)
                            
                            if dist <= obj.radius * 0.8: # Slightly smaller hitbox than the image radius
                                obj.active = False
                                
                                # Spawn water splashes / explosion
                                num_particles = 60 if obj.is_bomb else 40
                                for _ in range(num_particles):
                                    particles.append(Particle(obj.x, obj.y, obj.splash_color))
                                
                                if obj.is_bomb:
                                    lives -= 1
                                    if lives <= 0:
                                        game_over = True
                                else:
                                    score += 50000
                                    # Trigger blinking star rewards on top of the screen
                                    for _ in range(6):
                                        star_rewards.append(StarReward(is_center_burst=True))
                                    star_rewards.append(StarReward(x=WIDTH // 2 - 140, y=40))
                                    star_rewards.append(StarReward(x=WIDTH // 2 + 140, y=40))
                                    reward_texts.append(FloatingRewardText("+50,000", x=WIDTH // 2, y=65, font=small_font))
                                    
                    # Update previous position for the next frame
                    prev_finger_pos[idx] = pos
                    
                # Clean up lost hands from prev_finger_pos
                active_hands = set(curr_finger_pos.keys())
                for idx in list(prev_finger_pos.keys()):
                    if idx not in active_hands:
                        del prev_finger_pos[idx]

                # Update & Draw Objects
                for obj in objects:
                    obj.update()
                    if obj.active:
                        obj.draw(screen)
                
                # Update & Draw Particles
                for p in particles:
                    p.update()
                    p.draw(screen)
                    
                # Filter out inactive objects and finished particles
                objects = [o for o in objects if o.active]
                particles = [p for p in particles if p.life > 0]
                
                # Update & Draw Star Rewards (Top of Screen)
                for star in star_rewards:
                    star.update()
                    star.draw(screen)

                for rt in reward_texts:
                    rt.update()
                    rt.draw(screen)

                star_rewards = [s for s in star_rewards if s.life > 0]
                reward_texts = [rt for rt in reward_texts if rt.life > 0]

                # Draw UI
                score_text = font.render(f"Score: {score}", True, WHITE)
                score_shadow = font.render(f"Score: {score}", True, BLACK)
                screen.blit(score_shadow, (22, 22))
                screen.blit(score_text, (20, 20))
                
                lives_text = font.render(f"♥️: {lives}", True, RED)
                # lives_shadow = font.render(f"🖤: {lives}", True, BLACK)
                # screen.blit(lives_shadow, (WIDTH - 250 + 2, 22))
                screen.blit(lives_text, (WIDTH - 250, 20))
                
            else:
                # Game Over Screen
                overlay = pygame.Surface((WIDTH, HEIGHT))
                overlay.set_alpha(150)
                overlay.fill((0, 0, 0))
                screen.blit(overlay, (0, 0))
                
                go_text = font.render("GAME OVER", True, RED)
                restart_text = small_font.render("Press 'R' to Restart or 'ESC' to Quit", True, WHITE)
                score_display = small_font.render(f"Final Score: {score}", True, WHITE)
                
                screen.blit(go_text, (WIDTH//2 - go_text.get_width()//2, HEIGHT//2 - 100))
                screen.blit(score_display, (WIDTH//2 - score_display.get_width()//2, HEIGHT//2))
                screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 80))
            
            pygame.display.flip()
            clock.tick(FPS)
            
    finally:
        cap.release()
        pygame.quit()

if __name__ == "__main__":
    main()
