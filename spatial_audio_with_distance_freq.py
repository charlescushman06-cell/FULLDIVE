#!/usr/bin/env python3
"""
Dream Space Audio - Spatial environment through sound (HRTF version)
Train your brain to perceive 2D/3D space via audio encoding.

Uses Head-Related Transfer Functions (HRTF) for realistic 3D audio positioning.

Controls:
  WASD - Move (cardinal directions)
  Q/E  - Look left/right
  ESC  - Quit

Audio encoding:
  - Each direction sensor produces clicks convolved with real HRTF
  - Click rate = distance (faster = closer)
  - Your brain perceives sounds as positioned in 3D space around your head
"""

import numpy as np
import sounddevice as sd
import threading
import time
import sys
import math
import argparse
import tty
import termios
import select
from scipy import signal
import sofar

# ============ CONFIGURATION ============
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024

# Room dimensions - configurable per game mode
CELL_SIZE = 10.0  # 10 units wide hallways
MAZE_COLS = 80
MAZE_ROWS = 80
ROOM_WIDTH = CELL_SIZE * MAZE_COLS   # 800 units (default, overridden per mode)
ROOM_HEIGHT = CELL_SIZE * MAZE_ROWS  # 800 units (default, overridden per mode)
WALL_THICKNESS = 0.5  # For collision detection

# Vertical dimensions (3D simulation)
CEILING_HEIGHT = 8.0   # Ceiling is 8 units above floor
PLAYER_HEIGHT = 1.7    # Player "eyes" are at 1.7 units

# Game mode globals (set by mode selection)
GAME_MODE = 'maze'  # 'maze' or 'forest'
INTERNAL_WALLS = []
ORBS = []
WALL_GRID = {}

# HRTF data (loaded at startup)
HRTF = None
HRTF_POSITIONS = None  # (azimuth, elevation) pairs
HRTF_LEFT = None       # Left ear IRs
HRTF_RIGHT = None      # Right ear IRs


def load_hrtf(filepath="hrtf/mit_kemar.sofa"):
    """Load HRTF data from SOFA file"""
    global HRTF, HRTF_POSITIONS, HRTF_LEFT, HRTF_RIGHT
    
    print(f"Loading HRTF from {filepath}...")
    hrtf = sofar.read_sofa(filepath)
    
    HRTF = hrtf
    # Extract positions (azimuth, elevation) - ignore distance
    HRTF_POSITIONS = hrtf.SourcePosition[:, :2]  # (N, 2) - azimuth, elevation
    HRTF_LEFT = hrtf.Data_IR[:, 0, :]   # (N, 512) - left ear
    HRTF_RIGHT = hrtf.Data_IR[:, 1, :]  # (N, 512) - right ear
    
    print(f"  Loaded {len(HRTF_POSITIONS)} HRTF positions")
    print(f"  IR length: {HRTF_LEFT.shape[1]} samples ({HRTF_LEFT.shape[1]/SAMPLE_RATE*1000:.1f}ms)")


def find_nearest_hrtf(azimuth_deg, elevation_deg=0):
    """Find index of nearest HRTF measurement to given direction.
    
    Azimuth: 0° = front, 90° = right, -90° = left, 180° = back
    Elevation: 0° = horizontal, 90° = up, -90° = down
    """
    # Angular distance (handle wraparound)
    az_diff = np.abs(HRTF_POSITIONS[:, 0] - (azimuth_deg % 360))
    az_diff = np.minimum(az_diff, 360 - az_diff)
    el_diff = np.abs(HRTF_POSITIONS[:, 1] - elevation_deg)
    
    distances = np.sqrt(az_diff**2 + el_diff**2)
    return np.argmin(distances)


def interpolate_hrtf(azimuth_deg, elevation_deg=0, n_neighbors=4):
    """Get interpolated HRTF by blending nearest measurements.
    
    Returns (left_ir, right_ir) as weighted average of nearest HRTFs.
    This gives smooth transitions between measurement positions.
    """
    # Angular distance to all positions (handle wraparound)
    az_diff = np.abs(HRTF_POSITIONS[:, 0] - (azimuth_deg % 360))
    az_diff = np.minimum(az_diff, 360 - az_diff)
    el_diff = np.abs(HRTF_POSITIONS[:, 1] - elevation_deg)
    
    distances = np.sqrt(az_diff**2 + el_diff**2)
    
    # Get n nearest neighbors
    nearest_indices = np.argsort(distances)[:n_neighbors]
    nearest_distances = distances[nearest_indices]
    
    # If we're exactly on a measurement point, just return it
    if nearest_distances[0] < 0.1:
        return HRTF_LEFT[nearest_indices[0]].copy(), HRTF_RIGHT[nearest_indices[0]].copy()
    
    # Inverse distance weighting
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (nearest_distances + 0.01)
    weights /= weights.sum()  # Normalize to sum to 1
    
    # Weighted average of IRs
    left_ir = np.zeros_like(HRTF_LEFT[0])
    right_ir = np.zeros_like(HRTF_RIGHT[0])
    
    for idx, weight in zip(nearest_indices, weights):
        left_ir += HRTF_LEFT[idx] * weight
        right_ir += HRTF_RIGHT[idx] * weight
    
    return left_ir, right_ir


# Speed of sound for room acoustics (meters per second, but we use game units ~= meters)
SPEED_OF_SOUND = 343.0  # m/s


def generate_maze(cols, rows, cell_size, loop_factor=0.3):
    """Generate maze with loops (no dead-ends)."""
    import random
    
    cells = [[{'walls': [True, True, True, True], 'visited': False} 
              for _ in range(cols)] for _ in range(rows)]
    
    def get_neighbors(r, c):
        neighbors = []
        if r > 0 and not cells[r-1][c]['visited']:
            neighbors.append((r-1, c, 'top'))
        if c < cols-1 and not cells[r][c+1]['visited']:
            neighbors.append((r, c+1, 'right'))
        if r < rows-1 and not cells[r+1][c]['visited']:
            neighbors.append((r+1, c, 'bottom'))
        if c > 0 and not cells[r][c-1]['visited']:
            neighbors.append((r, c-1, 'left'))
        return neighbors
    
    def remove_wall(r1, c1, direction):
        if direction == 'top':
            cells[r1][c1]['walls'][0] = False
            cells[r1-1][c1]['walls'][2] = False
        elif direction == 'right':
            cells[r1][c1]['walls'][1] = False
            cells[r1][c1+1]['walls'][3] = False
        elif direction == 'bottom':
            cells[r1][c1]['walls'][2] = False
            cells[r1+1][c1]['walls'][0] = False
        elif direction == 'left':
            cells[r1][c1]['walls'][3] = False
            cells[r1][c1-1]['walls'][1] = False
    
    stack = [(0, 0)]
    cells[0][0]['visited'] = True
    
    while stack:
        r, c = stack[-1]
        neighbors = get_neighbors(r, c)
        
        if neighbors:
            nr, nc, direction = random.choice(neighbors)
            remove_wall(r, c, direction)
            cells[nr][nc]['visited'] = True
            stack.append((nr, nc))
        else:
            stack.pop()
    
    removable = []
    for r in range(rows):
        for c in range(cols):
            if r < rows - 1 and cells[r][c]['walls'][0]:
                removable.append((r, c, 'top'))
            if c < cols - 1 and cells[r][c]['walls'][1]:
                removable.append((r, c, 'right'))
    
    num_to_remove = int(len(removable) * loop_factor)
    walls_to_remove = random.sample(removable, min(num_to_remove, len(removable)))
    
    for r, c, direction in walls_to_remove:
        if direction == 'top':
            cells[r][c]['walls'][0] = False
            cells[r+1][c]['walls'][2] = False
        elif direction == 'right':
            cells[r][c]['walls'][1] = False
            cells[r][c+1]['walls'][3] = False
    
    walls = []
    for r in range(rows):
        for c in range(cols):
            x = c * cell_size
            y = r * cell_size
            
            if cells[r][c]['walls'][0]:
                walls.append((x, y + cell_size, x + cell_size, y + cell_size))
            if cells[r][c]['walls'][1]:
                walls.append((x + cell_size, y, x + cell_size, y + cell_size))
            if r == 0 and cells[r][c]['walls'][2]:
                walls.append((x, y, x + cell_size, y))
            if c == 0 and cells[r][c]['walls'][3]:
                walls.append((x, y, x, y + cell_size))
    
    return walls


def generate_forest(world_size=500):
    """Generate a forest world with cabins and scattered trees/bushes."""
    import random
    
    walls = []
    cabin_positions = []
    
    CABIN_SPACING = 125.0
    
    for cx in range(int(CABIN_SPACING), int(world_size), int(CABIN_SPACING)):
        for cy in range(int(CABIN_SPACING), int(world_size), int(CABIN_SPACING)):
            cabin_positions.append((float(cx), float(cy)))
    
    cabin_size = 8.0
    hs = cabin_size / 2
    
    for cabin_x, cabin_y in cabin_positions:
        walls.append((cabin_x - hs, cabin_y + hs, cabin_x + hs, cabin_y + hs))
        walls.append((cabin_x + hs, cabin_y - hs, cabin_x + hs, cabin_y + hs))
        walls.append((cabin_x - hs, cabin_y - hs, cabin_x - hs, cabin_y + hs))
        walls.append((cabin_x - hs, cabin_y - hs, cabin_x - 1, cabin_y - hs))
        walls.append((cabin_x + 1, cabin_y - hs, cabin_x + hs, cabin_y - hs))
    
    num_trees = 120
    tree_positions = []
    
    for _ in range(num_trees):
        for attempt in range(20):
            tx = random.uniform(20, world_size - 20)
            ty = random.uniform(20, world_size - 20)
            
            too_close_cabin = False
            for cx, cy in cabin_positions:
                if abs(tx - cx) < 20 and abs(ty - cy) < 20:
                    too_close_cabin = True
                    break
            if too_close_cabin:
                continue
            
            if tx < 40 and ty < 40:
                continue
            
            too_close = False
            for px, py, _ in tree_positions:
                if math.sqrt((tx - px)**2 + (ty - py)**2) < 8:
                    too_close = True
                    break
            
            if not too_close:
                tree_size = random.uniform(2, 4)
                tree_positions.append((tx, ty, tree_size))
                break
    
    for tx, ty, size in tree_positions:
        ts = size / 2
        walls.append((tx - ts, ty - ts, tx + ts, ty - ts))
        walls.append((tx + ts, ty - ts, tx + ts, ty + ts))
        walls.append((tx + ts, ty + ts, tx - ts, ty + ts))
        walls.append((tx - ts, ty + ts, tx - ts, ty - ts))
    
    num_bushes = 150
    
    for _ in range(num_bushes):
        for attempt in range(10):
            bx = random.uniform(10, world_size - 10)
            by = random.uniform(10, world_size - 10)
            
            too_close_cabin = False
            for cx, cy in cabin_positions:
                if abs(bx - cx) < 15 and abs(by - cy) < 15:
                    too_close_cabin = True
                    break
            if too_close_cabin:
                continue
            
            if bx < 30 and by < 30:
                continue
            
            bush_size = random.uniform(0.8, 1.5)
            bs = bush_size / 2
            
            walls.append((bx - bs, by - bs, bx + bs, by - bs))
            walls.append((bx + bs, by - bs, bx + bs, by + bs))
            walls.append((bx + bs, by + bs, bx - bs, by + bs))
            walls.append((bx - bs, by + bs, bx - bs, by - bs))
            break
    
    return walls, cabin_positions


# Spatial index for fast wall lookups
GRID_SIZE = 20.0

def build_wall_grid(room_width, room_height, internal_walls):
    """Build spatial index for walls"""
    global WALL_GRID
    WALL_GRID = {}
    
    boundary_walls = [
        (0, 0, room_width, 0),
        (room_width, 0, room_width, room_height),
        (room_width, room_height, 0, room_height),
        (0, room_height, 0, 0),
    ]
    all_walls = boundary_walls + internal_walls
    
    for wall in all_walls:
        x1, y1, x2, y2 = wall
        min_gx = int(min(x1, x2) / GRID_SIZE)
        max_gx = int(max(x1, x2) / GRID_SIZE)
        min_gy = int(min(y1, y2) / GRID_SIZE)
        max_gy = int(max(y1, y2) / GRID_SIZE)
        
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                key = (gx, gy)
                if key not in WALL_GRID:
                    WALL_GRID[key] = []
                WALL_GRID[key].append(wall)


def init_maze_mode():
    """Initialize the maze game mode"""
    global GAME_MODE, ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS, CELL_SIZE
    
    GAME_MODE = 'maze'
    CELL_SIZE = 10.0
    ROOM_WIDTH = CELL_SIZE * MAZE_COLS
    ROOM_HEIGHT = CELL_SIZE * MAZE_ROWS
    
    INTERNAL_WALLS = generate_maze(MAZE_COLS, MAZE_ROWS, CELL_SIZE)
    
    starting_top_wall = (0, CELL_SIZE, CELL_SIZE, CELL_SIZE)
    INTERNAL_WALLS = [w for w in INTERNAL_WALLS if w != starting_top_wall]
    starting_right_wall = (CELL_SIZE, 0, CELL_SIZE, CELL_SIZE)
    INTERNAL_WALLS = [w for w in INTERNAL_WALLS if w != starting_right_wall]
    
    ORB_SPACING = 200.0
    ORBS = []
    ORB_CHORDS = [
        [220, 277, 330],
        [196, 247, 294],
        [262, 330, 392],
        [294, 370, 440],
        [175, 220, 262],
        [247, 311, 370],
    ]
    
    for ox in range(int(ORB_SPACING/2), int(ROOM_WIDTH), int(ORB_SPACING)):
        for oy in range(int(ORB_SPACING/2), int(ROOM_HEIGHT), int(ORB_SPACING)):
            chord_idx = (ox // int(ORB_SPACING) + oy // int(ORB_SPACING)) % len(ORB_CHORDS)
            ORBS.append({
                'pos': (float(ox), float(oy)),
                'chord': ORB_CHORDS[chord_idx]
            })
    
    player['x'] = CELL_SIZE / 2
    player['y'] = CELL_SIZE / 2
    player['angle'] = 0.0
    player['target_angle'] = 0.0
    
    build_wall_grid(ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS)


def init_forest_mode():
    """Initialize the forest game mode"""
    global GAME_MODE, ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS
    
    GAME_MODE = 'forest'
    ROOM_WIDTH = 500.0
    ROOM_HEIGHT = 500.0
    
    INTERNAL_WALLS, cabin_positions = generate_forest(500)
    
    CABIN_CHORDS = [
        [262, 330, 392],
        [220, 277, 330],
        [196, 247, 294],
        [175, 220, 262],
        [294, 370, 440],
        [247, 311, 370],
        [233, 294, 349],
        [165, 208, 247],
        [277, 349, 415],
    ]
    
    ORBS = []
    for i, pos in enumerate(cabin_positions):
        chord = CABIN_CHORDS[i % len(CABIN_CHORDS)]
        ORBS.append({
            'pos': pos,
            'chord': chord
        })
    
    player['x'] = 25.0
    player['y'] = 25.0
    player['angle'] = math.radians(45)
    player['target_angle'] = player['angle']
    
    build_wall_grid(ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS)


def get_nearby_walls(x, y, radius=50.0):
    """Get walls near a position using spatial index"""
    walls = set()
    min_gx = int((x - radius) / GRID_SIZE)
    max_gx = int((x + radius) / GRID_SIZE)
    min_gy = int((y - radius) / GRID_SIZE)
    max_gy = int((y + radius) / GRID_SIZE)
    
    for gx in range(min_gx, max_gx + 1):
        for gy in range(min_gy, max_gy + 1):
            key = (gx, gy)
            if key in WALL_GRID:
                walls.update(WALL_GRID[key])
    
    return list(walls)


# Player state
player = {
    'x': 5.0,
    'y': 5.0,
    'angle': 0.0,
    'target_angle': 0.0,
    'move_speed': 0.5,
    'turn_speed': 0.0262,  # ~30 degrees/second at 20fps
    'turn_smooth': 0.6,
    'was_blocked': False
}


# ============ TERMINAL INPUT ============

class TerminalInput:
    """Non-blocking terminal input with time-based sticky keys.
    
    Keys stay 'held' for a duration after being pressed, allowing
    movement + turning at the same time despite terminal limitations.
    """
    def __init__(self):
        self.old_settings = None
        self.key_times = {}  # key -> last time seen
        self.sticky_duration = 0.25  # 250ms - keys stay "held" this long after last press
        
    def start(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
    def stop(self):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def update(self):
        now = time.time()
        
        # Read ALL available input
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'D':
                            self.key_times['left'] = now
                        elif ch3 == 'C':
                            self.key_times['right'] = now
                        elif ch3 == 'A':
                            self.key_times['up'] = now
                        elif ch3 == 'B':
                            self.key_times['down'] = now
                else:
                    self.key_times['esc'] = now
            else:
                self.key_times[ch.lower()] = now
        
        # Clean up old keys
        expired = [k for k, t in self.key_times.items() if now - t > self.sticky_duration * 2]
        for k in expired:
            del self.key_times[k]
    
    def is_pressed(self, key):
        if key not in self.key_times:
            return False
        return (time.time() - self.key_times[key]) < self.sticky_duration


# ============ HRTF AUDIO ENGINE ============

class HRTFAudioEngine:
    """Audio engine using interpolated HRTF convolution for smooth 3D spatial audio."""
    
    def __init__(self):
        self.running = True
        
        # === HORIZONTAL SENSORS ===
        # Angles relative to forward (0°), positive = right, negative = left
        self.sensor_angles = []
        
        # Fovea: 0°, ±2°, ±4°, ±6°, ±8°, ±10° (11 sensors)
        for deg in range(0, 11, 2):
            if deg == 0:
                self.sensor_angles.append(0)
            else:
                self.sensor_angles.extend([-deg, deg])
        
        # Parafovea: ±13°, ±16°, ±19° (6 sensors)
        for deg in range(13, 20, 3):
            self.sensor_angles.extend([-deg, deg])
        
        # Peripheral: ±22°, ±27°, ±32°, ±37° (8 sensors)
        for deg in range(22, 40, 5):
            self.sensor_angles.extend([-deg, deg])
        
        # Far: ±42°, ±49°, ±56° (6 sensors)
        for deg in range(42, 60, 7):
            self.sensor_angles.extend([-deg, deg])
        
        # Extreme: ±63°, ±73°, ±83°, ±90° (8 sensors)
        for deg in [63, 73, 83, 90]:
            self.sensor_angles.extend([-deg, deg])
        
        self.num_horizontal = len(self.sensor_angles)
        
        # === VERTICAL SENSORS ===
        # (azimuth_offset, elevation) - uses HRTF elevation for natural up/down perception
        # Elevation: positive = up, negative = down
        self.vertical_sensors = [
            # Upward sensors
            (0, 45),      # Up-forward
            (0, 70),      # Nearly straight up
            (-45, 40),    # Up-left
            (45, 40),     # Up-right
            # Downward sensors  
            (0, -30),     # Down-forward
            (0, -60),     # Steep down
            (-45, -30),   # Down-left
            (45, -30),    # Down-right
        ]
        self.num_vertical = len(self.vertical_sensors)
        self.vertical_timers = [0.0] * self.num_vertical
        
        self.num_sensors = self.num_horizontal + self.num_vertical
        print(f"  {self.num_horizontal} horizontal + {self.num_vertical} vertical = {self.num_sensors} total sensors")
        print(f"  HRTF interpolation enabled (smooth transitions)")
        print(f"  3D perception via HRTF elevation (up/down)")
        print(f"  Room acoustics enabled (early reflections)")
        
        # Click timing
        self.click_timers = [0.0] * self.num_horizontal
        
        # Pre-generate base click waveform (mono)
        self.base_click = self._generate_base_click()
        
        # HRTF cache for real-time interpolation
        # Key: (rounded_azimuth, rounded_elevation) -> (left_conv, right_conv)
        self.hrtf_cache = {}
        self.cache_resolution = 2.0  # Cache every 2 degrees
        
        # Room acoustics: reflection directions (relative to player)
        self.reflection_angles = [0, 45, 90, 135, 180, -135, -90, -45]
        self.reflection_timers = [0.0] * len(self.reflection_angles)
        self.reflection_click = self._generate_reflection_click()
        
        # Wall collision sound (centered)
        self.wall_hit = False
        self.wall_thud = self._generate_wall_thud()
        
        # Contact beep
        self.contact_timer = 0
        
        # Orb chord phases
        self.orb_phases = {}
        
        # Output buffer for overlap-add (longer for reflections)
        self.output_buffer_left = np.zeros(BUFFER_SIZE * 8)
        self.output_buffer_right = np.zeros(BUFFER_SIZE * 8)
        self.buffer_pos = 0
        
    def _generate_base_click(self, duration=0.008):
        """Generate base click waveform (mono) - broadband for distance filtering"""
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        # Soft click with smooth envelope - richer harmonics for better filtering
        envelope = np.sin(np.pi * t / duration)
        # Mix of frequencies for broadband click (500 + 1500 + 3000 Hz)
        tone = (np.sin(2 * np.pi * 500 * t) * 0.5 + 
                np.sin(2 * np.pi * 1500 * t) * 0.3 +
                np.sin(2 * np.pi * 3000 * t) * 0.2) * envelope
        return tone.astype(np.float32)
    
    def _filter_by_distance(self, audio, distance):
        """Apply distance-based low-pass filter. Close = bright, far = muffled."""
        # Cutoff frequency: close (0m) = 8000Hz, far (15m+) = 800Hz
        max_cutoff = 8000
        min_cutoff = 800
        max_dist = 15.0
        
        clamped_dist = max(0, min(distance, max_dist))
        # Linear interpolation of cutoff frequency
        cutoff = max_cutoff - (clamped_dist / max_dist) * (max_cutoff - min_cutoff)
        
        # Normalize cutoff for filter design (Nyquist = SAMPLE_RATE/2)
        normalized_cutoff = cutoff / (SAMPLE_RATE / 2)
        normalized_cutoff = max(0.01, min(0.99, normalized_cutoff))  # Keep in valid range
        
        # Design simple low-pass filter (2nd order Butterworth)
        try:
            b, a = signal.butter(2, normalized_cutoff, btype='low')
            filtered = signal.lfilter(b, a, audio)
            return filtered.astype(np.float32)
        except:
            return audio  # Fall back to unfiltered if filter fails
    
    def _generate_reflection_click(self, duration=0.006):
        """Generate softer click for reflections (more muffled)"""
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.sin(np.pi * t / duration)
        # Lower frequency, softer = absorbed by walls
        tone = np.sin(2 * np.pi * 350 * t) * envelope
        return tone.astype(np.float32)
    
    def _generate_wall_thud(self, duration=0.08):
        """Low thud for wall collision"""
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.exp(-t * 30)
        tone = (np.sin(2 * np.pi * 80 * t) * 0.6 + 
                np.sin(2 * np.pi * 120 * t) * 0.3 +
                np.random.uniform(-0.1, 0.1, samples) * envelope)
        return (tone * envelope * 0.5).astype(np.float32)
    
    def get_hrtf_click(self, world_angle_deg, elevation_deg=0, volume=0.5, distance=5.0):
        """Get HRTF-convolved click for a world angle + elevation, with distance-based filtering."""
        # Round to cache resolution for HRTF lookup
        az_key = round(world_angle_deg / self.cache_resolution) * self.cache_resolution
        el_key = round(elevation_deg / self.cache_resolution) * self.cache_resolution
        
        # Distance is bucketed for caching (0-3, 3-6, 6-10, 10-15, 15+)
        if distance < 3:
            dist_key = 1.5
        elif distance < 6:
            dist_key = 4.5
        elif distance < 10:
            dist_key = 8.0
        elif distance < 15:
            dist_key = 12.5
        else:
            dist_key = 20.0
        
        cache_key = (az_key, el_key, dist_key)
        
        if cache_key not in self.hrtf_cache:
            # SOFA convention: positive azimuth = left, we use positive = right
            sofa_azimuth = -az_key
            
            # Get interpolated HRTF with elevation
            left_ir, right_ir = interpolate_hrtf(sofa_azimuth, elevation_deg=el_key, n_neighbors=4)
            
            # Apply distance-based filtering BEFORE HRTF convolution
            filtered_click = self._filter_by_distance(self.base_click, dist_key)
            
            # Convolve filtered click with HRTF
            left_conv = signal.fftconvolve(filtered_click, left_ir, mode='full')
            right_conv = signal.fftconvolve(filtered_click, right_ir, mode='full')
            
            # Normalize
            max_val = max(np.max(np.abs(left_conv)), np.max(np.abs(right_conv)), 0.001)
            left_conv = (left_conv / max_val).astype(np.float32)
            right_conv = (right_conv / max_val).astype(np.float32)
            
            self.hrtf_cache[cache_key] = (left_conv, right_conv)
            
            # Limit cache size
            if len(self.hrtf_cache) > 2000:
                keys = list(self.hrtf_cache.keys())
                for k in keys[:300]:
                    del self.hrtf_cache[k]
        
        left, right = self.hrtf_cache[cache_key]
        return left * volume, right * volume
    
    def get_reflection_click(self, world_angle_deg, distance):
        """Get HRTF-convolved reflection click with distance-based filtering."""
        sofa_azimuth = -world_angle_deg
        
        # Get interpolated HRTF for reflection direction
        left_ir, right_ir = interpolate_hrtf(sofa_azimuth, elevation_deg=0, n_neighbors=4)
        
        # Convolve reflection click with HRTF
        left_conv = signal.fftconvolve(self.reflection_click, left_ir, mode='full')
        right_conv = signal.fftconvolve(self.reflection_click, right_ir, mode='full')
        
        # Distance-based attenuation and filtering
        # Further reflections are quieter and more muffled
        attenuation = max(0.1, 1.0 - distance / 30.0) * 0.3  # Max 30% of direct sound
        
        max_val = max(np.max(np.abs(left_conv)), np.max(np.abs(right_conv)), 0.001)
        left_conv = (left_conv / max_val * attenuation).astype(np.float32)
        right_conv = (right_conv / max_val * attenuation).astype(np.float32)
        
        return left_conv, right_conv
    
    def ray_hits_wall(self, x1, y1, x2, y2, rx, ry, rdx, rdy):
        """Check if ray hits wall segment, return distance or None"""
        wx, wy = x2 - x1, y2 - y1
        denom = rdx * wy - rdy * wx
        if abs(denom) < 1e-10:
            return None
        t = ((x1 - rx) * wy - (y1 - ry) * wx) / denom
        u = ((x1 - rx) * rdy - (y1 - ry) * rdx) / denom
        if t > 0 and 0 <= u <= 1:
            return t
        return None
    
    def get_distances(self):
        """Get distance to wall for each sensor, plus world angles for HRTF"""
        x, y, angle = player['x'], player['y'], player['angle']
        nearby_walls = get_nearby_walls(x, y, radius=30.0)
        
        distances = []
        world_angles = []  # Store world angles for HRTF lookup
        
        for sensor_angle in self.sensor_angles:
            # Convert sensor angle to world direction
            world_angle = angle + math.radians(sensor_angle)
            dx = math.sin(world_angle)
            dy = math.cos(world_angle)
            
            min_dist = 100.0
            for wall in nearby_walls:
                hit_dist = self.ray_hits_wall(wall[0], wall[1], wall[2], wall[3], x, y, dx, dy)
                if hit_dist is not None and hit_dist < min_dist:
                    min_dist = hit_dist
            
            distances.append(max(0.1, min_dist))
            world_angles.append(math.degrees(world_angle))
        
        return distances, world_angles
    
    def get_reflection_distances(self):
        """Get distances to walls for room acoustics reflections."""
        x, y, angle = player['x'], player['y'], player['angle']
        nearby_walls = get_nearby_walls(x, y, radius=50.0)
        
        reflection_data = []  # (distance, world_angle_deg)
        
        for ref_angle in self.reflection_angles:
            world_angle = angle + math.radians(ref_angle)
            dx = math.sin(world_angle)
            dy = math.cos(world_angle)
            
            min_dist = 100.0
            for wall in nearby_walls:
                hit_dist = self.ray_hits_wall(wall[0], wall[1], wall[2], wall[3], x, y, dx, dy)
                if hit_dist is not None and hit_dist < min_dist:
                    min_dist = hit_dist
            
            reflection_data.append((min_dist, math.degrees(world_angle)))
        
        return reflection_data
    
    def get_vertical_distances(self):
        """Get distances for vertical sensors (ceiling/floor awareness).
        
        Returns list of (distance, azimuth_world_deg, elevation_deg) for each vertical sensor.
        In our flat world, vertical distances are simulated based on ceiling/floor height.
        """
        x, y, angle = player['x'], player['y'], player['angle']
        nearby_walls = get_nearby_walls(x, y, radius=30.0)
        
        vertical_data = []
        
        for az_offset, elevation in self.vertical_sensors:
            # World azimuth
            world_az = math.degrees(angle) + az_offset
            
            # Calculate vertical distance component
            if elevation > 0:
                # Looking up - distance to ceiling
                # At 90° elevation, straight up to ceiling
                # At 45°, diagonal distance
                vert_dist = (CEILING_HEIGHT - PLAYER_HEIGHT) / math.sin(math.radians(abs(elevation)))
            else:
                # Looking down - distance to floor
                vert_dist = PLAYER_HEIGHT / math.sin(math.radians(abs(elevation)))
            
            # Also check horizontal component - might hit a wall before floor/ceiling
            # For diagonal rays, project horizontally and check walls
            horiz_component = vert_dist * math.cos(math.radians(abs(elevation)))
            
            if horiz_component > 0.5:  # Only check walls if there's horizontal travel
                horiz_angle = angle + math.radians(az_offset)
                dx = math.sin(horiz_angle)
                dy = math.cos(horiz_angle)
                
                for wall in nearby_walls:
                    hit_dist = self.ray_hits_wall(wall[0], wall[1], wall[2], wall[3], x, y, dx, dy)
                    if hit_dist is not None:
                        # Convert wall hit to diagonal distance at this elevation
                        diag_dist = hit_dist / math.cos(math.radians(abs(elevation)))
                        vert_dist = min(vert_dist, diag_dist)
            
            vertical_data.append((max(0.5, min(vert_dist, 20.0)), world_az, elevation))
        
        return vertical_data
    
    def distance_to_click_rate(self, distance):
        """Convert distance to clicks per second"""
        max_rate = 20.0
        min_rate = 1.0
        max_dist = 15.0
        clamped = max(0, min(distance, max_dist))
        return max_rate - (clamped / max_dist) * (max_rate - min_rate)
    
    def reflection_rate(self, distance):
        """Reflection click rate - slower than direct, based on room size feel."""
        # Reflections click slower - they indicate room boundaries
        max_rate = 8.0
        min_rate = 0.5
        max_dist = 30.0
        clamped = max(0, min(distance, max_dist))
        return max_rate - (clamped / max_dist) * (max_rate - min_rate)
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Real-time audio callback with interpolated HRTF and room acoustics"""
        if not self.running:
            outdata.fill(0)
            return
        
        # Get distances and world angles for all sensors
        distances, world_angles = self.get_distances()
        
        # Get reflection data for room acoustics
        reflection_data = self.get_reflection_distances()
        
        # Start with silence
        left_channel = np.zeros(frames, dtype=np.float32)
        right_channel = np.zeros(frames, dtype=np.float32)
        
        # Add any remaining output from previous buffer (overlap-add)
        take = min(frames, len(self.output_buffer_left) - self.buffer_pos)
        if take > 0:
            left_channel[:take] += self.output_buffer_left[self.buffer_pos:self.buffer_pos+take]
            right_channel[:take] += self.output_buffer_right[self.buffer_pos:self.buffer_pos+take]
            self.buffer_pos += take
        
        # Reset buffer if exhausted
        if self.buffer_pos >= len(self.output_buffer_left) - BUFFER_SIZE * 2:
            self.output_buffer_left = np.zeros(BUFFER_SIZE * 8, dtype=np.float32)
            self.output_buffer_right = np.zeros(BUFFER_SIZE * 8, dtype=np.float32)
            self.buffer_pos = 0
        
        dt = frames / SAMPLE_RATE
        
        # === DIRECT SOUND: Process each sensor with real-time HRTF interpolation ===
        for i, (dist, world_angle) in enumerate(zip(distances, world_angles)):
            rate = self.distance_to_click_rate(dist)
            self.click_timers[i] += dt
            
            if self.click_timers[i] >= 1.0 / rate:
                self.click_timers[i] = 0
                
                # Get HRTF-convolved click with distance-based filtering (bright=close, muffled=far)
                left_click, right_click = self.get_hrtf_click(world_angle, distance=dist)
                
                # Add to output (overlap-add for the tail)
                click_len = len(left_click)
                end_pos = min(frames, click_len)
                
                left_channel[:end_pos] += left_click[:end_pos]
                right_channel[:end_pos] += right_click[:end_pos]
                
                # Store overflow in buffer
                if click_len > frames:
                    overflow = click_len - frames
                    buf_end = min(overflow, len(self.output_buffer_left))
                    self.output_buffer_left[:buf_end] += left_click[frames:frames+buf_end]
                    self.output_buffer_right[:buf_end] += right_click[frames:frames+buf_end]
        
        # === ROOM ACOUSTICS: Early reflections ===
        for i, (ref_dist, ref_world_angle) in enumerate(reflection_data):
            if ref_dist > 25.0:  # Only add reflections for nearby walls
                continue
            
            rate = self.reflection_rate(ref_dist)
            self.reflection_timers[i] += dt
            
            if self.reflection_timers[i] >= 1.0 / rate:
                self.reflection_timers[i] = 0
                
                # Calculate delay based on sound travel time (round trip to wall)
                delay_samples = int((ref_dist * 2 / SPEED_OF_SOUND) * SAMPLE_RATE)
                delay_samples = min(delay_samples, BUFFER_SIZE * 4)  # Cap delay
                
                # Get reflection click with HRTF for wall direction
                left_ref, right_ref = self.get_reflection_click(ref_world_angle, ref_dist)
                
                # Add to buffer with delay (early reflection)
                ref_len = len(left_ref)
                start_pos = delay_samples
                end_pos = start_pos + ref_len
                
                # Place in output buffer (delayed)
                if start_pos < frames:
                    # Part goes in current frame
                    cur_end = min(frames, end_pos)
                    cur_len = cur_end - start_pos
                    left_channel[start_pos:cur_end] += left_ref[:cur_len]
                    right_channel[start_pos:cur_end] += right_ref[:cur_len]
                    
                    # Rest goes in overflow buffer
                    if end_pos > frames:
                        buf_start = 0
                        buf_len = min(end_pos - frames, len(self.output_buffer_left))
                        self.output_buffer_left[buf_start:buf_start+buf_len] += left_ref[cur_len:cur_len+buf_len]
                        self.output_buffer_right[buf_start:buf_start+buf_len] += right_ref[cur_len:cur_len+buf_len]
                else:
                    # All goes in overflow buffer (delayed)
                    buf_start = start_pos - frames
                    buf_end = min(buf_start + ref_len, len(self.output_buffer_left))
                    buf_len = buf_end - buf_start
                    if buf_len > 0:
                        self.output_buffer_left[buf_start:buf_end] += left_ref[:buf_len]
                        self.output_buffer_right[buf_start:buf_end] += right_ref[:buf_len]
        
        # === VERTICAL SENSORS: Ceiling/floor awareness with HRTF elevation ===
        vertical_data = self.get_vertical_distances()
        for i, (vert_dist, world_az, elevation) in enumerate(vertical_data):
            # Slower click rate for vertical - ambient awareness, not navigation
            rate = self.distance_to_click_rate(vert_dist) * 0.5  # Half speed
            self.vertical_timers[i] += dt
            
            if self.vertical_timers[i] >= 1.0 / rate:
                self.vertical_timers[i] = 0
                
                # Get HRTF-convolved click with proper elevation and distance filtering
                # Vertical sounds are slightly quieter to not overwhelm horizontal
                left_click, right_click = self.get_hrtf_click(world_az, elevation_deg=elevation, volume=0.35, distance=vert_dist)
                
                # Add to output
                click_len = len(left_click)
                end_pos = min(frames, click_len)
                
                left_channel[:end_pos] += left_click[:end_pos]
                right_channel[:end_pos] += right_click[:end_pos]
                
                if click_len > frames:
                    overflow = click_len - frames
                    buf_end = min(overflow, len(self.output_buffer_left))
                    self.output_buffer_left[:buf_end] += left_click[frames:frames+buf_end]
                    self.output_buffer_right[:buf_end] += right_click[frames:frames+buf_end]
        
        # Wall collision thud
        if self.wall_hit:
            self.wall_hit = False
            thud_len = min(len(self.wall_thud), frames)
            left_channel[:thud_len] += self.wall_thud[:thud_len]
            right_channel[:thud_len] += self.wall_thud[:thud_len]
        
        # Orb/cabin chords
        px, py = player['x'], player['y']
        t = np.arange(frames) / SAMPLE_RATE
        
        for orb_idx, orb in enumerate(ORBS):
            orb_x, orb_y = orb['pos']
            orb_dist = math.sqrt((orb_x - px)**2 + (orb_y - py)**2)
            
            if orb_dist > 150:
                continue
            
            orb_volume = max(0, min(0.15, (100 - orb_dist) / 100 * 0.15))
            
            # Angle to orb relative to player facing
            orb_world_angle = math.atan2(orb_x - px, orb_y - py)
            relative_angle = math.degrees(orb_world_angle - player['angle'])
            
            # Get HRTF for orb direction
            idx = find_nearest_hrtf(-relative_angle, elevation_deg=0)
            
            # Simple stereo panning for continuous tones (full HRTF convolution would be expensive)
            pan = math.sin(math.radians(relative_angle))
            left_vol = orb_volume * (1 - max(0, pan))
            right_vol = orb_volume * (1 + min(0, pan))
            
            # Generate chord
            for freq in orb['chord']:
                if orb_idx not in self.orb_phases:
                    self.orb_phases[orb_idx] = {}
                if freq not in self.orb_phases[orb_idx]:
                    self.orb_phases[orb_idx][freq] = 0.0
                
                phase = self.orb_phases[orb_idx][freq]
                tone = np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
                self.orb_phases[orb_idx][freq] = phase + 2 * np.pi * freq * dt
                
                left_channel += tone * left_vol / 3
                right_channel += tone * right_vol / 3
        
        # Output
        outdata[:, 0] = np.clip(left_channel, -1, 1)
        outdata[:, 1] = np.clip(right_channel, -1, 1)
    
    def start(self):
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            callback=self.audio_callback,
            blocksize=BUFFER_SIZE,
            dtype=np.float32
        )
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()


# ============ INPUT HANDLING ============

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def can_move_to(new_x, new_y):
    if new_x < WALL_THICKNESS or new_x > ROOM_WIDTH - WALL_THICKNESS:
        return False
    if new_y < WALL_THICKNESS or new_y > ROOM_HEIGHT - WALL_THICKNESS:
        return False
    
    nearby_walls = get_nearby_walls(new_x, new_y, radius=5.0)
    for wall in nearby_walls:
        x1, y1, x2, y2 = wall
        dist = point_to_segment_dist(new_x, new_y, x1, y1, x2, y2)
        if dist < WALL_THICKNESS:
            return False
    return True


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def lerp_angle(current, target, t):
    diff = normalize_angle(target - current)
    return current + diff * t


def handle_input(term, engine):
    term.update()
    
    new_x, new_y = player['x'], player['y']
    moved = False
    angle = player['angle']
    
    if term.is_pressed('w') or term.is_pressed('up'):
        new_x += math.sin(angle) * player['move_speed']
        new_y += math.cos(angle) * player['move_speed']
        moved = True
    if term.is_pressed('s') or term.is_pressed('down'):
        new_x -= math.sin(angle) * player['move_speed']
        new_y -= math.cos(angle) * player['move_speed']
        moved = True
    
    if term.is_pressed('a'):
        new_x -= math.cos(angle) * player['move_speed']
        new_y += math.sin(angle) * player['move_speed']
        moved = True
    if term.is_pressed('d'):
        new_x += math.cos(angle) * player['move_speed']
        new_y -= math.sin(angle) * player['move_speed']
        moved = True
    
    if term.is_pressed('q'):
        player['target_angle'] -= player['turn_speed']
    if term.is_pressed('e'):
        player['target_angle'] += player['turn_speed']
    
    player['angle'] = lerp_angle(player['angle'], player['target_angle'], player['turn_smooth'])
    player['angle'] = normalize_angle(player['angle'])
    player['target_angle'] = normalize_angle(player['target_angle'])
    
    if moved:
        actually_moved = False
        if can_move_to(new_x, new_y):
            player['x'] = new_x
            player['y'] = new_y
            actually_moved = True
        elif can_move_to(new_x, player['y']):
            player['x'] = new_x
            actually_moved = True
        elif can_move_to(player['x'], new_y):
            player['y'] = new_y
            actually_moved = True
        
        if not actually_moved:
            if not player['was_blocked']:
                engine.wall_hit = True
                player['was_blocked'] = True
        else:
            player['was_blocked'] = False
    else:
        player['was_blocked'] = False
    
    return term.is_pressed('esc')


# ============ DISPLAY ============

def draw_first_person(engine, width=120, height=24):
    x, y, angle = player['x'], player['y'], player['angle']
    fov = math.radians(100)
    nearby_walls = get_nearby_walls(x, y, radius=25.0)
    
    columns = []
    for col in range(width):
        ray_angle = angle - fov/2 + (col / width) * fov
        dx = math.sin(ray_angle)
        dy = math.cos(ray_angle)
        
        min_dist = 100.0
        for wall in nearby_walls:
            x1, y1, x2, y2 = wall
            wx, wy = x2 - x1, y2 - y1
            denom = dx * wy - dy * wx
            if abs(denom) < 1e-10:
                continue
            t = ((x1 - x) * wy - (y1 - y) * wx) / denom
            u = ((x1 - x) * dy - (y1 - y) * dx) / denom
            if t > 0 and 0 <= u <= 1:
                corrected_dist = t * math.cos(ray_angle - angle)
                min_dist = min(min_dist, corrected_dist)
        
        columns.append(min_dist)
    
    view = []
    shade_chars = ' ░▒▓█'
    
    for row in range(height):
        line = ''
        for col, dist in enumerate(columns):
            max_dist = 20.0
            wall_height = max(0, min(1, (max_dist - dist) / max_dist))
            half_wall = int(wall_height * height / 2)
            
            center = height // 2
            dist_from_center = abs(row - center)
            
            if dist_from_center <= half_wall:
                shade_val = max(0, min(1, 1 - dist/max_dist))
                shade_idx = int(shade_val * (len(shade_chars)-1))
                line += shade_chars[shade_idx]
            elif row < center:
                line += '·'
            else:
                line += '.'
        view.append(line)
    
    return '\n'.join(view)


def draw_map(map_size=21, view_radius=40.0):
    px, py = player['x'], player['y']
    
    world_min_x = px - view_radius
    world_max_x = px + view_radius
    world_min_y = py - view_radius
    world_max_y = py + view_radius
    
    scale = map_size / (view_radius * 2)
    
    def world_to_grid(wx, wy):
        gx = int((wx - world_min_x) * scale)
        gy = int((wy - world_min_y) * scale)
        return gx, gy
    
    grid = [['·' for _ in range(map_size)] for _ in range(map_size)]
    
    nearby_walls = get_nearby_walls(px, py, radius=view_radius + 10)
    
    for wall in nearby_walls:
        x1, y1, x2, y2 = wall
        
        if max(x1, x2) < world_min_x or min(x1, x2) > world_max_x:
            continue
        if max(y1, y2) < world_min_y or min(y1, y2) > world_max_y:
            continue
        
        gx1, gy1 = world_to_grid(x1, y1)
        gx2, gy2 = world_to_grid(x2, y2)
        
        gx1, gx2 = max(0, min(map_size-1, gx1)), max(0, min(map_size-1, gx2))
        gy1, gy2 = max(0, min(map_size-1, gy1)), max(0, min(map_size-1, gy2))
        
        if gy1 == gy2:
            row = map_size - 1 - gy1
            if 0 <= row < map_size:
                for gx in range(min(gx1, gx2), max(gx1, gx2) + 1):
                    if 0 <= gx < map_size:
                        grid[row][gx] = '█'
        elif gx1 == gx2:
            col = gx1
            if 0 <= col < map_size:
                for gy in range(min(gy1, gy2), max(gy1, gy2) + 1):
                    row = map_size - 1 - gy
                    if 0 <= row < map_size:
                        grid[row][col] = '█'
    
    for orb in ORBS:
        orb_x, orb_y = orb['pos']
        if orb_x < world_min_x or orb_x > world_max_x:
            continue
        if orb_y < world_min_y or orb_y > world_max_y:
            continue
        orb_gx, orb_gy = world_to_grid(orb_x, orb_y)
        orb_row = map_size - 1 - orb_gy
        if 0 <= orb_row < map_size and 0 <= orb_gx < map_size:
            grid[orb_row][orb_gx] = '◉'
    
    center = map_size // 2
    deg = math.degrees(player['angle']) % 360
    if 337.5 <= deg or deg < 22.5:
        arrow = '↑'
    elif 22.5 <= deg < 67.5:
        arrow = '↗'
    elif 67.5 <= deg < 112.5:
        arrow = '→'
    elif 112.5 <= deg < 157.5:
        arrow = '↘'
    elif 157.5 <= deg < 202.5:
        arrow = '↓'
    elif 202.5 <= deg < 247.5:
        arrow = '↙'
    elif 247.5 <= deg < 292.5:
        arrow = '←'
    else:
        arrow = '↖'
    
    grid[center][center] = arrow
    
    return '\n'.join([''.join(row) for row in grid])


def display_status(engine):
    deg = math.degrees(player['angle']) % 360
    compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int((deg + 22.5) / 45) % 8]
    
    nearest_orb = None
    orb_dist = float('inf')
    for orb in ORBS:
        ox, oy = orb['pos']
        d = math.sqrt((ox - player['x'])**2 + (oy - player['y'])**2)
        if d < orb_dist:
            orb_dist = d
            nearest_orb = orb
    
    orb_x, orb_y = nearest_orb['pos'] if nearest_orb else (0, 0)
    orb_angle = math.degrees(math.atan2(orb_x - player['x'], orb_y - player['y']) - player['angle']) % 360
    if orb_angle > 180:
        orb_angle -= 360
    
    mode_label = "MAZE" if GAME_MODE == 'maze' else "FOREST"
    orb_label = "ORB" if GAME_MODE == 'maze' else "CABIN"
    
    lines = []
    
    lines.append("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    lines.append(f"║                              DREAM SPACE - HRTF Spatial Audio [{mode_label}]                                                 ║")
    lines.append(f"║  Pos: ({player['x']:6.1f},{player['y']:6.1f})  Face: {deg:5.1f}° ({compass:2})    {orb_label}: {orb_dist:5.0f}m {orb_angle:+4.0f}°                                                     ║")
    lines.append("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  WASD=move  Q/E=look  ESC=quit                     Sensors: {engine.num_sensors}                                                       ║")
    lines.append("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    fp_width = 120
    fp_height = 24
    
    fp_view = draw_first_person(engine, width=fp_width, height=fp_height)
    fp_lines = fp_view.split('\n')
    
    view_radius = 35.0 if GAME_MODE == 'maze' else 60.0
    mini_map = draw_map(map_size=21, view_radius=view_radius)
    map_lines = mini_map.split('\n')
    
    map_offset_x = 2
    map_offset_y = 1
    
    for i, map_line in enumerate(map_lines):
        fp_row = map_offset_y + i
        if fp_row < len(fp_lines):
            fp_row_chars = list(fp_lines[fp_row])
            for j, ch in enumerate(map_line):
                col = map_offset_x + j
                if col < len(fp_row_chars):
                    fp_row_chars[col] = ch
            fp_lines[fp_row] = ''.join(fp_row_chars)
    
    lines.append("")
    lines.append("┌" + "─" * fp_width + "┐")
    for line in fp_lines:
        lines.append("│" + line + "│")
    lines.append("└" + "─" * fp_width + "┘")
    
    frame = '\033[2J\033[H' + '\n'.join(lines)
    sys.stdout.write(frame)
    sys.stdout.flush()


# ============ MAIN ============

def show_game_menu():
    print('\033[H\033[J', end='')
    print("╔═══════════════════════════════════════════════════════╗")
    print("║      DREAM SPACE - HRTF Spatial Audio Trainer         ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║                                                       ║")
    print("║   Select Game Mode:                                   ║")
    print("║                                                       ║")
    print("║   [1] MAZE    - 800x800 maze with multiple orbs       ║")
    print("║                 Navigate corridors, find beacons      ║")
    print("║                                                       ║")
    print("║   [2] FOREST  - 500x500 open world with cabins        ║")
    print("║                 Trees & bushes, find your way home    ║")
    print("║                                                       ║")
    print("║   [Q] Quit                                            ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    print("Press 1, 2, or Q: ", end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Dream Space - HRTF Spatial Audio Trainer')
    parser.add_argument('--device', '-d', type=int, help='Audio output device index')
    parser.add_argument('--list-devices', '-l', action='store_true', help='List audio devices')
    parser.add_argument('--mode', '-m', choices=['maze', 'forest'], help='Game mode (skip menu)')
    parser.add_argument('--hrtf', default='hrtf/mit_kemar.sofa', help='Path to HRTF SOFA file')
    args = parser.parse_args()
    
    if args.list_devices:
        print(sd.query_devices())
        return
    
    if args.device is not None:
        sd.default.device[1] = args.device
        print(f"Using: {sd.query_devices(args.device)['name']}")
        time.sleep(1)
    
    # Load HRTF
    load_hrtf(args.hrtf)
    
    # Game mode selection
    selected_mode = args.mode
    
    if not selected_mode:
        show_game_menu()
        
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                ch = sys.stdin.read(1).lower()
                if ch == '1':
                    selected_mode = 'maze'
                    break
                elif ch == '2':
                    selected_mode = 'forest'
                    break
                elif ch == 'q':
                    print("\nGoodbye!")
                    return
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    print('\033[H\033[J', end='')
    if selected_mode == 'maze':
        print("Generating maze...")
        init_maze_mode()
    else:
        print("Generating forest...")
        init_forest_mode()
    
    print()
    print("Initializing HRTF audio engine...")
    engine = HRTFAudioEngine()
    
    print()
    print("HRTF spatial audio: Sounds are positioned in 3D around your head")
    print("Click rate = distance (faster = closer)")
    print()
    print("Press any key to start...")
    
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    engine.start()
    
    term = TerminalInput()
    term.start()
    
    try:
        while True:
            should_quit = handle_input(term, engine)
            if should_quit:
                break
            
            display_status(engine)
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        pass
    finally:
        term.stop()
        engine.stop()
        print('\033[H\033[J', end='')
        print("Done.")


if __name__ == "__main__":
    main()
