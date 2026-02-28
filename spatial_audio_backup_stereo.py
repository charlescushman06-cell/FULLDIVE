#!/usr/bin/env python3
"""
Dream Space Audio - Spatial environment through sound
Train your brain to perceive 2D/3D space via audio encoding.

Controls:
  WASD - Move (cardinal directions)
  Q/E  - Look left/right
  ESC  - Quit

Audio encoding (minimal - for dream navigation):
  - FRONT: Low rumble (280 Hz) - both ears, impossible to miss
  - LEFT:  Click in LEFT EAR only
  - RIGHT: Click in RIGHT EAR only
  - (no back - you don't need it in dreams)
  
  Click rate = distance (faster = closer)
  
Your brain already knows stereo. Just learn: rumble = wall ahead.

When you turn, your "front" changes - so the frequencies swap accordingly!
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

def generate_maze(cols, rows, cell_size, loop_factor=0.3):
    """Generate maze with loops (no dead-ends). 
    
    Uses recursive backtracking, then removes extra walls to create loops.
    loop_factor: 0.0 = perfect maze (all dead-ends), 1.0 = very open (no dead-ends)
    """
    import random
    
    # Each cell tracks which walls exist: [top, right, bottom, left]
    cells = [[{'walls': [True, True, True, True], 'visited': False} 
              for _ in range(cols)] for _ in range(rows)]
    
    def get_neighbors(r, c):
        """Get unvisited neighbors"""
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
        """Remove wall between current cell and neighbor"""
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
    
    # Recursive backtracking from (0, 0)
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
    
    # Add loops: remove some remaining walls to prevent dead-ends
    # Collect all internal walls that still exist
    removable = []
    for r in range(rows):
        for c in range(cols):
            # Top wall (not on top edge)
            if r < rows - 1 and cells[r][c]['walls'][0]:
                removable.append((r, c, 'top'))
            # Right wall (not on right edge)
            if c < cols - 1 and cells[r][c]['walls'][1]:
                removable.append((r, c, 'right'))
    
    # Remove a percentage of walls to create loops
    num_to_remove = int(len(removable) * loop_factor)
    walls_to_remove = random.sample(removable, min(num_to_remove, len(removable)))
    
    for r, c, direction in walls_to_remove:
        if direction == 'top':
            cells[r][c]['walls'][0] = False
            cells[r+1][c]['walls'][2] = False
        elif direction == 'right':
            cells[r][c]['walls'][1] = False
            cells[r][c+1]['walls'][3] = False
    
    # Convert cell walls to line segments
    walls = []
    for r in range(rows):
        for c in range(cols):
            x = c * cell_size
            y = r * cell_size
            
            # Top wall
            if cells[r][c]['walls'][0]:
                walls.append((x, y + cell_size, x + cell_size, y + cell_size))
            # Right wall
            if cells[r][c]['walls'][1]:
                walls.append((x + cell_size, y, x + cell_size, y + cell_size))
            # Bottom wall (only for bottom row)
            if r == 0 and cells[r][c]['walls'][2]:
                walls.append((x, y, x + cell_size, y))
            # Left wall (only for left column)
            if c == 0 and cells[r][c]['walls'][3]:
                walls.append((x, y, x, y + cell_size))
    
    return walls


def generate_forest(world_size=500):
    """Generate a forest world with cabins and scattered trees/bushes.
    
    - Multiple cabins spread throughout (centering beacons with distinct chords)
    - Trees: larger obstacles (3-5m radius, represented as 4-wall squares)
    - Bushes: smaller obstacles (1-2m radius)
    """
    import random
    
    walls = []
    cabin_positions = []
    
    # Grid of cabins every 125 units (so you can always hear at least one)
    # 500 / 125 = 4, so we get a 3x3 grid interior + edges = good coverage
    CABIN_SPACING = 125.0
    
    for cx in range(int(CABIN_SPACING), int(world_size), int(CABIN_SPACING)):
        for cy in range(int(CABIN_SPACING), int(world_size), int(CABIN_SPACING)):
            cabin_positions.append((float(cx), float(cy)))
    
    # Build each cabin (8m x 8m with door on south)
    cabin_size = 8.0
    hs = cabin_size / 2
    
    for cabin_x, cabin_y in cabin_positions:
        # North wall
        walls.append((cabin_x - hs, cabin_y + hs, cabin_x + hs, cabin_y + hs))
        # East wall
        walls.append((cabin_x + hs, cabin_y - hs, cabin_x + hs, cabin_y + hs))
        # West wall
        walls.append((cabin_x - hs, cabin_y - hs, cabin_x - hs, cabin_y + hs))
        # South wall with door gap (2m opening in center)
        walls.append((cabin_x - hs, cabin_y - hs, cabin_x - 1, cabin_y - hs))
        walls.append((cabin_x + 1, cabin_y - hs, cabin_x + hs, cabin_y - hs))
    
    # Generate trees (larger, spread throughout)
    num_trees = 120
    tree_positions = []
    
    for _ in range(num_trees):
        for attempt in range(20):
            tx = random.uniform(20, world_size - 20)
            ty = random.uniform(20, world_size - 20)
            
            # Skip if too close to any cabin
            too_close_cabin = False
            for cx, cy in cabin_positions:
                if abs(tx - cx) < 20 and abs(ty - cy) < 20:
                    too_close_cabin = True
                    break
            if too_close_cabin:
                continue
            
            # Skip if too close to spawn point (bottom-left area)
            if tx < 40 and ty < 40:
                continue
            
            # Skip if too close to other trees
            too_close = False
            for px, py, _ in tree_positions:
                if math.sqrt((tx - px)**2 + (ty - py)**2) < 8:
                    too_close = True
                    break
            
            if not too_close:
                tree_size = random.uniform(2, 4)
                tree_positions.append((tx, ty, tree_size))
                break
    
    # Convert trees to wall segments (square obstacles)
    for tx, ty, size in tree_positions:
        ts = size / 2
        walls.append((tx - ts, ty - ts, tx + ts, ty - ts))  # South
        walls.append((tx + ts, ty - ts, tx + ts, ty + ts))  # East
        walls.append((tx + ts, ty + ts, tx - ts, ty + ts))  # North
        walls.append((tx - ts, ty + ts, tx - ts, ty - ts))  # West
    
    # Generate bushes (smaller, denser clusters)
    num_bushes = 150
    
    for _ in range(num_bushes):
        for attempt in range(10):
            bx = random.uniform(10, world_size - 10)
            by = random.uniform(10, world_size - 10)
            
            # Skip cabin areas
            too_close_cabin = False
            for cx, cy in cabin_positions:
                if abs(bx - cx) < 15 and abs(by - cy) < 15:
                    too_close_cabin = True
                    break
            if too_close_cabin:
                continue
            
            # Skip spawn area
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
GRID_SIZE = 20.0  # Grid cell size for spatial indexing

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
        # Find all grid cells this wall touches
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
    ROOM_WIDTH = CELL_SIZE * MAZE_COLS   # 800 units
    ROOM_HEIGHT = CELL_SIZE * MAZE_ROWS  # 800 units
    
    # Generate maze
    INTERNAL_WALLS = generate_maze(MAZE_COLS, MAZE_ROWS, CELL_SIZE)
    
    # Ensure starting cell (0,0) has an exit
    starting_top_wall = (0, CELL_SIZE, CELL_SIZE, CELL_SIZE)
    INTERNAL_WALLS = [w for w in INTERNAL_WALLS if w != starting_top_wall]
    starting_right_wall = (CELL_SIZE, 0, CELL_SIZE, CELL_SIZE)
    INTERNAL_WALLS = [w for w in INTERNAL_WALLS if w != starting_right_wall]
    
    # Grid of orbs every 200 units
    ORB_SPACING = 200.0
    ORBS = []
    ORB_CHORDS = [
        [220, 277, 330],   # A minor
        [196, 247, 294],   # G major
        [262, 330, 392],   # C major
        [294, 370, 440],   # D major
        [175, 220, 262],   # F major
        [247, 311, 370],   # B minor
    ]
    
    for ox in range(int(ORB_SPACING/2), int(ROOM_WIDTH), int(ORB_SPACING)):
        for oy in range(int(ORB_SPACING/2), int(ROOM_HEIGHT), int(ORB_SPACING)):
            chord_idx = (ox // int(ORB_SPACING) + oy // int(ORB_SPACING)) % len(ORB_CHORDS)
            ORBS.append({
                'pos': (float(ox), float(oy)),
                'chord': ORB_CHORDS[chord_idx]
            })
    
    # Player start position
    player['x'] = CELL_SIZE / 2
    player['y'] = CELL_SIZE / 2
    player['angle'] = 0.0
    player['target_angle'] = 0.0
    
    build_wall_grid(ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS)


def init_forest_mode():
    """Initialize the forest game mode"""
    global GAME_MODE, ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS
    
    GAME_MODE = 'forest'
    ROOM_WIDTH = 500.0   # 500m x 500m world
    ROOM_HEIGHT = 500.0
    
    # Generate forest with cabins
    INTERNAL_WALLS, cabin_positions = generate_forest(500)
    
    # Distinct chords for each cabin - spread across musical keys
    CABIN_CHORDS = [
        [262, 330, 392],   # C major (C, E, G) - bright, home
        [220, 277, 330],   # A minor (A, C#, E) - melancholic
        [196, 247, 294],   # G major (G, B, D) - open, pastoral
        [175, 220, 262],   # F major (F, A, C) - warm
        [294, 370, 440],   # D major (D, F#, A) - triumphant
        [247, 311, 370],   # B minor (B, D, F#) - mysterious
        [233, 294, 349],   # Bb major (Bb, D, F) - mellow
        [165, 208, 247],   # E minor (E, G, B) - somber
        [277, 349, 415],   # Db major (Db, F, Ab) - dreamy
    ]
    
    # Create orb at each cabin with distinct chord
    ORBS = []
    for i, pos in enumerate(cabin_positions):
        chord = CABIN_CHORDS[i % len(CABIN_CHORDS)]
        ORBS.append({
            'pos': pos,
            'chord': chord
        })
    
    # Player starts in bottom-left corner
    player['x'] = 25.0
    player['y'] = 25.0
    player['angle'] = math.radians(45)  # Face toward center
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

# Player state (positions set by init_*_mode)
player = {
    'x': 5.0,
    'y': 5.0,
    'angle': 0.0,  # radians, 0 = facing +Y (north)
    'target_angle': 0.0,  # For smooth turning
    'move_speed': 0.5,  # Faster for large maze
    'turn_speed': 0.0157,  # ~18 degrees/second at 20fps
    'turn_smooth': 0.6,  # How fast to lerp toward target angle (0-1, higher = faster)
    'was_blocked': False  # Track if we were blocked last frame (for single wall hit sound)
}

# Audio settings
REFERENCE_FREQ = 220  # Hz - center drone

# Minimal encoding: front + left/right only (no back - irrelevant in dreams)
FREQ_FRONT = 400    # Low-mid pitch = front (audible on all headphones)
FREQ_SIDE = 1200    # Higher pitch for left/right clicks (distinct from front)

CONTACT_FREQ = 1760  # Hz - wall hit beep

# ============ TERMINAL INPUT ============

class TerminalInput:
    """Non-blocking terminal input without needing root.
    
    Uses 'sticky keys' - keys stay active for a few frames after being pressed,
    allowing simultaneous movement + turning even with terminal key repeat limitations.
    """
    def __init__(self):
        self.old_settings = None
        self.key_timers = {}  # key -> frames remaining
        self.sticky_frames = 4  # How many frames a key stays "pressed" after last input
        
    def start(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
    def stop(self):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def update(self):
        """Check for keypresses, update key timers"""
        # Decrement all timers
        expired = []
        for key in self.key_timers:
            self.key_timers[key] -= 1
            if self.key_timers[key] <= 0:
                expired.append(key)
        for key in expired:
            del self.key_timers[key]
        
        # Read all available input
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # Escape sequence
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'D':  # Left arrow
                            self.key_timers['left'] = self.sticky_frames
                        elif ch3 == 'C':  # Right arrow
                            self.key_timers['right'] = self.sticky_frames
                        elif ch3 == 'A':  # Up arrow
                            self.key_timers['up'] = self.sticky_frames
                        elif ch3 == 'B':  # Down arrow
                            self.key_timers['down'] = self.sticky_frames
                else:
                    self.key_timers['esc'] = self.sticky_frames
            else:
                self.key_timers[ch.lower()] = self.sticky_frames
    
    def is_pressed(self, key):
        return key in self.key_timers


# ============ AUDIO GENERATORS ============

class AudioEngine:
    def __init__(self):
        self.phase = 0
        # ~63 sensors: maximum human-perceivable resolution (based on MAA)
        # Fovea (0-10°): every 2° = 11 sensors
        # Parafovea (10-20°): every 3° = 8 sensors  
        # Near peripheral (20-40°): every 5° = 16 sensors
        # Far peripheral (40-60°): every 7° = 12 sensors
        # Extreme (60-90°): every 10° = 14 sensors
        sensor_names = []
        
        # Fovea: 0°, ±2°, ±4°, ±6°, ±8°, ±10° (every 2°)
        for angle in range(0, 11, 2):
            if angle == 0:
                sensor_names.append('fov_0')
            else:
                sensor_names.extend([f'fov_L{angle}', f'fov_R{angle}'])
        
        # Parafovea: ±13°, ±16°, ±19° (every 3°, from 13 to 19)
        for angle in range(13, 20, 3):
            sensor_names.extend([f'para_L{angle}', f'para_R{angle}'])
        
        # Near peripheral: ±22°, ±27°, ±32°, ±37° (every 5°)
        for angle in range(22, 40, 5):
            sensor_names.extend([f'periph_L{angle}', f'periph_R{angle}'])
        
        # Far peripheral: ±42°, ±49°, ±56° (every 7°)
        for angle in range(42, 60, 7):
            sensor_names.extend([f'far_L{angle}', f'far_R{angle}'])
        
        # Extreme: ±63°, ±73°, ±83°, ±90° (every ~10°)
        for angle in [63, 73, 83, 90]:
            sensor_names.extend([f'ext_L{angle}', f'ext_R{angle}'])
        
        self.click_timers = {d: 0 for d in sensor_names}
        self.num_sensors = len(sensor_names)  # Track count
        self.contact_timer = 0
        self.running = True
        self.last_fc_dist = 0  # Debug: track front-center distance
        # Orb chord phases
        self.orb_phases = [0.0, 0.0, 0.0]
        # Wall collision trigger
        self.wall_hit = False
        
    def ray_hits_wall(self, x1, y1, x2, y2, rx, ry, rdx, rdy):
        """Check if ray hits a wall segment. Returns distance or None."""
        # Wall vector
        wx, wy = x2 - x1, y2 - y1
        
        # Solve for intersection
        denom = rdx * wy - rdy * wx
        if abs(denom) < 1e-10:
            return None  # Parallel
        
        t = ((x1 - rx) * wy - (y1 - ry) * wx) / denom
        u = ((x1 - rx) * rdy - (y1 - ry) * rdx) / denom
        
        if t > 0 and 0 <= u <= 1:
            return t
        return None
    
    def get_distances(self):
        """Calculate distance to walls in each direction"""
        x, y, angle = player['x'], player['y'], player['angle']
        
        # Maximum human-perceivable resolution (~63 sensors)
        # Spacing based on Minimum Audible Angle research
        
        dirs = {}
        
        # Fovea: 0°, ±2°, ±4°, ±6°, ±8°, ±10° (every 2°) - 11 sensors
        dirs['fov_0'] = (math.sin(angle), math.cos(angle))
        for deg in range(2, 11, 2):
            dirs[f'fov_L{deg}'] = (math.sin(angle - math.radians(deg)), math.cos(angle - math.radians(deg)))
            dirs[f'fov_R{deg}'] = (math.sin(angle + math.radians(deg)), math.cos(angle + math.radians(deg)))
        
        # Parafovea: ±13°, ±16°, ±19° (every 3°) - 6 sensors
        for deg in range(13, 20, 3):
            dirs[f'para_L{deg}'] = (math.sin(angle - math.radians(deg)), math.cos(angle - math.radians(deg)))
            dirs[f'para_R{deg}'] = (math.sin(angle + math.radians(deg)), math.cos(angle + math.radians(deg)))
        
        # Near peripheral: ±22°, ±27°, ±32°, ±37° (every 5°) - 8 sensors
        for deg in range(22, 40, 5):
            dirs[f'periph_L{deg}'] = (math.sin(angle - math.radians(deg)), math.cos(angle - math.radians(deg)))
            dirs[f'periph_R{deg}'] = (math.sin(angle + math.radians(deg)), math.cos(angle + math.radians(deg)))
        
        # Far peripheral: ±42°, ±49°, ±56° (every 7°) - 6 sensors
        for deg in range(42, 60, 7):
            dirs[f'far_L{deg}'] = (math.sin(angle - math.radians(deg)), math.cos(angle - math.radians(deg)))
            dirs[f'far_R{deg}'] = (math.sin(angle + math.radians(deg)), math.cos(angle + math.radians(deg)))
        
        # Extreme: ±63°, ±73°, ±83°, ±90° (every ~10°) - 8 sensors
        for deg in [63, 73, 83, 90]:
            dirs[f'ext_L{deg}'] = (math.sin(angle - math.radians(deg)), math.cos(angle - math.radians(deg)))
            dirs[f'ext_R{deg}'] = (math.sin(angle + math.radians(deg)), math.cos(angle + math.radians(deg)))
        
        # Get only nearby walls (spatial optimization)
        nearby_walls = get_nearby_walls(x, y, radius=30.0)
        
        distances = {}
        for name, (dx, dy) in dirs.items():
            min_dist = 100.0  # Max distance
            
            # Check nearby walls only
            for wall in nearby_walls:
                hit_dist = self.ray_hits_wall(wall[0], wall[1], wall[2], wall[3], x, y, dx, dy)
                if hit_dist is not None and hit_dist < min_dist:
                    min_dist = hit_dist
            
            distances[name] = max(0.1, min_dist)
        
        return distances
    
    def distance_to_click_rate(self, distance):
        """Convert distance to clicks per second (closer = faster)"""
        max_rate = 20.0
        min_rate = 1.0
        max_dist = 15.0  # ~1.5 cell widths for 10-unit cells
        
        # Linear: 10 clicks/s at dist=0, 0.5 clicks/s at dist=15
        clamped_dist = max(0, min(distance, max_dist))
        rate = max_rate - (clamped_dist / max_dist) * (max_rate - min_rate)
        return rate
    
    def generate_click(self, freq, duration=0.005):
        """Ultra-short blip - just a few cycles"""
        samples = int(SAMPLE_RATE * duration)  # 220 samples at 5ms
        t = np.linspace(0, duration, samples, False)
        # Simple half-sine envelope
        envelope = np.sin(np.pi * t / duration)
        tone = np.sin(2 * np.pi * freq * t) * envelope
        return tone * 0.7
    
    def generate_tone(self, freq, duration=0.08):
        """Generate a longer sustained tone (for front rumble)"""
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        # Soft attack and release
        envelope = np.sin(np.pi * t / duration)  # smooth bell curve
        tone = np.sin(2 * np.pi * freq * t) * envelope
        return tone * 0.7
    
    def generate_front_click(self, duration=0.005):
        """Ultra-short blip - crisp and distinct"""
        samples = int(SAMPLE_RATE * duration)  # 220 samples at 5ms
        t = np.linspace(0, duration, samples, False)
        # Simple half-sine - very clean
        envelope = np.sin(np.pi * t / duration)
        tone = np.sin(2 * np.pi * 400 * t) * envelope
        return tone * 0.8
    
    def generate_wall_hit(self, duration=0.08):
        """Low thud sound for wall collision - distinct from other sounds"""
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        # Low frequency thud with fast decay
        envelope = np.exp(-t * 30)
        # Mix of low freqs for "thud" feel
        tone = (np.sin(2 * np.pi * 80 * t) * 0.6 + 
                np.sin(2 * np.pi * 120 * t) * 0.3 +
                np.random.uniform(-0.1, 0.1, samples) * envelope)  # Add noise for texture
        return tone * envelope * 0.7
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Real-time audio callback"""
        if not self.running:
            outdata.fill(0)
            return
            
        t = np.arange(frames) / SAMPLE_RATE
        
        # Get current distances
        distances = self.get_distances()
        
        # Start clean (no drone - was muddying the low end)
        left_channel = np.zeros(frames)
        right_channel = np.zeros(frames)
        
        # HIGH-RES FOVEAL-DENSITY SENSORS: 21 sensors with graduated panning + volume
        # Config: (left_pan, right_pan, volume) - computed from angle
        
        def get_sensor_config(sensor_name):
            """Calculate panning and volume based on sensor angle"""
            # Parse angle from sensor name (format: prefix_L##, prefix_R##, or fov_0)
            if sensor_name == 'fov_0':
                deg = 0
            else:
                # Extract number from end of name
                parts = sensor_name.split('_')
                direction_num = parts[1]  # e.g., "L42" or "R10"
                num = int(direction_num[1:])  # extract number
                deg = -num if direction_num[0] == 'L' else num
            
            # Panning: gradual from center to sides
            pan = deg / 90.0  # -1 to +1
            left_pan = 0.5 - pan * 0.45  # 0.95 (left) to 0.05 (right)
            right_pan = 0.5 + pan * 0.45
            
            # Volume: equal across all sensors
            volume = 1.0
            
            return (left_pan, right_pan, volume)
        
        # All sensors (walls/objects only - no floor/ceiling)
        for direction in distances.keys():
            dist = distances[direction]
            rate = self.distance_to_click_rate(dist)
            
            self.click_timers[direction] += frames / SAMPLE_RATE
            
            if self.click_timers[direction] >= 1.0 / rate:
                self.click_timers[direction] = 0
                click = self.generate_front_click()
                
                if len(click) <= frames:
                    left_pan, right_pan, volume = get_sensor_config(direction)
                    left_channel[:len(click)] += click * left_pan * volume
                    right_channel[:len(click)] += click * right_pan * volume
        
        # Layer 3: Contact beep (if touching wall)
        if any(d < 0.5 for d in distances.values()):
            self.contact_timer += frames / SAMPLE_RATE
            if self.contact_timer > 0.2:
                self.contact_timer = 0
                beep = self.generate_click(CONTACT_FREQ, 0.05)
                if len(beep) <= frames:
                    left_channel[:len(beep)] += beep
                    right_channel[:len(beep)] += beep
        
        # Layer 3.5: Wall collision thud (when movement blocked)
        if self.wall_hit:
            self.wall_hit = False
            thud = self.generate_wall_hit()
            if len(thud) <= frames:
                left_channel[:len(thud)] += thud
                right_channel[:len(thud)] += thud
        
        # Layer 4: Centering orbs - ambient chord beacons spread throughout maze
        px, py = player['x'], player['y']
        orb_left = np.zeros(frames)
        orb_right = np.zeros(frames)
        
        for orb_idx, orb in enumerate(ORBS):
            orb_x, orb_y = orb['pos']
            
            # Distance to this orb
            orb_dist = math.sqrt((orb_x - px)**2 + (orb_y - py)**2)
            
            # Only play orbs within hearing range (150 units)
            if orb_dist > 150:
                continue
            
            # Volume: louder when closer (fades over 100 units)
            orb_volume = max(0, min(0.25, (100 - orb_dist) / 100 * 0.25))
            
            # Angle to orb relative to player facing
            orb_angle = math.atan2(orb_x - px, orb_y - py)  # Angle to orb in world space
            relative_angle = orb_angle - player['angle']  # Relative to player facing
            
            # Pan: -1 (left) to +1 (right)
            pan = math.sin(relative_angle)
            left_vol = orb_volume * (1 - max(0, pan))
            right_vol = orb_volume * (1 + min(0, pan))
            
            # Generate chord (3 frequencies) - use orb's unique chord
            for i, freq in enumerate(orb['chord']):
                # Use a phase offset based on orb index to keep them distinct
                phase_key = orb_idx * 3 + i
                if phase_key >= len(self.orb_phases):
                    self.orb_phases.extend([0.0] * (phase_key - len(self.orb_phases) + 1))
                tone = np.sin(2 * np.pi * freq * (t + self.orb_phases[phase_key]))
                self.orb_phases[phase_key] += frames / SAMPLE_RATE
                orb_left += tone * left_vol / 3
                orb_right += tone * right_vol / 3
        
        left_channel += orb_left
        right_channel += orb_right
        
        # Output stereo
        outdata[:, 0] = np.clip(left_channel, -1, 1)
        outdata[:, 1] = np.clip(right_channel, -1, 1)
    
    def start(self):
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            callback=self.audio_callback,
            blocksize=BUFFER_SIZE
        )
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()


# ============ INPUT HANDLING ============

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Distance from point to line segment"""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

def can_move_to(new_x, new_y):
    """Check if position is valid (not inside walls)"""
    # Check room bounds
    if new_x < WALL_THICKNESS or new_x > ROOM_WIDTH - WALL_THICKNESS:
        return False
    if new_y < WALL_THICKNESS or new_y > ROOM_HEIGHT - WALL_THICKNESS:
        return False
    
    # Check only nearby walls (spatial optimization)
    nearby_walls = get_nearby_walls(new_x, new_y, radius=5.0)
    for wall in nearby_walls:
        x1, y1, x2, y2 = wall
        dist = point_to_segment_dist(new_x, new_y, x1, y1, x2, y2)
        if dist < WALL_THICKNESS:
            return False
    return True

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def lerp_angle(current, target, t):
    """Smoothly interpolate between angles, handling wraparound"""
    diff = normalize_angle(target - current)
    return current + diff * t

def handle_input(term, engine):
    """Handle keyboard input - WASD moves relative to facing, Q/E looks around"""
    term.update()
    
    new_x, new_y = player['x'], player['y']
    moved = False
    angle = player['angle']
    
    # W/S: move forward/backward relative to facing
    if term.is_pressed('w') or term.is_pressed('up'):
        new_x += math.sin(angle) * player['move_speed']
        new_y += math.cos(angle) * player['move_speed']
        moved = True
    if term.is_pressed('s') or term.is_pressed('down'):
        new_x -= math.sin(angle) * player['move_speed']
        new_y -= math.cos(angle) * player['move_speed']
        moved = True
    
    # A/D: strafe left/right
    if term.is_pressed('a'):
        new_x -= math.cos(angle) * player['move_speed']
        new_y += math.sin(angle) * player['move_speed']
        moved = True
    if term.is_pressed('d'):
        new_x += math.cos(angle) * player['move_speed']
        new_y -= math.sin(angle) * player['move_speed']
        moved = True
    
    # Q/E for looking left/right (smooth turning)
    if term.is_pressed('q'):
        player['target_angle'] -= player['turn_speed']
    if term.is_pressed('e'):
        player['target_angle'] += player['turn_speed']
    
    # Smooth turn toward target angle
    player['angle'] = lerp_angle(player['angle'], player['target_angle'], player['turn_smooth'])
    
    # Keep angles normalized
    player['angle'] = normalize_angle(player['angle'])
    player['target_angle'] = normalize_angle(player['target_angle'])
    
    # Only move if valid position (wall collision)
    if moved:
        actually_moved = False
        if can_move_to(new_x, new_y):
            player['x'] = new_x
            player['y'] = new_y
            actually_moved = True
        elif can_move_to(new_x, player['y']):
            # Try sliding along X
            player['x'] = new_x
            actually_moved = True
        elif can_move_to(player['x'], new_y):
            # Try sliding along Y
            player['y'] = new_y
            actually_moved = True
        
        # Wall hit: only trigger sound on FIRST frame of collision
        if not actually_moved:
            if not player['was_blocked']:
                engine.wall_hit = True
                player['was_blocked'] = True
        else:
            player['was_blocked'] = False
    else:
        # Not trying to move, reset blocked state
        player['was_blocked'] = False
    
    return term.is_pressed('esc')


def distance_to_click_rate(distance):
    """Convert distance to clicks per second"""
    max_rate = 20.0
    min_rate = 1.0
    max_dist = 15.0  # ~1.5 cell widths
    
    clamped_dist = max(0, min(distance, max_dist))
    rate = max_rate - (clamped_dist / max_dist) * (max_rate - min_rate)
    return rate


def rate_bar(rate, max_rate=20.0, width=10):
    """Visual bar showing intensity"""
    filled = int((rate / max_rate) * width)
    return '█' * filled + '░' * (width - filled)


def draw_first_person(engine, width=60, height=15):
    """Render first-person ASCII view using raycasting"""
    x, y, angle = player['x'], player['y'], player['angle']
    
    # Field of view: 100 degrees (matching our 5 front sensors at ±50°)
    fov = math.radians(100)
    
    # Get nearby walls once (spatial optimization)
    nearby_walls = get_nearby_walls(x, y, radius=25.0)
    
    # Cast a ray for each column
    columns = []
    for col in range(width):
        # Ray angle: spread across FOV
        ray_angle = angle - fov/2 + (col / width) * fov
        dx = math.sin(ray_angle)
        dy = math.cos(ray_angle)
        
        # Find closest wall hit
        min_dist = 100.0
        
        for wall in nearby_walls:
            x1, y1, x2, y2 = wall
            # Ray-segment intersection
            wx, wy = x2 - x1, y2 - y1
            denom = dx * wy - dy * wx
            if abs(denom) < 1e-10:
                continue
            t = ((x1 - x) * wy - (y1 - y) * wx) / denom
            u = ((x1 - x) * dy - (y1 - y) * dx) / denom
            if t > 0 and 0 <= u <= 1:
                # Fix fisheye: multiply by cos of angle difference
                corrected_dist = t * math.cos(ray_angle - angle)
                min_dist = min(min_dist, corrected_dist)
        
        columns.append(min_dist)
    
    # Render: wall height inversely proportional to distance
    view = []
    shade_chars = ' ░▒▓█'  # Distance shading: far to near
    
    for row in range(height):
        line = ''
        for col, dist in enumerate(columns):
            # Wall height: closer = taller
            max_dist = 20.0
            wall_height = max(0, min(1, (max_dist - dist) / max_dist))
            half_wall = int(wall_height * height / 2)
            
            center = height // 2
            dist_from_center = abs(row - center)
            
            if dist_from_center <= half_wall:
                # Wall - shade by distance (clamp to valid range)
                shade_val = max(0, min(1, 1 - dist/max_dist))
                shade_idx = int(shade_val * (len(shade_chars)-1))
                line += shade_chars[shade_idx]
            elif row < center:
                line += '·'  # Ceiling
            else:
                line += '.'  # Floor
        view.append(line)
    
    return '\n'.join(view)


def draw_map(distances, map_size=21, view_radius=40.0):
    """Draw ASCII local mini-map centered on player"""
    
    # Player is always at center of map
    px, py = player['x'], player['y']
    
    # World coords visible: [px - view_radius, px + view_radius] etc
    world_min_x = px - view_radius
    world_max_x = px + view_radius
    world_min_y = py - view_radius
    world_max_y = py + view_radius
    
    scale = map_size / (view_radius * 2)
    
    def world_to_grid(wx, wy):
        """Convert world coords to grid coords"""
        gx = int((wx - world_min_x) * scale)
        gy = int((wy - world_min_y) * scale)
        return gx, gy
    
    # Create empty map
    grid = [['·' for _ in range(map_size)] for _ in range(map_size)]
    
    # Get only nearby walls (spatial optimization)
    nearby_walls = get_nearby_walls(px, py, radius=view_radius + 10)
    
    for wall in nearby_walls:
        x1, y1, x2, y2 = wall
        
        # Skip walls completely outside view
        if max(x1, x2) < world_min_x or min(x1, x2) > world_max_x:
            continue
        if max(y1, y2) < world_min_y or min(y1, y2) > world_max_y:
            continue
        
        gx1, gy1 = world_to_grid(x1, y1)
        gx2, gy2 = world_to_grid(x2, y2)
        
        # Clamp to grid
        gx1, gx2 = max(0, min(map_size-1, gx1)), max(0, min(map_size-1, gx2))
        gy1, gy2 = max(0, min(map_size-1, gy1)), max(0, min(map_size-1, gy2))
        
        # Draw line
        if gy1 == gy2:  # Horizontal wall
            row = map_size - 1 - gy1
            if 0 <= row < map_size:
                for gx in range(min(gx1, gx2), max(gx1, gx2) + 1):
                    if 0 <= gx < map_size:
                        grid[row][gx] = '█'
        elif gx1 == gx2:  # Vertical wall
            col = gx1
            if 0 <= col < map_size:
                for gy in range(min(gy1, gy2), max(gy1, gy2) + 1):
                    row = map_size - 1 - gy
                    if 0 <= row < map_size:
                        grid[row][col] = '█'
    
    # Draw all orbs within view
    for orb in ORBS:
        orb_x, orb_y = orb['pos']
        # Skip orbs outside view
        if orb_x < world_min_x or orb_x > world_max_x:
            continue
        if orb_y < world_min_y or orb_y > world_max_y:
            continue
        orb_gx, orb_gy = world_to_grid(orb_x, orb_y)
        orb_row = map_size - 1 - orb_gy
        if 0 <= orb_row < map_size and 0 <= orb_gx < map_size:
            grid[orb_row][orb_gx] = '◉'
    
    # Draw player at center with direction arrow
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


def display_status(distances, engine):
    """Full screen display with map and stats - buffered to prevent flicker"""
    rates = {d: distance_to_click_rate(dist) for d, dist in distances.items()}
    
    deg = math.degrees(player['angle']) % 360
    compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int((deg + 22.5) / 45) % 8]
    
    # Find nearest orb
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
    
    # Game mode label
    mode_label = "MAZE" if GAME_MODE == 'maze' else "FOREST"
    orb_label = "ORB" if GAME_MODE == 'maze' else "HOUSE"
    
    # Build entire frame in a buffer
    lines = []
    
    lines.append("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    lines.append(f"║                                    DREAM SPACE - Spatial Audio Trainer [{mode_label}]                                        ║")
    lines.append(f"║  Pos: ({player['x']:6.1f},{player['y']:6.1f})  Face: {deg:5.1f}° ({compass:2})    {orb_label}: {orb_dist:5.0f}m {orb_angle:+4.0f}°                                                     ║")
    lines.append("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
    # Compact sensor display
    fov_sensors = ['fov_0'] + [f'fov_{d}{a}' for a in range(2,11,2) for d in ['L','R']]
    fov_min = min(distances.get(s, 99) for s in fov_sensors if s in distances)
    lines.append(f"║  FOV: {fov_min:4.1f}m    WASD=move  Q/E=look  ESC=quit                                                                           ║")
    lines.append("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # First-person view with mini-map overlay - FULL SIZE
    fp_width = 120
    fp_height = 24
    
    # Generate first-person view as list of strings
    fp_view = draw_first_person(engine, width=fp_width, height=fp_height)
    fp_lines = fp_view.split('\n')
    
    # Generate mini-map (for overlay)
    view_radius = 35.0 if GAME_MODE == 'maze' else 60.0
    mini_map = draw_map(distances, map_size=21, view_radius=view_radius)
    map_lines = mini_map.split('\n')
    
    # Overlay mini-map on top-left of first-person view
    map_offset_x = 2  # Padding from left edge
    map_offset_y = 1  # Padding from top
    
    for i, map_line in enumerate(map_lines):
        fp_row = map_offset_y + i
        if fp_row < len(fp_lines):
            # Convert fp line to list for modification
            fp_row_chars = list(fp_lines[fp_row])
            # Overlay map characters
            for j, ch in enumerate(map_line):
                col = map_offset_x + j
                if col < len(fp_row_chars):
                    fp_row_chars[col] = ch
            fp_lines[fp_row] = ''.join(fp_row_chars)
    
    # Add border and output
    lines.append("")
    lines.append("┌" + "─" * fp_width + "┐")
    for line in fp_lines:
        lines.append("│" + line + "│")
    lines.append("└" + "─" * fp_width + "┘")
    
    # Output entire frame at once: clear screen and print
    frame = '\033[2J\033[H' + '\n'.join(lines)
    sys.stdout.write(frame)
    sys.stdout.flush()


# ============ MAIN ============

def show_game_menu():
    """Display game mode selection menu"""
    print('\033[H\033[J', end='')  # Clear screen
    print("╔═══════════════════════════════════════════════════════╗")
    print("║           DREAM SPACE - Spatial Audio Trainer         ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║                                                       ║")
    print("║   Select Game Mode:                                   ║")
    print("║                                                       ║")
    print("║   [1] MAZE    - 800x800 maze with multiple orbs       ║")
    print("║                 Navigate corridors, find beacons      ║")
    print("║                                                       ║")
    print("║   [2] FOREST  - 500x500 open world with house         ║")
    print("║                 Trees & bushes, find your way home    ║")
    print("║                                                       ║")
    print("║   [Q] Quit                                            ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    print("Press 1, 2, or Q: ", end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Dream Space - Spatial Audio Trainer')
    parser.add_argument('--device', '-d', type=int, help='Audio output device index')
    parser.add_argument('--list-devices', '-l', action='store_true', help='List audio devices')
    parser.add_argument('--mode', '-m', choices=['maze', 'forest'], help='Game mode (skip menu)')
    args = parser.parse_args()
    
    if args.list_devices:
        print(sd.query_devices())
        return
    
    if args.device is not None:
        sd.default.device[1] = args.device
        print(f"Using: {sd.query_devices(args.device)['name']}")
        time.sleep(1)
    
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
    
    # Initialize selected game mode
    print('\033[H\033[J', end='')
    if selected_mode == 'maze':
        print("Generating maze...")
        init_maze_mode()
    else:
        print("Generating forest...")
        init_forest_mode()
    
    print()
    print("Audio: Direction frequencies encode where walls are")
    print("       Click rate = distance (faster = closer)")
    print()
    print("Press any key to start...")
    
    # Wait for keypress
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    engine = AudioEngine()
    engine.start()
    
    term = TerminalInput()
    term.start()
    
    try:
        while True:
            should_quit = handle_input(term, engine)
            if should_quit:
                break
            
            distances = engine.get_distances()
            display_status(distances, engine)
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
