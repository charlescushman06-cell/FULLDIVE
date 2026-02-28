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
from scipy.io import wavfile
import sofar
import os

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

# Mouse tracking flag
MOUSE_ENABLED = True
FACE_TRACKER = None

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
    """Generate maze with loops and guaranteed connectivity.
    
    Uses recursive backtracking (guarantees all cells reachable),
    then adds loops for multiple paths (no dead-end traps).
    """
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
    
    # Recursive backtracking - guarantees ALL cells are connected
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
    
    # Add loops to create multiple paths (prevents dead-end traps)
    removable = []
    for r in range(rows):
        for c in range(cols):
            if r > 0 and cells[r][c]['walls'][0]:  # Top wall exists
                removable.append((r, c, 'top'))
            if c < cols - 1 and cells[r][c]['walls'][1]:  # Right wall exists
                removable.append((r, c, 'right'))
    
    num_to_remove = int(len(removable) * loop_factor)
    walls_to_remove = random.sample(removable, min(num_to_remove, len(removable)))
    
    for r, c, direction in walls_to_remove:
        if direction == 'top' and r > 0:
            cells[r][c]['walls'][0] = False
            cells[r-1][c]['walls'][2] = False
        elif direction == 'right' and c < cols - 1:
            cells[r][c]['walls'][1] = False
            cells[r][c+1]['walls'][3] = False
    
    # Convert to wall segments
    walls = []
    for r in range(rows):
        for c in range(cols):
            x = c * cell_size
            y = r * cell_size
            
            # Top wall of this cell
            if cells[r][c]['walls'][0] and r < rows - 1:
                walls.append((x, y + cell_size, x + cell_size, y + cell_size))
            # Right wall of this cell
            if cells[r][c]['walls'][1] and c < cols - 1:
                walls.append((x + cell_size, y, x + cell_size, y + cell_size))
            # Bottom boundary
            if r == 0 and cells[r][c]['walls'][2]:
                walls.append((x, y, x + cell_size, y))
            # Left boundary
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


# Monster state (for hunt mode)
monster = {
    'x': 0.0,
    'y': 0.0,
    'speed': 0.25,  # Slower than player
    'active': False,
    'radius': 1.5,  # Hitbox radius for raycast detection
    'chord': [110, 131, 165],  # A diminished (low, ominous) - A2, C#3, E3
}

# Hunt mode state
hunt_state = {
    'goal_pos': (250.0, 250.0),  # Center cabin
    'won': False,
    'lost': False,
    'cabin_size': 8.0,  # Size of goal cabin
    'catch_distance': 3.0  # How close monster needs to be to catch you
}


def init_hunt_mode():
    """Initialize the hunt game mode - escape the monster, reach the cabin!"""
    global GAME_MODE, ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS, monster, hunt_state
    
    GAME_MODE = 'hunt'
    ROOM_WIDTH = 500.0
    ROOM_HEIGHT = 500.0
    
    # Generate forest with cabins
    INTERNAL_WALLS, cabin_positions = generate_forest(500)
    
    # Only one orb - the goal cabin at center
    goal_pos = (250.0, 250.0)  # Center of map
    ORBS = [{
        'pos': goal_pos,
        'chord': [262, 330, 392]  # C major - "home" chord
    }]
    
    # Player starts in corner
    player['x'] = 30.0
    player['y'] = 30.0
    player['angle'] = math.radians(45)  # Face toward center
    player['target_angle'] = player['angle']
    
    # Monster starts on opposite side
    monster['x'] = 450.0
    monster['y'] = 450.0
    monster['speed'] = 0.3  # Slightly slower than player (0.5)
    monster['active'] = True
    
    # Reset hunt state
    hunt_state['goal_pos'] = goal_pos
    hunt_state['won'] = False
    hunt_state['lost'] = False
    
    build_wall_grid(ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS)


def init_maze_chase_mode():
    """Initialize maze chase - escape the monster through a maze!
    
    Compact 10x10 maze, monster hunting you, beacon reachable.
    """
    global GAME_MODE, ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS, CELL_SIZE
    global monster, hunt_state, MAZE_COLS, MAZE_ROWS
    
    GAME_MODE = 'maze_chase'
    
    # Compact maze - navigable but challenging
    MAZE_COLS = 10
    MAZE_ROWS = 10
    CELL_SIZE = 12.0  # Slightly wider corridors
    ROOM_WIDTH = CELL_SIZE * MAZE_COLS
    ROOM_HEIGHT = CELL_SIZE * MAZE_ROWS
    
    # Generate maze with lots of loops (0.5 = many escape routes)
    INTERNAL_WALLS = generate_maze(MAZE_COLS, MAZE_ROWS, CELL_SIZE, loop_factor=0.5)
    
    # Clear starting area walls
    start_walls_to_remove = [
        (0, CELL_SIZE, CELL_SIZE, CELL_SIZE),
        (CELL_SIZE, 0, CELL_SIZE, CELL_SIZE),
    ]
    INTERNAL_WALLS = [w for w in INTERNAL_WALLS if w not in start_walls_to_remove]
    
    # Goal is NOT in far corner - put it mid-right area (reachable!)
    goal_x = ROOM_WIDTH - CELL_SIZE * 2
    goal_y = ROOM_HEIGHT / 2  # Middle height
    
    # Single beacon - the exit
    ORBS = [{
        'pos': (goal_x, goal_y),
        'chord': [262, 330, 392]  # C major - safety
    }]
    
    # Player starts in bottom-left
    player['x'] = CELL_SIZE * 1.5
    player['y'] = CELL_SIZE * 1.5
    player['angle'] = math.radians(45)
    player['target_angle'] = player['angle']
    
    # Monster starts BEHIND the player (chasing from the start!)
    monster['x'] = CELL_SIZE * 3
    monster['y'] = CELL_SIZE * 3
    monster['speed'] = 0.35  # Faster - the pressure is real
    monster['active'] = True
    
    # Hunt state for win/lose
    hunt_state['goal_pos'] = (goal_x, goal_y)
    hunt_state['cabin_size'] = CELL_SIZE * 1.5  # Slightly larger goal area
    hunt_state['won'] = False
    hunt_state['lost'] = False
    hunt_state['catch_distance'] = 2.5
    
    build_wall_grid(ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS)
    
    print(f"  Maze: {MAZE_COLS}x{MAZE_ROWS} cells ({ROOM_WIDTH}x{ROOM_HEIGHT} units)")
    print(f"  Find the beacon and escape!")


# ============ TRAINING MODE LEVELS ============

TRAINING_LEVEL = 1
TRAINING_COMPLETE = False

def init_training_level(level):
    """Initialize a specific training level."""
    global GAME_MODE, ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS
    global monster, hunt_state, TRAINING_LEVEL, TRAINING_COMPLETE
    
    TRAINING_LEVEL = level
    TRAINING_COMPLETE = False
    GAME_MODE = 'training'
    monster['active'] = False
    
    if level == 1:
        init_training_distance()
    elif level == 2:
        init_training_direction()
    elif level == 3:
        init_training_corridor()
    elif level == 4:
        init_training_tjunction()
    elif level == 5:
        init_training_minimaze()
    elif level == 6:
        init_training_monster()
    elif level == 7:
        init_training_escape()
    else:
        print(f"  Unknown training level: {level}")
        return
    
    build_wall_grid(ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS)


def init_training_distance():
    """Level 1: Learn click rate = distance. Walk toward wall, notice clicks speed up."""
    global ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS
    
    print("  LEVEL 1: DISTANCE")
    print("  Walk toward the wall ahead (W key)")
    print("  Notice: FASTER clicks = CLOSER to wall")
    print("  Touch the wall to complete")
    
    ROOM_WIDTH = 50.0
    ROOM_HEIGHT = 50.0
    
    # Just outer walls
    INTERNAL_WALLS = [
        (0, 0, ROOM_WIDTH, 0),
        (ROOM_WIDTH, 0, ROOM_WIDTH, ROOM_HEIGHT),
        (ROOM_WIDTH, ROOM_HEIGHT, 0, ROOM_HEIGHT),
        (0, ROOM_HEIGHT, 0, 0),
    ]
    
    ORBS = []  # No beacon yet
    
    player['x'] = 25.0
    player['y'] = 10.0
    player['angle'] = math.radians(0)  # Face north (toward far wall)
    player['target_angle'] = player['angle']
    
    hunt_state['goal_pos'] = (25.0, 48.0)  # Touch the far wall
    hunt_state['cabin_size'] = 5.0
    hunt_state['won'] = False
    hunt_state['lost'] = False


def init_training_direction():
    """Level 2: Find the beacon. Empty room, beacon somewhere, walk to it."""
    global ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS
    import random
    
    print("  LEVEL 2: DIRECTION")
    print("  Find the beacon (you'll hear a chord)")
    print("  Turn with Q/E, walk to the sound")
    
    ROOM_WIDTH = 80.0
    ROOM_HEIGHT = 80.0
    
    INTERNAL_WALLS = [
        (0, 0, ROOM_WIDTH, 0),
        (ROOM_WIDTH, 0, ROOM_WIDTH, ROOM_HEIGHT),
        (ROOM_WIDTH, ROOM_HEIGHT, 0, ROOM_HEIGHT),
        (0, ROOM_HEIGHT, 0, 0),
    ]
    
    # Beacon in a random corner
    corners = [(15, 15), (15, 65), (65, 15), (65, 65)]
    beacon_pos = random.choice(corners)
    
    ORBS = [{
        'pos': beacon_pos,
        'chord': [262, 330, 392]  # C major
    }]
    
    player['x'] = 40.0
    player['y'] = 40.0
    player['angle'] = math.radians(random.randint(0, 360))
    player['target_angle'] = player['angle']
    
    hunt_state['goal_pos'] = beacon_pos
    hunt_state['cabin_size'] = 8.0
    hunt_state['won'] = False
    hunt_state['lost'] = False


def init_training_corridor():
    """Level 3: Navigate a simple L-shaped corridor."""
    global ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS
    
    print("  LEVEL 3: CORRIDOR")
    print("  Navigate the L-shaped corridor to the beacon")
    print("  Use clicks to feel the walls on each side")
    
    ROOM_WIDTH = 100.0
    ROOM_HEIGHT = 100.0
    
    # L-shaped corridor
    INTERNAL_WALLS = [
        # Outer walls
        (0, 0, ROOM_WIDTH, 0),
        (ROOM_WIDTH, 0, ROOM_WIDTH, ROOM_HEIGHT),
        (ROOM_WIDTH, ROOM_HEIGHT, 0, ROOM_HEIGHT),
        (0, ROOM_HEIGHT, 0, 0),
        # Inner walls forming L
        (20, 0, 20, 60),       # Left wall of vertical corridor
        (40, 20, 40, 80),      # Right wall of vertical corridor
        (40, 80, 100, 80),     # Top of horizontal corridor
        (20, 60, 80, 60),      # Bottom of horizontal corridor
    ]
    
    ORBS = [{
        'pos': (90.0, 70.0),
        'chord': [262, 330, 392]
    }]
    
    player['x'] = 30.0
    player['y'] = 10.0
    player['angle'] = math.radians(0)
    player['target_angle'] = player['angle']
    
    hunt_state['goal_pos'] = (90.0, 70.0)
    hunt_state['cabin_size'] = 8.0
    hunt_state['won'] = False
    hunt_state['lost'] = False


def init_training_tjunction():
    """Level 4: T-junction - choose left or right based on beacon."""
    global ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS
    import random
    
    print("  LEVEL 4: T-JUNCTION")
    print("  Walk forward, then choose LEFT or RIGHT")
    print("  Listen for which way the beacon is")
    
    ROOM_WIDTH = 100.0
    ROOM_HEIGHT = 100.0
    
    # T-junction
    INTERNAL_WALLS = [
        # Outer
        (0, 0, ROOM_WIDTH, 0),
        (ROOM_WIDTH, 0, ROOM_WIDTH, ROOM_HEIGHT),
        (ROOM_WIDTH, ROOM_HEIGHT, 0, ROOM_HEIGHT),
        (0, ROOM_HEIGHT, 0, 0),
        # T-junction walls
        (40, 0, 40, 50),       # Left of stem
        (60, 0, 60, 50),       # Right of stem
        (0, 50, 40, 50),       # Block left-stem
        (60, 50, 100, 50),     # Block right-stem
        (0, 70, 40, 70),       # Top left dead end
        (60, 70, 100, 70),     # Top right dead end
    ]
    
    # Beacon randomly on left or right
    if random.random() < 0.5:
        beacon_pos = (20.0, 60.0)  # Left
    else:
        beacon_pos = (80.0, 60.0)  # Right
    
    ORBS = [{
        'pos': beacon_pos,
        'chord': [262, 330, 392]
    }]
    
    player['x'] = 50.0
    player['y'] = 10.0
    player['angle'] = math.radians(0)
    player['target_angle'] = player['angle']
    
    hunt_state['goal_pos'] = beacon_pos
    hunt_state['cabin_size'] = 8.0
    hunt_state['won'] = False
    hunt_state['lost'] = False


def init_training_minimaze():
    """Level 5: 5x5 mini maze, no monster."""
    global ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS, CELL_SIZE
    
    print("  LEVEL 5: MINI MAZE")
    print("  Navigate a small 5x5 maze")
    print("  Find the beacon in the opposite corner")
    
    CELL_SIZE = 15.0
    cols, rows = 5, 5
    ROOM_WIDTH = cols * CELL_SIZE
    ROOM_HEIGHT = rows * CELL_SIZE
    
    INTERNAL_WALLS = generate_maze(cols, rows, CELL_SIZE, loop_factor=0.3)
    
    ORBS = [{
        'pos': (ROOM_WIDTH - CELL_SIZE/2, ROOM_HEIGHT - CELL_SIZE/2),
        'chord': [262, 330, 392]
    }]
    
    player['x'] = CELL_SIZE / 2
    player['y'] = CELL_SIZE / 2
    player['angle'] = math.radians(45)
    player['target_angle'] = player['angle']
    
    hunt_state['goal_pos'] = (ROOM_WIDTH - CELL_SIZE/2, ROOM_HEIGHT - CELL_SIZE/2)
    hunt_state['cabin_size'] = CELL_SIZE
    hunt_state['won'] = False
    hunt_state['lost'] = False


def init_training_monster():
    """Level 6: Track the monster in an empty room (no walls to worry about)."""
    global ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS, monster
    
    print("  LEVEL 6: MONSTER TRACKING")
    print("  Learn to hear the monster!")
    print("  Dark chord = presence, footsteps = movement")
    print("  Metallic clicks = you can SEE it (line of sight)")
    print("  Touch the beacon to escape")
    
    ROOM_WIDTH = 100.0
    ROOM_HEIGHT = 100.0
    
    INTERNAL_WALLS = [
        (0, 0, ROOM_WIDTH, 0),
        (ROOM_WIDTH, 0, ROOM_WIDTH, ROOM_HEIGHT),
        (ROOM_WIDTH, ROOM_HEIGHT, 0, ROOM_HEIGHT),
        (0, ROOM_HEIGHT, 0, 0),
    ]
    
    # Beacon in corner
    ORBS = [{
        'pos': (90.0, 90.0),
        'chord': [262, 330, 392]
    }]
    
    player['x'] = 20.0
    player['y'] = 20.0
    player['angle'] = math.radians(45)
    player['target_angle'] = player['angle']
    
    # Slow monster
    monster['x'] = 80.0
    monster['y'] = 50.0
    monster['speed'] = 0.15  # Very slow for learning
    monster['active'] = True
    
    hunt_state['goal_pos'] = (90.0, 90.0)
    hunt_state['cabin_size'] = 8.0
    hunt_state['won'] = False
    hunt_state['lost'] = False
    hunt_state['catch_distance'] = 3.0


def init_training_escape():
    """Level 7: Mini maze with slow monster - the real deal."""
    global ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS, ORBS, CELL_SIZE, monster
    
    print("  LEVEL 7: ESCAPE!")
    print("  Navigate the maze AND avoid the monster")
    print("  Everything you've learned - use it!")
    
    CELL_SIZE = 15.0
    cols, rows = 6, 6
    ROOM_WIDTH = cols * CELL_SIZE
    ROOM_HEIGHT = rows * CELL_SIZE
    
    INTERNAL_WALLS = generate_maze(cols, rows, CELL_SIZE, loop_factor=0.4)
    
    ORBS = [{
        'pos': (ROOM_WIDTH - CELL_SIZE/2, ROOM_HEIGHT - CELL_SIZE/2),
        'chord': [262, 330, 392]
    }]
    
    player['x'] = CELL_SIZE / 2
    player['y'] = CELL_SIZE / 2
    player['angle'] = math.radians(45)
    player['target_angle'] = player['angle']
    
    # Monster starts near goal
    monster['x'] = ROOM_WIDTH - CELL_SIZE * 1.5
    monster['y'] = ROOM_HEIGHT - CELL_SIZE * 1.5
    monster['speed'] = 0.2  # Moderate speed
    monster['active'] = True
    
    hunt_state['goal_pos'] = (ROOM_WIDTH - CELL_SIZE/2, ROOM_HEIGHT - CELL_SIZE/2)
    hunt_state['cabin_size'] = CELL_SIZE
    hunt_state['won'] = False
    hunt_state['lost'] = False
    hunt_state['catch_distance'] = 2.5


def check_training_complete():
    """Check if current training level is complete."""
    global TRAINING_COMPLETE, hunt_state
    
    if hunt_state.get('won'):
        TRAINING_COMPLETE = True
        return True
    return False


def update_monster():
    """Move monster toward player (simple chase AI)"""
    if not monster['active']:
        return
    
    # Direction to player
    dx = player['x'] - monster['x']
    dy = player['y'] - monster['y']
    dist = math.sqrt(dx*dx + dy*dy)
    
    if dist > 0:
        # Normalize and move
        dx /= dist
        dy /= dist
        
        new_x = monster['x'] + dx * monster['speed']
        new_y = monster['y'] + dy * monster['speed']
        
        # Simple collision - just move if valid
        if can_move_to(new_x, new_y):
            monster['x'] = new_x
            monster['y'] = new_y
        elif can_move_to(new_x, monster['y']):
            monster['x'] = new_x
        elif can_move_to(monster['x'], new_y):
            monster['y'] = new_y


def check_hunt_win_lose():
    """Check win/lose conditions for hunt mode"""
    global hunt_state
    
    if hunt_state['won'] or hunt_state['lost']:
        return
    
    # Check if player is touching goal cabin wall
    goal_x, goal_y = hunt_state['goal_pos']
    cabin_size = hunt_state.get('cabin_size', 8.0)
    hs = cabin_size / 2  # Half size
    
    px, py = player['x'], player['y']
    touch_distance = 1.5  # How close counts as "touching"
    
    # Check each wall of the cabin
    # Top wall: from (goal_x - hs, goal_y + hs) to (goal_x + hs, goal_y + hs)
    # Bottom wall (with door gap): two segments
    # Left wall, Right wall
    
    touching_cabin = False
    
    # Check if player is near any cabin wall
    # Simplified: check if player is within touch_distance of cabin bounding box
    if (goal_x - hs - touch_distance <= px <= goal_x + hs + touch_distance and
        goal_y - hs - touch_distance <= py <= goal_y + hs + touch_distance):
        # Player is in the cabin's bounding area, check if actually touching a wall
        # (not inside the cabin)
        dist_to_left = abs(px - (goal_x - hs))
        dist_to_right = abs(px - (goal_x + hs))
        dist_to_top = abs(py - (goal_y + hs))
        dist_to_bottom = abs(py - (goal_y - hs))
        
        min_wall_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        if min_wall_dist < touch_distance:
            touching_cabin = True
    
    if touching_cabin:
        hunt_state['won'] = True
        hunt_state['victory_triggered'] = True  # Signal to audio engine
        monster['active'] = False
        return
    
    # Check if monster caught player
    dist_to_monster = math.sqrt((player['x'] - monster['x'])**2 + (player['y'] - monster['y'])**2)
    
    if dist_to_monster < hunt_state['catch_distance']:
        hunt_state['lost'] = True
        hunt_state['death_triggered'] = True  # Signal to audio engine
        monster['active'] = False


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
    """Simple non-blocking terminal input with sticky keys."""
    def __init__(self):
        self.old_settings = None
        self.key_times = {}
        self.sticky_duration = 0.25
        
    def start(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
    def stop(self):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def update(self):
        now = time.time()
        
        # Read all available input
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                # Escape sequence - check for arrow keys
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[' and select.select([sys.stdin], [], [], 0.01)[0]:
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
    
    def get_mouse_turn(self):
        """No mouse - return 0"""
        return 0


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
        
        # Monster sound (hunt mode) - growling/rumbling sound
        self.monster_click = self._generate_monster_click()
        self.monster_click_timers = {}  # Timer per sensor that sees monster
        self.monster_chord_phases = {}  # Phase for each chord frequency
        for freq in monster.get('chord', [110, 131, 165]):
            self.monster_chord_phases[freq] = 0.0
        
        # Monster footsteps and breathing (real samples)
        self._load_footstep_samples()
        self.monster_footstep_timer = 0.0
        self.monster_last_pos = (0.0, 0.0)
        self.monster_breath_phase = 0.0
        
        # Player footsteps
        self.player_footstep_timer = 0.0
        self.player_last_pos = (0.0, 0.0)
        
        # Victory melody
        self.victory_melody = self._generate_victory_melody()
        self.victory_playing = False
        self.victory_position = 0
        
        # Death sound
        self.death_sound = self._generate_death_sound()
        self.death_playing = False
        self.death_position = 0
        
        # Countdown beeps
        self.countdown_beeps = self._generate_countdown_beeps()
        self.countdown_playing = False
        self.countdown_position = 0
        
        # Headphone test sounds
        self.left_test = self._generate_headphone_test("left")
        self.right_test = self._generate_headphone_test("right")
        
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
        """Apply distance-based low-pass filter. Close = bright, far = muffled.
        
        Compressed range (1500-5000Hz) prevents jarring brightness jumps.
        Original 800-8000Hz was 10x range; now 3.3x for gentler transitions.
        """
        # Cutoff frequency: close (0m) = 5000Hz, far (15m+) = 1500Hz
        max_cutoff = 5000   # Was 8000 - less harsh up close
        min_cutoff = 1500   # Was 800 - less muffled far away
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
    
    def _generate_monster_click(self, duration=0.012):
        """Distinct monster click - metallic/harsh, different from soft wall clicks.
        
        Only plays when monster is in line-of-sight (like an object).
        """
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.exp(-t * 200)  # Sharp attack, fast decay
        # Metallic ring - multiple inharmonic frequencies
        tone = (np.sin(2 * np.pi * 800 * t) * 0.4 +    # High ping
                np.sin(2 * np.pi * 1200 * t) * 0.3 +   # Metallic
                np.sin(2 * np.pi * 2000 * t) * 0.2 +   # Bright
                np.sin(2 * np.pi * 400 * t) * 0.1)     # Body
        return (tone * envelope * 0.7).astype(np.float32)
    
    def _load_footstep_samples(self):
        """Load real footstep samples from sounds/ directory."""
        sounds_dir = os.path.join(os.path.dirname(__file__), 'sounds')
        
        self.monster_steps = []
        self.player_step = None
        
        # Load monster footsteps (heavy, pitched down)
        for fname in ['monster_step_heavy_1.wav', 'monster_step_heavy_2.wav']:
            path = os.path.join(sounds_dir, fname)
            if os.path.exists(path):
                sr, data = wavfile.read(path)
                # Convert to float32 normalized
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                # Resample if needed
                if sr != SAMPLE_RATE:
                    data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
                self.monster_steps.append(data.astype(np.float32))
        
        # Load player footstep
        path = os.path.join(sounds_dir, 'player_step.wav')
        if os.path.exists(path):
            sr, data = wavfile.read(path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            if sr != SAMPLE_RATE:
                data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
            self.player_step = data.astype(np.float32)
        
        # Fallback to synthesized if files not found
        if not self.monster_steps:
            self.monster_steps = [self._generate_synth_footstep(heavy=True)]
        if self.player_step is None:
            self.player_step = self._generate_synth_footstep(heavy=False)
        
        self.monster_step_idx = 0  # Alternate between steps
    
    def _generate_synth_footstep(self, heavy=False, duration=0.08):
        """Fallback synthesized footstep."""
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        if heavy:
            envelope = np.exp(-t * 40)
            tone = (np.sin(2 * np.pi * 50 * t) * 0.5 +
                    np.sin(2 * np.pi * 80 * t) * 0.3 +
                    np.random.uniform(-0.15, 0.15, samples))
            return (tone * envelope * 0.8).astype(np.float32)
        else:
            envelope = np.exp(-t * 60)
            tone = (np.sin(2 * np.pi * 100 * t) * 0.3 +
                    np.sin(2 * np.pi * 250 * t) * 0.2)
            return (tone * envelope * 0.4).astype(np.float32)
    
    def get_monster_footstep(self):
        """Get next monster footstep (alternates left/right foot)."""
        step = self.monster_steps[self.monster_step_idx % len(self.monster_steps)]
        self.monster_step_idx += 1
        return step
    
    def get_player_footstep(self):
        """Get player footstep sample."""
        return self.player_step
    
    def _generate_victory_melody(self):
        """Generate a triumphant victory melody - you made it home!"""
        # C major arpeggio ascending, then resolve to high C
        # Notes: C4, E4, G4, C5, E5, G5, C6 (hold)
        notes = [
            (262, 0.15),   # C4
            (330, 0.15),   # E4
            (392, 0.15),   # G4
            (523, 0.15),   # C5
            (659, 0.15),   # E5
            (784, 0.15),   # G5
            (1047, 0.5),   # C6 - hold
        ]
        
        melody = []
        for freq, duration in notes:
            samples = int(SAMPLE_RATE * duration)
            t = np.linspace(0, duration, samples, False)
            # Soft attack, gentle decay
            attack = np.minimum(t * 20, 1.0)
            decay = np.exp(-t * 2)
            envelope = attack * decay
            # Pure tone with slight warmth (add octave below quietly)
            tone = (np.sin(2 * np.pi * freq * t) * 0.7 +
                    np.sin(2 * np.pi * freq * 0.5 * t) * 0.2 +
                    np.sin(2 * np.pi * freq * 2 * t) * 0.1)
            melody.extend(tone * envelope * 0.5)
        
        # Add final chord (C major, warm and full)
        chord_duration = 1.5
        samples = int(SAMPLE_RATE * chord_duration)
        t = np.linspace(0, chord_duration, samples, False)
        envelope = np.exp(-t * 1.5)
        chord = (np.sin(2 * np.pi * 523 * t) * 0.4 +   # C5
                 np.sin(2 * np.pi * 659 * t) * 0.3 +   # E5
                 np.sin(2 * np.pi * 784 * t) * 0.3 +   # G5
                 np.sin(2 * np.pi * 262 * t) * 0.2)    # C4 bass
        melody.extend(chord * envelope * 0.6)
        
        return np.array(melody, dtype=np.float32)
    
    def trigger_victory(self):
        """Start playing the victory melody."""
        self.victory_playing = True
        self.victory_position = 0
    
    def _generate_death_sound(self):
        """Generate a terrifying death sound - the monster got you."""
        # Descending dissonant chord crash
        death = []
        
        # Initial harsh impact
        impact_duration = 0.3
        samples = int(SAMPLE_RATE * impact_duration)
        t = np.linspace(0, impact_duration, samples, False)
        envelope = np.exp(-t * 5)
        # Dissonant cluster - minor 2nds and tritones
        impact = (np.sin(2 * np.pi * 100 * t) * 0.4 +
                  np.sin(2 * np.pi * 106 * t) * 0.3 +  # Minor 2nd - dissonant
                  np.sin(2 * np.pi * 141 * t) * 0.3 +  # Tritone
                  np.sin(2 * np.pi * 50 * t) * 0.5 +   # Sub bass
                  np.random.uniform(-0.3, 0.3, samples))  # Noise
        death.extend(impact * envelope * 0.8)
        
        # Descending growl
        growl_duration = 1.5
        samples = int(SAMPLE_RATE * growl_duration)
        t = np.linspace(0, growl_duration, samples, False)
        envelope = np.exp(-t * 1.5)
        # Pitch drops over time
        freq_drop = 80 * np.exp(-t * 2) + 30
        growl = (np.sin(2 * np.pi * np.cumsum(freq_drop) / SAMPLE_RATE) * 0.5 +
                 np.sin(2 * np.pi * np.cumsum(freq_drop * 1.5) / SAMPLE_RATE) * 0.3 +
                 np.random.uniform(-0.2, 0.2, samples))
        death.extend(growl * envelope * 0.6)
        
        return np.array(death, dtype=np.float32)
    
    def trigger_death(self):
        """Start playing the death sound."""
        self.death_playing = True
        self.death_position = 0
    
    def _generate_countdown_beeps(self):
        """Generate 3-2-1-GO countdown beeps."""
        beeps = []
        
        # Three warning beeps (lower pitch)
        for i in range(3):
            beep_duration = 0.15
            samples = int(SAMPLE_RATE * beep_duration)
            t = np.linspace(0, beep_duration, samples, False)
            envelope = np.sin(np.pi * t / beep_duration)
            tone = np.sin(2 * np.pi * 440 * t) * envelope * 0.4
            beeps.extend(tone)
            # Gap between beeps
            gap = int(SAMPLE_RATE * 0.85)  # 1 second total per beep
            beeps.extend(np.zeros(gap))
        
        # Final GO beep (higher pitch, longer)
        go_duration = 0.3
        samples = int(SAMPLE_RATE * go_duration)
        t = np.linspace(0, go_duration, samples, False)
        envelope = np.sin(np.pi * t / go_duration)
        # Major chord for "GO"
        go_tone = (np.sin(2 * np.pi * 880 * t) * 0.3 +
                   np.sin(2 * np.pi * 1109 * t) * 0.2 +  # Major 3rd
                   np.sin(2 * np.pi * 1319 * t) * 0.2)   # 5th
        beeps.extend(go_tone * envelope * 0.5)
        
        return np.array(beeps, dtype=np.float32)
    
    def trigger_countdown(self):
        """Start playing countdown."""
        self.countdown_playing = True
        self.countdown_position = 0
    
    def is_countdown_done(self):
        """Check if countdown finished."""
        return not self.countdown_playing
    
    def _generate_headphone_test(self, side):
        """Generate headphone test sound for left or right."""
        duration = 0.8
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Three quick beeps
        beep_len = int(SAMPLE_RATE * 0.1)
        gap_len = int(SAMPLE_RATE * 0.15)
        
        audio = np.zeros(samples)
        pos = 0
        for _ in range(3):
            if pos + beep_len > samples:
                break
            beep_t = np.linspace(0, 0.1, beep_len, False)
            envelope = np.sin(np.pi * beep_t / 0.1)
            freq = 600 if side == "left" else 800  # Slightly different pitch
            audio[pos:pos+beep_len] = np.sin(2 * np.pi * freq * beep_t) * envelope * 0.5
            pos += beep_len + gap_len
        
        return audio.astype(np.float32)
    
    def play_headphone_test(self, side):
        """Play headphone test synchronously."""
        if side == "left":
            stereo = np.column_stack([self.left_test, np.zeros_like(self.left_test)])
        else:
            stereo = np.column_stack([np.zeros_like(self.right_test), self.right_test])
        sd.play(stereo, SAMPLE_RATE)
        sd.wait()
    
    def _generate_breath_sample(self, inhale=True, duration=0.3):
        """Generate a single breath sound (inhale or exhale)."""
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        if inhale:
            # Inhale: rising intensity, higher frequencies
            envelope = np.sin(np.pi * t / duration) ** 0.7
            freq_mod = 1.0 + 0.3 * t / duration  # Rising pitch
        else:
            # Exhale: falling intensity, lower frequencies
            envelope = np.sin(np.pi * t / duration) ** 1.5
            freq_mod = 1.0 - 0.2 * t / duration  # Falling pitch
        
        # Breathy noise with resonance
        noise = np.random.uniform(-1, 1, samples)
        # Simple bandpass via combination of tones and noise
        breath = (noise * 0.4 +
                  np.sin(2 * np.pi * 200 * freq_mod * t) * 0.2 +
                  np.sin(2 * np.pi * 400 * freq_mod * t) * 0.15)
        
        return (breath * envelope * 0.5).astype(np.float32)
    
    def ray_hits_monster(self, x, y, dx, dy):
        """Check if ray from (x,y) in direction (dx,dy) hits the monster.
        
        Returns distance to monster if hit, None otherwise.
        Uses circle intersection for monster hitbox.
        """
        if not monster['active']:
            return None
        
        # Vector from ray origin to monster center
        mx, my = monster['x'], monster['y']
        radius = monster.get('radius', 1.5)
        
        # Ray-circle intersection
        # Ray: P = (x,y) + t*(dx,dy)
        # Circle: (P - M)^2 = r^2
        ox = x - mx
        oy = y - my
        
        a = dx*dx + dy*dy
        b = 2 * (ox*dx + oy*dy)
        c = ox*ox + oy*oy - radius*radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None  # No intersection
        
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        
        # Return nearest positive intersection
        if t1 > 0.1:  # Minimum distance
            return t1
        elif t2 > 0.1:
            return t2
        return None
    
    def get_monster_direction(self):
        """Get direction and distance to monster relative to player's head"""
        if not monster['active']:
            return None, None
        
        # Vector from player to monster
        dx = monster['x'] - player['x']
        dy = monster['y'] - player['y']
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 0.1:
            return 0, 0.1
        
        # Angle to monster in world space
        world_angle = math.atan2(dx, dy)
        
        # Convert to head-relative angle
        head_relative = math.degrees(world_angle - player['angle'])
        
        # Normalize to -180 to 180
        while head_relative > 180:
            head_relative -= 360
        while head_relative < -180:
            head_relative += 360
        
        return head_relative, dist
    
    def get_monster_visibility(self):
        """Check which sensors can see the monster (line-of-sight).
        
        Returns dict mapping sensor_index -> distance to monster.
        Only includes sensors that have clear line-of-sight.
        """
        if not monster['active']:
            return {}
        
        x, y, angle = player['x'], player['y'], player['angle']
        nearby_walls = get_nearby_walls(x, y, radius=30.0)
        
        visible_sensors = {}
        
        for i, sensor_angle in enumerate(self.sensor_angles):
            # Ray direction in world space
            world_angle = angle + math.radians(sensor_angle)
            dx = math.sin(world_angle)
            dy = math.cos(world_angle)
            
            # Check if ray hits monster
            monster_dist = self.ray_hits_monster(x, y, dx, dy)
            
            if monster_dist is None:
                continue  # Ray doesn't hit monster
            
            # Check if any wall is CLOSER than the monster (blocking view)
            blocked = False
            for wall in nearby_walls:
                wall_dist = self.ray_hits_wall(wall[0], wall[1], wall[2], wall[3], x, y, dx, dy)
                if wall_dist is not None and wall_dist < monster_dist:
                    blocked = True
                    break
            
            if not blocked:
                visible_sensors[i] = monster_dist
        
        return visible_sensors
    
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
        """Get distance to wall for each sensor.
        
        Ray direction uses world coordinates, but HRTF uses head-relative angles.
        """
        x, y, angle = player['x'], player['y'], player['angle']
        nearby_walls = get_nearby_walls(x, y, radius=30.0)
        
        distances = []
        
        for sensor_angle in self.sensor_angles:
            # Ray direction in world space (for wall intersection)
            world_angle = angle + math.radians(sensor_angle)
            dx = math.sin(world_angle)
            dy = math.cos(world_angle)
            
            min_dist = 100.0
            for wall in nearby_walls:
                hit_dist = self.ray_hits_wall(wall[0], wall[1], wall[2], wall[3], x, y, dx, dy)
                if hit_dist is not None and hit_dist < min_dist:
                    min_dist = hit_dist
            
            distances.append(max(0.1, min_dist))
        
        return distances
    
    def get_reflection_distances(self):
        """Get distances to walls for room acoustics reflections.
        
        Returns (distance, head_relative_angle) - reflections are positioned relative to head.
        """
        x, y, angle = player['x'], player['y'], player['angle']
        nearby_walls = get_nearby_walls(x, y, radius=50.0)
        
        reflection_data = []  # (distance, head_relative_angle_deg)
        
        for ref_angle in self.reflection_angles:
            world_angle = angle + math.radians(ref_angle)
            dx = math.sin(world_angle)
            dy = math.cos(world_angle)
            
            min_dist = 100.0
            for wall in nearby_walls:
                hit_dist = self.ray_hits_wall(wall[0], wall[1], wall[2], wall[3], x, y, dx, dy)
                if hit_dist is not None and hit_dist < min_dist:
                    min_dist = hit_dist
            
            # Return HEAD-RELATIVE angle, not world angle!
            reflection_data.append((min_dist, ref_angle))
        
        return reflection_data
    
    def get_vertical_distances(self):
        """Get distances for vertical sensors (ceiling/floor awareness).
        
        Returns list of (distance, head_relative_az, elevation_deg) for each vertical sensor.
        In our flat world, vertical distances are simulated based on ceiling/floor height.
        """
        x, y, angle = player['x'], player['y'], player['angle']
        nearby_walls = get_nearby_walls(x, y, radius=30.0)
        
        vertical_data = []
        
        for az_offset, elevation in self.vertical_sensors:
            # Use HEAD-RELATIVE azimuth, not world azimuth
            head_az = az_offset  # This is already head-relative!
            
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
            
            vertical_data.append((max(0.5, min(vert_dist, 20.0)), head_az, elevation))
        
        return vertical_data
    
    def distance_to_click_rate(self, distance):
        """Convert distance to clicks per second.
        
        Uses quadratic falloff for earlier warning - rate stays high longer
        as you approach, giving more gradual feedback instead of sudden ramp.
        
        Linear:    d=10 → 7.3 clicks/sec, d=5 → 13.7, d=2 → 17.5
        Quadratic: d=10 → 11.6 clicks/sec, d=5 → 17.9, d=2 → 19.7
        """
        max_rate = 20.0
        min_rate = 1.0
        max_dist = 15.0
        clamped = max(0, min(distance, max_dist))
        # Quadratic falloff: (d/max)^2 gives earlier warning
        normalized = clamped / max_dist
        return max_rate - (normalized ** 2) * (max_rate - min_rate)
    
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
        
        # Get distances for all sensors (HRTF uses head-relative angles)
        distances = self.get_distances()
        
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
        for i, dist in enumerate(distances):
            rate = self.distance_to_click_rate(dist)
            self.click_timers[i] += dt
            
            if self.click_timers[i] >= 1.0 / rate:
                self.click_timers[i] = 0
                
                # Use HEAD-RELATIVE angle (sensor position), not world angle!
                # HRTF is always relative to your head orientation
                head_relative_angle = self.sensor_angles[i]  # e.g., 0° = front, 90° = right
                
                # Get HRTF-convolved click with distance-based filtering (bright=close, muffled=far)
                left_click, right_click = self.get_hrtf_click(head_relative_angle, distance=dist)
                
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
        for i, (vert_dist, head_az, elevation) in enumerate(vertical_data):
            # Slower click rate for vertical - ambient awareness, not navigation
            rate = self.distance_to_click_rate(vert_dist) * 0.5  # Half speed
            self.vertical_timers[i] += dt
            
            if self.vertical_timers[i] >= 1.0 / rate:
                self.vertical_timers[i] = 0
                
                # Use HEAD-RELATIVE azimuth for vertical sensors too
                # Get HRTF-convolved click with proper elevation and distance filtering
                # Vertical sounds are slightly quieter to not overwhelm horizontal
                left_click, right_click = self.get_hrtf_click(head_az, elevation_deg=elevation, volume=0.35, distance=vert_dist)
                
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
        
        # === MONSTER AUDIO (hunt mode) ===
        # Two components:
        # 1. Dark chord - always audible when in range, spatially positioned (like cabin orbs)
        # 2. Metallic clicks - ONLY when monster is in line-of-sight (like wall objects)
        if GAME_MODE in ['hunt', 'maze_chase', 'training'] and monster['active']:
            monster_angle, monster_dist = self.get_monster_direction()
            
            # --- 1. MONSTER CHORD (always audible, ominous presence) ---
            if monster_angle is not None and monster_dist < 150:
                # Volume: audible from far, louder when close
                chord_volume = max(0.05, min(0.2, (120 - monster_dist) / 120 * 0.2))
                
                # Stereo panning based on monster direction
                pan = math.sin(math.radians(monster_angle))
                left_vol = chord_volume * (1 - max(0, pan))
                right_vol = chord_volume * (1 + min(0, pan))
                
                # Generate dark chord (A diminished - ominous)
                t_chord = np.arange(frames) / SAMPLE_RATE
                for freq in monster.get('chord', [110, 131, 165]):
                    if freq not in self.monster_chord_phases:
                        self.monster_chord_phases[freq] = 0.0
                    
                    phase = self.monster_chord_phases[freq]
                    # Add slight detuning and tremolo for creepiness
                    tremolo = 1.0 + 0.1 * np.sin(2 * np.pi * 3 * t_chord)  # 3Hz wobble
                    tone = np.sin(2 * np.pi * freq * t_chord + phase) * tremolo
                    self.monster_chord_phases[freq] = phase + 2 * np.pi * freq * dt
                    
                    left_channel += (tone * left_vol / 3).astype(np.float32)
                    right_channel += (tone * right_vol / 3).astype(np.float32)
            
            # --- 2. MONSTER CLICKS (line-of-sight only, like an object) ---
            visible_sensors = self.get_monster_visibility()
            
            for sensor_idx, monster_dist in visible_sensors.items():
                # Initialize timer for this sensor if needed
                if sensor_idx not in self.monster_click_timers:
                    self.monster_click_timers[sensor_idx] = 0.0
                
                # Click rate based on distance (faster = closer)
                rate = self.distance_to_click_rate(monster_dist) * 1.5  # Slightly faster than walls
                self.monster_click_timers[sensor_idx] += dt
                
                if self.monster_click_timers[sensor_idx] >= 1.0 / rate:
                    self.monster_click_timers[sensor_idx] = 0
                    
                    # Get HRTF-positioned metallic click (distinct from wall clicks)
                    head_relative_angle = self.sensor_angles[sensor_idx]
                    left_ir, right_ir = interpolate_hrtf(-head_relative_angle, elevation_deg=0, n_neighbors=4)
                    
                    # Convolve monster click with HRTF
                    left_click = signal.fftconvolve(self.monster_click, left_ir, mode='full')
                    right_click = signal.fftconvolve(self.monster_click, right_ir, mode='full')
                    
                    # Volume based on distance
                    vol = max(0.4, min(0.9, 1.0 - monster_dist / 30))
                    
                    max_val = max(np.max(np.abs(left_click)), np.max(np.abs(right_click)), 0.001)
                    left_click = (left_click / max_val * vol).astype(np.float32)
                    right_click = (right_click / max_val * vol).astype(np.float32)
                    
                    click_len = min(len(left_click), frames)
                    left_channel[:click_len] += left_click[:click_len]
                    right_channel[:click_len] += right_click[:click_len]
            
            # Clean up timers for sensors that no longer see monster
            dead_timers = [k for k in self.monster_click_timers if k not in visible_sensors]
            for k in dead_timers:
                del self.monster_click_timers[k]
            
            # --- 3. MONSTER FOOTSTEPS (spatial, based on movement) ---
            if monster_angle is not None:
                # Check if monster moved
                monster_moved = math.sqrt(
                    (monster['x'] - self.monster_last_pos[0])**2 +
                    (monster['y'] - self.monster_last_pos[1])**2
                )
                self.monster_last_pos = (monster['x'], monster['y'])
                
                if monster_moved > 0.01:  # Monster is moving
                    # Footstep rate: ~2 steps/sec normally, faster when close (hunting)
                    step_rate = 2.0 + max(0, (50 - monster_dist) / 50) * 1.5
                    self.monster_footstep_timer += dt
                    
                    if self.monster_footstep_timer >= 1.0 / step_rate:
                        self.monster_footstep_timer = 0
                        
                        # HRTF-positioned footstep (real sample, alternating feet)
                        left_ir, right_ir = interpolate_hrtf(-monster_angle, elevation_deg=-30, n_neighbors=4)
                        footstep_sample = self.get_monster_footstep()
                        
                        left_step = signal.fftconvolve(footstep_sample, left_ir, mode='full')
                        right_step = signal.fftconvolve(footstep_sample, right_ir, mode='full')
                        
                        # Volume based on distance (footsteps carry far)
                        vol = max(0.2, min(0.8, 1.0 - monster_dist / 80))
                        
                        max_val = max(np.max(np.abs(left_step)), np.max(np.abs(right_step)), 0.001)
                        left_step = (left_step / max_val * vol).astype(np.float32)
                        right_step = (right_step / max_val * vol).astype(np.float32)
                        
                        step_len = min(len(left_step), frames)
                        left_channel[:step_len] += left_step[:step_len]
                        right_channel[:step_len] += right_step[:step_len]
            
            # --- 4. MONSTER BREATHING (spatial, rate increases when close) ---
            if monster_angle is not None and monster_dist < 80:
                # Breath rate: slow when far (0.3 Hz), fast when close (1.5 Hz)
                breath_rate = 0.3 + (1.0 - monster_dist / 80) * 1.2
                
                # Continuous breathing using phase
                old_phase = self.monster_breath_phase
                self.monster_breath_phase += breath_rate * dt
                
                # Generate breath on phase boundaries (0 = inhale start, 0.5 = exhale start)
                if int(old_phase * 2) != int(self.monster_breath_phase * 2):
                    is_inhale = int(self.monster_breath_phase * 2) % 2 == 0
                    breath_duration = 0.4 / breath_rate  # Adjust duration to rate
                    breath_sound = self._generate_breath_sample(inhale=is_inhale, duration=min(0.5, breath_duration))
                    
                    # HRTF-positioned breathing
                    left_ir, right_ir = interpolate_hrtf(-monster_angle, elevation_deg=10, n_neighbors=4)
                    
                    left_breath = signal.fftconvolve(breath_sound, left_ir, mode='full')
                    right_breath = signal.fftconvolve(breath_sound, right_ir, mode='full')
                    
                    # Volume based on distance
                    vol = max(0.1, min(0.5, 1.0 - monster_dist / 60))
                    
                    max_val = max(np.max(np.abs(left_breath)), np.max(np.abs(right_breath)), 0.001)
                    left_breath = (left_breath / max_val * vol).astype(np.float32)
                    right_breath = (right_breath / max_val * vol).astype(np.float32)
                    
                    breath_len = min(len(left_breath), frames)
                    left_channel[:breath_len] += left_breath[:breath_len]
                    right_channel[:breath_len] += right_breath[:breath_len]
        
        # --- PLAYER FOOTSTEPS (centered, feedback for your own movement) ---
        player_moved = math.sqrt(
            (player['x'] - self.player_last_pos[0])**2 +
            (player['y'] - self.player_last_pos[1])**2
        )
        self.player_last_pos = (player['x'], player['y'])
        
        if player_moved > 0.01:  # Player is moving
            # Footstep rate: ~4 steps/sec when walking
            step_rate = 4.0
            self.player_footstep_timer += dt
            
            if self.player_footstep_timer >= 1.0 / step_rate:
                self.player_footstep_timer = 0
                
                # Player footsteps are centered (your own feet, real sample)
                footstep = self.get_player_footstep()
                step_len = min(len(footstep), frames)
                left_channel[:step_len] += footstep[:step_len] * 0.5  # Quieter for own steps
                right_channel[:step_len] += footstep[:step_len] * 0.5
        
        # Orb/cabin chords
        px, py = player['x'], player['y']
        t = np.arange(frames) / SAMPLE_RATE
        
        # In hunt mode, the goal cabin beacon needs to be audible from far away
        # Orb hearing distance varies by mode
        if GAME_MODE == 'hunt':
            max_orb_dist = 400.0  # Large forest
        elif GAME_MODE == 'maze_chase':
            max_orb_dist = 200.0  # Compact maze (120x120), always audible
        else:
            max_orb_dist = 150.0
        
        for orb_idx, orb in enumerate(ORBS):
            orb_x, orb_y = orb['pos']
            orb_dist = math.sqrt((orb_x - px)**2 + (orb_y - py)**2)
            
            if orb_dist > max_orb_dist:
                continue
            
            # Volume scales with distance - audible from far, louder when close
            if GAME_MODE == 'maze_chase':
                # Maze chase: smaller space, beacon always prominent
                orb_volume = max(0.05, min(0.25, (150 - orb_dist) / 150 * 0.25))
            elif GAME_MODE in ['hunt', 'training']:
                # Hunt/training: beacon audible from far, full volume at 50m
                orb_volume = max(0.02, min(0.2, (300 - orb_dist) / 300 * 0.2))
            else:
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
        
        # === VICTORY MELODY ===
        if GAME_MODE in ['hunt', 'maze_chase', 'training'] and hunt_state.get('victory_triggered') and not self.victory_playing:
            self.trigger_victory()
            hunt_state['victory_triggered'] = False  # Only trigger once
        
        if self.victory_playing:
            remaining = len(self.victory_melody) - self.victory_position
            to_play = min(frames, remaining)
            
            if to_play > 0:
                melody_chunk = self.victory_melody[self.victory_position:self.victory_position + to_play]
                left_channel[:to_play] += melody_chunk
                right_channel[:to_play] += melody_chunk
                self.victory_position += to_play
            else:
                self.victory_playing = False
        
        # === DEATH SOUND ===
        if GAME_MODE in ['hunt', 'maze_chase', 'training'] and hunt_state.get('death_triggered') and not self.death_playing:
            self.trigger_death()
            hunt_state['death_triggered'] = False
        
        if self.death_playing:
            remaining = len(self.death_sound) - self.death_position
            to_play = min(frames, remaining)
            
            if to_play > 0:
                death_chunk = self.death_sound[self.death_position:self.death_position + to_play]
                left_channel[:to_play] += death_chunk
                right_channel[:to_play] += death_chunk
                self.death_position += to_play
            else:
                self.death_playing = False
        
        # === COUNTDOWN ===
        if self.countdown_playing:
            remaining = len(self.countdown_beeps) - self.countdown_position
            to_play = min(frames, remaining)
            
            if to_play > 0:
                countdown_chunk = self.countdown_beeps[self.countdown_position:self.countdown_position + to_play]
                left_channel[:to_play] += countdown_chunk
                right_channel[:to_play] += countdown_chunk
                self.countdown_position += to_play
            else:
                self.countdown_playing = False
        
        # Output
        outdata[:, 0] = np.clip(left_channel, -1, 1)
        outdata[:, 1] = np.clip(right_channel, -1, 1)
    
    def start(self):
        # Create new stream (needed after stop/restart)
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            callback=self.audio_callback,
            blocksize=BUFFER_SIZE,
            dtype=np.float32
        )
        self.running = True
        self.stream.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


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
    
    # Face tracking (if enabled)
    if FACE_TRACKER is not None and FACE_TRACKER.is_face_detected():
        # Face tracking directly controls angle
        # Head yaw maps to player angle (scaled for comfortable range)
        head_yaw = FACE_TRACKER.get_head_yaw()
        # Scale: 45° head turn = 90° in-game turn (2x multiplier)
        face_angle = head_yaw * 2.0
        player['target_angle'] = face_angle
        player['angle'] = lerp_angle(player['angle'], player['target_angle'], 0.15)
    else:
        # Keyboard turning (Q/E)
        if term.is_pressed('q'):
            player['target_angle'] -= player['turn_speed']
        if term.is_pressed('e'):
            player['target_angle'] += player['turn_speed']
        
        # Mouse turning (move mouse left/right)
        mouse_turn = term.get_mouse_turn()
        if mouse_turn != 0:
            player['target_angle'] += mouse_turn
        
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
    
    # Draw monster (hunt mode)
    if GAME_MODE in ['hunt', 'maze_chase', 'training'] and monster['active']:
        mx, my = monster['x'], monster['y']
        if world_min_x <= mx <= world_max_x and world_min_y <= my <= world_max_y:
            mgx, mgy = world_to_grid(mx, my)
            m_row = map_size - 1 - mgy
            if 0 <= m_row < map_size and 0 <= mgx < map_size:
                grid[m_row][mgx] = '☠'  # Monster skull
    
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
    
    if GAME_MODE == 'maze':
        mode_label = "MAZE"
        orb_label = "ORB"
    elif GAME_MODE == 'forest':
        mode_label = "FOREST"
        orb_label = "CABIN"
    else:
        mode_label = "HUNT"
        orb_label = "CABIN"
    
    lines = []
    
    lines.append("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    
    if GAME_MODE in ['hunt', 'maze_chase', 'training']:
        monster_dist = math.sqrt((monster['x'] - player['x'])**2 + (monster['y'] - player['y'])**2)
        # Calculate head-relative angle to monster
        m_dx = monster['x'] - player['x']
        m_dy = monster['y'] - player['y']
        monster_world = math.atan2(m_dx, m_dy)
        monster_head_rel = math.degrees(monster_world - player['angle'])
        while monster_head_rel > 180: monster_head_rel -= 360
        while monster_head_rel < -180: monster_head_rel += 360
        
        lines.append(f"║                              DREAM SPACE - ESCAPE THE MONSTER! [{mode_label}]                                                 ║")
        lines.append(f"║  Pos: ({player['x']:6.1f},{player['y']:6.1f})  Face: {deg:5.1f}° ({compass:2})    CABIN: {orb_dist:4.0f}m {orb_angle:+4.0f}°    ☠ MONSTER: {monster_dist:4.0f}m {monster_head_rel:+4.0f}°               ║")
    else:
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
    print("║   Game Modes:                                         ║")
    print("║   [1] MAZE       - 800x800 maze, find the beacons     ║")
    print("║   [2] FOREST     - Open world with cabins & trees     ║")
    print("║   [3] HUNT       - Forest escape, monster chase!      ║")
    print("║   [4] MAZE CHASE - Monster in a maze! (HARD)          ║")
    print("║                                                       ║")
    print("║   Training:                                           ║")
    print("║   [5] TRAINING   - Learn to navigate step by step     ║")
    print("║                    7 progressive levels               ║")
    print("║                                                       ║")
    print("║   [Q] Quit                                            ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    print("Press 1-5 or Q: ", end='', flush=True)


def show_training_menu():
    print('\033[H\033[J', end='')
    print("╔═══════════════════════════════════════════════════════╗")
    print("║              TRAINING MODE                            ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║                                                       ║")
    print("║   [1] Distance   - Click rate = distance              ║")
    print("║   [2] Direction  - Find the beacon                    ║")
    print("║   [3] Corridor   - Navigate L-shaped path             ║")
    print("║   [4] T-Junction - Choose left or right               ║")
    print("║   [5] Mini Maze  - 5x5 maze, no monster               ║")
    print("║   [6] Monster    - Track the monster (open room)      ║")
    print("║   [7] Escape!    - Maze + monster (the real deal)     ║")
    print("║                                                       ║")
    print("║   [A] Auto       - Start from level 1, progress up    ║")
    print("║   [B] Back       - Return to main menu                ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    print("Press 1-7, A, or B: ", end='', flush=True)


def select_training_level():
    """Show training menu and get level selection."""
    show_training_menu()
    
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            ch = sys.stdin.read(1).lower()
            if ch in '1234567':
                return int(ch)
            elif ch == 'a':
                return 'auto'
            elif ch == 'b':
                return 'back'
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def run_training_session(start_level, engine):
    """Run training session starting from given level."""
    global GAME_MODE, TRAINING_LEVEL, TRAINING_COMPLETE, hunt_state, monster
    
    if start_level == 'auto':
        current_level = 1
    else:
        current_level = start_level
    
    while current_level <= 7:
        print('\033[H\033[J', end='')
        print(f"\n  Initializing Training Level {current_level}...\n")
        
        init_training_level(current_level)
        
        # Reset engine state
        engine.victory_playing = False
        engine.victory_position = 0
        engine.death_playing = False
        engine.death_position = 0
        
        time.sleep(1)
        
        # Countdown
        print("\n  Starting in...")
        engine.trigger_countdown()
        engine.start()
        time.sleep(3.5)
        
        term = TerminalInput()
        term.start()
        
        result = None
        
        try:
            while True:
                should_quit = handle_input(term, engine)
                if should_quit:
                    result = 'quit'
                    break
                
                # Update monster if active (levels 6 and 7)
                if monster['active']:
                    update_monster()
                    # Check if caught
                    dist_to_monster = math.sqrt(
                        (player['x'] - monster['x'])**2 + 
                        (player['y'] - monster['y'])**2
                    )
                    if dist_to_monster < hunt_state.get('catch_distance', 3.0):
                        hunt_state['lost'] = True
                        hunt_state['death_triggered'] = True
                        display_status(engine)
                        time.sleep(2)
                        result = 'lost'
                        break
                
                # Check win condition
                goal_x, goal_y = hunt_state['goal_pos']
                cabin_size = hunt_state.get('cabin_size', 8.0)
                hs = cabin_size / 2
                px, py = player['x'], player['y']
                
                # Check if touching goal area
                if (goal_x - hs - 1.5 <= px <= goal_x + hs + 1.5 and
                    goal_y - hs - 1.5 <= py <= goal_y + hs + 1.5):
                    hunt_state['won'] = True
                    hunt_state['victory_triggered'] = True
                    display_status(engine)
                    time.sleep(3)
                    result = 'won'
                    break
                
                display_status(engine)
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            result = 'quit'
        finally:
            term.stop()
            engine.stop()
        
        # Show result
        print('\033[H\033[J', end='')
        if result == 'won':
            print(f"\n  ✓ LEVEL {current_level} COMPLETE!\n")
            if current_level < 7:
                print(f"  [ENTER] Next level ({current_level + 1})")
                print("  [R] Retry this level")
                print("  [M] Training menu")
                print("  [Q] Quit")
                
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setcbreak(sys.stdin.fileno())
                    ch = sys.stdin.read(1).lower()
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
                if ch == 'q':
                    return 'quit'
                elif ch == 'r':
                    continue
                elif ch == 'm':
                    return 'menu'
                else:
                    current_level += 1
            else:
                print("  🎉 ALL TRAINING COMPLETE! 🎉")
                print("  You're ready for the real challenges!")
                print("\n  [ENTER] Return to menu")
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setcbreak(sys.stdin.fileno())
                    sys.stdin.read(1)
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                return 'menu'
        
        elif result == 'lost':
            print(f"\n  ✗ CAUGHT! Try again.\n")
            print("  [ENTER] Retry")
            print("  [M] Training menu")
            print("  [Q] Quit")
            
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                ch = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            if ch == 'q':
                return 'quit'
            elif ch == 'm':
                return 'menu'
            # else retry same level
        
        elif result == 'quit':
            return 'quit'
    
    return 'menu'


def headphone_check(engine):
    """Quick headphone L/R test."""
    print('\033[H\033[J', end='')
    print("╔═══════════════════════════════════════════════════════╗")
    print("║              🎧 HEADPHONE CHECK 🎧                    ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║                                                       ║")
    print("║   Put on your headphones now.                         ║")
    print("║   This game REQUIRES headphones for 3D audio.         ║")
    print("║                                                       ║")
    print("║   Press SPACE to test LEFT ear...                     ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        sys.stdin.read(1)
        
        print("\n   🔊 LEFT ear...")
        engine.play_headphone_test("left")
        time.sleep(0.5)
        
        print("   Press SPACE to test RIGHT ear...")
        sys.stdin.read(1)
        
        print("\n   🔊 RIGHT ear...")
        engine.play_headphone_test("right")
        time.sleep(0.5)
        
        print("\n   ✓ Headphones confirmed! Press any key to continue...")
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def run_tutorial(engine):
    """Quick 30-second tutorial - learn the click system."""
    global player, GAME_MODE, INTERNAL_WALLS, ROOM_WIDTH, ROOM_HEIGHT
    
    print('\033[H\033[J', end='')
    print("╔═══════════════════════════════════════════════════════╗")
    print("║              📖 QUICK TUTORIAL 📖                     ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║                                                       ║")
    print("║   Learn how to 'see' with sound in 30 seconds.        ║")
    print("║                                                       ║")
    print("║   [ENTER] Start tutorial                              ║")
    print("║   [S]     Skip tutorial (I know how to play)          ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        ch = sys.stdin.read(1).lower()
        if ch == 's':
            return  # Skip tutorial
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    # Simple tutorial room - just one wall in front
    GAME_MODE = 'tutorial'
    ROOM_WIDTH = 50.0
    ROOM_HEIGHT = 50.0
    INTERNAL_WALLS = []
    
    # Outer walls
    INTERNAL_WALLS = [
        (0, 0, ROOM_WIDTH, 0),  # Bottom
        (ROOM_WIDTH, 0, ROOM_WIDTH, ROOM_HEIGHT),  # Right
        (ROOM_WIDTH, ROOM_HEIGHT, 0, ROOM_HEIGHT),  # Top
        (0, ROOM_HEIGHT, 0, 0),  # Left
    ]
    
    build_wall_grid(ROOM_WIDTH, ROOM_HEIGHT, INTERNAL_WALLS)
    
    player['x'] = 25.0
    player['y'] = 25.0
    player['angle'] = math.radians(90)  # Face right (toward wall)
    player['target_angle'] = player['angle']
    
    engine.start()
    
    term = TerminalInput()
    term.start()
    
    print('\033[H\033[J', end='')
    print("\n  TUTORIAL: Listen to the clicks. Walk toward the wall (W key).")
    print("  Notice: FASTER clicks = CLOSER to the wall.")
    print("  Try walking forward and backward to feel the difference.\n")
    print("  Press SPACE when you understand, or wait 30 seconds.\n")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < 30:
            should_quit = handle_input(term, engine)
            if should_quit:
                break
            
            # Check for space to exit tutorial
            if term.is_pressed(' '):
                break
            
            display_status(engine)
            time.sleep(0.05)
    finally:
        term.stop()
        engine.stop()
    
    print('\033[H\033[J', end='')
    print("\n  ✓ Tutorial complete!")
    print("  Remember: Faster clicks = closer walls.")
    print("  Use Q/E to turn and listen for walls in different directions.\n")
    time.sleep(2)


def run_game_round(selected_mode, engine):
    """Run one round of the game. Returns 'restart', 'menu', or 'quit'."""
    global GAME_MODE, hunt_state
    
    print('\033[H\033[J', end='')
    if selected_mode == 'maze':
        print("Generating maze...")
        init_maze_mode()
    elif selected_mode == 'forest':
        print("Generating forest...")
        init_forest_mode()
    elif selected_mode == 'maze_chase':
        print("Generating maze chase arena...")
        init_maze_chase_mode()
    else:
        print("Generating hunt arena...")
        init_hunt_mode()
    
    # Reset audio engine state
    engine.victory_playing = False
    engine.victory_position = 0
    engine.death_playing = False
    engine.death_position = 0
    
    print('\033[H\033[J', end='')
    print("\n  Get ready...")
    print("  Find the cabin beacon (pleasant chord).")
    print("  Avoid the monster (dark chord + footsteps).\n")
    time.sleep(1)
    
    # Countdown
    print("  Starting in...")
    engine.trigger_countdown()
    engine.start()
    
    # Wait for countdown (approximately 3.5 seconds)
    time.sleep(3.5)
    
    term = TerminalInput()
    term.start()
    
    game_result = None
    
    try:
        while True:
            should_quit = handle_input(term, engine)
            if should_quit:
                game_result = 'quit'
                break
            
            # Hunt mode updates
            if GAME_MODE in ['hunt', 'maze_chase', 'training']:
                update_monster()
                check_hunt_win_lose()
                
                if hunt_state['won']:
                    display_status(engine)
                    # Let victory melody play
                    time.sleep(3.5)
                    game_result = 'won'
                    break
                    
                if hunt_state['lost']:
                    display_status(engine)
                    # Let death sound play
                    time.sleep(2.5)
                    game_result = 'lost'
                    break
            
            display_status(engine)
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        game_result = 'quit'
    finally:
        term.stop()
        engine.stop()
    
    # Show result screen
    print('\033[H\033[J', end='')
    if game_result == 'won':
        print("\n")
        print("  ╔═══════════════════════════════════════╗")
        print("  ║     🏆 YOU WIN! 🏆                    ║")
        print("  ║     You reached the cabin safely!    ║")
        print("  ╚═══════════════════════════════════════╝")
    elif game_result == 'lost':
        print("\n")
        print("  ╔═══════════════════════════════════════╗")
        print("  ║     💀 GAME OVER 💀                   ║")
        print("  ║     The monster caught you!          ║")
        print("  ╚═══════════════════════════════════════╝")
    
    if game_result in ['won', 'lost']:
        print("\n  [R] Restart")
        print("  [M] Main Menu")
        print("  [Q] Quit\n")
        
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                ch = sys.stdin.read(1).lower()
                if ch == 'r':
                    return 'restart'
                elif ch == 'm':
                    return 'menu'
                elif ch == 'q':
                    return 'quit'
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    return game_result or 'quit'


def main():
    parser = argparse.ArgumentParser(description='Dream Space - HRTF Spatial Audio Trainer')
    parser.add_argument('--device', '-d', type=int, help='Audio output device index')
    parser.add_argument('--list-devices', '-l', action='store_true', help='List audio devices')
    parser.add_argument('--mode', '-m', choices=['maze', 'forest', 'hunt', 'maze_chase', 'training'], help='Game mode (skip menu)')
    parser.add_argument('--hrtf', default='hrtf/mit_kemar.sofa', help='Path to HRTF SOFA file')
    parser.add_argument('--no-mouse', action='store_true', help='Disable mouse look (use Q/E to turn)')
    parser.add_argument('--face-tracking', '-f', action='store_true', help='Use webcam face tracking for head rotation')
    parser.add_argument('--skip-tutorial', action='store_true', help='Skip tutorial')
    parser.add_argument('--skip-headphone-check', action='store_true', help='Skip headphone check')
    args = parser.parse_args()
    
    # Store globally for TerminalInput
    global MOUSE_ENABLED, FACE_TRACKER
    MOUSE_ENABLED = not args.no_mouse
    FACE_TRACKER = None
    
    # Face tracking setup
    if args.face_tracking:
        try:
            from face_tracking import FaceTracker
            print("Starting face tracking...")
            FACE_TRACKER = FaceTracker(smoothing=0.4)
            FACE_TRACKER.start()
            print("Look straight ahead at the camera to calibrate...")
            time.sleep(2)  # Give time to calibrate
        except ImportError:
            print("Face tracking requires: pip install mediapipe opencv-python")
            print("Falling back to keyboard controls.")
            FACE_TRACKER = None
        except Exception as e:
            print(f"Face tracking error: {e}")
            print("Falling back to keyboard controls.")
            FACE_TRACKER = None
    
    if args.list_devices:
        print(sd.query_devices())
        return
    
    if args.device is not None:
        sd.default.device[1] = args.device
        print(f"Using: {sd.query_devices(args.device)['name']}")
        time.sleep(1)
    
    # Load HRTF
    load_hrtf(args.hrtf)
    
    # Create audio engine (used for tests and gameplay)
    print("Initializing HRTF audio engine...")
    engine = HRTFAudioEngine()
    
    # Headphone check
    if not args.skip_headphone_check:
        headphone_check(engine)
    
    # Tutorial (for hunt mode)
    first_run = True
    
    while True:
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
                    elif ch == '3':
                        selected_mode = 'hunt'
                        break
                    elif ch == '4':
                        selected_mode = 'maze_chase'
                        break
                    elif ch == '5':
                        selected_mode = 'training'
                        break
                    elif ch == 'q':
                        print("\nGoodbye!")
                        return
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        # Handle training mode selection
        if selected_mode == 'training':
            training_level = select_training_level()
            if training_level == 'back':
                args.mode = None
                continue
            result = run_training_session(training_level, engine)
            if result == 'quit':
                break
            args.mode = None
            continue
        
        # Tutorial for monster modes on first run
        if first_run and selected_mode in ['hunt', 'maze_chase'] and not args.skip_tutorial:
            run_tutorial(engine)
            first_run = False
        
        # Run the game
        result = run_game_round(selected_mode, engine)
        
        if result == 'quit':
            break
        elif result == 'menu':
            args.mode = None  # Reset to show menu
            continue
        elif result == 'restart':
            continue  # Same mode
    
    print('\033[H\033[J', end='')
    # Cleanup face tracking
    if FACE_TRACKER is not None:
        FACE_TRACKER.stop()
    
    print("Thanks for playing Dream Space!")


if __name__ == "__main__":
    main()
