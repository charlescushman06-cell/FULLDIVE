# Fulldive

**Your brain can learn to see with sound. Fulldive teaches it how.**

Walls click faster as you approach. Monsters breathe behind you. Safe havens hum in the distance. Close your eyes, put on headphones, and navigate a world rendered entirely by your mind. No screen. Just perception.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **HRTF Spatial Audio**: Real 3D sound positioning using MIT KEMAR dataset
- **Multiple Game Modes**: Maze, Forest, Hunt, Maze Chase
- **Progressive Training**: 7 levels from basic distance perception to monster escape
- **Face Tracking**: Optional webcam head tracking for immersive control
- **Audiograph**: Convert images to spatial audio (sensory substitution)

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/fulldive.git
cd fulldive

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Game

```bash
python spatial_audio.py
```

**First run will:**
1. Check your headphones (L/R test)
2. Show game menu
3. Optionally run a tutorial

### 3. Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move forward/left/back/right |
| Q/E | Turn left/right |
| ESC | Quit |
| Space | Skip tutorial section |

## Requirements

```
numpy
scipy
sounddevice
sofar
```

**Optional (for face tracking):**
```
mediapipe
opencv-python
```

**Optional (for audiograph):**
```
pillow
```

Install all:
```bash
pip install numpy scipy sounddevice sofar mediapipe opencv-python pillow
```

## Game Modes

### 1. Maze
Large 800x800 maze with multiple beacons. Practice navigation.

### 2. Forest  
Open world with cabins and trees. Find your way home.

### 3. Hunt
Forest escape - monster chases you! Reach the cabin to win.

### 4. Maze Chase
Monster in a compact maze. The ultimate challenge.

### 5. Training
Progressive 7-level training:
1. **Distance** - Learn click rate = distance
2. **Direction** - Find beacons by sound
3. **Corridor** - Navigate L-shaped paths
4. **T-Junction** - Audio-based decisions
5. **Mini Maze** - 5x5 maze navigation
6. **Monster** - Learn monster audio cues
7. **Escape** - Combine everything

## Audio Encoding

- **Click Rate**: Faster clicks = closer to wall
- **HRTF Positioning**: Sounds positioned in 3D around your head
- **Beacon Chords**: Pleasant chord guides you to goal
- **Monster Audio**:
  - Dark chord = monster presence (always audible)
  - Footsteps = monster moving
  - Breathing = monster getting close
  - Metallic clicks = monster in line-of-sight

## Face Tracking (Optional)

Use your webcam to control head rotation:

```bash
# Test face tracking standalone
python face_tracking.py

# Run game with face tracking
python spatial_audio.py --face-tracking
# or
python spatial_audio.py -f
```

**Requirements:**
- Webcam
- Good lighting
- Look straight ahead during calibration

**How it works:**
- Turn your head left/right to rotate in-game
- 45° head turn = 90° in-game rotation
- WASD still controls movement

## Audiograph - Image to Sound

Convert images to spatial audio for sensory substitution training:

```bash
# Convert an image to spatial audio
python audiograph.py image.png

# Scanning mode (left-to-right sweep)
python audiograph.py image.png --scan

# Custom duration
python audiograph.py image.png --duration 4.0

# Save to file
python audiograph.py image.png --save output.wav
```

**Encoding:**
- X-axis → HRTF spatial position (left/right)
- Y-axis → Frequency (top = high pitch, bottom = low)
- Brightness → Volume

### Audiograph Trainer

Progressive training to learn image-to-sound perception:

```bash
python audiograph_trainer.py
```

Levels: Spatial → Vertical → Shapes → Letters → Numbers → Objects

## Command Line Options

```
python spatial_audio.py [OPTIONS]

Options:
  -m, --mode MODE         Start directly in mode (maze/forest/hunt/maze_chase/training)
  -f, --face-tracking     Enable webcam face tracking
  -d, --device INDEX      Audio output device index
  -l, --list-devices      List available audio devices
  --hrtf PATH             Path to HRTF SOFA file (default: hrtf/mit_kemar.sofa)
  --no-mouse              Disable mouse look
  --skip-tutorial         Skip the tutorial
  --skip-headphone-check  Skip headphone L/R test
```

**Examples:**
```bash
# Quick start maze chase
python spatial_audio.py -m maze_chase --skip-headphone-check --skip-tutorial

# Start training directly
python spatial_audio.py -m training

# Face tracking with hunt mode
python spatial_audio.py -m hunt -f
```

## Project Structure

```
fulldive/
├── spatial_audio.py      # Main game
├── face_tracking.py      # Webcam head tracking
├── audiograph.py         # Image to audio converter
├── audiograph_trainer.py # Image perception training
├── hrtf/
│   └── mit_kemar.sofa    # HRTF data (MIT KEMAR)
├── sounds/
│   ├── monster_step_heavy_1.wav
│   ├── monster_step_heavy_2.wav
│   └── player_step.wav
└── README.md
```

## HRTF Data

The game uses the MIT KEMAR HRTF dataset for realistic 3D audio positioning. The SOFA file should be placed at `hrtf/mit_kemar.sofa`.

Download from: https://sofacoustics.org/data/database/mit/

## Tips for Best Experience

1. **Use good headphones** - HRTF doesn't work with speakers
2. **Quiet environment** - External sounds interfere with perception
3. **Start with training** - Learn the audio encoding before playing
4. **Close your eyes** - Forces your brain to rely on audio
5. **Go slow** - Build mental map before moving fast

## Troubleshooting

**No sound?**
- Check `python spatial_audio.py -l` for audio devices
- Use `python spatial_audio.py -d INDEX` to select device

**Face tracking not working?**
- Check webcam permissions
- Ensure good lighting
- Run `python face_tracking.py` to test standalone

**HRTF file not found?**
- Download MIT KEMAR SOFA file
- Place in `hrtf/mit_kemar.sofa`
- Or specify path: `--hrtf /path/to/file.sofa`

## Science Behind It

This project implements **sensory substitution** - using one sense (hearing) to convey information typically processed by another (vision). 

Key concepts:
- **HRTF**: Head-Related Transfer Functions encode how sound reaches each ear from different directions
- **Echolocation**: Click rate encoding mimics how bats perceive distance
- **Cross-modal plasticity**: The brain can learn to interpret audio as spatial information

Research shows 10-20 hours of training enables basic spatial perception through audio.

## License

MIT License - Use freely, attribution appreciated.

## Credits

- HRTF Data: MIT Media Lab (KEMAR dataset)
- Inspired by: The vOICe, BrainPort, and sensory substitution research
- Audio Engine: Built with NumPy, SciPy, and python-sounddevice
