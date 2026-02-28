#!/usr/bin/env python3
"""
Cushman Audiograph Trainer - Learn to "see" with sound

Progressive training from simple shapes to complex scenes.

Levels:
1. Spatial (left/center/right)
2. Vertical (high/low pitch = top/bottom)
3. Shapes (circle, square, triangle, line)
4. Letters (A-Z)
5. Numbers (0-9)
6. Objects (simple icons)
7. Scenes (complex images)
"""

import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont
import random
import time
import sys
import os
import json
import termios
import tty

# Import from audiograph
sys.path.insert(0, os.path.dirname(__file__))
from audiograph import (
    load_hrtf, image_to_matrix, matrix_to_audio_static, 
    matrix_to_audio_scan, SAMPLE_RATE
)

# Training state file
STATE_FILE = os.path.expanduser("~/.audiograph_progress.json")

def load_progress():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"level": 1, "score": 0, "total": 0, "history": []}

def save_progress(progress):
    with open(STATE_FILE, 'w') as f:
        json.dump(progress, f)


# ===== IMAGE GENERATORS =====

def gen_spatial_position():
    """Level 1: Left, center, or right dot"""
    positions = ['left', 'center', 'right']
    answer = random.choice(positions)
    
    img = Image.new('L', (200, 100), 0)
    draw = ImageDraw.Draw(img)
    
    if answer == 'left':
        draw.ellipse([20, 30, 60, 70], fill=255)
    elif answer == 'center':
        draw.ellipse([80, 30, 120, 70], fill=255)
    else:  # right
        draw.ellipse([140, 30, 180, 70], fill=255)
    
    return img, answer, positions

def gen_vertical_position():
    """Level 2: Top, middle, or bottom"""
    positions = ['top', 'middle', 'bottom']
    answer = random.choice(positions)
    
    img = Image.new('L', (100, 200), 0)
    draw = ImageDraw.Draw(img)
    
    if answer == 'top':
        draw.ellipse([35, 20, 65, 50], fill=255)
    elif answer == 'middle':
        draw.ellipse([35, 85, 65, 115], fill=255)
    else:  # bottom
        draw.ellipse([35, 150, 65, 180], fill=255)
    
    return img, answer, positions

def gen_shape():
    """Level 3: Basic shapes"""
    shapes = ['circle', 'square', 'triangle', 'hline', 'vline']
    answer = random.choice(shapes)
    
    img = Image.new('L', (150, 150), 0)
    draw = ImageDraw.Draw(img)
    
    if answer == 'circle':
        draw.ellipse([30, 30, 120, 120], fill=255)
    elif answer == 'square':
        draw.rectangle([35, 35, 115, 115], fill=255)
    elif answer == 'triangle':
        draw.polygon([(75, 25), (25, 125), (125, 125)], fill=255)
    elif answer == 'hline':
        draw.rectangle([20, 60, 130, 90], fill=255)
    elif answer == 'vline':
        draw.rectangle([60, 20, 90, 130], fill=255)
    
    return img, answer, shapes

def gen_letter():
    """Level 4: Capital letters"""
    letters = list('ACEILMNOT')  # Start with distinct letters
    answer = random.choice(letters)
    
    img = Image.new('L', (120, 150), 0)
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fall back to basic
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
    except:
        font = ImageFont.load_default()
    
    # Center the letter
    bbox = draw.textbbox((0, 0), answer, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (120 - w) // 2 - bbox[0]
    y = (150 - h) // 2 - bbox[1]
    draw.text((x, y), answer, fill=255, font=font)
    
    return img, answer, letters

def gen_number():
    """Level 5: Numbers 0-9"""
    numbers = list('0123456789')
    answer = random.choice(numbers)
    
    img = Image.new('L', (120, 150), 0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), answer, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (120 - w) // 2 - bbox[0]
    y = (150 - h) // 2 - bbox[1]
    draw.text((x, y), answer, fill=255, font=font)
    
    return img, answer, numbers

def gen_object():
    """Level 6: Simple objects/icons"""
    objects = {
        'face': lambda d: (d.ellipse([20,20,130,130], fill=128), 
                          d.ellipse([45,50,65,70], fill=255),
                          d.ellipse([85,50,105,70], fill=255),
                          d.arc([50,80,100,110], 0, 180, fill=255, width=5)),
        'house': lambda d: (d.polygon([(75,20), (20,70), (130,70)], fill=255),
                           d.rectangle([35,70,115,130], fill=200),
                           d.rectangle([60,90,90,130], fill=0)),
        'tree': lambda d: (d.ellipse([30,20,120,90], fill=200),
                          d.rectangle([65,80,85,140], fill=150)),
        'star': lambda d: d.polygon([(75,10), (90,60), (140,60), (100,90), 
                                     (115,140), (75,110), (35,140), (50,90),
                                     (10,60), (60,60)], fill=255),
        'arrow': lambda d: (d.polygon([(75,20), (40,70), (60,70), (60,130), 
                                       (90,130), (90,70), (110,70)], fill=255)),
    }
    
    answer = random.choice(list(objects.keys()))
    
    img = Image.new('L', (150, 150), 0)
    draw = ImageDraw.Draw(img)
    objects[answer](draw)
    
    return img, answer, list(objects.keys())


LEVEL_GENERATORS = {
    1: ("Spatial Position", gen_spatial_position, 10),
    2: ("Vertical Position", gen_vertical_position, 10),
    3: ("Basic Shapes", gen_shape, 15),
    4: ("Letters", gen_letter, 15),
    5: ("Numbers", gen_number, 15),
    6: ("Objects", gen_object, 20),
}


def play_image(img, use_scan=False, duration=2.0):
    """Convert image to audio and play it."""
    matrix = np.array(img.convert('L'), dtype=np.float32) / 255.0
    matrix = np.flipud(matrix)
    
    if use_scan:
        audio = matrix_to_audio_scan(matrix, duration=duration)
    else:
        audio = matrix_to_audio_static(matrix, duration=duration)
    
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def get_key():
    """Get single keypress."""
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def run_trial(level_num, use_scan=False):
    """Run a single trial for the given level."""
    level_name, generator, _ = LEVEL_GENERATORS[level_num]
    
    img, answer, options = generator()
    
    print(f"\n  🔊 Playing sound... (Level {level_num}: {level_name})")
    print(f"  Options: {', '.join(options)}")
    
    # Play the image
    duration = 2.0 if level_num <= 2 else 3.0
    play_image(img, use_scan=(level_num >= 4), duration=duration)
    
    # Get answer
    print(f"\n  Your answer (first letter): ", end='', flush=True)
    user_input = get_key()
    print(user_input)
    
    # Check if correct (match first letter)
    correct = False
    for opt in options:
        if opt.lower().startswith(user_input):
            if opt.lower() == answer.lower():
                correct = True
            break
    
    if correct:
        print(f"  ✓ Correct! It was: {answer}")
    else:
        print(f"  ✗ Wrong. It was: {answer}")
        # Show the image briefly
        print("  [Press any key to see the image, then continue]")
        get_key()
        # Save image temporarily and tell user to view it
        temp_path = "/tmp/audiograph_answer.png"
        img.save(temp_path)
        print(f"  Image saved to: {temp_path}")
    
    return correct


def run_level(level_num, progress):
    """Run all trials for a level."""
    level_name, generator, num_trials = LEVEL_GENERATORS[level_num]
    
    print(f"\n{'='*50}")
    print(f"  LEVEL {level_num}: {level_name}")
    print(f"  {num_trials} trials - need 70% to advance")
    print(f"{'='*50}")
    print("\n  Press any key to start...")
    get_key()
    
    correct = 0
    for i in range(num_trials):
        print(f"\n  --- Trial {i+1}/{num_trials} ---")
        if run_trial(level_num):
            correct += 1
        progress['total'] += 1
        progress['score'] += 1 if run_trial else 0
        time.sleep(0.5)
    
    accuracy = correct / num_trials
    print(f"\n{'='*50}")
    print(f"  Level {level_num} Complete!")
    print(f"  Score: {correct}/{num_trials} ({accuracy*100:.0f}%)")
    
    if accuracy >= 0.7:
        print(f"  ★ PASSED! Advancing to level {level_num + 1}")
        progress['level'] = min(level_num + 1, len(LEVEL_GENERATORS))
        return True
    else:
        print(f"  Try again - need 70% to advance")
        return False


def main():
    print('\033[H\033[J', end='')
    print("╔════════════════════════════════════════════════════════╗")
    print("║        CUSHMAN AUDIOGRAPH TRAINER                      ║")
    print("║        Learn to 'see' with sound                       ║")
    print("╠════════════════════════════════════════════════════════╣")
    print("║                                                        ║")
    print("║  Progressive training:                                 ║")
    print("║    1. Spatial position (left/right)                    ║")
    print("║    2. Vertical position (top/bottom)                   ║")
    print("║    3. Basic shapes                                     ║")
    print("║    4. Letters                                          ║")
    print("║    5. Numbers                                          ║")
    print("║    6. Objects                                          ║")
    print("║                                                        ║")
    print("║  Controls:                                             ║")
    print("║    Type first letter of your answer                    ║")
    print("║    Q to quit                                           ║")
    print("║                                                        ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    # Load HRTF
    print("\nLoading HRTF...")
    hrtf_path = os.path.join(os.path.dirname(__file__), 'hrtf/mit_kemar.sofa')
    load_hrtf(hrtf_path)
    
    # Load progress
    progress = load_progress()
    print(f"\nCurrent level: {progress['level']}")
    print(f"Total attempts: {progress['total']}")
    
    print("\n[ENTER] Start from current level")
    print("[R] Reset to level 1")
    print("[1-6] Jump to specific level")
    print("[Q] Quit")
    
    key = get_key()
    
    if key == 'q':
        return
    elif key == 'r':
        progress = {"level": 1, "score": 0, "total": 0, "history": []}
    elif key in '123456':
        progress['level'] = int(key)
    
    # Run training
    try:
        while progress['level'] <= len(LEVEL_GENERATORS):
            run_level(progress['level'], progress)
            save_progress(progress)
            
            print("\n[ENTER] Continue  [R] Retry level  [Q] Quit")
            key = get_key()
            if key == 'q':
                break
            elif key == 'r':
                progress['level'] = max(1, progress['level'] - 1)
    except KeyboardInterrupt:
        pass
    
    save_progress(progress)
    print("\nProgress saved. Goodbye!")


if __name__ == "__main__":
    main()
