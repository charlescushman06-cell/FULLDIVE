#!/usr/bin/env python3
"""
Dream Space Trainer - Learn the spatial audio encoding

Training modes:
1. Direction identification - hear a click, identify the direction
2. Distance estimation - hear clicks, estimate wall distance
3. Navigation challenge - find the center of the room by sound alone
4. Maze navigation - navigate a simple maze eyes closed
"""

import numpy as np
import sounddevice as sd
import time
import random
import math

SAMPLE_RATE = 44100

# Frequencies for each direction
FREQ_MAP = {
    'front': 880,
    'back': 440, 
    'left': 660,
    'right': 1100
}

def generate_tone(freq, duration=0.1, volume=0.3):
    """Generate a tone with envelope"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    envelope = np.exp(-t * 20) * (1 - np.exp(-t * 100))  # Attack + decay
    tone = np.sin(2 * np.pi * freq * t) * envelope * volume
    return tone.astype(np.float32)

def generate_clicks(freq, rate, duration=2.0, volume=0.3):
    """Generate repeated clicks at given rate"""
    samples = int(SAMPLE_RATE * duration)
    output = np.zeros(samples, dtype=np.float32)
    
    click = generate_tone(freq, duration=0.02, volume=volume)
    click_interval = int(SAMPLE_RATE / rate)
    
    pos = 0
    while pos + len(click) < samples:
        output[pos:pos+len(click)] += click
        pos += click_interval
    
    return output

def play_sound(audio):
    """Play audio and wait for completion"""
    sd.play(audio, SAMPLE_RATE)
    sd.wait()

def direction_training():
    """Mode 1: Learn to identify directions by frequency"""
    print("\n" + "="*50)
    print("DIRECTION TRAINING")
    print("="*50)
    print("\nLearn which frequency = which direction:")
    print("  Front = 880 Hz (high)")
    print("  Right = 1100 Hz (highest)")
    print("  Left  = 660 Hz (mid)")
    print("  Back  = 440 Hz (low)")
    print("\nI'll play a sound, you guess the direction.")
    print("Press Enter to start, 'q' to quit.\n")
    
    input()
    
    directions = list(FREQ_MAP.keys())
    correct = 0
    total = 0
    
    while True:
        direction = random.choice(directions)
        freq = FREQ_MAP[direction]
        
        print("Playing...")
        play_sound(generate_tone(freq, duration=0.3))
        time.sleep(0.2)
        play_sound(generate_tone(freq, duration=0.3))
        
        answer = input("\nDirection (front/back/left/right) or 'r' to replay, 'q' to quit: ").lower().strip()
        
        if answer == 'q':
            break
        if answer == 'r':
            play_sound(generate_tone(freq, duration=0.3))
            answer = input("Direction: ").lower().strip()
        
        total += 1
        if answer == direction:
            correct += 1
            print(f"✓ Correct! ({correct}/{total} = {100*correct/total:.0f}%)")
        else:
            print(f"✗ Wrong - it was {direction} ({freq} Hz)")
            print(f"  ({correct}/{total} = {100*correct/total:.0f}%)")
        print()
    
    print(f"\nFinal score: {correct}/{total}")

def distance_training():
    """Mode 2: Learn to estimate distance by click rate"""
    print("\n" + "="*50)
    print("DISTANCE TRAINING")  
    print("="*50)
    print("\nClick rate = distance to wall:")
    print("  Fast clicks (10/sec) = very close (~0.5 units)")
    print("  Slow clicks (1/sec)  = far away (~10 units)")
    print("\nI'll play clicks, you estimate the distance (1-10).")
    print("Press Enter to start, 'q' to quit.\n")
    
    input()
    
    correct = 0
    total = 0
    
    while True:
        # Random distance 1-10
        distance = random.uniform(1, 10)
        # Convert to click rate (inverse relationship)
        rate = 10.5 - distance  # 10 clicks/sec at dist=0.5, 0.5 at dist=10
        rate = max(0.5, min(10, rate))
        
        print("Playing...")
        play_sound(generate_clicks(880, rate, duration=3.0))
        
        answer = input("\nEstimate distance (1-10) or 'r' to replay, 'q' to quit: ").lower().strip()
        
        if answer == 'q':
            break
        if answer == 'r':
            play_sound(generate_clicks(880, rate, duration=3.0))
            answer = input("Distance: ").strip()
        
        try:
            guess = float(answer)
            error = abs(guess - distance)
            total += 1
            
            if error < 1.5:
                correct += 1
                print(f"✓ Close enough! Actual: {distance:.1f} ({correct}/{total})")
            else:
                print(f"✗ Actual distance: {distance:.1f} (you said {guess})")
            print()
        except ValueError:
            print("Enter a number 1-10")

def combined_training():
    """Mode 3: Direction + distance together"""
    print("\n" + "="*50)
    print("COMBINED TRAINING")
    print("="*50)
    print("\nNow both together: identify direction AND distance")
    print("Press Enter to start, 'q' to quit.\n")
    
    input()
    
    directions = list(FREQ_MAP.keys())
    
    while True:
        direction = random.choice(directions)
        distance = random.uniform(1, 10)
        rate = max(0.5, min(10, 10.5 - distance))
        freq = FREQ_MAP[direction]
        
        print("Playing...")
        play_sound(generate_clicks(freq, rate, duration=3.0))
        
        answer = input("\nDirection distance (e.g. 'front 3') or 'r'/'q': ").lower().strip()
        
        if answer == 'q':
            break
        if answer == 'r':
            play_sound(generate_clicks(freq, rate, duration=3.0))
            answer = input("Direction distance: ").lower().strip()
        
        parts = answer.split()
        if len(parts) >= 2:
            dir_guess = parts[0]
            try:
                dist_guess = float(parts[1])
                
                dir_correct = dir_guess == direction
                dist_correct = abs(dist_guess - distance) < 2
                
                if dir_correct and dist_correct:
                    print(f"✓ Both correct! ({direction}, {distance:.1f})")
                elif dir_correct:
                    print(f"~ Direction right, distance was {distance:.1f}")
                elif dist_correct:
                    print(f"~ Distance close, direction was {direction}")
                else:
                    print(f"✗ Was: {direction}, {distance:.1f}")
            except:
                print("Format: direction distance (e.g. 'front 5')")
        print()

def main():
    print("="*50)
    print("DREAM SPACE TRAINER")
    print("="*50)
    print()
    print("Training modes:")
    print("  1. Direction identification (learn the frequencies)")
    print("  2. Distance estimation (learn click rates)")
    print("  3. Combined (direction + distance)")
    print("  q. Quit")
    print()
    
    while True:
        choice = input("Select mode (1/2/3/q): ").strip()
        
        if choice == '1':
            direction_training()
        elif choice == '2':
            distance_training()
        elif choice == '3':
            combined_training()
        elif choice == 'q':
            break
        else:
            print("Enter 1, 2, 3, or q")

if __name__ == "__main__":
    main()
