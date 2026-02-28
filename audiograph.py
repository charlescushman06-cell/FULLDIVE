#!/usr/bin/env python3
"""
Cushman Audiograph - Image to Spatial Audio

Convert images to 3D spatial audio using HRTF.
- X-axis → HRTF spatial position (left-to-right = -90° to +90°)
- Y-axis → Frequency (top = high pitch, bottom = low pitch)
- Brightness → Volume

Usage:
    python audiograph.py image.png              # Static render
    python audiograph.py image.png --scan       # Scanning render
    python audiograph.py --camera               # Live camera feed
"""

import numpy as np
import sounddevice as sd
from scipy import signal
from PIL import Image
import argparse
import sys
import os

# Import HRTF from main game
sys.path.insert(0, os.path.dirname(__file__))
from spatial_audio import load_hrtf, interpolate_hrtf, SAMPLE_RATE

# Audio parameters
DURATION = 2.0  # seconds for static render
SCAN_DURATION = 3.0  # seconds for scan
NUM_FREQ_BANDS = 32  # vertical resolution
NUM_SPATIAL_BINS = 17  # horizontal resolution (-90° to +90° in 11.25° steps)

# Frequency range (Hz)
MIN_FREQ = 200
MAX_FREQ = 5000


def image_to_matrix(image_path, width=NUM_SPATIAL_BINS, height=NUM_FREQ_BANDS):
    """Load image and convert to brightness matrix."""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    # Convert to numpy, normalize to 0-1, flip vertically (top = high freq)
    matrix = np.array(img, dtype=np.float32) / 255.0
    matrix = np.flipud(matrix)  # Flip so top of image = high frequency
    
    return matrix


def matrix_to_audio_static(matrix, duration=DURATION):
    """Convert brightness matrix to spatial audio (all at once)."""
    height, width = matrix.shape
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    
    # Frequency bands (log scale)
    freqs = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), height)
    
    # Spatial positions (-90° to +90°)
    angles = np.linspace(-90, 90, width)
    
    left_channel = np.zeros(samples, dtype=np.float32)
    right_channel = np.zeros(samples, dtype=np.float32)
    
    print(f"Rendering {width}x{height} image to spatial audio...")
    
    for x in range(width):
        angle = angles[x]
        
        # Get HRTF for this angle
        left_ir, right_ir = interpolate_hrtf(-angle, elevation_deg=0, n_neighbors=4)
        
        # Generate column audio (sum of frequency bands weighted by brightness)
        column_audio = np.zeros(samples, dtype=np.float32)
        
        for y in range(height):
            brightness = matrix[y, x]
            if brightness < 0.05:  # Skip very dark pixels
                continue
            
            freq = freqs[y]
            # Generate tone with gentle envelope
            tone = np.sin(2 * np.pi * freq * t) * brightness
            column_audio += tone
        
        if np.max(np.abs(column_audio)) < 0.001:
            continue
        
        # Normalize column
        column_audio = column_audio / (height * 0.5)
        
        # Convolve with HRTF
        left_conv = signal.fftconvolve(column_audio, left_ir, mode='full')[:samples]
        right_conv = signal.fftconvolve(column_audio, right_ir, mode='full')[:samples]
        
        left_channel += left_conv.astype(np.float32)
        right_channel += right_conv.astype(np.float32)
    
    # Normalize output
    max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)), 0.001)
    left_channel = left_channel / max_val * 0.7
    right_channel = right_channel / max_val * 0.7
    
    return np.column_stack([left_channel, right_channel])


def matrix_to_audio_scan(matrix, duration=SCAN_DURATION):
    """Convert brightness matrix to scanning spatial audio (sweep left-to-right)."""
    height, width = matrix.shape
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    
    # Frequency bands (log scale)
    freqs = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), height)
    
    left_channel = np.zeros(samples, dtype=np.float32)
    right_channel = np.zeros(samples, dtype=np.float32)
    
    print(f"Scanning {width}x{height} image to spatial audio...")
    
    # Samples per column
    samples_per_col = samples // width
    
    for x in range(width):
        # Angle sweeps from -90° to +90°
        angle = -90 + (x / (width - 1)) * 180
        
        # Get HRTF for this angle
        left_ir, right_ir = interpolate_hrtf(-angle, elevation_deg=0, n_neighbors=4)
        
        # Time window for this column
        start_sample = x * samples_per_col
        end_sample = start_sample + samples_per_col
        col_t = t[start_sample:end_sample]
        col_samples = len(col_t)
        
        # Generate column audio
        column_audio = np.zeros(col_samples, dtype=np.float32)
        
        for y in range(height):
            brightness = matrix[y, x]
            if brightness < 0.05:
                continue
            
            freq = freqs[y]
            # Envelope for smooth transitions
            envelope = np.sin(np.pi * np.linspace(0, 1, col_samples))
            tone = np.sin(2 * np.pi * freq * col_t) * brightness * envelope
            column_audio += tone
        
        if np.max(np.abs(column_audio)) < 0.001:
            continue
        
        # Normalize
        column_audio = column_audio / (height * 0.3)
        
        # Convolve with HRTF
        left_conv = signal.fftconvolve(column_audio, left_ir, mode='full')[:col_samples]
        right_conv = signal.fftconvolve(column_audio, right_ir, mode='full')[:col_samples]
        
        # Add to output at correct position
        actual_end = min(end_sample, samples)
        actual_len = min(len(left_conv), actual_end - start_sample)
        
        left_channel[start_sample:start_sample+actual_len] += left_conv[:actual_len].astype(np.float32)
        right_channel[start_sample:start_sample+actual_len] += right_conv[:actual_len].astype(np.float32)
    
    # Normalize output
    max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)), 0.001)
    left_channel = left_channel / max_val * 0.7
    right_channel = right_channel / max_val * 0.7
    
    return np.column_stack([left_channel, right_channel])


def play_audio(audio):
    """Play stereo audio."""
    print("Playing... (put on headphones!)")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description='Cushman Audiograph - Image to Spatial Audio')
    parser.add_argument('image', nargs='?', help='Path to image file')
    parser.add_argument('--scan', action='store_true', help='Use scanning mode (left-to-right sweep)')
    parser.add_argument('--duration', '-d', type=float, default=2.0, help='Duration in seconds')
    parser.add_argument('--hrtf', default='hrtf/mit_kemar.sofa', help='Path to HRTF file')
    parser.add_argument('--save', '-s', help='Save to WAV file instead of playing')
    parser.add_argument('--cols', type=int, default=NUM_SPATIAL_BINS, help='Horizontal resolution')
    parser.add_argument('--rows', type=int, default=NUM_FREQ_BANDS, help='Vertical resolution (frequency bands)')
    args = parser.parse_args()
    
    if not args.image:
        print("Usage: python audiograph.py <image.png> [--scan] [--duration 2.0]")
        print("\nExample images to try:")
        print("  - A simple shape (circle, square)")
        print("  - A face")
        print("  - Text/letters")
        return
    
    # Load HRTF
    print(f"Loading HRTF from {args.hrtf}...")
    load_hrtf(args.hrtf)
    
    # Load image
    print(f"Loading image: {args.image}")
    matrix = image_to_matrix(args.image, width=args.cols, height=args.rows)
    print(f"Image size: {matrix.shape[1]}x{matrix.shape[0]} (width x height)")
    
    # Convert to audio
    if args.scan:
        audio = matrix_to_audio_scan(matrix, duration=args.duration)
    else:
        audio = matrix_to_audio_static(matrix, duration=args.duration)
    
    # Output
    if args.save:
        from scipy.io import wavfile
        wavfile.write(args.save, SAMPLE_RATE, audio)
        print(f"Saved to {args.save}")
    else:
        play_audio(audio)


if __name__ == "__main__":
    main()
