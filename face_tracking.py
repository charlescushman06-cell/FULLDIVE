#!/usr/bin/env python3
"""
Face tracking module for Dream Space

Uses MediaPipe Face Mesh to track head rotation via webcam.
Maps head yaw to player angle for hands-free navigation.

Requirements:
    pip install mediapipe opencv-python

Usage:
    from face_tracking import FaceTracker
    tracker = FaceTracker()
    tracker.start()
    
    # In game loop:
    yaw = tracker.get_head_yaw()  # Returns angle in radians
    
    tracker.stop()
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import math


class FaceTracker:
    def __init__(self, camera_index=0, smoothing=0.3):
        """
        Initialize face tracker.
        
        Args:
            camera_index: Webcam index (0 = default)
            smoothing: Smoothing factor (0-1, higher = smoother but more lag)
        """
        self.camera_index = camera_index
        self.smoothing = smoothing
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        
        # State
        self.running = False
        self.thread = None
        self.cap = None
        
        # Head orientation (radians)
        self.head_yaw = 0.0      # Left/right rotation
        self.head_pitch = 0.0   # Up/down tilt
        self.head_roll = 0.0    # Head tilt
        
        # Calibration - center position
        self.center_yaw = None
        self.calibrated = False
        
        # Smoothed values
        self.smooth_yaw = 0.0
        
    def start(self):
        """Start face tracking in background thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()
        
        print("Face tracking started. Look straight ahead to calibrate...")
        
    def stop(self):
        """Stop face tracking."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        if self.face_mesh:
            self.face_mesh.close()
            
    def calibrate(self):
        """Set current head position as center."""
        self.center_yaw = self.head_yaw
        self.calibrated = True
        print(f"Calibrated! Center yaw: {math.degrees(self.center_yaw):.1f}°")
        
    def get_head_yaw(self):
        """Get head yaw angle relative to calibrated center (radians)."""
        if not self.calibrated:
            return 0.0
        return self.smooth_yaw - self.center_yaw
    
    def get_head_pitch(self):
        """Get head pitch angle (radians). Positive = looking up."""
        return self.head_pitch
    
    def _tracking_loop(self):
        """Main tracking loop (runs in background thread)."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Auto-calibrate after a few frames
        calibrate_countdown = 30  # ~1 second at 30fps
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Calculate head orientation from key landmarks
                # Using nose tip, forehead, and chin for orientation
                nose_tip = landmarks[1]      # Nose tip
                forehead = landmarks[10]     # Forehead
                chin = landmarks[152]        # Chin
                left_eye = landmarks[33]     # Left eye outer corner
                right_eye = landmarks[263]   # Right eye outer corner
                
                # Calculate yaw from eye positions (simple but effective)
                eye_dx = right_eye.x - left_eye.x
                eye_dz = right_eye.z - left_eye.z
                
                # Yaw: when head turns right, left eye comes forward (negative z diff)
                # Scale factor determined empirically
                self.head_yaw = math.atan2(-eye_dz, eye_dx) * 2.5
                
                # Calculate pitch from nose-forehead-chin alignment
                vertical_dz = chin.z - forehead.z
                self.head_pitch = math.atan2(vertical_dz, 0.15) * 2
                
                # Smooth the yaw
                self.smooth_yaw = (self.smooth_yaw * self.smoothing + 
                                   self.head_yaw * (1 - self.smoothing))
                
                # Auto-calibrate
                if calibrate_countdown > 0:
                    calibrate_countdown -= 1
                    if calibrate_countdown == 0:
                        self.calibrate()
            
            time.sleep(0.016)  # ~60fps max
    
    def is_face_detected(self):
        """Check if a face is currently being tracked."""
        return self.calibrated


def test_face_tracking():
    """Test face tracking standalone."""
    print("Testing face tracking...")
    print("Look straight ahead, then turn your head left/right.")
    print("Press Ctrl+C to exit.\n")
    
    tracker = FaceTracker()
    tracker.start()
    
    try:
        while True:
            yaw_deg = math.degrees(tracker.get_head_yaw())
            pitch_deg = math.degrees(tracker.get_head_pitch())
            
            # Visual bar for yaw
            bar_width = 40
            center = bar_width // 2
            pos = int(center + (yaw_deg / 90) * center)
            pos = max(0, min(bar_width - 1, pos))
            bar = ['-'] * bar_width
            bar[center] = '|'
            bar[pos] = 'O'
            
            print(f"\rYaw: {yaw_deg:+6.1f}°  Pitch: {pitch_deg:+6.1f}°  [{''.join(bar)}]", end='')
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        tracker.stop()


if __name__ == "__main__":
    test_face_tracking()
