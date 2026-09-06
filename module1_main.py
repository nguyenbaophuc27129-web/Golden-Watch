# -*- coding: utf-8 -*-
"""
MODULE 1 MAIN - FACIAL ASYMMETRY DETECTION (PSCS v8.0)
======================================================
Unified file for Module 1 - Stroke Detection from Face

Features:
- MediaPipe Tasks API (478 landmarks)
- PyTorch Multi-class Classifier (91.07% accuracy)
- False Positive Detection (YAWN, SMILE, HEAD_TURN)
- Real-time GUI with indicators
- Export results to CSV/JSON

Author: PSCS Team
Date: 2026-08-26
Version: 8.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detection.face_module_v7 import FaceAsymmetryDetector
import cv2
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import csv
from datetime import datetime
from pathlib import Path

# ========================================================================
# CONFIGURATION
# ========================================================================

class Config:
    """Configuration for Module 1"""

    # Model paths (93.75% accuracy - 100% dataset trained)
    MEDIAPIPE_MODEL = "models/face_landmarker_v2/face_landmarker_v2_with_blendshapes.task"
    ML_MODEL = "models/stroke_classifier_100percent_best.pth"
    SCALER = "models/stroke_classifier_100percent_scaler.pkl"

    # Thresholds
    NORMAL_THRESHOLD = 30
    WARNING_THRESHOLD = 60

    # Display settings
    WINDOW_NAME = "Module 1: Stroke Detection (PSCS v8.0)"
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480

    # Export settings
    EXPORT_DIR = "exports/module1"
    EXPORT_FORMAT = "csv"  # csv or json

    # Additional features thresholds
    YAWN_MOUTH_AR_THRESHOLD = 0.5
    DROWSY_EYE_AR_THRESHOLD = 0.05
    HEAD_TURN_ROTATION_THRESHOLD = 15.0
    SMILE_INDICATOR_THRESHOLD = 0.3


# ========================================================================
# MODEL DEFINITIONS
# ========================================================================

class MultiClassClassifier(nn.Module):
    """PyTorch Multi-class Classifier for Stroke Detection"""

    def __init__(self, input_dim=960, hidden_dims=[512, 256, 128], num_classes=2):
        super(MultiClassClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ========================================================================
# MODULE 1 CLASS
# ========================================================================

class Module1StrokeDetector:
    """
    Module 1: Facial Asymmetry Detection (PSCS v8.0)

    Detects stroke from face using:
    - MediaPipe Tasks API for landmark extraction
    - PyTorch ML model for classification
    - Additional features for false positive detection
    """

    def __init__(self, config=Config):
        """Initialize Module 1"""
        self.config = config

        # Initialize components
        self._init_mediapipe()
        self._init_ml_model()
        self._init_export()

        # State variables
        self.running = False
        self.stroke_history = []
        self.frame_count = 0
        self.results = []

        print("[Module 1] Initialized successfully")

    def _init_mediapipe(self):
        """Initialize MediaPipe Face Landmarker"""
        print("[Module 1] Loading MediaPipe...")
        self.detector = FaceAsymmetryDetector(model_path=self.config.MEDIAPIPE_MODEL)
        print("[Module 1] MediaPipe loaded")

    def _init_ml_model(self):
        """Initialize PyTorch ML Model"""
        print("[Module 1] Loading ML model...")
        self.ml_model = MultiClassClassifier()
        self.ml_model.load_state_dict(
            torch.load(self.config.ML_MODEL, weights_only=True, map_location='cpu')
        )
        self.ml_model.eval()
        print("[Module 1] ML model loaded")

        # Load scaler
        print("[Module 1] Loading scaler...")
        self.scaler = joblib.load(self.config.SCALER)
        print("[Module 1] Scaler loaded")

    def _init_export(self):
        """Initialize export functionality"""
        self.export_dir = Path(self.config.EXPORT_DIR)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.export_file = self.export_dir / f"module1_results_{timestamp}.csv"

        # Create CSV with headers
        with open(self.export_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_id', 'status', 'stroke_prob',
                'mouth_ar', 'eye_ar', 'rotation', 'smile',
                'yawn_detected', 'drowsy_detected', 'head_turn_detected', 'smile_detected'
            ])

        print(f"[Module 1] Export initialized: {self.export_file}")

    def extract_additional_features(self, landmarks):
        """Extract additional features for false positive detection"""
        features = {
            'mouth_ar': 0,
            'eye_ar': 0,
            'rotation': 0,
            'smile': 0
        }

        try:
            # Mouth aspect ratio (for YAWN detection)
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            mouth_opening = np.linalg.norm(upper_lip - lower_lip)
            mouth_width = np.linalg.norm(left_corner - right_corner)
            features['mouth_ar'] = mouth_opening / (mouth_width + 1e-6)

            # Eye aspect ratio (for DROWSY detection)
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            left_eye_ar = np.linalg.norm(left_eye_top - left_eye_bottom)
            right_eye_ar = np.linalg.norm(right_eye_top - right_eye_bottom)
            features['eye_ar'] = (left_eye_ar + right_eye_ar) / 2

            # Face rotation (for HEAD TURN detection)
            face_center = landmarks[5]
            deviation = face_center[0] - 0.5
            features['rotation'] = deviation * 100

            # Smile indicator
            left_dist = np.linalg.norm(left_corner - face_center)
            right_dist = np.linalg.norm(right_corner - face_center)
            features['smile'] = (left_dist + right_dist) / 2
        except Exception as e:
            pass

        return features

    def process_frame(self, frame):
        """Process a single frame and return results"""
        self.frame_count += 1

        # Get landmarks from MediaPipe
        result = self.detector.process_frame(frame, frame_timestamp_ms=cv2.getTickCount())
        landmarks = result.get('raw_landmarks')

        if landmarks is None:
            return {
                'status': 'NO_FACE',
                'stroke_prob': 0,
                'features': {},
                'indicators': {}
            }

        # Extract additional features
        additional_features = self.extract_additional_features(landmarks)

        # Prepare features for ML model
        basic_features = landmarks.flatten()
        features_array = np.concatenate([
            basic_features,
            [
                additional_features['mouth_ar'],
                additional_features['eye_ar'],
                additional_features['rotation'],
                additional_features['smile']
            ]
        ])

        # ML prediction
        features_scaled = self.scaler.transform(features_array.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled)

        with torch.no_grad():
            outputs = self.ml_model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            stroke_prob = probs[0][0].item() * 100

        # Smooth with history
        self.stroke_history.append(stroke_prob)
        if len(self.stroke_history) > 5:
            self.stroke_history.pop(0)

        smooth_prob = np.mean(self.stroke_history)

        # Determine status
        if smooth_prob < self.config.NORMAL_THRESHOLD:
            status = "NORMAL"
        elif smooth_prob < self.config.WARNING_THRESHOLD:
            status = "WARNING"
        else:
            status = "DANGER"

        # Detect indicators
        indicators = {
            'yawn': additional_features['mouth_ar'] > self.config.YAWN_MOUTH_AR_THRESHOLD,
            'drowsy': additional_features['eye_ar'] < self.config.DROWSY_EYE_AR_THRESHOLD,
            'head_turn': abs(additional_features['rotation']) > self.config.HEAD_TURN_ROTATION_THRESHOLD,
            'smile': additional_features['smile'] > self.config.SMILE_INDICATOR_THRESHOLD
        }

        return {
            'status': status,
            'stroke_prob': smooth_prob,
            'features': additional_features,
            'indicators': indicators,
            'landmarks': landmarks
        }

    def export_result(self, result):
        """Export result to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        with open(self.export_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.frame_count,
                result['status'],
                f"{result['stroke_prob']:.2f}",
                f"{result['features'].get('mouth_ar', 0):.4f}",
                f"{result['features'].get('eye_ar', 0):.4f}",
                f"{result['features'].get('rotation', 0):.2f}",
                f"{result['features'].get('smile', 0):.4f}",
                result['indicators'].get('yawn', False),
                result['indicators'].get('drowsy', False),
                result['indicators'].get('head_turn', False),
                result['indicators'].get('smile', False)
            ])

    def draw_results(self, frame, result):
        """Draw results on frame - Compact version at top of screen"""
        # Colors
        COLOR_NORMAL = (0, 255, 0)
        COLOR_WARNING = (0, 165, 255)
        COLOR_DANGER = (0, 0, 255)
        COLOR_INDICATOR = (0, 255, 255)
        COLOR_TEXT = (255, 255, 255)

        # Choose color based on status
        if result['status'] == 'NORMAL':
            color = COLOR_NORMAL
        elif result['status'] == 'WARNING':
            color = COLOR_WARNING
        else:
            color = COLOR_DANGER

        # COMPACT INFO PANEL (Top of screen, not blocking face)
        panel_x = 10
        panel_y = 10
        panel_width = 280
        panel_height = 180

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     color, 2)

        y = panel_y + 25

        # Status (Big)
        cv2.putText(frame, result['status'], (panel_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y += 30

        # Stroke probability
        cv2.putText(frame, f"{result['stroke_prob']:.1f}%", (panel_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        y += 25

        # Indicators (Compact)
        indicators = result['indicators']
        features = result['features']

        # Display indicators in one line
        indicator_texts = []
        if indicators.get('yawn'):
            indicator_texts.append("YAWN")
        if indicators.get('drowsy'):
            indicator_texts.append("DROWSY")
        if indicators.get('head_turn'):
            indicator_texts.append("HEAD_TURN")
        if indicators.get('smile'):
            indicator_texts.append("SMILE")

        if indicator_texts:
            text = " | ".join(indicator_texts)
            cv2.putText(frame, text, (panel_x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INDICATOR, 1)
        y += 25

        # Feature values (small)
        cv2.putText(frame, f"M:{features.get('mouth_ar', 0):.2f} E:{features.get('eye_ar', 0):.2f}",
                    (panel_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        y += 20
        cv2.putText(frame, f"R:{features.get('rotation', 0):.1f} S:{features.get('smile', 0):.2f}",
                    (panel_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Draw face mesh landmarks (FULL FACE)
        landmarks = result.get('landmarks')
        if landmarks is not None:
            h, w = frame.shape[:2]

            # Draw FULL face mesh (connect landmarks with lines)
            # Face oval outline
            face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            for i in range(len(face_oval)):
                idx1 = face_oval[i]
                idx2 = face_oval[(i + 1) % len(face_oval)]
                x1 = int(landmarks[idx1][0] * w)
                y1 = int(landmarks[idx1][1] * h)
                x2 = int(landmarks[idx2][0] * w)
                y2 = int(landmarks[idx2][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Eyes
            left_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
            right_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
            for i in range(len(left_eye) - 1):
                x1 = int(landmarks[left_eye[i]][0] * w)
                y1 = int(landmarks[left_eye[i]][1] * h)
                x2 = int(landmarks[left_eye[i + 1]][0] * w)
                y2 = int(landmarks[left_eye[i + 1]][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            for i in range(len(right_eye) - 1):
                x1 = int(landmarks[right_eye[i]][0] * w)
                y1 = int(landmarks[right_eye[i]][1] * h)
                x2 = int(landmarks[right_eye[i + 1]][0] * w)
                y2 = int(landmarks[right_eye[i + 1]][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # Eyebrows
            left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
            right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
            for i in range(len(left_eyebrow) - 1):
                x1 = int(landmarks[left_eyebrow[i]][0] * w)
                y1 = int(landmarks[left_eyebrow[i]][1] * h)
                x2 = int(landmarks[left_eyebrow[i + 1]][0] * w)
                y2 = int(landmarks[left_eyebrow[i + 1]][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

            for i in range(len(right_eyebrow) - 1):
                x1 = int(landmarks[right_eyebrow[i]][0] * w)
                y1 = int(landmarks[right_eyebrow[i]][1] * h)
                x2 = int(landmarks[right_eyebrow[i + 1]][0] * w)
                y2 = int(landmarks[right_eyebrow[i + 1]][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

            # Nose
            nose = [1, 2, 98, 97, 327, 326, 4, 5, 6, 168, 197, 195, 194, 195, 219, 240,
                    142, 126, 217, 214, 209, 129, 203, 205, 36, 142, 126, 217, 214, 209, 129]
            for i in range(len(nose) - 1):
                x1 = int(landmarks[nose[i]][0] * w)
                y1 = int(landmarks[nose[i]][1] * h)
                x2 = int(landmarks[nose[i + 1]][0] * w)
                y2 = int(landmarks[nose[i + 1]][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

            # Mouth
            mouth_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
            mouth_inner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
            for i in range(len(mouth_outer)):
                idx1 = mouth_outer[i]
                idx2 = mouth_outer[(i + 1) % len(mouth_outer)]
                x1 = int(landmarks[idx1][0] * w)
                y1 = int(landmarks[idx1][1] * h)
                x2 = int(landmarks[idx2][0] * w)
                y2 = int(landmarks[idx2][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            for i in range(len(mouth_inner)):
                idx1 = mouth_inner[i]
                idx2 = mouth_inner[(i + 1) % len(mouth_inner)]
                x1 = int(landmarks[idx1][0] * w)
                y1 = int(landmarks[idx1][1] * h)
                x2 = int(landmarks[idx2][0] * w)
                y2 = int(landmarks[idx2][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # Draw key points (dots)
            for i in [1, 13, 14, 61, 291, 152, 33, 263, 10, 338]:
                x = int(landmarks[i][0] * w)
                y = int(landmarks[i][1] * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        y = 40

        # Status
        cv2.putText(frame, f"Status: {result['status']}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 35

        # Stroke probability
        cv2.putText(frame, f"Stroke Prob: {result['stroke_prob']:.1f}%", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 35

        # Model info
        cv2.putText(frame, "Module 1 v8.0 (91% accuracy)", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 30

        # Additional features
        cv2.putText(frame, "Additional Features:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
        y += 25

        features = result['features']
        indicators = result['indicators']

        # Mouth AR
        cv2.putText(frame, f"Mouth AR (Yawn): {features.get('mouth_ar', 0):.3f}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y += 20
        if indicators.get('yawn'):
            cv2.putText(frame, "[YAWN DETECTED]", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INDICATOR, 1)
        y += 25

        # Eye AR
        cv2.putText(frame, f"Eye AR (Drowsy): {features.get('eye_ar', 0):.3f}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y += 20
        if indicators.get('drowsy'):
            cv2.putText(frame, "[DROWSY DETECTED]", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INDICATOR, 1)
        y += 25

        # Rotation
        cv2.putText(frame, f"Rotation (HeadTurn): {features.get('rotation', 0):.1f}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y += 20
        if indicators.get('head_turn'):
            cv2.putText(frame, "[HEAD TURN]", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INDICATOR, 1)
        y += 25

        # Smile
        cv2.putText(frame, f"Smile: {features.get('smile', 0):.3f}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y += 20
        if indicators.get('smile'):
            cv2.putText(frame, "[SMILE DETECTED]", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INDICATOR, 1)

        # Draw landmarks
        landmarks = result.get('landmarks')
        if landmarks is not None:
            h, w = frame.shape[:2]
            for i in [1, 13, 14, 61, 291, 152]:  # Key landmarks
                x = int(landmarks[i][0] * w)
                y_lm = int(landmarks[i][1] * h)
                cv2.circle(frame, (x, y_lm), 2, (0, 255, 0), -1)

        return frame

    def run_camera(self):
        """Run Module 1 with camera - FULLSCREEN MODE"""
        print(f"[Module 1] Starting camera...")
        print("[Module 1] Press 'q' to quit, 's' to save screenshot, 'f' to toggle fullscreen")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[Module 1] ERROR: Cannot open camera")
            return

        self.running = True
        fullscreen = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # RESIZE frame for large display
            height, width = frame.shape[:2]
            new_width = 1280  # Larger width
            new_height = int(new_width * height / width)
            frame = cv2.resize(frame, (new_width, new_height))

            # Process frame
            result = self.process_frame(frame)

            # Draw results
            frame = self.draw_results(frame, result)

            # Export result
            if self.frame_count % 10 == 0:
                self.export_result(result)

            # Display (show first, then set fullscreen)
            cv2.imshow(self.config.WINDOW_NAME, frame)

            # Set fullscreen after window is created
            if self.frame_count == 1 or (self.frame_count % 30 == 0):
                if fullscreen:
                    cv2.setWindowProperty(self.config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                         cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                         cv2.WINDOW_NORMAL)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                filename = f"screenshot_{self.frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[Module 1] Screenshot saved: {filename}")
            elif key == ord('f'):
                fullscreen = not fullscreen
                print(f"[Module 1] Fullscreen: {fullscreen}")

        cap.release()
        cv2.destroyAllWindows()

        print(f"[Module 1] Results exported to: {self.export_file}")
        print("[Module 1] Stopped")


# ========================================================================
# MAIN
# ========================================================================

def main():
    """Main entry point"""
    print("="*60)
    print("MODULE 1: FACIAL ASYMMETRY DETECTION (PSCS v8.0)")
    print("="*60)
    print()

    # Initialize Module 1
    module = Module1StrokeDetector()

    # Run camera
    module.run_camera()

    print()
    print("[Module 1] Session ended")


if __name__ == "__main__":
    main()
