# -*- coding: utf-8 -*-
"""
MODULE 1 TEST SUITE - Follow TEST_GUIDE.md
Automated testing for all 6 test cases
"""

import sys
import os
import time

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.face_module_v7 import FaceAsymmetryDetector
import cv2
import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path

# Configuration
MEDIAPIPE_MODEL = os.path.join(parent_dir, "models/face_landmarker_v2/face_landmarker_v2_with_blendshapes.task")
ML_MODEL = os.path.join(parent_dir, "models/stroke_classifier_100percent_best.pth")
SCALER = os.path.join(parent_dir, "models/stroke_classifier_100percent_scaler.pkl")

class MultiClassClassifier(nn.Module):
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

# Load models
print("="*70)
print("MODULE 1 TEST SUITE - Following TEST_GUIDE.md")
print("="*70)
print()

print("Loading models...")
detector = FaceAsymmetryDetector(model_path=MEDIAPIPE_MODEL)
ml_model = MultiClassClassifier()
ml_model.load_state_dict(torch.load(ML_MODEL, weights_only=True, map_location='cpu'))
ml_model.eval()
scaler = joblib.load(SCALER)
print("Models loaded!")
print()

# Test results storage
test_results = {}

def extract_features(landmarks):
    """Extract additional features"""
    try:
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_opening = np.linalg.norm(upper_lip - lower_lip)
        mouth_width = np.linalg.norm(left_corner - right_corner)
        mouth_ar = mouth_opening / (mouth_width + 1e-6)

        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        left_eye_ar = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_ar = np.linalg.norm(right_eye_top - right_eye_bottom)
        eye_ar = (left_eye_ar + right_eye_ar) / 2

        face_center = landmarks[5]
        deviation = face_center[0] - 0.5
        rotation = deviation * 100

        left_dist = np.linalg.norm(left_corner - face_center)
        right_dist = np.linalg.norm(right_corner - face_center)
        smile = (left_dist + right_dist) / 2

        return {
            'mouth_ar': mouth_ar,
            'eye_ar': eye_ar,
            'rotation': rotation,
            'smile': smile
        }
    except:
        return {'mouth_ar': 0, 'eye_ar': 0, 'rotation': 0, 'smile': 0}

def process_and_display(frame_num, test_name, duration_seconds=5):
    """Process frame and display results"""
    global_timestamp = 0

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"[ERROR] {test_name}: Cannot open camera")
        return None

    print(f"\n{'='*70}")
    print(f"TEST CASE: {test_name}")
    print(f"{'='*70}")
    print(f"Press ENTER when ready to start...")
    input()

    results = []
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed >= duration_seconds:
            break

        frame_count += 1

        try:
            result = detector.process_frame(frame, frame_timestamp_ms=global_timestamp)
            global_timestamp += 33
        except:
            continue

        if result.get('status') in ['NO_FACE', 'NO_DETECTOR']:
            continue

        landmarks = result.get('raw_landmarks')
        if landmarks is None:
            continue

        features = extract_features(landmarks)

        # ML prediction
        basic_features = landmarks.flatten()
        features_array = np.concatenate([
            basic_features,
            [features['mouth_ar'], features['eye_ar'], features['rotation'], features['smile']]
        ])

        features_scaled = scaler.transform(features_array.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled)

        with torch.no_grad():
            outputs = ml_model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            stroke_prob = probs[0][0].item() * 100

        # Determine status
        if stroke_prob < 30:
            status = "NORMAL"
        elif stroke_prob < 60:
            status = "WARNING"
        else:
            status = "DANGER"

        # Indicators
        indicators = {
            'yawn': features['mouth_ar'] > 0.5,
            'drowsy': features['eye_ar'] < 0.05,
            'head_turn': abs(features['rotation']) > 15.0,
            'smile': features['smile'] > 0.3
        }

        results.append({
            'stroke_prob': stroke_prob,
            'status': status,
            'indicators': indicators
        })

        # Display
        h, w = frame.shape[:2]
        new_width = 1280
        new_height = int(new_width * h / w)
        frame = cv2.resize(frame, (new_width, new_height))

        # Draw status box
        color = (0, 255, 0) if status == "NORMAL" else (0, 165, 255) if status == "WARNING" else (0, 0, 255)
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 120), color, 2)
        cv2.putText(frame, f"{test_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Stroke Prob: {stroke_prob:.1f}%", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Indicators
        y = 115
        for ind_name, ind_value in indicators.items():
            if ind_value:
                cv2.putText(frame, f"[{ind_name.upper()}]", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y += 20

        cv2.imshow('Module 1 Test Suite', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if results:
        avg_prob = np.mean([r['stroke_prob'] for r in results])
        final_status = results[-1]['status']

        print(f"\nRESULTS: {test_name}")
        print(f"   Frames captured: {len(results)}")
        print(f"   Average Stroke Prob: {avg_prob:.2f}%")
        print(f"   Final Status: {final_status}")

        # Check indicators
        indicators_count = {}
        for r in results:
            for ind_name, id in r['indicators'].items():
                if id:
                    indicators_count[ind_name] = indicators_count.get(ind_name, 0) + 1

        if indicators_count:
            print(f"   Indicators detected:")
            for ind_name, count in indicators_count.items():
                if count > 0:
                    print(f"     - {ind_name.upper()}: {count}/{len(results)} frames")

        return {
            'avg_prob': avg_prob,
            'final_status': final_status,
            'frames': len(results),
            'indicators': indicators_count
        }

    return None

def analyze_result(result, test_name):
    """Analyze if test passed or failed"""
    passed = False
    diagnosis = ""

    if test_name == "Test Case 1: NORMAL (Baseline)":
        passed = (result['final_status'] == 'NORMAL' and
                 result['avg_prob'] < 30 and
                 sum(result['indicators'].values()) == 0)
        diagnosis = "PASS" if passed else "FAIL"

    elif test_name == "Test Case 2: MIMIC STROKE":
        passed = (result['final_status'] in ['WARNING', 'DANGER'] and
                 result['avg_prob'] > 30)
        diagnosis = "PASS" if passed else "FAIL (Model khong detect đột quỵ)"

    elif test_name == "Test Case 3: YAWN (False Positive Check)":
        has_yawn = result['indicators'].get('yawn', 0) > 0
        is_normal = result['final_status'] == 'NORMAL'
        passed = (has_yawn and is_normal)
        diagnosis = "PASS" if passed else "FAIL (Expected YAWN indicator + NORMAL status)"

    elif test_name == "Test Case 4: SMILE (False Positive Check)":
        has_smile = result['indicators'].get('smile', 0) > 0
        is_normal = result['final_status'] == 'NORMAL'
        passed = (has_smile and is_normal)
        diagnosis = "PASS" if passed else "FAIL (Expected SMILE indicator + NORMAL status)"

    elif test_name == "Test Case 5: HEAD TURN (False Positive Check)":
        has_turn = result['indicators'].get('head_turn', 0) > 0
        is_normal = result['final_status'] == 'NORMAL'
        passed = (has_turn and is_normal)
        diagnosis = "PASS" if passed else "FAIL (Expected HEAD TURN indicator + NORMAL status)"

    elif test_name == "Test Case 6: COMBINATION":
        # For combination, at least 2 indicators and normal status
        has_multiple = sum(1 for v in result['indicators'].values() if v) >= 2
        is_normal = result['final_status'] == 'NORMAL'
        passed = (has_multiple and is_normal)
        diagnosis = "PASS" if passed else "FAIL (Expected multiple indicators + NORMAL status)"

    print(f"   {diagnosis}")
    return passed

# ============================================================================
# RUN ALL TESTS
# ============================================================================

print("TEST SUITE OVERVIEW:")
print("   Test 1: NORMAL (Baseline)")
print("   Test 2: MIMIC STROKE")
print("   Test 3: YAWN (False Positive Check)")
print("   Test 4: SMILE (False Positive Check)")
print("   Test 5: HEAD TURN (False Positive Check)")
print("   Test 6: COMBINATION")
print()

all_passed = []
test_names = [
    "Test Case 1: NORMAL (Baseline)",
    "Test Case 2: MIMIC STROKE",
    "Test Case 3: YAWN (False Positive Check)",
    "Test Case 4: SMILE (False Positive Check)",
    "Test Case 5: HEAD TURN (False Positive Check)",
    "Test Case 6: COMBINATION"
]

for idx, test_name in enumerate(test_names):
    print(f"\n{'='*70}")
    print(f"TEST CASE {idx+1}/6: {test_name}")
    print(f"{'='*70}")

    result = process_and_display(idx + 1, test_name, duration_seconds=5)

    if result:
        passed = analyze_result(result, test_name)
        all_passed.append(passed)

# ============================================================================
# FINAL SCORING
# ============================================================================

print(f"\n{'='*70}")
print(f"FINAL SCORING")
print(f"{'='*70}")
print()

score = sum(all_passed)
total = len(all_passed)
percentage = (score / total * 100) if total > 0 else 0

print(f"   Total Tests: {total}")
print(f"   Passed: {score}")
print(f"   Failed: {total - score}")
print(f"   Score: {percentage:.1f}%")
print()

if percentage >= 100:
    print("   EXCELLENT! (>90% accuracy)")
elif percentage >= 80:
    print("   GOOD! (80-90% accuracy)")
elif percentage >= 70:
    print("   ACCEPTABLE (70-80% accuracy)")
else:
    print("   NEED IMPROVEMENT")

print(f"\n{'='*70}")
print(f"TEST SUITE COMPLETE")
print(f"{'='*70}")
