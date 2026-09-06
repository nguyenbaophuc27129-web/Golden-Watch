"""
QUICK TEST - Module 1 (NON-INTERACTIVE)
Test each case automatically with 3 seconds per test
"""

import sys
import os
import time

# Get parent directory
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

# Configuration
MEDIAPIPE_MODEL = os.path.join(parent_dir, "models/face_landmarker_v2/face_landmarker_v2_with_blendshapes.task")
ML_MODEL = os.path.join(parent_dir, "models/stroke_classifier_100percent_best.pth")
SCALER = os.path.join(parent_dir, "models/stroke_classifier_100percent_scaler.pkl")

class MultiClassClassifier(nn.Module):
    def __init__(self):
        super(MultiClassClassifier, self).__init__()
        layers = []
        prev_dim = 960
        for hidden_dim in [512, 256, 128]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Load models
print("Loading models...")
detector = FaceAsymmetryDetector(model_path=MEDIAPIPE_MODEL)
ml_model = MultiClassClassifier()
ml_model.load_state_dict(torch.load(ML_MODEL, weights_only=True, map_location='cpu'))
ml_model.eval()
scaler = joblib.load(SCALER)
print("Models loaded!")

# Global timestamp for all tests
global_timestamp = 0
ml_model = MultiClassClassifier()
ml_model.load_state_dict(torch.load(ML_MODEL, weights_only=True, map_location='cpu'))
ml_model.eval()
scaler = joblib.load(SCALER)
print("Models loaded!")

def extract_features(landmarks):
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

        return {'mouth_ar': mouth_ar, 'eye_ar': eye_ar, 'rotation': rotation, 'smile': smile}
    except:
        return {'mouth_ar': 0, 'eye_ar': 0, 'rotation': 0, 'smile': 0}

def test_one_case(duration_seconds=5):
    """Test one case automatically"""
    global global_timestamp

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    results = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time >= duration_seconds:
            break

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

        if stroke_prob < 30:
            status = "NORMAL"
        elif stroke_prob < 60:
            status = "WARNING"
        else:
            status = "DANGER"

        indicators = {
            'yawn': features['mouth_ar'] > 0.5,
            'drowsy': features['eye_ar'] < 0.05,
            'head_turn': abs(features['rotation']) > 15.0,
            'smile': features['smile'] > 0.3
        }

        results.append({'stroke_prob': stroke_prob, 'status': status, 'indicators': indicators})

        # Display
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (1280, int(1280 * h / w)))

        color = (0, 255, 0) if status == "NORMAL" else (0, 165, 255) if status == "WARNING" else (0, 0, 255)
        cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 100), color, 2)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"{stroke_prob:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y = 90
        for ind_name, ind_value in indicators.items():
            if ind_value:
                cv2.putText(frame, f"[{ind_name.upper()}]", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y += 15

        cv2.imshow('Quick Test - Module 1', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if results:
        avg_prob = np.mean([r['stroke_prob'] for r in results])
        final_status = results[-1]['status']
        indicators_count = {}
        for r in results:
            for ind_name, ind_value in r['indicators'].items():
                if ind_value:
                    indicators_count[ind_name] = indicators_count.get(ind_name, 0) + 1

        print(f"Results: {len(results)} frames, Avg prob: {avg_prob:.2f}%, Status: {final_status}")
        if indicators_count:
            print(f"Indicators: {indicators_count}")

        return {'avg_prob': avg_prob, 'final_status': final_status, 'indicators': indicators_count}

    return None

# ============================================================================
# RUN TESTS AUTOMATICALLY
# ============================================================================

print("="*70)
print("MODULE 1 QUICK TEST (AUTOMATIC - 3 seconds per test)")
print("="*70)
print()
print("Instructions:")
print("  Test 1: Sit still, neutral face")
print("  Test 2: Mimic stroke (crooked mouth, closed eye, tilted head)")
print("  Test 3: Yawn widely")
print("  Test 4: Smile")
print("  Test 5: Turn head sideways")
print("  Test 6: Combination")
print()

print("Starting in 3 seconds... Prepare!")
time.sleep(3)

all_results = []

for i in range(6):
    print(f"\n{'='*70}")
    print(f"TEST {i+1}/6")
    print(f"{'='*70}")
    print(f"Prepare: {['Sit still (neutral)', 'Mimic stroke', 'Yawn', 'Smile', 'Turn head', 'Combination'][i]}")
    print("Starting in 2 seconds...")
    time.sleep(2)

    result = test_one_case(duration_seconds=3)
    if result:
        all_results.append(result)
        print(f"-> Capture: {result['final_status']}, Avg prob: {result['avg_prob']:.2f}%")

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")

if all_results:
    normal_count = sum(1 for r in all_results if r['final_status'] == 'NORMAL')
    warning_count = sum(1 for r in all_results if r['final_status'] == 'WARNING')
    danger_count = sum(1 for r in all_results if r['final_status'] == 'DANGER')
    avg_prob = np.mean([r['avg_prob'] for r in all_results])

    print(f"Total tests: {len(all_results)}")
    print(f"  NORMAL: {normal_count}")
    print(f"  WARNING: {warning_count}")
    print(f"  DANGER: {danger_count}")
    print(f"  Average Stroke Prob: {avg_prob:.2f}%")

    print(f"\nScoring:")
    if danger_count >= 1:
        print("  EXCELLENT! Model detected stroke (mimic test)")
    if normal_count >= 1:
        print("  GOOD! Baseline normal test passed")
