"""
MODULE 5: VISUAL FIELD / EYE MOVEMENT DETECTION - PSCS v8.1
Phát hiện khiếm thị trường và bất thường vận chuyển mắt do đột quỵ

Tác giả: PSCS Team
Ngày: 31/08/2026 (Updated with ML model support)
"""

import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import joblib

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[VISUAL] OpenCV not available. Install: pip install opencv-python")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[VISUAL] MediaPipe not available. Install: pip install mediapipe")


class VisualFieldDetector:
    """Detector phát hiện visual field deficit và eye movement abnormality"""

    def __init__(self, ml_model_path=None, scaler_path=None):
        """Khởi tạo Visual Field Detector với ML Model support"""
        # Initialize MediaPipe Face Mesh
        self.face_mesh = None
        self.detector_type = None

        if MEDIAPIPE_AVAILABLE:
            try:
                mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector_type = 'face_mesh'
                print("[VISUAL] Using MediaPipe Face Mesh (solutions API)")
            except Exception as e:
                print(f"[VISUAL] Face Mesh init error: {e}")
                self.detector_type = 'fallback'
        else:
            self.detector_type = 'fallback'

        # Initialize ML model attributes
        self.ml_model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load ML model if paths provided
        if ml_model_path and scaler_path:
            self._load_ml_model(ml_model_path, scaler_path)

        # Eye movement history
        self.left_gaze_history = []
        self.right_gaze_history = []
        self.history_maxlen = 30

        # Thresholds
        self.thresholds = {
            'gaze_asymmetry_max': 15,
            'eye_openness_min': 0.3,
            'face_symmetry_max': 20,
            'ml_threshold': 50  # ML model threshold
        }

    def _load_ml_model(self, model_path, scaler_path):
        """Load ML model cho visual field detection"""
        try:
            # Define model architecture (must match training)
            class VisualFieldClassifier(nn.Module):
                def __init__(self):
                    super(VisualFieldClassifier, self).__init__()
                    self.network = nn.Sequential(
                        nn.Linear(20, 40),
                        nn.BatchNorm1d(40),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(40, 80),
                        nn.BatchNorm1d(80),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(80, 40),
                        nn.BatchNorm1d(40),
                        nn.ReLU(),
                        nn.Linear(40, 2)
                    )

                def forward(self, x):
                    return self.network(x)

            self.ml_model = VisualFieldClassifier()
            self.ml_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
            self.ml_model.to(self.device)
            self.ml_model.eval()
            print(f"[VISUAL] ML model loaded from: {model_path}")
        except Exception as e:
            print(f"[VISUAL] Error loading ML model: {e}")

        try:
            self.scaler = joblib.load(scaler_path)
            print(f"[VISUAL] Scaler loaded from: {scaler_path}")
        except Exception as e:
            print(f"[VISUAL] Error loading scaler: {e}")

    def extract_visual_features(self, landmarks):
        """
        Extract features for ML model (20 features)

        Features:
        ┌─────────────────────────────────────────────────────────┐
        │  1-12: Eye gaze points (x, y at 3 time points per eye)    │
        │  13-20: Visual field test results (8 positions)          │
        └─────────────────────────────────────────────────────────┘
        """
        if landmarks is None:
            return np.zeros(20)

        features = []

        # Eye landmark indices
        LEFT_EYE_INDICES = [33, 160]  # Left and right eye corners
        RIGHT_EYE_INDICES = [263, 385]

        # Get eye centers (simplified)
        left_eye_center = np.array([landmarks[33].x, landmarks[33].y])
        right_eye_center = np.array([landmarks[263].x, landmarks[263].y])

        # Generate 3 time points of gaze data (simulated)
        for t in range(3):
            # Add small variation for natural movement
            noise_x = np.random.normal(0, 0.01)
            noise_y = np.random.normal(0, 0.01)

            left_gaze = left_eye_center + np.array([noise_x, noise_y])
            right_gaze = right_eye_center + np.array([noise_x, noise_y])

            features.extend([left_gaze[0], left_gaze[1]])
            features.extend([right_gaze[0], right_gaze[1]])

        # Calculate eye openness (EAR)
        LEFT_EYE = [33, 160, 158, 153, 144, 163]
        RIGHT_EYE = [362, 385, 387, 380, 373, 393]

        left_eye_pts = np.array([[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE if i < len(landmarks)])
        right_eye_pts = np.array([[landmarks[i].x, landmarks[i].y] for i in RIGHT_EYE if i < len(landmarks)])

        left_ear = self._eye_aspect_ratio(left_eye_pts)
        right_ear = self._eye_aspect_ratio(right_eye_pts)

        # Simulate visual field test results (8 stimuli positions)
        # In real system, this would be from actual visual field testing
        # For now, use rule-based estimation from eye openness

        if left_ear < 0.2 or right_ear < 0.2:
            # One eye closed - severe defect
            visual_field_results = [1, 0, 1, 1, 1, 1, 0, 0]  # Example pattern
        elif abs(left_ear - right_ear) > 0.15:
            # Significant asymmetry - possible hemianopia
            visual_field_results = [1, 0, 1, 1, 1, 1, 0, 1]
        else:
            # Normal - can see most stimuli
            n_seen = np.random.choice([6, 7, 8], p=[0.1, 0.3, 0.6])
            visual_field_results = [1] * n_seen + [0] * (8 - n_seen)
            np.random.shuffle(visual_field_results)

        features.extend(visual_field_results)

        return np.array(features)

    def detect_visual_field_abnormality(self, frame):
        """
        Phát hiện visual field deficit từ frame (with ML model support)

        Args:
            frame: numpy array (H, W, 3) - BGR or RGB image

        Returns:
            result: dict với visual analysis results
        """
        result = {
            'status': 'NO_FACE',
            'visual_prob': 0.0,
            'nihss_gaze_score': 0,
            'nihss_visual_score': 0,
            'metrics': {}
        }

        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            result['status'] = 'NO_DETECTOR'
            return result

        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else frame
        else:
            frame_rgb = frame

        try:
            # Process with MediaPipe Face Mesh
            mp_results = self.face_mesh.process(frame_rgb)

            if mp_results.multi_face_landmarks is None or len(mp_results.multi_face_landmarks) == 0:
                return result

            landmarks = mp_results.multi_face_landmarks[0].landmark

            # Extract features for ML
            features = self.extract_visual_features(landmarks)

            # Calculate metrics
            metrics = self._calculate_metrics(landmarks)
            result['metrics'] = metrics

            # ML classification if model available
            if self.ml_model is not None and self.scaler is not None:
                try:
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
                    features_tensor = torch.FloatTensor(features_scaled).to(self.device)

                    with torch.no_grad():
                        self.ml_model.eval()
                        outputs = self.ml_model.network(features_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        visual_prob = probs[0][1].item() * 100  # Probability of defect
                    result['visual_prob'] = visual_prob
                except Exception as e:
                    print(f"[VISUAL] ML prediction error: {e}")
                    result['visual_prob'] = self._rule_based_score(metrics)
            else:
                result['visual_prob'] = self._rule_based_score(metrics)

            # Determine status
            if result['visual_prob'] < 50:
                result['status'] = 'NORMAL'
            elif result['visual_prob'] < 75:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'DANGER'

            # Map to NIHSS
            result['nihss_gaze_score'] = self._map_to_nihss_gaze(result['visual_prob'])
            result['nihss_visual_score'] = self._map_to_nihss_visual(result['visual_prob'])

        except Exception as e:
            print(f"[VISUAL] Detection error: {e}")
            result['status'] = 'ERROR'

        return result

    def _calculate_metrics(self, landmarks):
        """Calculate visual metrics from landmarks"""
        # Eye landmark indices
        LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Get eye landmarks
        left_eye_pts = np.array([[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE if i < len(landmarks)])
        right_eye_pts = np.array([[landmarks[i].x, landmarks[i].y] for i in RIGHT_EYE if i < len(landmarks)])

        # Calculate eye openness
        left_ear = self._eye_aspect_ratio(left_eye_pts)
        right_ear = self._eye_aspect_ratio(right_eye_pts)

        # Get eye centers
        left_center = np.mean(left_eye_pts, axis=0)
        right_center = np.mean(right_eye_pts, axis=0)

        return {
            'left_eye_openness': left_ear,
            'right_eye_openness': right_ear,
            'left_gaze_x': 0.0,
            'left_gaze_y': 0.0,
            'right_gaze_x': 0.0,
            'right_gaze_y': 0.0,
            'gaze_asymmetry': abs(left_center[0] - right_center[0]) * 100,
            'pupil_asymmetry': 0.0,
            'blink_detected': left_ear < 0.2 or right_ear < 0.2
        }

    def _eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR) for openness"""
        if len(eye_points) < 6:
            return 0.5

        # Vertical distances
        p1 = eye_points[1]
        p2 = eye_points[5]
        v1 = np.linalg.norm(p1 - p2)

        # Horizontal distance
        p3 = eye_points[0]
        p4 = eye_points[3]
        h1 = np.linalg.norm(p3 - p4)

        if h1 == 0:
            return 0.0

        return v1 / h1

    def _rule_based_score(self, metrics):
        """Tính Visual Field Abnormality Score using rules (fallback)"""
        score = 0.0

        # 1. High gaze asymmetry (40 points)
        if metrics.get('gaze_asymmetry', 0) > self.thresholds['gaze_asymmetry_max']:
            score += 40
        elif metrics.get('gaze_asymmetry', 0) > self.thresholds['gaze_asymmetry_max'] * 0.7:
            score += 20

        # 2. One eye closed (30 points)
        if metrics.get('left_eye_openness', 0.5) < self.thresholds['eye_openness_min']:
            score += 30
        elif metrics.get('right_eye_openness', 0.5) < self.thresholds['eye_openness_min']:
            score += 30
        elif metrics.get('left_eye_openness', 0.5) < 0.5:
            score += 15

        # 3. Blink detected (10 points)
        if metrics.get('blink_detected', False):
            score += 10

        return min(score, 100.0)

    def _map_to_nihss_gaze(self, visual_score):
        """Mapping visual score sang NIHSS Item 4 (Best Gaze)"""
        if visual_score < 50:
            return 0
        elif visual_score < 75:
            return 1
        else:
            return 2

    def _map_to_nihss_visual(self, visual_score):
        """Mapping visual score sang NIHSS Item 3 (Visual)"""
        if visual_score < 50:
            return 0
        elif visual_score < 70:
            return 1
        elif visual_score < 85:
            return 2
        else:
            return 3


# Test function
def test_visual_module():
    """Test Visual Module với ML model"""
    print("="*70)
    print("MODULE 5 VISUAL FIELD TEST (with ML Model)")
    print("="*70)
    print()

    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))

    # Find latest ML model
    import glob
    ml_models = glob.glob(os.path.join(project_dir, "models/visual_field_*.pth"))
    if ml_models:
        latest_model = sorted(ml_models)[-1]
        scaler_path = latest_model.replace('.pth', '_scaler.pkl')
        print(f"[INFO] Using ML model: {os.path.basename(latest_model)}")
    else:
        latest_model = None
        scaler_path = None
        print(f"[INFO] No ML model found, using rule-based")

    # Initialize detector
    detector = VisualFieldDetector(ml_model_path=latest_model, scaler_path=scaler_path)

    if not CV2_AVAILABLE:
        print("[ERROR] OpenCV not available")
        return

    print("[OK] Visual Field Detector initialized")
    print()
    print("Instructions:")
    print("  - Look at the camera")
    print("  - Follow the visual prompts")
    print()
    print("Press ENTER to start...")
    input()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[OK] Camera opened")
    print("Testing for 10 seconds...")
    print()

    import time
    start_time = time.time()

    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect_visual_field_abnormality(frame)

        if result['status'] != 'NO_FACE':
            print(f"Status: {result['status']}, Visual Prob: {result['visual_prob']:.1f}%, "
                  f"NIHSS Gaze: {result['nihss_gaze_score']}, Visual: {result['nihss_visual_score']}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Test complete!")


if __name__ == "__main__":
    test_visual_module()
