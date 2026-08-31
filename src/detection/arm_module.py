"""
MODULE 3: ARM WEAKNESS DETECTION - PSCS v8.1
Phát hiện yếu tay do đột quỵ sử dụng YOLOv8n-Pose + ML Model

Tác giả: PSCS Team
Ngày: 31/08/2026 (Updated with ML model support)
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import joblib
from ultralytics import YOLO


class ArmWeaknessDetector:
    """Detector phát hiện yếu tay (Arm Weakness) từ pose estimation"""

    # YOLOv8n-Pose keypoints:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder
    # 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist
    # 11-16: hips, knees, ankles

    def __init__(self, pose_model_path=None, ml_model_path=None, scaler_path=None):
        """
        Khởi tạo Arm Weakness Detector với ML Model support

        Args:
            pose_model_path: Đường dẫn đến YOLOv8n-pose model
            ml_model_path: Đường dẫn đến ML model (.pth)
            scaler_path: Đường dẫn đến scaler (.pkl)
        """
        # Setup paths
        if pose_model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(os.path.dirname(current_dir))
            pose_model_path = os.path.join(project_dir, "models/yolov8n-pose.pt")

        # Load YOLOv8n-pose
        if os.path.exists(pose_model_path):
            self.pose_model = YOLO(pose_model_path)
            print(f"[ARM] YOLOv8n-pose loaded from: {pose_model_path}")
        else:
            print(f"[ARM] Pose model not found: {pose_model_path}")
            self.pose_model = None

        # Initialize ML model attributes
        self.ml_model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load ML model if paths provided
        if ml_model_path and scaler_path:
            self._load_ml_model(ml_model_path, scaler_path)

        # Thresholds
        self.thresholds = {
            'arm_drop_max': 30,
            'movement_range_min': 20,
            'asymmetry_max': 25,
            'speed_ratio_max': 0.5,
            'consistency_min': 0.6,
            'ml_threshold': 50  # ML model threshold
        }

        # History để track movement
        self.left_arm_history = []
        self.right_arm_history = []
        self.movement_frames = 0

    def _load_ml_model(self, model_path, scaler_path):
        """Load ML model cho arm weakness detection"""
        try:
            # Define model architecture (must match training)
            class ArmWeaknessClassifier(nn.Module):
                def __init__(self):
                    super(ArmWeaknessClassifier, self).__init__()
                    self.network = nn.Sequential(
                        nn.Linear(16, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(32, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32, 2)
                    )

                def forward(self, x):
                    return self.network(x)

            self.ml_model = ArmWeaknessClassifier()
            self.ml_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
            self.ml_model.to(self.device)
            self.ml_model.eval()
            print(f"[ARM] ML model loaded from: {model_path}")
        except Exception as e:
            print(f"[ARM] Error loading ML model: {e}")

        try:
            self.scaler = joblib.load(scaler_path)
            print(f"[ARM] Scaler loaded from: {scaler_path}")
        except Exception as e:
            print(f"[ARM] Error loading scaler: {e}")

    def extract_arm_keypoints(self, keypoints):
        """
        Trích xuất arm keypoints từ pose

        Args:
            keypoints: numpy array (17, 3) from YOLOv8n-pose

        Returns:
            dict với shoulder, elbow, wrist coordinates
        """
        if keypoints is None or len(keypoints) < 11:
            return None

        result = {
            'left_shoulder': keypoints[5],
            'right_shoulder': keypoints[6],
            'left_elbow': keypoints[7],
            'right_elbow': keypoints[8],
            'left_wrist': keypoints[9],
            'right_wrist': keypoints[10]
        }

        return result

    def extract_arm_features(self, arm_data):
        """
        Extract features for ML model (16 features)

        Features:
        ┌─────────────────────────────────────────────────────────┐
        │  1-12: Keypoint coordinates (x, y for 6 joints)       │
        │  13-16: Calculated features (drift, span, asymmetry)   │
        └─────────────────────────────────────────────────────────┘
        """
        if arm_data is None:
            return np.zeros(16)

        features = []

        # Keypoint coordinates (12 features)
        features.extend([
            arm_data['left_shoulder'][0], arm_data['left_shoulder'][1],
            arm_data['left_elbow'][0], arm_data['left_elbow'][1],
            arm_data['left_wrist'][0], arm_data['left_wrist'][1],
            arm_data['right_shoulder'][0], arm_data['right_shoulder'][1],
            arm_data['right_elbow'][0], arm_data['right_elbow'][1],
            arm_data['right_wrist'][0], arm_data['right_wrist'][1]
        ])

        # Calculated features (4 features)
        left_shoulder = np.array(arm_data['left_shoulder'][:2])
        right_shoulder = np.array(arm_data['right_shoulder'][:2])
        left_wrist = np.array(arm_data['left_wrist'][:2])
        right_wrist = np.array(arm_data['right_wrist'][:2])

        # Left arm drift rate
        left_drift = left_wrist[1] - left_shoulder[1]

        # Right arm drift rate
        right_drift = right_wrist[1] - right_shoulder[1]

        # Arm span
        arm_span = abs(left_wrist[0] - right_wrist[0])

        # Asymmetry score
        asymmetry = abs(left_drift - right_drift)

        features.extend([left_drift, right_drift, arm_span, asymmetry])

        return np.array(features)

    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """Tính góc cánh tay (shoulder-elbow-wrist)"""
        if shoulder[2] < 0.5 or elbow[2] < 0.5 or wrist[2] < 0.5:
            return None

        v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
        v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def detect_arm_weakness(self, frame):
        """
        Phát hiện yếu tay từ frame (with ML model support)

        Args:
            frame: numpy array (H, W, 3) - RGB image

        Returns:
            result: dict với status, arm_prob, nihss_score, metrics
        """
        result = {
            'status': 'NO_PERSON',
            'arm_prob': 0.0,
            'nihss_score': 0,
            'metrics': {}
        }

        if self.pose_model is None:
            return result

        # YOLOv8n-pose detection
        pose_results = self.pose_model(frame, verbose=False)

        if len(pose_results) == 0 or len(pose_results[0].keypoints) == 0:
            return result

        # Get keypoints
        keypoints = pose_results[0].keypoints.xy[0].cpu().numpy()

        # Extract arm keypoints
        arm_data = self.extract_arm_keypoints(keypoints)

        if arm_data is None:
            result['status'] = 'NO_POSE'
            return result

        # Extract features for ML
        features = self.extract_arm_features(arm_data)

        # Calculate metrics
        metrics = self._calculate_arm_metrics(arm_data)
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
                    arm_prob = probs[0][1].item() * 100  # Probability of weakness
                result['arm_prob'] = arm_prob
            except Exception as e:
                print(f"[ARM] ML prediction error: {e}")
                result['arm_prob'] = self._rule_based_score(metrics)
        else:
            result['arm_prob'] = self._rule_based_score(metrics)

        # Determine status
        if result['arm_prob'] < 50:
            result['status'] = 'NORMAL'
        elif result['arm_prob'] < 75:
            result['status'] = 'WARNING'
        else:
            result['status'] = 'DANGER'

        # Map to NIHSS Item 5 (Motor Arm)
        result['nihss_score'] = self._map_to_nihss(result['arm_prob'])

        return result

    def _calculate_arm_metrics(self, arm_data):
        """Tính tất cả arm metrics"""
        metrics = {}

        left_shoulder = arm_data['left_shoulder']
        right_shoulder = arm_data['right_shoulder']
        left_elbow = arm_data['left_elbow']
        right_elbow = arm_data['right_elbow']
        left_wrist = arm_data['left_wrist']
        right_wrist = arm_data['right_wrist']

        # Arm angles
        left_angle = self.calculate_arm_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self.calculate_arm_angle(right_shoulder, right_elbow, right_wrist)

        metrics['left_arm_angle'] = left_angle if left_angle is not None else 0
        metrics['right_arm_angle'] = right_angle if right_angle is not None else 0

        # Arm height
        left_height = left_wrist[1] - left_shoulder[1] if left_shoulder[2] > 0.5 and left_wrist[2] > 0.5 else 0
        right_height = right_wrist[1] - right_shoulder[1] if right_shoulder[2] > 0.5 and right_wrist[2] > 0.5 else 0

        metrics['left_arm_height'] = left_height
        metrics['right_arm_height'] = right_height

        # Arm drop
        left_drop = max(0, left_height)
        right_drop = max(0, right_height)

        metrics['left_arm_drop'] = left_drop
        metrics['right_arm_drop'] = right_drop

        # Asymmetry
        if left_angle is not None and right_angle is not None:
            metrics['angle_asymmetry'] = abs(left_angle - right_angle)
        else:
            metrics['angle_asymmetry'] = 0

        metrics['height_asymmetry'] = abs(left_height - right_height)

        # Weak arm detection
        metrics['weak_arm'] = self._identify_weak_arm(metrics)

        return metrics

    def _identify_weak_arm(self, metrics):
        """Identify which arm is weak"""
        if metrics['left_arm_drop'] > metrics['right_arm_drop'] + 20:
            return 'left'
        elif metrics['right_arm_drop'] > metrics['left_arm_drop'] + 20:
            return 'right'
        elif metrics['left_arm_angle'] < metrics['right_arm_angle'] - 15:
            return 'left'
        elif metrics['right_arm_angle'] < metrics['left_arm_angle'] - 15:
            return 'right'
        else:
            return 'none'

    def _rule_based_score(self, metrics):
        """Calculate arm weakness score using rules (fallback)"""
        score = 0.0

        weak_arm = metrics['weak_arm']
        if weak_arm == 'left':
            drop = metrics['left_arm_drop']
        elif weak_arm == 'right':
            drop = metrics['right_arm_drop']
        else:
            drop = max(metrics['left_arm_drop'], metrics['right_arm_drop'])

        # Arm drop (40 points)
        if drop > 100:
            score += 40
        elif drop > 50:
            score += 20
        elif drop > 30:
            score += 10

        # Angle asymmetry (30 points)
        if metrics['angle_asymmetry'] > 25:
            score += 30
        elif metrics['angle_asymmetry'] > 15:
            score += 20
        elif metrics['angle_asymmetry'] > 10:
            score += 10

        # Height asymmetry (20 points)
        if metrics['height_asymmetry'] > 80:
            score += 20
        elif metrics['height_asymmetry'] > 50:
            score += 10

        # Extended arm (10 points)
        weak_angle = metrics['left_arm_angle'] if weak_arm == 'left' else metrics['right_arm_angle']
        if weak_arm != 'none':
            if weak_angle < 70:
                score += 10
            elif weak_angle < 80:
                score += 5

        return min(score, 100.0)

    def _map_to_nihss(self, arm_score):
        """Mapping arm score sang NIHSS Item 5 (Motor Arm)"""
        if arm_score < 50:
            return 0  # No drift
        elif arm_score < 65:
            return 1  # Mild drift
        elif arm_score < 80:
            return 2  # Some effort against gravity
        else:
            return 3  # No effort against gravity


# Test function
def test_arm_module():
    """Test Arm Module với ML model"""
    print("="*70)
    print("MODULE 3 ARM WEAKNESS TEST (with ML Model)")
    print("="*70)
    print()

    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))

    # Find latest ML model
    import glob
    ml_models = glob.glob(os.path.join(project_dir, "models/arm_weakness_*.pth"))
    if ml_models:
        latest_model = sorted(ml_models)[-1]
        scaler_path = latest_model.replace('.pth', '_scaler.pkl')
        print(f"[INFO] Using ML model: {os.path.basename(latest_model)}")
    else:
        latest_model = None
        scaler_path = None
        print(f"[INFO] No ML model found, using rule-based")

    # Initialize detector
    detector = ArmWeaknessDetector(ml_model_path=latest_model, scaler_path=scaler_path)

    if detector.pose_model is None:
        print("[ERROR] Cannot load YOLO model")
        return

    print("[OK] YOLOv8n-pose loaded")
    print()

    # Test with existing video/image if available
    test_images = glob.glob(os.path.join(project_dir, "data/datasets/pose/*.*"))

    if test_images:
        print(f"[INFO] Found {len(test_images)} test images")
        for img_path in test_images[:3]:
            print(f"\nTesting: {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = detector.detect_arm_weakness(img_rgb)
                print(f"  Status: {result['status']}")
                print(f"  Arm Weakness: {result['arm_prob']:.2f}%")
                print(f"  NIHSS Item 5: {result['nihss_score']}/4")
    else:
        print("\nNo test images found. Use webcam test instead.")
        print("Instructions:")
        print("  - Raise both arms forward")
        print("  - Hold for 5 seconds")
        print("  - Press ENTER to start...")
        input()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return

        import time
        start_time = time.time()

        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_arm_weakness(frame_rgb)

            if result['status'] != 'NO_PERSON':
                print(f"Status: {result['status']}, Arm Prob: {result['arm_prob']:.1f}%, NIHSS: {result['nihss_score']}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    print()
    print("Test complete!")


if __name__ == "__main__":
    test_arm_module()
