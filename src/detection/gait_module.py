"""
MODULE 4: GAIT ABNORMALITY DETECTION - PSCS v8.0
Phát hiện bất thường dáng đi do đột quỵ sử dụng time series analysis

Tác giả: PSCS Team
Ngày: 29/08/2026
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import glob


class GaitAbnormalityDetector:
    """Detector phát hiện gait abnormality từ time series"""

    def __init__(self, model_path=None, scaler_path=None):
        """
        Khởi tạo Gait Abnormality Detector

        Args:
            model_path: Đường dẫn đến ML model (.pth)
            scaler_path: Đường dẫn đến scaler (.pkl)
        """
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model nếu có
        if model_path and os.path.exists(model_path):
            self._load_ml_model(model_path, scaler_path)

        # Thresholds cho rule-based detection
        self.thresholds = {
            'stride_length_min': 0.5,      # meters - stride length
            'stride_length_max': 0.8,      # meters
            'cadence_min': 100,             # steps/min
            'cadence_max': 130,             # steps/min
            'stride_time_var_max': 0.05,   # variance in stride time
            'asymmetry_max': 0.15,          # left-right asymmetry
            'velocity_var_max': 0.1         # variance in velocity
        }

    def _load_ml_model(self, model_path, scaler_path):
        """Load ML model cho gait abnormality detection"""
        try:
            # Create wrapper class matching training script structure
            class GaitClassifier(nn.Module):
                def __init__(self):
                    super(GaitClassifier, self).__init__()
                    self.network = nn.Sequential(
                        nn.Linear(8, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(32, 2)
                    )

                def forward(self, x):
                    return self.network(x)

            self.model = GaitClassifier()
            self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"[GAIT] ML model loaded from: {model_path}")
        except Exception as e:
            print(f"[GAIT] Error loading ML model: {e}")

        try:
            self.scaler = joblib.load(scaler_path)
            print(f"[GAIT] Scaler loaded from: {scaler_path}")
        except Exception as e:
            print(f"[GAIT] Error loading scaler: {e}")

    def load_gait_data(self, file_path):
        """
        Load gait time series data từ file

        Args:
            file_path: Đường dẫn đến file .txt

        Returns:
            data: numpy array (N, 2) - [time, measurement]
        """
        try:
            data = np.loadtxt(file_path)
            if len(data.shape) == 1:
                data = np.reshape(data, (-1, 2))
            return data
        except Exception as e:
            print(f"[GAIT] Error loading {file_path}: {e}")
            return None

    def extract_gait_features(self, time_series):
        """
        Extract features từ gait time series

        Args:
            time_series: numpy array (N, 2) - [time, measurement]

        Returns:
            features: numpy array (8,) - extracted features
        """
        if time_series is None or len(time_series) < 10:
            return np.zeros(8)

        try:
            # time_series format: [time, value] where value is foot sensor measurement
            time = time_series[:, 0]
            value = time_series[:, 1]

            # 1. Stride length (from value changes - foot sensor measurements)
            # Value represents vertical displacement of foot during gait
            if len(value) > 1:
                stride_length = np.mean(np.abs(np.diff(value))) if len(value) > 1 else 0
                # Normalize from sensor units to meters (assuming cm scale)
                stride_length = stride_length / 100.0  # convert to meters
            else:
                stride_length = 0

            # 2. Cadence (steps per minute) - from sampling frequency
            if len(time) > 1:
                time_diff = np.mean(np.diff(time))
                if time_diff > 0:
                    cadence = 60.0 / time_diff  # steps per minute
                else:
                    cadence = 120  # default
            else:
                cadence = 120

            # 3. Stride time variability
            if len(time) > 1:
                stride_time_var = np.var(np.diff(time)) / np.mean(np.diff(time)) if np.mean(np.diff(time)) > 0 else 0
            else:
                stride_time_var = 0

            # 4. Signal magnitude variability
            magnitude_var = np.var(value) / (np.mean(np.abs(value)) + 1e-6)

            # 5. Velocity (stride length / time)
            if len(time) > 1 and len(value) > 1:
                velocity = stride_length / np.mean(np.diff(time)) if np.mean(np.diff(time)) > 0 else 0
            else:
                velocity = 0

            # 6. Acceleration (change in velocity over time)
            accel = 0
            if len(value) > 2 and len(time) > 2:
                velocities = []
                for i in range(1, len(value)):
                    dv = abs(value[i] - value[i-1])
                    dt = time[i] - time[i-1]
                    if dt > 0:
                        v = dv / dt
                        velocities.append(v)
                if len(velocities) > 1:
                    accel = np.std(velocities)  # Use variability as acceleration proxy

            # 7. Regularity (inverse of variance)
            regularity = 1.0 / (1.0 + magnitude_var)

            # 8. Symmetry (proxy - assume left-right balance from signal)
            symmetry = 1.0 - min(magnitude_var * 0.1, 1.0)

            features = np.array([
                stride_length,  # Normalized stride length
                cadence / 100.0,  # Normalized cadence
                stride_time_var,  # Stride time variability
                magnitude_var,  # Signal variability
                velocity / 10.0,  # Normalized velocity
                accel / 100.0,  # Normalized acceleration
                regularity,  # Regularity score
                symmetry  # Symmetry score
            ])

            return features

        except Exception as e:
            print(f"[GAIT] Error extracting features: {e}")
            return np.zeros(8)

    def detect_gait_abnormality(self, time_series):
        """
        Phát hiện gait abnormality từ time series

        Args:
            time_series: numpy array (N, 2)

        Returns:
            result: dict với status, probability, NIHSS score
        """
        result = {
            'status': 'NO_DATA',
            'gait_prob': 0.0,
            'nihss_score': 0,
            'metrics': {}
        }

        if time_series is None or len(time_series) < 10:
            return result

        # Extract features
        features = self.extract_gait_features(time_series)

        # Calculate metrics
        metrics = {
            'stride_length': features[0],
            'cadence': features[1] * 100,  # Convert back
            'stride_time_var': features[2],
            'magnitude_var': features[3],
            'velocity': features[4] * 10,
            'acceleration': features[5] * 100,
            'regularity': features[6],
            'symmetry': features[7]
        }
        result['metrics'] = metrics

        # ML prediction nếu có model
        if self.model is not None and self.scaler is not None:
            try:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)

                with torch.no_grad():
                    self.model.eval()  # Ensure eval mode
                    outputs = self.model.network(features_tensor)  # Use .network
                    probs = torch.softmax(outputs, dim=1)
                    gait_prob = probs[0][1].item() * 100  # Probability of abnormal
                result['gait_prob'] = gait_prob
            except Exception as e:
                print(f"[GAIT] ML prediction error: {e}")
                result['gait_prob'] = self._rule_based_score(metrics)
        else:
            result['gait_prob'] = self._rule_based_score(metrics)

        # Determine status (ROC optimized threshold: 64%)
        if result['gait_prob'] < 64:
            result['status'] = 'NORMAL'
        elif result['gait_prob'] < 80:
            result['status'] = 'WARNING'
        else:
            result['status'] = 'DANGER'

        # Map to NIHSS Item 6 (Motor Leg)
        result['nihss_score'] = self._map_to_nihss(result['gait_prob'])

        return result

    def _rule_based_score(self, metrics):
        """Calculate gait abnormality score using rules"""
        score = 0.0

        # 1. Stride length abnormal (25 points)
        if metrics['stride_length'] < self.thresholds['stride_length_min']:
            score += 25
        elif metrics['stride_length'] > self.thresholds['stride_length_max']:
            score += 25

        # 2. Cadence abnormal (25 points)
        if metrics['cadence'] < self.thresholds['cadence_min']:
            score += 25
        elif metrics['cadence'] > self.thresholds['cadence_max']:
            score += 25

        # 3. High variability (20 points)
        if metrics['stride_time_var'] > self.thresholds['stride_time_var_max']:
            score += 20

        # 4. Low regularity (20 points)
        if metrics['regularity'] < 0.5:
            score += 20

        # 5. High asymmetry (10 points)
        if metrics['symmetry'] < 1 - self.thresholds['asymmetry_max']:
            score += 10

        return min(score, 100.0)

    def _map_to_nihss(self, gait_score):
        """
        Mapping gait score sang NIHSS Item 6 (Motor Leg)

        Args:
            gait_score: Gait abnormality score (0-100)

        Returns:
            nihss_score: NIHSS score (0-4)
        """
        if gait_score < 64:
            return 0  # No drift (ROC optimized threshold)
        elif gait_score < 75:
            return 1  # Mild drift
        elif gait_score < 85:
            return 2  # Some effort against gravity
        else:
            return 3  # No movement against gravity


# Real-time gait detection using YOLOv8n-Pose
class GaitPoseDetector:
    """Real-time gait detection từ camera pose estimation"""

    def __init__(self, model_path=None):
        """Initialize pose detector for gait analysis"""
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(project_dir, "models/yolov8n-pose.pt")

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[GAIT] YOLOv8n-pose loaded for real-time gait detection")
        except Exception as e:
            print(f"[GAIT] Error loading YOLO: {e}")
            self.model = None

        # Gait analysis parameters
        self.hip_history = []  # Store hip positions over time
        self.knee_history = []  # Store knee positions
        self.ankle_history = []  # Store ankle positions
        self.history_maxlen = 30  # Store last 30 frames
        self.gait_cycles = 0  # Count gait cycles

    def detect_gait_from_frame(self, frame):
        """
        Detect gait abnormality from camera frame using pose estimation

        Args:
            frame: numpy array (H, W, 3) - RGB image

        Returns:
            result: dict với gait analysis results
        """
        result = {
            'status': 'NO_PERSON',
            'gait_prob': 0.0,
            'nihss_score': 0,
            'metrics': {}
        }

        if self.model is None:
            return result

        try:
            # YOLOv8n-pose detection
            results = self.model(frame, verbose=False)

            if len(results) == 0 or len(results[0].keypoints) == 0:
                return result

            # Get keypoints (use .xy for pose keypoints, not .xyxy which is for bounding boxes)
            keypoints = results[0].keypoints.xy[0].cpu().numpy()

            # Extract lower body keypoints for gait
            # 11: left_hip, 12: right_hip
            # 13: left_knee, 14: right_knee
            # 15: left_ankle, 16: right_ankle
            if keypoints.shape[0] >= 17:
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                left_knee = keypoints[13]
                right_knee = keypoints[14]
                left_ankle = keypoints[15]
                right_ankle = keypoints[16]

                # Store in history
                self.hip_history.append((left_hip, right_hip))
                self.knee_history.append((left_knee, right_knee))
                self.ankle_history.append((left_ankle, right_ankle))

                # Keep only recent history
                if len(self.hip_history) > self.history_maxlen:
                    self.hip_history.pop(0)
                    self.knee_history.pop(0)
                    self.ankle_history.pop(0)

                # Calculate gait metrics
                metrics = self._calculate_gait_metrics()
                result['metrics'] = metrics

                # Score gait abnormality
                result['gait_prob'] = self._calculate_gait_score(metrics)

                # Determine status
                if result['gait_prob'] < 30:
                    result['status'] = 'NORMAL'
                elif result['gait_prob'] < 60:
                    result['status'] = 'WARNING'
                else:
                    result['status'] = 'DANGER'

                # NIHSS mapping
                result['nihss_score'] = self._map_to_nihss(result['gait_prob'])

                # Person detected - status already set to NORMAL/WARNING/DANGER above
                # Do not overwrite the classification status

        except Exception as e:
            print(f"[GAIT] Frame analysis error: {e}")

        return result

    def _calculate_gait_metrics(self):
        """Calculate gait metrics from pose history"""
        metrics = {
            'step_regularity': 0.0,
            'left_symmetry': 0.0,
            'right_symmetry': 0.0,
            'hip_stability': 0.0,
            'knee_range': 0.0,
            'ankle_range': 0.0
        }

        if len(self.knee_history) < 5:
            return metrics

        # Calculate ranges
        left_knee_y = [k[0][1] for k in self.knee_history]
        right_knee_y = [k[1][1] for k in self.knee_history]
        left_ankle_y = [a[0][1] for a in self.ankle_history]
        right_ankle_y = [a[1][1] for a in self.ankle_history]

        metrics['knee_range'] = max(left_knee_y) - min(left_knee_y) if left_knee_y else 0
        metrics['ankle_range'] = max(left_ankle_y) - min(left_ankle_y) if left_ankle_y else 0

        # Symmetry (left vs right)
        left_movement = np.std(left_knee_y) if len(left_knee_y) > 0 else 0
        right_movement = np.std(right_knee_y) if len(right_knee_y) > 0 else 0
        total_movement = left_movement + right_movement + 1e-6

        metrics['left_symmetry'] = left_movement / total_movement
        metrics['right_symmetry'] = right_movement / total_movement

        # Regularity from stride consistency
        knee_diffs = []
        for i in range(1, len(self.knee_history)):
            diff = abs(self.knee_history[i][0][1] - self.knee_history[i-1][0][1])
            knee_diffs.append(diff)

        if knee_diffs:
            metrics['step_regularity'] = 1.0 / (1.0 + np.std(knee_diffs) / (np.mean(knee_diffs) + 1e-6))

        # Hip stability
        hip_y_diffs = []
        for i in range(1, len(self.hip_history)):
            diff = abs(self.hip_history[i][0][1] - self.hip_history[i-1][0][1])
            hip_y_diffs.append(diff)

        if hip_y_diffs:
            metrics['hip_stability'] = 1.0 / (1.0 + np.std(hip_y_diffs) / (np.mean(hip_y_diffs) + 1e-6))

        return metrics

    def _calculate_gait_score(self, metrics):
        """Calculate gait abnormality score"""
        score = 0.0

        # Low step regularity (30 points)
        if metrics['step_regularity'] < 0.7:
            score += 30
        elif metrics['step_regularity'] < 0.5:
            score += 15

        # High asymmetry (30 points)
        asymmetry = abs(metrics['left_symmetry'] - 0.5)
        if asymmetry > 0.3:
            score += 30
        elif asymmetry > 0.2:
            score += 15

        # Low hip stability (20 points)
        if metrics['hip_stability'] < 0.8:
            score += 20

        # Limited range of motion (20 points)
        if metrics['knee_range'] < 50:
            score += 10
        if metrics['ankle_range'] < 80:
            score += 10

        return min(score, 100.0)

    def _map_to_nihss(self, gait_score):
        """Map gait score to NIHSS Item 6 (ROC optimized threshold: 64%)"""
        if gait_score < 64:
            return 0  # No drift (ROC optimized)
        elif gait_score < 75:
            return 1  # Mild drift
        elif gait_score < 85:
            return 2  # Some effort against gravity
        else:
            return 3  # No movement against gravity
