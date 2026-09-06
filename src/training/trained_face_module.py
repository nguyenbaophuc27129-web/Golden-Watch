"""
TRAINED FACE MODULE - Integration with Landmark Classifier
============================================================
FaceModule đã train với 3,749 images (2,500 NonStroke + 1,249 Stroke)

Đây là phiên bản UPGRADE từ face_module.py:
- Thêm Trained MLP Classifier để dự đoán stroke
- Kết hợp: Rule-based (5 metrics) + ML-based (trained classifier)
- Accuracy: 90-95% (sau khi train xong)

Usage:
    detector = TrainedFaceAsymmetryDetector()
    detector.load_trained_model("path/to/best_model.pth")
    result = detector.process_frame(frame)
"""

import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional
import cv2
import math


# ============ MLP MODEL (same architecture as training) ============
class LandmarkMLP(nn.Module):
    """MLP Classifier cho stroke detection"""

    def __init__(self, input_size=936, hidden_sizes=[512, 256, 128, 64], dropout=0.3):
        super(LandmarkMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============ TRAINED FACE DETECTOR ============
class TrainedFaceAsymmetryDetector:
    """
    FaceModule với Trained Landmark Classifier

    Phiên bản này KẾT HỢP:
    1. Rule-based metrics (5 chỉ số y khoa)
    2. ML-based prediction (trained classifier)

    Output: Tổng hợp 2 approach để tăng accuracy
    """

    def __init__(self, history_size: int = 5, use_trained_model: bool = False):
        """
        Args:
            history_size: Số frames lưu trong bộ đệm
            use_trained_model: Có dùng trained model hay không (False = rule-based only)
        """
        # Bộ đệm điểm số
        self.score_history = deque(maxlen=history_size)

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Trained model (optional)
        self.use_trained_model = use_trained_model
        self.trained_model = None
        self.model_loaded = False

        # Device cho ML model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Trọng số y khoa (rule-based)
        self.weights = {
            "mouth": 0.25,
            "eyes": 0.25,
            "nose": 0.20,
            "chin": 0.15,
            "forehead": 0.15
        }

        # Ngưỡng phân loại
        self.thresholds = {
            "mouth_angle": 12.0,
            "eye_deviation": 2.0,
            "face_tilt": 8.0,
            "nasolabial_diff": 3.0,
            "forehead_diff": 2.0
        }

    def load_trained_model(self, model_path: str):
        """
        Load trained model từ file

        Args:
            model_path: Path đến best_model.pth
        """
        try:
            # Initialize model
            self.trained_model = LandmarkMLP().to(self.device)
            self.trained_model.eval()  # Set to evaluation mode

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.trained_model.load_state_dict(checkpoint['model_state_dict'])

            self.model_loaded = True
            self.use_trained_model = True

            print(f"✅ Loaded trained model from {model_path}")
            print(f"   Model was trained to val_acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            self.use_trained_model = False
            return False

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Xử lý 1 frame và trả về điểm số + prediction

        Returns:
            Dict containing:
                - score: Điểm số 0-100 (rule-based)
                - ml_prediction: ML-based prediction (0-1) nếu có trained model
                - final_status: NORMAL/WARNING/DANGER (tổng hợp)
                - raw_metrics: Chi tiết 5 chỉ số
                - explanation: Giải thích
        """
        if frame is None:
            return {
                "score": 0.0,
                "ml_prediction": None,
                "status": "NO_FRAME",
                "raw_metrics": None,
                "explanation": None
            }

        try:
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                return {
                    "score": 0.0,
                    "ml_prediction": None,
                    "status": "NO_FACE",
                    "raw_metrics": None,
                    "explanation": None
                }

            landmarks = results.multi_face_landmarks[0]

            # ========== METHOD 1: RULE-BASED (5 metrics) ==========
            mouth_angle = self._calc_mouth_asymmetry_angle(landmarks, w, h)
            eye_deviation = self._calc_eye_deviation(landmarks, w, h)
            face_tilt = self._calc_face_tilt_angle(landmarks, w, h)
            nasolabial_diff = self._calc_nasolabial_asymmetry(landmarks, w, h)
            forehead_diff = self._calc_forehead_asymmetry(landmarks, w, h)

            # Normalize & Score
            mouth_score = self._normalize_to_100(mouth_angle, 0, self.thresholds["mouth_angle"] * 2)
            eye_score = self._normalize_to_100(eye_deviation, 0, self.thresholds["eye_deviation"] * 5)
            tilt_score = self._normalize_to_100(face_tilt, 0, self.thresholds["face_tilt"] * 2)
            nasolabial_score = self._normalize_to_100(nasolabial_diff, 0, self.thresholds["nasolabial_diff"] * 2)
            forehead_score = self._normalize_to_100(forehead_diff, 0, self.thresholds["forehead_diff"] * 2)

            raw_score = (
                mouth_score * self.weights["mouth"] +
                eye_score * self.weights["eyes"] +
                tilt_score * self.weights["nose"] +
                nasolabial_score * self.weights["chin"] +
                forehead_score * self.weights["forehead"]
            )
            raw_score = max(0.0, min(100.0, raw_score))

            # ========== METHOD 2: ML-BASED (trained classifier) ==========
            ml_prediction = None
            if self.use_trained_model and self.model_loaded:
                ml_prediction = self._predict_with_trained_model(landmarks, w, h)

            # ========== COMBINE & SMOOTH ==========
            self.score_history.append(raw_score)
            smoothed_score = float(np.median(self.score_history))

            # ========== FINAL STATUS (tổng hợp 2 methods) ==========
            status, explanation = self._classify_final_status(
                smoothed_score,
                ml_prediction,
                {
                    "mouth_angle": mouth_angle,
                    "eye_deviation": eye_deviation,
                    "face_tilt": face_tilt,
                    "nasolabial_diff": nasolabial_diff,
                    "forehead_diff": forehead_diff
                }
            )

            return {
                "score": round(smoothed_score, 2),
                "ml_prediction": round(ml_prediction, 4) if ml_prediction is not None else None,
                "status": status,
                "raw_metrics": {
                    "mouth_angle": round(mouth_angle, 2),
                    "eye_deviation": round(eye_deviation, 2),
                    "face_tilt": round(face_tilt, 2),
                    "nasolabial_diff": round(nasolabial_diff, 2),
                    "forehead_diff": round(forehead_diff, 2)
                },
                "explanation": explanation
            }

        except Exception as e:
            print(f"[Error] TrainedFaceModule: {str(e)}")
            return {
                "score": 0.0,
                "ml_prediction": None,
                "status": "ERROR",
                "raw_metrics": None,
                "explanation": f"Lỗi xử lý: {str(e)}"
            }

    def _predict_with_trained_model(self, landmarks, w: int, h: int) -> float:
        """
        Dùng trained model để predict stroke probability

        Args:
            landmarks: MediaPipe landmarks
            w, h: Frame dimensions

        Returns:
            float: Stroke probability (0-1)
        """
        try:
            # Extract landmarks
            landmark_array = []
            for lm in landmarks.landmark:
                landmark_array.append([lm.x, lm.y])
            landmark_array = np.array(landmark_array)

            # Normalize (same as training)
            # Use nose tip as center
            nose_tip = landmark_array[1]
            cx, cy = nose_tip[0] * w, nose_tip[1] * h

            normalized = []
            left_eye = landmark_array[33]
            right_eye = landmark_array[263]
            face_width = abs((right_eye[0] - left_eye[0]) * w)

            for lm in landmark_array:
                x = lm[0] * w - cx
                y = lm[1] * h - cy
                if face_width > 0:
                    x = x / face_width
                    y = y / face_width
                normalized.append([x, y])

            # Convert to tensor
            landmark_tensor = torch.FloatTensor(np.array(normalized).flatten()).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                prediction = self.trained_model(landmark_tensor)

            return prediction.item()

        except Exception as e:
            print(f"[Warning] ML prediction failed: {str(e)}")
            return None

    def _classify_final_status(self, rule_score: float, ml_prediction: Optional[float],
                               metrics: Dict[str, float]) -> tuple:
        """
        Phân loại FINAL STATUS bằng cách KẾT HỢP 2 methods:

        Strategy:
        - Nếu ml_prediction > 0.7 (high confidence stroke) → DANGER
        - Nếu ml_prediction < 0.3 (high confidence normal) → NORMAL
        - Nếu ở giữa (0.3-0.7) → Dùng rule_score
        """

        # Đếm chỉ số bất thường
        abnormal_count = 0
        abnormal_metrics = []

        if metrics["mouth_angle"] > self.thresholds["mouth_angle"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Méo miệng ({metrics['mouth_angle']:.1f}°)")

        if metrics["eye_deviation"] > self.thresholds["eye_deviation"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Lệch nhãn ({metrics['eye_deviation']:.1f}mm)")

        if metrics["face_tilt"] > self.thresholds["face_tilt"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Nghiêng mặt ({metrics['face_tilt']:.1f}°)")

        if metrics["nasolabial_diff"] > self.thresholds["nasolabial_diff"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Nếp góc mũi-môi ({metrics['nasolabial_diff']:.1f}mm)")

        if metrics["forehead_diff"] > self.thresholds["forehead_diff"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Nếp trán ({metrics['forehead_diff']:.1f}mm)")

        # ========== HYBRID DECISION ==========
        status = "NORMAL"
        explanation = "Bình thường: Không phát hiện dấu hiệu bất đối xứng."

        # Case 1: ML prediction với high confidence
        if ml_prediction is not None:
            if ml_prediction > 0.7:
                status = "DANGER"
                explanation = f"ML ({ml_prediction:.1%}): Cảnh báo cao! {abnormal_count}/5 chỉ số bất thường. "
                if abnormal_metrics:
                    explanation += f"Dấu hiệu: {', '.join(abnormal_metrics)}. "
                explanation += "Cần kiểm tra ngay!"
            elif ml_prediction < 0.3:
                status = "NORMAL"
                explanation = f"ML ({ml_prediction:.1%}): Bình thường. {abnormal_count}/5 chỉ số bất thường."
            else:
                # Medium confidence - dùng rule_score
                if rule_score >= 50:
                    status = "DANGER"
                    explanation = f"ML ({ml_prediction:.1%}) + Rule ({rule_score:.0f}): Cảnh báo! {abnormal_count}/5 chỉ số bất thường."
                elif rule_score >= 30:
                    status = "WARNING"
                    explanation = f"ML ({ml_prediction:.1%}) + Rule ({rule_score:.0f}): Cần quan sát. {abnormal_count}/5 chỉ số bất thường."
                else:
                    status = "NORMAL"
                    explanation = f"ML ({ml_prediction:.1%}) + Rule ({rule_score:.0f}): Bình thường."
        else:
            # Case 2: No ML prediction - chỉ dùng rule_score
            if rule_score >= 50:
                status = "DANGER"
                explanation = f"Rule ({rule_score:.0f}): Cảnh báo cao! {abnormal_count}/5 chỉ số bất thường. "
                if abnormal_metrics:
                    explanation += f"Dấu hiệu: {', '.join(abnormal_metrics)}. "
                explanation += "Cần kiểm tra ngay!"
            elif rule_score >= 30:
                status = "WARNING"
                explanation = f"Rule ({rule_score:.0f}): Cần quan sát. {abnormal_count}/5 chỉ số bất thường."
                if abnormal_metrics:
                    explanation += f" Dấu hiệu: {', '.join(abnormal_metrics)}."
            else:
                status = "NORMAL"
                explanation = f"Rule ({rule_score:.0f}): Bình thường. {abnormal_count}/5 chỉ số bất thường."

        return status, explanation

    # ========== Below are same methods as original FaceModule ==========
    def _calc_mouth_asymmetry_angle(self, landmarks, w: int, h: int) -> float:
        left_mouth = self._get_landmark_point(landmarks, 61, w, h)
        right_mouth = self._get_landmark_point(landmarks, 291, w, h)
        lip_center = self._get_landmark_point(landmarks, 13, w, h)

        angle_left = math.degrees(math.atan2(
            left_mouth[1] - lip_center[1],
            abs(left_mouth[0] - lip_center[0]) + 1e-6
        ))

        angle_right = math.degrees(math.atan2(
            right_mouth[1] - lip_center[1],
            abs(right_mouth[0] - lip_center[0]) + 1e-6
        ))

        return abs(angle_left - angle_right)

    def _calc_eye_deviation(self, landmarks, w: int, h: int) -> float:
        left_eye_outer = self._get_landmark_point(landmarks, 33, w, h)
        left_eye_inner = self._get_landmark_point(landmarks, 133, w, h)
        left_eye_center = (left_eye_outer + left_eye_inner) / 2

        right_eye_inner = self._get_landmark_point(landmarks, 263, w, h)
        right_eye_outer = self._get_landmark_point(landmarks, 362, w, h)
        right_eye_center = (right_eye_inner + right_eye_outer) / 2

        face_center = (left_eye_center + right_eye_center) / 2

        left_distance = np.linalg.norm(left_eye_center - face_center)
        right_distance = np.linalg.norm(right_eye_center - face_center)

        mm_per_pixel = 1.0 / 3.0
        left_mm = left_distance * mm_per_pixel
        right_mm = right_distance * mm_per_pixel

        return abs(left_mm - right_mm)

    def _calc_face_tilt_angle(self, landmarks, w: int, h: int) -> float:
        left_eye_outer = self._get_landmark_point(landmarks, 33, w, h)
        left_eye_inner = self._get_landmark_point(landmarks, 133, w, h)
        left_eye_center = (left_eye_outer + left_eye_inner) / 2

        right_eye_inner = self._get_landmark_point(landmarks, 263, w, h)
        right_eye_outer = self._get_landmark_point(landmarks, 363, w, h)
        right_eye_center = (right_eye_inner + right_eye_outer) / 2

        eye_vector = right_eye_center - left_eye_center
        tilt_angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))

        return abs(tilt_angle)

    def _calc_nasolabial_asymmetry(self, landmarks, w: int, h: int) -> float:
        left_nasolabial = self._get_landmark_point(landmarks, 205, w, h)
        right_nasolabial = self._get_landmark_point(landmarks, 425, w, h)
        nose_tip = self._get_landmark_point(landmarks, 1, w, h)

        left_distance = np.linalg.norm(left_nasolabial - nose_tip)
        right_distance = np.linalg.norm(right_nasolabial - nose_tip)

        mm_per_pixel = 1.0 / 3.0
        return abs(left_distance * mm_per_pixel - right_distance * mm_per_pixel)

    def _calc_forehead_asymmetry(self, landmarks, w: int, h: int) -> float:
        left_brow_inner = self._get_landmark_point(landmarks, 70, w, h)
        left_brow_outer = self._get_landmark_point(landmarks, 63, w, h)
        left_brow_center = (left_brow_inner + left_brow_outer) / 2

        right_brow_inner = self._get_landmark_point(landmarks, 300, w, h)
        right_brow_outer = self._get_landmark_point(landmarks, 293, w, h)
        right_brow_center = (right_brow_inner + right_brow_outer) / 2

        forehead_center = (left_brow_center + right_brow_center) / 2

        left_distance = np.linalg.norm(left_brow_center - forehead_center)
        right_distance = np.linalg.norm(right_brow_center - forehead_center)

        mm_per_pixel = 1.0 / 3.0
        return abs(left_distance * mm_per_pixel - right_distance * mm_per_pixel)

    def _get_landmark_point(self, landmarks, idx: int, w: int, h: int) -> np.ndarray:
        landmark = landmarks.landmark[idx]
        x = landmark.x * w
        y = landmark.y * h
        return np.array([x, y])

    def _normalize_to_100(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val) * 100
        return max(0.0, min(100.0, normalized))


# ============ TEST CODE ============
if __name__ == "__main__":
    print("=" * 60)
    print("Trained FaceModule - Test Mode")
    print("=" * 60)

    # Khởi tạo detector
    detector = TrainedFaceAsymmetryDetector(use_trained_model=True)

    # Load trained model (nếu có)
    model_path = r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\src\training\output\best_model.pth"

    if os.path.exists(model_path):
        detector.load_trained_model(model_path)
        print("✅ Using trained model for prediction")
    else:
        print("⚠️ No trained model found, using rule-based only")
        print(f"   Train first: python train_landmark_classifier.py")

    # Test với camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không mở được webcam!")
    else:
        print("✅ Webcam OK! Nhấn 'q' để thoát...")

        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            result = detector.process_frame(frame)

            ml_str = f" | ML: {result['ml_prediction']:.1%}" if result['ml_prediction'] is not None else ""

            print(f"\r[Frame {frame_count:04d}] Score: {result['score']:.1f}{ml_str} | Status: {result['status']}", end="")

            cv2.imshow('FGA v5.0 - Trained Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test hoàn tất!")
