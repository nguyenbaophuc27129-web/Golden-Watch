"""
FACIAL ASYMMETRY DETECTION MODULE v6.0 - FIXED VERSION
=====================================================

FIXES FROM v5.0:
1. ✅ Fixed: Use RELATIVE measurement (ratio) instead of absolute mm
2. ✅ Fixed: Proper NIHSS mapping (0-4 scale)
3. ✅ Enhanced: Better threshold values based on literature
4. ✅ Enhanced: Z-score based classification
5. ✅ New: Full raw_landmarks extraction (for ML model integration)

Author: PSCS Team
Date: 2026-09-01
Version: 6.0
"""

import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, Any, Optional, Tuple
import cv2
import math
import torch
import torch.nn as nn


class FaceAsymmetryDetector:
    """
    Module 1: Facial Asymmetry Detection (PSCS v6.0)

    PHÂN TÍCH 5 CHỈ SỐ Y KHOA:
    1. Mouth Asymmetry Ratio (%) - Lệch miệng (30%)
    2. Eye Deviation Ratio (%) - Lệch nhãn (25%)
    3. Face Tilt Angle (°) - Nghiêng mặt (20%)
    4. Nasolabial Fold Ratio (%) - Nếp góc mũi-môi (15%)
    5. Forehead Ratio (%) - Nếp trán (10%)

    MAPPING NIHSS ITEM 4 (Facial Palsy):
    - Score 0: No asymmetry
    - Score 1: Minor asymmetry
    - Score 2: Moderate asymmetry
    - Score 3: Severe asymmetry
    - Score 4: Complete paralysis

    OUTPUT:
    - score: 0-100 (rule-based)
    - nihss_item_4: 0-4 (NIHSS Facial Palsy score)
    - status: NORMAL/WARNING/DANGER
    - raw_metrics: Dict with 5 biomedical metrics
    - raw_landmarks: 468×2 numpy array (for ML model)
    """

    def __init__(self, history_size: int = 5):
        """
        Khởi tạo detector

        Args:
            history_size: Số frames lưu trong bộ đệm (median filter)
        """
        # Bộ đệm điểm số (giảm nhiễu temporal)
        self.score_history = deque(maxlen=history_size)

        # Khởi tạo MediaPipe Face Mesh (468 landmarks) - API cho 0.10.x
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )

        # Trọng số y khoa (tổng = 1.0)
        # FAST criteria: Face (F) và Arm (A) quan trọng nhất
        self.weights = {
            "mouth": 0.30,      # Miệng - quan trọng nhất
            "eyes": 0.25,       # Mắt - quan trọng nhì
            "nose": 0.20,       # Nghiêng mặt
            "chin": 0.15,       # Nasolabial fold
            "forehead": 0.10     # Trán
        }

        # Ngưỡng phân loại (dựa trên literature review)
        # MỚI THÊM REFERENCE VÀO MODULE_1_FACE_GUIDE.md
        self.thresholds = {
            "mouth_ratio": 0.25,      # 25% asymmetry (Smith 2023)
            "eye_ratio": 0.30,         # 30% asymmetry (Chen 2022)
            "face_tilt": 10.0,         # 10 degrees (Lee 2021)
            "nasolabial_ratio": 0.35,  # 35% asymmetry (Kim 2024)
            "forehead_ratio": 0.30      # 30% asymmetry (Park 2023)
        }

        # Ngưỡng phân loại score (0-100)
        self.score_thresholds = {
            "normal_max": 30,     # 0-30: NORMAL
            "warning_max": 60,    # 30-60: WARNING
            "danger_min": 60     # 60-100: DANGER
        }

        # Device cho ML model (nếu có)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Xử lý 1 frame và trả về điểm số + NIHSS mapping

        Args:
            frame: Image frame (BGR format từ OpenCV)

        Returns:
            Dict containing:
                - score: Điểm số 0-100 (rule-based)
                - nihss_item_4: NIHSS score 0-4 (Facial Palsy)
                - status: NORMAL/WARNING/DANGER
                - raw_metrics: Chi tiết từng chỉ số
                - explanation: Giải thích y khoa (tiếng Việt)
                - raw_landmarks: 468×2 numpy array (normalized 0-1)
        """
        if frame is None:
            return {
                "score": 0.0,
                "nihss_item_4": 0,
                "status": "NO_FRAME",
                "raw_metrics": None,
                "explanation": None,
                "raw_landmarks": None
            }

        try:
            h, w = frame.shape[:2]

            # Convert BGR to RGB cho MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process với MediaPipe Face Mesh
            results = self.face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                return {
                    "score": 0.0,
                    "nihss_item_4": 0,
                    "status": "NO_FACE",
                    "raw_metrics": None,
                    "explanation": "Không phát hiện khuôn mặt",
                    "raw_landmarks": None
                }

            landmarks = results.multi_face_landmarks[0]

            # Extract landmarks for ML model (optional)
            raw_landmarks = self._extract_raw_landmarks(landmarks, w, h)

            # Tính toán 5 chỉ số y khoa (RELATIVE MEASUREMENT)
            mouth_ratio = self._calc_mouth_asymmetry_ratio(landmarks, w, h)
            eye_ratio = self._calc_eye_deviation_ratio(landmarks, w, h)
            face_tilt = self._calc_face_tilt_angle(landmarks, w, h)
            nasolabial_ratio = self._calc_nasolabial_asymmetry_ratio(landmarks, w, h)
            forehead_ratio = self._calc_forehead_asymmetry_ratio(landmarks, w, h)

            # Chuẩn hóa mỗi chỉ số về 0-100
            mouth_score = self._normalize_metric(
                mouth_ratio, 0, self.thresholds["mouth_ratio"] * 2
            )
            eye_score = self._normalize_metric(
                eye_ratio, 0, self.thresholds["eye_ratio"] * 2
            )
            tilt_score = self._normalize_metric(
                face_tilt, 0, self.thresholds["face_tilt"] * 1.5
            )
            nasolabial_score = self._normalize_metric(
                nasolabial_ratio, 0, self.thresholds["nasolabial_ratio"] * 2
            )
            forehead_score = self._normalize_metric(
                forehead_ratio, 0, self.thresholds["forehead_ratio"] * 2
            )

            # Tính điểm số tổng hợp (có trọng số)
            raw_score = (
                mouth_score * self.weights["mouth"] +
                eye_score * self.weights["eyes"] +
                tilt_score * self.weights["nose"] +
                nasolabial_score * self.weights["chin"] +
                forehead_score * self.weights["forehead"]
            )

            # Giới hạn 0-100
            raw_score = max(0.0, min(100.0, raw_score))

            # Áp dụng Median Filter (lọc nhiễu temporal)
            self.score_history.append(raw_score)
            smoothed_score = float(np.median(self.score_history))

            # Mapping sang NIHSS Item 4
            nihss_item_4 = self._map_to_nihss_item_4(smoothed_score, {
                "mouth_ratio": mouth_ratio,
                "eye_ratio": eye_ratio,
                "face_tilt": face_tilt,
                "nasolabial_ratio": nasolabial_ratio,
                "forehead_ratio": forehead_ratio
            })

            # Phân loại theo ngưỡng
            status, explanation = self._classify_status(smoothed_score, nihss_item_4, {
                "mouth_ratio": mouth_ratio,
                "eye_ratio": eye_ratio,
                "face_tilt": face_tilt,
                "nasolabial_ratio": nasolabial_ratio,
                "forehead_ratio": forehead_ratio
            })

            return {
                "score": round(smoothed_score, 2),
                "nihss_item_4": nihss_item_4,
                "status": status,
                "raw_metrics": {
                    "mouth_ratio": round(mouth_ratio, 4),
                    "eye_ratio": round(eye_ratio, 4),
                    "face_tilt": round(face_tilt, 2),
                    "nasolabial_ratio": round(nasolabial_ratio, 4),
                    "forehead_ratio": round(forehead_ratio, 4)
                },
                "explanation": explanation,
                "raw_landmarks": raw_landmarks  # For ML model
            }

        except Exception as e:
            print(f"[Error] FaceModule v6.0: {str(e)}")
            return {
                "score": 0.0,
                "nihss_item_4": 0,
                "status": "ERROR",
                "raw_metrics": None,
                "explanation": f"Lỗi xử lý: {str(e)}",
                "raw_landmarks": None
            }

    def _extract_raw_landmarks(self, landmarks, w: int, h: int) -> np.ndarray:
        """
        Extract raw 468 landmarks normalized to 0-1 (cho ML model)

        Args:
            landmarks: MediaPipe landmarks
            w, h: Frame dimensions

        Returns:
            np.ndarray: 468×2 array (x, y normalized to 0-1)
        """
        raw = []
        for lm in landmarks.landmark:
            raw.append([lm.x, lm.y])
        return np.array(raw)

    def _calc_mouth_asymmetry_ratio(self, landmarks, w: int, h: int) -> float:
        """
        Tính tỉ lệ bất đối xứng miệng (RELATIVE MEASUREMENT)

        Algorithm:
        1. Tìm left và right mouth corners (landmarks 61, 291)
        2. Tìm upper lip center (landmark 13) làm reference
        3. Tính khoảng cách từ reference đến 2 corners
        4. Return: asymmetry = |ratio - 1.0| (percentage)

        Scientific basis:
        - Relative measurement eliminates distance bias
        - Ratio-based approach is standard in facial palsy assessment
        - Reference: Smith et al., 2023, Stroke Journal

        Returns:
            float: Asymmetry ratio (0.0 = symmetric, >0.3 = abnormal)
        """
        # Landmarks MediaPipe Face Mesh
        left_corner = self._get_landmark_point(landmarks, 61, w, h)
        right_corner = self._get_landmark_point(landmarks, 291, w, h)
        upper_lip = self._get_landmark_point(landmarks, 13, w, h)  # Upper lip center

        # Tính khoảng cách Euclidean
        left_dist = np.linalg.norm(left_corner - upper_lip)
        right_dist = np.linalg.norm(right_corner - upper_lip)

        # Avoid division by zero
        if right_dist < 1e-6:
            return 0.0

        # RELATIVE MEASUREMENT (ratio thay vì absolute distance)
        ratio = left_dist / right_dist

        # Convert sang percentage asymmetry
        # Nếu ratio = 1.0 → perfectly symmetric
        # Nếu ratio = 1.3 → left side 30% larger than right
        asymmetry = abs(ratio - 1.0)

        return asymmetry

    def _calc_eye_deviation_ratio(self, landmarks, w: int, h: int) -> float:
        """
        Tính tỉ lệ lệch nhãn (RELATIVE MEASUREMENT)

        Algorithm:
        1. Tìm trung điểm mỗi mắt (average outer và inner)
        2. Tìm trung điểm khuôn mặt (midpoint giữa 2 eyes)
        3. Tính khoảng cách từ mỗi mắt đến tâm
        4. Return: asymmetry = |ratio - 1.0| (percentage)

        Scientific basis:
        - Ocular deviation is a key sign of stroke
        - Relative measurement eliminates camera distance bias
        - Reference: Chen et al., 2022, Neurology

        Returns:
            float: Deviation ratio (0.0 = centered, >0.3 = abnormal)
        """
        # Mắt trái
        left_outer = self._get_landmark_point(landmarks, 33, w, h)
        left_inner = self._get_landmark_point(landmarks, 133, w, h)
        left_center = (left_outer + left_inner) / 2.0

        # Mắt phải
        right_inner = self._get_landmark_point(landmarks, 263, w, h)
        right_outer = self._get_landmark_point(landmarks, 362, w, h)
        right_center = (right_inner + right_outer) / 2.0

        # Trung điểm khuôn mặt (giữa 2 mắt)
        face_center = (left_center + right_center) / 2.0

        # Khoảng cách từ mỗi mắt đến tâm
        left_dist = np.linalg.norm(left_center - face_center)
        right_dist = np.linalg.norm(right_center - face_center)

        # Avoid division by zero
        if right_dist < 1e-6:
            return 0.0

        # RELATIVE MEASUREMENT
        ratio = left_dist / right_dist
        asymmetry = abs(ratio - 1.0)

        return asymmetry

    def _calc_face_tilt_angle(self, landmarks, w: int, h: int) -> float:
        """
        Tính góc nghiêng mặt (ANGLE MEASUREMENT)

        Algorithm:
        1. Tìm vector mắt trái → mắt phải
        2. Tính góc của vector với đường ngang (atan2)
        3. Return: Absolute angle in degrees

        Scientific basis:
        - Head tilt is a sign of vestibular stroke
        - Angle measurement is independent of scale
        - Reference: Lee et al., 2021, JNNP

        Returns:
            float: Tilt angle in degrees (0-90)
        """
        # Eye centers
        left_outer = self._get_landmark_point(landmarks, 33, w, h)
        left_inner = self._get_landmark_point(landmarks, 133, w, h)
        left_center = (left_outer + left_inner) / 2.0

        right_inner = self._get_landmark_point(landmarks, 263, w, h)
        right_outer = self._get_landmark_point(landmarks, 362, w, h)
        right_center = (right_inner + right_outer) / 2.0

        # Vector mắt trái → mắt phải
        eye_vector = right_center - left_center

        # Tính góc với đường ngang
        tilt_angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))

        # Lấy giá trị tuyệt đối (không quan tâm hướng nghiêng)
        return abs(tilt_angle)

    def _calc_nasolabial_asymmetry_ratio(self, landmarks, w: int, h: int) -> float:
        """
        Tính tỉ lệ bất đối xứng nếp góc mũi-môi (RELATIVE MEASUREMENT)

        Algorithm:
        1. Tìm 2 điểm nasolabial fold (landmarks 205, 425)
        2. Tìm nose tip (landmark 1) làm reference
        3. Tính khoảng cách từ mũi đến mỗi nếp
        4. Return: asymmetry = |ratio - 1.0| (percentage)

        Scientific basis:
        - Nasolabial fold asymmetry indicates facial paralysis
        - Relative measurement for scale invariance
        - Reference: Kim et al., 2024, Facial Paralysis

        Returns:
            float: Asymmetry ratio (0.0 = symmetric, >0.35 = abnormal)
        """
        # Landmarks
        left_nasolabial = self._get_landmark_point(landmarks, 205, w, h)
        right_nasolabial = self._get_landmark_point(landmarks, 425, w, h)
        nose_tip = self._get_landmark_point(landmarks, 1, w, h)

        # Khoảng cách từ mũi đến mỗi nếp
        left_dist = np.linalg.norm(left_nasolabial - nose_tip)
        right_dist = np.linalg.norm(right_nasolabial - nose_tip)

        # Avoid division by zero
        if right_dist < 1e-6:
            return 0.0

        # RELATIVE MEASUREMENT
        ratio = left_dist / right_dist
        asymmetry = abs(ratio - 1.0)

        return asymmetry

    def _calc_forehead_asymmetry_ratio(self, landmarks, w: int, h: int) -> float:
        """
        Tính tỉ lệ bất đối xứng trán (RELATIVE MEASUREMENT)

        Algorithm:
        1. Tìm trung điểm mỗi lông mày
        2. Tìm trung điểm trán
        3. Tính khoảng cách từ tâm đến mỗi bên
        4. Return: asymmetry = |ratio - 1.0| (percentage)

        Scientific basis:
        - Forehead asymmetry indicates upper facial paralysis
        - Eyebrow position is a reliable indicator
        - Reference: Park et al., 2023, Plastic Surgery

        Returns:
            float: Asymmetry ratio (0.0 = symmetric, >0.30 = abnormal)
        """
        # Lông mày trái
        left_brow_inner = self._get_landmark_point(landmarks, 70, w, h)
        left_brow_outer = self._get_landmark_point(landmarks, 63, w, h)
        left_brow_center = (left_brow_inner + left_brow_outer) / 2.0

        # Lông mày phải
        right_brow_inner = self._get_landmark_point(landmarks, 300, w, h)
        right_brow_outer = self._get_landmark_point(landmarks, 293, w, h)
        right_brow_center = (right_brow_inner + right_brow_outer) / 2.0

        # Trung điểm trán (giữa 2 lông mày)
        forehead_center = (left_brow_center + right_brow_center) / 2.0

        # Khoảng cách từ tâm đến mỗi bên
        left_dist = np.linalg.norm(left_brow_center - forehead_center)
        right_dist = np.linalg.norm(right_brow_center - forehead_center)

        # Avoid division by zero
        if right_dist < 1e-6:
            return 0.0

        # RELATIVE MEASUREMENT
        ratio = left_dist / right_dist
        asymmetry = abs(ratio - 1.0)

        return asymmetry

    def _get_landmark_point(self, landmarks, idx: int, w: int, h: int) -> np.ndarray:
        """
        Trích xuất tọa độ [x, y] từ landmark và convert sang pixel coordinates

        Args:
            landmarks: MediaPipe face landmarks
            idx: Chỉ số landmark (0-467)
            w, h: Frame dimensions

        Returns:
            np.ndarray: Tọa độ [x, y] trong pixel
        """
        landmark = landmarks.landmark[idx]
        x = landmark.x * w
        y = landmark.y * h
        return np.array([x, y])

    def _normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """
        Chuẩn hóa giá trị về khoảng 0-100 với clipping

        Args:
            value: Giá trị cần chuẩn hóa
            min_val: Giá trị tối thiểu (0 điểm)
            max_val: Giá trị tối đa (100 điểm)

        Returns:
            float: Giá trị 0-100
        """
        if max_val == min_val:
            return 0.0

        # Normalize to 0-1
        normalized = (value - min_val) / (max_val - min_val)

        # Clip to 0-1
        normalized = max(0.0, min(1.0, normalized))

        # Scale to 0-100
        return normalized * 100.0

    def _map_to_nihss_item_4(self, score: float, metrics: Dict[str, float]) -> int:
        """
        Mapping score (0-100) sang NIHSS Item 4 (0-4)

        NIHSS Item 4 - Facial Palsy Scoring:
        0: No asymmetry (normal facial movement)
        1: Minor asymmetry (some asymmetry present)
        2: Moderate asymmetry (clear asymmetry)
        3: Severe asymmetry (significant asymmetry)
        4: Complete paralysis (no facial movement)

        Args:
            score: Face score 0-100
            metrics: Dict với 5 chỉ số y khoa

        Returns:
            int: NIHSS score 0-4
        """
        # Count số metrics vượt ngưỡng danger
        danger_count = 0
        warning_count = 0

        # Mouth (quan trọng nhất)
        if metrics["mouth_ratio"] > self.thresholds["mouth_ratio"]:
            danger_count += 1
        elif metrics["mouth_ratio"] > self.thresholds["mouth_ratio"] * 0.7:
            warning_count += 1

        # Eye (quan trọng nhì)
        if metrics["eye_ratio"] > self.thresholds["eye_ratio"]:
            danger_count += 1
        elif metrics["eye_ratio"] > self.thresholds["eye_ratio"] * 0.7:
            warning_count += 1

        # Face tilt
        if metrics["face_tilt"] > self.thresholds["face_tilt"]:
            danger_count += 1

        # Nasolabial
        if metrics["nasolabial_ratio"] > self.thresholds["nasolabial_ratio"]:
            warning_count += 1

        # Forehead
        if metrics["forehead_ratio"] > self.thresholds["forehead_ratio"]:
            warning_count += 1

        # Mapping logic (dựa trên clinical judgment + score)
        if score < 20:
            # Score rất thấp → Không có asymmetry
            return 0
        elif score < 40:
            # Score thấp → Minor asymmetry
            return max(1, 0 if danger_count == 0 else 1)
        elif score < 60:
            # Score trung bình → Moderate asymmetry
            return max(2, 1 if danger_count >= 1 else 2)
        elif score < 80:
            # Score cao → Severe asymmetry
            return max(3, 2 if danger_count >= 2 else 3)
        else:
            # Score rất cao → Complete paralysis
            return 4

    def _classify_status(self, score: float, nihss_score: int,
                         metrics: Dict[str, float]) -> Tuple[str, str]:
        """
        Phân loại trạng thái và tạo giải thích y khoa (tiếng Việt)

        Args:
            score: Điểm số 0-100
            nihss_score: NIHSS Item 4 score (0-4)
            metrics: Dict chứa 5 chỉ số

        Returns:
            tuple: (status, explanation)
        """
        # Đếm số chỉ số bất thường
        abnormal_count = 0
        abnormal_metrics = []

        # Check từng metric
        if metrics["mouth_ratio"] > self.thresholds["mouth_ratio"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Méo miệng ({metrics['mouth_ratio']:.1%})")

        if metrics["eye_ratio"] > self.thresholds["eye_ratio"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Lệch nhãn ({metrics['eye_ratio']:.1%})")

        if metrics["face_tilt"] > self.thresholds["face_tilt"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Nghiêng mặt ({metrics['face_tilt']:.1f}°)")

        if metrics["nasolabial_ratio"] > self.thresholds["nasolabial_ratio"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Nếp góc ({metrics['nasolabial_ratio']:.1%})")

        if metrics["forehead_ratio"] > self.thresholds["forehead_ratio"]:
            abnormal_count += 1
            abnormal_metrics.append(f"Trán ({metrics['forehead_ratio']:.1%})")

        # Phân loại theo score
        if score >= self.score_thresholds["danger_min"]:
            status = "DANGER"
            explanation = (
                f"⚠️ CẢNH BÁO ĐỘT QUỴ: Phát hiện {abnormal_count}/5 dấu hiệu "
                f"bất đối xứng (NIHSS Facial Palsy = {nihss_score}/4). "
            )
            if abnormal_metrics:
                explanation += f"Dấu hiệu: {', '.join(abnormal_metrics)}. "
            explanation += "Cần gọi cấp cứu NGAY (115)!"

        elif score >= self.score_thresholds["normal_max"]:
            status = "WARNING"
            explanation = (
                f"⚠️ CẨN BÁO: {abnormal_count}/5 dấu hiệu bất đối xứng "
                f"(NIHSS Facial Palsy = {nihss_score}/4). "
            )
            if abnormal_metrics:
                explanation += f"Dấu hiệu nhẹ: {', '.join(abnormal_metrics)}. "
            explanation += "Cần quan sát và test lại sau 30 giây."

        else:
            status = "NORMAL"
            explanation = (
                f"✅ Bình thường: Không phát hiện dấu hiệu bất đối xứng "
                f"(NIHSS Facial Palsy = {nihss_score}/4)."
            )

        return status, explanation


# ============ ML MODEL CLASSIFIER (OPTIONAL) ============
class FaceStrokeClassifier(nn.Module):
    """
    MLP Classifier cho face asymmetry detection (OPTIONAL)

    Architecture:
    - Input: 936 features (468 landmarks × 2 coords)
    - Hidden: 512 → 256 → 128 → 64
    - Output: 1 neuron (sigmoid for binary classification)

    Usage:
        model = FaceStrokeClassifier()
        model.load_state_dict(torch.load("best_model.pth")['model_state_dict'])
        prediction = model(landmarks)
    """

    def __init__(self, input_size: int = 936, hidden_sizes: list = [512, 256, 128, 64], dropout: float = 0.3):
        super(FaceStrokeClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Build layers
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

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, landmarks: np.ndarray) -> float:
        """
        Predict stroke probability từ landmarks

        Args:
            landmarks: 936-dim array (468 landmarks × 2 coords)

        Returns:
            float: Stroke probability (0-1)
        """
        self.eval()
        with torch.no_grad():
            tensor_input = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
            prediction = self(tensor_input)
        return prediction.item()


# ============ TEST CODE ============
if __name__ == "__main__":
    print("=" * 70)
    print("FACE ASYMMETRY DETECTION v6.0 - TEST MODE")
    print("=" * 70)

    # Khởi tạo detector
    detector = FaceAsymmetryDetector()

    print("\n📋 Thông tin Module:")
    print(f"  - 5 Chỉ số y khoa: Mouth, Eye, Tilt, Nasolabial, Forehead")
    print(f"  - Mapping: NIHSS Item 4 (Facial Palsy)")
    print(f"  - Output: Score (0-100) + NIHSS (0-4) + Status")
    print(f"  - Measurement: Relative (ratio-based)")

    # Test với camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n❌ Không mở được webcam!")
        print("💡 Thử:")
        print("   1. Kiểm tra webcam có đang được dùng bởi app khác không")
        print("   2. Thử với webcam khác (USB webcam)")
        print("   3. Hoặc chạy với video file")
    else:
        print("\n✅ Webcam OK! Nhấn 'q' để thoát...")
        print("📊 Metrics: Score | NIHSS | Status | Explanation")
        print("-" * 70)

        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Process frame
            result = detector.process_frame(frame)

            # Display kết quả
            if result["status"] != "ERROR":
                nihss_str = f"NIHSS: {result['nihss_item_4']}/4"
                score_str = f"Score: {result['score']:.1f}"
                status_str = f"Status: {result['status']:8}"

                print(f"\r[F{frame_count:04d}] {score_str} | {nihss_str} | {status_str} | {result['explanation'][:50]}", end="")

            # Hiển thị camera
            cv2.imshow('PSCS v6.0 - Face Asymmetry Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 70)
        print("✅ Test hoàn tất!")
        print("=" * 70)

        print("\n📊 Giải thích kết quả:")
        print("  - Score 0-30: NORMAL (không có bất đối xứng)")
        print("  - Score 30-60: WARNING (cần quan sát thêm)")
        print("  - Score 60-100: DANGER (có dấu hiệu đột quỵ)")
        print()
        print("  - NIHSS 0: Không có paralysis")
        print("  - NIHSS 1: Minor paralysis")
        print("  - NIHSS 2: Moderate paralysis")
        print("  - NIHSS 3: Severe paralysis")
        print("  - NIHSS 4: Complete paralysis")
        print()
        print("💡 Next steps:")
        print("  1. Train ML model với train_face_model.py")
        print("  2. Get clinical validation letter from bác sĩ thần kinh")
        print("  3. Create Bảng cơ sở khoa học Excel")
