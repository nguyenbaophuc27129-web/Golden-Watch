import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, Any, Optional
import cv2
import math

class FaceAsymmetryDetector:
    """
    Module 01: Facial Asymmetry Detection (FGA v5.0)

    Phân tích 5 chỉ số y khoa để phát hiện rối loạn mặt:
    1. Mouth Asymmetry Angle (°) - Góc bất đối xứng miệng
    2. Eye Deviation (mm) - Lệch nhãn
    3. Face Tilt Angle (°) - Góc nghiêng mặt
    4. Nasolabial Fold Asymmetry (mm) - Chênh lệch nếp góc mũi-môi
    5. Forehead Asymmetry (mm) - Chênh lệch trán

    Mục tiêu: Phát hiện đột quỵ với Sensitivity >85%
    """

    def __init__(self, history_size: int = 5):
        """
        Khởi tạo detector

        Args:
            history_size: Số frames lưu trong bộ đệm (median filter)
        """
        # Bộ đệm lưu lịch sử điểm số để lọc trung vị (giảm nhiễu)
        self.score_history = deque(maxlen=history_size)

        # Khởi tạo MediaPipe Face Mesh (468 landmarks)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Trọng số y khoa (tổng = 1.0)
        # Mouth và Eyes quan trọng nhất (FAST criteria)
        self.weights = {
            "mouth": 0.25,
            "eyes": 0.25,
            "nose": 0.20,  # Face tilt
            "chin": 0.15,  # Nasolabial fold
            "forehead": 0.15
        }

        # Ngưỡng phân loại (có thể điều chỉnh sau calibration)
        self.thresholds = {
            "mouth_angle": 12.0,  # °
            "eye_deviation": 2.0,   # mm
            "face_tilt": 8.0,      # °
            "nasolabial_diff": 3.0, # mm
            "forehead_diff": 2.0     # mm
        }

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Xử lý 1 frame và trả về điểm số rối loạn mặt

        Args:
            frame: Image frame (BGR format từ OpenCV)

        Returns:
            Dict containing:
                - score: Điểm số 0-100
                - status: NORMAL/WARNING/DANGER
                - raw_metrics: Chi tiết từng chỉ số
                - explanation: Giải thích y khoa
        """
        if frame is None:
            return {"score": 0.0, "status": "NO_FRAME", "raw_metrics": None, "explanation": None}

        try:
            h, w = frame.shape[:2]

            # Convert BGR to RGB cho MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process với MediaPipe Face Mesh
            results = self.face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                return {"score": 0.0, "status": "NO_FACE", "raw_metrics": None, "explanation": None}

            landmarks = results.multi_face_landmarks[0]

            # Tính toán 5 chỉ số y khoa
            mouth_angle = self._calc_mouth_asymmetry_angle(landmarks, w, h)
            eye_deviation = self._calc_eye_deviation(landmarks, w, h)
            face_tilt = self._calc_face_tilt_angle(landmarks, w, h)
            nasolabial_diff = self._calc_nasolabial_asymmetry(landmarks, w, h)
            forehead_diff = self._calc_forehead_asymmetry(landmarks, w, h)

            # Chuẩn hóa mỗi chỉ số về 0-100
            mouth_score = self._normalize_to_100(mouth_angle, 0, self.thresholds["mouth_angle"] * 2)
            eye_score = self._normalize_to_100(eye_deviation, 0, self.thresholds["eye_deviation"] * 5)
            tilt_score = self._normalize_to_100(face_tilt, 0, self.thresholds["face_tilt"] * 2)
            nasolabial_score = self._normalize_to_100(nasolabial_diff, 0, self.thresholds["nasolabial_diff"] * 2)
            forehead_score = self._normalize_to_100(forehead_diff, 0, self.thresholds["forehead_diff"] * 2)

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

            # Áp dụng Median Filter (lọc nhiễu)
            self.score_history.append(raw_score)
            smoothed_score = float(np.median(self.score_history))

            # Phân loại theo ngưỡng
            status, explanation = self._classify_status(smoothed_score, {
                "mouth_angle": mouth_angle,
                "eye_deviation": eye_deviation,
                "face_tilt": face_tilt,
                "nasolabial_diff": nasolabial_diff,
                "forehead_diff": forehead_diff
            })

            return {
                "score": round(smoothed_score, 2),
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
            print(f"[Error] FaceModule: {str(e)}")
            return {"score": 0.0, "status": "ERROR", "raw_metrics": None, "explanation": f"Lỗi xử lý: {str(e)}"}

    def _calc_mouth_asymmetry_angle(self, landmarks, w: int, h: int) -> float:
        """
        Tính góc bất đối xứng miệng

        Logic:
        1. Lấy 2 góc miệng (trái và phải)
        2. Tính góc mỗi miệng với đường ngang
        3. Chênh lệch 2 góc = độ bất đối xứng

        Landmarks MediaPipe:
        - Left mouth corner: 61
        - Right mouth corner: 291
        - Left lip top: 61
        - Right lip top: 291
        """
        # Lấy tọa độ các điểm miệng
        left_mouth = self._get_landmark_point(landmarks, 61, w, h)
        right_mouth = self._get_landmark_point(landmarks, 291, w, h)

        # Lấy điểm tham chiếu (giữa môi)
        lip_center = self._get_landmark_point(landmarks, 0, w, h)  # Chin point
        lip_center = self._get_landmark_point(landmarks, 13, w, h)  # Upper lip

        # Tính góc miệng trái với đường ngang
        angle_left = math.degrees(math.atan2(
            left_mouth[1] - lip_center[1],
            abs(left_mouth[0] - lip_center[0]) + 1e-6
        ))

        # Tính góc miệng phải với đường ngang
        angle_right = math.degrees(math.atan2(
            right_mouth[1] - lip_center[1],
            abs(right_mouth[0] - lip_center[0]) + 1e-6
        ))

        # Chênh lệch góc = độ bất đối xứng
        asymmetry_angle = abs(angle_left - angle_right)

        return asymmetry_angle

    def _calc_eye_deviation(self, landmarks, w: int, h: int) -> float:
        """
        Tính lệch nhãn (mm)

        Logic:
        1. Lấy trung điểm mắt trái và mắt phải
        2. So sánh với trung điểm khuôn mặt
        3. Khoảng cách lệch = độ lệch nhãn

        Landmarks MediaPipe:
        - Left eye center: Giữa 33 (outer) và 133 (inner)
        - Right eye center: Giữa 362 (outer) và 263 (inner)
        """
        # Mắt trái
        left_eye_outer = self._get_landmark_point(landmarks, 33, w, h)
        left_eye_inner = self._get_landmark_point(landmarks, 133, w, h)
        left_eye_center = (left_eye_outer + left_eye_inner) / 2

        # Mắt phải
        right_eye_inner = self._get_landmark_point(landmarks, 263, w, h)
        right_eye_outer = self._get_landmark_point(landmarks, 362, w, h)
        right_eye_center = (right_eye_inner + right_eye_outer) / 2

        # Trung điểm khuôn mặt (giữa 2 mắt)
        face_center = (left_eye_center + right_eye_center) / 2

        # Khoảng cách từ mỗi mắt đến tâm
        left_distance = np.linalg.norm(left_eye_center - face_center)
        right_distance = np.linalg.norm(right_eye_center - face_center)

        # Chuyển đổi pixel sang mm (giả sử 1mm ≈ 3 pixels ở khoảng cách camera)
        mm_per_pixel = 1.0 / 3.0
        left_mm = left_distance * mm_per_pixel
        right_mm = right_distance * mm_per_pixel

        # Chênh lệch = độ lệch nhãn
        deviation = abs(left_mm - right_mm)

        return deviation

    def _calc_face_tilt_angle(self, landmarks, w: int, h: int) -> float:
        """
        Tính góc nghiêng mặt

        Logic:
        1. Lấy đường nối 2 mắt (mắt trái → mắt phải)
        2. Tính góc của đường này với đường ngang
        3. Góc nghiêng = độ lệch từ 0° (ngang)

        Landmarks:
        - Left eye center: Giữa 33 và 133
        - Right eye center: Giữa 362 và 263
        """
        # Mắt trái
        left_eye_outer = self._get_landmark_point(landmarks, 33, w, h)
        left_eye_inner = self._get_landmark_point(landmarks, 133, w, h)
        left_eye_center = (left_eye_outer + left_eye_inner) / 2

        # Mắt phải
        right_eye_inner = self._get_landmark_point(landmarks, 263, w, h)
        right_eye_outer = self._get_landmark_point(landmarks, 362, w, h)
        right_eye_center = (right_eye_inner + right_eye_outer) / 2

        # Vector mắt trái → mắt phải
        eye_vector = right_eye_center - left_eye_center

        # Tính góc với đường ngang
        tilt_angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))

        # Lấy giá trị tuyệt đối (không quan tâm nghiêng trái hay phải)
        return abs(tilt_angle)

    def _calc_nasolabial_asymmetry(self, landmarks, w: int, h: int) -> float:
        """
        Tính chênh lệch nếp góc mũi-môi (nasolabial fold)

        Logic:
        1. Lấy 2 điểm nếp góc mũi-môi (trái và phải)
        2. So sánh độ sâu của 2 nếp
        3. Chênh lệch độ sâu = độ bất đối xứng

        Landmarks (approximate):
        - Left nasolabial fold: Khu vực giữa mũi và miệng trái
        - Right nasolabial fold: Khu vực giữa mũi và miệng phải
        """
        # Sử dụng landmark 205 (trái) và 425 (phải) - khu vực nếp góc
        left_nasolabial = self._get_landmark_point(landmarks, 205, w, h)
        right_nasolabial = self._get_landmark_point(landmarks, 425, w, h)

        # Lấy điểm mũi làm tham chiếu
        nose_tip = self._get_landmark_point(landmarks, 1, w, h)

        # Khoảng cách từ mũi đến mỗi nếp
        left_distance = np.linalg.norm(left_nasolabial - nose_tip)
        right_distance = np.linalg.norm(right_nasolabial - nose_tip)

        # Chuyển đổi sang mm
        mm_per_pixel = 1.0 / 3.0
        left_mm = left_distance * mm_per_pixel
        right_mm = right_distance * mm_per_pixel

        # Chênh lệch
        asymmetry = abs(left_mm - right_mm)

        return asymmetry

    def _calc_forehead_asymmetry(self, landmarks, w: int, h: int) -> float:
        """
        Tính chênh lệch trán

        Logic:
        1. Lấy điểm giữa trán (giữa 2 lông mày)
        2. So sánh khoảng cách đến 2 bên thái dương
        3. Chênh lệch = độ bất đối xứng

        Landmarks:
        - Left eyebrow center: Giữa 70 và 63
        - Right eyebrow center: Giữa 300 và 293
        """
        # Lông mày trái
        left_brow_inner = self._get_landmark_point(landmarks, 70, w, h)
        left_brow_outer = self._get_landmark_point(landmarks, 63, w, h)
        left_brow_center = (left_brow_inner + left_brow_outer) / 2

        # Lông mày phải
        right_brow_inner = self._get_landmark_point(landmarks, 300, w, h)
        right_brow_outer = self._get_landmark_point(landmarks, 293, w, h)
        right_brow_center = (right_brow_inner + right_brow_outer) / 2

        # Lấy trung điểm trán (giữa 2 lông mày)
        forehead_center = (left_brow_center + right_brow_center) / 2

        # Khoảng cách đến từng lông mày
        left_distance = np.linalg.norm(left_brow_center - forehead_center)
        right_distance = np.linalg.norm(right_brow_center - forehead_center)

        # Chuyển đổi sang mm
        mm_per_pixel = 1.0 / 3.0
        left_mm = left_distance * mm_per_pixel
        right_mm = right_distance * mm_per_pixel

        # Chênh lệch
        asymmetry = abs(left_mm - right_mm)

        return asymmetry

    def _get_landmark_point(self, landmarks, idx: int, w: int, h: int) -> np.ndarray:
        """
        Trích xuất tọa độ [x, y] từ landmark và chuẩn hóa theo kích thước frame

        Args:
            landmarks: MediaPipe face landmarks
            idx: Chỉ số landmark
            w: Chiều rộng frame
            h: Chiều cao frame

        Returns:
            np.ndarray: Tọa độ [x, y] trong pixel
        """
        landmark = landmarks.landmark[idx]
        x = landmark.x * w
        y = landmark.y * h
        return np.array([x, y])

    def _normalize_to_100(self, value: float, min_val: float, max_val: float) -> float:
        """
        Chuẩn hóa giá trị về khoảng 0-100

        Args:
            value: Giá trị cần chuẩn hóa
            min_val: Giá trị tối thiểu (0 điểm)
            max_val: Giá trị tối đa (100 điểm)

        Returns:
            float: Giá trị 0-100
        """
        if max_val == min_val:
            return 0.0

        normalized = (value - min_val) / (max_val - min_val) * 100
        return max(0.0, min(100.0, normalized))

    def _classify_status(self, score: float, metrics: Dict[str, float]) -> tuple:
        """
        Phân loại trạng thái và tạo giải thích y khoa

        Args:
            score: Điểm số 0-100
            metrics: Dict chứa 5 chỉ số

        Returns:
            tuple: (status, explanation)
        """
        # Đếm số chỉ số vượt ngưỡng
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

        # Phân loại theo điểm số
        if score >= 50:
            status = "DANGER"
            explanation = f"CẢNH BÁO: Phát hiện {abnormal_count}/5 dấu hiệu bất đối xứng. "
            if abnormal_metrics:
                explanation += f"Dấu hiệu: {', '.join(abnormal_metrics)}. "
            explanation += "Cần kiểm tra ngay theo FAST!"
        elif score >= 30:
            status = "WARNING"
            explanation = f"CẢNH BÁO: {abnormal_count}/5 dấu hiệu bất đối xứng. "
            if abnormal_metrics:
                explanation += f"Dấu hiệu nhẹ: {', '.join(abnormal_metrics)}. "
            explanation += "Cần quan sát thêm."
        else:
            status = "NORMAL"
            explanation = "Bình thường: Không phát hiện dấu hiệu bất đối xứng khuôn mặt."

        return status, explanation

# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("FaceModule v5.0 - Test Mode")
    print("=" * 60)

    # Khởi tạo detector
    detector = FaceAsymmetryDetector()

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

            # Process frame
            result = detector.process_frame(frame)

            # Hiển thị kết quả
            print(f"\r[Frame {frame_count:04d}] Score: {result['score']:.1f} | Status: {result['status']}", end="")

            # Hiển thị (optional)
            cv2.imshow('FGA v5.0 - Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test hoàn tất!")
