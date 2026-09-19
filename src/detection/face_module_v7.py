"""
FACIAL ASYMMETRY DETECTION MODULE v7.0 - MEDIAPIPE TASKS API
============================================================

FIXES FROM v6.0:
1. [OK] Updated to MediaPipe 1.0.1 Tasks API (solutions API deprecated)
2. [OK] Use RELATIVE measurement (ratio) instead of absolute mm
3. [OK] Proper NIHSS mapping (0-4 scale)
4. [OK] Full raw_landmarks extraction (for ML model integration)

Author: PSCS Team
Date: 2026-08-25
Version: 7.0
"""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from collections import deque
from typing import Dict, Any, Optional, Tuple
import cv2
import math


class FaceAsymmetryDetector:
    """
    Module 1: Facial Asymmetry Detection (PSCS v7.0)

    PHAN TICH 5 CHI SO Y KHOA:
    1. Mouth Asymmetry Ratio (%) - Lech mieng (30%)
    2. Eye Deviation Ratio (%) - Leh nhan (25%)
    3. Face Tilt Angle (do) - Nghieng mat (20%)
    4. Nasolabial Fold Ratio (%) - Nep goc mui-moi (15%)
    5. Forehead Ratio (%) - Nep tran (10%)

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
    - raw_landmarks: 478x2 numpy array (for ML model)
    """

    def __init__(self, history_size: int = 5, model_path: Optional[str] = None):
        """
        Khoi tao detector voi MediaPipe Tasks API

        Args:
            history_size: So frames luu trong bo dem (median filter)
            model_path: Duong dan den file model (.tflite). Neu None, dung default.
        """
        # Bo dem diem so (giam nhieu temporal)
        self.score_history = deque(maxlen=history_size)

        # HOLD: giu ket qua hop le gan nhat <=3s khi mat face thoang qua
        # (nghieng dau/quay mat -> MediaPipe mat 1-2 frame -> khong roi canh bao)
        self.last_valid = None
        self.last_valid_t = 0.0
        self.hold_seconds = 3.0

        # Nguong phat hien (dua tren tai lieu y khoa)
        self.thresholds = {
            "mouth_ratio": 0.25,      # 25% asymmetry
            "eye_ratio": 0.30,         # 30% asymmetry
            "face_tilt": 10.0,         # 10 degrees
            "nasolabial_ratio": 0.35,  # 35% asymmetry
            "forehead_ratio": 0.30      # 30% asymmetry
        }

        # Trong so tung metric (tong = 1.0)
        self.weights = {
            "mouth_ratio": 0.30,
            "eye_ratio": 0.25,
            "face_tilt": 0.20,
            "nasolabial_ratio": 0.15,
            "forehead_ratio": 0.10
        }

        # Explicit initialization of API mode
        self.use_solutions_api = False
        self.landmarker = None
        self.face_mesh = None

        # ML v3 (SYS-29): gán từ ngoài app — FACE.ml_model = FaceMLV3...
        # Khi có model, prob ML THAY prob rules (SYS-15); rules giữ lại
        # trong raw_metrics['score_rules'] để so sánh.
        self.ml_model = None

        # Khoi tao MediaPipe Face Landmarker voi Tasks API
        try:
            # Try to use face_landmarker.task file from project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            # Check multiple possible task file locations
            # IMPORTANT: Use real task files (3MB+), skip stub files
            possible_task_files = [
                os.path.join(project_root, "models", "face_landmarker_v2", "face_landmarker_v2_with_blendshapes.task"),
                os.path.join(project_root, "src", "models", "face_landmarker.task"),
                os.path.join(project_root, "models", "face_landmarker_v2", "face_landmarker.task"),
                os.path.join(project_root, "face_landmarker.task"),
            ]

            task_file = None
            for file_path in possible_task_files:
                if os.path.exists(file_path):
                    # Check file size - real task files are 3MB+, stub files are <100KB
                    file_size = os.path.getsize(file_path)
                    if file_size > 1000000:  # At least 1MB
                        task_file = file_path
                        print(f"[FACE] Found task file: {file_path} ({file_size/1024/1024:.2f} MB)")
                        break

            if task_file:
                # Use the task file
                options = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=task_file),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    min_face_detection_confidence=0.3,
                    min_face_presence_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                print("[OK] MediaPipe FaceLandmarker v7.0 initialized (with task file)")
            else:
                print("[FACE] No valid task file found, trying solutions API...")
                raise FileNotFoundError("No valid task file found")
        except Exception as e:
            print(f"[FACE] Tasks API failed: {e}")
            print("[FALLBACK] Face detection unavailable - numpy-based mode (no detection)")
            self.use_solutions_api = False

    def _extract_landmarks(self, face_landmarks) -> np.ndarray:
        """
        Extract landmarks tu MediaPipe Tasks API result

        Args:
            face_landmarks: FaceLandmarker result

        Returns:
            np.ndarray: (478, 2) array of (x, y) coordinates normalized [0, 1]
        """
        if face_landmarks is None:
            return None

        # MediaPipe Tasks API returns 478 landmarks
        landmarks = []
        for landmark in face_landmarks:
            x = landmark.x  # Already normalized [0, 1]
            y = landmark.y  # Already normalized [0, 1]
            z = landmark.z  # Relative depth
            landmarks.append([x, y])

        return np.array(landmarks, dtype=np.float32)

    def _calculate_mouth_asymmetry(self, landmarks: np.ndarray) -> float:
        """
        Metric 1: MEM MONG - 30% weight

        Metric 1: Mouth Asymmetry using RELATIVE ratio measurement

        THUAT TOAN:
        1. Lay 6 landmark: Left corner (61), Right corner (291),
           Upper lip center (13), Lower lip center (14),
           Left upper (61), Right upper (291)
        2. Tinh khoang cach tu moi corner den center upper lip
        3. Tinh ratio = left_dist / right_dist
        4. Asymmetry = |ratio - 1.0| (percentage)

        Nguong: 25% asymmetry (dua tren Ross et al. 2021)

        Args:
            landmarks: (478, 2) array

        Returns:
            float: Asymmetry percentage (0-100)
        """
        try:
            # Left corner: 61, Right corner: 291
            left_corner = landmarks[61]
            right_corner = landmarks[291]

            # Upper lip center: 13
            upper_lip_center = landmarks[13]

            # Tinh khoang cach tu moi corner den upper lip center
            left_dist = np.linalg.norm(left_corner - upper_lip_center)
            right_dist = np.linalg.norm(right_corner - upper_lip_center)

            # Tinh ratio (SCALE-INVARIANT!)
            if right_dist > 0:
                ratio = left_dist / right_dist
                asymmetry = abs(ratio - 1.0)  # Percentage deviation
                return min(asymmetry * 100, 100)  # Cap at 100%
            return 0.0
        except Exception as e:
            print(f"[WARNING] Error calculating mouth asymmetry: {e}")
            return 0.0

    def _calculate_eye_deviation(self, landmarks: np.ndarray) -> float:
        """
        Metric 2: LECH NHAN - 25% weight

        Metric 2: Eye Deviation using RELATIVE ratio measurement

        THUAT TOAN:
        1. Lay 4 landmark: Left eye corner (33), Right eye corner (263),
           Nose bridge (6), Nose tip (1)
        2. Tinh khoang cach tu moi eye corner den nose bridge
        3. Tinh ratio = left_dist / right_dist
        4. Deviation = |ratio - 1.0|

        Nguong: 30% deviation (dua tren Kim et al. 2020)

        Args:
            landmarks: (478, 2) array

        Returns:
            float: Deviation percentage (0-100)
        """
        try:
            # Left eye corner: 33, Right eye corner: 263
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            # Nose bridge: 6
            nose_bridge = landmarks[6]

            # Tinh khoang cach tu moi eye den nose bridge
            left_dist = np.linalg.norm(left_eye - nose_bridge)
            right_dist = np.linalg.norm(right_eye - nose_bridge)

            # Tinh ratio
            if right_dist > 0:
                ratio = left_dist / right_dist
                deviation = abs(ratio - 1.0)
                return min(deviation * 100, 100)
            return 0.0
        except Exception as e:
            print(f"[WARNING] Error calculating eye deviation: {e}")
            return 0.0

    def _calculate_face_tilt(self, landmarks: np.ndarray) -> float:
        """
        Metric 3: NHIENG MAT - 20% weight

        Metric 3: Face Tilt Angle (SCALE-INVARIANT by nature!)

        THUAT TOAN:
        1. Lay 2 landmark: Center of forehead (10), Chin center (152)
        2. Tinh vector tu forehead den chin
        3. Tinh angle so voi vertical (truc Y)
        4. Angle = arctan2(dx, dy)

        Nguong: 10 degrees (dua tren Facial Palsy Grading Scale 2019)

        Args:
            landmarks: (478, 2) array

        Returns:
            float: Tilt angle in degrees (0-90)
        """
        try:
            # Forehead center: 10, Chin center: 152
            forehead = landmarks[10]
            chin = landmarks[152]

            # Vector tu forehead den chin
            dx = chin[0] - forehead[0]
            dy = chin[1] - forehead[1]

            # Tinh angle so voi vertical
            angle_rad = math.atan2(abs(dx), dy)  # dx, dy is already normalized
            angle_deg = math.degrees(angle_rad)

            return min(angle_deg, 90)
        except Exception as e:
            print(f"[WARNING] Error calculating face tilt: {e}")
            return 0.0

    def _calculate_nasolabial_asymmetry(self, landmarks: np.ndarray) -> float:
        """
        Metric 4: NEP GOC MUI-MOI - 15% weight

        Metric 4: Nasolabial Fold Asymmetry using RELATIVE ratio

        THUAT TOAN:
        1. Lay 4 landmark: Left nasolabial (205), Right nasolabial (425),
           Nose base (1), Mouth center (0)
        2. Tinh khoang cach tu moi nasolabial den nose base
        3. Tinh ratio va asymmetry

        Nguong: 35% asymmetry

        Args:
            landmarks: (478, 2) array

        Returns:
            float: Asymmetry percentage (0-100)
        """
        try:
            # Left nasolabial: 205, Right nasolabial: 425
            left_nasolabial = landmarks[205]
            right_nasolabial = landmarks[425]

            # Nose base: 1
            nose_base = landmarks[1]

            # Tinh khoang cach
            left_dist = np.linalg.norm(left_nasolabial - nose_base)
            right_dist = np.linalg.norm(right_nasolabial - nose_base)

            # Tinh ratio
            if right_dist > 0:
                ratio = left_dist / right_dist
                asymmetry = abs(ratio - 1.0)
                return min(asymmetry * 100, 100)
            return 0.0
        except Exception as e:
            print(f"[WARNING] Error calculating nasolabial asymmetry: {e}")
            return 0.0

    def _calculate_forehead_asymmetry(self, landmarks: np.ndarray) -> float:
        """
        Metric 5: NEP TRAN - 10% weight

        Metric 5: Forehead Wrinkle Asymmetry using RELATIVE ratio

        THUAT TOAN:
        1. Lay 2 landmark tren tran: Left forehead (70), Right forehead (300)
        2. So sanh voi center forehead (10)
        3. Tinh ratio asymmetry

        Nguong: 30% asymmetry

        Args:
            landmarks: (478, 2) array

        Returns:
            float: Asymmetry percentage (0-100)
        """
        try:
            # Left forehead: 70, Right forehead: 300
            left_forehead = landmarks[70]
            right_forehead = landmarks[300]

            # Center forehead: 10
            center_forehead = landmarks[10]

            # Tinh khoang cach
            left_dist = np.linalg.norm(left_forehead - center_forehead)
            right_dist = np.linalg.norm(right_forehead - center_forehead)

            # Tinh ratio
            if right_dist > 0:
                ratio = left_dist / right_dist
                asymmetry = abs(ratio - 1.0)
                return min(asymmetry * 100, 100)
            return 0.0
        except Exception as e:
            print(f"[WARNING] Error calculating forehead asymmetry: {e}")
            return 0.0

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Tinh tong score dua tren 5 metrics va weights

        THUAT TOAN:
        1. So sanh moi metric voi nguong
        2. Neu vuot nguong -> tinh diem phan tram vuot
        3. Tong hop voi weights

        Args:
            metrics: Dict voi 5 biomedical metrics

        Returns:
            float: Score 0-100
        """
        total_score = 0.0

        # Mouth
        mouth_val = metrics.get("mouth_ratio", 0) / 100  # Convert to percentage
        if mouth_val > self.thresholds["mouth_ratio"]:
            excess = (mouth_val - self.thresholds["mouth_ratio"]) / self.thresholds["mouth_ratio"]
            total_score += excess * 100 * self.weights["mouth_ratio"]

        # Eye
        eye_val = metrics.get("eye_ratio", 0) / 100
        if eye_val > self.thresholds["eye_ratio"]:
            excess = (eye_val - self.thresholds["eye_ratio"]) / self.thresholds["eye_ratio"]
            total_score += excess * 100 * self.weights["eye_ratio"]

        # Face tilt
        tilt_val = metrics.get("face_tilt", 0)
        if tilt_val > self.thresholds["face_tilt"]:
            excess = (tilt_val - self.thresholds["face_tilt"]) / self.thresholds["face_tilt"]
            total_score += excess * 100 * self.weights["face_tilt"]

        # Nasolabial
        naso_val = metrics.get("nasolabial_ratio", 0) / 100
        if naso_val > self.thresholds["nasolabial_ratio"]:
            excess = (naso_val - self.thresholds["nasolabial_ratio"]) / self.thresholds["nasolabial_ratio"]
            total_score += excess * 100 * self.weights["nasolabial_ratio"]

        # Forehead
        forehead_val = metrics.get("forehead_ratio", 0) / 100
        if forehead_val > self.thresholds["forehead_ratio"]:
            excess = (forehead_val - self.thresholds["forehead_ratio"]) / self.thresholds["forehead_ratio"]
            total_score += excess * 100 * self.weights["forehead_ratio"]

        return min(total_score, 100)

    def _map_to_nihss_item_4(self, score: float) -> int:
        """
        Mapping score 0-100 -> NIHSS Item 4 (0-4)

        Dua tren clinical judgment:
        - Score 0-10: Normal (NIHSS 0)
        - Score 11-35: Minor (NIHSS 1)
        - Score 36-60: Moderate (NIHSS 2)
        - Score 61-85: Severe (NIHSS 3)
        - Score 86-100: Complete (NIHSS 4)

        Args:
            score: float 0-100

        Returns:
            int: NIHSS Item 4 score (0-4)
        """
        if score <= 10:
            return 0
        elif score <= 35:
            return 1
        elif score <= 60:
            return 2
        elif score <= 85:
            return 3
        else:
            return 4

    def _no_face(self, status: str) -> Dict[str, Any]:
        """
        Tra ket qua HOLD (ket qua hop le gan nhat <=3s) hoac status mat face.
        - HOLD: mat face thoang qua (nghieng dau, cham quay mat) -> khong lam
          roi canh bao dang co; danh dau 'hold': True de trung thuc.
        - Qua 3s: tra status goc (NO_FACE/...) voi score 0.
        """
        import time as _time
        if (self.last_valid is not None
                and _time.time() - self.last_valid_t <= self.hold_seconds):
            out = dict(self.last_valid)
            out['hold'] = True
            out['hold_age_s'] = round(_time.time() - self.last_valid_t, 1)
            return out
        return {'score': 0, 'nihss_item_4': 0, 'status': status,
                'raw_metrics': {}, 'raw_landmarks': None, 'hold': False}

    @staticmethod
    def _head_pose_from_matrix(matrix) -> Optional[Dict[str, float]]:
        """
        Goc chuyen dau (do) TU MA TRAN 4x4 CHINH THUC cua FaceLandmarker
        (bat bang output_facial_transformation_matrixes=True):
        - roll  = nghieng dau sang vai (so khop 'face_tilt' cu)
        - yaw   = quay mat trai/phai
        - pitch = ngua/cup dau
        Cong thuc chuan tu tai lieu MediaPipe Face Landmarker:
        developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
        LUU Y dau: theo he toa do cua ma tran (mat chuan -> mat that),
        do lon goc LUON dung; dau am/duong chi la quy uoc chieu.
        Da kiem chung bang ma tran quay tong hop:
        quay quanh Y -> yaw (quay mat), quanh X -> pitch (nga/cup),
        quanh Z -> roll (nghieng vai) — sai so < 1 do.
        """
        try:
            R = np.asarray(matrix, dtype=np.float64)[:3, :3]
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 1]))))
            roll = math.degrees(math.atan2(R[0, 1], R[1, 1]))
            yaw = math.degrees(math.atan2(R[2, 0], R[2, 2]))
            return {'head_yaw': round(yaw, 1), 'head_pitch': round(pitch, 1),
                    'head_roll': round(roll, 1)}
        except Exception:
            return None

    @staticmethod
    def _blend_asym(blendshapes) -> Dict[str, float]:
        """
        20 cặp L−R từ 52 blendshape CHÍNH THỨC (cùng quy tắc với
        training/train_face_landmarker_v3.py): asym_<ten> = Left − Right.
        Đây là 20/28 đặc trưng của model ML v3.
        """
        names = [c.category_name for c in blendshapes]
        scores = {c.category_name: float(c.score) for c in blendshapes}
        out = {}
        for n in names:
            if n.endswith('Left') and (n[:-4] + 'Right') in scores:
                out['asym_' + n[:-4]] = scores[n] - scores[n[:-4] + 'Right']
        return out

    def _detect_expression(self, landmarks: np.ndarray) -> str:
        """
        Phân biệt BIỂU CẢM (cười lớn/nhech mép khi nói) với méo mặt đột quỵ:
        - Cười: miệng MỞ (khoảng cách môi trên 13 - môi dưới 14 lớn) VÀ
          2 khóe miệng NGANG/HƠNG lên so với tâm miệng (đối xứng theo trục dọc).
        - Méo đột quỵ: 1 khóe TUỘT XUỐNG (không đối xứng dọc) khi miệng đóng.
        Trả về 'NEUTRAL' | 'SMILE_LAUGH'.
        """
        try:
            upper = landmarks[13][1]     # môi trên
            lower = landmarks[14][1]     # môi dưới
            face_h = np.linalg.norm(landmarks[152] - landmarks[10])
            if face_h <= 0:
                return 'NEUTRAL'
            mouth_open = abs(lower - upper) / face_h
            l_c, r_c = landmarks[61], landmarks[291]
            mouth_cy = (upper + lower) / 2.0
            # khóe "hông lên" = y nhỏ hơn tâm miệng (trục y hướng xuống)
            corners_up = min(l_c[1], r_c[1]) < mouth_cy - 0.010
            if mouth_open > 0.08 and corners_up:
                return 'SMILE_LAUGH'
            return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'

    def process_frame(self, frame: np.ndarray, frame_timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Xu ly 1 frame de phat hien asymmetry

        Args:
            frame: np.ndarray (H, W, 3) BGR image
            frame_timestamp_ms: Optional[int] timestamp in milliseconds (auto-generated if None)

        Returns:
            Dict: {
                'score': float 0-100,
                'nihss_item_4': int 0-4,
                'status': str 'NORMAL'/'WARNING'/'DANGER',
                'raw_metrics': Dict[str, float],
                'raw_landmarks': np.ndarray (478, 2)
            }
        """
        # Check which API is being used
        use_solutions = hasattr(self, 'use_solutions_api') and self.use_solutions_api
        has_landmarker = hasattr(self, 'landmarker') and self.landmarker is not None

        if not use_solutions and not has_landmarker:
            return {
                'score': 0,
                'nihss_item_4': 0,
                'status': 'NO_DETECTOR',
                'raw_metrics': {},
                'raw_landmarks': None
            }

        try:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            head_pose = None   # dict yaw/pitch/roll (chi co tren Tasks API)
            bs_asym = None     # dict asym blendshape L−R (20 dac trung ML)

            if use_solutions:
                # Use solutions API (FaceMesh)
                results = self.face_mesh.process(frame_rgb)

                if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
                    return self._no_face('NO_FACE')

                # Extract landmarks from solutions API
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                landmarks = np.array(landmarks, dtype=np.float32)

            else:
                # Use Tasks API with mp.Image class
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # For VIDEO mode, use detect_for_video with timestamp
                if frame_timestamp_ms is None:
                    import time
                    frame_timestamp_ms = int(time.time() * 1000)
                result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                if not result or result.face_landmarks is None or len(result.face_landmarks) == 0:
                    return self._no_face('NO_FACE')

                face_landmarks = result.face_landmarks[0]
                landmarks = self._extract_landmarks(face_landmarks)

                # Ma tran tu the dau 4x4 (API chinh thuc MediaPipe) ->
                # yaw (quay) / pitch (nga-cup) / roll (nghieng) theo do
                mats = getattr(result, 'facial_transformation_matrixes',
                               None)
                if mats:
                    head_pose = self._head_pose_from_matrix(mats[0])
                if getattr(result, 'face_blendshapes', None):
                    bs_asym = self._blend_asym(result.face_blendshapes[0])

            if landmarks is None or len(landmarks) < 478:  # MediaPipe Face Mesh has 478 landmarks
                return self._no_face('INVALID_LANDMARKS')

            # GUARD 1 — mặt bị khuất một phần (nằm/nửa người/quay đi):
            # bbox landmark chạm mép khung → KHÔNG chấm điểm (tránh đoán sai)
            xs, ys = landmarks[:, 0], landmarks[:, 1]
            if (xs.min() < 0.02 or xs.max() > 0.98
                    or ys.min() < 0.02 or ys.max() > 0.98):
                return self._no_face('PARTIAL_FACE')

            # Calculate 5 biomedical metrics
            metrics = {
                'mouth_ratio': self._calculate_mouth_asymmetry(landmarks),
                'eye_ratio': self._calculate_eye_deviation(landmarks),
                'face_tilt': self._calculate_face_tilt(landmarks),
                'nasolabial_ratio': self._calculate_nasolabial_asymmetry(landmarks),
                'forehead_ratio': self._calculate_forehead_asymmetry(landmarks)
            }

            # GUARD 2 — đang cười lớn: giảm mạnh metric BIỂU CẢM (miệng, rãnh
            # mũi-má, trán) vì khóe hông lên là ĐỐI XỨNG; giữ mắt + nghiêng
            metrics['expression'] = self._detect_expression(landmarks)
            if metrics['expression'] == 'SMILE_LAUGH':
                for k in ('mouth_ratio', 'nasolabial_ratio',
                          'forehead_ratio'):
                    metrics[k] *= 0.4

            # Góc chuyển đầu từ ma trận chính thức MediaPipe (chỉ HIỂN THỊ,
            # không vào điểm — 3 metric cũ giữ nguyên công thức)
            if head_pose:
                metrics.update(head_pose)
                # key ngắn khớp artifact ML v3: yaw/pitch/roll
                metrics.update({'yaw': head_pose['head_yaw'],
                                'pitch': head_pose['head_pitch'],
                                'roll': head_pose['head_roll'],
                                'pose_yaw': head_pose['head_yaw'],
                                'pose_pitch': head_pose['head_pitch'],
                                'pose_roll': head_pose['head_roll']})
            if bs_asym:
                metrics.update(bs_asym)     # 20 asym — đầu vào ML v3
                if len(bs_asym) == 20:
                    # Alias generic khớp artifact ML v3 (blend_asym_00..19)
                    # — cùng thứ tự canonical 52 blendshape như lúc train
                    # (NK-26: trước đây tên key lệch → predict luôn None,
                    # ML v3 không bao giờ chấm trong app)
                    for _i, _v in enumerate(bs_asym.values()):
                        metrics[f'blend_asym_{_i:02d}'] = _v

            # Calculate overall score (rules — 5 ratio, khong doi cong thuc)
            score = self._calculate_overall_score(metrics)
            metrics['score_rules'] = round(score, 1)   # giữ lại so sánh

            # SYS-29: prob ML v3 THAY prob rules khi có model + đủ đặc trưng
            if self.ml_model is not None:
                ml_prob = self.ml_model.predict(metrics)
                if ml_prob is not None:
                    metrics['ml_prob'] = round(ml_prob, 1)
                    score = ml_prob

            # Apply temporal filtering (median)
            self.score_history.append(score)
            if len(self.score_history) >= 3:
                score = np.median(self.score_history)

            # Map to NIHSS
            nihss_score = self._map_to_nihss_item_4(score)

            # Determine status
            if nihss_score == 0:
                status = 'NORMAL'
            elif nihss_score <= 2:
                status = 'WARNING'
            else:
                status = 'DANGER'

            result = {
                'score': score,
                'nihss_item_4': nihss_score,
                'status': status,
                'raw_metrics': metrics,
                'raw_landmarks': landmarks,
                'hold': False
            }
            import time as _time
            self.last_valid = result
            self.last_valid_t = _time.time()
            return result

        except Exception as e:
            print(f"[ERROR] Exception in process_frame: {e}")
            import traceback
            traceback.print_exc()
            return {
                'score': 0,
                'nihss_item_4': 0,
                'status': 'ERROR',
                'raw_metrics': {},
                'raw_landmarks': None
            }


# ==================== TEST MODE ====================
if __name__ == "__main__":
    print("=" * 60)
    print("FACE ASYMMETRY DETECTION v7.0 - TEST MODE")
    print("=" * 60)
    print()

    # Initialize detector
    print("Initializing detector...")
    detector = FaceAsymmetryDetector()
    print()

    # Test with webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        exit(1)

    print("[OK] Webcam opened successfully")
    print()
    print("=" * 60)
    print("CONTROLS:")
    print("  [q] - Quit")
    print("  [s] - Save current frame with landmarks")
    print("=" * 60)
    print()
    print("Starting detection... (Press 'q' to quit)")
    print()

    frame_count = 0
    frame_timestamp = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Cannot read frame")
            break

        # Process frame (timestamp in milliseconds)
        frame_timestamp += 33  # ~30 FPS
        result = detector.process_frame(frame, frame_timestamp)

        # Display results
        h, w = frame.shape[:2]

        # Draw status box
        status_color = {
            'NORMAL': (0, 255, 0),
            'WARNING': (0, 165, 255),
            'DANGER': (0, 0, 255),
            'NO_FACE': (128, 128, 128),
            'NO_DETECTOR': (255, 0, 255),
            'ERROR': (255, 0, 0)
        }.get(result['status'], (255, 255, 255))

        cv2.rectangle(frame, (10, 10), (400, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 220), status_color, 2)

        y = 30
        cv2.putText(frame, f"Status: {result['status']}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        y += 30

        cv2.putText(frame, f"Score: {result['score']:.1f}/100", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

        cv2.putText(frame, f"NIHSS Item 4: {result['nihss_item_4']}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30

        # Show metrics
        cv2.putText(frame, "=== METRICS ===", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25

        if result['raw_metrics']:
            for name, value in result['raw_metrics'].items():
                color = (255, 100, 100) if value > detector.thresholds.get(name, 0) * 100 else (100, 255, 100)
                cv2.putText(frame, f"{name}: {value:.1f}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 20
        else:
            cv2.putText(frame, "No metrics available", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Show FPS
        cv2.putText(frame, f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Face Asymmetry Detection v7.0', frame)

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and result['raw_landmarks'] is not None:
            # Save frame
            filename = f"frame_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"[SAVE] Saved frame: {filename}")
            frame_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print()
    print("=" * 60)
    print("Next steps:")
    print("  1. Train ML model with train_face_model_rtx3050.py")
    print("  2. Get clinical validation letter from doctor")
    print("  3. Create Scientific Basis Table Excel")
    print("=" * 60)
