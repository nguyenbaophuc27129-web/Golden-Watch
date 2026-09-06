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
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
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

            if use_solutions:
                # Use solutions API (FaceMesh)
                results = self.face_mesh.process(frame_rgb)

                if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
                    return {
                        'score': 0,
                        'nihss_item_4': 0,
                        'status': 'NO_FACE',
                        'raw_metrics': {},
                        'raw_landmarks': None
                    }

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
                    return {
                        'score': 0,
                        'nihss_item_4': 0,
                        'status': 'NO_FACE',
                        'raw_metrics': {},
                        'raw_landmarks': None
                    }

                face_landmarks = result.face_landmarks[0]
                landmarks = self._extract_landmarks(face_landmarks)

            if landmarks is None or len(landmarks) < 478:  # MediaPipe Face Mesh has 478 landmarks
                return {
                    'score': 0,
                    'nihss_item_4': 0,
                    'status': 'INVALID_LANDMARKS',
                    'raw_metrics': {},
                    'raw_landmarks': None
                }

            # Calculate 5 biomedical metrics
            metrics = {
                'mouth_ratio': self._calculate_mouth_asymmetry(landmarks),
                'eye_ratio': self._calculate_eye_deviation(landmarks),
                'face_tilt': self._calculate_face_tilt(landmarks),
                'nasolabial_ratio': self._calculate_nasolabial_asymmetry(landmarks),
                'forehead_ratio': self._calculate_forehead_asymmetry(landmarks)
            }

            # Calculate overall score
            score = self._calculate_overall_score(metrics)

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

            return {
                'score': score,
                'nihss_item_4': nihss_score,
                'status': status,
                'raw_metrics': metrics,
                'raw_landmarks': landmarks
            }

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
