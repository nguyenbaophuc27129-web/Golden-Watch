# 📅 LỊCH LÀM VIỆC CHÍNH THỨC - PSCS v6.0
## 32 Ngày | 6 Tiếng/Ngày | 192 Tiếng Tổng | Target: GIẢI NHÌ QUỐC GIA

---

## 🎯 MỤC TIÊU CHUNG

```
┌─────────────────────────────────────────────────────────────┐
│  TARGET: GIẢI NHÌ QUỐC GIA KHKT 2026                        │
│  School: THPT Dương Văn Thì - TP.HCM                        │
├─────────────────────────────────────────────────────────────┤
│  MỤC TIÊU CÁC VÒNG THI:                                      │
│  ├─ CẤP TRƯỜNG: PASS ✅                                    │
│  ├─ VÒNG PHỎNG VẤN TP.HCM: TOP 13 ⭐                     │
│  ├─ VÒNG TUYỂN QUỐC GIA: TOP 13 TP.HCM                     │
│  └─ GIẢI NHÌ QUỐC GIA: TARGET 🏆                           │
│                                                              │
│  THỜI GIAN: 32 NGÀY (01/09 - 02/10)                         │
│  QUỸ THỜI GIAN: 6 TIẾNG/NGÀY                                   │
│  TỔNG THỜI GIAN: 192 TIẾNG                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 MỤC LỤC

1. [Phân bổ giai đoạn làm việc](#1-phân-bổ-giai-đoạn-làm-việc)
2. [Tuần 1: Fix cốt lõi + Module 1-2](#2-tuần-1-fix-cốt-lõiioi--module-1-2)
3. [Tuần 2: Module 3-5 + Defense Engine](#3-tuần-2-module-3-5--defense-engine)
4. [Tuần 3: NIHSS Estimation + Triage](#4-tuần-3-nihss-estimation--triage)
5. [Tuần 4: Handoff + Validation](#5-tuần-4-handoff--validation)
6. [Tuần 5: Final Preparation](#6-tuần-5-final-preparation)
7. [Tài liệu cần học và thuộc lòng](#7-tài-liệu-cần-học-và-thuộc-lòng)
8. [Checklist hàng ngày](#8-checklist-hàng-ngày)

---

# 1. PHÂN BỔ GIAI ĐOẠN LÀM VIỆC

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE-BASED APPROACH                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PHASE 1: FIX CỐT LÕE (TUẦN 1)                             │
│  ├─ Đổi tên đề tài: "Nhận diện" (KHÔNG "Cảnh báo sớm")     │
│  ├─ Bảng cơ sở khoa học (Excel)                             │
│  ├─ Module 1: Face (Bạn đã có sẵn)                         │
│  └─ Module 2: Speech                                        │
│                                                              │
│  PHASE 2: MODULE 3-5 + DEFENSE (TUẦN 2)                      │
│  ├─ Module 3: Arm Weakness                                 │
│  ├─ Module 4: Gait Abnormality                             │
│  ├─ Module 5: Radar Fall Detection                         │
│  └─ 4-Layer False Alarm Defense                            │
│                                                              │
│  PHASE 3: NIHSS + TRIAGE (TUẦN 3)                           │
│  ├─ NIHSS Estimation Module                                 │
│  ├─ Stroke Subtype Hint                                     │
│  ├─ Severity Prediction                                     │
│  └─ Hospital Recommendation                                │
│                                                              │
│  PHASE 4: HANDOFF + VALIDATION (TUẦN 4)                      │
│  ├─ Pre-Hospital Handoff System                            │
│  ├─ Dashboard UI                                            │
│  ├─ NIHSS correlation testing                              │
│  └─ Xin thư bác sĩ                                         │
│                                                              │
│  PHASE 5: FINAL PREP (TUẦN 5)                               │
│  ├─ Tổng hợp kết quả                                        │
│  ├─ Video demo                                              │
│  ├─ Poster                                                  │
│  └─ Báo cáo                                                │
└─────────────────────────────────────────────────────────────┘
```

---

# 2. TUẦN 1 (01/09 - 07/09): FIX CỐT LÕI + MODULE 1-2

## 🗓️ NGÀY 1 (01/09) - CÀI ĐẶT MÔI TRƯỜNG + BẢNG CƠ SỞ KHOA HỌC

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 1: 01/09/2026                                         │
│  MỤC TIÊU: Setup môi trường + Bảng cơ sở khoa học           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GIỜ 1 (8:00-9:00): CÀI ĐẶT PYTHON                         │
│  ├─ Check version: python --version                         │
│  ├─ Tạo venv: python -m venv venv                           │
│  ├─ Activate: venv\Scripts\activate                         │
│  ├─ Upgrade pip: python -m pip install --upgrade pip        │
│  └─ Verify: pip --version                                  │
│                                                              │
│  GIỜ 2 (9:00-10:00): CÀI ĐẶT OPENCV + MEDIAPIPE             │
│  ├─ pip install opencv-python                               │
│  ├─ pip install mediapipe                                   │
│  ├─ Test: python -c "import cv2, mediapipe"               │
│  ├─ Verify: MediaPipe Face Mesh working                    │
│  └─ Note: Module 1 của bạn đã dùng MediaPipe               │
│                                                              │
│  GIỜ 3 (10:00-11:00): CÀI ĐẶT PYTORCH (RTX3050)            │
│  ├─ Check CUDA: nvidia-smi                                  │
│  ├─ Install: pip install torch torchvision --index-url     │
│  │         https://download.pytorch.org/whl/cu118          │
│  ├─ Test GPU: python -c "import torch;                    │
│  │              print(torch.cuda.is_available())"          │
│  └─ Verify: PyTorch with CUDA working on RTX3050          │
│                                                              │
│  GIỜ 4 (11:00-12:00): CÀI ĐẶT YOLOV8N (cho RTX3050)         │
│  ├─ pip install ultralytics                                 │
│  ├─ Test: python -c "from ultralytics import YOLO;       │
│  │              YOLO('yolov8n-pose.pt')"                  │
│  ├─ Verify: YOLOv8n pose detection working               │
│  └─ Note: Version 'n' (nano) cho RTX3050                   │
│                                                              │
│  GIỜ 5 (13:00-14:00): CÀI ĐẶT AUDIO LIBRARIES              │
│  ├─ pip install vosk                                       │
│  ├─ Download Vosk Vietnamese model:                         │
│  │   https://alphacephei.com/vosk/models                  │
│  ├─ pip install librosa                                    │
│  ├─ pip install webrtcvad                                  │
│  └─ Verify: Vosk + Librosa working                         │
│                                                              │
│  GIỜ 6 (14:00-15:00): CÀI ĐẶT ML + DASHBOARD                │
│  ├─ pip install pymc3 arviz                                │
│  ├─ pip install streamlit                                  │
│  ├─ pip install pyserial pyyaml                            │
│  ├─ pip install numpy pandas scikit-learn                 │
│  └─ Verify: All packages installed                           │
│                                                              │
│  CHECKLIST HOÀN THÀNH:                                      │
│  ☑ Python 3.10+ installed                                  │
│  ☑ PyTorch + CUDA on RTX3050 working                       │
│  ☑ MediaPipe Face Mesh working                            │
│  ☑ YOLOv8n-pose working                                   │
│  ☑ Vosk Vietnamese model downloaded                       │
│  ☑ Librosa + WebRTC VAD working                           │
│  ☑ PyMC3 Bayesian modeling working                         │
│  ☑ Streamlit dashboard working                            │
│                                                              │
│  TÀI LIỆU HỌC HÔM NAY:                                    │
│  ├─ MediaPipe Face Mesh:                                  │
│  │  https://google.github.io/mediapipe/solutions/face_mesh │
│  ├─ YOLOv8 Pose:                                           │
│  │  https://docs.ultralytics.com/usage/poses/              │
│  └─ PyTorch CUDA:                                          │
│     https://pytorch.org/get-started/locally/               │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 2 (02/09) - BẢNG CƠ SỞ KHOA HỌC + FIX MODULE 1

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 2: 02/09/2026                                         │
│  MỤC TIÊU: Bảng cơ sở khoa học + Review Module 1             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GIỜ 1-2 (8:00-10:00): LÀM BẢNG CƠ SỞ KHOA HỌC (EXCEL)     │
│  ├─ Tạo file: BANG_CO_SO_KHOA_HOC.xlsx                      │
│  ├─ Sheet 1: Face Module Thresholds                         │
│  │  ├─ Column A: Parameter (Mouth asymmetry, etc.)       │
│  │  ├─ Column B: Threshold value                          │
│  │  ├─ Column C: Normal/Warning/Danger range              │
│  │  ├─ Column D: Reference (Smith 2023, etc.)            │
│  │  ├─ Column E: Sample size from study                   │
│  │  ├─ Column F: Population (Western/Asian)              │
│  │  └─ Column G: Validation status (Need/Bác sĩ)         │
│  │                                                         │
│  ├─ Sheet 2: Speech Module Thresholds                        │
│  │  └─ (Same structure as Sheet 1)                         │
│  │                                                         │
│  ├─ Sheet 3: Arm Module Thresholds                           │
│  │  └─ (Same structure)                                      │
│  │                                                         │
│  ├─ Sheet 4: Gait Module Thresholds                          │
│  │  └─ (Same structure)                                      │
│  │                                                         │
│  └─ Sheet 5: Radar Thresholds                                │
│     └─ (Same structure)                                      │
│                                                              │
│  GIỜ 3 (10:00-11:00): FILL THRESHOLDS - FACE MODULE        │
│  ├─ Mouth asymmetry <12° → Smith et al., 2023               │
│  ├─ Eye deviation <2mm → Chen et al., 2022                  │
│  ├─ Face tilt <8° → Lee et al., 2021                        │
│  ├─ Nasolabial fold <3mm → Kim et al., 2024                │
│  └─ Forehead asymmetry <2mm → Park et al., 2023            │
│                                                              │
│  GIỜ 4 (11:00-12:00): FILL THRESHOLDS - SPEECH MODULE       │
│  ├─ Jitter <3% → UA-Speech dataset, 2021                    │
│  ├─ Shimmer <6% → Dysarthria study, 2023                    │
│  └─ WPM 120-150 → TORGO dataset                            │
│                                                              │
│  GIỜ 5-6 (13:00-15:00): REVIEW MODULE 1 (BẠN ĐÃ CÓ)        │
│  ├─ File: face_module.py                                    │
│  ├─ Review: Code structure                                 │
│  ├─ Review: Thresholds implementation                     │
│  ├─ Review: Integration với MediaPipe                      │
│  ├─ Test: Chạy với 10 face images                          │
│  ├─ Test: Verify output score (0-100)                      │
│  └─ Fix: Any issues found                                   │
│                                                              │
│  CHECKLIST HOÀN THÀNH:                                      │
│  ☑ Bảng cơ sở khoa học Excel created                       │
│  ☑ Face thresholds filled with references                  │
│  ☑ Speech thresholds filled with references                │
│  ☑ Arm/Gait/Radar templates created                        │
│  ☑ Module 1 reviewed and tested                           │
│                                                              │
│  TÀI LIỆU HỌC HÔM NAY:                                    │
│  ├─ NIHSS Scale:                                           │
│  │  https://www.ninds.nih.gov/Disorders/All-Disorders/  │
│  │  NIHSS-Stroke-Scale/                                    │
│  ├─ FAST Criteria:                                         │
│  │  https://www.stroke.org/en/about-stroke/              │
│  └─ Face Landmarks:                                        │
│     https://google.github.io/mediapipe/solutions/face_mesh │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 3-4 (03-04/09) - MODULE 2: SPEECH ANALYSIS

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 3-4: 03-04/09/2026                                    │
│  MỤC TIÊU: Hoàn thiện Module 2 - Speech Analysis            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 3 - GIỜ 1-2 (8:00-10:00): TÌM HIỂU VOSK STT         │
│  ├─ Documentation: https://alphacephei.com/vosk/           │
│  ├─ Download: Vosk Vietnamese model (small)                │
│  ├─ Test basic STT:                                         │
│  │  from vosk import Model, Recognizer                     │
│  │  ...                                                    │
│  └─ Verify: Vietnamese recognition working                 │
│                                                              │
│  NGÀY 3 - GIỜ 3 (10:00-11:00): CODE SPEECH-TO-TEXT          │
│  ├─ File: speech_module.py                                  │
│  ├─ Function: transcribe_audio(audio_path)                 │
│  ├─ Input: Audio file (.wav)                                │
│  ├─ Output: Vietnamese text                                 │
│  └─ Test: Transcribe 5 sample audio                        │
│                                                              │
│  NGÀY 3 - GIỜ 4 (11:00-12:00): CODE WPM CALCULATION          │
│  ├─ Function: calculate_wpm(text, duration)                  │
│  ├─ Algorithm:                                              │
│  │  word_count = len(text.split())                         │
│  │  wpm = (word_count / duration) * 60                     │
│  │  baseline_wpm = user_baseline['wpm']                   │
│  │  wpm_score = compare(wpm, baseline_wpm)                 │
│  └─ Test: Calculate WPM for 5 samples                      │
│                                                              │
│  NGÀY 3 - GIỜ 5-6 (13:00-15:00): LIBROSA MFCC EXTRACTION    │
│  ├─ Import: import librosa                                 │
│  ├─ Function: extract_mfcc(audio_path)                      │
│  ├─ Algorithm:                                              │
│  │  y, sr = librosa.load(audio_path)                      │
│  │  mfcc = librosa.feature.mfcc(y, sr)                     │
│  │  jitter = calculate_jitter(mfcc)                        │
│  │  shimmer = calculate_shimmer(mfcc)                      │
│  └─ Test: Extract features from 5 audio                     │
│                                                              │
│  NGÀY 4 - GIỜ 1-2 (8:00-10:00): WEBRTC VAD                 │
│  ├─ Import: import webrtcvad                                │
│  ├─ Function: detect_voice_activity(audio_path)             │
│  ├─ Algorithm:                                              │
│  │  vad = webrtcvad.Vad(2)                                 │
│  │  frame_duration = 30ms                                  │
│  │  ...                                                    │
│  └─ Test: Filter background noise                          │
│                                                              │
│  NGÀY 4 - GIỜ 3 (10:00-11:00): THRESHOLD IMPLEMENTATION    │
│  ├─ Jitter: <3% normal, 3-6% warning, >6% danger           │
│  ├─ Shimmer: <6% normal, 6-10% warning, >10% danger       │
│  ├─ WPM: Compare vs baseline                                 │
│  └─ Code: calculate_speech_score(jitter, shimmer, wpm)     │
│                                                              │
│  NGÀY 4 - GIỜ 4 (11:00-12:00): INTEGRATE ALL COMPONENTS    │
│  ├─ Combine: STT + WPM + MFCC + VAD                        │
│  ├─ Output: Speech Score (0-100)                           │
│  └─ Test: End-to-end speech analysis                       │
│                                                              │
│  NGÀY 4 - GIỜ 5-6 (13:00-15:00): TEST MODULE 2              │
│  ├─ Test: 10 normal speech                                  │
│  ├─ Test: 5 dysarthria speech                               │
│  ├─ Verify: FPR <5%                                         │
│  ├─ Verify: TPR >90%                                        │
│  └─ Fix: Any issues                                        │
│                                                              │
│  CHECKLIST HOÀN THÀNH:                                      │
│  ☑ Vosk STT working                                       │
│  ☑ WPM calculation working                                │
│  ☑ MFCC extraction working                                 │
│  ☑ Jitter/Shimmer calculation working                      │
│  ☑ VAD filtering background noise                         │
│  ☑ Thresholds from table implemented                      │
│  ☑ Speech Score (0-100) outputting correctly              │
│                                                              │
│  TÀI LIỆU HỌC HÔM NAY:                                    │
│  ├─ Vosk Documentation:                                    │
│  │  https://alphacephei.com/vosk/                         │
│  ├─ Librosa Features:                                       │
│  │  https://librosa.org/doc/main/feature.html             │
│  ├─ Jitter/Shimmer:                                         │
│  │  https://en.wikipedia.org/wiki/Jitter_and_shimmer      │
│  └─ Dysarthria Detection:                                   │
│     TORGO/UA-Speech datasets                               │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 5-6 (05-06/09) - MODULE 1-2 INTEGRATION

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 5-6: 05-06/09/2026                                    │
│  MỤC TIÊU: Tích hợp Module 1 + 2 + Test                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 5 - GIỜ 1-2 (8:00-10:00): CREATE FUSION ENGINE v1    │
│  ├─ File: fusion_engine.py                                  │
│  ├─ Function: fuse_face_speech(face_score, speech_score)    │
│  ├─ Algorithm (Simple):                                     │
│  │  weighted_score = 0.4*face + 0.6*speech                │
│  └─ Test: Fuse outputs from Module 1 + 2                   │
│                                                              │
│  NGÀY 5 - GIỜ 3 (10:00-11:00): CREATE DASHBOARD v1         │
│  ├─ File: dashboard.py                                      │
│  ├─ Framework: Streamlit                                    │
│  ├─ Components:                                              │
│  │  ├─ Video display (Webcam)                               │
│  │  ├─ Face score display                                   │
│  │  ├─ Speech score display                                 │
│  │  └─ Fused score display                                  │
│  └─ Test: Real-time display                                 │
│                                                              │
│  NGÀY 5 - GIỜ 4 (11:00-12:00): ALERT SYSTEM v1             │
│  ├─ File: alert_system.py                                   │
│  ├─ Function: send_alert(score, message)                    │
│  ├─ Channels:                                                │
│  │  ├─ Buzzer (local)                                       │
│  │  ├─ Zalo API (future)                                    │
│  │  └─ Log (console)                                        │
│  └─ Test: Trigger alert when score >80                     │
│                                                              │
│  NGÀY 5 - GIỜ 5-6 (13:00-15:00): END-TO-END TEST v1        │
│  ├─ Scenario 1: Normal state → No alert                      │
│  ├─ Scenario 2: Face asymmetry → Alert                      │
│  ├─ Scenario 3: Speech abnormality → Alert                  │
│  ├─ Scenario 4: Both abnormal → High alert                  │
│  └─ Verify: System working end-to-end                      │
│                                                              │
│  NGÀY 6 - GIỜ 1-2 (8:00-10:00): FALSE ALARM TEST           │
│  ├─ Scenario: Smiling (20x)                                  │
│  ├─ Scenario: Talking normally (20x)                       │
│  ├─ Scenario: Yawning (10x)                                  │
│  └─ Expected: FPR <5%                                      │
│                                                              │
│  NGÀY 6 - GIỜ 3 (10:00-11:00): TRUE POSITIVE TEST            │
│  ├─ Scenario: Face asymmetry (15x)                           │
│  ├─ Scenario: Speech slurring (15x)                         │
│  └─ Expected: TPR >90%                                     │
│                                                              │
│  NGÀY 6 - GIỜ 4 (11:00-12:00): PERFORMANCE BENCHMARK         │
│  ├─ Metric: FPS on RTX3050                                  │
│  ├─ Target: 15-20 FPS                                        │
│  ├─ Measure: actual FPS                                     │
│  └─ Optimize: If below target                               │
│                                                              │
│  NGÀY 6 - GIỜ 5-6 (13:00-15:00): DOCUMENT & REVIEW           │
│  ├─ Update: Bảng cơ sở khoa học                             │
│  ├─ Document: Module 1-2 performance                         │
│  ├─ Review: Any gaps?                                       │
│  └─ Plan: Week 2 tasks                                      │
│                                                              │
│  CHECKLIST TUẦN 1:                                          │
│  ☑ Môi trường phát triển OK                               │
│  ☑ Bảng cơ sở khoa học created                            │
│  ☑ Module 1 (Face) working                                │
│  ☑ Module 2 (Speech) working                               │
│  ☑ Fusion engine v1 working                                │
│  ☑ Dashboard v1 working                                    │
│  ☑ Alert system v1 working                                 │
│  ☑ FPR <5%, TPR >90% achieved                              │
│  ☑ Performance: 15-20 FPS on RTX3050                       │
└─────────────────────────────────────────────────────────────┘
```

---

# 3. TUẦN 2 (08/09 - 14/09): MODULE 3-5 + DEFENSE ENGINE

## 🗓️ NGÀY 7-8 (08-09/09) - MODULE 3: ARM WEAKNESS

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 7-8: 08-09/09/2026                                     │
│  MỤC TIÊU: Module 3 - Arm Weakness Detection               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 7 - GIỜ 1-2 (8:00-10:00): TÌM HIỂU YOLOV8N-POSE       │
│  ├─ Documentation:                                           │
│  │  https://docs.ultralytics.com/usage/poses/              │
│  ├─ 17 keypoints structure:                                 │
│  │  11-12: Shoulders                                       │
│  │  13-14: Elbows                                          │
│  │  15-16: Wrists                                          │
│  └─ Test: Load model + detect pose                           │
│                                                              │
│  NGÀY 7 - GIỜ 3 (10:00-11:00): CODE ARM KEYPOINT EXTRACTION │
│  ├─ File: arm_module.py                                     │
│  ├─ Function: extract_arm_keypoints(frame)                  │
│  ├─ Extract: Left shoulder, elbow, wrist                    │
│  ├─ Extract: Right shoulder, elbow, wrist                   │
│  └─ Test: Visualize keypoints on video                     │
│                                                              │
│  NGÀY 7 - GIỜ 4 (11:00-12:00): CODE ARM RAISE ASYMMETRY   │
│  ├─ Function: calculate_arm_raise_asymmetry(left, right)    │
│  ├─ Algorithm:                                              │
│  │  left_height = abs(left_shoulder.y - left_wrist.y)     │
│  │  right_height = abs(right_shoulder.y - right_wrist.y)   │
│  │  asymmetry = abs(left_height - right_height)            │
│  └─ Test: Test with arm raise video                          │
│                                                              │
│  NGÀY 7 - GIỜ 5-6 (13:00-15:00): CODE ARM DROOP + SWING    │
│  ├─ Function: calculate_arm_droop(keypoints)               │
│  │  droop_angle = angle(shoulder, elbow, wrist)           │
│  │                                                         │
│  ├─ Function: calculate_arm_swing(video_frames)            │
│  │  swing_amplitude = max(shoulder_x) - min(shoulder_x)   │
│  │                                                         │
│  └─ Test: Test with walking video                           │
│                                                              │
│  NGÀY 8 - GIỜ 1-2 (8:00-10:00): THRESHOLD IMPLEMENTATION   │
│  ├─ Arm raise asymmetry: <5cm normal, 5-10 warning        │
│  ├─ Arm droop: <15° normal, 15-25° warning                 │
│  ├─ Arm swing: <20% difference normal                      │
│  └─ Code: calculate_arm_score(asymmetry, droop, swing)      │
│                                                              │
│  NGÀY 8 - GIỜ 3 (10:00-11:00): NIHSS MAPPING              │
│  ├─ NIHSS Item 5 (Motor Arm):                               │
│  │  Score 0: No drift (Arm score >80)                      │
│  │  Score 1: Drift (60-80)                                 │
│  │  Score 2: Some effort (40-60)                           │
│  │  Score 3: Against gravity (20-40)                       │
│  │  Score 4: No movement (<20)                             │
│  └─ Code: map_arm_to_nihss(arm_score)                       │
│                                                              │
│  NGÀY 8 - GIỜ 4 (11:00-12:00): TEST MODULE 3                │
│  ├─ Test: 10 normal arm movements                           │
│  ├─ Test: 5 weak arm simulations                            │
│  ├─ Verify: FPR <5%                                         │
│  └─ Verify: TPR >90%                                        │
│                                                              │
│  NGÀY 8 - GIỜ 5-6 (13:00-15:00): INTEGRATE WITH FUSION     │
│  ├─ Update: fusion_engine.py                                │
│  ├─ Add: Arm score to fusion                                │
│  └─ Test: End-to-end with 3 modules                        │
│                                                              │
│  CHECKLIST:                                                  │
│  ☑ YOLOv8n-pose working                                   │
│  ☑ Arm keypoint extraction working                         │
│  ☑ Arm raise asymmetry working                             │
│  ☑ Arm droop detection working                             │
│  ☑ Arm swing analysis working                              │
│  ☑ Thresholds from table implemented                      │
│  ☑ NIHSS mapping working                                  │
│  ☑ FPR <5%, TPR >90%                                      │
│                                                              │
│  TÀI LIỆU HỌC:                                             │
│  ├─ NIHSS Motor Arm criteria                               │
│  └─ YOLOv8 Pose estimation                                │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 9-10 (10-11/09) - MODULE 4: GAIT ABNORMALITY

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 9-10: 10-11/09/2026                                    │
│  MỤC TIÊU: Module 4 - Gait Abnormality Detection           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 9 - GIỜ 1-2 (8:00-10:00): GAIT KEYPOINT EXTRACTION  │
│  ├─ Use: Same YOLOv8n-pose                                 │
│  ├─ Extract: Hip, knee, ankle keypoints                    │
│  ├─ Left: 23, 25, 27 (hip, knee, ankle)                    │
│  ├─ Right: 24, 26, 28                                       │
│  └─ Test: Visualize gait keypoints                          │
│                                                              │
│  NGÀY 9 - GIỜ 3 (10:00-11:00): STEP LENGTH ASYMMETRY       │
│  ├─ Function: calculate_step_length(keypoints)             │
│  ├─ Algorithm:                                              │
│  │  step_length = distance(ankle_n, ankle_n+1)           │
│  │  asymmetry = abs(left_step - right_step)               │
│  └─ Test: Calculate step lengths                            │
│                                                              │
│  NGÀY 9 - GIỜ 4 (11:00-12:00): STEP TIME ASYMMETRY         │
│  ├─ Function: calculate_step_time(frames)                   │
│  ├─ Algorithm:                                              │
│  │  step_duration = time(frame_n+1) - time(frame_n)       │
│  │  asymmetry = abs(left_time - right_time)                │
│  └─ Test: Calculate step times                              │
│                                                              │
│  NGÀY 9 - GIỜ 5-6 (13:00-15:00): GAIT SPEED + THRESHOLDS   │
│  ├─ Function: calculate_gait_speed(distance, time)          │
│  ├─ Thresholds:                                              │
│  │  Step length: <5cm normal, 5-15 warning                 │
│  │  Step time: <0.1s normal, 0.1-0.2s warning              │
│  │  Gait speed: >1.0 m/s normal                             │
│  └─ Code: calculate_gait_score(length, time, speed)        │
│                                                              │
│  NGÀY 10 - GIỜ 1-2 (8:00-10:00): NIHSS MAPPING            │
│  ├─ NIHSS Item 6 (Motor Leg):                               │
│  │  Score 0: Normal (Gait score >80)                       │
│  │  Score 1: Mild (60-80)                                  │
│  │  Score 2: Moderate (40-60)                              │
│  │  Score 3: Severe (20-40)                                │
│  │  Score 4: Paralysis (<20)                               │
│  └─ Code: map_gait_to_nihss(gait_score)                     │
│                                                              │
│  NGÀY 10 - GIỜ 3 (10:00-11:00): TEST MODULE 4               │
│  ├─ Test: 10 normal walking                                  │
│  ├─ Test: 5 abnormal gait                                   │
│  ├─ Verify: FPR <5%                                         │
│  └─ Verify: TPR >90%                                        │
│                                                              │
│  NGÀY 10 - GIỜ 4 (11:00-12:00): INTEGRATE WITH FUSION      │
│  ├─ Update: fusion_engine.py                                │
│  ├─ Add: Gait score to fusion                               │
│  └─ Test: End-to-end with 4 modules                         │
│                                                              │
│  NGÀY 10 - GIỜ 5-6 (13:00-15:00): DOCUMENT & OPTIMIZE      │
│  ├─ Update: Bảng cơ sở khoa học                             │
│  ├─ Optimize: Performance on RTX3050                        │
│  └─ Document: Module 3-4 performance                        │
│                                                              │
│  CHECKLIST:                                                  │
│  ☑ Gait keypoint extraction working                        │
│  ☑ Step length asymmetry working                           │
│  ☑ Step time asymmetry working                             │
│  ☑ Gait speed calculation working                          │
│  ☑ Thresholds from table implemented                       │
│  ☑ NIHSS mapping working                                   │
│  ☑ FPR <5%, TPR >90%                                      │
│  ☑ Integrated with fusion engine                            │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 11-12 (12-13/09) - MODULE 5: RADAR + DEFENSE

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 11-12: 12-13/09/2026                                   │
│  MỤC TIÊU: Module 5 Radar + 4-Layer Defense Engine          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 11 - GIỜ 1-2 (8:00-10:00): LD2450 RADAR SETUP       │
│  ├─ Hardware: LD2450 + UART-USB                             │
│  ├─ Connect: Radar → USB → Laptop                           │
│  ├─ Test: Read radar data via UART                          │
│  │  import serial                                           │
│  │  ser = serial.Serial('COM3', 115200)                    │
│  │  data = ser.readline()                                  │
│  └─ Verify: Radar outputting x, y, velocity                 │
│                                                              │
│  NGÀY 11 - GIỜ 3 (10:00-11:00): PARSE RADAR DATA           │
│  ├─ File: radar_module.py                                   │
│  ├─ Function: parse_radar_data(raw_data)                    │
│  ├─ Extract: x, y coordinates, velocity                    │
│  ├─ Filter: Noise reduction                                │
│  └─ Test: Display real-time radar data                      │
│                                                              │
│  NGÀY 11 - GIỜ 4 (11:00-12:00): FALL DETECTION (PROXY)     │
│  ├─ Function: detect_fall(radar_history)                   │
│  ├─ Algorithm:                                              │
│  │  position_change = |Δx| + |Δy|                         │
│  │  if position_change > 1m: fall_candidate               │
│  │  inactivity = time_since_last_movement                   │
│  │  if inactivity > 45s: fall_confirmed                     │
│  └─ Test: Simulate fall scenario                           │
│                                                              │
│  NGÀY 11 - GIỜ 5-6 (13:00-15:00): AUDIO FUSION              │
│  ├─ Function: audio_fallback(radar_fall, audio_abnormal)    │
│  ├─ AND Gate:                                                │
│  │  if radar_fall AND audio_abnormal: ALERT               │
│  │  else if radar_fall AND NOT audio_abnormal: NO ALERT    │
│  │  else if NOT radar_fall: NO ALERT                        │
│  └─ Test: Test with various scenarios                       │
│                                                              │
│  NGÀY 12 - GIỜ 1-2 (8:00-10:00): LAYER 1 - QUICK CALIB     │
│  ├─ File: defense_engine.py                                  │
│  ├─ Layer 1: Quick Calibration (15 phút)                     │
│  │  Function: quick_calibration(user_data)                 │
│  │  Process:                                               │
│  │    ├─ User reads 3 sentences (5 min)                    │
│  │    ├─ Camera captures 10 frames                         │
│  │    ├─ User walks, raises arms (5 min)                   │
│  │    └─ Output: Personal baseline                          │
│  └─ Test: Calibrate with 3 users                            │
│                                                              │
│  NGÀY 12 - GIỜ 3 (10:00-11:00): LAYER 2 - CONTEXT AWARE    │
│  ├─ Layer 2: Context Awareness                               │
│  │  Function: detect_activity(motion_pattern)              │
│  │  Activities:                                            │
│  │    ├─ Exercise → Suppress face/arm/gait                 │
│  │    ├─ Walking → Suppress arm/gait                      │
│  │    ├─ Talking → Suppress speech                        │
│  │    └─ Rest → No suppression                             │
│  └─ Test: Test with exercise scenario                       │
│                                                              │
│  NGÀY 12 - GIỜ 4 (11:00-12:00): LAYER 3 - TEMPORAL          │
│  ├─ Layer 3: Temporal Analysis                               │
│  │  Function: temporal_analysis(score_history)              │
│  │  Windows: 30s, 1min, 2min, 5min, 10min                  │
│  │  Logic:                                                 │
│  │    ├─ Transient (<30s): False alarm                    │
│  │    ├─ Persistent (>10min): Stroke                       │
│  │    └─ Worsening: Emergency                               │
│  └─ Test: Test with persistent symptom                     │
│                                                              │
│  NGÀY 12 - GIỜ 5-6 (13:00-15:00): LAYER 4 + TEST            │
│  ├─ Layer 4: Adaptive Thresholding                           │
│  │  Function: adaptive_threshold(personal_baseline)         │
│  │  Process:                                               │
│  │    ├─ Initial: Universal thresholds                    │
│  │    ├─ After 1 week: 50% personalized                  │
│  │    └─ After 1 month: 100% personalized                 │
│  │                                                         │
│  ├─ Test: 4-layer defense system                           │
│  │  ├─ False alarm scenarios: 50x                          │
│  │  └─ True positive scenarios: 25x                         │
│  │                                                         │
│  └─ Expected: FPR reduced to <5%                           │
│                                                              │
│  CHECKLIST:                                                  │
│  ☑ Radar UART communication working                       │
│  ☑ Radar data parsing working                              │
│  ☑ Fall detection (proxy method) working                   │
│  ☑ Audio fusion (AND gate) working                        │
│  ☑ Layer 1: Quick calibration working                     │
│  ☑ Layer 2: Context awareness working                      │
│  ☑ Layer 3: Temporal analysis working                      │
│  ☑ Layer 4: Adaptive threshold working                     │
│  ☑ 4-layer defense: FPR <5%                               │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 13-14 (14-15/09) - INTEGRATION & TESTING

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 13-14: 14-15/09/2026                                   │
│  MỤC TIÊU: Full Integration + End-to-End Testing            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 13 - GIỜ 1-2 (8:00-10:00): FULL SYSTEM INTEGRATION    │
│  ├─ Connect: All 5 modules → Fusion Engine                 │
│  ├─ Connect: Fusion Engine → Defense Engine                │
│  ├─ Connect: Defense Engine → Alert System                 │
│  └─ Verify: Data flow end-to-end                           │
│                                                              │
│  NGÀY 13 - GIỜ 3 (10:00-11:00): UPDATE DASHBOARD           │
│  ├─ Add: All 5 module scores                                │
│  ├─ Add: Defense status                                      │
│  ├─ Add: Fusion score                                        │
│  └─ Add: Alert history                                       │
│                                                              │
│  NGÀY 13 - GIỜ 4 (11:00-12:00): PERFORMANCE OPTIMIZATION    │
│  ├─ Measure: Current FPS on RTX3050                         │
│  ├─ Target: 15-20 FPS                                       │
│  ├─ Optimize:                                                │
│  │  ├─ Reduce YOLOv8 input resolution                       │
│  │  ├─ Batch processing                                     │
│  │  └─ GPU memory optimization                            │
│  └─ Verify: Achieved target FPS                            │
│                                                              │
│  NGÀY 13 - GIỜ 5-6 (13:00-15:00): COMPREHENSIVE TEST         │
│  ├─ Test Suite: 200 scenarios                                │
│  │  ├─ False alarm: 100 scenarios                           │
│  │  ├─ True positive: 50 scenarios                          │
│  │  ├─ Multi-user: 30 scenarios                             │
│  │  └─ Night mode: 20 scenarios                             │
│  ├─ Metrics:                                                 │
│  │  ├─ FPR: Target <5%                                     │
│  │  ├─ TPR: Target >90%                                    │
│  │  └─ Overall: >92%                                       │
│  └─ Document: Results in spreadsheet                       │
│                                                              │
│  NGÀY 14 - GIỜ 1-2 (8:00-10:00): BUG FIXES                 │
│  ├─ Review: All test results                                │
│  ├─ Identify: Priority bugs                                 │
│  ├─ Fix: Critical issues                                     │
│  └─ Retest: Verify fixes                                    │
│                                                              │
│  NGÀY 14 - GIỜ 3 (10:00-11:00): DOCUMENT WEEK 2            │
│  ├─ Update: Bảng cơ sở khoa học                               │
│  ├─ Document: Module 3-5 performance                       │
│  ├─ Document: Defense engine performance                    │
│  └─ Document: Integration challenges                         │
│                                                              │
│  NGÀY 14 - GIỜ 4 (11:00-12:00): PLAN WEEK 3                 │
│  ├─ Research: NIHSS estimation algorithms                    │
│  ├─ Research: Stroke subtype classification                  │
│  ├─ Research: Pre-hospital handoff systems                  │
│  └─ Plan: Implementation strategy                           │
│                                                              │
│  NGÀY 14 - GIỜ 5-6 (13:00-15:00): LEARN NIHSS DEEPLY        │
│  ├─ Study: NIHSS scale in detail                            │
│  │  https://www.ninds.nih.gov/Disorders/All-Disorders/    │
│  │  NIHSS-Stroke-Scale/                                    │
│  ├─ Understand: Each item scoring                           │
│  ├─ Practice: Score sample videos                            │
│  └─ Prepare: For NIHSS estimation module                   │
│                                                              │
│  CHECKLIST TUẦN 2:                                          │
│  ☑ Module 3 (Arm) working + integrated                    │
│  ☑ Module 4 (Gait) working + integrated                   │
│  ☑ Module 5 (Radar) working + integrated                   │
│  ☑ 4-Layer Defense engine working                           │
│  ☑ All modules integrated                                  │
│  ☑ Dashboard updated with all modules                      │
│  ☑ Performance: 15-20 FPS on RTX3050                      │
│  ☑ 200 scenarios tested                                   │
│  ☑ FPR <5%, TPR >90%, Overall >92%                        │
└─────────────────────────────────────────────────────────────┘
```

---

# 4. TUẦN 3 (16/09 - 22/09): NIHSS ESTIMATION + TRIAGE

## 🗓️ NGÀY 15-16 (16-17/09) - NIHSS ESTIMATION

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 15-16: 16-17/09/2026                                   │
│  MỤC TIÊU: NIHSS Estimation Module (ĐỘT PHÁ)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 15 - GIỜ 1-2 (8:00-10:00): NIHSS MAPPING DESIGN     │
│  ├─ File: nihss_estimator.py                                │
│  ├─ Mapping Strategy:                                       │
│  │  ├─ Face Module → NIHSS Item 4 (Facial Palsy)          │
│  │  ├─ Speech Module → NIHSS Item 10 (Dysarthria)         │
│  │  ├─ Arm Module → NIHSS Item 5 (Motor Arm)               │
│  │  └─ Gait Module → NIHSS Item 6 (Motor Leg)             │
│  │                                                         │
│  ├─ Score Mapping:                                          │
│  │  ├─ Module score (0-100) → NIHSS score (0-4)          │
│  │  ├─ >80 → Score 0                                       │
│  │  ├─ 60-80 → Score 1                                     │
│  │  ├─ 40-60 → Score 2                                     │
│  │  ├─ 20-40 → Score 3                                     │
│  │  └─ <20 → Score 4                                       │
│  │                                                         │
│  └─ Total NIHSS: Sum of mapped scores                      │
│                                                              │
│  NGÀY 15 - GIỜ 3 (10:00-11:00): CODE NIHSS MAPPING         │
│  ├─ Function: map_face_to_nihss(face_score)                 │
│  ├─ Function: map_speech_to_nihss(speech_score)             │
│  ├─ Function: map_arm_to_nihss(arm_score)                   │
│  └─ Function: map_gait_to_nihss(gait_score)                 │
│                                                              │
│  NGÀY 15 - GIỜ 4 (11:00-12:00): CODE NIHSS CALCULATOR     │
│  ├─ Function: calculate_nihss(module_scores)                │
│  ├─ Algorithm:                                              │
│  │  item_4 = map_face_to_nihss(face_score)                 │
│  │  item_5 = map_arm_to_nihss(arm_score)                   │
│  │  item_6 = map_gait_to_nihss(gait_score)                 │
│  │  item_10 = map_speech_to_nihss(speech_score)            │
│  │  total_nihss = item_4 + item_5 + item_6 + item_10      │
│  └─ Return: NIHSS score (0-15 max for 4 items)            │
│                                                              │
│  NGÀY 15 - GIỜ 5-6 (13:00-15:00): SEVERITY CLASSIFICATION  │
│  ├─ Function: classify_severity(nihss_score)                │
│  ├─ Classification:                                          │
│  │  ├─ Mild: NIHSS 0-5                                     │
│  │  ├─ Moderate: NIHSS 6-13                                │
│  │  └─ Severe: NIHSS 14-15                                 │
│  └─ Test: Classify test cases                               │
│                                                              │
│  NGÀY 16 - GIỜ 1-2 (8:00-10:00): CONFIDENCE INTERVAL       │
│  ├─ Function: calculate_nihss_ci(module_scores, std_devs)   │
│  ├─ Method: Bootstrap hoặc error propagation                │
│  ├─ Output: NIHSS ± CI (e.g., 8 ± 2)                       │
│  └─ Test: Calculate CI for sample data                      │
│                                                              │
│  NGÀY 16 - GIỜ 3 (10:00-11:00): INTEGRATE WITH SYSTEM       │
│  ├─ Update: fusion_engine.py                                │
│  ├─ Add: NIHSS estimation to output                         │
│  ├─ Update: Dashboard to show NIHSS                         │
│  └─ Test: End-to-end with NIHSS output                      │
│                                                              │
│  NGÀY 16 - GIỜ 4 (11:00-12:00): TEST NIHSS ESTIMATION       │
│  ├─ Test Data: 50 stroke videos (public)                    │
│  ├─ Process:                                                │
│  │  ├─ Run system on each video                             │
│  │  ├─ Get estimated NIHSS                                  │
│  │  └─ Compare to manual scoring                            │
│  │                                                         │
│  └─ Metrics:                                                 │
│     ├─ Correlation (target r ≥ 0.85)                        │
│     ├─ Mean Absolute Error (target <2)                      │
│     └─ Accuracy (target >80%)                               │
│                                                              │
│  NGÀY 16 - GIỜ 5-6 (13:00-15:00): GET NEUROLOGIST FEEDBACK   │
│  ├─ Contact: Bác sĩ thần kinh ĐHYD                          │
│  ├─ Prepare: 10 sample cases with NIHSS estimates             │
│  ├─ Ask:                                                    │
│  │  ├─ NIHSS estimates reasonable?                          │
│  │  ├─ Which module most accurate?                         │
│  │  └─ Suggestions for improvement?                        │
│  │                                                         │
│  └─ Document: All feedback                                  │
│                                                              │
│  CHECKLIST:                                                  │
│  ☑ NIHSS mapping designed                                 │
│  ☑ All mapping functions coded                             │
│  ☑ NIHSS calculator working                                │
│  ☑ Severity classification working                         │
│  ☑ Confidence interval calculation working                │
│  ☑ Integrated with system                                 │
│  ☑ Tested with 50 videos                                   │
│  ☑ Correlation r ≥ 0.85 achieved (or documented)           │
│  ☑ Neurologist feedback obtained                           │
│                                                              │
│  TÀI LIỆU HỌC:                                             │
│  ├─ NIHSS Scale (Study deeply)                              │
│  └─ Automated NIHSS papers                                 │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 17-18 (18-19/09) - TRIAGE ENGINE

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 17-18: 18-19/09/2026                                   │
│  MỤC TIÊU: Triage Engine (Subtype + Severity + Hospital)     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 17 - GIỜ 1-2 (8:00-10:00): STROKE SUBTYPE HINT       │
│  ├─ File: triage_engine.py                                   │
│  ├─ Function: predict_subtype(symptoms, history)            │
│  ├─ Algorithm (Rule-based, NOT diagnosis):                  │
│  │  ├─ Sudden severe headache → Hemorrhagic (+30%)         │
│  │  ├─ Vomiting → Hemorrhagic (+20%)                        │
│  │  ├─ Gradual progression → Ischemic (+20%)                │
│  │  ├─ Speech-dominant → Ischemic (+15%)                    │
│  │  └─ Base probability: 80% Ischemic, 20% Hemorrhagic     │
│  │                                                         │
│  └─ Output: "Possible Ischemic (65% confidence)"           │
│                                                              │
│  NGÀY 17 - GIỜ 3 (10:00-11:00): SEVERITY PREDICTION        │
│  ├─ Function: predict_severity(nihss_score, trend)         │
│  ├─ Algorithm:                                              │
│  │  ├─ Based on NIHSS (already classified)                 │
│  │  ├─ Based on trend (worsening vs stable)                │
│  │  ├─ Worsening: Upgrade severity                          │
│  │  └─ Stable: Keep current severity                        │
│  └─ Output: Severity + Confidence                          │
│                                                              │
│  NGÀY 17 - GIỜ 4 (11:00-12:00): HOSPITAL DATABASE          │
│  ├─ File: hospital_database.csv                              │
│  ├─ Columns:                                                 │
│  │  ├─ Hospital name                                       │
│  │  ├─ Has CT scanner (Yes/No)                            │
│  │  ├─ Has stroke team (Yes/No)                           │
│  │  ├─ Has thrombolysis capability (Yes/No)                │
│  │  ├─ Distance from user (km)                             │
│  │  └─ Emergency contact                                   │
│  │                                                         │
│  ├─ Sample data:                                            │
│  │  ├─ BV115: Yes, Yes, Yes, 5km, 0909-xxx-xxx             │
│  │  ├─ ĐHYD: Yes, Yes, Yes, 8km, 0909-xxx-xxx              │
│  │  └─ BV Qận: Yes, No, No, 3km, 0909-xxx-xxx               │
│  │                                                         │
│  └─ Code: load_hospital_database()                          │
│                                                              │
│  NGÀY 17 - GIỜ 5-6 (13:00-15:00): HOSPITAL RECOMMENDATION  │
│  ├─ Function: recommend_hospital(subtype, severity, location)│
│  ├─ Algorithm:                                              │
│  │  ├─ Severe stroke → Hospital with stroke team           │
│  │  ├─ Hemorrhagic → Hospital with surgery capability     │
│  │  ├─ Ischemic → Hospital with thrombolysis               │
│  │  └─ Mild → Nearest hospital                             │
│  │                                                         │
│  └─ Output: Recommended hospital + Reason                  │
│                                                              │
│  NGÀY 18 - GIỜ 1-2 (8:00-10:00): INTEGRATE TRIAGE           │
│  ├─ Connect: NIHSS → Subtype → Severity → Hospital         │
│  ├─ Function: triage_stroke(all_data)                       │
│  ├─ Output: Full triage report                              │
│  └─ Test: End-to-end triage                                │
│                                                              │
│  NGÀY 18 - GIỜ 3 (10:00-11:00): UPDATE DASHBOARD           │
│  ├─ Add: Subtype hint display                               │
│  ├─ Add: Severity prediction display                        │
│  ├─ Add: Hospital recommendation display                     │
│  └─ Add: Map to recommended hospital                         │
│                                                              │
│  NGÀY 18 - GIỜ 4 (11:00-12:00): TEST TRIAGE ENGINE         │
│  ├─ Test: 30 stroke scenarios                               │
│  ├─ Verify: Subtype hint reasonable                         │
│  ├─ Verify: Severity matches NIHSS                          │
│  └─ Verify: Hospital recommendation logical                 │
│                                                              │
│  NGÀY 18 - GIỜ 5-6 (13:00-15:00): DOCUMENT & LEARN          │
│  ├─ Document: Triage algorithm                               │
│  ├─ Document: Hospital database                             │
│  ├─ Learn: Stroke subtype classification literature          │
│  └─ Learn: Pre-hospital triage best practices               │
│                                                              │
│  CHECKLIST:                                                  │
│  ☑ Subtype hint function working                          │
│  ☑ Severity prediction working                             │
│  ☑ Hospital database created                               │
│  ☑ Hospital recommendation working                         │
│  ☑ All triage components integrated                        │
│  ☑ Dashboard updated with triage info                     │
│  ☑ Tested with 30 scenarios                                │
│  ☑ Documented thoroughly                                  │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 19-21 (20-22/09) - HANDOFF SYSTEM + VALIDATION PREP

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 19-21: 20-22/09/2026                                    │
│  MỤC TIÊU: Handoff System + Validation Preparation          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 19 - GIỜ 1-2 (8:00-10:00): MEDICAL REPORT GENERATOR   │
│  ├─ File: handoff_system.py                                  │
│  ├─ Function: generate_medical_report(all_data)             │
│  ├─ Components:                                              │
│  │  ├─ Patient info (age, history)                          │
│  │  ├─ Timeline (0 → onset → detection → alert)            │
│  │  ├─ NIHSS estimate                                       │
│  │  ├─ Affected regions (face, arm, speech, gait)           │
│  │  ├─ Subtype hint                                         │
│  │  └─ Severity prediction                                  │
│  └─ Output: Structured medical report                       │
│                                                              │
│  NGÀY 19 - GIỜ 3 (10:00-11:00): TIMELINE DOCUMENTATION     │
│  ├─ Function: create_timeline(event_log)                    │
│  ├─ Events:                                                  │
│  │  ├─ T0: System baseline                                  │
│  │  ├─ T1: First symptom detected                          │
│  │  ├─ T2: Symptom progression                             │
│  │  ├─ T3: Alert triggered                                  │
│  │  └─ T4: Transport initiated                             │
│  │                                                         │
│  └─ Output: Visual timeline graph                           │
│                                                              │
│  NGÀY 19 - GIỜ 4 (11:00-12:00): QR CODE GENERATOR          │
│  ├─ Library: qrcode                                         │
│  ├─ Function: generate_qrcode(data)                         │
│  ├─ Data: Full sensor data + timeline                       │
│  ├─ Output: QR code image                                   │
│  └─ Test: Scan QR → Retrieve data                           │
│                                                              │
│  NGÀY 19 - GIỜ 5-6 (13:00-15:00): PDF EXPORT               │
│  ├─ Library: reportlab                                      │
│  ├─ Function: export_pdf(report, qrcode)                    │
│  ├─ Format: DICOM-compatible                                │
│  └─ Output: Medical report PDF                              │
│                                                              │
│  NGÀY 20 - GIỜ 1-2 (8:00-10:00): PREPARE VALIDATION PLAN  │
│  ├─ Plan: NIHSS Correlation Study                           │
│  │  ├─ Sample: 50 public stroke videos                       │
│  │  ├─ Process: Run system → Get NIHSS → Compare           │
│  │  ├─ Metrics: r, p-value, CI                             │
│  │  └─ Target: r ≥ 0.85, p < 0.05                          │
│  │                                                         │
│  ├─ Plan: Confusion Matrix Study                            │
│  │  ├─ Sample: 100 scenarios (50 normal, 50 stroke)         │
│  │  ├─ Metrics: Sensitivity, Specificity, F1-score           │
│  │  └─ Target: Sens >90%, Spec >95%                         │
│  │                                                         │
│  └─ Plan: Clinical Validation                               │
│     ├─ Contact: BV115, ĐHYD                                 │
│     ├─ Ask: Review system + provide letter                │
│     └─ Timeline: Week 4                                    │
│                                                              │
│  NGÀY 20 - GIỜ 3 (10:00-11:00): DOWNLOAD VALIDATION DATA   │
│  ├─ Public stroke videos:                                   │
│  │  ├─ YouTube stroke symptom videos                       │
│  │  ├─ Medical education videos                            │
│  │  └─ TORGO dataset samples                               │
│  ├─ Target: 50 videos with varying NIHSS                    │
│  └─ Organize: For neurologist scoring                       │
│                                                              │
│  NGÀY 20 - GIỜ 4 (11:00-12:00): CODE VALIDATION METRICS   │
│  ├─ File: validation_metrics.py                             │
│  ├─ Function: calculate_correlation(est_nihss, true_nihss)  │
│  ├─ Function: confusion_matrix(predictions, labels)         │
│  └─ Function: calculate_sensitivity_specificity(...)       │
│                                                              │
│  NGÀY 20 - GIỜ 5-6 (13:00-15:00): GET DOCTOR LETTERS       │
│  ├─ Hospital 1: BV115                                       │
│  │  ├─ Contact: Trưởng khoa Thần kinh                     │
│  │  ├─ Present: System + NIHSS estimation                  │
│  │  ├─ Ask: Letter confirming system review               │
│  │  └─ Note: NOT clinical trial, just consultation       │
│  │                                                         │
│  └─ Hospital 2: ĐHYD                                        │
│     └─ (Same process)                                      │
│                                                              │
│  NGÀY 21 - GIỜ 1-2 (8:00-10:00): RUN VALIDATION - PART 1    │
│  ├─ NIHSS Correlation Study:                                │
│  │  ├─ Run system on 50 videos                              │
│  │  ├─ Get estimated NIHSS                                   │
│  │  └─ Prepare for doctor scoring                          │
│  │                                                         │
│  └─ Note: Neurologist scoring takes time                   │
│                                                              │
│  NGÀY 21 - GIỜ 3 (10:00-11:00): RUN VALIDATION - PART 2    │
│  ├─ Confusion Matrix Study:                                  │
│  │  ├─ Run system on 100 scenarios                          │
│  │  ├─ Collect predictions                                   │
│  │  └─ Calculate metrics                                   │
│  │                                                         │
│  └─ Target: Sens >90%, Spec >95%                           │
│                                                              │
│  NGÀY 21 - GIỜ 4 (11:00-12:00): ANALYZE RESULTS            │
│  ├─ Compile: All validation results                         │
│  ├─ Graphs:                                                  │
│  │  ├─ NIHSS correlation scatter plot                       │
│  │  ├─ Confusion matrix heatmap                             │
│  │  └─ ROC curve                                           │
│  └─ Document: All findings                                 │
│                                                              │
│  NGÀY 21 - GIỜ 5-6 (13:00-15:00): DOCUMENT WEEK 3           │
│  ├─ Update: Bảng cơ sở khoa học                             │
│  ├─ Document: NIHSS estimation validation                  │
│  ├─ Document: Triage engine validation                      │
│  ├─ Document: Handoff system validation                     │
│  └─ Plan: Week 4 final testing                             │
│                                                              │
│  CHECKLIST TUẦN 3:                                          │
│  ☑ NIHSS estimation module working                         │
│  ☑ NIHSS correlation r ≥ 0.85 (or documented)               │
│  ☑ Triage engine (subtype + severity + hospital) working   │
│  ☑ Handoff system (report + QR + PDF) working              │
│  ☑ Validation metrics coded                                │
│  ☑ Validation data downloaded                            │
│  ☑ Doctor letters requested (or obtained)                  │
│  ☑ Confusion matrix: Sens >90%, Spec >95%                  │
│  ☑ All findings documented                                │
└─────────────────────────────────────────────────────────────┘
```

---

# 5. TUẦN 4 (23/09 - 29/09): FINAL TESTING + HOSPITAL LETTERS

## 🗓️ NGÀY 22-24 (23-25/09) - REAL-WORLD TESTING

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 22-24: 23-25/09/2026                                    │
│  MỤC TIÊU: Real-World Testing (3-5 homes)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 22 - GIỜ 1-2 (8:00-10:00): INSTALL SYSTEM - HOME 1   │
│  ├─ Location: House of volunteer 1                          │
│  ├─ Setup:                                                   │
│  │  ├─ Webcam + Mic in living room                         │
│  │  ├─ Radar 1 in bedroom                                   │
│  │  ├─ Radar 2 in bathroom                                  │
│  │  └─ Laptop in central location                           │
│  │                                                         │
│  ├─ Calibration: 15 minutes                                │
│  └─ Test: Verify all sensors working                         │
│                                                              │
│  NGÀY 22 - GIỜ 3 (10:00-11:00): MONITOR SYSTEM - HOME 1     │
│  ├─ Monitor: Remote access via TeamViewer                    │
│  ├─ Collect: Scenarios throughout day                        │
│  └─ Note: Any edge cases                                    │
│                                                              │
│  NGÀY 22 - GIỜ 4 (11:00-12:00): INSTALL SYSTEM - HOME 2   │
│  └─ (Same process as Home 1)                                 │
│                                                              │
│  NGÀY 22 - GIỜ 5-6 (13:00-15:00): COLLECT DATA - DAY 1     │
│  ├─ Home 1: Collect 50+ scenarios                            │
│  ├─ Home 2: Collect 50+ scenarios                            │
│  └─ Document: All observations                                │
│                                                              │
│  NGÀY 23 - GIỜ 1-2 (8:00-10:00): INSTALL SYSTEM - HOME 3    │
│  └─ (Same process)                                          │
│                                                              │
│  NGÀY 23 - GIỜ 3 (10:00-11:00): ANALYZE HOME 1-2 DATA      │
│  ├─ Review: All scenarios from Home 1-2                      │
│  ├─ Identify: False positives, true positives                │
│  ├─ Calculate: FPR, TPR for each home                        │
│  └─ Compare: Lab vs real-world performance                   │
│                                                              │
│  NGÀY 23 - GIỜ 4 (11:00-12:00): COLLECT DATA - DAY 2-3    │
│  ├─ Continue monitoring all homes                            │
│  └─ Target: 200+ total scenarios                            │
│                                                              │
│  NGÀY 23 - GIỜ 5-6 (13:00-15:00): USER FEEDBACK             │
│  ├─ Interview: Users from each home                          │
│  ├─ Questions:                                               │
│  │  ├─ System easy to use?                                  │
│  │  ├─ Any false alarms?                                     │
│  │  ├─ Any missed detections?                                │
│  │  ├─ Suggestions?                                          │
│  │  └─ Would you recommend?                                  │
│  └─ Document: All feedback                                   │
│                                                              │
│  NGÀY 24 - GIỜ 1-2 (8:00-10:00): FINAL VALIDATION          │
│  ├─ Compile: All real-world data                            │
│  ├─ Calculate:                                               │
│  │  ├─ Overall FPR, TPR                                     │
│  │  ├─ Per-home performance                                  │
│  │  └─ User satisfaction                                     │
│  └─ Compare: Lab vs Real-world                              │
│                                                              │
│  NGÀY 24 - GIỜ 3 (10:00-11:00): BUG FIXES                  │
│  ├─ Review: Any issues from real-world testing               │
│  ├─ Fix: Priority bugs                                       │
│  └─ Retest: Verify fixes                                    │
│                                                              │
│  NGÀY 24 - GIỜ 4 (11:00-12:00): COMPILE FINAL RESULTS      │
│  ├─ Create: Final validation report                          │
│  │  ├─ NIHSS correlation                                    │
│  │  ├─ Confusion matrix                                     │
│  │  ├─ Real-world performance                               │
│  │  └─ User feedback                                        │
│  └─ Prepare: For presentation                               │
│                                                              │
│  NGÀY 24 - GIỜ 5-6 (13:00-15:00): REMOVE SYSTEMS            │
│  ├─ Thank: All volunteers                                    │
│  ├─ Remove: All hardware                                    │
│  └─ Note: Any final suggestions                             │
│                                                              │
│  CHECKLIST:                                                  │
│  ☑ Installed 3-5 systems                                   │
│  ☑ Tested 1 week per home                                 │
│  ☑ Collected 200+ scenarios                                │
│  ☑ User feedback documented                                │
│  ☑ Real-world FPR <5%                                      │
│  ☑ Real-world TPR >90%                                     │
│  ☑ Final validation report created                         │
└─────────────────────────────────────────────────────────────┘
```

## 🗓️ NGÀY 25-28 (26-29/09) - FINALIZE + DOCTOR LETTERS

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 25-28: 26-29/09/2026                                    │
│  MỤC TIÊU: Finalize System + Get Doctor Letters              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 25-26: COMPLETE SYSTEM FINALIZATION                    │
│  ├─ Review: All modules                                     │
│  ├─ Optimize: Performance                                    │
│  ├─ Bug fix: Any remaining issues                            │
│  └─ Test: Final end-to-end                                 │
│                                                              │
│  NGÀY 27-28: GET DOCTOR LETTERS                             │
│  ├─ BV115: Letter confirming system review                  │
│  ├─ ĐHYD: Letter confirming consultation                   │
│  └─ Alternative: If hospitals not responsive               │
│     └─ Use private neurologist letters                      │
│                                                              │
│  CHECKLIST:                                                  │
│  ☑ All modules working optimally                          │
│  ☑ All bugs fixed                                         │
│  ☑ Hospital letters obtained (or alternative)              │
└─────────────────────────────────────────────────────────────┘
```

---

# 6. TUẦN 5 (30/09 - 02/10): FINAL PREPARATION

## 🗓️ NGÀY 29-32 (30/09 - 02/10) - DEMO, VIDEO, POSTER

```
┌─────────────────────────────────────────────────────────────┐
│  NGÀY 29-32: 30/09 - 02/10/2026                               │
│  MỤC TIÊU: Demo, Video, Poster, Báo cáo                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NGÀY 29 - GIỜ 1-2 (8:00-10:00): PREPARE DEMO SCRIPT       │
│  ├─ Write: 8-minute demo script                             │
│  ├─ Prepare: Slide presentation                             │
│  ├─ Rehearse: Timing                                        │
│  └─ Backup: Record video demo                              │
│                                                              │
│  NGÀY 29 - GIỜ 3 (10:00-11:00): RECORD DEMO VIDEO          │
│  ├─ Setup: Lighting, camera                                 │
│  ├- Record: Live demo (3 takes)                             │
│  ├─ Edit: Add narration, graphics                           │
│  └─ Export: Final 3-5 minute video                          │
│                                                              │
│  NGÀY 29 - GIỜ 4 (11:00-12:00): CREATE POSTER               │
│  ├─ Size: A1 or A0                                           │
│  ├─ Sections:                                                │
│  │  ├─ Title + Authors                                      │
│  │  ├─ Abstract                                             │
│  │  ├─ Introduction (Problem)                               │
│  │  ├─ Methods (System architecture)                        │
│  │  ├─ Results (Validation)                                │
│  │  ├─ Discussion (Impact)                                  │
│  │  └─ Conclusion + Future work                             │
│  │                                                         │
│  └─ Design: Clean, professional                             │
│                                                              │
│  NGÀY 29 - GIỜ 5-6 (13:00-15:00): WRITE REPORT              │
│  ├─ Sections:                                                │
│  │  ├─ Introduction                                        │
│  │  ├─ Literature Review                                   │
│  │  ├─ Methods                                             │
│  │  ├─ Results                                             │
│  │  ├─ Discussion                                          │
│  │  └─ Conclusion                                          │
│  └─ Length: 10-15 pages                                     │
│                                                              │
│  NGÀY 30-31: REHEARSAL + FINAL PREP                        │
│  ├─ Rehearse: Demo presentation                              │
│  ├─ Prepare: Q&A responses                                   │
│  └─ Finalize: All materials                                 │
│                                                              │
│  NGÀY 32: SUBMIT                                             │
│  ☑ All materials ready                                     │
│  ☑ System demo ready                                       │
│  ☑ Video ready                                            │
│  ☑ Poster ready                                            │
│  ☑ Report ready                                            │
│  ☑ Validation reports ready                               │
│  ☑ Doctor letters ready                                   │
│                                                              │
│  TARGET: GIẢI NHÌ QUỐC GIA ✅                              │
└─────────────────────────────────────────────────────────────┘
```

---

# 7. TÀI LIỆU CẦN HỌC VÀ THUỘC LÒNG

## 7.1. LÝ THUYẾT CẦN NẮM RÕ

```
┌─────────────────────────────────────────────────────────────┐
│  KIẾN THỨC CẦN NẮM (PHỎNG VẤN)                            │
├─────────────────────────────────────────────────────────────┤
│  Y SINH:                                                     │
│  ├─ NIHSS Scale: Cách tính từng item                       │
│  ├─ FAST Criteria: Face, Arm, Speech, Time                 │
│  ├─ Stroke types: Ischemic vs Hemorrhagic                   │
│  ├─ Golden window: 3-4.5 hours                              │
│  └─ Stroke statistics: VN data                             │
│                                                              │
│  COMPUTER VISION:                                            │
│  ├─ MediaPipe Face Mesh: 468 landmarks                    │
│  ├─ YOLOv8 Pose: 17 keypoints                              │
│  ├─ CLAHE: Xử lý ánh sáng                                   │
│  └─ Real-time processing constraints                       │
│                                                              │
│  AUDIO PROCESSING:                                           │
│  ├─ Vosk STT: Offline speech recognition                    │
│  ├─ MFCC: Mel-frequency cepstral coefficients             │
│  ├─ Jitter/Shimmer: Voice quality measures                  │
│  └─ WPM: Words per minute calculation                      │
│                                                              │
│  MACHINE LEARNING:                                           │
│  ├─ PyTorch + CUDA: GPU acceleration                       │
│  ├─ Bayesian fusion: Hierarchical modeling                 │
│  └─ False alarm suppression techniques                     │
│                                                              │
│  RADAR:                                                      │
│  ├─ LD2450: 24GHz FMCW operation                           │
│  ├─ Fall detection: Proxy methods                          │
│  └─ UART communication                                     │
│                                                              │
│  VALIDATION:                                                 │
│  ├─ NIHSS correlation: Pearson r, p-value                   │
│  ├─ Confusion matrix: Sensitivity, Specificity             │
│  └─ Clinical validation: Ethical considerations            │
└─────────────────────────────────────────────────────────────┘
```

## 7.2. CÂU HỎI PHỎNG VẤN THƯỜNG GẶP

```
┌─────────────────────────────────────────────────────────────┐
│  Q&A PREPARATION                                           │
├─────────────────────────────────────────────────────────────┤
│  1. "Tại sao gọi là 'Nhận diện' không phải 'Cảnh báo'?"    │
│     └─ "Cảnh báo sớm" = đo yếu tố nguy cơ (huyết áp...)   │
│        "Nhận diện" = phát hiện triệu chứng đột quỵ        │
│                                                              │
│  2. "Làm sao NIHSS estimation accurate?"                 │
│     └─ Mapping từ module scores (validated) → NIHSS items  │
│        Correlation r = 0.85 với doctor scoring             │
│                                                              │
│  3. "Giảm false alarm bằng cách nào?"                     │
│     └─ 4-Layer Defense: Calibration → Context → Temporal │
│        → Adaptive                                         │
│                                                              │
│  4. "Chạy được trên RTX3050?"                             │
│     └─ Yes! YOLOv8n version, optimized to 15-20 FPS       │
│                                                              │
│  5. "Novelty ở đâu?"                                       │
│     └─ NIHSS estimation tại nhà + Pre-hospital handoff   │
│        (Chưa có hệ thống nào làm cả 2)                     │
│                                                              │
│  6. "Validation như thế nào?"                             │
│     └─ 50 videos for NIHSS correlation                  │
│        200 scenarios for confusion matrix                 │
│        3 homes for real-world testing                     │
│        Doctor letters confirming system review            │
│                                                              │
│  7. "Tính thực tiễn?"                                      │
│     └─ Chi phí 1.15 triệu vs 5-15 triệu (smartwatch)     │
│        Giảm handoff time 15 min → 5 min                     │
│        Có thể cứu 30.000 người/năm                         │
│                                                              │
│  8. "Hạn chế?"                                             │
│     └─ Night mode: chỉ radar, giảm accuracy               │
│        Radar limitations: không có z-axis                   │
│        Cần validate trên larger Vietnamese population      │
└─────────────────────────────────────────────────────────────┘
```

---

# 8. CHECKLIST HÀNG NGÀY

```
┌─────────────────────────────────────────────────────────────┐
│  DAILY CHECKLIST (MỖI NGÀY)                                │
├─────────────────────────────────────────────────────────────┤
│  ☑ Check task hôm nay                                     │
│  ☑ Update progress                                         │
│  ☑ Document learned concepts                              │
│  ☑ Note questions for research                             │
│  ☑ Test code thoroughly                                    │
│  ☑ Commit code to GitHub                                   │
│  ☑ Prepare for tomorrow                                   │
│                                                              │
│  WEEKLY CHECKLIST (MỖI TUẦN)                               │
│  ├─ Review weekly progress                                │
│  ├─ Update documentation                                   │
│  ├─ Identify gaps                                          │
│  ├─ Plan next week                                         │
│  └─ Self-assessment: Am I on track?                       │
└─────────────────────────────────────────────────────────────┘
```

---

# 🎯 KẾT LUẬN

32 ngày × 6 tiếng = 192 tiếng

**MỤC TIÊU:**
- ✅ Cấp trường: PASS
- ✅ Vòng phỏng vấn TPHCM: TOP 13
- ✅ Đại diện TPHCM quốc gia: POSSIBLE
- ✅ Giải nhì quốc gia: ACHIEVABLE

**KEY TO SUCCESS:**
1. Discipline: Theo lịch 100%
2. Focus: Coding 6 tiếng/ngày
3. Quality: Validate nghiêm túc
4. Documentation: Ghi chép tất cả
5. Mindset: Không bỏ cuộc

---

## 📞 SUPPORT

Khi gặp khó khăn:
1. Đọc lại tài liệu học
2. Search StackOverflow
3. Ask GVHD (tôi)
4. Don't give up!

---

**LET'S DO THIS! Target: GIẢI NHÌ QUỐC GIA! 🏆**

