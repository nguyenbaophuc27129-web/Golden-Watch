# MODELS — file weights KHÔNG kèm trong bộ code

Bộ `Golden_Watch_KHKT` chỉ chứa CODE chính thức. Để chạy được, copy các file
weights sau từ repo làm việc `fga_project/` vào đúng vị trí dưới đây:

## Bắt buộc (ML models + MediaPipe)

| Copy từ `fga_project/` | Vào `Golden_Watch_KHKT/` | Dùng cho |
|---|---|---|
| `models/speech_torgo_20260828_211130.pth` + `_scaler.pkl` | `models/` | M2 Speech (TORGO 83.07%) |
| `models/arm_weakness_20260830_200657.pth` + `_scaler.pkl` | `models/` | M3 Arm (99.50%) |
| `models/gait_classifier_20260829_120925.pth` + `_scaler.pkl` | `models/` | M4 Gait (86.13%) |
| `models/face_landmarker_v2.task` | `models/` | M1 Face (MediaPipe) |
| `src/yolov8n-pose.pt` | `src/yolov8n-pose.pt` | M3 + M4 (YOLO pose) |

## Tùy chọn

| Copy từ | Vào | Dùng cho |
|---|---|---|
| `models/vosk-model-vn-0.4/` (~45MB) | `models/vosk-model-vn-0.4/` | M2 speech-to-text tiếng Việt (đếm WPM) |

Lệnh copy nhanh (từ thư mục cha của cả 2 project):

```bash
cd fga_project
cp models/speech_torgo_20260828_211130.pth* ../Golden_Watch_KHKT/models/
cp models/arm_weakness_20260830_200657.pth* ../Golden_Watch_KHKT/models/
cp models/gait_classifier_20260829_120925.pth* ../Golden_Watch_KHKT/models/
cp models/face_landmarker_v2.task ../Golden_Watch_KHKT/models/
cp src/yolov8n-pose.pt ../Golden_Watch_KHKT/src/
```

Lý do không kèm weights: .pth/.task nặng (~200MB), nộp hồ sơ KHKT gửi code +
báo cáo + video; weights demo trực tiếp trên máy thi công trình.
