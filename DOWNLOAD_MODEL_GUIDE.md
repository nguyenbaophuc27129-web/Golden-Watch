# HƯỚNG DẪN DOWNLOAD FACE LANDMARKER MODEL CHO v7.0

## CÁCH 1: TỪ KAGGLE (KHUYẾN NGHỊ - NHANH NHẤT)

### Bước 1: Đăng nhập Kaggle
1. Vào: https://www.kaggle.com/
2. Đăng ký tài khoản (miễn phí)
3. Đăng nhập

### Bước 2: Accept Terms
1. Vào: https://www.kaggle.com/models/google/mediapipe/framework/pyTorch/variations/face-landmarker-v2
2. Click "Use" hoặc "Download"
3. Chọn "PyTorch" framework
4. Chọn version mới nhất

### Bước 3: Download
1. Click "Download" button
2. File: `face_landmarker_v2.task` (~10MB)
3. Lưu vào: `C:\Users\Admin\Documents\NCKHKT_26\fga_project\models\`

### Bước 4: Verify
```bash
# Check file size (phải ~10MB)
dir C:\Users\Admin\Documents\NCKHKT_26\fga_project\models\face_landmarker_v2.task

# Hoặc trong Python:
python -c "import os; size = os.path.getsize('models/face_landmarker_v2.task'); print(f'Size: {size/1024/1024:.2f} MB')"
```

---

## CÁCH 2: TỐI ƯU - DÙNG v6.0 TRONG LUC CHỜ MODEL

Nếu không muốn download ngay, có thể dùng v6.0 tạm thời:

### Option A: Dùng Python 3.11 + MediaPipe 0.9.x
```bash
# Cần cài Python 3.11 (không dùng 3.13)
py -3.11 -m pip install mediapipe==0.9.0.1

# Chạy face_module_v6.py
py -3.11 src/detection/face_module_v6.py
```

### Option B: Mock test (không detect thật)
- Dùng code v7.0 nhưng sẽ chạy ở "NO_DETECTOR" mode
- Trả về dummy data để test UI/UX

---

## CÁCH 3: BUILD TỪ SOURCE (PHỨC TẠP)

Không khuyến khích - cần:
1. Cài Bazel
2. Clone MediaPipe repo
3. Build từ source

---

## SAU KHI CÓ MODEL FILE:

### Test v7.0
```bash
cd C:\Users\Admin\Documents\NCKHKT_26\fga_project
python src/detection/face_module_v7.py
```

### Kết quả mong đợi:
```
============================================================
FACE ASYMMETRY DETECTION v7.0 - TEST MODE
============================================================

Initializing detector...
[OK] MediaPipe FaceLandmarker v7.0 initialized

Opening webcam...
[OK] Webcam opened successfully

Starting detection... (Press 'q' to quit)
```

### Controls:
- `[q]` - Quit
- `[s]` - Save frame with landmarks

---

## TROUBLESHOOTING:

### Lỗi "ExternalFile must specify..."
→ Model file không tồn tại hoặc path sai

### Lỗi "Status luôn NO_DETECTOR"
→ MediaPipe không khởi tạo được, check model file size

### Lỗi "UnicodeEncodeError"
→ Set terminal UTF-8: `chcp 65001` (Windows)

---

## FILE STRUCTURE SAU KHI DOWNLOAD:

```
fga_project/
├── models/
│   └── face_landmarker_v2.task  <-- File này (~10MB)
├── src/
│   ├── detection/
│   │   ├── face_module_v6.py
│   │   └── face_module_v7.py
│   └── training/
│       └── train_face_model_rtx3050.py
└── ...
```

---

## THAM KHẢO:

- Kaggle Model: https://www.kaggle.com/models/google/mediapipe
- MediaPipe GitHub: https://github.com/google-ai-edge/mediapipe
- Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
