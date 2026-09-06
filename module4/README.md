# MODULE 4: GAIT ABNORMALITY DETECTION (PSCS v8.0)

## TÓM TẮT

Module 4 phát hiện bất thường dáng đi (Gait Abnormality) do đột quỵ sử dụng time series analysis và pose estimation.

---

## CHỨC NĂNG

- **Time Series Analysis**: Phân tích dữ liệu gait từ file .txt (stride interval time series)
- **ML Classification**: Sử dụng MLP đã train trên Gait in Aging Dataset
- **Real-time Detection**: Phát hiện dáng đi từ camera bằng YOLOv8n-Pose
- **8 Gait Features**:
  1. Stride length (độ dài bước chân)
  2. Cadence (tốc độ bước chân)
  3. Stride time variability (biến thiên thời gian bước)
  4. Magnitude variability (biến thiên độ lớn)
  5. Velocity (tốc độ)
  6. Acceleration (gia tốc)
  7. Regularity (tính đều đặn)
  8. Symmetry (tính đối xứng)
- Mapping sang NIHSS Item 6 (Motor Leg)
- Phân loại: NORMAL / WARNING / DANGER

---

## CẤU TRÚC THỨ MỤC

```
module4/
├── README.md (file này)
├── module4_main.py (FILE CHÍNH - Chạy này)
└── train_gait_model.py (Training script)

src/detection/
└── gait_module.py (Module chính)
```

---

## CÁCH CHẠY

### Mode 1: Test với Dataset Samples (Khuyến nghị)
```bash
cd C:/Users/Admin/Documents/NCKHKT_26/fga_project
py -3.11 module4/module4_main.py --mode samples
```

### Mode 2: Test với Single File
```bash
py -3.11 module4/module4_main.py --mode file --file "path/to/gait/file.txt"
```

### Mode 3: Test với Webcam (Real-time)
```bash
py -3.11 module4/module4_main.py --mode webcam --duration 10
```

---

## YÊU CẦU HỆ THỐNG

### Phần mềm:
- Python 3.11+
- PyTorch
- OpenCV (opencv-python)
- NumPy
- Scikit-learn
- Ultralytics (cho YOLOv8n-Pose)

### Cài đặt:
```bash
py -3.11 -m pip install torch opencv-python numpy scikit-learn ultralytics
```

---

## GAIT IN AGING AND DISEASE DATASET

### Dataset Structure:
```
gait-in-aging-and-disease-database-1.0.0/
├── y1-y5-si.txt    # 5 healthy young adults (23-29 yrs) - Normal
├── o1-o5-si.txt    # 5 healthy old adults (71-77 yrs) - Normal
└── pd1-pd5-si.txt  # 5 Parkinson's patients (60-77 yrs) - Abnormal
```

### Data Format:
- 2 columns: [time (seconds), stride interval (seconds)]
- Stride interval = thời gian giữa các lần chân tương tự chạm đất

### Training Results:
```
Dataset: Gait in Aging and Disease (v2 with sliding window)
Samples: 162 (145 normal, 17 abnormal)
Window size: 100, Stride: 50

Training Results:
- Accuracy:  96.88%
- Precision: 75.00%
- Recall:    100.00%
- F1-Score:  0.8571
- Sensitivity: 100.00% (Không bỏ sót case bất thường)
- Specificity: 96.55%

Model: gait_classifier_20260829_120925.pth
```

---

## THRESHOLDS:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Stride length | 0.5 - 0.8 m | Độ dài bước chân normal |
| Cadence | 100 - 130 steps/min | Tốc độ bước chân |
| Stride time var | < 0.05 | Biến thiên thời gian bước |
| Asymmetry | < 15% | Chênh lệch trái-phải |

---

## GIẢI THÍCH OUTPUT

### Status:
- **NORMAL**: Gait prob < 30% → Dáng đi bình thường
- **WARNING**: Gait prob 30-60% → Cần chú ý
- **DANGER**: Gait prob > 60% → Bất thường dáng đi

### NIHSS Item 6 (Motor Leg):
- **0**: Không drift
- **1**: Mild drift
- **2**: Có effort chống trọng lực
- **3**: Không movement chống trọng lực

### Metrics:
- **stride_length**: Độ dài bước chân (normalized)
- **cadence**: Tốc độ bước chân (steps/min)
- **stride_time_var**: Biến thiên thời gian bước
- **magnitude_var**: Biến thiên độ lớn signal
- **velocity**: Tốc độ di chuyển
- **acceleration**: Gia tốc
- **regularity**: Tính đều đặn (0-1)
- **symmetry**: Tính đối xứng (0-1)

---

## CÁCH TEST ĐỂ BIẾT CHÍNH XÁC

### Test 1: Normal Gait Samples
```bash
py -3.11 module4/module4_main.py --mode samples
```
**Kết quả mong đợi:**
- o1-o5, y1-y5: Status NORMAL, Gait prob < 30%
- pd1-pd5: Status DANGER, Gait prob > 60%

### Test 2: Real-time Detection
```bash
py -3.11 module4/module4_main.py --mode webcam --duration 10
```
**Hướng dẫn:**
1. Đứng trước camera, toàn thân hiển thị
2. Đi bộ tại chỗ hoặc đi ngang qua camera
3. Quan sát kết quả trên màn hình

---

## TROUBLESHOOTING

### Lỗi: "Cannot load ML model"
- **Giải pháp**: Kiểm tra path: `gait_classifier_20260829_120925.pth`

### Lỗi: "Cannot open camera"
- **Giải pháp**: Kiểm tra webcam có đang được sử dụng bởi ứng dụng khác không

### Lỗi: "No person detected"
- **Giải pháp**: Đảm bảo toàn thân hiển thị trong camera

### Lỗi: "ultralytics not found"
- **Giải pháp**: `py -3.11 -m pip install ultralytics`

---

## TRAINING MODEL

Để train lại model với dataset:

```bash
py -3.11 module4/train_gait_model.py --gait_path "path/to/gait/dataset"
```

**Training Parameters:**
- Epochs: 150 (with early stopping)
- Batch size: 32
- Learning rate: 0.001
- Hidden dims: [64, 32]
- Dropout: 0.5
- Weight decay: 0.001
- Window size: 100
- Stride: 50

---

## NIHSS MAPPING:

Module 4 mapping gait abnormality score sang NIHSS Item 6:

| Gait Score | NIHSS Score | Description |
|-----------|-------------|-------------|
| 0-29% | 0 | No drift |
| 30-49% | 1 | Mild drift |
| 50-69% | 2 | Some effort against gravity |
| 70-100% | 3 | No movement against gravity |

---

## FILES:

| File | Mô tả |
|------|-------|
| gait_module.py | Gait detection core module |
| module4_main.py | Main entry point |
| train_gait_model.py | Training script |

---

## LIÊN HỆ:

- Team: PSCS Team
- Date: 2026-08-29
- Version: 8.0
