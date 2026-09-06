# MODULE 3: ARM WEAKNESS DETECTION (PSCS v8.0)

## TÓM TẮT

Module 3 phát hiện yếu tay (Arm Weakness) do đột quỵ sử dụng YOLOv8n-Pose cho pose estimation.

---

## CHỨC NĂNG

- Pose estimation với YOLOv8n-Pose (17 keypoints)
- Phân tích góc cánh tay (shoulder-elbow-wrist)
- Phát hiện arm drop (tay rơi)
- Phát hiện asymmetry giữa tay trái và phải
- Mapping sang NIHSS Item 5 (Motor Arm)
- Phân loại: NORMAL / WARNING / DANGER

---

## CẤU TRÚC THỨ MỤC

```
module3/
├── README.md (file này)
└── module3_main.py (FILE CHÍNH - Chạy này)

src/detection/
└── arm_module.py (Module chính)
```

---

## CÁCH CHẠY

### Cách 1: Full Test (10 giây)
```bash
cd C:/Users/Admin/Documents/NCKHKT_26/fga_project
py -3.11 module3/module3_main.py
```

### Cách 2: Quick Test (1 frame)
```bash
py -3.11 module3/module3_main.py --mode quick
```

---

## YÊU CẦU HỆ THỐNG

### Phần mềm:
- Python 3.11+
- YOLOv8n-Pose (ultralytics)
- OpenCV (opencv-python)
- NumPy

### Cài đặt:
```bash
py -3.11 -m pip install ultralytics opencv-python numpy
```

---

## YOLOV8N-POSE KEYPOINTS:

```
┌─────────────────────────────────────────────────────────────┐
│  YOLOv8n-Pose - 17 Keypoints                             │
├─────────────────────────────────────────────────────────────┤
│  0: nose        9: left_wrist                             │
│  1: left_eye    10: right_wrist                           │
│  2: right_eye   11: left_hip                              │
│  3: left_ear    12: right_hip                             │
│  4: right_ear   13: left_knee                             │
│  5: left_shoulder  14: right_knee                        │
│  6: right_shoulder 15: left_ankle                         │
│  7: left_elbow   16: right_ankle                         │
│  8: right_elbow                                            │
└─────────────────────────────────────────────────────────────┘

Arm Keypoints Used:
- Shoulder (5, 6)
- Elbow (7, 8)
- Wrist (9, 10)
```

---

## THRESHOLDS:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Arm drop | > 100 pixels | Tay rơi quá nhiều |
| Movement range | > 20 degrees | Phạm vi chuyển động |
| Asymmetry | > 25 degrees | Chênh lệch trái-phải |
| Speed ratio | < 0.5 | Tốc độ tay yếu/normal |

---

## GIẢI THÍCH OUTPUT

### Status trên màn hình:
- **NORMAL**: Arm prob < 30% → Không có yếu tay
- **WARNING**: Arm prob 30-60% → Cần chú ý
- **DANGER**: Arm prob > 60% → Có thể yếu tay

### NIHSS Item 5 (Motor Arm):
- **0**: Không drift
- **1**: Drift nhưng giữ được position
- **2**: Có effort chống trọng lực
- **3**: Không movement chống trọng lực
- **4**: Không movement

### Metrics:
- **left_arm_angle**: Góc cánh tay trái (degrees)
- **right_arm_angle**: Góc cánh tay phải (degrees)
- **left_arm_drop**: Tay trái rơi (pixels)
- **right_arm_drop**: Tay phải rơi (pixels)
- **angle_asymmetry**: Chênh lệch góc trái-phải
- **weak_arm**: Tay nào yếu (left/right/none)

---

## CÁCH TEST ĐỂ BIẾT CHÍNH XÁC

### Test 1: Normal Arms
1. Chạy module3_main.py
2. Giơ cả 2 tay lên trước ngực, giữ thẳng
3. **Kết quả mong đợi:**
   - Status: NORMAL
   - Arm prob: < 30%
   - weak_arm: none

### Test 2: Mimic Arm Weakness
1. Chạy module3_main.py
2. Giơ 1 tay (trái hoặc phải) nhưng để "rơi" xuống
3. **Kết quả mong đợi:**
   - Status: WARNING hoặc DANGER
   - Arm prob: > 30%
   - weak_arm: left hoặc right

### Test 3: Asymmetry Test
1. Chạy module3_main.py
2. Giơ tay trái cao, tay phải thấp
3. **Kết quả mong đợi:**
   - angle_asymmetry cao
   - Status: WARNING

---

## TROUBLESHOOTING

### Lỗi: "Cannot load YOLO model"
- **Giải pháp**: Kiểm tra path: `yolov8n-pose.pt`

### Lỗi: "Cannot open camera"
- **Giải pháp**: Kiểm tra webcam có đang được sử dụng bởi ứng dụng khác không

### Lỗi: "No person detected"
- **Giải pháp**: Đảm bảo toàn thân hiển thị trong camera

### Lỗi: "ultralytics not found"
- **Giải pháp**: `py -3.11 -m pip install ultralytics`

---

## NIHSS MAPPING:

Module 3 mapping arm weakness score sang NIHSS Item 5:

| Arm Score | NIHSS Score | Description |
|-----------|-------------|-------------|
| 0-29% | 0 | No drift |
| 30-49% | 1 | Drift, holds position |
| 50-69% | 2 | Some effort against gravity |
| 70-100% | 3 | No movement against gravity |

---

## FILES:

| File | Mô tả |
|------|-------|
| arm_module.py | Arm weakness detection core module |
| module3_main.py | Main entry point |

---

## LIÊN HỆ:

- Team: PSCS Team
- Date: 2026-08-29
- Version: 8.0
