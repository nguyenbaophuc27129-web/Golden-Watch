# MODULE 5: VISUAL FIELD / EYE MOVEMENT DETECTION (PSCS v8.0)

## TÓM TẮT

Module 5 phát hiện khiếm thị trường (Visual Field Deficit) và bất thường vận chuyển mắt do đột quỵ sử dụng eye tracking.

---

## CHỨC NĂNG

- **Eye Tracking**: Theo dõi chuyển động mắt bằng MediaPipe Face Mesh
- **Gaze Direction Detection**: Phát hiện hướng nhìn (left, right, up, down)
- **Eye Openness Analysis**: Phân tích độ mở mí mắt (eye aspect ratio)
- **Pupil Symmetry Check**: Kiểm tra đối xứng đồng tử
- **Gaze Asymmetry Detection**: Phát hiện bất đối xứng hướng nhìn giữa 2 mắt
- **Blink Detection**: Phát hiện chớp mắt
- **Visual Field Test**: Test thị trường theo 5 hướng
- Mapping sang NIHSS Item 4 (Best Gaze) & Item 5 (Visual Fields)
- Phân loại: NORMAL / WARNING / DANGER

---

## CẤU TRÚC THỨ MỤC

```
module5/
├── README.md (file này)
└── module5_main.py (FILE CHÍNH - Chạy này)

src/detection/
└── visual_module.py (Module chính)
```

---

## CÁCH CHẠY

### Cách 1: Full Test (15 giây - Khuyến nghị)
```bash
cd C:/Users/Admin/Documents/NCKHKT_26/fga_project
py -3.11 module5/module5_main.py
```

### Cách 2: Quick Test (1 frame)
```bash
py -3.11 module5/module5_main.py --mode quick
```

### Cách 3: Custom Duration
```bash
py -3.11 module5/module5_main.py --duration 20
```

---

## YÊU CẦU HỆ THỐNG

### Phần mềm:
- Python 3.11+
- MediaPipe (cho Face Mesh)
- OpenCV (opencv-python)
- NumPy
- SciPy (cho ConvexHull)

### Cài đặt:
```bash
# MediaPipe có vấn đề với version 1.0+ (API thay đổi)
# Cần cài đặt version cũ hơn nếu có thể
py -3.11 -m pip install opencv-python numpy scipy
# mediapipe sẽ dùng fallback mode nếu API không tương thích
```

---

## MEDIAPIPE FACE MESH - EYE LANDMARKS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MediaPipe Face Mesh - Eye Landmarks                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Left Eye (16 landmarks):                                                   │
│    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161  │
│                                                                              │
│  Right Eye (16 landmarks):                                                  │
│    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385   │
│                                                                              │
│  Iris (5 landmarks each):                                                   │
│    Left: 468, 469, 470, 471, 472                                           │
│    Right: 473, 474, 475, 476, 477                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## THRESHOLDS:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Gaze asymmetry | < 15° | Chênh lệch hướng nhìn 2 mắt |
| Eye openness | > 0.3 | Tỷ lệ mở mắt (EAR) |
| Pupil asymmetry | < 0.2 | Chênh lệch kích thước đồng tử |
| Blink rate | 10-30/min | Tốc độ chớp mắt normal |

---

## GIẢI THÍCH OUTPUT

### Status:
- **NORMAL**: Visual prob < 30% → Thị trường bình thường
- **WARNING**: Visual prob 30-60% → Cần chú ý
- **DANGER**: Visual prob > 60% → Khiếm thị trường

### NIHSS Item 4 (Best Gaze):
- **0**: Normal - cả 2 mắt cùng hướng
- **1**: Partial gaze palsy - một mắt không hướng đầy đủ
- **2**: Forced deviation - cả 2 mắt bị ép về một phía

### NIHSS Item 5 (Visual Fields):
- **0**: Không mất thị trường
- **1**: Partial hemianopia (một phần)
- **2**: Complete hemianopia (một bên hoàn toàn)
- **3**: Bilateral hemianopia (hai bên)

### Metrics:
- **left_eye_openness**: Độ mở mắt trái (0-1)
- **right_eye_openness**: Độ mở mắt phải (0-1)
- **left_gaze_x/y**: Hướng nhìn mắt trái (-1 đến 1)
- **right_gaze_x/y**: Hướng nhìn mắt phải (-1 đến 1)
- **gaze_asymmetry**: Chênh lệch hướng nhìn (degrees)
- **pupil_asymmetry**: Chênh lệch kích thước đồng tử (0-1)
- **blink_detected**: Phát hiện chớp mắt

---

## CÁCH TEST ĐỂ BIẾT CHÍNH XÁC

### Test 1: Normal Visual Field
1. Chạy module5_main.py
2. Nhìn thẳng vào camera, theo dõi hướng dẫn
3. **Kết quả mong đợi:**
   - Status: NORMAL
   - Visual prob: < 30%
   - Gaze asymmetry: < 10°

### Test 2: Simulate Visual Field Deficit
1. Chạy module5_main.py
2. Che một bên mắt (giả lập hemianopia)
3. **Kết quả mong đợi:**
   - Status: WARNING hoặc DANGER
   - Visual prob: > 30%
   - Gaze asymmetry cao

### Test 3: Eye Movement Test
1. Chạy module5_main.py
2. Khi được hướng dẫn "Look LEFT", chỉ nhìn sang trái
3. Khi được hướng dẫn "Look RIGHT", chỉ nhìn sang phải
4. **Kết quả mong đợi:**
   - Gaze direction thay đổi theo hướng dẫn
   - Status: NORMAL nếu theo dõi tốt

---

## VISUAL FIELD TEST PROTOCOL

Thực hiện test theo 5 hướng:

```
1. Look LEFT   (3 giây)  → Test hemianopia trái
2. Look RIGHT  (3 giây)  → Test hemianopia phải
3. Look UP     (3 giây)  → Test thị trường trên
4. Look DOWN   (3 giây)  → Test thị trường dưới
5. Look CENTER (3 giây)  → Test vị trí trung tâm
```

**Tổng thời gian: ~15 giây**

---

## TROUBLESHOOTING

### Lỗi: "MediaPipe not available"
- **Giải pháp**: `py -3.11 -m pip install mediapipe`

### Lỗi: "Cannot open camera"
- **Giải pháp**: Kiểm tra webcam có đang được sử dụng bởi ứng dụng khác không

### Lỗi: "No face detected"
- **Giải pháp**: Đảm bảo mặt hiển thị rõ trong camera, đủ ánh sáng

### Lỗi: "scipy not found"
- **Giải pháp**: `py -3.11 -m pip install scipy`

---

## NIHSS MAPPING:

Module 5 mapping visual field abnormality sang NIHSS:

### NIHSS Item 4 (Best Gaze):
| Visual Score | NIHSS Score | Description |
|-------------|-------------|-------------|
| 0-29% | 0 | Normal |
| 30-59% | 1 | Partial gaze palsy |
| 60-100% | 2 | Forced deviation |

### NIHSS Item 5 (Visual Fields):
| Visual Score | NIHSS Score | Description |
|-------------|-------------|-------------|
| 0-29% | 0 | No visual loss |
| 30-49% | 1 | Partial hemianopia |
| 50-69% | 2 | Complete hemianopia |
| 70-100% | 3 | Bilateral hemianopia |

---

## STROKE SYMPTOMS DETECTED:

### 1. Hemianopia (Một nửa thị trường)
- **Symptom**: Không nhìn thấy một phía
- **Detection**: Gaze asymmetry cao, mắt không hướng về phía đó
- **Related to**: Lesion ở visual pathway (occipital lobe)

### 2. Gaze Palsy (Liệt vận động mắt)
- **Symptom**: Mắt không thể hướng về phía đó
- **Detection**: Hạn chế chuyển động mắt
- **Related to**: Lesion ở brainstem (pons, midbrain)

### 3. Ptosis (Sụp mi)
- **Symptom**: Mí mắt sụp xuống
- **Detection**: Eye openness thấp
- **Related to**: Lesion ở oculomotor nerve (CN III)

### 4. Anisocoria (Không đều đồng tử)
- **Symptom**: Đồng tử 2 bên không đều
- **Detection**: Pupil asymmetry cao
- **Related to**: Lesion ở sympathetic/parasympathetic pathways

---

## FILES:

| File | Mô tả |
|------|-------|
| visual_module.py | Visual field detection core module |
| module5_main.py | Main entry point |

---

## LIÊN HỆ:

- Team: PSCS Team
- Date: 2026-08-29
- Version: 8.0
