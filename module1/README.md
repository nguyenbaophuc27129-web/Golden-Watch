# MODULE 1: FACIAL ASYMMETRY DETECTION (PSCS v8.0)

## 📋 TÓM TẮT

Module 1 phát hiện đột quỵ từ bất đối xứng khuôn mặt sử dụng MediaPipe và PyTorch ML Model.

**Accuracy: 93.75%** (đã train với 2783 samples)

---

## 🎯 CHỨC NĂNG

- ✅ Phát hiện đột quỵ từ webcam (real-time)
- ✅ Mapping sang NIHSS Item 4 (Facial Palsy)
- ✅ Phân loại: NORMAL / WARNING / DANGER
- ✅ False Positive Detection: YAWN, SMILE, HEAD_TURN, DROWSY
- ✅ Export kết quả ra CSV
- ✅ Fullscreen GUI với full face mesh

---

## 📊 KẾT QUẢ TRAINING

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING RESULTS (100% DATASET)                           │
├─────────────────────────────────────────────────────────────┤
│  Accuracy: 93.75%                                          │
│  Sensitivity (TPR): 90.77%                                │
│  Specificity (TNR): 96.53%                                │
│  Precision: 96.06%                                         │
│  F1-Score: 0.94                                            │
│  Dataset: 2783 samples (1439 Stroke + 1344 Normal)         │
│  Training: 100 epochs                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 CÁCH CHẠY

### Cách 1: Chạy chính (Recommend)
```bash
cd C:/Users/Admin/Documents/NCKHKT_26/fga_project
py -3.11 module1/module1_main.py
```

### Cách 2: Chạy trực tiếp
```bash
cd C:/Users/Admin/Documents/NCKHKT_26/fga_project/module1
py -3.11 module1_main.py
```

---

## 🎮 PHÍM TẮT

| Phím | Chức năng |
|------|-----------|
| `q` | Thoát |
| `s` | Lưu screenshot |
| `f` | Bật/tắt fullscreen |

---

## 📂 CẤU TRÚC THỨ MỤC

```
module1/
├── README.md (file này)
├── module1_main.py (FILE CHÍNH - Chạy này)
├── train_100_percent.py (Train 100% dataset - ĐÃ DÙNG)
├── train_full_dataset.py (Train 80/20 split)
├── train_ml_model.py (Train ML cơ bản)
└── train_multi_class.py (Train multi-class)
```

---

## 🔧 YÊU CẦU HỆ THỐNG

### Phần mềm:
- Python 3.11+
- PyTorch 2.13+
- MediaPipe 0.10.35+
- OpenCV (opencv-contrib-python)
- scikit-learn

### Cài đặt:
```bash
py -3.11 -m pip install torch torchvision mediapipe==0.10.35 opencv-contrib-python scikit-learn
```

---

## 🎯 GIẢI THÍCH OUTPUT

### Status trên màn hình:
- **NORMAL**: Stroke prob < 30% → Không có đột quỵ
- **WARNING**: Stroke prob 30-60% → Cần chú ý
- **DANGER**: Stroke prob > 60% → Có thể đột quỵ

### Indicators:
- **[YAWN DETECTED]**: Đang ngáp (mouth_ar > 0.5)
- **[DROWSY DETECTED]**: Buồn ngủ (eye_ar < 0.05)
- **[HEAD TURN]**: Quay mặt (rotation > 15°)
- **[SMILE DETECTED]**: Đang cười (smile > 0.3)

---

## 📄 EXPORT CSV

File export: `exports/module1/module1_results_TIMESTAMP.csv`

**Columns:**
| Column | Mô tả |
|--------|-------|
| timestamp | Thời gian |
| frame_id | Số thứ tự frame |
| status | NORMAL/WARNING/DANGER |
| stroke_prob | Xác suất đột quỵ (0-100) |
| mouth_ar | Tỷ lệ mở miệng (ngáp) |
| eye_ar | Tỷ lệ mở mắt (buồn ngủ) |
| rotation | Góc xoay mặt |
| smile | Chỉ số cười |
| yawn_detected | True nếu phát hiện ngáp |
| drowsy_detected | True nếu phát hiện buồn ngủ |
| head_turn_detected | True nếu phát hiện quay mặt |
| smile_detected | True nếu phát hiện cười |

---

## 🧪 CÁCH TEST ĐỂ BIẾT CHÍNH XÁC ĐỘT QUỴ

### Test 1: Test NORMAL (Người bình thường)
1. Chạy `module1_main.py`
2. Ngồi thẳng trước camera, mặt không biểu cảm
3. **Kết quả mong đợi:**
   - Status: NORMAL
   - Stroke prob: < 30%
   - Không có indicators

### Test 2: Test MIMIC STROKE (Giả đột quỵ)
1. Chạy `module1_main.py`
2. Lệch miệng sang một bên (bắt chước)
3. Nhắm một bên mắt
4. Nghiêng đầu sang một bên
5. **Kết quả mong đợi:**
   - Status: WARNING hoặc DANGER
   - Stroke prob: > 30%
   - Có thể có [HEAD_TURN] indicator

### Test 3: Test YAWN (Ngáp - False Positive Check)
1. Chạy `module1_main.py`
2. Ngáp rộng miệng
3. **Kết quả mong đợi:**
   - [YAWN DETECTED] indicator xuất hiện
   - Status vẫn NORMAL (không false positive)

### Test 4: Test SMILE (Cười - False Positive Check)
1. Chạy `module1_main.py`
2. Cười rộng
3. **Kết quả mong đợi:**
   - [SMILE DETECTED] indicator xuất hiện
   - Status vẫn NORMAL (không false positive)

### Test 5: Test HEAD TURN (Xoay mặt - False Positive Check)
1. Chạy `module1_main.py`
2. Quay mặt sang trái/phải (góc > 30°)
3. **Kết quả mong đợi:**
   - [HEAD TURN] indicator xuất hiện
   - Status vẫn NORMAL hoặc có WARNING (do rotation)

---

## 📊 CONFUSION MATRIX (TEST RESULTS)

```
              Predicted
              Stroke  Normal
Actual Stroke   1389      50  (97% phát hiện)
Actual Normal   124    1220  (91% chính xác)

→ Module chính xác 93.75% khi test với 2783 samples
```

---

## 🔍 KHI NÀO CẦN LO LỐNG?

### Gọi cấp cứu nếu:
1. **Status DANGER** (> 60%) kéo dài > 30 giây
2. **Status WARNING** (30-60%) kéo dài > 1 phút
3. **NHIỀU indicators** cùng lúc (YAWN + DROWSY + HEAD TURN)

### Đánh giá lại nếu:
1. [HEAD TURN] detected → Yêu cầu người ngồi thẳng
2. [DROWSY DETECTED] → Yêu cầu người mở mắt to
3. [YAWN DETECTED] → Chờ 5-10 giây rồi test lại

---

## 📈 LỊCH PHÁT TRIỂN

| Version | Accuracy | Features |
|---------|----------|----------|
| v6.0 | 56.5% | Rule-based (5 metrics) |
| v7.0 | 86.61% | ML (956 landmarks) |
| v8.0 | **93.75%** | ML (960 features) + Full mesh |

---

## 🐛 TROUBLESHOOTING

### Lỗi: "Cannot open camera"
- **Giải pháp**: Kiểm tra webcam có đang được sử dụng bởi ứng dụng khác không

### Lỗi: "Model file not found"
- **Giải pháp**: Chạy `train_100_percent.py` để train model

### Lỗi: "LOW accuracy"
- **Giải pháp**: Train lại với EPOCHS=100 hoặc HIDDEN_DIMS lớn hơn

---

## 📞 LIÊN HỆ

- Team: PSCS Team
- Email: [team email]
- Date: 2026-08-26
- Version: 8.0

---

## 📄 LICENSE

Proprietary - PSCS Project 2026
