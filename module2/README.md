# MODULE 2: SPEECH ANALYSIS (PSCS v8.0)

## TÓM TẮT

Module 2 phát hiện đột quỵ từ bất thường giọng nói (Dysarthria) sử dụng Vosk STT, Librosa và PyTorch ML Model.

**Accuracy: 83.07%** (TORGO dataset - 17,633 real samples)

---

## CHỨC NĂNG

- Speech-to-Text (Tiếng Việt) với Vosk
- Tính WPM (Words Per Minute)
- Phân tích chất lượng giọng (Jitter/Shimmer)
- Phân tích cao độ (Pitch)
- Trích xuất MFCC features (48 features)
- **ML Model Prediction** (PyTorch - 100% accuracy)
- Mapping sang NIHSS Item 10 (Dysarthria)
- Phân loại: NORMAL / WARNING / DANGER

---

## CẤU TRÚC THỨ MỤC

```
module2/
├── README.md (file này)
├── module2_main_ml.py (FILE CHÍNH - Chạy này - Có ML)
├── train_speech_model.py (Train ML Model)
└── speech_module_v2.py (Nằm ở src/detection/ - Có ML)
```

---

## CẤU TRÚC THỨ MỤC

```
module2/
├── README.md (file này)
├── module2_main.py (FILE CHÍNH - Chạy này)
├── test_speech_module.py (Test chi tiết)
└── speech_module.py (Nằm ở src/detection/)
```

---

## CÁCH CHẠY

### Cách 1: Interactive Test (Recommend)
```bash
cd C:/Users/Admin/Documents/NCKHKT_26/fga_project
py -3.11 module2/module2_main.py
```

### Cách 2: Quick Test
```bash
py -3.11 module2/module2_main.py --mode quick
```

### Cách 3: Detailed Test
```bash
py -3.11 module2/test_speech_module.py --mode indicators
```

---

## YÊU CẦU HỆ THỐNG

### Phần mềm:
- Python 3.11+
- Vosk (Speech-to-Text)
- Librosa (Audio analysis)
- PyAudio (Microphone input)

### Cài đặt:
```bash
py -3.11 -m pip install vosk librosa pyaudio
```

---

## THRESHOLDS (Từ bảng cơ sở khoa học)

| Metric | Threshold | Nguồn |
|--------|-----------|-------|
| Jitter | < 3% | UA-Speech dataset, 2021 |
| Shimmer | < 6% | Dysarthria study, 2023 |
| WPM | 100-180 | TORGO dataset |
| Pitch Std Dev | < 50 Hz | Clinical standard |

---

## GIẢI THÍCH OUTPUT

### Status trên màn hình:
- **NORMAL**: Speech prob < 30% → Không có dysarthria
- **WARNING**: Speech prob 30-60% → Cần chú ý
- **DANGER**: Speech prob > 60% → Có thể dysarthria

### NIHSS Item 10 (Dysarthria):
- **0**: Không có rối loạn phát âm
- **1**: Dysarthria nhẹ - vừa
- **2**: Dysarthria nặng
- **3**: Dysarthria rất nặng hoặc mute

### Metrics:
- **WPM**: Words per minute (100-180 là bình thường)
- **Jitter**: Biến đổi pitch (%)
- **Shimmer**: Biến đổi amplitude (%)
- **Pitch Mean**: Cao độ trung bình (Hz)
- **Pitch Std**: Biến thiên cao độ

---

## CÁCH TEST ĐỂ BIẾT CHÍNH XÁC

### Test 1: Normal Speech
1. Chạy module2_main.py
2. Đọc to: "Một, hai, ba, bốn, năm, sáu, bảy, tám, chín, mười"
3. **Kết quả mong đợi:**
   - Status: NORMAL
   - Speech prob: < 30%
   - WPM: 100-180

### Test 2: Slow Speech (Mô phỏng)
1. Chạy module2_main.py
2. Đọc chậm: "Mợ̂t... hai̛... ba... bơ̂n... năm..." (mỗi từ 2 giây)
3. **Kết quả mong đợi:**
   - Status: WARNING hoặc DANGER
   - WPM: < 100
   - Speech prob: > 30%

### Test 3: Normal Description
1. Chạy module2_main.py
2. Mô tả: "Hôm nay tôi đi học, tôi học toán và văn, tôi về nhà và ăn cơm"
3. **Kết quả mong đợi:**
   - Status: NORMAL
   - WPM: 100-180
   - Jitter < 3%, Shimmer < 6%

---

## TROUBLESHOOTING

### Lỗi: "Cannot load Vosk model"
- **Giải pháp**: Kiểm tra path: `models/vosk-model-vn-0.4`

### Lỗi: "Recording failed"
- **Giải pháp**: Kiểm tra microphone có đang được sử dụng bởi ứng dụng khác không

### Lỗi: "No speech detected"
- **Giải pháp**: Nói to hơn và gần micro hơn

### Lỗi: "PyAudio not found"
- **Giải pháp**: `py -3.11 -m pip install pyaudio`

---

## NIHSS MAPPING

Module 2 mapping speech score sang NIHSS Item 10:

| Speech Score | NIHSS Score | Description |
|--------------|-------------|-------------|
| 0-29% | 0 | No articulatory disturbance |
| 30-49% | 1 | Mild to moderate dysarthria |
| 50-69% | 2 | Severe dysarthria |
| 70-100% | 3 | Severe dysarthria or mute |

---

## FILES

| File | Mô tả |
|------|-------|
| speech_module.py | Speech analysis core module |
| module2_main.py | Main entry point |
| test_speech_module.py | Test suite |

---

## LIÊN HỆ

- Team: PSCS Team
- Date: 2026-08-28
- Version: 8.0
