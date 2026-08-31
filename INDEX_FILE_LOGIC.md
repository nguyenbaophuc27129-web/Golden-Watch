# 📚 INDEX - FILE LOGIC & GIẢI THÍCH CODE

**PSCS v8.0 - Pre-Hospital Stroke Care System**
**Tác giả:** PSCS Team
**Ngày:** 30/08/2026

---

## 📋 DANH MỤC TÀI LIỆU

```
┌─────────────────────────────────────────────────────────────┐
│  Tài liệu này giải thích logic code từng module             │
│  Mục đích: Hiểu cách hoạt động của hệ thống                │
│  Đối tượng: Nhà tuyển học, BGK, người duyệt code            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📖 CÁC FILE MODULE:

### 1️⃣ MODULE 1: FACE ASYMMETRY DETECTION
**File:** `file_logic_giai_thich_code_Module1.md`

**Nội dung:**
- 🎯 Mục tiêu: Phát hiện bất thường khuôn mặt (mặt lệch)
- 🔄 Data flow: YOLO → MediaPipe → Feature Extraction → ML → NIHSS
- 📐 Logic:
  - Face detection (YOLOv8n-face)
  - Landmark extraction (468 points)
  - Feature extraction (96 features: eyes, mouth, symmetry)
  - ML classification (PyTorch NN)
  - NIHSS mapping (Item 4: Facial Palsy)

**Key Metrics:**
- Accuracy: 93.75%
- F1 Score: 0.93
- AUC-ROC: 0.97

---

### 2️⃣ MODULE 2: SPEECH DYSARTHRIA DETECTION
**File:** `file_logic_giai_thich_code_Module2.md`

**Nội dung:**
- 🎯 Mục tiêu: Phát hiện rối loạn ngôn ngữ (dysarthria)
- 🔄 Data flow: Audio → Vosk STT → Feature Extraction → ML → NIHSS
- 📐 Logic:
  - Speech recognition (Vosk STT - Vietnamese)
  - Text features (64: speech rate, articulation, fluency)
  - Audio features (64: pitch, energy, spectral)
  - ML classification (PyTorch NN, 128 inputs)
  - NIHSS mapping (Item 9: Language, Item 10: Dysarthria)

**Key Metrics:**
- Accuracy: 83.07%
- Dataset: 17,633 samples (TORGO)
- AUC-ROC: 0.89

---

### 3️⃣ MODULE 3: ARM WEAKNESS DETECTION
**File:** `file_logic_giai_thich_code_Module3.md`

**Nội dung:**
- 🎯 Mục tiêu: Phát hiện yếu tay (arm weakness)
- 🔄 Data flow: Camera → YOLOv8n-Pose → Movement Tracking → Rule-Based → NIHSS
- 📐 Logic:
  - Pose detection (YOLOv8n-Pose, 17 keypoints)
  - Arm tracking (6 keypoints per arm)
  - Drift detection (vertical drift rate)
  - ROM analysis (range of motion)
  - Rule-based scoring (100 points total)
  - NIHSS mapping (Item 5: Motor Arm)

**Status:** ⚠️ Rule-based only - NEEDS ML model

**Target After Training:**
- Accuracy: 80-85%
- FPR: <15%

---

### 4️⃣ MODULE 4: GAIT ABNORMALITY DETECTION ⭐
**File:** `file_logic_giai_thich_code_Module4.md`

**Nội dung:**
- 🎯 Mục tiêu: Phát hiện bất thường dáng đi (gait abnormality)
- 🔄 Data flow: Time Series → Feature Extraction → ML → ROC Optimized → NIHSS
- 📐 Logic:
  - Gait feature extraction (8 features)
  - ML classification (PyTorch NN)
  - ⭐ **ROC optimized threshold: 64%**
  - Rule-based scoring (with scientific sources)
  - NIHSS mapping (Item 6: Motor Leg)

**Key Metrics:**
- Accuracy: 86.13%
- FPR: 11.3% ✅ (below 15% target!)
- ROC AUC: 0.98
- Statistical Power: 100%

**⭐ Status:** ✅ **COMPETITION-READY!** (After ROC optimization)

---

### 5️⃣ MODULE 5: VISUAL FIELD DEFECT DETECTION
**File:** `file_logic_giai_thich_code_Module5.md`

**Nội dung:**
- 🎯 Mục tiêu: Phát hiện khiếm khuyết thị trường (visual field defect)
- 🔄 Data flow: Camera → MediaPipe → Pupil Tracking → Visual Field Test → NIHSS
- 📐 Logic:
  - Eye landmark detection (MediaPipe Face Mesh)
  - Pupil detection & tracking
  - Visual field testing (8 stimuli positions)
  - Rule-based scoring (100 points total)
  - NIHSS mapping (Item 3: Visual)

**Status:** ⚠️ Rule-based only - NEEDS ML model

**Target After Training:**
- Accuracy: 80-85%
- FPR: <15%

---

## 🔗 NIHSS MAPPING (TẤT CẢ MODULES)

```
┌─────────────────────────────────────────────────────────────┐
│  NIHSS Item              Module                Score Range   │
├─────────────────────────────────────────────────────────────┤
│  Item 1a: Consciousness   Module 1 (Face)         0-3        │
│  Item 3: Visual           Module 5 (Visual)       0-3        │
│  Item 4: Facial Palsy     Module 1 (Face)         0-3        │
│  Item 5: Motor Arm        Module 3 (Arm)          0-4        │
│  Item 6: Motor Leg       Module 4 (Gait)         0-4        │
│  Item 9: Best Language    Module 2 (Speech)       0-3        │
│  Item 10: Dysarthria      Module 2 (Speech)       0-2        │
└─────────────────────────────────────────────────────────────┘
```

**Tổng NIHSS Score:** 0-42 points
- **0-4:** Minor stroke
- **5-15:** Moderate stroke
- **16-20:** Moderate to severe
- **21-42:** Severe stroke

---

## 📊 OVERALL SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ Camera  │ │ Micro   │ │ Camera  │ │ Camera  │          │
│  │ Module1 │ │ Module2 │ │ Module3 │ │ Module4 │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│       │           │           │           │                  │
│       ▼           ▼           ▼           ▼                  │
│  ┌─────────────────────────────────────────────────┐     │
│  │         DETECTION MODULES (5 modules)             │     │
│  │  - Face Asymmetry                                 │     │
│  │  - Speech Dysarthria                              │     │
│  │  - Arm Weakness                                  │     │
│  │  - Gait Abnormality ⭐ FIXED                     │     │
│  │  - Visual Field Defect                           │     │
│  └─────────────────────────────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────┐     │
│  │         RESULT AGGREGATION                        │     │
│  │  - Individual module scores                       │     │
│  │  - NIHSS item mapping                            │     │
│  │  - Overall stroke probability                     │     │
│  └─────────────────────────────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────┐     │
│  │         OUTPUT & ALERTS                           │     │
│  │  - Stroke probability (0-100%)                   │     │
│  │  - NIHSS total score (0-42)                      │     │
│  │  - Hospital recommendation                        │     │
│  │  - Emergency alert (if stroke detected)           │     │
│  └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 TRẠNG THÁI MODULES

```
┌─────────────────────────────────────────────────────────────┐
│  Module  Status   ML Model  Accuracy  FPR    Competition    │
├─────────────────────────────────────────────────────────────┤
│  M1 Face  ✅ READY   ✅ Yes    93.75%   3.5%   ✅ YES       │
│  M2 Speech ✅ READY   ✅ Yes    83.07%   8.5%   ✅ YES       │
│  M3 Arm   ⚠️ WEAK   ❌ No      N/A      ~15%   ⚠️ NEEDS    │
│  M4 Gait  ✅ READY   ✅ Yes    86.13%   11.3%  ✅ YES ⭐    │
│  M5 Visual ⚠️ WEAK   ❌ No      N/A      ~20%   ⚠️ NEEDS    │
└─────────────────────────────────────────────────────────────┘

⭐ = Module 4 FIXED with ROC optimization (30/08/2026)
```

---

## 📚 THAM KHẢO THÊM

### Technical Documentation:
- `QUY_TRAC_BAT_DAU_NGAY.md` - Quick start guide
- `TONG_QUAN_DU_AN_FINAL.md` - Project overview
- `NANG_CAP.md` - Week 2 upgrade plan
- `THANG_DO_CHUAN_QUOC_TE.md` - International standards

### Validation Documents:
- `TONG_HOP_DANH_GIA_FINAL.md` - Overall assessment
- `NGAY1_RESULT_MODULE4_FIX.md` - Module 4 fix results

### Test Scripts:
- `test_all_modules.py` - System test
- `roc_analysis_module4.py` - ROC analysis script
- `comprehensive_test_module4.py` - Module 4 validation

---

## 🔍 CÁCH ĐỌC TÀI LIỆU NÀY

**Để hiểu từng module:**
1. Đọc file logic của module tương ứng
2. Xem "DATA FLOW" để hiểu quy trình
3. Xem "LOGIC CỐT LÕI" để hiểu chi tiết từng hàm
4. Xem "THRESHOLDS" để hiểu các ngưỡng dùng
5. Xem "PERFORMANCE METRICS" để xem kết quả

**Để debug/lỗi:**
1. Xem "COMMON ISSUES & SOLUTIONS" trong mỗi file
2. Check "KEY FUNCTIONS SUMMARY" để hiểu API
3. Xem "USAGE EXAMPLE" để xem cách dùng

**Để training ML model (Module 3, 5):**
1. Xem "NEXT STEPS" trong file Module 3 và 5
2. Follow Training workflow trong documentation
3. Validate với target metrics

---

## 💡 MẸO DÙNG TÀI LIỆU

**Cho Nhà tuyển học:**
- Đọc "DATA FLOW" để hiểu architecture
- Đọc "PERFORMANCE METRICS" để đánh giá technical excellence
- Đọc "NIHSS MAPPING" để hiểu clinical validity

**Cho BGK:**
- Đọc "THRESHOLDS" để kiểm tra scientific validity
- Đọc "LOGIC CỐT LÕI" để hiểu algorithm
- Đọc "SCIENTIFIC VALIDATION" để check sources

**Cho Developer:**
- Đọc "KEY FUNCTIONS SUMMARY" để understand API
- Đọc "USAGE EXAMPLE" để implement
- Đọc "COMMON ISSUES" để debug

---

*Index Document Version: 1.0*
*Last Updated: 30/08/2026*
*PSCS v8.0 - Pre-Hospital Stroke Care System*

---

**🎯 TỔNG KẾT:**

Đã tạo xong 5 file logic giải thích code cho từng module:

1. ✅ `file_logic_giai_thich_code_Module1.md` - Face Asymmetry Detection
2. ✅ `file_logic_giai_thich_code_Module2.md` - Speech Dysarthria Detection
3. ✅ `file_logic_giai_thich_code_Module3.md` - Arm Weakness Detection
4. ✅ `file_logic_giai_thich_code_Module4.md` - Gait Abnormality Detection ⭐ FIXED
5. ✅ `file_logic_giai_thich_code_Module5.md` - Visual Field Defect Detection
6. ✅ `INDEX_FILE_LOGIC.md` - File index này

Tất cả file đều có:
- 🎯 Mục tiêu module
- 🔄 Data flow (quy trình xử lý)
- 📐 Logic chi tiết từng hàm
- 📊 Thresholds & ngưỡng
- 🔍 Key functions summary
- 📈 Performance metrics
- 🚨 Common issues & solutions
- 📝 Usage examples

**Sẵn sàng cho BGK và Nhà tuyển học! 🎓**
