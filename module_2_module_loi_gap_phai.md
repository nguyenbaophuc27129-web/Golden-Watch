# MODULE 2 - SPEECH ANALYSIS - ERROR LOG

## File: module2_main_ml.py
**Version:** v8.0
**Last Updated:** 30/08/2026

---

## ❌ LỖI TÌM THẤY (CRITICAL ERRORS)

### Lỗi 1: Vosk Model Path Validation Missing
**Severity:** HIGH
**Status:** DOCUMENTED

**Issue:**
```python
# Line 28: Vosk model path không validated
VOSK_MODEL = os.path.join(parent_dir, "models/vosk-model-vn-0.4")
```

**Problem:**
- Không check xem model có tồn tại không
- Vosk model rất lớn (~500MB) - có thể chưa download
- Crashes khi model missing

**Fix Needed:**
```python
if not os.path.exists(VOSK_MODEL):
    print(f"[ERROR] Vosk model not found: {VOSK_MODEL}")
    print("Please download from: https://alphacephei.com/vosk/models")
    return
```

---

### Lỗi 2: ML Model Scaler Missing Validation
**Severity:** HIGH
**Status:** DOCUMENTED

**Issue:**
```python
# Line 30-31: Model files không validated
ML_MODEL = os.path.join(parent_dir, "models/speech_torgo_20260828_211130.pth")
SCALER = os.path.join(parent_dir, "models/speech_torgo_20260828_211130_scaler.pkl")
```

**Problem:**
- Không check xem files có tồn tại
- Scaler file critical cho ML predictions
- System crashes nếu files missing

**Fix Needed:**
```python
if not os.path.exists(ML_MODEL):
    print(f"[ERROR] ML model not found: {ML_MODEL}")
    return
if not os.path.exists(SCALER):
    print(f"[ERROR] Scaler not found: {SCALER}")
    return
```

---

## ⚠️ VẤN ĐỀ CẦN LƯU Ý (WARNINGS)

### Vấn đề 1: Hardcoded Thresholds - No Scientific Basis
**Severity:** HIGH
**Status:** CRITICAL ISSUE

**Issue:**
Thresholds trong speech_module_v2.py không có scientific justification:

```python
# These thresholds need validation!
JITTER_THRESHOLD = 3.0  # %
SHIMMER_THRESHOLD = 6.0  # %
WPM_MIN = 100
WPM_MAX = 180
PITCH_STD_THRESHOLD = 50  # Hz
```

**Problem:**
- Không có link đến research papers
- Không có clinical validation
- Values có thể outdated hoặc incorrect

**Sources Needed:**
1. Jitter threshold: Research paper?
2. Shimmer threshold: Research paper?
3. WPM range: Clinical standard?
4. Pitch threshold: Published research?

---

### Vấn đề 2: No Vietnamese STT Optimization
**Severity:** MEDIUM
**Status:** LIMITATION

**Issue:**
- Vosk model cho Vietnamese có thể không accurate
- Noise rejection không optimized
- Vietnamese phoneme recognition limited

**Current Performance:**
- Accuracy: 83.07% (on TORGO dataset - English)
- Vietnamese performance: Unknown

---

### Vấn đề 3: Recording Duration Fixed
**Severity:** LOW
**Status:** LIMITATION

**Issue:**
```python
# Recording time fixed at 5 seconds
RECORDING_DURATION = 5
```

**Problem:**
- 5 seconds có thể không đủ cho dysarthria detection
- Elderly patients có thể nói chậm hơn
- Should be configurable

---

## 🔧 CẦN CẢI THIỆN (IMPROVEMENTS NEEDED)

### 1. Add Noise Reduction
**Current:** Basic recording
**Needed:** Noise cancellation, echo removal

### 2. Vietnamese Language Optimization
**Current:** Generic Vosk model
**Needed:** Fine-tuned Vietnamese STT

### 3. Longer Recording Time
**Current:** 5 seconds fixed
**Needed:** Configurable 5-15 seconds

### 4. Real-time Visualization
**Current:** Terminal output
**Needed:** Waveform visualization, pitch graph

---

## 📊 TRAINING STATUS

### Model Training Completed:
- ✅ Dataset: TORGO v2 (17,633 samples)
- ✅ Accuracy: 83.07%
- ✅ Precision: 83.30%
- ✅ Recall: 85.00%
- ✅ F1-Score: 0.8415
- ✅ Model: `speech_torgo_20260828_211130.pth`

### Validation Needed:
- ❌ Vietnamese speech validation
- ❌ Cross-validation (5-fold)
- ❌ ROC/AUC analysis
- ❌ Clinical testing with patients

---

## 🎯 PRIORITY FIXES (Week 2)

### Priority 1: Add Model Validation
**Timeline:** 30 minutes
**Impact:** Critical - Prevent crashes

### Priority 2: Document Threshold Sources
**Timeline:** 2 hours
**Impact:** High - Scientific credibility

### Priority 3: Vietnamese Speech Testing
**Timeline:** 4 hours
**Impact:** Medium - Validate Vietnamese performance

### Priority 4: Waveform Visualization
**Timeline:** 3 hours
**Impact:** Medium - User experience

---

## 📝 THANG ĐO CHUẨN QUỐC TẾ (NEED SOURCES!)

### 1. Jitter (Biến thiên pitch)
**Current Threshold:** <3%
**Needed Source:** Research paper linking jitter to dysarthria

**Possible Sources:**
- [SEARCH NEEDED] UA-Speech dataset analysis
- [SEARCH NEEDED] Dysarthria acoustic analysis papers

### 2. Shimmer (Biến thiên amplitude)
**Current Threshold:** <6%
**Needed Source:** Research paper linking shimmer to dysarthria

**Possible Sources:**
- [SEARCH NEEDED] Voice quality analysis
- [SEARCH NEEDED] Pathological speech characteristics

### 3. WPM (Words Per Minute)
**Current Range:** 100-180
**Needed Source:** Clinical standard for speech rate

**Possible Sources:**
- [SEARCH NEEDED] Speech rate norms
- [SEARCH NEEDED] TORGO dataset documentation

### 4. Pitch Characteristics
**Current Threshold:** Std <50Hz
**Needed Source:** Research on pitch in dysarthria

**Possible Sources:**
- [SEARCH NEEDED] F0 characteristics in dysarthria
- [SEARCH NEEDED] Pitch variability studies

---

*Error Log Generated: 30/08/2026*
*CRITICAL: Need to find scientific sources for all thresholds!*
