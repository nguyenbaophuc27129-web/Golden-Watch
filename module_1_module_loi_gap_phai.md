# MODULE 1 - FACE ASYMMETRY DETECTION - ERROR LOG

## File: module1_main.py
**Version:** v8.0
**Last Updated:** 30/08/2026

---

## ❌ LỖI TÌM THẤY (CRITICAL ERRORS)

### Lỗi 1: Line 29 - Typo trong import statement
**Severity:** HIGH
**Status:** FIXED

**Error:**
```python
8from detection.face_module_v7 import FaceAsymmetryDetector
```

**Explanation:** Chữ "8" ở đầu dòng là typo, có thể do copy-paste error.

**Fix:**
```python
from detection.face_module_v7 import FaceAsymmetryDetector
```

**Impact:** Module không thể import được, toàn bộ system không chạy.

---

### Lỗi 2: MediaPipe Tasks API Incompatibility
**Severity:** HIGH
**Status:** FALLBACK MODE

**Issue:**
```python
# Line 54: Tasks API model path
def MEDIAPIPE_MODEL(self):
    return os.path.join(self.parent_dir, "models/face_landmarker_v2/face_landmarker_v2_with_blendshapes.task")
```

**Problem:**
- MediaPipe 1.0 Tasks API requires proper file handling
- Model file có thể không tồn tại hoặc corrupted
- System falls back to Solutions API (468 landmarks instead of 478)

**Fix Applied:**
- Fallback logic implemented in face_module_v7.py
- Solutions API used when Tasks API fails

**Current Status:** Working in fallback mode

---

## ⚠️ VẤN ĐỀ CẦN LƯU Ý (WARNINGS)

### Vấn đề 1: Hardcoded Thresholds
**Severity:** MEDIUM
**Status:** DOCUMENTED

**Issue:**
```python
# Line 65-66: Thresholds hardcoded
NORMAL_THRESHOLD = 30
WARNING_THRESHOLD = 60
```

**Problem:**
- Thresholds 30%, 60% không có scientific justification
- Không có clinical validation
- Không có link đến research papers

**Recommendation:**
- Document source của thresholds
- Hoặc implement ROC analysis để optimize

---

### Vấn đề 2: No Error Handling cho Model Loading
**Severity:** MEDIUM
**Status:** NEEDS FIX

**Issue:**
```python
# Line 88-100: Model definition không có error handling
class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim=960, hidden_dims=[512, 256, 128], num_classes=2):
```

**Problem:**
- Không check xem model file có tồn tại không
- Không có fallback khi model loading fails
- Crashes khi model files missing

**Recommendation:**
```python
try:
    model = torch.load(ML_MODEL)
except FileNotFoundError:
    print(f"[ERROR] Model file not found: {ML_MODEL}")
    return None
```

---

## 🔧 CẦN CẢI THIỆN (IMPROVEMENTS NEEDED)

### 1. GUI Interface
**Status:** MISSING

**Current:** Terminal-based output
**Needed:** Graphical user interface với:
- Real-time visualization
- Clear pass/fail indicators
- User-friendly controls

### 2. Export Functionality
**Status:** IMPLEMENTED BUT NOT TESTED

**Current:** Code có export nhưng chưa verified
**Needed:** Test export to CSV/JSON

### 3. Batch Testing
**Status:** MISSING

**Current:** Only single-frame testing
**Needed:** Batch testing với multiple images/videos

---

## 📊 TRAINING STATUS

### Model Training Completed:
- ✅ Dataset: 2783 samples (1439 Stroke + 1344 Normal)
- ✅ Accuracy: 93.75%
- ✅ Model file: `stroke_classifier_100percent_best.pth`
- ✅ Scaler: `stroke_classifier_100percent_scaler.pkl`

### Validation Needed:
- ❌ Cross-validation (5-fold)
- ❌ ROC/AUC analysis
- ❌ Confidence intervals
- ❌ Statistical significance testing

---

## 🎯 PRIORITY FIXES (Week 2)

### Priority 1: Fix Import Typo
**Timeline:** 5 minutes
**Impact:** Critical - Module không chạy

### Priority 2: Add Error Handling
**Timeline:** 30 minutes
**Impact:** High - Prevent crashes

### Priority 3: Implement ROC Analysis
**Timeline:** 2 hours
**Impact:** Medium - Optimize thresholds

### Priority 4: Create GUI
**Timeline:** 4 hours
**Impact:** High - User experience

---

## 📝 NOTES

- Module using MediaPipe Face Mesh với 468/478 landmarks
- ML classifier trained on stroke dataset
- False positive detection implemented (YAWN, SMILE, HEAD_TURN, DROWSY)
- Export functionality exists but needs testing

---

*Error Log Generated: 30/08/2026*
*Next Review: After fixes applied*
