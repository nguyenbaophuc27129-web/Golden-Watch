# MODULE 3 - ARM WEAKNESS DETECTION - ERROR LOG

## File: module3_main.py
**Version:** v8.0
**Last Updated:** 30/08/2026

---

## ❌ LỖI TÌM THẤY (CRITICAL ERRORS)

### Lỗi 1: YOLO Model Path Validation Missing
**Severity:** HIGH
**Status:** DOCUMENTED

**Issue:**
```python
# Line 28: YOLO model không validated
YOLO_MODEL = os.path.join(parent_dir, "models/yolov8n-pose.pt")
```

**Problem:**
- Không check model existence
- YOLOv8n-pose.pt rất lớn (~6MB)
- System crashes nếu model missing

**Fix Needed:**
```python
if not os.path.exists(YOLO_MODEL):
    print(f"[ERROR] YOLO model not found: {YOLO_MODEL}")
    print("Download from: https://github.com/ultralytics/ultralytics")
    return
```

---

### Lỗi 2: Keypoints API Compatibility (FIXED)
**Severity:** HIGH
**Status:** FIXED

**Previous Issue:**
```python
# OLD CODE (ultralytics <8.0):
keypoints = results[0].keypoints.xyxy[0].cpu().numpy()
```

**Fix Applied:**
```python
# NEW CODE (ultralytics >=8.0):
keypoints = results[0].keypoints.xy[0].cpu().numpy()
```

**Status:** ✅ Fixed in arm_module.py

---

## ⚠️ VẤN ĐỀ CẦN LƯU Ý (WARNINGS)

### Vấn đề 1: Rule-Based Detection - No ML Training
**Severity:** MEDIUM
**Status:** BY DESIGN

**Issue:**
- Module 3 sử dụng rule-based detection
- Không có ML model trained
- Accuracy unknown

**Current Approach:**
```python
# Rule-based thresholds
ARM_DROP_THRESHOLD = 100  # pixels
MOVEMENT_RANGE_THRESHOLD = 20  # degrees
ASYMMETRY_THRESHOLD = 25  # degrees
```

**Problem:**
- Thresholds không có scientific basis
- Không có accuracy metrics
- Không có clinical validation

---

### Vấn đề 2: Thresholds No Scientific Basis
**Severity:** HIGH
**Status:** CRITICAL ISSUE

**Issue:**
```python
# These thresholds arbitrary!
ARM_DROP_THRESHOLD = 100  # pixels - Why 100?
MOVEMENT_RANGE_THRESHOLD = 20  # degrees - Why 20?
ASYMMETRY_THRESHOLD = 25  # degrees - Why 25?
SPEED_RATIO_THRESHOLD = 0.5  # Why 0.5?
```

**Problem:**
- No research paper links
- No clinical validation
- No biomechanics basis

**Sources Needed:**
1. Arm drop threshold - Biomechanics paper?
2. Movement range - Clinical study?
3. Asymmetry - Stroke rehabilitation research?
4. Speed ratio - Motor control paper?

---

### Vấn đề 3: Pixel-based Measurements
**Severity:** MEDIUM
**Status:** LIMITATION

**Issue:**
```python
# Arm drop measured in pixels!
arm_drop = abs(start_y - end_y)
if arm_drop > 100:  # pixels???
```

**Problem:**
- Pixels not real-world units
- Changes with camera distance
- Not standardized

**Should Use:**
- Centimeters (with calibration)
- Degrees (angle-based)
- Normalized measurements

---

## 🔧 CẦN CẢI THIỆN (IMPROVEMENTS NEEDED)

### 1. Add ML Model
**Current:** Rule-based only
**Needed:** Train ML model on stroke arm weakness data

### 2. Normalize Measurements
**Current:** Pixel-based
**Needed:** Real-world units (cm, degrees)

### 3. Camera Calibration
**Current:** No calibration
**Needed:** Automatic calibration for real measurements

### 4. Biomechanics Validation
**Current:** Arbitrary thresholds
**Needed:** Validate against biomechanics research

---

## 📊 TRAINING STATUS

### ML Training: NOT COMPLETED
- ❌ No ML model trained
- ❌ No accuracy metrics
- ❌ No validation dataset
- ❌ No performance evaluation

**Current:** Rule-based system only

---

## 🎯 PRIORITY FIXES (Week 2)

### Priority 1: Add Model Validation
**Timeline:** 15 minutes
**Impact:** Critical - Prevent crashes

### Priority 2: Find Scientific Sources
**Timeline:** 3 hours
**Impact:** High - Clinical credibility

### Priority 3: Implement Angle-based Measurements
**Timeline:** 2 hours
**Impact:** Medium - More accurate

### Priority 4: Train ML Model (Optional)
**Timeline:** 8 hours
**Impact:** High - Improved accuracy

---

## 📝 THANG ĐO CHUẨN QUỐC TẾ (NEED SOURCES!)

### 1. Arm Drop (Drift)
**Current Threshold:** >100 pixels
**Needed Source:** Research on arm drift in stroke

**Possible Sources:**
- [SEARCH NEEDED] NIHSS motor arm research
- [SEARCH NEEDED] Post-stroke arm weakness studies

### 2. Movement Range
**Current Threshold:** >20 degrees
**Needed Source:** Biomechanics of arm movement

**Possible Sources:**
- [SEARCH NEEDED] Shoulder biomechanics
- [SEARCH NEEDED] Range of motion norms

### 3. Asymmetry
**Current Threshold:** >25 degrees
**Needed Source:** Arm asymmetry in stroke

**Possible Sources:**
- [SEARCH NEEDED] Hemiparesis research
- [SEARCH NEEDED] Motor asymmetry studies

### 4. Speed Ratio
**Current Threshold:** <0.5
**Needed Source:** Movement speed in stroke

**Possible Sources:**
- [SEARCH NEEDED] Motor velocity research
- [SEARCH NEEDED] Stroke kinematics papers

---

## 🔍 NIHSS ITEM 5 MAPPING

**Current Mapping:**
| Arm Score | NIHSS Score | Description |
|------------|-------------|-------------|
| 0-29% | 0 | No drift |
| 30-49% | 1 | Drift, holds position |
| 50-69% | 2 | Some effort against gravity |
| 70-100% | 3 | No movement against gravity |

**Problem:**
- Mapping TỰ tạo, không có clinical correlation
- No validation with real patients
- No research backing

**Needed:**
- Clinical validation study
- Correlation analysis với NIHSS assessment

---

*Error Log Generated: 30/08/2026*
*CRITICAL: Module 3 needs complete redesign for clinical use!*
