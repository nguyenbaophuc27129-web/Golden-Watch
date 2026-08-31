# MODULE 5 - VISUAL FIELD / EYE MOVEMENT DETECTION - ERROR LOG

## File: module5_main.py
**Version:** v8.0
**Last Updated:** 30/08/2026

---

## ❌ LỖI TÌM THẤY (CRITICAL ERRORS)

### Lỗi 1: MediaPipe Fallback Mode
**Severity:** HIGH
**Status:** DOCUMENTED

**Issue:**
```python
# Module 5 using MediaPipe fallback mode
# Eye landmarks detected but not fully utilized
```

**Problem:**
- MediaPipe API issues
- Not all 468 landmarks utilized
- Eye tracking limited accuracy

**Current Status:** Working in fallback mode

---

## ⚠️ VẤN ĐỀ CẦN LƯU Ý (WARNINGS)

### Vấn đề 1: Thresholds No Scientific Basis
**Severity:** HIGH
**Status:** CRITICAL ISSUE

**Issue:**
```python
# Thresholds arbitrary!
GAZE_ASYMMETRY_THRESHOLD = 15  # degrees - Why 15?
EYE_OPENNESS_THRESHOLD = 0.3    # Why 0.3?
PUPIL_ASYMMETRY_THRESHOLD = 0.2  # Why 0.2?
```

**Problem:**
- No research paper backing
- No clinical validation
- Values arbitrarily chosen

---

### Vấn đề 2: No ML Model
**Severity:** MEDIUM
**Status:** BY DESIGN

**Issue:**
- Module 5 rule-based only
- No ML model trained
- No accuracy metrics

**Current Approach:**
```python
# Rule-based thresholds
if gaze_asymmetry > 15:
    return "GAZE_ASYMMETRY_DETECTED"
```

**Problem:**
- No learning from data
- No accuracy optimization
- Limited to predefined rules

---

### Vấn đề 3: NIHSS Mapping Double-counted
**Severity:** MEDIUM
**Status:** DOCUMENTED

**Issue:**
```python
# Module 5 maps to BOTH NIHSS Item 4 AND Item 5
# This creates confusion!
```

**Problem:**
- NIHSS Item 4: Best Gaze
- NIHSS Item 5: Visual Fields
- Module 5 tries to do both

**Clarification Needed:**
- Which symptoms map to which NIHSS item?
- Hemianopia → Item 5 (Visual Fields)
- Gaze palsy → Item 4 (Best Gaze)
- Don't mix them!

---

## 🔧 CẦN CẢI THIỆN (IMPROVEMENTS NEEDED)

### 1. Separate Gaze vs Visual Field Detection
**Current:** Combined detection
**Needed:** Separate, clear detection

### 2. Add ML Model
**Current:** Rule-based only
**Needed:** Train ML model on eye tracking data

### 3. Improve Eye Tracking Accuracy
**Current:** Basic eye tracking
**Needed:** Advanced algorithms, calibration

### 4. Add Pupilometry
**Current:** Basic pupil asymmetry
**Needed:** Full pupilometry analysis

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

### Priority 1: Find Scientific Sources
**Timeline:** 3 hours
**Impact:** HIGH - Clinical credibility

### Priority 2: Separate NIHSS Mapping
**Timeline:** 1 hour
**Impact:** MEDIUM - Clear logic

### Priority 3: Improve Eye Tracking
**Timeline:** 4 hours
**Impact:** HIGH - Better accuracy

### Priority 4: Train ML Model (Optional)
**Timeline:** 8 hours
**Impact:** HIGH - Improved accuracy

---

## 📝 THANG ĐO CHUẨN QUỐC TẾ (NEED SOURCES!)

### 1. Gaze Asymmetry
**Current Threshold:** <15°
**Needed Source:** Research on gaze in stroke

**Possible Sources:**
- [SEARCH NEEDED] Gaze palsy in stroke
- [SEARCH NEEDED] Eye movement disorders research
- [SEARCH NEEDED] Neuro-ophthalmology papers

### 2. Eye Openness (EAR)
**Current Threshold:** >0.3
**Needed Source:** Eye aspect ratio research

**Possible Sources:**
- [SEARCH NEEDED] Eye aspect ratio validation
- [SEARCH NEEDED] Ptosis measurement papers
- [SEARCH NEEDED] Blink detection research

### 3. Pupil Asymmetry
**Current Threshold:** <0.2
**Needed Source:** Anisocoria research

**Possible Sources:**
- [SEARCH NEEDED] Pupillary asymmetry norms
- [SEARCH NEEDED] Anisocoria in stroke
- [SEARCH NEEDED] Neurological pupil testing

### 4. Visual Field Defects
**Current:** No quantitative threshold
**Needed Source:** Hemianopia detection

**Possible Sources:**
- [SEARCH NEEDED] Visual field testing
- [SEARCH NEEDED] Hemianopia quantification
- [SEARCH NEEDED] Perimetry research

---

## 🔍 NIHSS ITEMS 4 & 5 MAPPING

### NIHSS Item 4: Best Gaze
**Current Mapping:**
| Visual Score | NIHSS Score | Description |
|--------------|-------------|-------------|
| 0-29% | 0 | Normal |
| 30-59% | 1 | Partial gaze palsy |
| 60-100% | 2 | Forced deviation |

**Detection:**
- Gaze asymmetry
- Eye movement limitation
- Forced deviation

---

### NIHSS Item 5: Visual Fields
**Current Mapping:**
| Visual Score | NIHSS Score | Description |
|--------------|-------------|-------------|
| 0-29% | 0 | No visual loss |
| 30-49% | 1 | Partial hemianopia |
| 50-69% | 2 | Complete hemianopia |
| 70-100% | 3 | Bilateral hemianopia |

**Detection:**
- Visual field testing
- Hemianopia detection
- Quadrantanopia detection

---

## 🚨 CRITICAL ISSUES

### Issue 1: No Scientific Validation
**Status:** CRITICAL

**Problem:**
- Module 5 thresholds completely arbitrary
- No research backing
- No clinical validation

**Impact:**
- Not deployable for clinical use
- No credibility with medical community

---

### Issue 2: No ML Model
**Status:** HIGH PRIORITY

**Problem:**
- Rule-based system limited
- Cannot learn from data
- Cannot optimize accuracy

**Impact:**
- Lower accuracy than possible
- No adaptability

---

### Issue 3: NIHSS Mapping Confusion
**Status:** NEEDS CLARIFICATION

**Problem:**
- Module 5 maps to 2 NIHSS items
- Confusion about which symptoms map to which item
- Potential double-counting

**Impact:**
- Inaccurate NIHSS scoring
- Clinical confusion

---

## ✅ RECOMMENDED FIXES

### Immediate (Day 1):
1. Document current limitations
2. Separate NIHSS Item 4 vs 5 detection
3. Add warning about rule-based limitations

### Short-term (Day 2-3):
1. Find scientific sources for thresholds
2. Create comprehensive testing protocol
3. Add quantitative measurements

### Long-term (Week 2+):
1. Train ML model on eye tracking data
2. Clinical validation
3. Real patient testing

---

## 📊 COMPARISON: MODULE 5 vs LITERATURE

### Gaze Palsy Detection
**Literature:** Various methods exist
**Module 5:** Basic gaze tracking
**Gap:** Limited accuracy, no validation

### Visual Field Testing
**Literature:** Perimetry, confrontation testing
**Module 5:** Simplified 5-direction test
**Gap:** Not comprehensive enough

### Pupil Assessment
**Literature:** Pupillometry, anisocoria measurement
**Module 5:** Basic asymmetry detection
**Gap:** Limited quantification

---

*Error Log Generated: 30/08/2026*
*CRITICAL: Module 5 needs complete overhaul for clinical use!*
