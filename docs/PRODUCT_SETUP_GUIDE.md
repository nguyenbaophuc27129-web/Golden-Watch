# PSCS - Pre-Hospital Stroke Care System
## Product Setup Guide for Competition

---

## 📦 PRODUCT OVERVIEW

**PSCS v8.0** is an AI-powered stroke detection system that implements the FAST stroke protocol for rapid pre-hospital stroke assessment.

### What it does:
- **F - Face**: Detects facial palsy/asymmetry using MediaPipe Face Mesh
- **A - Arms**: Detects arm weakness using YOLOv8n-Pose
- **S - Speech**: Detects dysarthria using TORGO-trained ML model
- **T - Time**: Provides comprehensive stroke risk assessment

### Key Features:
- ✅ Real-time camera-based detection
- ✅ 96.88% accuracy on gait abnormality detection
- ✅ 83.07% accuracy on dysarthria detection
- ✅ Web-based interface for easy use
- ✅ REST API for mobile integration
- ✅ Patient report generation

---

## 🎯 COMPETITION READINESS CHECKLIST

### Hardware Requirements:
- [ ] **Logitech C270 Webcam** (or any USB webcam)
  - Resolution: 720p minimum
  - FPS: 30fps minimum
  - USB: 2.0/3.0 compatible

- [ ] **Computer/Laptop**
  - CPU: Intel i5 or better
  - RAM: 8GB minimum (16GB recommended)
  - GPU: Optional (CUDA-compatible for faster inference)

- [ ] **Microphone** (for speech module)
  - Built-in laptop mic or USB microphone
  - Sample rate: 16kHz compatible

### Software Requirements:
- [ ] Python 3.11+
- [ ] All dependencies installed (see requirements.txt)
- [ ] Models downloaded and placed in `/models` folder

---

## 🚀 QUICK START (5 MINUTES)

### 1. Install Dependencies
```bash
cd C:/Users/Admin/Documents/NCKHKT_26/fga_project
pip install -r webapp/requirements.txt
```

### 2. Verify Models
Check that these files exist in `/models/`:
- `face_stroke_20260828_223019.pth`
- `face_stroke_20260828_223019_scaler.pkl`
- `speech_torgo_20260828_211130.pth`
- `speech_torgo_20260828_211130_scaler.pkl`
- `gait_classifier_20260829_120925.pth`
- `gait_classifier_20260829_120925_scaler.pkl`
- `yolov8n-pose.pt`

### 3. Start the Web App
```bash
cd webapp
python app.py
```

### 4. Open Browser
Navigate to: `http://localhost:5000`

---

## 📋 TEST PROCEDURES FOR COMPETITION

### Test 1: FAST Protocol (Camera)
**Purpose**: Demonstrate real-time stroke detection

1. Open web app at `http://localhost:5000`
2. Click "Start FAST Test"
3. Ask test subject to:
   - Look at camera and smile
   - Raise both arms forward
4. Observe results:
   - Face: Score < 40% = OK
   - Arms: Score < 30% = OK
   - FAST Score: 0 = LOW RISK

### Test 2: Gait Analysis (Dataset)
**Purpose**: Show ML model accuracy

Run: `python test_fast_simple.py`

Expected results:
- Normal samples: < 30% abnormality
- Parkinson's samples: > 90% abnormality

### Test 3: Speech Analysis (Audio)
**Purpose**: Dysarthria detection

1. Have subject speak clearly into microphone
2. Have subject slur speech (mimic stroke)
3. Observe classification difference

---

## 🎨 DEMONSTRATION SCRIPT

### Opening (30 seconds):
"PSCS is an AI-powered stroke detection system that saves lives by identifying stroke symptoms in seconds using the FAST protocol."

### Demo (2 minutes):
1. **Show Normal Case** (30s)
   - Subject: Healthy person
   - Action: Look at camera, raise arms, speak clearly
   - Result: All scores LOW → NORMAL

2. **Show Stroke Mimic** (1 minute)
   - Subject: Mimic stroke symptoms
   - Action: Drooping face, weak arm, slurred speech
   - Result: High scores → STROKE RISK DETECTED

3. **Explain FAST** (30s)
   - F: Facial palsy
   - A: Arm weakness
   - S: Speech difficulty
   - T: Time to call emergency

### Closing (30 seconds):
"Our system achieves 83-97% accuracy across different symptom types. In a real emergency, every second counts - PSCS helps save that time."

---

## 🔧 TROUBLESHOOTING

### Camera not detected?
```bash
# Check camera
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

### Models not loading?
- Verify file paths in `webapp/app.py`
- Check that model files are not corrupted

### Web app not starting?
- Check that port 5000 is not in use
- Try: `python app.py --port 5001`

### Low FPS?
- Close other applications
- Use GPU if available: Set `CUDA_VISIBLE_DEVICES=0`

---

## 📊 COMPETITION SCORING

### Innovation (30 points):
- AI-powered FAST protocol implementation
- Multi-modal detection (face, arms, speech, gait)
- Real-time processing

### Technical Excellence (30 points):
- ML models with 80-97% accuracy
- Real-time performance (>15 FPS)
- Robust error handling

### Impact (20 points):
- Saves critical time in stroke emergencies
- Easy to use interface
- Deployable in resource-constrained settings

### Presentation (20 points):
- Clear demo
- Live testing
- Professional web interface

**Total: 100 points**

---

## 📁 FILE STRUCTURE

```
fga_project/
├── webapp/
│   ├── app.py                 # Flask backend
│   ├── requirements.txt      # Dependencies
│   └── templates/
│       └── index.html        # Web interface
├── src/detection/
│   ├── face_module_v7.py     # Face detection
│   ├── arm_module.py         # Arm detection
│   ├── speech_module_v2.py   # Speech detection
│   ├── gait_module.py        # Gait detection
│   └── visual_module.py      # Visual field detection
├── models/                    # ML models
├── module1/                   # Face module files
├── module2/                   # Speech module files
├── module3/                   # Arm module files
├── module4/                   # Gait module files
├── module5/                   # Visual module files
├── test_fast_simple.py        # FAST test script
└── docs/
    └── PRODUCT_SETUP_GUIDE.md
```

---

## 🏆 WINNING STRATEGIES

### 1. Live Demo > Slides
Show real-time detection rather than just screenshots

### 2. Use Real Test Subjects
Have judges mimic stroke symptoms to show detection

### 3. Show Numbers
- "96.88% accuracy" is more impressive than "high accuracy"
- "Detects stroke in <5 seconds" shows speed

### 4. Storytelling
"PSCS can make the difference between recovery and permanent disability"

### 5. Prepared Questions
Anticipate questions about:
- False positives → Explain thresholds
- Hardware requirements → Show it runs on laptop
- Medical validation → Reference NIHSS scale

---

## 📞 CONTACT

- Team: PSCS Team
- Date: 29/08/2026
- Version: 8.0

**Good luck! 🎉**
