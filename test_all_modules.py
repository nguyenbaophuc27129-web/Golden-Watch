# -*- coding: utf-8 -*-
"""
Quick test of all Core Detection modules after fixes
"""

import sys
import os
sys.path.insert(0, 'src')

import datetime

print("="*80)
print("TESTING ALL CORE DETECTION MODULES AFTER FIXES")
print("="*80)
print()
print(f"Test Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

results = {}

# Test 1: Face Module
print("[1/5] Testing Face Module...")
try:
    from detection.face_module_v7 import FaceAsymmetryDetector
    detector = FaceAsymmetryDetector(history_size=5)
    print("  OK Face module loaded successfully")
    results['face'] = 'OK'
except Exception as e:
    print(f"  X Face module error: {e}")
    results['face'] = 'FAIL'

# Test 2: Speech Module
print("[2/5] Testing Speech Module...")
try:
    from detection.speech_module_v2 import SpeechAnalysisModule
    speech_model = "models/speech_torgo_20260828_211130.pth"
    speech_scaler = "models/speech_torgo_20260828_211130_scaler.pkl"
    if os.path.exists(speech_model) and os.path.exists(speech_scaler):
        detector = SpeechAnalysisModule(
            vosk_model_path=None,
            ml_model_path=speech_model,
            scaler_path=speech_scaler
        )
        print("  OK Speech module loaded successfully")
        results['speech'] = 'OK'
    else:
        print("  ! Speech model files not found")
        results['speech'] = 'NO_MODEL'
except Exception as e:
    print(f"  X Speech module error: {e}")
    results['speech'] = 'FAIL'

# Test 3: Arm Module
print("[3/5] Testing Arm Module...")
try:
    from detection.arm_module import ArmWeaknessDetector
    yolo_model = "models/yolov8n-pose.pt"

    # Find latest ML model
    import glob
    ml_models = glob.glob("models/arm_weakness_*.pth")
    if ml_models:
        latest_ml_model = sorted(ml_models)[-1]
        ml_scaler = latest_ml_model.replace('.pth', '_scaler.pkl')
        print(f"  [INFO] Using ML model: {os.path.basename(latest_ml_model)}")
    else:
        latest_ml_model = None
        ml_scaler = None
        print(f"  [INFO] No ML model found, using rule-based")

    if os.path.exists(yolo_model):
        detector = ArmWeaknessDetector(
            pose_model_path=yolo_model,
            ml_model_path=latest_ml_model,
            scaler_path=ml_scaler
        )
        print("  OK Arm module loaded successfully (with ML model support!)")
        results['arm'] = 'OK'
    else:
        print("  ! YOLO model not found")
        results['arm'] = 'NO_MODEL'
except Exception as e:
    print(f"  ❌ Arm module error: {e}")
    results['arm'] = 'FAIL'

# Test 4: Gait Module
print("[4/5] Testing Gait Module...")
try:
    from detection.gait_module import GaitAbnormalityDetector
    gait_model = "models/gait_classifier_20260829_120925.pth"
    gait_scaler = "models/gait_classifier_20260829_120925_scaler.pkl"
    if os.path.exists(gait_model) and os.path.exists(gait_scaler):
        detector = GaitAbnormalityDetector(
            model_path=gait_model,
            scaler_path=gait_scaler
        )
        print("  OK Gait module loaded successfully")
        results['gait'] = 'OK'
    else:
        print("  ! Gait model files not found")
        results['gait'] = 'NO_MODEL'
except Exception as e:
    print(f"  ❌ Gait module error: {e}")
    results['gait'] = 'FAIL'

# Test 5: Visual Module
print("[5/5] Testing Visual Module...")
try:
    from detection.visual_module import VisualFieldDetector

    # Find latest ML model
    import glob
    ml_models = glob.glob("models/visual_field_*.pth")
    if ml_models:
        latest_ml_model = sorted(ml_models)[-1]
        ml_scaler = latest_ml_model.replace('.pth', '_scaler.pkl')
        print(f"  [INFO] Using ML model: {os.path.basename(latest_ml_model)}")
    else:
        latest_ml_model = None
        ml_scaler = None
        print(f"  [INFO] No ML model found, using rule-based")

    detector = VisualFieldDetector(
        ml_model_path=latest_ml_model,
        scaler_path=ml_scaler
    )
    print("  OK Visual module loaded successfully (with ML model support!)")
    results['visual'] = 'OK'
except Exception as e:
    print(f"  ❌ Visual module error: {e}")
    results['visual'] = 'FAIL'

print()
print("="*80)
print("MODULE LOAD TEST SUMMARY")
print("="*80)
print()

for module, status in results.items():
    symbol = "[OK]" if status == 'OK' else "[!]" if status == 'NO_MODEL' else "[X]"
    print(f"{symbol} {module.upper():12} : {status}")

print()
print("="*80)

# Save results
import json
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
results_file = f'test_fixes_{timestamp}.json'
with open(results_file, 'w') as f:
    json.dump({
        'timestamp': datetime.datetime.now().isoformat(),
        'results': results
    }, f, indent=2)

print(f"Results saved to: {results_file}")
print()
print("="*80)
