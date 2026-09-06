# -*- coding: utf-8 -*-
"""
AUTO TEST MODULE 2 - No interaction required
Test speech module with simulated features
"""

import sys
import os

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

import torch
import numpy as np
from detection.speech_module_v2 import DysarthriaClassifier
import joblib

# Configuration
ML_MODEL = os.path.join(parent_dir, "models/speech_torgo_20260828_211130.pth")
SCALER = os.path.join(parent_dir, "models/speech_torgo_20260828_211130_scaler.pkl")

print("="*70)
print("MODULE 2 AUTO TEST - TORGO MODEL VERIFICATION")
print("="*70)
print()

# Load model
print("Loading TORGO model...")
model = DysarthriaClassifier()
model.load_state_dict(torch.load(ML_MODEL, weights_only=True, map_location='cpu'))
model.eval()

scaler = joblib.load(SCALER)

print(f"Model loaded: {ML_MODEL}")
print(f"Scaler loaded: {SCALER}")
print()

# Test with simulated features
print("Testing with simulated features...")
print()

test_cases = [
    {
        "name": "Normal Speech",
        "features": np.array([
            # MFCC mean (13) - normal variation
            0.1, 0.2, 0.15, 0.1, 0.05, 0.1, 0.15, 0.1, 0.05, 0.1, 0.15, 0.1, 0.05,
            # MFCC std (13) - low variation
            0.3, 0.25, 0.2, 0.3, 0.25, 0.2, 0.3, 0.25, 0.2, 0.3, 0.25, 0.2, 0.3,
            # MFCC delta (13) - stable
            0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05,
            # Pitch (4) - stable
            150, 15, 120, 180,
            # Energy (3) - normal
            0.001, 0.0002, 0.002,
            # ZCR (2) - normal
            0.1, 0.02
        ])
    },
    {
        "name": "Dysarthria Speech",
        "features": np.array([
            # MFCC mean (13) - high variation
            0.5, 0.8, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5,
            # MFCC std (13) - high variation
            0.8, 0.7, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.9, 0.8,
            # MFCC delta (13) - unstable
            0.3, 0.4, 0.35, 0.3, 0.4, 0.35, 0.3, 0.4, 0.35, 0.3, 0.4, 0.35, 0.3,
            # Pitch (4) - variable
            140, 50, 90, 190,
            # Energy (3) - irregular
            0.0008, 0.0004, 0.0015,
            # ZCR (2) - higher
            0.12, 0.04
        ])
    }
]

for test in test_cases:
    print(f"Test: {test['name']}")
    print("-" * 50)

    # Scale and predict
    features_scaled = scaler.transform(test['features'].reshape(1, -1))
    features_tensor = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)
        dysarthria_prob = probs[0][1].item() * 100
        normal_prob = probs[0][0].item() * 100

    # Determine status
    if dysarthria_prob < 30:
        status = "NORMAL"
        color = "OK"
    elif dysarthria_prob < 60:
        status = "WARNING"
        color = "WARN"
    else:
        status = "DANGER"
        color = "DANGER"

    print(f"  Normal Probability: {normal_prob:.1f}%")
    print(f"  Dysarthria Probability: {dysarthria_prob:.1f}%")
    print(f"  Status: [{color}] {status}")
    print()

# Model info
print("="*70)
print("MODEL INFORMATION")
print("="*70)
print()

import json
info_path = ML_MODEL.replace('.pth', '_info.json')
with open(info_path, 'r') as f:
    info = json.load(f)

print(f"Dataset: {info['dataset']}")
print(f"Accuracy: {info['accuracy']*100:.2f}%")
print(f"Precision: {info['precision']*100:.2f}%")
print(f"Recall: {info['recall']*100:.2f}%")
print(f"Specificity: {info['specificity']*100:.2f}%")
print()

print("="*70)
print("MODULE 2 AUTO TEST COMPLETE")
print("="*70)
print()
print("To test with REAL microphone:")
print("  py -3.11 module2/module2_main_ml.py")
print()
