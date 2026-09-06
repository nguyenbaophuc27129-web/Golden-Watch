# -*- coding: utf-8 -*-
"""
TEST MODULE 2 WITH REAL TORGO AUDIO FILES
Test model with actual samples from TORGO dataset
"""

import sys
import os
import glob

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

import torch
import numpy as np
import librosa
from detection.speech_module_v2 import DysarthriaClassifier
import joblib

# Configuration
ML_MODEL = os.path.join(parent_dir, "models/speech_torgo_20260828_211130.pth")
SCALER = os.path.join(parent_dir, "models/speech_torgo_20260828_211130_scaler.pkl")
TORGO_PATH = os.path.join(parent_dir, "data/datasets/speech/TORGO Dataset for Dysarthric Speech - Audio Files")

SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 13

print("="*70)
print("MODULE 2 TEST - REAL TORGO AUDIO SAMPLES")
print("="*70)
print()

# Load model
print("Loading TORGO model...")
model = DysarthriaClassifier()
model.load_state_dict(torch.load(ML_MODEL, weights_only=True, map_location='cpu'))
model.eval()

scaler = joblib.load(SCALER)
print("Model loaded!")
print()


def extract_features(audio_path):
    """Extract 48 features from audio"""
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = np.mean(np.diff(mfcc, axis=1), axis=1)

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            pitch_mean = [np.mean(pitch_values)]
            pitch_std = [np.std(pitch_values)]
            pitch_min = [np.min(pitch_values)]
            pitch_max = [np.max(pitch_values)]
        else:
            pitch_mean = [0]
            pitch_std = [0]
            pitch_min = [0]
            pitch_max = [0]

        # Energy
        frame_length = int(SAMPLE_RATE * 0.02)
        energy = []
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))

        if len(energy) > 0:
            energy_mean = [np.mean(energy)]
            energy_std = [np.std(energy)]
            energy_range = [np.max(energy) - np.min(energy)]
        else:
            energy_mean = [0]
            energy_std = [0]
            energy_range = [0]

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = [np.mean(zcr)]
        zcr_std = [np.std(zcr)]

        return np.concatenate([
            mfcc_mean, mfcc_std, mfcc_delta,
            pitch_mean, pitch_std, pitch_min, pitch_max,
            energy_mean, energy_std, energy_range,
            zcr_mean, zcr_std
        ])
    except Exception as e:
        print(f"    Error: {e}")
        return None


# Test with real TORGO files
test_configs = [
    {"name": "Normal (Female Control)", "path": "F_Con/wav_arrayMic_FC01S01", "expected": "NORMAL"},
    {"name": "Normal (Male Control)", "path": "M_Con/wav_arrayMic_MC01S01", "expected": "NORMAL"},
    {"name": "Dysarthria (Female)", "path": "F_Dys/wav_arrayMic_F01", "expected": "DANGER/WARNING"},
    {"name": "Dysarthria (Male)", "path": "M_Dys/wav_arrayMic_M01S01", "expected": "DANGER/WARNING"}
]

results_summary = []

for config in test_configs:
    test_path = os.path.join(TORGO_PATH, config['path'])

    if not os.path.exists(test_path):
        print(f"Skip: {config['name']} (path not found: {test_path})")
        continue

    wav_files = glob.glob(os.path.join(test_path, "*.wav"))[:5]  # Test 5 files each

    if len(wav_files) == 0:
        print(f"Skip: {config['name']} (no WAV files)")
        continue

    print(f"Test: {config['name']}")
    print(f"  Path: {config['path']}")
    print(f"  Files: {len(wav_files)} samples")
    print()

    predictions = []

    for wav_file in wav_files:
        features = extract_features(wav_file)
        if features is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            features_tensor = torch.FloatTensor(features_scaled)

            with torch.no_grad():
                outputs = model(features_tensor)
                probs = torch.softmax(outputs, dim=1)
                dysarthria_prob = probs[0][1].item() * 100

            if dysarthria_prob < 30:
                status = "NORMAL"
            elif dysarthria_prob < 60:
                status = "WARNING"
            else:
                status = "DANGER"

            predictions.append(dysarthria_prob)

    if len(predictions) > 0:
        avg_prob = np.mean(predictions)
        final_status = "NORMAL" if avg_prob < 30 else "WARNING" if avg_prob < 60 else "DANGER"

        print(f"  Average Dysarthria Prob: {avg_prob:.1f}%")
        print(f"  Status: {final_status}")
        print(f"  Expected: {config['expected']}")
        print()

        results_summary.append({
            "name": config['name'],
            "avg_prob": avg_prob,
            "status": final_status,
            "expected": config['expected']
        })

# Summary
print("="*70)
print("TEST SUMMARY")
print("="*70)
print()

for result in results_summary:
    match = "✓" if result['expected'] in result['status'] else "✗"
    print(f"{match} {result['name']}: {result['status']} ({result['avg_prob']:.1f}%)")

print()
print("="*70)
