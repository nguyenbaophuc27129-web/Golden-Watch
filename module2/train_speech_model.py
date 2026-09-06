# -*- coding: utf-8 -*-
"""
TRAIN SPEECH MODEL - Module 2 (PSCS v8.0)
Train ML model cho dysarthria detection

Usage:
    py -3.11 module2/train_speech_model.py

Author: PSCS Team
Date: 28/08/2026
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
from tqdm import tqdm
import librosa
from datetime import datetime

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.speech_module import SpeechAnalysisModule

# Configuration
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_MFCC = 13

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3

# Synthetic dataset size
NUM_SAMPLES_PER_CLASS = 1500  # 1500 stroke + 1500 normal = 3000 total


class SpeechDataset(Dataset):
    """Dataset cho speech features"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DysarthriaClassifier(nn.Module):
    """MLP Classifier cho dysarthria detection"""

    def __init__(self, input_dim=30, hidden_dims=[128, 64, 32], num_classes=2):
        super(DysarthriaClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(DROPOUT)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def extract_features_from_audio(audio_path):
    """Extract features từ file audio"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

        # Truncate or pad to DURATION seconds
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        features = {}

        # 1. MFCC features (13 coefficients + mean/std)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        features['mfcc_delta'] = np.mean(np.diff(mfcc, axis=1), axis=1)

        # 2. Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            features['pitch_mean'] = [np.mean(pitch_values)]
            features['pitch_std'] = [np.std(pitch_values)]
            features['pitch_min'] = [np.min(pitch_values)]
            features['pitch_max'] = [np.max(pitch_values)]
        else:
            features['pitch_mean'] = [0]
            features['pitch_std'] = [0]
            features['pitch_min'] = [0]
            features['pitch_max'] = [0]

        # 3. Energy features
        frame_length = int(SAMPLE_RATE * 0.02)
        energy = []
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))

        if len(energy) > 0:
            features['energy_mean'] = [np.mean(energy)]
            features['energy_std'] = [np.std(energy)]
            features['energy_range'] = [np.max(energy) - np.min(energy)]
        else:
            features['energy_mean'] = [0]
            features['energy_std'] = [0]
            features['energy_range'] = [0]

        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = [np.mean(zcr)]
        features['zcr_std'] = [np.std(zcr)]

        # Concatenate all features
        feature_vector = np.concatenate([
            features['mfcc_mean'],      # 13
            features['mfcc_std'],       # 13
            features['mfcc_delta'],     # 13
            features['pitch_mean'],     # 1
            features['pitch_std'],      # 1
            features['pitch_min'],      # 1
            features['pitch_max'],      # 1
            features['energy_mean'],    # 1
            features['energy_std'],     # 1
            features['energy_range'],   # 1
            features['zcr_mean'],       # 1
            features['zcr_std']         # 1
        ])  # Total: 13+13+13+1+1+1+1+1+1+1+1+1 = 48 features

        return feature_vector

    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(48)


def generate_synthetic_dataset(num_samples=3000):
    """
    Tạo dataset synthetic cho training

    Returns:
        features: numpy array (n_samples, n_features)
        labels: numpy array (n_samples,) - 0=Normal, 1=Dysarthria
    """
    print("="*70)
    print("GENERATING SYNTHETIC DATASET")
    print("="*70)
    print()

    np.random.seed(42)

    n_features = 48
    features = np.zeros((num_samples, n_features))
    labels = np.zeros(num_samples)

    # Normal speech samples (50%)
    num_normal = num_samples // 2

    print(f"Generating {num_normal} NORMAL speech samples...")

    for i in range(num_normal):
        # Normal speech characteristics
        mfcc_mean = np.random.randn(13) * 0.5  # Small variation
        mfcc_std = np.abs(np.random.randn(13)) * 0.3 + 0.1
        mfcc_delta = np.random.randn(13) * 0.1

        pitch_mean = np.random.normal(150, 20)
        pitch_std = np.random.normal(15, 5)
        pitch_min = pitch_mean - np.random.normal(30, 10)
        pitch_max = pitch_mean + np.random.normal(30, 10)

        energy_mean = np.random.normal(0.001, 0.0003)
        energy_std = np.random.normal(0.0002, 0.00005)
        energy_range = np.random.normal(0.002, 0.0005)

        zcr_mean = np.random.normal(0.1, 0.02)
        zcr_std = np.random.normal(0.02, 0.005)

        features[i] = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_delta,
            [pitch_mean, pitch_std, pitch_min, pitch_max],
            [energy_mean, energy_std, energy_range],
            [zcr_mean, zcr_std]
        ])
        labels[i] = 0  # Normal

    # Dysarthria (stroke) samples (50%)
    num_stroke = num_samples - num_normal

    print(f"Generating {num_stroke} DYSARTHRIA speech samples...")

    for i in range(num_normal, num_samples):
        # Dysarthria characteristics
        # - Higher jitter/shimmer (simulated by MFCC variation)
        # - Slower speech rate (simulated by energy patterns)
        # - Higher pitch variability
        # - Irregular rhythm (simulated by ZCR)

        severity = np.random.uniform(0.5, 1.0)  # Severity factor

        mfcc_mean = np.random.randn(13) * (1.5 + severity)  # Higher variation
        mfcc_std = np.abs(np.random.randn(13)) * (0.5 + severity * 0.5) + 0.2
        mfcc_delta = np.random.randn(13) * (0.3 + severity * 0.5)

        pitch_mean = np.random.normal(140, 30)  # More variable
        pitch_std = np.random.normal(30, 15) * (1 + severity)  # Higher variability
        pitch_min = pitch_mean - np.random.normal(40, 15)
        pitch_max = pitch_mean + np.random.normal(40, 15)

        energy_mean = np.random.normal(0.0008, 0.0004)  # Lower energy
        energy_std = np.random.normal(0.0003, 0.0001) * (1 + severity)
        energy_range = np.random.normal(0.0015, 0.0006)

        zcr_mean = np.random.normal(0.12, 0.03)  # Higher ZCR
        zcr_std = np.random.normal(0.03, 0.01) * (1 + severity)

        features[i] = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_delta,
            [pitch_mean, pitch_std, pitch_min, pitch_max],
            [energy_mean, energy_std, energy_range],
            [zcr_mean, zcr_std]
        ])
        labels[i] = 1  # Dysarthria

    print()
    print(f"Dataset generated: {num_samples} samples, {n_features} features")
    print(f"  Normal: {num_normal}, Dysarthria: {num_stroke}")
    print()

    return features, labels


def calculate_jitter_shimmer_from_features(features):
    """Tính jitter/shimmer approximated từ MFCC features"""
    # Jitter ~ variability in pitch-related MFCC
    # Shimmer ~ variability in energy-related features

    # Use first few MFCC coeffs as proxy for pitch variation
    jitter_proxy = np.std(features[:3]) * 10  # Scale to percentage

    # Use energy variation as shimmer proxy
    shimmer_proxy = np.std(features[26:29]) * 100  # Scale to percentage

    return jitter_proxy, shimmer_proxy


def train_model():
    """Train model chính"""

    print("="*70)
    print("MODULE 2: TRAINING SPEECH CLASSIFIER")
    print("="*70)
    print()

    # 1. Generate dataset
    features, labels = generate_synthetic_dataset(NUM_SAMPLES_PER_CLASS * 2)

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()

    # 3. Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print()

    # 4. Create datasets
    train_dataset = SpeechDataset(X_train_scaled, y_train)
    test_dataset = SpeechDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 5. Initialize model
    input_dim = features.shape[1]
    model = DysarthriaClassifier(input_dim=input_dim, hidden_dims=HIDDEN_DIMS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Model initialized on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # 6. Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # 7. Training loop
    print("Starting training...")
    print()

    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for features_batch, labels_batch in test_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels_batch.size(0)
                test_correct += (predicted == labels_batch).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        scheduler.step(test_loss)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
            print()

    # 8. Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features_batch, labels_batch in test_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(features_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print("="*70)
    print("TRAINING RESULTS")
    print("="*70)
    print()
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Dysarthria  Normal")
    print(f"Actual Dysarthria   {conf_matrix[1][1]:4d}     {conf_matrix[1][0]:4d}")
    print(f"Actual Normal        {conf_matrix[0][1]:4d}     {conf_matrix[0][0]:4d}")
    print()

    # Sensitivity (TPR) = Recall for positive class
    sensitivity = recall * 100

    # Specificity (TNR) = TN / (TN + FP)
    specificity = (conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])) * 100

    print(f"Sensitivity (TPR): {sensitivity:.2f}%")
    print(f"Specificity (TNR): {specificity:.2f}%")
    print()

    # 9. Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = os.path.join(parent_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"speech_classifier_{timestamp}.pth")
    scaler_path = os.path.join(model_dir, f"speech_classifier_{timestamp}_scaler.pkl")
    info_path = os.path.join(model_dir, f"speech_classifier_{timestamp}_info.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    info = {
        "timestamp": timestamp,
        "input_dim": input_dim,
        "hidden_dims": HIDDEN_DIMS,
        "num_classes": 2,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "num_samples": NUM_SAMPLES_PER_CLASS * 2,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": conf_matrix.tolist()
    }

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print("="*70)
    print("MODEL SAVED")
    print("="*70)
    print()
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Info: {info_path}")
    print()


if __name__ == "__main__":
    try:
        train_model()
        print("Training complete!")
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user")
    except Exception as e:
        print()
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
