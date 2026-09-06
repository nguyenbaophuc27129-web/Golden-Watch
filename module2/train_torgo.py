# -*- coding: utf-8 -*-
"""
TRAIN WITH TORGO DATASET - Module 2 (PSCS v8.0)
Train ML model cho dysarthria detection với TORGO dataset

TORGO Dataset: https://www.cs.toronto.edu/~complingweb/torgo.php
- 8 dysarthric speakers
- 7 control speakers
- ~10,000 audio files

Usage:
    py -3.11 module2/train_torgo.py --torgo_path /path/to/torgo

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
import argparse
import glob

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

# Configuration
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_MFCC = 13

# Training hyperparameters (IMPROVED REGULARIZATION)
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_DIMS = [256, 128, 64]  # Larger network
DROPOUT = 0.5  # Higher dropout
WEIGHT_DECAY = 0.001  # L2 regularization
EARLY_STOPPING_PATIENCE = 20
MIN_DELTA = 0.001


class TORGODataset(Dataset):
    """Dataset cho TORGO audio files"""

    def __init__(self, features, labels, speaker_ids=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.speaker_ids = speaker_ids if speaker_ids else [0] * len(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.speaker_ids[idx]


class DysarthriaClassifier(nn.Module):
    """MLP Classifier cho dysarthria detection"""

    def __init__(self, input_dim=48, hidden_dims=[128, 64, 32], num_classes=2):
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
    """Extract features từ file audio (48 features)"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

        # Truncate or pad to DURATION seconds
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        features_dict = {}

        # 1. MFCC features (13+13+13 = 39 features)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        features_dict['mfcc_mean'] = np.mean(mfcc, axis=1)
        features_dict['mfcc_std'] = np.std(mfcc, axis=1)
        features_dict['mfcc_delta'] = np.mean(np.diff(mfcc, axis=1), axis=1)

        # 2. Pitch features (4 features)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            features_dict['pitch_mean'] = [np.mean(pitch_values)]
            features_dict['pitch_std'] = [np.std(pitch_values)]
            features_dict['pitch_min'] = [np.min(pitch_values)]
            features_dict['pitch_max'] = [np.max(pitch_values)]
        else:
            features_dict['pitch_mean'] = [0]
            features_dict['pitch_std'] = [0]
            features_dict['pitch_min'] = [0]
            features_dict['pitch_max'] = [0]

        # 3. Energy features (3 features)
        frame_length = int(SAMPLE_RATE * 0.02)
        energy = []
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))

        if len(energy) > 0:
            features_dict['energy_mean'] = [np.mean(energy)]
            features_dict['energy_std'] = [np.std(energy)]
            features_dict['energy_range'] = [np.max(energy) - np.min(energy)]
        else:
            features_dict['energy_mean'] = [0]
            features_dict['energy_std'] = [0]
            features_dict['energy_range'] = [0]

        # 4. ZCR features (2 features)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features_dict['zcr_mean'] = [np.mean(zcr)]
        features_dict['zcr_std'] = [np.std(zcr)]

        # Concatenate all features (48 features)
        feature_vector = np.concatenate([
            features_dict['mfcc_mean'],
            features_dict['mfcc_std'],
            features_dict['mfcc_delta'],
            features_dict['pitch_mean'],
            features_dict['pitch_std'],
            features_dict['pitch_min'],
            features_dict['pitch_max'],
            features_dict['energy_mean'],
            features_dict['energy_std'],
            features_dict['energy_range'],
            features_dict['zcr_mean'],
            features_dict['zcr_std']
        ])

        return feature_vector

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return np.zeros(48)


def load_torgo_dataset(torgo_path):
    """
    Load TORGO dataset from directory

    Actual structure:
    torgo_path/
    ├── F_Con/ (Female Control - Normal)
    │   └── wav_arrayMic_FC01S01/
    │       └── *.wav
    ├── F_Dys/ (Female Dysarthria)
    │   └── wav_arrayMic_*/
    │       └── *.wav
    ├── M_Con/ (Male Control - Normal)
    │   └── wav_arrayMic_*/
    │       └── *.wav
    └── M_Dys/ (Male Dysarthria)
        └── wav_arrayMic_*/
            └── *.wav

    Returns:
        features: numpy array
        labels: numpy array (0=NonDysarthria, 1=Dysarthria)
        speaker_ids: list
    """
    print("="*70)
    print("LOADING TORGO DATASET")
    print("="*70)
    print()

    features = []
    labels = []
    speaker_ids = []
    file_paths = []

    # Control (Normal) speakers - F_Con and M_Con
    for control_type in ['F_Con', 'M_Con']:
        control_dir = os.path.join(torgo_path, control_type)
        if os.path.exists(control_dir):
            print(f"Loading {control_type} (Normal) samples from: {control_dir}")

            for speaker_dir in os.listdir(control_dir):
                speaker_path = os.path.join(control_dir, speaker_dir)
                if os.path.isdir(speaker_path):
                    wav_files = glob.glob(os.path.join(speaker_path, "*.wav"))
                    print(f"  {speaker_dir}: {len(wav_files)} files")

                    for wav_file in wav_files:
                        features.append(extract_features_from_audio(wav_file))
                        labels.append(0)  # Normal
                        speaker_ids.append(hash(speaker_dir) % 10000)
                        file_paths.append(wav_file)

    # Dysarthria speakers - F_Dys and M_Dys
    for dys_type in ['F_Dys', 'M_Dys']:
        dys_dir = os.path.join(torgo_path, dys_type)
        if os.path.exists(dys_dir):
            print(f"Loading {dys_type} (Dysarthria) samples from: {dys_dir}")

            for speaker_dir in os.listdir(dys_dir):
                speaker_path = os.path.join(dys_dir, speaker_dir)
                if os.path.isdir(speaker_path):
                    wav_files = glob.glob(os.path.join(speaker_path, "*.wav"))
                    print(f"  {speaker_dir}: {len(wav_files)} files")

                    for wav_file in wav_files:
                        features.append(extract_features_from_audio(wav_file))
                        labels.append(1)  # Dysarthria
                        speaker_ids.append(hash(speaker_dir) % 10000)
                        file_paths.append(wav_file)

    if len(features) == 0:
        print()
        print("[ERROR] No audio files found!")
        print()
        print("Expected TORGO structure:")
        print("  torgo_path/")
        print("  ├── F_Con/ (Female Control - Normal)")
        print("  │   └── wav_arrayMic_*/")
        print("  │       └── *.wav")
        print("  ├── F_Dys/ (Female Dysarthria)")
        print("  ├── M_Con/ (Male Control - Normal)")
        print("  └── M_Dys/ (Male Dysarthria)")
        print()
        return None, None, None

    features = np.array(features)
    labels = np.array(labels)

    print()
    print(f"Dataset loaded: {len(features)} samples, {features.shape[1]} features")
    dysarthria_count = np.sum(labels == 1)
    normal_count = np.sum(labels == 0)
    print(f"  Dysarthria: {dysarthria_count}")
    print(f"  Normal: {normal_count}")
    print()

    return features, labels, speaker_ids


def train_model(torgo_path=None, features=None, labels=None, speaker_ids=None):
    """Train model với TORGO dataset"""

    print("="*70)
    print("MODULE 2: TRAINING WITH TORGO DATASET")
    print("="*70)
    print()

    # Load dataset
    if torgo_path and os.path.exists(torgo_path):
        features, labels, speaker_ids = load_torgo_dataset(torgo_path)
        if features is None:
            return
    elif features is None:
        print("[ERROR] Please provide --torgo_path or pre-computed features")
        return

    # Split train/test (stratified by speaker to avoid speaker bias)
    # Get unique speakers
    unique_speakers = list(set(speaker_ids))

    # Split by speaker
    train_speakers = unique_speakers[:int(len(unique_speakers) * 0.8)]
    test_speakers = unique_speakers[int(len(unique_speakers) * 0.8):]

    train_indices = [i for i, s_id in enumerate(speaker_ids) if s_id in train_speakers]
    test_indices = [i for i, s_id in enumerate(speaker_ids) if s_id in test_speakers]

    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]

    print(f"Train set: {len(X_train)} samples ({len(train_speakers)} speakers)")
    print(f"Test set: {len(X_test)} samples ({len(test_speakers)} speakers)")
    print()

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print()

    # Create datasets
    train_dataset = TORGODataset(X_train_scaled, y_train, [speaker_ids[i] for i in train_indices])
    test_dataset = TORGODataset(X_test_scaled, y_test, [speaker_ids[i] for i in test_indices])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    input_dim = features.shape[1]
    model = DysarthriaClassifier(input_dim=input_dim, hidden_dims=HIDDEN_DIMS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Model initialized on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training setup (IMPROVED)
    # Class weights for imbalanced dataset
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Early stopping
    best_accuracy = 0.0
    best_model_state = None
    patience_counter = 0

    # Training loop
    print("Starting training...")
    print()

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features_batch, labels_batch, _ in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for features_batch, labels_batch, _ in test_loader:
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

        scheduler.step()

        # Early stopping check
        if test_accuracy > best_accuracy + MIN_DELTA:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
            print()

    # Final evaluation (use best model)
    if best_model_state is not None:
        print(f"Loading best model (accuracy: {best_accuracy:.2f}%)")
        model.load_state_dict(best_model_state)

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features_batch, labels_batch, _ in test_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(features_batch)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
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

    sensitivity = recall * 100
    specificity = (conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])) * 100 if (conf_matrix[0][0] + conf_matrix[0][1]) > 0 else 0

    print(f"Sensitivity (TPR): {sensitivity:.2f}%")
    print(f"Specificity (TNR): {specificity:.2f}%")
    print()

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = os.path.join(parent_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"speech_torgo_{timestamp}.pth")
    scaler_path = os.path.join(model_dir, f"speech_torgo_{timestamp}_scaler.pkl")
    info_path = os.path.join(model_dir, f"speech_torgo_{timestamp}_info.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    info = {
        "timestamp": timestamp,
        "dataset": "TORGO",
        "input_dim": input_dim,
        "hidden_dims": HIDDEN_DIMS,
        "num_classes": 2,
        "epochs": epoch + 1,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": conf_matrix.tolist(),
        "note": "Trained with improved regularization (higher dropout, L2, early stopping, class weights, gradient clipping)"
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
    parser = argparse.ArgumentParser(description='Train with TORGO dataset')
    parser.add_argument('--torgo_path', type=str, help='Path to TORGO dataset directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')

    args = parser.parse_args()

    try:
        if args.torgo_path:
            train_model(torgo_path=args.torgo_path)
        else:
            print("Usage:")
            print("  py -3.11 module2/train_torgo.py --torgo_path /path/to/torgo")
            print()
            print("Download TORGO from:")
            print("  https://www.kaggle.com/datasets/sauravmaheshkar/torgo-dataset")
            print("  https://www.cs.toronto.edu/~complingweb/torgo.php")
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user")
    except Exception as e:
        print()
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
