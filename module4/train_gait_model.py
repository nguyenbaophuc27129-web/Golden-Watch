# -*- coding: utf-8 -*-
"""
TRAIN GAIT MODEL - Module 4 (PSCS v8.0)
Train ML model cho gait abnormality detection với Gait in Aging dataset

Usage:
    py -3.11 module4/train_gait_model.py --gait_path "path/to/gait/dataset"

Author: PSCS Team
Date: 29/08/2026
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
from datetime import datetime
import argparse
import glob
import pandas as pd

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.gait_module import GaitAbnormalityDetector

# Configuration
# Training hyperparameters
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_DIMS = [64, 32]
DROPOUT = 0.5
WEIGHT_DECAY = 0.001
EARLY_STOP_PATIENCE = 25


class GaitDataset(Dataset):
    """Dataset cho gait time series data"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GaitClassifier(nn.Module):
    """MLP Classifier cho gait abnormality detection"""

    def __init__(self, input_dim=8, hidden_dims=[64, 32], num_classes=2):
        super(GaitClassifier, self).__init__()
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


def load_gait_dataset(gait_path, window_size=100, stride=50):
    """
    Load Gait in Aging and Disease Dataset with sliding window

    Args:
        gait_path: Đường dẫn đến thư mục dataset
        window_size: Kích thước window để tạo samples
        stride: Bước nhảy giữa các windows

    Returns:
        features: numpy array
        labels: numpy array (0=Normal/Control, 1=Abnormal/PD)
    """
    print("="*70)
    print("LOADING GAIT IN AGING AND DISEASE DATASET")
    print("="*70)
    print(f"Window size: {window_size}, Stride: {stride}")
    print()

    detector = GaitAbnormalityDetector()

    features = []
    labels = []
    file_names = []

    # Load normal/control files (y1-y5 = young/normal, o1-o5 = old/normal)
    normal_patterns = []
    for i in range(1, 6):
        normal_patterns.append(f"y{i}-*si.txt")  # Young normal
        normal_patterns.append(f"o{i}-*si.txt")  # Old normal

    for pattern in normal_patterns:
        full_pattern = os.path.join(gait_path, pattern)
        files = glob.glob(full_pattern)

        for file in files:
            try:
                data = detector.load_gait_data(file)
                if data is not None and len(data) >= window_size:
                    # Use sliding window to create multiple samples
                    for start_idx in range(0, len(data) - window_size + 1, stride):
                        window_data = data[start_idx:start_idx + window_size]
                        feat = detector.extract_gait_features(window_data)
                        features.append(feat)
                        labels.append(0)  # Normal
                        file_names.append(f"{file}_win{start_idx}")
            except Exception as e:
                continue

    print(f"Normal (Control): {len([l for l in labels if l == 0])} samples")

    # Load Parkinson's Disease files (pd1-pd5 = abnormal)
    for i in range(1, 6):
        pattern = os.path.join(gait_path, f"pd{i}-*si.txt")
        files = glob.glob(pattern)

        for file in files:
            try:
                data = detector.load_gait_data(file)
                if data is not None and len(data) >= window_size:
                    # Use sliding window
                    for start_idx in range(0, len(data) - window_size + 1, stride):
                        window_data = data[start_idx:start_idx + window_size]
                        feat = detector.extract_gait_features(window_data)
                        features.append(feat)
                        labels.append(1)  # Abnormal (Parkinson's)
                        file_names.append(f"{file}_win{start_idx}")
            except Exception as e:
                continue

    print(f"Abnormal (Parkinson's): {len([l for l in labels if l == 1])} samples")
    print()

    features = np.array(features)
    labels = np.array(labels)

    print(f"Total samples: {len(features)}")
    print(f"Features per sample: {features.shape[1]}")
    print()

    return features, labels, file_names


def train_model(gait_path=None, features=None, labels=None):
    """Train model với gait dataset"""

    print("="*70)
    print("MODULE 4: TRAINING GAIT CLASSIFIER")
    print("="*70)
    print()

    # Load dataset
    if gait_path and os.path.exists(gait_path):
        features, labels, file_names = load_gait_dataset(gait_path)
        if features is None:
            return
    else:
        print("[ERROR] Please provide --gait_path")
        return

    # Check class balance
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution:")
    for cls, count in zip(unique, counts):
        class_name = "Normal" if cls == 0 else "Abnormal"
        print(f"  {class_name}: {count} samples")
    print()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print()

    # Create datasets
    train_dataset = GaitDataset(X_train_scaled, y_train)
    test_dataset = GaitDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)

    # Initialize model
    input_dim = features.shape[1]
    model = GaitClassifier(input_dim=input_dim, hidden_dims=HIDDEN_DIMS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Model initialized on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training setup with improved regularization
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Early stopping
    best_test_accuracy = 0.0
    patience_counter = 0

    # Training loop
    print("Starting training...")
    print()

    best_accuracy = 0.0

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

        scheduler.step(test_loss)

        # Early stopping
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
            print()

    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features_batch, labels_batch in test_loader:
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
    print(f"              Abnormal  Normal")
    print(f"Actual Abnormal    {conf_matrix[1][1]:4d}     {conf_matrix[1][0]:4d}")
    print(f"Actual Normal       {conf_matrix[0][1]:4d}     {conf_matrix[0][0]:4d}")
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

    model_path = os.path.join(model_dir, f"gait_classifier_{timestamp}.pth")
    scaler_path = os.path.join(model_dir, f"gait_classifier_{timestamp}_scaler.pkl")
    info_path = os.path.join(model_dir, f"gait_classifier_{timestamp}_info.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    info = {
        "timestamp": timestamp,
        "dataset": "Gait in Aging and Disease (v2 with sliding window)",
        "input_dim": input_dim,
        "hidden_dims": HIDDEN_DIMS,
        "num_classes": 2,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
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
    parser = argparse.ArgumentParser(description='Train Gait Model')
    parser.add_argument('--gait_path', type=str, required=True,
                       help='Path to Gait in Aging and Disease dataset')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')

    args = parser.parse_args()

    try:
        train_model(gait_path=args.gait_path)
        print("Training complete!")
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user")
    except Exception as e:
        print()
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
