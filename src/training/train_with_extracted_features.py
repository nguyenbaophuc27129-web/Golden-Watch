"""
STEP 2: TRAIN ML CLASSIFIER FROM EXTRACTED FEATURES (NO MEDIAPIPE NEEDED)
=========================================================================

This trains the classifier using pre-extracted features.
NO MediaPipe model file needed!

Author: PSCS Team
Date: 2026-08-25
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuration
CONFIG = {
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.001,
    "early_stopping_patience": 5,
    "train_val_split": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print("=" * 60)
print("TRAIN FACE CLASSIFIER FROM EXTRACTED FEATURES")
print("=" * 60)
print(f"Device: {CONFIG['device'].upper()}")
print()

# Load features
print("Loading features...")
nonstroke_feat = np.load("C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/processed/nonstroke_features.npy")
stroke_feat = np.load("C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/processed/stroke_features.npy")

print(f"NonStroke features shape: {nonstroke_feat.shape}")
print(f"Stroke features shape: {stroke_feat.shape}")
print()

# Create labels
X = np.vstack([nonstroke_feat, stroke_feat])
y = np.array([0] * len(nonstroke_feat) + [1] * len(stroke_feat))

print(f"Total samples: {len(X)}")
print(f"NonStroke: {np.sum(y == 0)}, Stroke: {np.sum(y == 1)}")
print()

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=CONFIG["train_val_split"], random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")
print()


# Dataset class
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Model
class FaceClassifier(nn.Module):
    def __init__(self, input_size):
        super(FaceClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 2)  # Binary: 0=NonStroke, 1=Stroke
        )

    def forward(self, x):
        return self.network(x)


# Initialize
input_size = X_train.shape[1]
model = FaceClassifier(input_size).to(CONFIG["device"])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# Data loaders
train_dataset = FeatureDataset(X_train, y_train)
val_dataset = FeatureDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

print(f"Model input size: {input_size}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")
print()


# Training loop
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []
val_accuracies = []

print("Starting training...")
print()

for epoch in range(CONFIG["epochs"]):
    # Training
    model.train()
    train_loss = 0.0

    for features, labels in train_loader:
        features, labels = features.to(CONFIG["device"]), labels.to(CONFIG["device"])

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(CONFIG["device"]), labels.to(CONFIG["device"])

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    val_acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        # Save best model
        os.makedirs("C:/Users/Admin/Documents/NCKHKT_26/fga_project/models", exist_ok=True)
        torch.save(model.state_dict(),
                  "C:/Users/Admin/Documents/NCKHKT_26/fga_project/models/face_classifier_from_features.pth")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

print()
print("=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print()


# Final evaluation
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for features, labels in val_loader:
        features, labels = features.to(CONFIG["device"]), labels.to(CONFIG["device"])

        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

try:
    auc_roc = roc_auc_score(all_labels, all_probs)
except:
    auc_roc = 0.0

print("VALIDATION METRICS:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}  (Sensitivity)")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc_roc:.4f}")
print()

# PASS/FAIL indicators
print("QUALITY CHECK:")
print(f"  Accuracy >0.90:  {'[PASS]' if accuracy > 0.90 else '[FAIL]'}")
print(f"  Recall >0.90:    {'[PASS]' if recall > 0.90 else '[FAIL]'}  (Critical for stroke detection)")
print(f"  F1-Score >0.90:  {'[PASS]' if f1 > 0.90 else '[FAIL]'}")
print(f"  AUC-ROC >0.90:   {'[PASS]' if auc_roc > 0.90 else '[FAIL]'}")
print()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

print("CONFUSION MATRIX:")
print("                Predicted")
print("               NonStroke  Stroke")
print(f"Actual NonStroke  {cm[0][0]:6d}  {cm[0][1]:4d}")
print(f"       Stroke     {cm[1][0]:4d}  {cm[1][1]:6d}")
print()

# Save report
os.makedirs("C:/Users/Admin/Documents/NCKHKT_26/fga_project/results", exist_ok=True)

report = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "auc_roc": float(auc_roc),
    "confusion_matrix": cm.tolist(),
    "config": CONFIG
}

with open("C:/Users/Admin/Documents/NCKHKT_26/fga_project/results/training_report_from_features.json", "w") as f:
    json.dump(report, f, indent=2)

print("Report saved to: results/training_report_from_features.json")
print()
print("=" * 60)
print("NEXT STEPS:")
print("1. Test model with real webcam input")
print("2. Download face_landmarker.task for full functionality")
print("3. Or continue with feature-based approach")
print("=" * 60)
