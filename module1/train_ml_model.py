# -*- coding: utf-8 -*-
"""
Train ML Model with raw_landmarks from stroke dataset
Using PyTorch MLP classifier
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "C:/Users/Admin/Documents/NCKHKT_26/fga_project/src")

from detection.face_module_v7 import FaceAsymmetryDetector
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# Paths
DATASET_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/datasets/face/Annotated stroke and non stroke Dataset"
STROKE_DIR = f"{DATASET_DIR}/Stroke"
NONSTROKE_DIR = f"{DATASET_DIR}/NonStroke"
MODEL_PATH = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/models/face_landmarker_v2/face_landmarker_v2_with_blendshapes.task"
OUTPUT_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/models"

print("=== TRAIN ML MODEL WITH RAW LANDMARKS ===")

# Load images
def load_images(directory, label, limit=None):
    images = []
    dir_path = Path(directory)
    if not dir_path.exists():
        return images

    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        for img_path in dir_path.glob(ext):
            if limit and count >= limit:
                break
            images.append((str(img_path), label))
            count += 1
        if limit and count >= limit:
            break
    return images

print("Loading dataset (limited to 500 each for speed)...")
stroke_images = load_images(STROKE_DIR, 1, limit=500)
nonstroke_images = load_images(NONSTROKE_DIR, 0, limit=500)

print(f"Stroke: {len(stroke_images)}")
print(f"NonStroke: {len(nonstroke_images)}")

# Extract features
detector = FaceAsymmetryDetector(model_path=MODEL_PATH)

# Global timestamp for ALL frames (MediaPipe Tasks API requirement)
global_timestamp = 0

def extract_features(images, use_global_timestamp=True):
    features = []
    labels = []

    global global_timestamp
    local_timestamp = 0

    for img_path, label in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = detector.process_frame(img, frame_timestamp_ms=global_timestamp if use_global_timestamp else local_timestamp)
        global_timestamp += 33
        local_timestamp += 33

        # Skip if no face detected
        if result.get('status') in ['NO_FACE', 'NO_DETECTOR']:
            continue

        # Get raw landmarks (478, 2)
        landmarks = result.get('raw_landmarks')
        if landmarks is None:
            continue

        # Flatten to 956 features (478 x 2)
        features.append(landmarks.flatten())
        labels.append(label)

    return np.array(features), np.array(labels)

print("Extracting features from Stroke images...")
X_stroke, y_stroke = extract_features(stroke_images, use_global_timestamp=True)
print(f"Stroke: {len(X_stroke)} valid samples")

print("Extracting features from NonStroke images...")
X_nonstroke, y_nonstroke = extract_features(nonstroke_images, use_global_timestamp=True)
print(f"NonStroke: {len(X_nonstroke)} valid samples")

# Combine
X = np.vstack([X_stroke, X_nonstroke])
y = np.hstack([y_stroke, y_nonstroke])

print(f"\nTotal samples: {len(X)}")
print(f"Stroke: {sum(y==1)}, NonStroke: {sum(y==0)}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_torch = torch.FloatTensor(X_train_scaled)
y_train_torch = torch.LongTensor(y_train)
X_test_torch = torch.FloatTensor(X_test_scaled)
y_test_torch = torch.LongTensor(y_test)

# Create Dataset
class FaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FaceDataset(X_train_torch, y_train_torch)
test_dataset = FaceDataset(X_test_torch, y_test_torch)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define MLP Model
class StrokeClassifier(nn.Module):
    def __init__(self, input_dim=956, hidden_dims=[512, 256, 128], output_dim=2):
        super(StrokeClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Initialize model
model = StrokeClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print("\n=== TRAINING ===")
print(f"Model: MLP with layers [956 -> 512 -> 256 -> 128 -> 2]")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
num_epochs = 50
best_acc = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()

    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    val_acc = 100 * val_correct / val_total

    scheduler.step(val_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        # Save model weights only
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/stroke_classifier_best.pth')
        # Save scaler separately
        import joblib
        joblib.dump(scaler, f'{OUTPUT_DIR}/stroke_classifier_scaler.pkl')

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(test_loader):.4f}, Val Acc: {val_acc:.2f}%')

print(f'\nBest Validation Accuracy: {best_acc:.2f}%')

# Load best model for final evaluation
model.load_state_dict(torch.load(f'{OUTPUT_DIR}/stroke_classifier_best.pth', weights_only=True))
model.eval()

# Final evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n=== FINAL RESULTS ===")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall (Sensitivity): {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"\nConfusion Matrix:")
print("              Predicted")
print("              Stroke  Normal")
print(f"Actual Stroke   {cm[1][1]:4d}    {cm[1][0]:4d}")
print(f"Actual Normal  {cm[0][1]:4d}    {cm[0][0]:4d}")

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"\nSpecificity (TNR): {specificity*100:.2f}%")

# Save model info
model_info = {
    'best_accuracy': best_acc,
    'final_accuracy': acc * 100,
    'precision': precision * 100,
    'recall': recall * 100,
    'specificity': specificity * 100,
    'f1_score': f1 * 100,
    'confusion_matrix': cm.tolist(),
    'input_dim': 956,
    'hidden_dims': [512, 256, 128]
}

with open(f'{OUTPUT_DIR}/stroke_classifier_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\nModel saved to: {OUTPUT_DIR}/stroke_classifier_best.pth")
print(f"Model info saved to: {OUTPUT_DIR}/stroke_classifier_info.json")
print("\nTraining: DONE!")
