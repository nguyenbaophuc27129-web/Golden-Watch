# -*- coding: utf-8 -*-
"""
TRAIN MODULE 1 - 100% DATASET (NO SPLIT)
===========================================
Train with ALL images and Test with ALL images (separate evaluation)

NO TRAIN/TEST SPLIT - 100% training + 100% testing
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
from sklearn.preprocessing import StandardScaler
import json
import time

# ========================================================================
# CONFIGURATION
# ========================================================================

class TrainConfig:
    """100% Training Configuration"""

    # Dataset
    STROKE_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/datasets/face/Annotated stroke and non stroke Dataset/Stroke"
    NORMAL_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/datasets/face/Annotated stroke and non stroke Dataset/NonStroke"

    # 100% TRAINING
    EPOCHS = 100  # Tăng lên 100-200 để accuracy cao hơn
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # MODEL
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT = 0.3

    # OUTPUT
    MODEL_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/models"
    MODEL_NAME = "stroke_classifier_100percent"

# ========================================================================
# MODEL
# ========================================================================

class StrokeClassifier(nn.Module):
    def __init__(self, input_dim=960, hidden_dims=None, dropout=0.3):
        super(StrokeClassifier, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ========================================================================
# DATA LOADING
# ========================================================================

def load_all_images(directory, label):
    """Load ALL images"""
    images = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"WARNING: Directory not found: {directory}")
        return images

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        for img_path in dir_path.glob(ext):
            images.append((str(img_path), label))

    return images

def extract_features_all(images, detector):
    """Extract features from ALL images"""
    features = []
    labels = []
    image_paths = []  # Keep track of paths for analysis

    total = len(images)
    global_timestamp = 0

    print(f"Processing {total} images...")

    for idx, (img_path, label) in enumerate(images):
        if (idx + 1) % 500 == 0:
            print(f"  Progress: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)")

        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            result = detector.process_frame(img, frame_timestamp_ms=global_timestamp)
            global_timestamp += 33
        except Exception as e:
            continue

        if result.get('status') in ['NO_FACE', 'NO_DETECTOR']:
            continue

        landmarks = result.get('raw_landmarks')
        if landmarks is None:
            continue

        # Features
        basic_features = landmarks.flatten()

        try:
            # Mouth AR
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            mouth_opening = np.linalg.norm(upper_lip - lower_lip)
            mouth_width = np.linalg.norm(left_corner - right_corner)
            mouth_ar = mouth_opening / (mouth_width + 1e-6)

            # Eye AR
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            left_eye_ar = np.linalg.norm(left_eye_top - left_eye_bottom)
            right_eye_ar = np.linalg.norm(right_eye_top - right_eye_bottom)
            eye_ar = (left_eye_ar + right_eye_ar) / 2

            # Rotation
            face_center = landmarks[5]
            deviation = face_center[0] - 0.5
            rotation = deviation * 100

            # Smile
            left_dist = np.linalg.norm(left_corner - face_center)
            right_dist = np.linalg.norm(right_corner - face_center)
            smile = (left_dist + right_dist) / 2

            additional_features = [mouth_ar, eye_ar, rotation, smile]
        except:
            additional_features = [0, 0, 0, 0]

        combined_features = np.concatenate([basic_features, additional_features])
        features.append(combined_features)
        labels.append(label)
        image_paths.append(img_path)

    return np.array(features), np.array(labels), image_paths

# ========================================================================
# TRAIN 100%
# ========================================================================

def train_100_percent():
    """Train with 100% dataset and test 100% dataset"""

    config = TrainConfig()

    print("="*70)
    print("TRAIN MODULE 1 - 100% DATASET (NO SPLIT)")
    print("="*70)
    print(f"\nEPOCHS: {config.EPOCHS}")
    print(f"HIDDEN_DIMS: {config.HIDDEN_DIMS}")

    print("\n" + "="*70)
    print("LOADING DATASET...")
    print("="*70)

    # Load ALL images
    print("\nLoading STROKE images...")
    stroke_images = load_all_images(config.STROKE_DIR, 0)
    print(f"  Found: {len(stroke_images)} images")

    print("\nLoading NORMAL images...")
    normal_images = load_all_images(config.NORMAL_DIR, 1)
    print(f"  Found: {len(normal_images)} images")

    total_images = len(stroke_images) + len(normal_images)
    print(f"\nTOTAL IMAGES IN DATASET: {total_images}")

    print("\n" + "="*70)
    print("EXTRACTING FEATURES FROM ALL IMAGES...")
    print("="*70)

    detector = FaceAsymmetryDetector(
        model_path="C:/Users/Admin/Documents/NCKHKT_26/fga_project/models/face_landmarker_v2/face_landmarker_v2_with_blendshapes.task"
    )

    print("\nExtracting Stroke features...")
    X_stroke, y_stroke, paths_stroke = extract_features_all(stroke_images, detector)
    print(f"  Valid: {len(X_stroke)}/{len(stroke_images)}")

    print("\nExtracting Normal features...")
    X_normal, y_normal, paths_normal = extract_features_all(normal_images, detector)
    print(f"  Valid: {len(X_normal)}/{len(normal_images)}")

    # Combine ALL
    X = np.vstack([X_stroke, X_normal])
    y = np.hstack([y_stroke, y_normal])
    all_paths = paths_stroke + paths_normal

    print(f"\nTOTAL VALID SAMPLES: {len(X)}")
    print(f"  Stroke: {sum(y==0)}, Normal: {sum(y==1)}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PyTorch Dataset (100% for training)
    class FaceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = FaceDataset(X_scaled, y)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    print("\n" + "="*70)
    print("TRAINING WITH 100% DATASET...")
    print("="*70)

    # Initialize model
    model = StrokeClassifier(
        input_dim=X.shape[1],
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\nModel: {X.shape[1]} -> {config.HIDDEN_DIMS} -> 2")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    num_epochs = config.EPOCHS
    best_acc = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

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

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) - Train: {train_acc:.2f}%')

        # Save best
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), f'{config.MODEL_DIR}/{config.MODEL_NAME}_best.pth')
            import joblib
            joblib.dump(scaler, f'{config.MODEL_DIR}/{config.MODEL_NAME}_scaler.pkl')

    total_time = time.time() - start_time

    print(f'\nTraining completed in {total_time/60:.1f} minutes')
    print(f'Best Training Accuracy: {best_acc:.2f}%')

    print("\n" + "="*70)
    print("TESTING WITH 100% DATASET...")
    print("="*70)

    # Load best model
    model.load_state_dict(torch.load(f'{config.MODEL_DIR}/{config.MODEL_NAME}_best.pth', weights_only=True))
    model.eval()

    # Test with ALL data
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs[:, 1].numpy())  # Stroke probability

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy on 100% dataset: {acc*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print("              Predicted")
    print("              Stroke  Normal")
    print(f"Actual Stroke   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual Normal  {cm[1][0]:4d}    {cm[1][1]:4d}")

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\nDetailed Metrics:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Sensitivity (Recall): {sensitivity*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")

    print("\nPer-class Report:")
    print(classification_report(all_labels, all_preds, target_names=['Stroke', 'Normal']))

    # Save results
    model_info = {
        'training_mode': '100_percent_no_split',
        'accuracy': float(acc * 100),
        'sensitivity': float(sensitivity * 100),
        'specificity': float(specificity * 100),
        'precision': float(precision * 100),
        'confusion_matrix': cm.tolist(),
        'total_images': int(total_images),
        'valid_samples': int(len(X)),
        'stroke_samples': int(sum(y==0)),
        'normal_samples': int(sum(y==1)),
        'config': {
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'hidden_dims': config.HIDDEN_DIMS
        },
        'training_time_minutes': float(total_time / 60)
    }

    with open(f'{config.MODEL_DIR}/{config.MODEL_NAME}_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\nModel saved to: {config.MODEL_DIR}/{config.MODEL_NAME}_best.pth")
    print(f"Info saved to: {config.MODEL_DIR}/{config.MODEL_NAME}_info.json")

    print("\n" + "="*70)
    print("TRAINING & TESTING COMPLETE!")
    print("="*70)

    print(f"\n✅ Trained with 100% dataset: {len(X)} samples")
    print(f"✅ Tested with 100% dataset: {len(X)} samples")
    print(f"✅ Final Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train_100_percent()
