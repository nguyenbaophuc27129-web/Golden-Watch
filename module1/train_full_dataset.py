# -*- coding: utf-8 -*-
"""
TRAIN MODULE 1 - FULL DATASET (7498 images)
============================================
Train with ALL images from "Annotated stroke and non stroke Dataset"

CONFIGURABLE PARAMETERS:
- TRAIN_SAMPLES: How many images to use (None = ALL)
- EPOCHS: Training iterations
- BATCH_SIZE: Training batch size
- LEARNING_RATE: Adam learning rate
- HIDDEN_DIMS: Neural network hidden layers
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
import time

# ========================================================================
# CONFIGURATION - USER CAN MODIFY THESE
# ========================================================================

class TrainConfig:
    """Training Configuration - MODIFY THESE FOR YOUR NEEDS"""

    # Dataset
    STROKE_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/datasets/face/Annotated stroke and non stroke Dataset/Stroke"
    NORMAL_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/datasets/face/Annotated stroke and non stroke Dataset/NonStroke"

    # TRAINING PARAMETERS - MODIFY TO IMPROVE ACCURACY
    TRAIN_SAMPLES = None  # None = ALL images, or set number (e.g., 1000)
    EPOCHS = 50  # Increase for better accuracy (try 100, 200)
    BATCH_SIZE = 32  # Increase if GPU available (64, 128)
    LEARNING_RATE = 0.001  # Decrease for better convergence (0.0001)

    # MODEL ARCHITECTURE - MODIFY TO IMPROVE PERFORMANCE
    HIDDEN_DIMS = [512, 256, 128]  # Try [1024, 512, 256] for more capacity
    DROPOUT = 0.3  # Increase if overfitting (0.5)

    # DATA SPLIT
    TEST_SIZE = 0.2  # 20% for testing
    RANDOM_STATE = 42

    # OUTPUT
    MODEL_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/models"
    MODEL_NAME = "stroke_classifier_full"

# ========================================================================
# MODEL DEFINITION
# ========================================================================

class StrokeClassifier(nn.Module):
    """PyTorch Stroke Classifier - Configurable Architecture"""

    def __init__(self, input_dim=960, hidden_dims=None, dropout=0.3, num_classes=2):
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

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ========================================================================
# DATA LOADING
# ========================================================================

def load_all_images(directory, label):
    """Load ALL images from directory"""
    images = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"WARNING: Directory not found: {directory}")
        return images

    print(f"Loading from {directory}...")

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        for img_path in dir_path.glob(ext):
            images.append((str(img_path), label))

    return images

def extract_features(images, detector):
    """Extract features from images with progress bar"""
    features = []
    labels = []

    total = len(images)
    global_timestamp = 0

    for idx, (img_path, label) in enumerate(images):
        if (idx + 1) % 100 == 0:
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

        # Basic features
        basic_features = landmarks.flatten()

        # Additional features
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

    return np.array(features), np.array(labels)

# ========================================================================
# TRAINING
# ========================================================================

def train_full_dataset():
    """Train with FULL dataset"""

    config = TrainConfig()

    print("="*70)
    print("TRAIN MODULE 1 - FULL DATASET")
    print("="*70)
    print(f"\nCONFIGURATION:")
    print(f"  TRAIN_SAMPLES: {config.TRAIN_SAMPLES if config.TRAIN_SAMPLES else 'ALL'}")
    print(f"  EPOCHS: {config.EPOCHS}")
    print(f"  BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"  LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"  HIDDEN_DIMS: {config.HIDDEN_DIMS}")
    print(f"  DROPOUT: {config.DROPOUT}")

    print("\n" + "="*70)
    print("STEP 1: LOAD DATASET")
    print("="*70)

    # Load images
    print("\nLoading STROKE images...")
    stroke_images = load_all_images(config.STROKE_DIR, 0)
    print(f"  Found: {len(stroke_images)} images")

    print("\nLoading NORMAL images...")
    normal_images = load_all_images(config.NORMAL_DIR, 1)
    print(f"  Found: {len(normal_images)} images")

    total_images = len(stroke_images) + len(normal_images)
    print(f"\nTOTAL: {total_images} images")

    # Limit if specified
    if config.TRAIN_SAMPLES:
        stroke_images = stroke_images[:config.TRAIN_SAMPLES]
        normal_images = normal_images[:config.TRAIN_SAMPLES]
        print(f"Limited to: {len(stroke_images) + len(normal_images)} images")

    print("\n" + "="*70)
    print("STEP 2: EXTRACT FEATURES")
    print("="*70)

    # Initialize detector
    print("\nInitializing MediaPipe...")
    detector = FaceAsymmetryDetector(
        model_path="C:/Users/Admin/Documents/NCKHKT_26/fga_project/models/face_landmarker_v2/face_landmarker_v2_with_blendshapes.task"
    )

    print("\nExtracting features from Stroke images...")
    X_stroke, y_stroke = extract_features(stroke_images, detector)
    print(f"  Valid: {len(X_stroke)}/{len(stroke_images)}")

    print("\nExtracting features from Normal images...")
    X_normal, y_normal = extract_features(normal_images, detector)
    print(f"  Valid: {len(X_normal)}/{len(normal_images)}")

    # Combine
    X = np.vstack([X_stroke, X_normal])
    y = np.hstack([y_stroke, y_normal])

    print(f"\nTOTAL VALID SAMPLES: {len(X)}")
    print(f"  Stroke: {sum(y==0)}, Normal: {sum(y==1)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    print(f"\nTRAIN: {len(X_train)}, TEST: {len(X_test)}")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PyTorch Dataset
    class FaceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = FaceDataset(X_train_scaled, y_train)
    test_dataset = FaceDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print("\n" + "="*70)
    print("STEP 3: TRAIN MODEL")
    print("="*70)

    # Initialize model
    model = StrokeClassifier(
        input_dim=X.shape[1],
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT,
        num_classes=2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\nModel Architecture:")
    print(f"  Input: {X.shape[1]} features")
    print(f"  Hidden: {config.HIDDEN_DIMS}")
    print(f"  Output: 2 classes")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    num_epochs = config.EPOCHS
    best_acc = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
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

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{config.MODEL_DIR}/{config.MODEL_NAME}_best.pth')
            import joblib
            joblib.dump(scaler, f'{config.MODEL_DIR}/{config.MODEL_NAME}_scaler.pkl')

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) - Train: {train_acc:.2f}%, Val: {val_acc:.2f}%')

    total_time = time.time() - start_time

    print(f'\nTraining completed in {total_time/60:.1f} minutes')
    print(f'Best Validation Accuracy: {best_acc:.2f}%')

    # Final evaluation
    print("\n" + "="*70)
    print("STEP 4: FINAL EVALUATION")
    print("="*70)

    model.load_state_dict(torch.load(f'{config.MODEL_DIR}/{config.MODEL_NAME}_best.pth', weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(y_batch.numpy())

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {acc*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print("              Predicted")
    print("              Stroke  Normal")
    print(f"Actual Stroke   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual Normal  {cm[1][0]:4d}    {cm[1][1]:4d}")

    print("\nPer-class Report:")
    print(classification_report(all_labels, all_preds, target_names=['Stroke', 'Normal']))

    # Save model info
    model_info = {
        'accuracy': float(acc * 100),
        'confusion_matrix': cm.tolist(),
        'total_samples': int(total_images),
        'valid_samples': int(len(X)),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'config': {
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'hidden_dims': config.HIDDEN_DIMS,
            'dropout': config.DROPOUT
        },
        'training_time_minutes': float(total_time / 60)
    }

    with open(f'{config.MODEL_DIR}/{config.MODEL_NAME}_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\nModel saved to: {config.MODEL_DIR}/{config.MODEL_NAME}_best.pth")
    print(f"Scaler saved to: {config.MODEL_DIR}/{config.MODEL_NAME}_scaler.pkl")
    print(f"Info saved to: {config.MODEL_DIR}/{config.MODEL_NAME}_info.json")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

    print("\nTO IMPROVE ACCURACY:")
    print("1. Increase EPOCHS to 100, 200")
    print("2. Modify HIDDEN_DIMS to [1024, 512, 256]")
    print("3. Decrease LEARNING_RATE to 0.0001")
    print("4. Increase BATCH_SIZE if GPU available")

if __name__ == "__main__":
    train_full_dataset()
