# -*- coding: utf-8 -*-
"""
Train Multi-class Classifier for Module 1
Classes: STROKE, YAWN, SMILE, HEAD_TURN, NORMAL
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

print("=== MULTI-CLASS CLASSIFIER: STROKE vs YAWN vs SMILE vs HEAD_TURN vs NORMAL ===")

# Paths
DATASET_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/data/datasets/face"
MODEL_PATH = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/models/face_landmarker_v2/face_landmarker_v2_with_blendshapes.task"
OUTPUT_DIR = "C:/Users/Admin/Documents/NCKHKT_26/fga_project/models"

# Class definitions (5-class for future)
FULL_CLASSES = {
    'STROKE': 0,
    'YAWN': 1,
    'SMILE': 2,
    'HEAD_TURN': 3,
    'NORMAL': 4
}

# Current 2-class mapping
CLASSES = {
    'STROKE': 0,
    'NORMAL': 1
}

CLASS_NAMES = list(CLASSES.keys())
NUM_CLASSES = len(CLASSES)

print(f"\nFull Classes (future): {list(FULL_CLASSES.keys())}")
print(f"Current Classes: {CLASS_NAMES}")

# Load dataset function
def load_images_from_dirs(dataset_dirs):
    """
    Load images from multiple directories with labels

    Args:
        dataset_dirs: Dict of {class_name: directory_path}

    Returns:
        List of (image_path, label)
    """
    all_images = []

    for class_name, dir_path in dataset_dirs.items():
        if not Path(dir_path).exists():
            print(f"WARNING: Directory not found: {dir_path}")
            continue

        print(f"Loading {class_name} from: {dir_path}")

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            for img_path in Path(dir_path).glob(ext):
                all_images.append((str(img_path), CLASSES[class_name]))

        print(f"  Found {sum(1 for p, l in all_images if l == CLASSES[class_name])} images")

    return all_images

# Use current 2-class dataset
dataset_dirs = {
    'STROKE': f"{DATASET_DIR}/Annotated stroke and non stroke Dataset/Stroke",
    'NORMAL': f"{DATASET_DIR}/Annotated stroke and non stroke Dataset/NonStroke",
}

# TODO: Add datasets for YAWN, SMILE, HEAD_TURN for 5-class training

print("\nNote: Currently using STROKE and NORMAL datasets.")
print("TODO: Add YAWN, SMILE, HEAD_TURN datasets for full multi-class training")
print("\nFor demonstration, we'll create a 2-class model first (STROKE vs NORMAL)")
print("Then extend to 5-class when datasets are available.\n")

# Initialize detector
detector = FaceAsymmetryDetector(model_path=MODEL_PATH)

# Global timestamp for ALL images
global_timestamp = 0

# Extract features
def extract_features_with_augmentation(images, detector, use_global_timestamp=True):
    """
    Extract features from images with optional augmentation

    For YAWN: Detect mouth opening
    For SMILE: Detect mouth corners
    For HEAD_TURN: Detect face profile
    """
    features = []
    labels = []
    raw_metrics_list = []

    global global_timestamp
    local_timestamp = 0

    for img_path, label in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = detector.process_frame(img, frame_timestamp_ms=global_timestamp if use_global_timestamp else local_timestamp)
        global_timestamp += 33
        local_timestamp += 33

        if result.get('status') in ['NO_FACE', 'NO_DETECTOR']:
            continue

        landmarks = result.get('raw_landmarks')
        if landmarks is None:
            continue

        # Basic features: flattened landmarks (956)
        basic_features = landmarks.flatten()

        # ADDITIONAL FEATURES FOR BETTER CLASSIFICATION:

        # 1. Mouth aspect ratio (for YAWN detection)
        # Landmarks: upper lip (13), lower lip (14), left corner (61), right corner (291)
        try:
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_corner = landmarks[61]
            right_corner = landmarks[291]

            # Mouth opening (vertical distance)
            mouth_opening = np.linalg.norm(upper_lip - lower_lip)

            # Mouth width (horizontal distance)
            mouth_width = np.linalg.norm(left_corner - right_corner)

            # Mouth aspect ratio
            mouth_ar = mouth_opening / (mouth_width + 1e-6)
        except:
            mouth_ar = 0

        # 2. Eye aspect ratio (for drowsiness/closure)
        # Landmarks: eye top/bottom
        try:
            # Left eye: 159 (top), 145 (bottom)
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            left_eye_ar = np.linalg.norm(left_eye_top - left_eye_bottom)

            # Right eye: 386 (top), 374 (bottom)
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            right_eye_ar = np.linalg.norm(right_eye_top - right_eye_bottom)

            eye_ar = (left_eye_ar + right_eye_ar) / 2
        except:
            eye_ar = 0

        # 3. Face rotation angle (for HEAD_TURN detection)
        # Use nose bridge and chin to estimate rotation
        try:
            nose_tip = landmarks[1]
            chin = landmarks[152]
            face_center = landmarks[5]

            # Calculate deviation from center
            deviation = face_center[0] - 0.5  # 0.5 is center of normalized coordinates
            rotation = deviation * 100  # Scale up
        except:
            rotation = 0

        # 4. Smile detection
        try:
            # Distance from lip corners to face center
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            face_center = landmarks[5]

            # Calculate smile curvature
            left_dist = np.linalg.norm(left_corner - face_center)
            right_dist = np.linalg.norm(right_corner - face_center)

            # Smile indicator: corners pulled outward/upward
            smile_indicator = (left_dist + right_dist) / 2
        except:
            smile_indicator = 0

        # Combine all features
        additional_features = [mouth_ar, eye_ar, rotation, smile_indicator]
        combined_features = np.concatenate([basic_features, additional_features])

        features.append(combined_features)
        labels.append(label)
        raw_metrics_list.append(result.get('raw_metrics', {}))

    return np.array(features), np.array(labels), raw_metrics_list

print("Loading dataset...")
all_images = load_images_from_dirs(dataset_dirs)
print(f"Total images loaded: {len(all_images)}")

if len(all_images) == 0:
    print("ERROR: No images loaded!")
    sys.exit(1)

# Limit samples for speed
print("\nExtracting features (limited to 500 each)...")
stroke_images = [img for img in all_images if img[1] == CLASSES['STROKE']][:500]
normal_images = [img for img in all_images if img[1] == CLASSES['NORMAL']][:500]

limited_images = stroke_images + normal_images

print(f"Processing {len(limited_images)} images...")

X_stroke, y_stroke, metrics_stroke = extract_features_with_augmentation(stroke_images, detector, use_global_timestamp=True)
print(f"Stroke: {len(X_stroke)} valid samples")

X_normal, y_normal, metrics_normal = extract_features_with_augmentation(normal_images, detector, use_global_timestamp=True)
print(f"Normal: {len(X_normal)} valid samples")

# Combine
X = np.vstack([X_stroke, X_normal])
y = np.hstack([y_stroke, y_normal])

print(f"\nTotal features shape: {X.shape}")
print(f"  Basic landmarks: 956")
print(f"  Additional features: 4 (mouth_ar, eye_ar, rotation, smile)")
print(f"  Total: {X.shape[1]}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Multi-class Model
class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim=960, hidden_dims=[512, 256, 128], num_classes=5):
        super(MultiClassClassifier, self).__init__()

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

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# For now, use 2 classes (STROKE vs NORMAL)
model = MultiClassClassifier(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print("\n=== TRAINING MULTI-CLASS MODEL ===")
print(f"Model: MLP with layers [{X.shape[1]} -> 512 -> 256 -> 128 -> {2}]")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Classes: 2 (STROKE=0, NORMAL=1)")
print("\nNote: Change num_classes=5 when YAWN, SMILE, HEAD_TURN datasets available")

# Training
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
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/multi_class_classifier_best.pth')
        import joblib
        joblib.dump(scaler, f'{OUTPUT_DIR}/multi_class_scaler.pkl')

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

print(f'\nBest Validation Accuracy: {best_acc:.2f}%')

# Load best model
model.load_state_dict(torch.load(f'{OUTPUT_DIR}/multi_class_classifier_best.pth', weights_only=True))
model.eval()

# Evaluation
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

print("\n=== FINAL RESULTS ===")
print(f"Accuracy: {acc*100:.2f}%")
print(f"\nConfusion Matrix:")
print("              Predicted")
print("              Stroke  Normal")
print(f"Actual Stroke   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"Actual Normal  {cm[1][0]:4d}    {cm[1][1]:4d}")

print("\nPer-class Report:")
target_names = ['Stroke', 'Normal']
print(classification_report(all_labels, all_preds, target_names=target_names))

# Save model info
model_info = {
    'accuracy': acc * 100,
    'num_classes': 2,
    'class_names': ['Stroke', 'Normal'],
    'input_dim': X.shape[1],
    'feature_types': {
        'landmarks': 956,
        'mouth_ar': 1,
        'eye_ar': 1,
        'rotation': 1,
        'smile': 1
    },
    'confusion_matrix': cm.tolist()
}

with open(f'{OUTPUT_DIR}/multi_class_classifier_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\nModel saved to: {OUTPUT_DIR}/multi_class_classifier_best.pth")
print(f"Scaler saved to: {OUTPUT_DIR}/multi_class_scaler.pkl")
print(f"Info saved to: {OUTPUT_DIR}/multi_class_classifier_info.json")

print("\n=== TODO FOR 5-CLASS MODEL ===")
print("1. Collect YAWN dataset (images of people yawning)")
print("2. Collect SMILE dataset (images of people smiling)")
print("3. Collect HEAD_TURN dataset (profile view faces)")
print("4. Update dataset_dirs with these paths")
print("5. Change num_classes=5 in model")
print("6. Update CLASS_NAMES = ['STROKE', 'YAWN', 'SMILE', 'HEAD_TURN', 'NORMAL']")

print("\nTraining: DONE!")
