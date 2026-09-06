"""
TRAIN LANDMARK CLASSIFIER FOR STROKE DETECTION
===============================================
Phương án C: MLP Classifier trên 468 MediaPipe Landmarks

Ưu điểm cho chuẩn y sinh:
- Sử dụng đặc trưng y học (facial landmarks) có thể giải thích được
- Accuracy: 90-95% (theo literature)
- Có thể so sánh với các chỉ số y khoa (FAST criteria)

Thời gian train: ~10 ngày (bao gồm data prep, train, validate, tune)
"""

import os
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path

# ============ CONFIG ============
CONFIG = {
    "dataset_path": r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\data\datasets\face\Annotated stroke and non stroke Dataset",
    "output_dir": r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\src\training\output",
    "image_size": (224, 224),
    "test_size": 0.2,
    "val_size": 0.2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "early_stopping_patience": 10,
    "random_seed": 42
}

# ============ STEP 1: DATASET LOADING ============
class StrokeLandmarkDataset(Dataset):
    """
    Dataset cho training landmark classifier

    Input: Image + Bounding Box
    Output: 468 landmarks (normalized) + Label (0=NonStroke, 1=Stroke)
    """

    def __init__(self, data_frame, transform=None):
        """
        Args:
            data_frame: DataFrame with columns: image_path, label, bbox
        """
        self.data_frame = data_frame
        self.transform = transform

        # Khởi tạo MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=True  # Important cho batch processing
        )

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Lấy thông tin từ DataFrame
        img_path = self.data_frame.iloc[idx]['image_path']
        label = self.data_frame.iloc[idx]['label']
        bbox = self.data_frame.iloc[idx]['bbox']  # [x, y, w, h] normalized

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Cannot load {img_path}")
            # Return zero landmarks if image fails
            landmarks = np.zeros((468, 2))
            return torch.FloatTensor(landmarks.flatten()), torch.LongTensor([label])[0]

        h, w = image.shape[:2]

        # Crop theo bounding box (tăng accuracy bằng cách focus vào face)
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        bw = int(bbox[2] * w)
        bh = int(bbox[3] * h)

        # Padding để không bị cắt mất phần mặt
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x1 + bw + 2*pad)
        y2 = min(h, y1 + bh + 2*pad)

        face_crop = image[y1:y2, x1:x2]

        # Extract landmarks
        landmarks = self._extract_landmarks(face_crop)

        # Normalize landmarks về [-1, 1] (relative to crop center)
        if landmarks is not None:
            landmarks = self._normalize_landmarks(landmarks, face_crop.shape[:2])
        else:
            landmarks = np.zeros((468, 2))

        return torch.FloatTensor(landmarks.flatten()), torch.LongTensor([label])[0]

    def _extract_landmarks(self, image):
        """Extract 468 landmarks từ image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        # Extract x, y coordinates (468 landmarks)
        landmark_array = []
        for lm in landmarks.landmark:
            landmark_array.append([lm.x, lm.y])

        return np.array(landmark_array)

    def _normalize_landmarks(self, landmarks, img_shape):
        """
        Normalize landmarks về [-1, 1] với tâm là (0, 0)

        Đây là bước QUAN TRỌNG cho biomedical standards:
        - Chuẩn hóa theo face center
        - Loại bỏ biến do khoảng cách camera
        - Loại bỏ biến do kích thước mặt
        """
        h, w = img_shape

        # Center landmarks around face center (landmark 5 or 1)
        # Landmark 1: Nose tip
        nose_tip = landmarks[1]
        cx, cy = nose_tip[0] * w, nose_tip[1] * h

        normalized = []
        for lm in landmarks:
            # Convert to pixel coordinates
            x = lm[0] * w - cx
            y = lm[1] * h - cy

            # Normalize by face width (distance between outer eyes: landmarks 33 and 263)
            # This makes the scale invariant
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            face_width = abs((right_eye[0] - left_eye[0]) * w)

            if face_width > 0:
                x = x / face_width
                y = y / face_width

            normalized.append([x, y])

        return np.array(normalized)


# ============ STEP 2: MLP CLASSIFIER ============
class LandmarkMLP(nn.Module):
    """
    MLP Classifier cho stroke detection dựa trên 468 landmarks

    Architecture:
    - Input: 936 features (468 landmarks × 2 coords)
    - Hidden layers: 512 → 256 → 128 → 64
    - Output: 1 neuron (sigmoid for binary classification)

    Đây là ARCHITECTURE ĐƠN GIẢN nhưng HIỆU QUẢ cho biomedical tasks:
    - Không overfit như CNN deep
    - Có thể giải thích được (feature importance)
    - Train nhanh trên CPU
    """

    def __init__(self, input_size=936, hidden_sizes=[512, 256, 128, 64], dropout=0.3):
        super(LandmarkMLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Build hidden layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # BatchNorm cho stability
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============ STEP 3: TRAINING FUNCTIONS ============
def prepare_dataset():
    """
    Chuẩn bị dataset từ folder structure

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    print("=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60)

    dataset_path = CONFIG["dataset_path"]

    # Scan dataset
    data = []

    # Load NonStroke
    nonstroke_path = os.path.join(dataset_path, "NonStroke")
    if os.path.exists(nonstroke_path):
        print(f"Loading NonStroke images from {nonstroke_path}...")
        for txt_file in tqdm(os.listdir(nonstroke_path)):
            if txt_file.endswith('.txt'):
                img_file = txt_file.replace('.txt', '.jpg')
                img_path = os.path.join(nonstroke_path, img_file)

                # Check if image exists
                if not os.path.exists(img_path):
                    continue

                # Read annotation
                txt_path = os.path.join(nonstroke_path, txt_file)
                with open(txt_path, 'r') as f:
                    line = f.readline().strip()
                    parts = line.split()
                    label = int(parts[0])  # Should be 0 for NonStroke
                    bbox = list(map(float, parts[1:]))  # [x, y, w, h]

                data.append({
                    'image_path': img_path,
                    'label': label,
                    'bbox': bbox
                })

    # Load Stroke
    stroke_path = os.path.join(dataset_path, "Stroke")
    if os.path.exists(stroke_path):
        print(f"Loading Stroke images from {stroke_path}...")
        for txt_file in tqdm(os.listdir(stroke_path)):
            if txt_file.endswith('.txt'):
                img_file = txt_file.replace('.txt', '.jpg')
                img_path = os.path.join(stroke_path, img_file)

                if not os.path.exists(img_path):
                    continue

                txt_path = os.path.join(stroke_path, txt_file)
                with open(txt_path, 'r') as f:
                    line = f.readline().strip()
                    parts = line.split()
                    label = int(parts[0])  # Should be 1 for Stroke
                    bbox = list(map(float, parts[1:]))

                data.append({
                    'image_path': img_path,
                    'label': label,
                    'bbox': bbox
                })

    print(f"\n✅ Total images loaded: {len(data)}")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save dataset info
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "dataset_info.csv", index=False)

    # Count labels
    label_counts = df['label'].value_counts()
    print(f"\nLabel distribution:")
    print(f"  NonStroke (0): {label_counts.get(0, 0)}")
    print(f"  Stroke (1): {label_counts.get(1, 0)}")

    # Split dataset
    train_df, temp_df = train_test_split(
        df,
        test_size=CONFIG["test_size"] + CONFIG["val_size"],
        random_state=CONFIG["random_seed"],
        stratify=df['label']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=CONFIG["test_size"] / (CONFIG["test_size"] + CONFIG["val_size"]),
        random_state=CONFIG["random_seed"],
        stratify=temp_df['label']
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Create datasets
    train_dataset = StrokeLandmarkDataset(train_df)
    val_dataset = StrokeLandmarkDataset(val_df)
    test_dataset = StrokeLandmarkDataset(test_df)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train một epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for landmarks, labels in tqdm(train_loader, desc="Training"):
        landmarks = landmarks.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        # Forward
        outputs = model(landmarks)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for landmarks, labels in tqdm(val_loader, desc="Validating"):
            landmarks = landmarks.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            # Forward
            outputs = model(landmarks)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_model():
    """
    Main training function
    """
    print("\n" + "=" * 60)
    print("LANDMARK CLASSIFIER TRAINING - STROKE DETECTION")
    print("=" * 60)
    print(f"\nConfig:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Set random seed
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Prepare dataset
    train_loader, val_loader, test_loader = prepare_dataset()

    # Initialize model
    print("\n" + "=" * 60)
    print("STEP 2: INITIALIZING MODEL")
    print("=" * 60)

    model = LandmarkMLP(
        input_size=936,  # 468 landmarks × 2 coords
        hidden_sizes=[512, 256, 128, 64],
        dropout=0.3
    ).to(device)

    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING")
    print("=" * 60)

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print stats
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            output_dir = Path(CONFIG["output_dir"])
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': CONFIG
            }, output_dir / "best_model.pth")
            print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")

            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n⏹ Early stopping triggered!")
                break

    # Plot training history
    plot_training_history(history)

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("STEP 4: FINAL EVALUATION")
    print("=" * 60)

    # Load best model
    output_dir = Path(CONFIG["output_dir"])
    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate_test_set(
        model, test_loader, criterion, device
    )

    # Save results
    save_training_results(history, test_acc, test_preds, test_labels)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir / 'best_model.pth'}")
    print(f"  Results saved to: {output_dir}")


def evaluate_test_set(model, test_loader, criterion, device):
    """Evaluate on test set with detailed metrics"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for landmarks, labels in tqdm(test_loader, desc="Testing"):
            landmarks = landmarks.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(landmarks)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    test_loss = test_loss / len(test_loader)

    predicted = (all_preds > 0.5).astype(int)
    test_acc = 100 * (predicted == all_labels).sum() / len(all_labels)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, predicted, target_names=['NonStroke', 'Stroke']))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, predicted)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nSensitivity (Recall for Stroke): {sensitivity:.4f}")
    print(f"Specificity (Recall for NonStroke): {specificity:.4f}")

    # AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_preds)
        print(f"AUC-ROC: {auc:.4f}")
    except:
        auc = None

    # Plot confusion matrix
    plot_confusion_matrix(cm)

    return test_loss, test_acc, all_preds, all_labels


def plot_training_history(history):
    """Plot training curves"""
    output_dir = Path(CONFIG["output_dir"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    print(f"📈 Training history plot saved to {output_dir / 'training_history.png'}")


def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    output_dir = Path(CONFIG["output_dir"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NonStroke', 'Stroke'],
                yticklabels=['NonStroke', 'Stroke'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set')
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    print(f"📊 Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")


def save_training_results(history, test_acc, test_preds, test_labels):
    """Save training results to JSON"""
    output_dir = Path(CONFIG["output_dir"])

    results = {
        'config': CONFIG,
        'training_history': history,
        'final_test_accuracy': float(test_acc),
        'training_completed': True
    }

    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to {output_dir / 'training_results.json'}")


# ============ MAIN ============
if __name__ == "__main__":
    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start training
    train_model()

    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review results in: {output_dir}")
    print(f"2. Best model: {output_dir / 'best_model.pth'}")
    print(f"3. Integrate with FaceModule: update face_module.py to use trained model")
