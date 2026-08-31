#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏋️ TRAIN MODULE 5: VISUAL FIELD DEFECT DETECTION - ML MODEL
Train Neural Network để phát hiện khiếm khuyết thị trường

Tác giả: PSCS Team
Ngày: 31/08/2026
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from datetime import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)


# ============================================================================
# STEP 1: GENERATE SYNTHETIC TRAINING DATA
# ============================================================================

def generate_visual_field_training_data(n_samples=1000):
    """
    Tạo synthetic training data cho visual field defect detection

    Features (20 total):
    ┌─────────────────────────────────────────────────────────┐
    │  Eye Tracking (12 features):                              │
    │  1-6: Left eye gaze (x, y at 3 time points)               │
    │  7-12: Right eye gaze (x, y at 3 time points)              │
│                                                          │
│  Visual Field Test (8 features):                            │
│  13-20: Stimuli detection at 8 positions                    │
│         (center, TL, T, TR, R, BR, B, BL)                 │
│                                                          │
│  Total: 20 features                                        │
└─────────────────────────────────────────────────────────┘
    """
    print("\n[STEP 1] Generating synthetic training data...")
    print(f"Target: {n_samples} samples")

    X = []
    y = []

    for i in range(n_samples):
        # Generate eye tracking data
        sample = []

        # Normal eye position (centered, with some variation)
        base_x = 0.5
        base_y = 0.5

        # Generate 3 time points for each eye
        left_gaze_points = []
        right_gaze_points = []

        for t in range(3):
            # Add some natural variation
            noise_x = np.random.normal(0, 0.02)
            noise_y = np.random.normal(0, 0.02)

            left_gaze_points.extend([
                base_x + noise_x,
                base_y + noise_y
            ])

            right_gaze_points.extend([
                base_x + noise_x,
                base_y + noise_y
            ])

        sample.extend(left_gaze_points)
        sample.extend(right_gaze_points)

        # Decide label (0: normal, 1: abnormal)
        label = np.random.choice([0, 1], p=[0.65, 0.35])  # 35% abnormal

        # Visual field test results (8 stimuli positions)
        # Positions: center, TL, T, TR, R, BR, B, BL
        visual_field_results = []

        if label == 0:
            # Normal: can see most stimuli (6-8/8)
            n_seen = np.random.choice([6, 7, 8], p=[0.1, 0.3, 0.6])
        else:
            # Abnormal: missing stimuli depending on defect type
            defect_types = ['left_hemianopia', 'right_hemianopia',
                          'quadrantanopia', 'partial']
            defect = np.random.choice(defect_types, p=[0.25, 0.25, 0.3, 0.2])

            if defect == 'left_hemianopia':
                # Miss left side stimuli (TL, BL, L)
                visual_field_results = [1, 0, 1, 1, 1, 1, 0, 0]  # TL=0, BL=0, L=0
            elif defect == 'right_hemianopia':
                # Miss right side stimuli (TR, BR, R)
                visual_field_results = [1, 1, 1, 0, 0, 0, 1, 1]  # TR=0, BR=0, R=0
            elif defect == 'quadrantanopia':
                # Miss one quadrant (random)
                quadrant = np.random.choice([0, 1, 2, 3])
                if quadrant == 0:  # Top-left
                    visual_field_results = [1, 0, 1, 1, 1, 1, 1, 1]
                elif quadrant == 1:  # Top-right
                    visual_field_results = [1, 1, 1, 0, 1, 1, 1, 1]
                elif quadrant == 2:  # Bottom-right
                    visual_field_results = [1, 1, 1, 1, 0, 0, 1, 1]
                else:  # Bottom-left
                    visual_field_results = [1, 1, 1, 1, 1, 0, 0, 1]
            else:  # Partial defect
                n_seen = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
                visual_field_results = [1] * n_seen + [0] * (8 - n_seen)
                np.random.shuffle(visual_field_results)

        if label == 0:
            # Generate random seen stimuli for normal
            n_seen = np.random.choice([6, 7, 8], p=[0.1, 0.3, 0.6])
            visual_field_results = [1] * n_seen + [0] * (8 - n_seen)
            np.random.shuffle(visual_field_results)

        sample.extend(visual_field_results)

        X.append(sample)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"✓ Generated {len(X)} samples")
    print(f"  - Normal: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  - Abnormal: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")

    return X, y


# ============================================================================
# STEP 2: CREATE PYTORCH DATASET
# ============================================================================

class VisualFieldDataset(Dataset):
    """Dataset cho visual field defect detection"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# STEP 3: DEFINE NEURAL NETWORK MODEL
# ============================================================================

class VisualFieldClassifier(nn.Module):
    """
    Neural Network cho Visual Field Defect Detection

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Input: 20 features (eye tracking + visual field test)   │
    │    ↓                                                     │
    │  Dense(20 → 40) + ReLU + BatchNorm + Dropout(0.3)        │
    │    ↓                                                     │
    │  Dense(40 → 80) + ReLU + BatchNorm + Dropout(0.3)        │
    │    ↓                                                     │
    │  Dense(80 → 40) + ReLU + BatchNorm                       │
    │    ↓                                                     │
    │  Dense(40 → 2) + Softmax                                 │
    │    ↓                                                     │
    │  Output: [Normal_prob, Defect_prob]                      │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super(VisualFieldClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(40, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(80, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(),

            nn.Linear(40, 2)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# STEP 4: TRAINING FUNCTION
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train model với validation"""

    print("\n[STEP 2] Creating datasets...")

    # Create datasets
    train_dataset = VisualFieldDataset(X_train, y_train)
    val_dataset = VisualFieldDataset(X_val, y_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")

    # Initialize model
    print("\n[STEP 3] Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = VisualFieldClassifier().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    print(f"\n[STEP 4] Training for {epochs} epochs...")

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 20

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n✓ Training completed")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model


# ============================================================================
# STEP 5: EVALUATION & METRICS
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model và calculate comprehensive metrics"""

    print("\n[STEP 5] Evaluating model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.LongTensor(y_test).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)

    # Convert to numpy
    y_pred = predictions.cpu().numpy()
    y_proba = probs[:, 1].cpu().numpy()
    y_true = y_test

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # ROC AUC
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = 0.5

    # Print results
    print(f"\n{'='*60}")
    print(f"MODULE 5 - VISUAL FIELD DEFECT: TEST RESULTS")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"               Normal  Abnormal")
    print(f"Actual Normal    {tn:4d}    {fp:4d}")
    print(f"       Abnormal  {fn:4d}    {tp:4d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"  FPR:         {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  ROC AUC:     {auc:.4f}")
    print(f"\n{'='*60}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'fpr': fpr,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }


# ============================================================================
# STEP 6: SAVE MODEL
# ============================================================================

def save_model(model, scaler, metrics, output_dir='models'):
    """Save model, scaler, và metadata"""

    print("\n[STEP 6] Saving model...")

    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model
    model_path = os.path.join(output_dir, f'visual_field_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved: {model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, f'visual_field_{timestamp}_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'metrics': metrics,
        'input_features': 20,
        'architecture': 'VisualFieldClassifier',
        'description': 'Visual field defect detection from eye tracking',
        'nihss_item': 'Item 3 - Visual'
    }

    metadata_path = os.path.join(output_dir, f'visual_field_{timestamp}_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")

    return model_path, scaler_path, metadata_path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training pipeline"""
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*70)
    print("TRAINING MODULE 5: VISUAL FIELD DEFECT ML MODEL")
    print("   PSCS v8.0 - Pre-Hospital Stroke Care System")
    print("="*70)

    # Configuration
    N_SAMPLES = 1000
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    EPOCHS = 100
    BATCH_SIZE = 32
    RANDOM_STATE = 42

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    try:
        # Step 1: Generate data
        X, y = generate_visual_field_training_data(N_SAMPLES)

        # Step 2: Split data (train/val/test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
        )

        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        # Step 3: Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Step 4: Train model
        model = train_model(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        # Step 5: Evaluate model
        metrics = evaluate_model(model, X_test_scaled, y_test)

        # Step 6: Save model
        model_path, scaler_path, metadata_path = save_model(model, scaler, metrics)

        # Final summary
        print("\n" + "="*70)
        print("✅ MODULE 5 ML MODEL TRAINING COMPLETED!")
        print("="*70)
        print(f"\nModel Performance:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  FPR: {metrics['fpr']*100:.2f}%")
        print(f"  AUC: {metrics['auc']:.4f}")

        if metrics['fpr'] < 0.15:
            print(f"\n✅ FPR < 15%: COMPETITION-READY!")
        else:
            print(f"\n⚠️ FPR ≥ 15%: Needs improvement")

        print(f"\nOutput files:")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print(f"  Metadata: {metadata_path}")

        print("\n" + "="*70)
        print("Ready to integrate into Module 5!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
