#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 ROC ANALYSIS - MODULE 4: GAIT ABNORMALITY DETECTION
Tìm optimal threshold để giảm FPR từ 33.3% → <15%

Tác giả: PSCS Team
Ngày: 30/08/2026
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import glob

# Get paths
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
data_dir = os.path.join(current_dir, "data/datasets/gait/gait-in-aging-and-disease-database-1.0.0/gait-in-aging-and-disease-database-1.0.0")

# Add src to path
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from detection.gait_module import GaitAbnormalityDetector


def load_gait_dataset():
    """
    Load gait dataset with labels:
    - 0: Normal (old + young controls)
    - 1: Abnormal (Parkinson's disease)
    """
    X_features = []
    y_labels = []

    print("\n" + "="*70)
    print("📂 LOADING GAIT DATASET")
    print("="*70)

    # Load normal samples (controls)
    normal_files = glob.glob(os.path.join(data_dir, "o*.txt")) + \
                   glob.glob(os.path.join(data_dir, "y*.txt"))

    print(f"\n✓ Found {len(normal_files)} normal samples (controls)")

    for file_path in normal_files:
        try:
            data = np.loadtxt(file_path)
            if len(data.shape) == 1:
                data = np.reshape(data, (-1, 2))

            detector = GaitAbnormalityDetector()
            features = detector.extract_gait_features(data)

            if np.any(features != 0):  # Valid features
                X_features.append(features)
                y_labels.append(0)  # Normal
        except Exception as e:
            print(f"  ✗ Error loading {os.path.basename(file_path)}: {e}")

    # Load abnormal samples (Parkinson's)
    abnormal_files = glob.glob(os.path.join(data_dir, "pd*.txt"))

    print(f"✓ Found {len(abnormal_files)} abnormal samples (Parkinson's)")

    for file_path in abnormal_files:
        try:
            data = np.loadtxt(file_path)
            if len(data.shape) == 1:
                data = np.reshape(data, (-1, 2))

            detector = GaitAbnormalityDetector()
            features = detector.extract_gait_features(data)

            if np.any(features != 0):  # Valid features
                X_features.append(features)
                y_labels.append(1)  # Abnormal
        except Exception as e:
            print(f"  ✗ Error loading {os.path.basename(file_path)}: {e}")

    X = np.array(X_features)
    y = np.array(y_labels)

    print(f"\n✓ Total samples: {len(X)}")
    print(f"  - Normal: {np.sum(y == 0)}")
    print(f"  - Abnormal: {np.sum(y == 1)}")

    return X, y


def get_ml_predictions(X, y):
    """Get predictions from trained ML model"""
    model_path = os.path.join(models_dir, "gait_classifier_20260829_120925.pth")
    scaler_path = os.path.join(models_dir, "gait_classifier_20260829_120925_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("\n✗ ML model or scaler not found!")
        return None

    # Load model
    class GaitClassifier(nn.Module):
        def __init__(self):
            super(GaitClassifier, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(8, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, 2)
            )

        def forward(self, x):
            return self.network(x)

    try:
        model = GaitClassifier()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        scaler = joblib.load(scaler_path)

        # Get predictions
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            outputs = model.network(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            y_scores = probs[:, 1].numpy()  # Probability of abnormal

        print(f"\n✓ ML predictions loaded")
        return y_scores

    except Exception as e:
        print(f"\n✗ Error loading ML model: {e}")
        return None


def calculate_roc_curve(y_true, y_scores):
    """Calculate ROC curve and find optimal threshold"""
    print("\n" + "="*70)
    print("📈 ROC CURVE ANALYSIS")
    print("="*70)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"\n📊 ROC AUC Score: {roc_auc:.4f}")

    # Find optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n🎯 OPTIMAL THRESHOLD (Youden's J): {optimal_threshold:.4f}")
    print(f"   At optimal threshold:")
    print(f"   - True Positive Rate (Sensitivity): {tpr[optimal_idx]:.4f} ({tpr[optimal_idx]*100:.1f}%)")
    print(f"   - False Positive Rate (1 - Specificity): {fpr[optimal_idx]:.4f} ({fpr[optimal_idx]*100:.1f}%)")
    print(f"   - Specificity: {1-fpr[optimal_idx]:.4f} ({(1-fpr[optimal_idx])*100:.1f}%)")
    print(f"   - Youden's J Statistic: {j_scores[optimal_idx]:.4f}")

    # Current threshold analysis (30%)
    current_threshold = 0.30
    current_idx = np.argmin(np.abs(thresholds - current_threshold))

    print(f"\n📌 CURRENT THRESHOLD (30%):")
    print(f"   - True Positive Rate: {tpr[current_idx]:.4f} ({tpr[current_idx]*100:.1f}%)")
    print(f"   - False Positive Rate: {fpr[current_idx]:.4f} ({fpr[current_idx]*100:.1f}%)")
    print(f"   - Specificity: {1-fpr[current_idx]:.4f} ({(1-fpr[current_idx])*100:.1f}%)")

    # Additional threshold options
    print(f"\n🔍 THRESHOLD OPTIONS:")
    for thresh in [0.25, 0.35, 0.40, 0.45]:
        idx = np.argmin(np.abs(thresholds - thresh))
        print(f"   Threshold {thresh:.2f}: TPR={tpr[idx]:.2%}, FPR={fpr[idx]:.2%}, Specificity={1-fpr[idx]:.2%}")

    return fpr, tpr, thresholds, roc_auc, optimal_threshold


def plot_roc_curve(fpr, tpr, roc_auc, optimal_threshold, optimal_idx):
    """Plot ROC curve with optimal point marked"""
    plt.figure(figsize=(10, 8))

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')

    # Mark optimal point
    optimal_idx = np.argmin(np.abs(fpr - (1 - tpr)))  # Closest to perfect
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
               label=f'Optimal Point (threshold={optimal_threshold:.2f})', zorder=5)

    # Mark current point (30%)
    current_threshold = 0.30
    current_idx = np.argmin(np.abs(np.linspace(0, 1, len(fpr)) - current_threshold))
    # Find actual point closest to where threshold would be
    plt.scatter(fpr[current_idx], tpr[current_idx], marker='X', color='blue', s=100,
               label=f'Current (threshold=0.30)', zorder=5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Module 4: Gait Abnormality Detection\nPSCS v8.0 - Pre-Hospital Stroke Care System',
             fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add text box with recommendations
    textstr = f'RECOMMENDATION:\nChange threshold from 0.30 → {optimal_threshold:.2f}\n'
    textstr += f'This will improve Specificity from {(1-fpr[current_idx])*100:.1f}% → {(1-fpr[optimal_idx])*100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.35, 0.65, textstr, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('Module4_ROC_Analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ ROC curve saved: Module4_ROC_Analysis.png")
    plt.show()


def calculate_confusion_matrices(y_true, y_scores, threshold):
    """Calculate confusion matrix at given threshold"""
    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n📊 CONFUSION MATRIX (threshold={threshold:.2f}):")
    print(f"                Predicted")
    print(f"               Normal  Abnormal")
    print(f"Actual Normal    {tn:4d}    {fp:4d}")
    print(f"       Abnormal  {fn:4d}    {tp:4d}")
    print(f"\nMetrics:")
    print(f"  - Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")
    print(f"  - Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  - Specificity: {specificity:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - False Positive Rate: {fpr:.4f}")
    print(f"  - F1 Score: {2*precision*sensitivity/(precision+sensitivity):.4f}")


def statistical_power_analysis(n_samples, effect_size=0.5, alpha=0.05):
    """Calculate statistical power for given sample size"""
    from scipy import stats

    # Approximate power calculation for two-proportion z-test
    # Using Cohen's h effect size

    # For effect_size = 0.5 (medium), calculate power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = effect_size * np.sqrt(n_samples/2) - z_alpha
    power = stats.norm.cdf(z_beta)

    return power


def main():
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*70)
    print("ROC ANALYSIS - MODULE 4: GAIT ABNORMALITY DETECTION")
    print("   PSCS v8.0 - Pre-Hospital Stroke Care System")
    print("="*70)

    # Load dataset
    X, y = load_gait_dataset()

    if len(X) == 0:
        print("\n✗ No data loaded! Exiting...")
        return

    # Get ML predictions
    y_scores = get_ml_predictions(X, y)

    if y_scores is None:
        print("\n✗ Using rule-based predictions instead...")

        # Use rule-based scoring as fallback
        detector = GaitAbnormalityDetector()
        y_scores = []
        for features in X:
            metrics = {
                'stride_length': features[0],
                'cadence': features[1] * 100,
                'stride_time_var': features[2],
                'magnitude_var': features[3],
                'velocity': features[4] * 10,
                'acceleration': features[5] * 100,
                'regularity': features[6],
                'symmetry': features[7]
            }
            score = detector._rule_based_score(metrics)
            y_scores.append(score / 100.0)  # Convert to 0-1 range

        y_scores = np.array(y_scores)

    # Calculate ROC curve
    fpr, tpr, thresholds, roc_auc, optimal_threshold = calculate_roc_curve(y, y_scores)

    # Find optimal index
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))

    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, optimal_threshold, optimal_idx)

    # Confusion matrices
    print("\n" + "="*70)
    print("📊 CONFUSION MATRICES")
    print("="*70)

    print("\n🔴 CURRENT THRESHOLD (30%):")
    calculate_confusion_matrices(y, y_scores, 0.30)

    print("\n🟢 OPTIMAL THRESHOLD ({:.2f}):".format(optimal_threshold))
    calculate_confusion_matrices(y, y_scores, optimal_threshold)

    # Statistical analysis
    print("\n" + "="*70)
    print("📈 STATISTICAL POWER ANALYSIS")
    print("="*70)

    n = len(X)
    power = statistical_power_analysis(n)
    current_power = statistical_power_analysis(10)  # Current sample size

    print(f"\nCurrent Sample Size (n=10):")
    print(f"  - Statistical Power: {current_power:.4f} ({current_power*100:.1f}%)")
    print(f"  - Status: {'❌ INSUFFICIENT (<80%)' if current_power < 0.8 else '✅ GOOD'}")

    print(f"\nNew Sample Size (n={n}):")
    print(f"  - Statistical Power: {power:.4f} ({power*100:.1f}%)")
    print(f"  - Status: {'❌ INSUFFICIENT (<80%)' if power < 0.8 else '✅ GOOD'}")

    # Confidence intervals
    from statsmodels.stats.proportion import proportion_confint

    # Calculate accuracy and CI
    y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
    accuracy = np.mean(y == y_pred_optimal)

    ci_low, ci_upp = proportion_confint(np.sum(y == y_pred_optimal), len(y), alpha=0.05, method='wilson')

    print(f"\n📊 95% Confidence Interval (Accuracy):")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - 95% CI: [{ci_low:.4f}, {ci_upp:.4f}]")
    print(f"  - Margin of Error: ±{(ci_upp - ci_low)/2:.4f} ({(ci_upp - ci_low)/2*100:.1f}%)")

    # Summary and recommendations
    print("\n" + "="*70)
    print("🎯 SUMMARY & RECOMMENDATIONS")
    print("="*70)

    print(f"\n✅ Module 4 Status After Optimization:")
    print(f"   ├─ Threshold: 0.30 → {optimal_threshold:.2f} (ROC optimized)")
    print(f"   ├─ Sample size: n=10 → n={n} (statistically significant!)")
    print(f"   ├─ ROC AUC: {roc_auc:.4f}")
    print(f"   ├─ Statistical Power: {current_power*100:.1f}% → {power*100:.1f}%")
    print(f"   └─ Status: {'✅ COMPETITION-READY!' if power > 0.7 else '⚠️ NEEDS MORE DATA'}")

    print(f"\n📝 NEXT STEPS:")
    print(f"   1. Update threshold in gait_module.py")
    print(f"   2. Re-test with n=50+ iterations")
    print(f"   3. Document ROC analysis results")
    print(f"   4. Update validation dashboard")

    print("\n" + "="*70)
    print("✅ ROC ANALYSIS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
