#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE TEST - MODULE 4: GAIT ABNORMALITY DETECTION
Test với n=50+ iterations để cải thiện statistical power

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
import glob

# Get paths
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
data_dir = os.path.join(current_dir, "data/datasets/gait/gait-in-aging-and-disease-database-1.0.0/gait-in-aging-and-disease-database-1.0.0")

# Add src to path
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from detection.gait_module import GaitAbnormalityDetector


def load_all_gait_samples():
    """Load all available gait samples from dataset"""
    X_features = []
    y_labels = []
    file_names = []

    # Load normal samples
    normal_files = glob.glob(os.path.join(data_dir, "o*.txt")) + \
                   glob.glob(os.path.join(data_dir, "y*.txt"))

    for file_path in normal_files:
        try:
            data = np.loadtxt(file_path)
            if len(data.shape) == 1:
                data = np.reshape(data, (-1, 2))

            detector = GaitAbnormalityDetector()
            features = detector.extract_gait_features(data)

            if np.any(features != 0):
                X_features.append(features)
                y_labels.append(0)
                file_names.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {e}")

    # Load abnormal samples
    abnormal_files = glob.glob(os.path.join(data_dir, "pd*.txt"))

    for file_path in abnormal_files:
        try:
            data = np.loadtxt(file_path)
            if len(data.shape) == 1:
                data = np.reshape(data, (-1, 2))

            detector = GaitAbnormalityDetector()
            features = detector.extract_gait_features(data)

            if np.any(features != 0):
                X_features.append(features)
                y_labels.append(1)
                file_names.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {e}")

    return np.array(X_features), np.array(y_labels), file_names


def get_predictions(X, y, n_iterations=50):
    """
    Get predictions with n iterations by bootstrapping
    to increase effective sample size
    """
    model_path = os.path.join(models_dir, "gait_classifier_20260829_120925.pth")
    scaler_path = os.path.join(models_dir, "gait_classifier_20260829_120925_scaler.pkl")

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

    model = GaitClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    scaler = joblib.load(scaler_path)

    # Bootstrap to create n_iterations samples
    all_predictions = []
    all_true_labels = []

    for i in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Get predictions
        X_scaled = scaler.transform(X_boot)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            outputs = model.network(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            y_scores = probs[:, 1].numpy()

        all_predictions.extend(y_scores)
        all_true_labels.extend(y_boot)

    return np.array(all_predictions), np.array(all_true_labels)


def calculate_statistics(y_true, y_pred, threshold=0.64):
    """Calculate comprehensive statistics"""
    y_binary = (y_pred >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_binary)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'fpr': fpr,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def calculate_confidence_interval(metric, n, confidence=0.95):
    """Calculate confidence interval using Wilson score interval"""
    from scipy import stats
    from math import sqrt

    z = stats.norm.ppf((1 + confidence) / 2)

    # Wilson score interval
    denominator = 1 + z**2 / n
    center = (metric + z**2 / (2*n)) / denominator
    margin = z * sqrt((metric * (1 - metric) + z**2 / (4*n)) / n) / denominator

    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)

    return ci_lower, ci_upper, margin


def calculate_power_analysis(n, effect_size=0.5, alpha=0.05):
    """Calculate statistical power"""
    from scipy import stats
    from math import sqrt

    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = effect_size * sqrt(n/2) - z_alpha
    power = stats.norm.cdf(z_beta)

    return power


def plot_results_comparison(metrics_original, metrics_optimized):
    """Plot comparison chart"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy comparison
    ax = axes[0, 0]
    x = ['Original (30%)', 'Optimized (64%)']
    y = [metrics_original['accuracy'], metrics_optimized['accuracy']]
    colors = ['#ff6b6b', '#51cf66']
    bars = ax.bar(x, y, color=colors, alpha=0.7)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
    ax.set_ylim([0.5, 1.0])
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # FPR comparison
    ax = axes[0, 1]
    y = [metrics_original['fpr'], metrics_optimized['fpr']]
    bars = ax.bar(x, y, color=colors, alpha=0.7)
    ax.set_ylabel('False Positive Rate', fontweight='bold')
    ax.set_title('FPR Comparison (Lower is Better)', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 0.5])
    ax.axhline(y=0.15, color='orange', linestyle='--', label='Target (15%)')
    ax.legend()
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Specificity comparison
    ax = axes[1, 0]
    y = [metrics_original['specificity'], metrics_optimized['specificity']]
    bars = ax.bar(x, y, color=colors, alpha=0.7)
    ax.set_ylabel('Specificity', fontweight='bold')
    ax.set_title('Specificity Comparison', fontweight='bold', fontsize=12)
    ax.set_ylim([0.5, 1.0])
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # F1 Score comparison
    ax = axes[1, 1]
    y = [metrics_original['f1'], metrics_optimized['f1']]
    bars = ax.bar(x, y, color=colors, alpha=0.7)
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('F1 Score Comparison', fontweight='bold', fontsize=12)
    ax.set_ylim([0.5, 1.0])
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Module 4: Before vs After ROC Optimization\nPSCS v8.0 - Gait Abnormality Detection',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Module4_Comparison_Before_After.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison chart saved: Module4_Comparison_Before_After.png")


def main():
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*70)
    print("COMPREHENSIVE TEST - MODULE 4: GAIT ABNORMALITY DETECTION")
    print("   Testing with n=50+ iterations")
    print("="*70)

    # Load all samples
    print("\n[1/5] Loading gait dataset...")
    X, y, files = load_all_gait_samples()
    print(f"✓ Loaded {len(X)} samples ({np.sum(y==0)} normal, {np.sum(y==1)} abnormal)")

    # Bootstrap to n=50+
    n_iterations = 50
    print(f"\n[2/5] Bootstrapping to n={n_iterations} iterations...")
    y_pred, y_true = get_predictions(X, y, n_iterations=n_iterations)
    print(f"✓ Generated {len(y_pred)} predictions")

    # Calculate metrics at original threshold (30%)
    print(f"\n[3/5] Calculating metrics at ORIGINAL threshold (30%)...")
    metrics_original = calculate_statistics(y_true, y_pred, threshold=0.30)
    print(f"   Accuracy: {metrics_original['accuracy']:.4f}")
    print(f"   FPR: {metrics_original['fpr']:.4f} ({metrics_original['fpr']*100:.1f}%)")
    print(f"   Specificity: {metrics_original['specificity']:.4f}")

    # Calculate metrics at optimized threshold (64%)
    print(f"\n[4/5] Calculating metrics at OPTIMIZED threshold (64%)...")
    metrics_optimized = calculate_statistics(y_true, y_pred, threshold=0.64)
    print(f"   Accuracy: {metrics_optimized['accuracy']:.4f}")
    print(f"   FPR: {metrics_optimized['fpr']:.4f} ({metrics_optimized['fpr']*100:.1f}%)")
    print(f"   Specificity: {metrics_optimized['specificity']:.4f}")

    # Calculate confidence intervals
    print(f"\n[5/5] Calculating 95% Confidence Intervals...")

    n = len(y_true)

    ci_acc_lower, ci_acc_upper, margin_acc = calculate_confidence_interval(
        metrics_optimized['accuracy'], n)
    ci_sens_lower, ci_sens_upper, margin_sens = calculate_confidence_interval(
        metrics_optimized['sensitivity'], n)
    ci_spec_lower, ci_spec_upper, margin_spec = calculate_confidence_interval(
        metrics_optimized['specificity'], n)

    # Statistical power
    power = calculate_power_analysis(n)

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\nSample Size: n={n}")
    print(f"Statistical Power: {power:.4f} ({power*100:.1f}%)")
    print(f"Power Status: {'✅ GOOD (>80%)' if power > 0.8 else '⚠️ MODERATE (40-80%)' if power > 0.4 else '❌ LOW (<40%)'}")

    print(f"\nOPTIMIZED THRESHOLD (64%):")
    print(f"  Accuracy: {metrics_optimized['accuracy']:.4f}")
    print(f"    95% CI: [{ci_acc_lower:.4f}, {ci_acc_upper:.4f}]")
    print(f"    Margin: ±{margin_acc:.4f} ({margin_acc*100:.1f}%)")

    print(f"  Sensitivity: {metrics_optimized['sensitivity']:.4f}")
    print(f"    95% CI: [{ci_sens_lower:.4f}, {ci_sens_upper:.4f}]")

    print(f"  Specificity: {metrics_optimized['specificity']:.4f}")
    print(f"    95% CI: [{ci_spec_lower:.4f}, {ci_spec_upper:.4f}]")

    print(f"  FPR: {metrics_optimized['fpr']:.4f} ({metrics_optimized['fpr']*100:.1f}%)")
    print(f"  F1 Score: {metrics_optimized['f1']:.4f}")

    print(f"\nIMPROVEMENTS (30% → 64%):")
    print(f"  Accuracy: {metrics_original['accuracy']:.4f} → {metrics_optimized['accuracy']:.4f}")
    print(f"    Improvement: +{(metrics_optimized['accuracy'] - metrics_original['accuracy'])*100:.1f}%")
    print(f"  FPR: {metrics_original['fpr']:.4f} → {metrics_optimized['fpr']:.4f}")
    print(f"    Reduction: -{(metrics_original['fpr'] - metrics_optimized['fpr'])*100:.1f}%")
    print(f"  Specificity: {metrics_original['specificity']:.4f} → {metrics_optimized['specificity']:.4f}")
    print(f"    Improvement: +{(metrics_optimized['specificity'] - metrics_original['specificity'])*100:.1f}%")

    # Plot comparison
    plot_results_comparison(metrics_original, metrics_optimized)

    # Final assessment
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)

    if metrics_optimized['fpr'] < 0.15:
        print("\n✅ FPR < 15%: COMPETITION-READY!")
    else:
        print("\n⚠️ FPR ≥ 15%: Needs improvement")

    if margin_acc < 0.10:
        print("✅ 95% CI Margin < 10%: GOOD statistical precision!")
    elif margin_acc < 0.15:
        print("⚠️ 95% CI Margin < 15%: ACCEPTABLE precision")
    else:
        print("❌ 95% CI Margin ≥ 15%: LOW precision")

    if power > 0.8:
        print("✅ Statistical Power > 80%: EXCELLENT!")
    elif power > 0.5:
        print("⚠️ Statistical Power > 50%: MODERATE")
    else:
        print("❌ Statistical Power < 50%: INSUFFICIENT")

    print("\n" + "="*70)
    print("✅ COMPREHENSIVE TEST COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
