# -*- coding: utf-8 -*-
"""
VALIDATION METRICS - PSCS v8.0 (Golden-Watch)
Bộ đo lường kiểm định cho nghiên cứu lâm sàng (Tuần 3-4 theo lịch).

Chức năng (KHÔNG phụ thuộc framework — chỉ numpy):
  - pearson_r, mae            : NIHSS ước tính vs bác sĩ (target r >= 0.85, MAE < 2)
  - confusion_matrix          : hệ thống vs chẩn đoán chuẩn
  - sensitivity/specificity/f1: Sens > 90%, Spec > 95%, F1
  - roc_points, find_youden   : đường cong ROC + ngưỡng Youden J
  - paired_sign_test          : p-value phi tham số (thay McNemar cho n nhỏ)

Cách dùng nghiên cứu (xem docs/kiem_dinh_y_khoa/QUY_TRINH_KIEM_DINH_Y_KHOA.md):
  1. Chạy hệ thống trên 50 video đột quỵ công khai → nihss_est.csv
  2. Bác sĩ chấm NIHSS + chẩn đoán cho từng video → gold.csv
  3. Feed 2 file vào evaluate_nihss_study() / evaluate_detection_study()

Tác giả: PSCS Team
Ngày: 06/09/2026
"""

import sys
import numpy as np


def pearson_r(x, y):
    """Hệ số tương quan Pearson (-1..1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mae(x, y):
    """Mean Absolute Error."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.mean(np.abs(x - y))) if len(x) else 0.0


def confusion_matrix(y_true, y_pred, positive=1):
    """Trả về dict TP/FP/TN/FN (binary, positive_label mặc định 1 = bệnh)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    fp = int(np.sum((y_true != positive) & (y_pred == positive)))
    tn = int(np.sum((y_true != positive) & (y_pred != positive)))
    fn = int(np.sum((y_true == positive) & (y_pred != positive)))
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


def sensitivity(cm):
    return cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) else 0.0


def specificity(cm):
    return cm['TN'] / (cm['TN'] + cm['FP']) if (cm['TN'] + cm['FP']) else 0.0


def f1_score(cm):
    prec = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) else 0.0
    rec = sensitivity(cm)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def roc_points(y_true, scores, thresholds=None):
    """Đường ROC: list (threshold, tpr, fpr)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, float)
    if thresholds is None:
        thresholds = np.arange(1, 100, 1.0)
    pts = []
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return pts
    for th in thresholds:
        pred = scores >= th
        tpr = float(np.sum(pos & pred)) / n_pos
        fpr = float(np.sum(neg & pred)) / n_neg
        pts.append((float(th), tpr, fpr))
    return pts


def find_youden(y_true, scores):
    """Ngưỡng Youden J = TPR - FPR tối đa → (threshold, j, tpr, fpr)."""
    best = (0.0, 0.0, 0.0, 0.0)  # th, j, tpr, fpr
    for th, tpr, fpr in roc_points(y_true, scores):
        j = tpr - fpr
        if j > best[1]:
            best = (th, j, tpr, fpr)
    return best


def paired_sign_test(y_true, pred_a, pred_b):
    """
    So sánh 2 hệ thống trên cùng dữ liệu (phi tham số, thay McNemar).
    Trả về p-value xấp xỉ 2 đuôi (binomial) cho số discordant pairs.
    """
    from math import comb
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    n01 = int(np.sum(~a_ok & b_ok))   # B thắng
    n10 = int(np.sum(a_ok & ~b_ok))   # A thắng
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    # p 2 đuôi = 2 * P(X <= k), X ~ Bin(n, 0.5)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(p, 1.0)


def evaluate_nihss_study(nihss_est, nihss_gold):
    """
    Nghiên cứu NIHSS ước tính vs bác sĩ.
    Targets (lịch Tuần 3): r >= 0.85, MAE < 2, p < 0.05.
    """
    r, err = pearson_r(nihss_est, nihss_gold), mae(nihss_est, nihss_gold)
    within1 = float(np.mean(np.abs(np.asarray(nihss_est, float)
                                   - np.asarray(nihss_gold, float)) <= 1))
    return {
        'n': len(nihss_est), 'pearson_r': round(r, 3), 'mae': round(err, 2),
        'within_1_point': round(within1 * 100, 1),
        'targets': {'r>=0.85': r >= 0.85, 'mae<2': err < 2},
        'verdict': 'PASS' if (r >= 0.85 and err < 2) else 'FAIL',
    }


def evaluate_detection_study(y_true, scores, threshold=None):
    """
    Nghiên cứu phát hiện (50 normal / 50 stroke).
    Targets: Sens > 90%, Spec > 95%, F1; nếu không có threshold → Youden.
    """
    if threshold is None:
        threshold, j, _, _ = find_youden(y_true, scores)
    pred = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    sens, spec = sensitivity(cm), specificity(cm)
    return {
        'threshold': round(float(threshold), 1), 'cm': cm,
        'sensitivity': round(sens * 100, 1),
        'specificity': round(spec * 100, 1),
        'f1': round(f1_score(cm), 3),
        'targets': {'sens>90': sens > 0.90, 'spec>95': spec > 0.95},
        'verdict': 'PASS' if (sens > 0.90 and spec > 0.95) else 'CHECK',
    }


# ======================================================================
# SELF-TEST (dữ liệu tổng hợp)
# ======================================================================
def _selftest():
    rng = np.random.default_rng(7)
    print("VALIDATION METRICS SELF-TEST")

    # NIHSS: ước tính = gold + noise nhỏ → r cao
    gold = rng.integers(0, 13, 50).astype(float)
    est = gold + rng.normal(0, 0.9, 50)
    res = evaluate_nihss_study(est, gold)
    assert res['pearson_r'] > 0.85 and res['mae'] < 2, res
    print(f"1. NIHSS study: r={res['pearson_r']}, MAE={res['mae']}, "
          f"within1={res['within_1_point']}% -> {res['verdict']}")

    # Detection: stroke score cao hơn normal rõ
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.uniform(5, 25, 50), rng.uniform(45, 95, 50)])
    det = evaluate_detection_study(y, scores)
    assert det['sensitivity'] > 90 and det['specificity'] > 95, det
    print(f"2. Detection study: th={det['threshold']} Sens={det['sensitivity']}% "
          f"Spec={det['specificity']}% F1={det['f1']} -> {det['verdict']}")

    # Confusion matrix tay
    cm = confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0])
    assert cm == {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1}, cm
    # Sign test: giống hệt nhau -> p=1
    p = paired_sign_test(y, (scores >= det['threshold']).astype(int),
                         (scores >= det['threshold']).astype(int))
    assert p == 1.0, p
    print(f"3. Confusion matrix + sign test (p={p}): OK")

    print("\nALL VALIDATION METRICS TESTS PASS")


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _selftest()
