# -*- coding: utf-8 -*-
"""
UNCERTAINTY + ABSTENTION — PSCS v9.0 (trụ cột A, NK-10)
=======================================================
"Biết mình không biết": mỗi điểm số đi kèm ĐỘ BẤT ĐỊNH; khi bất định quá
ngưỡng → hệ KHÔNG tự đoán quyết mà đánh dấu NEEDS_CHECK (người xác nhận).

Nguyên tắc an toàn (giữ đúng triết lý Defense L-08):
  - Abstention KHÔNG BAO GIỜ hạ cấp EMERGENCY — chỉ bổ sung needs_check
    ở vùng bất định. Báo động thật không bao giờ bị "nuốt".
  - Mọi hàm thuần numpy, không phụ thuộc sklearn/torch → an toàn runtime.

Nội dung:
  1. Platt calibration (Newton 1D) + phiên bản cross-validated (trung thực)
  2. Ensemble uncertainty: mean/std nhiều model hoặc nhiều seed
  3. Margin: khoảng cách tới ngưỡng quyết định
  4. Abstain mask: theo độ bất định (top-frac)
  5. Risk–coverage curve + selective summary (đánh giá)

Tác giả: PSCS Team
Ngày: 12/09/2026
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. PLATT CALIBRATION (logistic trên 1D score)
# ----------------------------------------------------------------------
def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_platt(scores, labels, ridge=1e-4, max_iter=200, tol=1e-10):
    """
    Fit Platt scaling: prob = sigmoid(a * score + b) bằng Newton 1D
    (log-loss + L2 nhẹ lên `a` để ổn định khi fold nhỏ).

    scores: raw prob/score (0..1 hoặc bất kỳ) — shape (n,)
    labels: 0/1 — shape (n,)
    Returns: (a, b)
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if len(s) < 3 or len(np.unique(y)) < 2:
        return 1.0, 0.0   # không đủ dữ liệu → identity

    a, b = 1.0, 0.0
    for _ in range(max_iter):
        z = a * s + b
        p = _sigmoid(z)
        # gradient & hessian của log-loss + ridge*a^2
        g_a = float(np.sum((p - y) * s) + ridge * a)
        g_b = float(np.sum(p - y))
        w = p * (1.0 - p)
        h_aa = float(np.sum(w * s * s)) + ridge
        h_ab = float(np.sum(w * s))
        h_bb = float(np.sum(w))
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 0 or not np.isfinite(det):
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a, b = a - da, b - db
        if abs(da) < tol and abs(db) < tol:
            break
    return float(a), float(b)


def apply_platt(scores, a, b):
    """prob đã calibrate = sigmoid(a*score + b)."""
    return _sigmoid(a * np.asarray(scores, dtype=np.float64) + b)


def cv_platt_oof(scores, labels, fold_ids):
    """
    Calibrate TRUNG THỰC: với mỗi fold → fit Platt trên các fold KHÁC,
    transform fold hiện tại (out-of-fold). Trả về mảng prob đã calibrate.
    fold_ids: mảng group id (người/block) cùng độ dài scores.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    folds = np.asarray(fold_ids)
    out = np.full(len(s), np.nan)
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        a, b = fit_platt(s[tr], y[tr])
        out[te] = apply_platt(s[te], a, b)
    out[np.isnan(out)] = apply_platt(s[np.isnan(out)], 1.0, 0.0)
    return out


# ----------------------------------------------------------------------
# 2-3. ĐỘ BẤT ĐỊNH
# ----------------------------------------------------------------------
def ensemble(probs_matrix):
    """
    probs_matrix: (n_samples, n_models_or_seeds) — prob của cùng mẫu từ
    nhiều model/seed. Returns: (mean, std) — mean là prob hệ, std là
    ĐỘ BẤT ĐỊNH ensemble (bất đồng giữa các thành viên).
    """
    p = np.asarray(probs_matrix, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    mean = p.mean(axis=1)
    std = p.std(axis=1, ddof=1) if p.shape[1] > 1 else np.zeros(len(p))
    return mean, std


def margin(prob, threshold):
    """
    Khoảng cách tới ngưỡng quyết định, chuẩn hoá [0,1].
    CÀNG NHỎ CÀNG BẤT ĐỊNH (vùng xám quanh ngưỡng).
    """
    return np.abs(np.asarray(prob, dtype=np.float64) - float(threshold))


def combined_uncertainty(unc_std, marg, w_std=0.5):
    """
    Gộp 2 tín hiệu bất định về chung thang [0,1] (cao = bất định hơn):
      - unc_std: std ensemble (đã chia 0.25 để về ~[0,1] vì std tối đa 0.5)
      - marg: margin tới ngưỡng → đổi dấu (1 − margin/0.5)
    w_std = trọng số của std ensemble.
    """
    u1 = np.clip(np.asarray(unc_std, dtype=np.float64) / 0.5, 0, 1)
    u2 = np.clip(1.0 - np.asarray(marg, dtype=np.float64) / 0.5, 0, 1)
    return float(w_std) * u1 + (1.0 - w_std) * u2


# ----------------------------------------------------------------------
# 4. ABSTAIN MASK
# ----------------------------------------------------------------------
def abstain_topfrac(uncertainty, frac):
    """
    Chọn frac (0..1) mẫu BẤT ĐỊNH NHẤT theo uncertainty.
    Returns: mask True = nên abstain (đưa người xác nhận).
    """
    u = np.asarray(uncertainty, dtype=np.float64)
    n_abstain = int(round(frac * len(u)))
    if n_abstain <= 0:
        return np.zeros(len(u), dtype=bool)
    order = np.argsort(-u, kind='stable')   # bất định nhất trước
    mask = np.zeros(len(u), dtype=bool)
    mask[order[:n_abstain]] = True
    return mask


# ----------------------------------------------------------------------
# 5. ĐÁNH GIÁ: RISK–COVERAGE + SELECTIVE SUMMARY
# ----------------------------------------------------------------------
def _binary_counts(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp


def selective_point(y, prob, mask_keep, threshold):
    """Metrics trên phần ĐƯỢC GIỮ (mask_keep=True) tại ngưỡng threshold."""
    yy, pp = np.asarray(y), np.asarray(prob)
    if mask_keep.sum() == 0:
        return {'coverage': 0.0, 'n': 0}
    yk, pk = yy[mask_keep], pp[mask_keep]
    pred = (pk >= threshold).astype(int)
    tp, fn, tn, fp = _binary_counts(yk, pred)
    return {
        'coverage': round(float(mask_keep.mean()), 4),
        'n': int(mask_keep.sum()),
        'sens': round(100 * tp / max(tp + fn, 1), 1),
        'spec': round(100 * tn / max(tn + fp, 1), 1),
        'fpr': round(100 * fp / max(fp + tn, 1), 1),
        'err_rate': round(100 * (fp + fn) / max(len(yk), 1), 2),
    }


def risk_coverage(y, prob, uncertainty, threshold, n_points=50):
    """
    Đường risk–coverage: từ coverage 100% (giữ tất cả) xuống ~50%
    (abstain nửa bất định nhất). Trả về dict các mảng để vẽ figure.
    """
    y = np.asarray(y)
    u = np.asarray(uncertainty, dtype=np.float64)
    covs, errs, sens, fprs = [], [], [], []
    for frac in np.linspace(0.0, 0.5, n_points):
        m_abstain = abstain_topfrac(u, frac)
        pt = selective_point(y, prob, ~m_abstain, threshold)
        covs.append(pt['coverage'])
        errs.append(pt['err_rate'])
        sens.append(pt.get('sens'))
        fprs.append(pt.get('fpr'))
    return {'coverage': covs, 'err_rate': errs, 'sens': sens, 'fpr': fprs}


def selective_summary(y, prob, uncertainty, threshold,
                      abstain_fracs=(0.1, 0.2, 0.3)):
    """
    Tổng kết selective prediction cho 1 module:
      - base: metrics giữ 100%
      - từng mức abstain: metrics phần giữ + lỗi trong phần abstain
      - unc_auc: AUC của (uncertainty → dự đoán sai) — "bất định có biết lỗi?"
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    p = np.asarray(prob, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    err = ((p >= threshold).astype(int) != y).astype(int)

    base = selective_point(y, p, np.ones(len(y), dtype=bool), threshold)

    steps = {}
    for frac in abstain_fracs:
        m_ab = abstain_topfrac(u, frac)
        keep = selective_point(y, p, ~m_ab, threshold)
        # trong phần bị abstain: bao nhiêu % thực sự là lỗi?
        ab_err = float(err[m_ab].mean()) * 100 if m_ab.sum() else 0.0
        steps[f'abstain_{int(frac*100)}'] = {
            'kept': keep,
            'err_rate_among_abstained': round(ab_err, 2),
            'err_reduction_kept': round(base['err_rate'] - keep['err_rate'], 2),
        }

    unc_auc = None
    if 0 < err.sum() < len(err):
        unc_auc = round(float(roc_auc_score(err, u)), 3)

    return {'threshold': round(float(threshold), 3), 'base': base,
            'steps': steps, 'unc_error_auc': unc_auc, 'n': len(y)}
