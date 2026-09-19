# -*- coding: utf-8 -*-
"""
NK-24 — KIỂM ĐỊNH THỐNG KÊ BỔ SUNG (McNemar · DeLong · Calibration · PR)
Điền các thang đo còn ghi "Kế hoạch/THIẾU" trong BANG_THANG_DO_KIEM_DINH.md:

  §4 McNemar       — so 2 model CÙNG fold, CÙNG từng mẫu (face: LogReg 5 ratio
                     [bộ cũ, đại diện face_asym_v2] vs face v3 28 ft [chính thức])
                     — test chính xác binomial trên cặp đúng/sai.
  §1 DeLong paired — so AUC 2 model trên CÙNG mẫu có covariance (DeLong 1988):
                     (a) face 5-ft vs 28-ft OOF; (b) speech LOSO LogReg vs MLP
                     (dùng đúng oof_predictions.csv của NK-03 — KHÔNG chạy lại).
  §4 Calibration   — đường reliability + Brier + ECE cho face v3: raw vs
                     Platt (nested — fit trong fold, KHÔNG nhìn test) — bổ sung
                     cho NK-10 (Platt) bằng hình chuẩn khoa học.
  PR-AUC           — dữ liệu lệch lớp (van Rijsbergen 1979).

NGUYÊN TẮC: mọi số đo trên DỮ LIỆU THẬT có sẵn; KHÔNG đụng models/;
face dùng đúng cache _face_v3_features_cache.npz (3,715 ảnh × 28 ft) và
CÙNG OOF fold GroupKFold(5) theo block cho 2 model (so sánh công bằng —
mỗi ảnh được dự đoán đúng 1 lần bởi model chưa thấy block của nó); seed 42.

Chạy: PYTHONUTF8=1 python training/eval_statistical_tests.py
Output: test_results/stat_tests_<ts>/ (summary.json + 3 PNG)
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_CACHE = os.path.join(ROOT, 'test_results', '_face_v3_features_cache.npz')
SPEECH_OOF = os.path.join(
    ROOT, 'test_results', 'speech_speaker_loso_20260910_214834',
    'oof_predictions.csv')
SEED = 42


# ----------------------------------------------------------------------
# DeLong (1988) — bản paired nhanh (Sun & Xu 2014), vectorized numpy
# ----------------------------------------------------------------------
def _midrank(x):
    """Midrank (xử lý tie đúng chuẩn)."""
    J = np.argsort(x, kind='mergesort')
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _delong_components(y, probs_all):
    """probs_all: (k_model, n) — y: (n,) 0/1. Trả aucs (k,), v01, v10."""
    k, n = probs_all.shape
    m = int(y.sum())            # số dương
    nn = n - m                  # số âm
    # sắp: dương trước, âm sau (thứ tự GIỐNG NHAU cho mọi model → paired)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    order = np.concatenate([pos_idx, neg_idx])
    P = probs_all[:, order[:m]]          # (k, m)
    Nn = probs_all[:, order[m:]]         # (k, nn)
    tx = np.empty((k, m))
    ty = np.empty((k, nn))
    tz = np.empty((k, n))
    probs_sorted = probs_all[:, order]
    for r in range(k):
        tx[r] = _midrank(P[r])
        ty[r] = _midrank(Nn[r])
        tz[r] = _midrank(probs_sorted[r])
    aucs = (tz[:, :m].sum(axis=1) / (m * nn)
            - m * (m + 1) / (2.0 * m * nn))
    v01 = (tz[:, :m] - tx) / nn          # (k, m)
    v10 = 1.0 - (tz[:, m:] - ty) / m     # (k, nn)
    return aucs, v01, v10, m, nn


def delong_paired(y, p1, p2):
    """So sánh AUC paired 2 model (cùng mẫu). Trả dict đầy đủ."""
    from scipy.stats import norm
    P = np.vstack([p1, p2])
    aucs, v01, v10, m, nn = _delong_components(y, P)
    # DeLong chuẩn (m ≠ n được): cấu trúc = Cov phần dương + Cov phần âm
    # v01: (k, m) — mỗi mẫu DƯƠNG 1 quan sát k-chiều; v10: (k, nn) — mẫu ÂM
    l = np.array([1.0, -1.0])
    sx = np.cov(v01)                       # (2,2) trên m quan sát
    sy = np.cov(v10)                       # (2,2) trên nn quan sát
    var_delta = (float(l @ sx @ l) / m + float(l @ sy @ l) / nn)
    delta = float(aucs[0] - aucs[1])
    if var_delta > 0:
        z = abs(delta) / np.sqrt(var_delta)
        p = float(2.0 * (1.0 - norm.cdf(z)))
    else:
        z, p = 0.0, 1.0
    lo = delta - 1.959964 * np.sqrt(max(var_delta, 0.0))
    hi = delta + 1.959964 * np.sqrt(max(var_delta, 0.0))
    return {'auc_1': round(float(aucs[0]), 4), 'auc_2': round(float(aucs[1]), 4),
            'delta_auc': round(delta, 4),
            'ci95_delta': [round(lo, 4), round(hi, 4)],
            'var_delta': var_delta, 'z': round(float(z), 3),
            'p_value': float(f'{p:.3e}'), 'n_pos': m, 'n_neg': nn}


# ----------------------------------------------------------------------
# McNemar chính xác (binomial) trên cặp đúng/sai
# ----------------------------------------------------------------------
def mcnemar_exact(y, pred1, pred2):
    """Bảng 2×2: cả2đúng / chỉ1đúng / chỉ2đúng / cả2sai → test chính xác."""
    from scipy.stats import binomtest
    c1 = (pred1 == y)
    c2 = (pred2 == y)
    n01 = int((c1 & ~c2).sum())   # model1 đúng, model2 sai
    n10 = int((~c1 & c2).sum())   # model1 sai, model2 đúng
    b, c = n01, n10
    res = binomtest(b, b + c, 0.5) if (b + c) > 0 else None
    p = res.pvalue if res else 1.0
    return {'n01_model1_right_model2_wrong': b,
            'n10_model1_wrong_model2_right': c,
            'p_exact': float(f'{p:.3e}'),
            'acc_1': round(float(c1.mean()), 4),
            'acc_2': round(float(c2.mean()), 4), 'n': int(len(y))}


# ----------------------------------------------------------------------
# Youden threshold
# ----------------------------------------------------------------------
def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i])


def ece_bins(y, p, n_bins=10):
    """Expected Calibration Error — 10 bin đều theo xác suất."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        ece += m.mean() * abs(acc - conf)
        rows.append({'bin': b, 'n': int(m.sum()), 'conf': round(conf, 4),
                     'acc': round(acc, 4)})
    return round(float(ece), 4), rows


# ----------------------------------------------------------------------
# FACE: 2 model, CÙNG OOF fold (GroupKFold 5 theo block)
# ----------------------------------------------------------------------
def face_oof_two_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    d = np.load(FACE_CACHE, allow_pickle=True)
    X, y, blocks = d['X'], d['y'].astype(int), d['blocks']
    X5 = X[:, :5]        # 5 ratio lâm sàng (bộ cũ — face_asym_v2)
    X28 = X              # 28 ft v3 (5 ratio + 20 asym + 3 pose)

    Cs = (0.03, 0.1, 0.3, 1.0, 3.0)

    def oof_one(Xin):
        oof = np.zeros(len(y))
        gkf = GroupKFold(5)
        for ti, vi in gkf.split(Xin, y, blocks):
            # chọn C bằng OOF nội bộ (giống protocol v3) — công bằng cho
            # cả 2 model, mỗi model C riêng
            best = None
            for C in Cs:
                o = np.zeros(len(ti))
                for t2, v2 in GroupKFold(5).split(Xin[ti], y[ti], blocks[ti]):
                    sc = StandardScaler().fit(Xin[ti][t2])
                    clf = LogisticRegression(C=C, max_iter=2000,
                                             random_state=SEED)
                    clf.fit(sc.transform(Xin[ti][t2]), y[ti][t2])
                    o[v2] = clf.predict_proba(sc.transform(Xin[ti][v2]))[:, 1]
                auc = roc_auc_score(y[ti], o)
                if best is None or auc > best[1]:
                    best = (C, auc)
            C = best[0]
            sc = StandardScaler().fit(Xin[ti])
            clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
            clf.fit(sc.transform(Xin[ti]), y[ti])
            oof[vi] = clf.predict_proba(sc.transform(Xin[vi]))[:, 1]
        return oof

    print('  FACE OOF model cũ (5 ratio)…')
    p5 = oof_one(X5)
    print('  FACE OOF model v3 (28 ft)…')
    p28 = oof_one(X28)
    return y, p5, p28


def face_platt_nested(y, p28, blocks):
    """Platt calibrate HONEST: fit trong fold (inner OOF), apply ngoài fold.
    Đầu vào p28 là OOF outer → dùng chính fold outer để fit/apply tách biệt:
    mỗi outer fold: fit (a,b) trên ToÀN bộ phần train (các fold khác),
    apply lên fold test. Không rò rỉ vì Platt chỉ dùng nhãn phần train."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from fusion.uncertainty import fit_platt, apply_platt
    from sklearn.model_selection import GroupKFold
    out = np.zeros(len(p28))
    for ti, vi in GroupKFold(5).split(p28, y, blocks):
        a, b = fit_platt(p28[ti], y[ti])
        out[vi] = apply_platt(p28[vi], a, b)
    return out, [float('nan')]  # tham số ghép không cần — ghi per-fold riêng


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('== NK-24 KIỂM ĐỊNH THỐNG KÊ BỔ SUNG (McNemar · DeLong · Calib · PR) ==')
    t0 = time.time()
    out_dir = os.path.join(
        ROOT, 'test_results',
        f'stat_tests_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    summary = {'protocol': 'NK-24 stat tests: McNemar/DeLong/Calibration/PR — '
                           'OOF cùng fold, seed 42, KHÔNG đụng models/'}

    # ---------------- FACE ----------------
    print('FACE — 2 model cùng OOF fold…')
    y_f, p5, p28 = face_oof_two_models()
    t5 = youden_threshold(y_f, p5)
    t28 = youden_threshold(y_f, p28)
    mcn = mcnemar_exact(y_f, (p5 >= t5).astype(int),
                        (p28 >= t28).astype(int))
    dl_f = delong_paired(y_f, p28, p5)   # hướng: v3 − cũ
    print(f"  McNemar: {mcn['n01_model1_right_model2_wrong']} vs "
          f"{mcn['n10_model1_wrong_model2_right']} p={mcn['p_exact']:.2e}")
    print(f"  DeLong face: AUC v3 {dl_f['auc_1']} vs cũ {dl_f['auc_2']} "
          f"Δ={dl_f['delta_auc']} p={dl_f['p_value']:.2e}")

    # Calibration face v3: raw vs Platt nested
    from sklearn.model_selection import GroupKFold
    blocks = np.load(FACE_CACHE, allow_pickle=True)['blocks']
    p_platt, _ = face_platt_nested(y_f, p28, blocks)
    from sklearn.metrics import brier_score_loss
    brier_raw = round(float(brier_score_loss(y_f, p28)), 4)
    brier_pl = round(float(brier_score_loss(y_f, p_platt)), 4)
    ece_raw, bins_raw = ece_bins(y_f, p28)
    ece_pl, bins_pl = ece_bins(y_f, p_platt)
    print(f'  Brier raw {brier_raw} → Platt {brier_pl} · ECE {ece_raw} → {ece_pl}')

    from sklearn.metrics import average_precision_score, roc_curve
    ap28 = round(float(average_precision_score(y_f, p28)), 4)
    base = round(float(y_f.mean()), 4)
    print(f'  PR-AUC v3 {ap28} (baseline dương {base})')

    summary['face'] = {
        'protocol': 'OOF GroupKFold(5) theo block, 3,715 ảnh, C chọn inner-OOF',
        'mcnemar_5ft_vs_28ft': mcn,
        'delong_v3_vs_5ft': dl_f,
        'thresholds_youden': {'5ft': round(t5, 4), '28ft': round(t28, 4)},
        'brier_raw': brier_raw, 'brier_platt': brier_pl,
        'ece_raw': ece_raw, 'ece_platt': ece_pl,
        'bins_raw': bins_raw, 'bins_platt': bins_pl,
        'pr_auc_v3': ap28, 'pr_baseline': base}

    # ---------------- SPEECH (dùng đúng OOF NK-03) ----------------
    print('SPEECH — DeLong LogReg vs MLP (OOF LOSO NK-03)…')
    import csv
    rows = []
    with open(SPEECH_OOF, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['label']), float(r['p_logreg']),
                         float(r['p_mlp'])))
    arr = np.array(rows)
    y_s, plr, pmlp = arr[:, 0].astype(int), arr[:, 1], arr[:, 2]
    dl_s = delong_paired(y_s, plr, pmlp)
    t_lr, t_ml = youden_threshold(y_s, plr), youden_threshold(y_s, pmlp)
    mcn_s = mcnemar_exact(y_s, (plr >= t_lr).astype(int),
                          (pmlp >= t_ml).astype(int))
    ap_s = round(float(average_precision_score(y_s, plr)), 4)
    print(f"  DeLong speech: LR {dl_s['auc_1']} vs MLP {dl_s['auc_2']} "
          f"Δ={dl_s['delta_auc']} p={dl_s['p_value']:.2e}")
    summary['speech'] = {'protocol': 'dùng đúng oof_predictions.csv NK-03 '
                                     '(1,100 file LOSO người thật)',
                         'delong_logreg_vs_mlp': dl_s,
                         'mcnemar_logreg_vs_mlp': mcn_s,
                         'pr_auc_logreg': ap_s,
                         'pr_baseline': round(float(y_s.mean()), 4)}

    # ---------------- FIGURES ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1) Reliability face v3 (raw vs Platt) + histogram
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=170)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Hoàn hảo')
    fpr, tpr, _ = roc_curve(y_f, p28)
    ax.plot([b['conf'] for b in bins_raw], [b['acc'] for b in bins_raw],
            'o-', color='#c0392b', lw=1.8,
            label=f'Raw (ECE {ece_raw:.3f}, Brier {brier_raw:.3f})')
    ax.plot([b['conf'] for b in bins_pl], [b['acc'] for b in bins_pl],
            's-', color='#1a6faf', lw=1.8,
            label=f'Platt (ECE {ece_pl:.3f}, Brier {brier_pl:.3f})')
    ax.set_xlabel('Xác suất dự đoán (bin trung bình)')
    ax.set_ylabel('Tỷ lệ dương thật trong bin')
    ax.set_title('Face v3 — đường reliability (OOF 3,715 ảnh)')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.hist(p28, bins=40, alpha=0.55, color='#c0392b', label='Raw')
    ax.hist(np.clip(p_platt, 0, 1), bins=40, alpha=0.55, color='#1a6faf',
            label='Platt')
    ax.set_xlabel('Xác suất dự đoán'); ax.set_ylabel('Số ảnh')
    ax.set_title('Phân phối prob (cắt clip [0,1])')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'reliability_face_v3.png'))
    plt.close(fig)

    # 2) Paired ROC face (cũ vs v3) + chú thích McNemar/DeLong
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=170)
    for p, c, lb in ((p5, '#7f8c8d', f"5 ratio (cũ) AUC {dl_f['auc_2']}"),
                     (p28, '#1a6faf', f"v3 28 ft AUC {dl_f['auc_1']}")):
        fpr, tpr, _ = roc_curve(y_f, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=lb)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title('Face — 2 model CÙNG OOF fold (3,715 ảnh)\n'
                 f"DeLong ΔAUC {dl_f['delta_auc']:.3f} "
                 f"[{dl_f['ci95_delta'][0]:.3f},{dl_f['ci95_delta'][1]:.3f}] "
                 f"p={dl_f['p_value']:.1e} · "
                 f"McNemar p={mcn['p_exact']:.1e}")
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'paired_tests_face.png'))
    plt.close(fig)

    # 3) Paired ROC speech + PR cả 2 module
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=170)
    ax = axes[0]
    for p, c, lb in ((plr, '#1a6faf', f"LogReg AUC {dl_s['auc_1']}"),
                     (pmlp, '#e67e22', f"MLP AUC {dl_s['auc_2']}")):
        fpr, tpr, _ = roc_curve(y_s, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=lb)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title('Speech LOSO (1,100 file) — LogReg vs MLP\n'
                 f"DeLong ΔAUC {dl_s['delta_auc']:.3f} "
                 f"p={dl_s['p_value']:.2f} · McNemar p={mcn_s['p_exact']:.2f}")
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_f, p28)
    prec, _ = None, None
    from sklearn.metrics import precision_recall_curve
    pr, rc, _ = precision_recall_curve(y_f, p28)
    ax.plot(rc, pr, color='#1a6faf', lw=2,
            label=f'Face v3 (AP {ap28}, baseline {base})')
    pr2, rc2, _ = precision_recall_curve(y_s, plr)
    ax.plot(rc2, pr2, color='#27ae60', lw=2,
            label=f'Speech LogReg (AP {ap_s}, baseline '
                  f"{summary['speech']['pr_baseline']})")
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Đường Precision–Recall (dữ liệu lệch lớp)')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'paired_tests_speech_pr.png'))
    plt.close(fig)

    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(f'\nĐã lưu: {out_dir}')
    print(f'Thời gian: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
