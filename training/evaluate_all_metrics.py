# -*- coding: utf-8 -*-
"""
BỘ KIỂM ĐỊNH ĐẦY ĐỦ — THANG ĐO CHUẨN AI + CHUẨN Y TẾ (evaluate_all_metrics.py)

Trả lời yêu cầu: mỗi mô hình PHẢI có số liệu thật cho các thang đo chuẩn:
  Phân loại   : Accuracy, Precision, Recall/Sensitivity, Specificity, F1,
                NPV, AUC-ROC (+ Wilson 95% CI mọi tỷ lệ, bootstrap CI cho AUC)
  Phân cụm    : Silhouette Score (gait 3 nhóm y/o/pd; face 2 lớp)
  Hồi quy     : NIHSS — KHÔNG có ground truth công khai chấm điểm item
                → KHÔNG bịa số; ghi rõ THIẾU, dùng Bland-Altman + weighted
                kappa SAU khi có 50 video NIHSS chuẩn (L-01A)

Nguồn số liệu 100% THẬT:
  1. FACE rules  : test_results/face_rules_eval_*.csv  (module chạy thật)
  2. FACE ML     : train MỚI Logistic+MLP 5 đặc trưng, GroupKFold(5) THEO BLOCK
                   trên face_signal_test CSV (chống leakage ảnh liên tiếp)
  3. GAIT v2     : tái lập LOSO 15 subject PhysioNet (import pipeline v2)
  4. SPEECH TORGO: module2_test JSON (per-session predictions thật)

Biểu đồ (matplotlib): ROC ×4, confusion matrix ×4, calibration face-ML,
forest plot Cohen's d, PCA+silhouette gait.

Output: test_results/metrics_pack_<ts>/ (json + csv + png)

Tham chiếu thang đo: BANG_THANG_DO_KIEM_DINH.md
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, silhouette_score)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, current_dir)

from train_gait_model_v2 import (load_subjects, train_fold,  # noqa: E402
                                 DEFAULT_GAIT_PATH, SEED)

TEST_DIR = os.path.join(parent_dir, 'test_results')

FEATS = ['mouth_ratio', 'eye_ratio', 'face_tilt', 'nasolabial_ratio',
         'forehead_ratio']

# Ngưỡng vận hành thực tế của từng module (đang chạy trong app)
TH_FACE_RULES = 30.0     # WARNING ≥ 30
TH_FACE_ML = 0.5         # xác suất sau sigmoid
TH_GAIT = None           # → dùng Youden tối ưu từ OOF (metadata v2: 0.10)
TH_SPEECH = 30.0         # threshold_alarm = 30 (app)


# ======================================================================
# HÀM METRIC CHUẨN
# ======================================================================
def wilson_ci(k, n, z=1.96):
    """Wilson score interval (Wilson 1927, JASA) — n nhỏ vẫn hợp lệ."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(p, 4), round(max(0.0, center - half), 4), \
        round(min(1.0, center + half), 4)


def bootstrap_auc_ci(y, p, n_iter=1000, seed=SEED):
    """Bootstrap percentile 95% CI cho AUC (Efron; Hanley–McNeil 1982 ý nghĩa)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def cohen_d_ci(a, b):
    """Cohen's d + 95% CI xấp xỉ (Cohen 1988; SE theo Hedges & Olkin 1985)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
    d = (a.mean() - b.mean()) / s if s > 1e-9 else 0.0
    se = np.sqrt((na + nb) / (na * nb) + d * d / (2 * (na + nb)))
    return round(float(d), 3), round(float(d - 1.96 * se), 3), \
        round(float(d + 1.96 * se), 3)


def classification_pack(y_true, prob, threshold, name):
    """Trọn bộ thang đo phân loại tại 1 ngưỡng + AUC không phụ thuộc ngưỡng."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_true, prob))

    pack = {
        'module': name,
        'n': int(len(y_true)),
        'n_positive': int(y_true.sum()),
        'n_negative': int((y_true == 0).sum()),
        'threshold': round(float(threshold), 4),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'accuracy': wilson_ci(tp + tn, len(y_true)),
        'precision_ppv': wilson_ci(tp, tp + fp),
        'recall_sensitivity': wilson_ci(tp, tp + fn),
        'specificity': wilson_ci(tn, tn + fp),
        'npv': wilson_ci(tn, tn + fn),
        'f1': round(float(f1_score(y_true, pred, zero_division=0)), 3),
        'auc_roc': round(auc, 3),
        'auc_ci95_bootstrap': bootstrap_auc_ci(y_true, prob),
        'youden_j': round(float(
            (tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1)) - 1), 3),
    }
    return pack, y_true, prob, pred


def fmt_pack(p):
    """1 dòng CSV/JSON gọn."""
    return {
        'module': p['module'], 'n': p['n'],
        'TP': p['TP'], 'FP': p['FP'], 'FN': p['FN'], 'TN': p['TN'],
        'accuracy': p['accuracy'][0],
        'accuracy_ci95': f"{p['accuracy'][1]}-{p['accuracy'][2]}",
        'precision': p['precision_ppv'][0],
        'precision_ci95': f"{p['precision_ppv'][1]}-{p['precision_ppv'][2]}",
        'recall_sens': p['recall_sensitivity'][0],
        'recall_ci95': f"{p['recall_sensitivity'][1]}-{p['recall_sensitivity'][2]}",
        'specificity': p['specificity'][0],
        'spec_ci95': f"{p['specificity'][1]}-{p['specificity'][2]}",
        'npv': p['npv'][0], 'npv_ci95': f"{p['npv'][1]}-{p['npv'][2]}",
        'f1': p['f1'], 'auc_roc': p['auc_roc'],
        'auc_ci95': f"{p['auc_ci95_bootstrap'][0]}-{p['auc_ci95_bootstrap'][1]}",
        'youden_j': p['youden_j'], 'threshold': p['threshold'],
    }


def save_cm(y_true, pred, name, out_dir):
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center', fontsize=14,
                color='white' if v > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['Dự đoán Âm tính', 'Dự đoán Dương tính'])
    ax.set_yticks([0, 1], ['Thực tế Âm tính', 'Thực tế Dương tính'])
    ax.set_title(f'Confusion Matrix — {name}')
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{name}_confusion_matrix.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_roc(y_true, prob, name, out_dir, auc=None):
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc_v = auc if auc is not None else roc_auc_score(y_true, prob)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_v:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Ngẫu nhiên (0.5)')
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(f'Đường ROC — {name}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    path = os.path.join(out_dir, f'{name}_roc.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ======================================================================
# 1) FACE — RULES (module chạy thật trên app)
# ======================================================================
def eval_face_rules(out_dir):
    files = sorted(glob.glob(os.path.join(
        TEST_DIR, 'face_rules_eval_*.csv')))
    df = pd.read_csv(files[-1])
    df = df[df['status'] != 'NO_FACE'].copy()
    y, score = df['label'].values, df['score'].values
    pack, y, prob, pred = classification_pack(
        y, score / 100.0, TH_FACE_RULES / 100.0, 'face_rules')
    pack['threshold'] = TH_FACE_RULES  # thang 0-100 cho dễ đọc
    pack['note'] = ('Loại NO_FACE trước khi tính (đã loại 38% ảnh chất lượng '
                    'dataset kém — L-01/M1-04); ngưỡng dương = 30 (WARNING)')
    roc_p = save_roc(y, prob, 'face_rules', out_dir)
    cm_p = save_cm(y, pred, 'face_rules', out_dir)
    return pack, roc_p, cm_p


# ======================================================================
# 2) FACE — ML 5 ĐẶC TRƯNG (TRAIN MỚI, GroupKFold THEO BLOCK)
# ======================================================================
def eval_face_ml(out_dir):
    files = sorted(glob.glob(os.path.join(
        TEST_DIR, 'face_signal_test_*.csv')))
    df = pd.read_csv(files[-1])
    df = df[df['valid'] == True].dropna(subset=FEATS).copy()  # noqa: E712
    X, y, g = df[FEATS].values, df['label'].values, df['block_id'].values

    def fit_mlp(Xtr, ytr):
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        m = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(32, 16), nn.ReLU(),
                          nn.Linear(16, 1)).to(dev)
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
        pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                          dtype=torch.float32, device=dev)
        lf = nn.BCEWithLogitsLoss(pos_weight=pw)
        op = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(300):
            op.zero_grad()
            loss = lf(m(Xt).squeeze(-1), yt)
            loss.backward()
            op.step()
        m.eval()
        return m, dev

    gkf = GroupKFold(n_splits=5)
    p_lr = np.zeros(len(y))
    p_mlp = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        lr.fit(sc.transform(X[tr]), y[tr])
        p_lr[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        m, dev = fit_mlp(X[tr], y[tr])
        import torch
        with torch.no_grad():
            p_mlp[te] = torch.sigmoid(m(torch.tensor(
                X[te], dtype=torch.float32, device=dev)).squeeze(-1)).cpu().numpy()

    packs = {}
    plots = []
    for nm, p, th in [('face_ml_logreg', p_lr, TH_FACE_ML),
                      ('face_ml_mlp', p_mlp, TH_FACE_ML)]:
        pack, y2, pr, pd_ = classification_pack(y, p, th, nm)
        packs[nm] = pack
        plots.append(save_roc(y2, pr, nm, out_dir, auc=pack['auc_roc']))
        plots.append(save_cm(y2, pd_, nm, out_dir))

    # Calibration curve (Hosmer–Lemeshow ý nghĩa; Steyerberg 2009)
    frac_pos, mean_pred = calibration_curve(y, p_lr, n_bins=8, strategy='quantile')
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot(mean_pred, frac_pos, 'o-', label='Logistic (out-of-fold)')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Hoàn hảo')
    ax.set_xlabel('Xác suất dự đoán')
    ax.set_ylabel('Tỷ lệ dương thực tế')
    ax.set_title('Calibration — face ML Logistic')
    ax.legend()
    fig.tight_layout()
    cal_p = os.path.join(out_dir, 'face_ml_calibration.png')
    fig.savefig(cal_p, dpi=130)
    plt.close(fig)
    plots.append(cal_p)

    # Forest plot Cohen's d ± CI (5 đặc trưng)
    ds = [cohen_d_ci(df[df.label == 1][f], df[df.label == 0][f])
          for f in FEATS]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ypos = np.arange(len(FEATS))[::-1]
    for yy, (d, lo, hi) in zip(ypos, ds):
        ax.plot([lo, hi], [yy, yy], lw=2)
        ax.plot(d, yy, 'o')
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_yticks(ypos, FEATS)
    ax.set_xlabel("Cohen's d (stroke − non-stroke) ± 95% CI")
    ax.set_title('Effect size 5 đặc trưng bất đối xứng mặt')
    fig.tight_layout()
    forest_p = os.path.join(out_dir, 'face_forest_effects.png')
    fig.savefig(forest_p, dpi=130)
    plt.close(fig)
    plots.append(forest_p)

    packs['face_ml_silhouette'] = round(float(
        silhouette_score(StandardScaler().fit_transform(X), y)), 3)

    # Lưu artifact model Logistic (deploy được: coef + scaler params)
    sc_all = StandardScaler().fit(X)
    lr_all = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_all.fit(sc_all.transform(X), y)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    art = {
        'version': 'face_asym_v2_5feat',
        'created': ts,
        'validation': 'GroupKFold(5) theo block_id — không leakage ảnh liên tiếp',
        'features': FEATS,
        'scaler_mean': sc_all.mean_.tolist(),
        'scaler_scale': sc_all.scale_.tolist(),
        'coef': lr_all.coef_[0].tolist(),
        'intercept': float(lr_all.intercept_[0]),
        'threshold_warning': 0.5,
        'auc_oof': packs['face_ml_logreg']['auc_roc'],
        'note': 'M1-07: chỉ nối vào app SAU khi qua protocol B (SYS-15); '
                'nghệ thuật: 5 đặc trưng bất đối xứng, class-weight balanced',
    }
    art_path = os.path.join(parent_dir, 'models',
                            f'face_asym_v2_5feat_{ts}.json')
    with open(art_path, 'w', encoding='utf-8') as f:
        json.dump(art, f, ensure_ascii=False, indent=2)
    packs['model_artifact'] = art_path
    return packs, plots


# ======================================================================
# 3) GAIT v2 — LOSO 15 SUBJECT (tái lập pipeline đã train)
# ======================================================================
def eval_gait_loso(out_dir):
    from detection.gait_module import GaitAbnormalityDetector
    det = GaitAbnormalityDetector()
    subjects = load_subjects(DEFAULT_GAIT_PATH, det)

    all_prob, all_y, all_sid = [], [], []
    for i, sub in enumerate(subjects):
        X_tr = np.vstack([s['features'] for j, s in enumerate(subjects)
                          if j != i])
        y_tr = np.concatenate([np.full(len(s['features']), s['label'])
                               for j, s in enumerate(subjects) if j != i])
        prob = train_fold(X_tr, y_tr, sub['features'], 'cpu')
        all_prob.extend(prob.tolist())
        all_y.extend([sub['label']] * len(prob))
        all_sid.extend([sub['id']] * len(prob))

    y = np.array(all_y)
    p = np.array(all_prob)
    df = pd.DataFrame({'sid': all_sid, 'y': y, 'p': p})

    # Chọn ngưỡng Youden tối ưu TỪ OOF (trung thực — v2 metadata: 0.10)
    thresholds = np.linspace(0.01, 0.99, 99)
    js = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        js.append(tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1)
    th = float(thresholds[int(np.argmax(js))])

    pack, y2, pr, pred = classification_pack(y, p, th, 'gait_loso_v2')
    pack['threshold'] = round(th, 3)
    pack['note'] = ('LOSO subject-level (không leakage); ngưỡng Youden chọn '
                    'trên OOF; Parkinson = proxy bất thường dáng đi, KHÔNG '
                    'phải đột quỵ (hạn chế đã ghi metadata)')
    plots = [save_roc(y2, pr, 'gait_loso_v2', out_dir),
             save_cm(y2, pred, 'gait_loso_v2', out_dir)]

    # Subject-level (15 điểm): median prob mỗi subject
    sub_med = df.groupby('sid').agg(y=('y', 'first'), p=('p', 'median'))
    spack, sy, spr, spred = classification_pack(
        sub_med['y'].values, sub_med['p'].values, th, 'gait_loso_subject15')
    packs = {'gait_loso_v2': pack, 'gait_subject15': spack}

    # ----- Silhouette + PCA 3 nhóm (y/o/pd) — thang đo phân cụm -----
    X_all = np.vstack([s['features'] for s in subjects])
    groups = []
    for s in subjects:
        groups.extend(['pd'] * len(s['features']) if s['id'].startswith('pd')
                      else ['y'] * len(s['features']) if s['id'].startswith('y')
                      else ['o'] * len(s['features']))
    groups = np.array(groups)
    Xs = StandardScaler().fit_transform(X_all)
    sil_2 = round(float(silhouette_score(Xs, (groups == 'pd').astype(int))), 3)
    sil_3 = round(float(silhouette_score(Xs, groups)), 3)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colors = {'y': '#1a7a3a', 'o': '#b8860b', 'pd': '#c81e1e'}
    for gname in ['y', 'o', 'pd']:
        m = groups == gname
        ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.6, c=colors[gname],
                   label={'y': 'Trẻ khỏe (y1-y5)', 'o': 'Già khỏe (o1-o5)',
                          'pd': 'Parkinson (pd1-pd5)'}[gname])
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
    ax.set_title(f'Gait features — PCA + phân cụm\n'
                 f'Silhouette: 2 lớp={sil_2}, 3 nhóm={sil_3}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    pca_p = os.path.join(out_dir, 'gait_silhouette_pca.png')
    fig.savefig(pca_p, dpi=130)
    plt.close(fig)
    plots.append(pca_p)

    packs['gait_silhouette_2class'] = sil_2
    packs['gait_silhouette_3group'] = sil_3
    return packs, plots


# ======================================================================
# 4) SPEECH — TORGO (per-session predictions thật)
# ======================================================================
def eval_speech(out_dir):
    files = sorted(glob.glob(os.path.join(TEST_DIR, 'module2_test_*.json')))
    d = json.load(open(files[-1], encoding='utf-8'))
    y, p = [], []
    for r in d['results']:
        y.append(0 if 'NORMAL' in r['true'].upper() else 1)
        p.append(float(r['speech_prob']) / 100.0)
    y, p = np.array(y), np.array(p)
    pack, y2, pr, pred = classification_pack(
        y, p, TH_SPEECH / 100.0, 'speech_torgo')
    pack['threshold'] = TH_SPEECH
    pack['note'] = ('TORGO (tiếng Anh) per-session, median-3-cửa-sổ M2-10; '
                    'HẠN CHẾ: chưa có tiếng Việt — L-02')
    plots = [save_roc(y2, pr, 'speech_torgo', out_dir),
             save_cm(y2, pred, 'speech_torgo', out_dir)]
    return pack, plots


# ======================================================================
# MAIN
# ======================================================================
def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(TEST_DIR, f'metrics_pack_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output: {out_dir}\n')

    all_rows = []
    gallery = {'roc': [], 'cm': [], 'other': []}

    print('===== 1/4 FACE rules =====')
    pk, roc, cm = eval_face_rules(out_dir)
    all_rows.append(fmt_pack(pk)); gallery['roc'].append(roc)
    gallery['cm'].append(cm)

    print('===== 2/4 FACE ML (train mới M1-07) =====')
    face_pks, plots = eval_face_ml(out_dir)
    for nm in ('face_ml_logreg', 'face_ml_mlp'):
        all_rows.append(fmt_pack(face_pks[nm]))
    gallery['roc'] += plots[:4]
    gallery['other'] += plots[4:]

    print('===== 3/4 GAIT LOSO v2 =====')
    pks, plots = eval_gait_loso(out_dir)
    all_rows.append(fmt_pack(pks['gait_loso_v2']))
    all_rows.append(fmt_pack(pks['gait_subject15']))
    gallery['roc'].append(plots[0]); gallery['cm'].append(plots[1])
    gallery['other'].append(plots[2])

    print('===== 4/4 SPEECH TORGO =====')
    pk, plots = eval_speech(out_dir)
    all_rows.append(fmt_pack(pk))
    gallery['roc'] += plots[:1]; gallery['cm'] += plots[1:]

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, 'bang_so_lieu.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    report = {
        'purpose': 'Bộ số liệu kiểm định chuẩn AI + y tế (evaluate_all_metrics)',
        'created': datetime.now().isoformat(),
        'metrics_definitions': 'BANG_THANG_DO_KIEM_DINH.md',
        'results': all_rows,
        'silhouette': {
            'gait_2class': pks.get('gait_silhouette_2class'),
            'gait_3group': pks.get('gait_silhouette_3group'),
            'face_ml_logreg': face_pks.get('face_ml_silhouette'),
        },
        'model_artifacts': {
            'face_logreg': face_pks.get('model_artifact'),
            'gait_v2': 'models/gait_classifier_v2_20260907_210433.pth',
        },
        'missing_data_audit': {
            'arm': 'THIẾU — model train trên synthetic data, chưa có eval trên '
                   'dữ liệu thật (cần video thật tay giơ/đóng mở — A-01)',
            'nihss_regression': 'THIẾU — không có NIHSS gold-standard công '
                                'khai; MAE/RMSE/R²/Bland-Altman sẽ tính sau '
                                'L-01A (50 video NIHSS chuẩn)',
            'radar': 'THIẾU — cần test radar thật 08/09 (M5-01)',
            'speech_vn': 'THIẾU — TORGO tiếng Anh; cần thu tiếng Việt (L-02)',
        },
        'plots': gallery,
    }
    jp = os.path.join(out_dir, 'metrics_summary.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('\n===== BẢNG SỐ LIỆU =====')
    print(df.to_string(index=False))
    print(f"\nCSV : {csv_path}\nJSON: {jp}")
    print(f"Model Logistic mặt đã lưu: {face_pks.get('model_artifact')}")


if __name__ == '__main__':
    main()
