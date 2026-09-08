# -*- coding: utf-8 -*-
"""
BỘ BIỂU ĐỒ KHOA HỌC ĐO ĐỘ CHÍNH XÁC — ĐỊNH LƯỢNG (plot_accuracy_charts.py)

Vẽ 5 biểu đồ chuẩn cho báo cáo/poster (100% số thật OOF/LOSO, seed 42):
  A. ROC gộp 4 module trên 1 hệ trục (so sánh trực quan)
  B. Precision–Recall curve (bắt buộc khi dữ liệu lệch lớp — gait 145/17)
  C. Bar chart nhóm: Accuracy/Precision/Sensitivity/Specificity/F1 theo module
     tại ngưỡng vận hành thật (face-rules 30, face-ML 0.5, gait Youden, speech 30)
  D. Sensitivity & Specificity THEO NGƯỠNG + điểm Youden tối ưu (face-ML, gait)
  E. Forest AUC ± 95% CI bootstrap (bản tóm tắt định lượng 1 hình)

Output: test_results/charts_accuracy_<ts>/ (5 PNG + chi_muc.json)
Chạy:   python training/plot_accuracy_charts.py
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
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, current_dir)

from train_gait_model_v2 import load_subjects, train_fold, \
    DEFAULT_GAIT_PATH, SEED  # noqa: E402

TEST_DIR = os.path.join(parent_dir, 'test_results')
FEATS = ['mouth_ratio', 'eye_ratio', 'face_tilt', 'nasolabial_ratio',
         'forehead_ratio']

COLORS = {'face_rules': '#7f8c8d', 'face_ml': '#1a5276',
          'gait': '#b9770e', 'speech': '#1e8449'}
LABELS = {'face_rules': 'Méo mặt — rules (app)',
          'face_ml': 'Méo mặt — ML 5 đặc trưng',
          'gait': 'Dáng đi — LOSO PhysioNet',
          'speech': 'Nói khó — TORGO'}
OPS_TH = {'face_rules': 0.30, 'face_ml': 0.50, 'gait': None, 'speech': 0.30}


# ---------------------------------------------------------------- data
def load_data():
    """Trả dict {key: (y, prob)} — toàn bộ out-of-fold/LOSO."""
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = {}

    fr = sorted(glob.glob(os.path.join(TEST_DIR, 'face_rules_eval_*.csv')))[-1]
    df = pd.read_csv(fr)
    df = df[df['status'] != 'NO_FACE']
    data['face_rules'] = (df['label'].values, df['score'].values / 100.0)

    fs = sorted(glob.glob(os.path.join(TEST_DIR, 'face_signal_test_*.csv')))[-1]
    df = pd.read_csv(fs)
    df = df[df['valid'] == True].dropna(subset=FEATS)  # noqa: E712
    X, y, g = df[FEATS].values, df['label'].values, df['block_id'].values
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        lr.fit(sc.transform(X[tr]), y[tr])
        p[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    data['face_ml'] = (y, p)

    from detection.gait_module import GaitAbnormalityDetector
    subjects = load_subjects(DEFAULT_GAIT_PATH, GaitAbnormalityDetector())
    ys, ps = [], []
    for i, sub in enumerate(subjects):
        Xtr = np.vstack([s['features'] for j, s in enumerate(subjects)
                         if j != i])
        ytr = np.concatenate([np.full(len(s['features']), s['label'])
                              for j, s in enumerate(subjects) if j != i])
        ps.extend(train_fold(Xtr, ytr, sub['features'], 'cpu'))
        ys.extend([sub['label']] * len(sub['features']))
    data['gait'] = (np.array(ys), np.array(ps))

    mj = sorted(glob.glob(os.path.join(TEST_DIR, 'module2_test_*.json')))[-1]
    d = json.load(open(mj, encoding='utf-8'))
    ys, ps = [], []
    for r in d['results']:
        ys.append(0 if 'NORMAL' in r['true'].upper() else 1)
        ps.append(float(r['speech_prob']) / 100.0)
    data['speech'] = (np.array(ys), np.array(ps))

    # Ngưỡng Youden cho gait (chọn trên OOF — như train v2)
    y, p = data['gait']
    ths = np.linspace(0.01, 0.99, 99)
    js = [confusion_matrix(y, (p >= t).astype(int), labels=[0, 1])
          for t in ths]
    jsc = [tp / (tp + fn) + tn / (tn + fp) - 1
           for tn, fp, fn, tp in [m.ravel() for m in js]]
    OPS_TH['gait'] = float(ths[int(np.argmax(jsc))])
    return data


def wilson(k, n, z=1.96):
    """Trả (điểm giữa, biên dưới, biên trên)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def boot_auc(y, p, n_iter=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_iter):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def metrics_at(y, p, th):
    tn, fp, fn, tp = confusion_matrix(y, (p >= th).astype(int),
                                      labels=[0, 1]).ravel()
    acc = wilson(tp + tn, len(y))          # (điểm, lo, hi)
    prc = wilson(tp, tp + fp)
    sen = wilson(tp, tp + fn)
    spe = wilson(tn, tn + fp)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {'Acc': acc, 'Prec': prc, 'Sens': sen, 'Spec': spe, 'F1': f1}


# ---------------------------------------------------------------- charts
def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    data = load_data()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(TEST_DIR, f'charts_accuracy_{ts}')
    os.makedirs(out, exist_ok=True)
    keys = ['face_rules', 'face_ml', 'gait', 'speech']

    # ---------- A. ROC GỘP ----------
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for k in keys:
        y, p = data[k]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, color=COLORS[k],
                label=f'{LABELS[k]} — AUC {roc_auc_score(y, p):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Ngẫu nhiên (0.5)')
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('A. Đường ROC so sánh 4 bộ phân loại\n'
                 '(đánh giá out-of-fold / LOSO — seed 42)')
    ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'A_roc_gop.png'), dpi=150)
    plt.close(fig)

    # ---------- B. PRECISION–RECALL ----------
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for k in keys:
        y, p = data[k]
        pr, rc, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rc, pr, lw=2, color=COLORS[k],
                label=f'{LABELS[k]} — AP {ap:.3f}')
        base = y.mean()
        ax.hlines(base, 0, 1, colors=COLORS[k], linestyles=':', lw=0.8,
                  alpha=0.5)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title('B. Đường Precision–Recall (nét chấm = tỷ lệ dương nền)')
    ax.legend(fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'B_precision_recall.png'), dpi=150)
    plt.close(fig)

    # ---------- C. BAR NHÓM METRICS ----------
    labels5 = ['Acc', 'Prec', 'Sens', 'Spec', 'F1']
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    width = 0.2
    for i, k in enumerate(keys):
        th = OPS_TH[k]
        m = metrics_at(*data[k], th)
        vals = [m[l][0] if isinstance(m[l], tuple) else m[l] for l in labels5]
        # thanh lỗi Wilson 2 phía: [center−lo, hi−center]
        err_lo = [max(0.0, v - m[l][1]) if isinstance(m[l], tuple) else 0
                  for l, v in zip(labels5, vals)]
        err_hi = [max(0.0, m[l][2] - v) if isinstance(m[l], tuple) else 0
                  for l, v in zip(labels5, vals)]
        xs = np.arange(len(labels5)) + (i - 1.5) * width
        ax.bar(xs, vals, width * 0.92, color=COLORS[k],
               yerr=[err_lo, err_hi], error_kw={'lw': 0.8}, label=LABELS[k])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.025, f'{v:.2f}', ha='center', fontsize=6.5)
    ax.set_xticks(np.arange(len(labels5)),
                  ['Accuracy', 'Precision', 'Sensitivity', 'Specificity',
                   'F1'])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Giá trị (0–1) · thanh lỗi = Wilson 95% CI')
    ax.set_title('C. Số liệu định lượng tại ngưỡng vận hành '
                 f"(gait Youden {OPS_TH['gait']:.2f}, còn lại ngưỡng app)")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'C_bar_metrics.png'), dpi=150)
    plt.close(fig)

    # ---------- D. SENS/SPEC THEO NGƯỠNG ----------
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, k in zip(axes, ['face_ml', 'gait']):
        y, p = data[k]
        ths = np.linspace(0.01, 0.99, 99)
        sens, spec = [], []
        for t in ths:
            tn, fp, fn, tp = confusion_matrix(
                y, (p >= t).astype(int), labels=[0, 1]).ravel()
            sens.append(tp / max(tp + fn, 1))
            spec.append(tn / max(tn + fp, 1))
        ax.plot(ths, sens, color='#c0392b', lw=2, label='Sensitivity')
        ax.plot(ths, spec, color='#2471a3', lw=2, label='Specificity')
        th = OPS_TH[k]
        ax.axvline(th, color='k', ls='--', lw=1)
        ax.text(th, 0.03, f' ngưỡng {th:.2f}', fontsize=8, rotation=90)
        ax.set_xlabel('Ngưỡng quyết định')
        ax.set_title(f"D. {LABELS[k].split(' — ')[1]}")
        ax.legend(fontsize=8, loc='center right')
    axes[0].set_ylabel('Giá trị')
    fig.suptitle('D. Đánh đổi Sensitivity/Specificity theo ngưỡng '
                 '(vạch đứt = ngưỡng đã chọn)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out, 'D_sens_spec_theo_ngoang.png'), dpi=150)
    plt.close(fig)

    # ---------- E. FOREST AUC ± CI ----------
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    yps = np.arange(len(keys))[::-1]
    idx = {k: i for i, k in enumerate(keys)}
    for k in keys:
        y, p = data[k]
        auc = roc_auc_score(y, p)
        lo, hi = boot_auc(y, p)
        yy = yps[idx[k]]
        ax.plot([lo, hi], [yy, yy], lw=2.4, color=COLORS[k])
        ax.plot(auc, yy, 'o', color=COLORS[k], ms=8)
        ax.text(hi + 0.012, yy, f'{auc:.3f} [{lo:.3f}–{hi:.3f}]',
                va='center', fontsize=8)
    ax.axvline(0.5, color='k', ls='--', lw=1)
    ax.text(0.5, len(keys) - 0.35, 'ngẫu nhiên', fontsize=7, ha='center')
    ax.set_yticks(yps, [LABELS[k].split(' — ')[0] for k in keys])
    ax.set_xlim(0.45, 1.12)
    ax.set_xlabel('AUC-ROC ± 95% CI (bootstrap 1000 lần, seed 42)')
    ax.set_title('E. Tóm tắt định lượng: độ chính xác phân biệt của từng module')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'E_forest_auc_ci.png'), dpi=150)
    plt.close(fig)

    # ---------- INDEX ----------
    summary = {}
    for k in keys:
        y, p = data[k]
        summary[k] = {
            'n': int(len(y)), 'n_positive': int(np.sum(y)),
            'auc': round(float(roc_auc_score(y, p)), 3),
            'auc_ci95': [round(v, 3) for v in boot_auc(y, p)],
            'op_threshold': round(OPS_TH[k], 3),
            **{l: (round(v[0], 3) if isinstance(v, tuple) else round(v, 3))
               for l, v in metrics_at(y, p, OPS_TH[k]).items()},
        }
    with open(os.path.join(out, 'chi_muc.json'), 'w', encoding='utf-8') as f:
        json.dump({'folder': out, 'summary': summary,
                   'charts': sorted(os.listdir(out))}, f,
                  ensure_ascii=False, indent=2)

    print('Biểu đồ đã vẽ →', out)
    for f in sorted(os.listdir(out)):
        print('  ', f)
    print('\nTÓM TẮT ĐỊNH LƯỢNG:')
    for k in keys:
        s = summary[k]
        print(f"  {LABELS[k]:34s} AUC {s['auc']:.3f} "
              f"[{s['auc_ci95'][0]:.3f}–{s['auc_ci95'][1]:.3f}] "
              f"Acc {s['Acc']:.2f} Sens {s['Sens']:.2f} Spec {s['Spec']:.2f} "
              f"F1 {s['F1']:.2f} (n={s['n']}, th={s['op_threshold']})")


if __name__ == '__main__':
    main()
