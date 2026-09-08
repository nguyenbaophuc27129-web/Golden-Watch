# -*- coding: utf-8 -*-
"""
TRAIN GAIT MODEL v2 — SUBJECT-LEVEL SPLIT (PSCS v8.0)
Sửa lỗi L-04: v1 chia window ngẫu nhiên train/test → leakage (window của cùng
1 subject nằm ở cả 2 tập → accuracy phình to).

v2:
  - Dữ liệu: PhysioNet "Gait in Aging and Disease Database v1.0.0" (DỮ LIỆU THẬT)
    15 subject: o1-o5 (người già khỏe), y1-y5 (trẻ khỏe) = NORMAL;
                pd1-pd5 (Parkinson) = ABNORMAL (bất thường dáng đi).
  - Đánh giá: LOSO (Leave-One-Subject-Out) 15 folds — subject test chưa từng
    xuất hiện trong train → không leakage.
  - Báo cáo: window-level Sens/Spec/Acc + Wilson 95% CI + ROC AUC;
             subject-level (median prob/subject) — 15 quyết định thật.
  - Model cuối: train trên cả 15 subject, lưu kèm metadata JSON (nguồn, LOSO).

Usage:
    python training/train_gait_model_v2.py [--gait_path PATH]
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

from detection.gait_module import GaitAbnormalityDetector  # noqa: E402

# Hyperparams (giữ nguyên v1 để so sánh công bằng)
EPOCHS = 150
BATCH_SIZE = 32
LR = 0.001
HIDDEN_DIMS = [64, 32]
DROPOUT = 0.5
WEIGHT_DECAY = 0.001
EARLY_STOP_PATIENCE = 25
WINDOW_SIZE = 100
STRIDE = 50
SEED = 42

DEFAULT_GAIT_PATH = (r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\data"
                     r"\datasets\gait\gait-in-aging-and-disease-database-1.0.0"
                     r"\gait-in-aging-and-disease-database-1.0.0")

DATASET_CITATION = (
    "Hausdorff JM et al. Gait in Aging and Disease Database v1.0.0. "
    "PhysioNet (physionet.org/content/gait-in-aging-and-disease-database/1.0.0/); "
    "những người tham gia đi bộ trên thảm lực đo lực thẳng đứng, 15 subject "
    "(5 người già khỏe o1-o5, 5 trẻ khỏe y1-y5, 5 Parkinson pd1-pd5)."
)


class GaitClassifierV2(nn.Module):
    """Kiến trúc GIỐNG HỆT gait_module._load_ml_model để load_state_dict khớp."""

    def __init__(self, input_dim=8):
        super().__init__()
        layers = []
        prev = input_dim
        for h in HIDDEN_DIMS:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def wilson_ci(k, n, z=1.96):
    """Khoảng tin cậy Wilson 95% cho tỷ lệ (n nhỏ vẫn hợp lệ)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(p, 4), round(max(0, center - half), 4),
            round(min(1, center + half), 4))


def load_subjects(gait_path, detector):
    """Load 15 subject → list dict {subject_id, label, windows_features}."""
    subjects = []

    def load_group(prefixes, label):
        for pat in prefixes:
            for f in glob.glob(os.path.join(gait_path, pat)):
                data = detector.load_gait_data(f)
                if data is None or len(data) < WINDOW_SIZE + 10:
                    continue
                feats = []
                for s in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
                    feats.append(detector.extract_gait_features(
                        data[s:s + WINDOW_SIZE]))
                feats = np.array(feats, dtype=np.float32)
                feats = feats[~np.all(feats == 0, axis=1)]  # bỏ window lỗi
                sid = os.path.basename(f).split('-')[0]
                subjects.append({'id': sid, 'label': label, 'features': feats,
                                 'file': os.path.basename(f)})
                print(f"  {os.path.basename(f):20s} label={label} "
                      f"windows={len(feats)}")

    print("NORMAL (y1-y5, o1-o5):")
    load_group([f"y{i}-*si.txt" for i in range(1, 6)] +
               [f"o{i}-*si.txt" for i in range(1, 6)], 0)
    print("ABNORMAL — Parkinson (pd1-pd5):")
    load_group([f"pd{i}-*si.txt" for i in range(1, 6)], 1)

    assert len(subjects) >= 10, f"Quá ít subject: {len(subjects)}"
    return subjects


def train_fold(X_tr, y_tr, X_te, device):
    """Train 1 fold, trả probs cho X_te."""
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    cls_count = np.bincount(y_tr, minlength=2).astype(np.float32)
    weights = torch.tensor(cls_count.sum() / (2 * cls_count + 1e-9),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    model = GaitClassifierV2().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)

    X_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    n = len(X_t)
    best_loss, wait = float('inf'), 0
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        ep_loss, nb = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            out = model(X_t[idx])
            loss = loss_fn(out, y_t[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        ep_loss /= max(nb, 1)
        if ep_loss < best_loss - 1e-4:
            best_loss, wait = ep_loss, 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                break

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_te_s, dtype=torch.float32, device=device))
        prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gait_path', default=DEFAULT_GAIT_PATH)
    args = ap.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | Seed {SEED}")
    detector = GaitAbnormalityDetector()  # không load model cũ
    subjects = load_subjects(args.gait_path, detector)

    n_sub = len(subjects)
    n_win = sum(len(s['features']) for s in subjects)
    n_pos = sum(1 for s in subjects if s['label'] == 1)
    print(f"\n{n_sub} subjects ({n_pos} Parkinson, {n_sub - n_pos} control), "
          f"{n_win} windows")

    # ---------------- LOSO 15 FOLDS ----------------
    print("\n" + "=" * 70)
    print("LEAVE-ONE-SUBJECT-OUT (15 folds — không leakage subject)")
    print("=" * 70)

    all_prob, all_true, fold_rows = [], [], []
    for i, test_sub in enumerate(subjects):
        X_tr = np.vstack([s['features'] for j, s in enumerate(subjects)
                          if j != i])
        y_tr = np.concatenate([np.full(len(s['features']), s['label'])
                               for j, s in enumerate(subjects) if j != i])
        X_te = test_sub['features']
        prob = train_fold(X_tr, y_tr, X_te, device)

        all_prob.extend(prob.tolist())
        all_true.extend([test_sub['label']] * len(prob))
        win_pred = (prob >= 0.5).astype(int)
        win_acc = float((win_pred == test_sub['label']).mean())
        fold_rows.append({'test_subject': test_sub['id'],
                          'label': test_sub['label'],
                          'n_windows': len(prob),
                          'median_prob': float(np.median(prob)),
                          'window_acc': round(win_acc, 4)})
        print(f"Fold {i + 1:2d}/{n_sub}  test={test_sub['id']:5s} "
              f"label={test_sub['label']}  wins={len(prob):3d}  "
              f"median_prob={np.median(prob):.3f}  win_acc={win_acc:.2f}")

    all_prob = np.array(all_prob)
    all_true = np.array(all_true)

    # Ngưỡng Youden trên pooled LOSO probs (mild optimism — công bố rõ)
    best_th, best_j = 0.5, -1
    for th in np.arange(0.05, 0.96, 0.05):
        pred = (all_prob >= th).astype(int)
        tp = int(((pred == 1) & (all_true == 1)).sum())
        fn = int(((pred == 0) & (all_true == 1)).sum())
        tn = int(((pred == 0) & (all_true == 0)).sum())
        fp = int(((pred == 1) & (all_true == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if tpr - fpr > best_j:
            best_j, best_th = tpr - fpr, float(th)

    pred = (all_prob >= best_th).astype(int)
    tp = int(((pred == 1) & (all_true == 1)).sum())
    fn = int(((pred == 0) & (all_true == 1)).sum())
    tn = int(((pred == 0) & (all_true == 0)).sum())
    fp = int(((pred == 1) & (all_true == 0)).sum())
    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)
    acc_ci = wilson_ci(tp + tn, len(all_true))
    auc = float(roc_auc_score(all_true, all_prob))

    # Subject-level: median prob mỗi subject vs best_th
    med = [f['median_prob'] for f in fold_rows]
    lab = [f['label'] for f in fold_rows]
    sub_pred = (np.array(med) >= best_th).astype(int)
    sub_acc = float((sub_pred == np.array(lab)).mean())
    sub_correct = [f"{f['test_subject']}({'ok' if p == l else 'MISS'})"
                   for f, p, l in zip(fold_rows, sub_pred, lab)]

    print("\n----- KẾT QUẢ LOSO (window-level) -----")
    print(f"Threshold (Youden): {best_th:.2f}")
    print(f"Sensitivity (PD) : {tp}/{tp + fn} = {sens_ci[0]:.2%} "
          f"CI95 [{sens_ci[1]:.2%}, {sens_ci[2]:.2%}]")
    print(f"Specificity      : {tn}/{tn + fp} = {spec_ci[0]:.2%} "
          f"CI95 [{spec_ci[1]:.2%}, {spec_ci[2]:.2%}]")
    print(f"Accuracy         : {tp + tn}/{len(all_true)} = {acc_ci[0]:.2%} "
          f"CI95 [{acc_ci[1]:.2%}, {acc_ci[2]:.2%}]")
    print(f"ROC AUC          : {auc:.3f}")
    print(f"Subject-level acc: {int(sub_acc * len(lab))}/{len(lab)} = "
          f"{sub_acc:.2%}")
    print(f"Chi tiết subject : {', '.join(sub_correct)}")

    # ---------------- TRAIN MODEL CUỐI TRÊN CẢ 15 SUBJECT ----------------
    print("\nTrain model cuối trên toàn bộ 15 subject...")
    X_all = np.vstack([s['features'] for s in subjects])
    y_all = np.concatenate([np.full(len(s['features']), s['label'])
                            for s in subjects])
    scaler = StandardScaler().fit(X_all)
    X_s = scaler.transform(X_all)
    cls_count = np.bincount(y_all, minlength=2).astype(np.float32)
    weights = torch.tensor(cls_count.sum() / (2 * cls_count + 1e-9),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    model = GaitClassifierV2().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    X_t = torch.tensor(X_s, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_all, dtype=torch.long, device=device)
    gen = torch.Generator().manual_seed(SEED)
    n = len(X_t)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
    model = model.cpu().eval()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(os.path.join(parent_dir, 'models'), exist_ok=True)
    model_path = os.path.join(parent_dir, 'models',
                              f'gait_classifier_v2_{ts}.pth')
    scaler_path = model_path.replace('.pth', '_scaler.pkl')
    meta_path = model_path.replace('.pth', '_metadata.json')
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        'version': 'v2-subject-split',
        'created': ts,
        'fixes': 'L-04: bỏ window-random split (leakage) → LOSO subject-level',
        'dataset': DATASET_CITATION,
        'n_subjects': n_sub,
        'n_windows': n_win,
        'labels': '0 = control (y1-y5, o1-o5), 1 = Parkinson (pd1-pd5)',
        'limitation': ('Parkinson là proxy bất thường dáng đi, KHÔNG phải '
                       'đột quỵ; tín hiệu thảm lực ≠ video pose (cùng bộ 8 '
                       'features). Cần kiểm chứng thêm trên video thật.'),
        'hyperparams': {'epochs': EPOCHS, 'batch': BATCH_SIZE, 'lr': LR,
                        'hidden': HIDDEN_DIMS, 'dropout': DROPOUT,
                        'weight_decay': WEIGHT_DECAY, 'window': WINDOW_SIZE,
                        'stride': STRIDE, 'seed': SEED,
                        'class_weights': True},
        'loso': {
            'threshold_youden': round(best_th, 2),
            'sensitivity': {'value': sens_ci[0], 'ci95': sens_ci[1:]},
            'specificity': {'value': spec_ci[0], 'ci95': spec_ci[1:]},
            'accuracy': {'value': acc_ci[0], 'ci95': acc_ci[1:]},
            'roc_auc': round(auc, 3),
            'subject_level_acc': round(sub_acc, 4),
            'folds': fold_rows,
        },
        'files': {'model': os.path.basename(model_path),
                  'scaler': os.path.basename(scaler_path)},
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu:\n  {model_path}\n  {scaler_path}\n  {meta_path}")
    print("\nLƯU Ý: ngưỡng runtime trong gait_module.py vẫn là 64 — "
          f"ngưỡng LOSO Youden mới = {best_th:.2f} (x100). Quyết định cập "
          "nhật runtime là việc riêng (ghi LOI_SO M4-06).")


if __name__ == '__main__':
    main()
