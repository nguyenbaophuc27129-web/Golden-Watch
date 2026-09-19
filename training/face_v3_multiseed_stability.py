# -*- coding: utf-8 -*-
"""
F-1a — FACE v3 MULTI-SEED STABILITY (NK-04)
Số AUC 0.943 của face v3 là 1 lần chia block (seed 42). Giám khảo sẽ hỏi
"chia khác thì sao?". Script này lặp protocol v3 với 10 seed (42..51):
trích đặc trưng MediaPipe 1 LẦN (cache .npz) → chỉ random lại phép chia
block train/test → phân phối AUC test / sens / spec của nhóm G4_all.

KHÔNG đụng artifact models/ — chỉ đo ổn định. Kết quả trung thực: nếu
AUC dao động nhiều, phải công bố khoảng, không công bố 1 số.
Chạy: PYTHONUTF8=1 python training/face_v3_multiseed_stability.py
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

import train_face_landmarker_v3 as v3   # tái dùng toàn bộ protocol

CACHE = os.path.join(ROOT, 'test_results', '_face_v3_features_cache.npz')
SEEDS = list(range(42, 52))
GROUP_TAG = 'G4_all'


def get_features():
    """Trích 1 lần (hoặc nạp cache) → X(G4_all), y, blocks."""
    if os.path.exists(CACHE):
        print(f'Nạp cache: {CACHE}')
        d = np.load(CACHE, allow_pickle=True)
        return d['X'], d['y'], d['blocks'], list(d['asym_names'])
    print('Trích đặc trưng MediaPipe (1 lần, lưu cache)…')
    rows = v3.scan_dataset()
    Xr, y, blocks, bs_names, _ = v3.extract_all(rows)
    asym_names, il, ir = v3.blend_asym_pairs(bs_names)
    n_bs = len(bs_names)
    L = 5 + np.array(il)
    R = 5 + np.array(ir)
    asym_X = Xr[:, L] - Xr[:, R]
    pose_X = Xr[:, n_bs + 5:n_bs + 8]
    X = np.hstack([Xr[:, :5], asym_X, pose_X])
    np.savez_compressed(CACHE, X=X, y=y, blocks=blocks,
                        asym_names=np.array(asym_names, dtype=object))
    print(f'Cache đã lưu: {CACHE} ({X.shape})')
    return X, y, blocks, asym_names


def run_seed(X, y, blocks, seed):
    """Nhân bản evaluate_group của v3 nhưng seed hoá phép chia block."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.RandomState(seed)
    groups = np.unique(blocks)
    gs = rng.permutation(groups)
    tr_g = set(gs[:int(len(gs) * 0.8)])
    tr_m = np.array([b in tr_g for b in blocks])
    te_m = ~tr_m
    tr_idx = np.where(tr_m)[0]

    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        oof_tr = np.zeros(len(tr_idx))
        for ti, vi in GroupKFold(5).split(X[tr_idx], y[tr_idx],
                                          blocks[tr_idx]):
            sc = StandardScaler().fit(X[tr_idx][ti])
            clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
            clf.fit(sc.transform(X[tr_idx][ti]), y[tr_idx][ti])
            oof_tr[vi] = clf.predict_proba(
                sc.transform(X[tr_idx][vi]))[:, 1]
        auc = roc_auc_score(y[tr_idx], oof_tr)
        if best is None or auc > best[1]:
            best = (C, auc, oof_tr)
    C, auc_oof, oof_tr = best
    fpr, tpr, thr = roc_curve(y[tr_idx], oof_tr)
    threshold = float(thr[int(np.argmax(tpr - fpr))])

    sc = StandardScaler().fit(X[tr_idx])
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
    clf.fit(sc.transform(X[tr_idx]), y[tr_idx])
    p_te = clf.predict_proba(sc.transform(X[te_m]))[:, 1]
    y_te = y[te_m]
    auc_te = float(roc_auc_score(y_te, p_te))
    pred = (p_te >= threshold).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    tn = int(((pred == 0) & (y_te == 0)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {'seed': seed, 'C': C, 'auc_oof_train': round(auc_oof, 4),
            'auc_test': round(auc_te, 4), 'sens_test': round(sens, 4),
            'spec_test': round(spec, 4), 'n_train': int(tr_m.sum()),
            'n_test': int(te_m.sum())}


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('== F-1a FACE v3 MULTI-SEED STABILITY (NK-04) ==')
    t0 = time.time()
    X, y, blocks, _ = get_features()
    print(f'Đặc trưng: {X.shape}, {time.time()-t0:.0f}s')

    runs = []
    for s in SEEDS:
        r = run_seed(X, y, blocks, s)
        runs.append(r)
        print(f"  seed {s}: AUC test {r['auc_test']:.4f} "
              f"sens {r['sens_test']*100:.1f}% spec {r['spec_test']*100:.1f}% "
              f"(C={r['C']})")

    aucs = np.array([r['auc_test'] for r in runs])
    sens = np.array([r['sens_test'] for r in runs])
    spec = np.array([r['spec_test'] for r in runs])
    summary = {'protocol': 'face v3 G4_all multi-seed (F-1a, NK-04)',
               'seeds': SEEDS,
               'auc_mean': round(float(aucs.mean()), 4),
               'auc_sd': round(float(aucs.std(ddof=1)), 4),
               'auc_min': round(float(aucs.min()), 4),
               'auc_max': round(float(aucs.max()), 4),
               'sens_mean': round(float(sens.mean()), 4),
               'spec_mean': round(float(spec.mean()), 4),
               'runs': runs}
    out_dir = os.path.join(
        ROOT, 'test_results',
        f'face_v3_multiseed_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=170)
    ax.plot(SEEDS, aucs, 'o-', color='#1a6faf', lw=2)
    ax.axhline(aucs.mean(), color='#c0392b', ls='--', lw=1.5,
               label=f"mean {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}")
    ax.fill_between([SEEDS[0] - 1, SEEDS[-1] + 1],
                    aucs.mean() - aucs.std(ddof=1),
                    aucs.mean() + aucs.std(ddof=1),
                    color='#c0392b', alpha=0.08)
    ax.set_xlabel('Seed phép chia block train/test')
    ax.set_ylabel('AUC test (block-aware, chạm test 1 lần)')
    ax.set_title('Face v3 (G4_all) — ổn định theo 10 seed')
    ax.set_ylim(0.85, 1.0)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'multiseed_stability.png'))

    print(f"\nAUC test: mean {summary['auc_mean']} ± {summary['auc_sd']} "
          f"[{summary['auc_min']}–{summary['auc_max']}] · "
          f"sens {summary['sens_mean']} · spec {summary['spec_mean']}")
    print(f'Đã lưu: {out_dir}')


if __name__ == '__main__':
    main()
