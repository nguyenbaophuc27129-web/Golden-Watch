# -*- coding: utf-8 -*-
"""
TRỤ CỘT A (v3.0, NK-10) — UNCERTAINTY + ABSTENTION EVAL
=======================================================
Câu hỏi khoa học: "Hệ thống có BIẾT mình không biết không?"
  - Với mỗi module có OOF per-sample (face, speech): tính ĐỘ BẤT ĐỊNH
    từ ensemble (nhiều seed/model) + margin tới ngưỡng.
  - Abstain 10/20/30% mẫu bất định nhất → đo sens/spec/FPR trên phần GIỮ
    + tỷ lệ lỗi TRONG phần bị bỏ (bất định phải tập trung ở chỗ hay sai).

Protocol (ghi TRƯỚC, seed 42):
  FACE : cache _face_v3_features_cache.npz (3,715 ảnh × 28 ft) → OOF
         GroupKFold-5 theo BLOCK × 10 seed (42–51); C tune inner
         GroupKFold trên phần train. prob hệ = mean 10 seed,
         uncertainty = std 10 seed. Khác multiseed NK-04 (80/20) vì cần
         OOF PER-IMAGE — khai báo trung thực trong output.
  SPEECH: dùng ĐÚNG OOF CSV của NK-03 (LOSO NGƯỜI THẬT, 1,100 file,
         seed 42). prob hệ = mean(p_logreg, p_mlp); uncertainty =
         std 2 model (bất đồng kiến trúc).
  Threshold: Youden trên OOF pooled từng module (ghi rõ trong JSON).
Output: test_results/abstention_<ts>/ (JSON + CSV per-sample + 2 PNG).
Chạy: PYTHONUTF8=1 python training/eval_abstention.py [--fast]
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.fusion.uncertainty import (ensemble, cv_platt_oof, margin,
                                    combined_uncertainty,
                                    selective_summary, risk_coverage)

CACHE = os.path.join(ROOT, 'test_results', '_face_v3_features_cache.npz')
SPEECH_OOF = os.path.join(
    ROOT, 'test_results', 'speech_speaker_loso_20260910_214834',
    'oof_predictions.csv')
SEEDS = list(range(42, 52))


def youden_threshold(y, p):
    """Ngưỡng Youden J tối đa trên OOF pooled (giống pooled_metrics NK-03)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, p)), 4)


# ----------------------------------------------------------------------
# FACE — OOF 10-seed ensemble
# ----------------------------------------------------------------------
def face_oof_ensemble(X, y, blocks, log=print):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    P = np.full((n, len(SEEDS)), np.nan)
    for si, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed)
        groups = np.unique(blocks)
        perm = rng.permutation(groups)
        chunks = np.array_split(perm, 5)   # 5 fold ~20% block mỗi fold
        for vi, chunk in enumerate(chunks):
            te_m = np.isin(blocks, chunk)
            tr_idx = np.where(~te_m)[0]
            # tune C bằng GroupKFold nội bộ trên phần train (giống NK-04)
            from sklearn.model_selection import GroupKFold
            best = (None, -1)
            for C in (0.03, 0.1, 0.3, 1.0, 3.0):
                oof_tr = np.zeros(len(tr_idx))
                for ti, v in GroupKFold(5).split(X[tr_idx], y[tr_idx],
                                                 blocks[tr_idx]):
                    sc = StandardScaler().fit(X[tr_idx][ti])
                    clf = LogisticRegression(C=C, max_iter=2000,
                                             random_state=42)
                    clf.fit(sc.transform(X[tr_idx][ti]), y[tr_idx][ti])
                    oof_tr[v] = clf.predict_proba(
                        sc.transform(X[tr_idx][v]))[:, 1]
                from sklearn.metrics import roc_auc_score
                auc_tr = roc_auc_score(y[tr_idx], oof_tr)
                if auc_tr > best[1]:
                    best = (C, auc_tr)
            sc = StandardScaler().fit(X[tr_idx])
            clf = LogisticRegression(C=best[0], max_iter=2000,
                                     random_state=42)
            clf.fit(sc.transform(X[tr_idx]), y[tr_idx])
            P[te_m, si] = clf.predict_proba(sc.transform(X[te_m]))[:, 1]
        log(f'  seed {seed}: xong ({si + 1}/{len(SEEDS)})')
    assert not np.isnan(P).any(), 'OOF face thiếu ô'
    return P


# ----------------------------------------------------------------------
# SPEECH — nạp OOF CSV của NK-03
# ----------------------------------------------------------------------
def speech_load():
    rows = []
    with open(SPEECH_OOF, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['label']), float(r['p_logreg']),
                         float(r['p_mlp']), r['speaker']))
    y = np.array([r[0] for r in rows])
    P = np.array([[r[1], r[2]] for r in rows])
    speakers = np.array([r[3] for r in rows])
    return y, P, speakers


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true',
                    help='smoke: 2 seed face')
    args = ap.parse_args()
    if args.fast:
        global SEEDS
        SEEDS = [42, 43]

    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('== TRỤ CỘT A — ABSTENTION EVAL (NK-10) ==')

    out_dir = os.path.join(
        ROOT, 'test_results',
        f'abstention_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    res = {'protocol': {
        'seed': 42,
        'face': ('OOF GroupKFold-5 theo block × %d seed %s, prob hệ = mean, '
                 'uncertainty = std ensemble; C tune inner GroupKFold '
                 '(khác NK-04 80/20 vì cần OOF per-image)' % (
                     len(SEEDS), SEEDS)),
        'speech': ('tái dùng OOF CSV NK-03 (LOSO người thật seed 42); '
                   'prob hệ = mean(p_logreg,p_mlp), uncertainty = std'),
        'threshold': 'Youden trên OOF pooled từng module',
        'uncertainty_runtime': 'combined = 0.5·std/0.5 + 0.5·(1 − margin/0.5)',
        'abstain_rule': 'top-frac bất định nhất → NEEDS_CHECK (không hạ EMERGENCY)',
    }, 'modules': {}}

    per_sample = {}

    # ---------------- FACE ----------------
    print('\n[FACE] OOF %d-seed ensemble (cache npz)…' % len(SEEDS))
    d = np.load(CACHE, allow_pickle=True)
    X, y_f, blocks = d['X'], d['y'], d['blocks']
    P_f = face_oof_ensemble(X, y_f, blocks)
    p_f, std_f = ensemble(P_f)
    thr_f = youden_threshold(y_f, p_f)
    m_f = margin(p_f, thr_f)
    unc_f = combined_uncertainty(std_f, m_f)
    # calibrate trung thực theo block (sanity AUC không đổi — Platt đơn điệu)
    p_f_cal = cv_platt_oof(p_f, y_f, blocks)
    m_f = selective_summary(y_f, p_f, unc_f, thr_f)
    res['modules']['face'] = {
        'auc_ensemble_mean': auc(y_f, p_f),
        'auc_platt_cv': auc(y_f, p_f_cal),
        'auc_single_seed_range': [auc(y_f, P_f[:, 0]), auc(y_f, P_f[:, -1])],
        'threshold_youden': round(thr_f, 3),
        'selective': m_f,
    }
    per_sample['face'] = (y_f, p_f, std_f, unc_f, thr_f)
    print(f"  AUC ensemble {res['modules']['face']['auc_ensemble_mean']} "
          f"(platt-cv {res['modules']['face']['auc_platt_cv']}) · thr "
          f"{thr_f:.3f} · err base {m_f['base']['err_rate']}%")
    for k, v in m_f['steps'].items():
        print(f"  {k}: giữ {v['kept']['coverage']:.0%} → err "
              f"{v['kept']['err_rate']}% (−{v['err_reduction_kept']}đ) · "
              f"FPR {v['kept']['fpr']}% (base {m_f['base']['fpr']}%) · "
              f"sens {v['kept']['sens']}% · lỗi trong phần bỏ "
              f"{v['err_rate_among_abstained']}%")

    # ---------------- SPEECH ----------------
    print('\n[SPEECH] nạp OOF NK-03 (LOSO người thật)…')
    y_s, P_s, speakers = speech_load()
    p_s, std_s = ensemble(P_s)
    thr_s = youden_threshold(y_s, p_s)
    m_s = margin(p_s, thr_s)
    unc_s = combined_uncertainty(std_s, m_s)
    p_s_cal = cv_platt_oof(p_s, y_s, speakers)
    sel_s = selective_summary(y_s, p_s, unc_s, thr_s)
    res['modules']['speech'] = {
        'auc_logreg': auc(y_s, P_s[:, 0]),
        'auc_mlp': auc(y_s, P_s[:, 1]),
        'auc_ensemble_mean': auc(y_s, p_s),
        'auc_platt_cv': auc(y_s, p_s_cal),
        'threshold_youden': round(thr_s, 3),
        'selective': sel_s,
    }
    per_sample['speech'] = (y_s, p_s, std_s, unc_s, thr_s)
    print(f"  LogReg {res['modules']['speech']['auc_logreg']} · MLP "
          f"{res['modules']['speech']['auc_mlp']} · ensemble "
          f"{res['modules']['speech']['auc_ensemble_mean']} · thr {thr_s:.3f}"
          f" · err base {sel_s['base']['err_rate']}%")
    for k, v in sel_s['steps'].items():
        print(f"  {k}: giữ {v['kept']['coverage']:.0%} → err "
              f"{v['kept']['err_rate']}% (−{v['err_reduction_kept']}đ) · "
              f"FPR {v['kept']['fpr']}% (base {sel_s['base']['fpr']}%) · "
              f"sens {v['kept']['sens']}% · lỗi trong phần bỏ "
              f"{v['err_rate_among_abstained']}%")

    res['headline'] = {}
    for mod, sel in (('face', m_f), ('speech', sel_s)):
        res['headline'][mod] = {
            'unc_error_auc': sel['unc_error_auc'],
            'abstain_20': sel['steps'].get('abstain_20'),
        }
    print(f"\nunc_error_auc (bất định có dự đoán được lỗi?): face "
          f"{m_f['unc_error_auc']} · speech {sel_s['unc_error_auc']}")

    # ---------------- CSV per-sample ----------------
    with open(os.path.join(out_dir, 'per_sample_uncertainty.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['module', 'label', 'prob_system', 'unc_std',
                    'unc_combined', 'youden_thr', 'pred_correct'])
        for mod, (yy, pp, ss, uu, tt) in per_sample.items():
            correct = ((pp >= tt).astype(int) == yy)
            for i in range(len(yy)):
                w.writerow([mod, int(yy[i]), round(float(pp[i]), 4),
                            round(float(ss[i]), 4), round(float(uu[i]), 4),
                            round(tt, 3), int(correct[i])])

    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---------------- FIGURES ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=170)
    for ax, (name, (yy, pp, ss, uu, tt)) in zip(axes, per_sample.items()):
        rc = risk_coverage(yy, pp, uu, tt, n_points=41)
        ax.plot([c * 100 for c in rc['coverage']], rc['err_rate'], 'o-',
                ms=3, color='#1a6faf', lw=1.8,
                label='lỗi trên phần ĐƯỢC GIỮ')
        for frac, col in ((0.1, '#e67e22'), (0.2, '#c0392b'), (0.3, '#8e44ad')):
            ax.axvline((1 - frac) * 100, color=col, ls=':', lw=1)
            ax.text((1 - frac) * 100, ax.get_ylim()[1] * 0.97,
                    f'abstain {int(frac*100)}%', fontsize=7, rotation=90,
                    va='top', ha='right', color=col)
        ax.set_xlabel('Coverage % (phần giữ lại)')
        ax.set_ylabel('% lỗi trên phần giữ')
        ax.set_title(f'{name.upper()} — risk–coverage (bỏ mẫu bất định nhất)')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'abstention_curves.png'))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=170)
    for ax, (name, (yy, pp, ss, uu, tt)) in zip(axes, per_sample.items()):
        correct = ((pp >= tt).astype(int) == yy)
        bins = np.linspace(0, max(uu.max(), 0.01), 40)
        ax.hist(uu[correct], bins=bins, alpha=0.6, color='#1a6faf',
                label=f'đúng (n={int(correct.sum())})', density=True)
        ax.hist(uu[~correct], bins=bins, alpha=0.6, color='#c0392b',
                label=f'sai (n={int((~correct).sum())})', density=True)
        ax.set_xlabel('độ bất định tổng hợp')
        ax.set_ylabel('mật độ')
        ax.set_title(f'{name.upper()} — bất định: đúng vs sai')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'uncertainty_separation.png'))

    print(f'\nĐã lưu: {out_dir} (JSON + CSV + 2 PNG)')


if __name__ == '__main__':
    main()
