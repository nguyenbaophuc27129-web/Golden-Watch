# -*- coding: utf-8 -*-
"""
S-2 (NK-03 kế tiếp) — SPEECH openSMILE eGeMAPSv02 vs 48 FEATURES (LOSO)
=======================================================================
So sánh công bằng TRÊN CÙNG bộ file + CÙNG fold người của lần chạy
48 features chính thức (speech_speaker_loso_20260910_214834):
  - Nạp danh sách file từ oof_predictions.csv của lần chạy gốc.
  - Trích openSMILE eGeMAPSv02 (88 features chuẩn tài liệu rối loạn nói).
  - LOSO LogReg (đầu tuyến tính — bài học HistGB) cùng protocol.

Kết quả dù tốt hay kém hơn 0.620 đều công bố: tốt hơn → nâng độ chính xác
người-level; kém hơn → chứng minh 48 features tự trích đã đủ, không cần
phụ thuộc thư viện ngoài (cũng là kết quả khoa học).
Chạy: PYTHONUTF8=1 python training/eval_speech_opensmile_loso.py
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

import eval_speech_speaker_loso as s1   # tái dùng protocol + fold

REF_DIR = os.path.join(ROOT, 'test_results',
                       'speech_speaker_loso_20260910_214834')
SEED = 42


def extract_opensmile(rows):
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals)
    X, meta, fail = [], [], 0
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            df = smile.process_file(r['path'])
            v = df.iloc[0].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(v)):
                raise ValueError('non-finite')
        except Exception:
            fail += 1
            continue
        X.append(v)
        meta.append(r)
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s', flush=True)
    X = np.array(X, dtype=np.float64)
    # điền NaN còn sót bằng median cột (trung thực: đếm số ô điền)
    n_nan = int(np.isnan(X).sum())
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    idx = np.where(np.isnan(X))
    X[idx] = col_med[idx[1]]
    print(f'openSMILE xong: {len(meta)} file, {fail} lỗi, '
          f'{n_nan} ô NaN điền median, {time.time()-t0:.0f}s')
    return X, meta


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('== S-2 openSMILE eGeMAPSv02 vs 48 features (LOSO người thật) ==')

    # --- nạp bộ file gốc từ OOF CSV của lần chạy 48 ft ---
    ref_csv = os.path.join(REF_DIR, 'oof_predictions.csv')
    ref = pd.read_csv(ref_csv)
    ref_set = set(ref['path'].astype(str))
    print(f'Lần chạy gốc: {len(ref_set)} file từ {os.path.basename(REF_DIR)}')

    all_rows = s1.scan_sessions(s1.DEFAULT_TORGO)
    rows = [r for r in all_rows
            if os.path.basename(r['path']) in ref_set]
    print(f'Khớp lại đường dẫn: {len(rows)}/{len(ref_set)} file')

    X, meta = extract_opensmile(rows)
    y = np.array([m['label'] for m in meta])
    print(f'eGeMAPSv02: {X.shape[1]} features × {len(meta)} file · '
          f'{len(set(m["speaker"] for m in meta))} người')

    # --- LOSO LogReg, cùng logic fold người như S-1 ---
    print('[LOSO] LogReg trên 88 ft eGeMAPSv02 …', flush=True)
    oof, folds = s1.loso(X, meta, s1.train_logreg)
    m = s1.pooled_metrics(y, oof)
    print(f"  LogReg 88ft: AUC {m['auc']} thr {m['threshold']} · "
          f"Sens {m['sens']} {m['sens_ci']} · Spec {m['spec']} {m['spec_ci']}")

    # --- số đối chiếu từ lần chạy gốc (đọc summary) ---
    with open(os.path.join(REF_DIR, 'summary.json'), encoding='utf-8') as f:
        ref_sum = json.load(f)
    ref_auc48 = ref_sum['p1']['logreg']['auc']

    out_dir = os.path.join(
        ROOT, 'test_results',
        f'speech_opensmile_loso_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    res = {'protocol': 'S-2 openSMILE eGeMAPSv02 LOSO (so 48 ft cùng file)',
           'seed': SEED, 'n_files': len(meta), 'n_features': X.shape[1],
           'opensmile_logreg': m,
           'ref_48ft_logreg_auc': ref_auc48,
           'delta_auc': round(float(m['auc']) - ref_auc48, 4)}
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    import csv
    with open(os.path.join(out_dir, 'oof_opensmile.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path', 'speaker', 'mic', 'label', 'p_logreg_88ft'])
        for i, mm in enumerate(meta):
            w.writerow([os.path.basename(mm['path']), mm['speaker'],
                        mm['mic'], mm['label'], round(float(oof[i]), 4)])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=170)
    labels = ['48 ft tự trích\n(production)', 'eGeMAPSv02 88 ft\n(openSMILE)']
    aucs = [ref_auc48, m['auc']]
    bars = ax.bar(labels, aucs, color=['#1a6faf', '#8e44ad'], width=0.55)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.3f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('AUC LOSO theo người thật')
    ax.set_title(f"openSMILE vs 48 ft — cùng {len(meta)} file, cùng fold "
                 f"(Δ = {res['delta_auc']:+.3f})")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'feature_set_comparison.png'))

    print(f"\nΔ AUC (88ft − 48ft) = {res['delta_auc']:+.4f}")
    print(f'Đã lưu: {out_dir}')
    return res


if __name__ == '__main__':
    main()
