# -*- coding: utf-8 -*-
"""
BLOCKSWEEP FIGURE (nâng tầm #1) — Quét kích thước block × họ mô hình
chỉ báo leakage cho dataset face Kaggle.

Số nguồn: test_results/leak_check_blocksize.json (training/leak_check_blocksize.py).
Kết luận khoa học:
  - LogisticRegression ổn định AUC 0.91–0.94 mọi cỡ block  → tổng quát hoá thật
  - HistGradientBoosting ≈ 1.000 kể cả block //1000 (8 block) → memorize người
    (filename chứa subject) — AUC 1.0 = CHỈ SỐ LEAKAGE, không dùng báo cáo.

Figure này đưa vào báo cáo (mục Tiến hành nghiên cứu) + poster + trả lời
giám khảo: "vì sao không công bố con số 100%?".
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    src = os.path.join(ROOT, 'test_results', 'leak_check_blocksize.json')
    with open(src, encoding='utf-8') as f:
        rows = json.load(f)

    blocks = [r['block'] for r in rows]
    logreg = [r['auc_logreg'] for r in rows]
    histgb = [r['auc_histgb'] for r in rows]
    n_blocks = [r['n_blocks'] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=170)

    ax.plot(blocks, logreg, 'o-', color='#1a6faf', lw=2.4, ms=8,
            label='LogisticRegression (tuyến tính)')
    ax.plot(blocks, histgb, 's-', color='#c0392b', lw=2.4, ms=8,
            label='HistGradientBoosting (cây)')

    ax.axhline(0.5, color='gray', ls=':', lw=1.2)
    ax.text(blocks[-1], 0.507, 'ngẫu nhiên (0.5)', ha='right',
            fontsize=9, color='gray')

    # vùng AUC ~1.0 = nghi leakage
    ax.axhspan(0.98, 1.005, color='#c0392b', alpha=0.06)
    ax.text(blocks[0], 0.985, 'vùng nghi leakage (AUC ≈ 1.0)',
            fontsize=9, color='#c0392b', va='top')

    for x, y, n in zip(blocks, logreg, n_blocks):
        ax.annotate(f'{y:.3f}\n({n} block)', (x, y), textcoords='offset points',
                    xytext=(0, -30), ha='center', fontsize=8.5, color='#1a6faf')
    for x, y in zip(blocks, histgb):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=8.5, color='#c0392b')

    ax.set_xscale('log')
    ax.set_xticks(blocks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Kích thước block khi chia GroupKFold (ảnh/block) — '
                  'block càng lớn càng tách xa người', fontsize=10)
    ax.set_ylabel('AUC-ROC (out-of-fold)', fontsize=10)
    ax.set_title('Giao thức BlockSweep — quét block size × họ mô hình:\n'
                 'AUC của mô hình cây không giảm khi block = 1000 → chỉ báo leakage',
                 fontsize=11.5)
    ax.set_ylim(0.45, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(loc='center left', fontsize=10, framealpha=0.9)

    out_dir = os.path.join(ROOT, 'test_results')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'blocksweep_leakage_diagnostic.png')
    fig.tight_layout()
    fig.savefig(out)
    print(f'Đã lưu: {out}')
    print(f'LogReg : {[round(v, 4) for v in logreg]}  (ổn định → tổng quát)')
    print(f'HistGB : {[round(v, 4) for v in histgb]}  (≈1.0 mọi block → memorize)')
    return out


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
