# -*- coding: utf-8 -*-
r"""
E-3 (NK-08) — ARM TRÊN DỮ LIỆU THẬT ĐẦU TIÊN
=============================================
Protocol E3 (KE_HOACH v2.0): đội tự quay 3–5 người × 2 điều kiện trước camera:
  thu_tuc\*.mp4  — giơ 2 tay ngang giữ ~20s (label 0, bình thường)
  tha_tay\*.mp4  — giơ 2 tay rồi THẢ RƠI 1 tay ~20s (label 1, mô phỏng NIHSS item5)

Chạy ArmWeaknessDetector THẬT (YOLO pose + ML .pth production) trên từng frame
(2 fps), gộp prob theo median từng video → AUC/sens/spec + Wilson CI.

n nhỏ — GHI RÕ "smoke test, n video nhỏ" trong báo cáo (trung thực).
Chạy: PYTHONUTF8=1 python training/eval_arm_e3.py --thu thu_tuc --tha tha_tay
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, 'src')
MODELS = os.path.join(ROOT, 'models')

VID_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def scan_videos(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(VID_EXT))


def score_video(path, arm, sample_hz=2.0):
    """Chạy arm detector thật trên video → dict prob trung bình + số frame."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_hz)))
    probs, n_frames, i = [], 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            n_frames += 1
            r = arm.detect_arm_weakness(rgb)
            if r.get('status') not in ('NO_PERSON', 'NO_POSE'):
                probs.append(float(r.get('arm_prob', 0) or 0))
        i += 1
    cap.release()
    if not probs:
        return {'prob': None, 'n_valid': 0, 'n_frames': n_frames,
                'dur_s': i / fps}
    return {'prob': float(np.median(probs)),
            'prob_mean': float(np.mean(probs)),
            'prob_max': float(np.max(probs)),
            'n_valid': len(probs), 'n_frames': n_frames, 'dur_s': i / fps}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (round(100 * (c - h) / d, 1), round(100 * p, 1),
            round(100 * (c + h) / d, 1))


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--thu', default='thu_tuc',
                    help='thư mục video giơ 2 tay bình thường (label 0)')
    ap.add_argument('--tha', default='tha_tay',
                    help='thư mục video thả 1 tay (label 1)')
    args = ap.parse_args()

    print('== E-3 ARM TRÊN VIDEO THẬT (smoke test n nhỏ) ==')
    sys.path.insert(0, SRC_DIR)
    from detection.arm_module import ArmWeaknessDetector
    arm = ArmWeaknessDetector(
        pose_model_path=os.path.join(SRC_DIR, 'yolov8n-pose.pt'),
        ml_model_path=os.path.join(MODELS, 'arm_weakness_20260830_200657.pth'),
        scaler_path=os.path.join(MODELS,
                                 'arm_weakness_20260830_200657_scaler.pkl'))

    rows = []
    for folder, label, tag in ((args.thu, 0, 'thu_tuc'),
                               (args.tha, 1, 'tha_tay')):
        vids = scan_videos(folder)
        print(f'{tag}: {len(vids)} video')
        for vp in vids:
            s = score_video(vp, arm)
            if s is None or s['prob'] is None:
                print(f"  {os.path.basename(vp)}: KHÔNG có frame hợp lệ "
                      f"({s['n_frames'] if s else '?'} frame đọc được)")
                continue
            rows.append({'video': os.path.basename(vp), 'cond': tag,
                         'label': label, **s})
            print(f"  {os.path.basename(vp)}: prob {s['prob']:.1f} "
                  f"(mean {s['prob_mean']:.1f}, max {s['prob_max']:.1f}, "
                  f"{s['n_valid']}/{s['n_frames']} frame)")

    if len({r['label'] for r in rows}) < 2:
        print('\n[!] Cần ÍT NHẤT 1 video mỗi điều kiện để eval. Thoát.')
        return

    y = np.array([r['label'] for r in rows])
    p = np.array([r['prob'] for r in rows])

    from sklearn.metrics import roc_auc_score, roc_curve
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None
    fpr, tpr, thr = roc_curve(y, p)
    j = int(np.argmax(tpr - fpr))
    best_thr = float(thr[j])

    pred = (p >= best_thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens_ci = wilson(tp, tp + fn)
    spec_ci = wilson(tn, tn + fp)

    out_dir = os.path.join(
        ROOT, 'test_results',
        f'arm_e3_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    res = {'protocol': 'E3 arm real video (smoke test, n nhỏ)',
           'n_videos': len(rows),
           'n_thu_tuc': int((y == 0).sum()), 'n_tha_tay': int((y == 1).sum()),
           'auc': None if auc is None else round(auc, 4),
           'threshold_youden': round(best_thr, 1),
           'sens': sens_ci, 'spec': spec_ci,
           'confusion': {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp},
           'rows': [{k: (round(v, 2) if isinstance(v, float) else v)
                     for k, v in r.items()} for r in rows]}
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=170)
    cols = ['#1a6faf' if r['label'] == 0 else '#c0392b' for r in rows]
    names = [r['video'][:14] for r in rows]
    axes[0].bar(range(len(rows)), p, color=cols)
    axes[0].axhline(best_thr, color='k', ls='--', lw=1.2,
                    label=f'ngưỡng Youden {best_thr:.0f}')
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('arm_prob (median video)')
    axes[0].set_title(f"E3 — prob từng video (AUC "
                      f"{'-' if auc is None else f'{auc:.3f}'})")
    axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
    if auc is not None:
        axes[1].plot(fpr, tpr, lw=2.2, color='#1a6faf',
                     label=f'AUC {auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'k:', lw=1)
        axes[1].set_xlabel('FPR'); axes[1].set_ylabel('Sensitivity')
        axes[1].set_title('ROC — arm video thật (n nhỏ)')
        axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'arm_e3_eval.png'))

    print(f"\nAUC: {'—' if auc is None else f'{auc:.4f}'} · ngưỡng Youden "
          f"{best_thr:.1f} · Sens {sens_ci} · Spec {spec_ci}")
    print(f'Confusion: TP {tp} / FN {fn} / TN {tn} / FP {fp}')
    print(f'Đã lưu: {out_dir} (ghi NK-09 vào nhat ky.md)')


if __name__ == '__main__':
    main()
