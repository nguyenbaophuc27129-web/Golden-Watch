# -*- coding: utf-8 -*-
"""
EVALUATE FACE RULES — đánh giá face_module_v7 (RULE-BASED runtime) trên
tập TEST block-split của dataset công khai "Annotated stroke and non stroke".

Mục đích (fix L-04):
  - Runtime KHÔNG dùng model PyTorch 93.75% (model đó chưa được wire vào app).
  - Bằng chứng phải đo trên cái đang chạy thật: bộ rules MediaPipe landmark.
  - Đánh giá ở mức ẢNH trên tập TEST block-split (không leakage frame liên tiếp).

Output: test_results/face_rules_eval_<ts>.json + .csv + roc png

Usage:
    python training/evaluate_face_rules.py [--max_images 0]
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from detection.face_module_v7 import FaceAsymmetryDetector  # noqa: E402

DATASET_PATH = (r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\data"
                r"\datasets\face\Annotated stroke and non stroke Dataset")
OUT_DIR = os.path.join(parent_dir, 'test_results')
SEED = 42
BLOCK = 50


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(p, 4), round(max(0, c - h), 4), round(min(1, c + h), 4)]


def scan_dataset():
    rows = []
    for sub, lab in [('NonStroke', 0), ('Stroke', 1)]:
        p = os.path.join(DATASET_PATH, sub)
        for t in sorted(os.listdir(p)):
            if t.endswith('.txt'):
                ip = os.path.join(p, t.replace('.txt', '.jpg'))
                if os.path.exists(ip):
                    bbox = None
                    try:
                        with open(os.path.join(p, t)) as f:
                            parts = f.readline().split()
                        if len(parts) >= 5:
                            bbox = list(map(float, parts[1:5]))
                    except Exception:
                        pass
                    rows.append({'image_path': ip, 'label': lab,
                                 'bbox': bbox,
                                 'block_id': f"{lab}_{int(t.split('_')[1].split('.')[0]) // BLOCK}"})
    return pd.DataFrame(rows)


def crop_face(img, bbox, margin=0.15):
    """Crop theo YOLO bbox (cx, cy, w, h normalized) + margin."""
    if img is None or bbox is None:
        return img
    H, W = img.shape[:2]
    cx, cy, w, h = bbox
    x1 = int(max(0, (cx - w / 2) * W - margin * W))
    y1 = int(max(0, (cy - h / 2) * H - margin * H))
    x2 = int(min(W, (cx + w / 2) * W + margin * W))
    y2 = int(min(H, (cy + h / 2) * H + margin * H))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return img
    return img[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_images', type=int, default=0,
                    help='0 = toàn bộ tập test')
    args = ap.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    df = scan_dataset()
    blocks = df.drop_duplicates('block_id')[['block_id', 'label']]
    _, temp_b = train_test_split(blocks, test_size=0.4,
                                 random_state=SEED, stratify=blocks['label'])
    _, test_b = train_test_split(temp_b, test_size=0.5,
                                 random_state=SEED, stratify=temp_b['label'])
    test_df = df[df['block_id'].isin(test_b['block_id'])].reset_index(drop=True)
    print(f"Tập TEST: {len(test_df)} ảnh / {len(test_b)} blocks "
          f"(stroke {(test_df.label == 1).sum()}, "
          f"nonstroke {(test_df.label == 0).sum()})")

    if args.max_images:
        half = args.max_images // 2
        pos = test_df[test_df.label == 1].head(half)
        neg = test_df[test_df.label == 0].head(args.max_images - half)
        test_df = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)
        print(f"Chạy nhanh {len(test_df)} ảnh (--max_images)")

    det = FaceAsymmetryDetector(history_size=1)  # tắt median temporal
    rows = []
    t0 = datetime.now()
    for k, (_, r) in enumerate(test_df.iterrows()):
        img = cv2.imread(r['image_path'])
        res = {'status': 'READ_ERROR', 'score': 0.0}
        if img is not None:
            crop = crop_face(img, r['bbox'])
            # VIDEO mode yêu cầu timestamp tăng đơn điệu → dùng counter riêng
            res = det.process_frame(crop, frame_timestamp_ms=1000 + k * 33)
        rows.append({'image': os.path.basename(r['image_path']),
                     'label': int(r['label']),
                     'score': float(res.get('score', 0) or 0),
                     'status': res.get('status', 'ERROR')})
        if (k + 1) % 100 == 0:
            el = (datetime.now() - t0).total_seconds()
            print(f"  {k + 1}/{len(test_df)} ({el:.0f}s)")

    out = pd.DataFrame(rows)
    valid = out[~out['status'].isin(['NO_FACE', 'READ_ERROR', 'ERROR',
                                     'NO_DETECTOR', 'INVALID_LANDMARKS'])]
    n_invalid = len(out) - len(valid)
    y = valid['label'].values
    s = valid['score'].values

    # Youden trên chính tập test (chỉ là mô tả điểm số, không tune model)
    fpr, tpr, th = roc_curve(y, s)
    j = tpr - fpr
    best_i = int(np.argmax(j))
    best_th = float(th[best_i])

    pred = (s >= best_th).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())

    auc = float(roc_auc_score(y, s))
    sens, spec, acc = (wilson_ci(tp, tp + fn), wilson_ci(tn, tn + fp),
                       wilson_ci(tp + tn, len(y)))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    report = {
        'what': 'Đánh giá face_module_v7 RULE-BASED (runtime thật) trên ảnh test',
        'dataset': ('Annotated stroke and non stroke Dataset (công khai) — '
                    'tập TEST block-split, không leakage frame'),
        'created': datetime.now().isoformat(),
        'n_test_images': len(out), 'n_valid': len(valid),
        'n_no_face_or_error': n_invalid,
        'n_stroke': n_pos, 'n_nonstroke': n_neg,
        'youden_threshold_score': round(best_th, 2),
        'auc': round(auc, 4),
        'sensitivity': {'tp': tp, 'fn': fn, **dict(zip(
            ['value', 'ci_low', 'ci_high'], sens))},
        'specificity': {'tn': tn, 'fp': fp, **dict(zip(
            ['value', 'ci_low', 'ci_high'], spec))},
        'accuracy': dict(zip(['value', 'ci_low', 'ci_high'], acc)),
        'note': ('Youden chọn TRÊN tập test → chỉ mang tính mô tả; kết luận '
                 'chốt cần thu thêm webcam thật (protocol B).'),
        'per_image_file': None,
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUT_DIR, f'face_rules_eval_{ts}.csv')
    out.to_csv(csv_path, index=False)
    report['per_image_file'] = os.path.basename(csv_path)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.scatter(fpr[best_i], tpr[best_i], c='red',
                    label=f"Youden th={best_th:.1f}")
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
        plt.title('Face rules (runtime) — test block-split')
        plt.tight_layout()
        png = os.path.join(OUT_DIR, f'face_rules_eval_{ts}_roc.png')
        plt.savefig(png, dpi=120)
        report['roc_png'] = os.path.basename(png)
    except Exception as e:
        print(f"(Không vẽ được ROC: {e})")

    jp = os.path.join(OUT_DIR, f'face_rules_eval_{ts}.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== KẾT QUẢ FACE RULES (runtime thật) =====")
    print(f"Ảnh test: {len(out)} (hợp lệ {len(valid)}, NO_FACE/lỗi {n_invalid})")
    print(f"Stroke/NonStroke: {n_pos}/{n_neg}")
    print(f"AUC = {auc:.3f}")
    print(f"Youden score threshold = {best_th:.2f}")
    print(f"Sensitivity = {tp}/{tp + fn} = {sens[0]:.2%} "
          f"CI [{sens[1]:.2%}, {sens[2]:.2%}]")
    print(f"Specificity = {tn}/{tn + fp} = {spec[0]:.2%} "
          f"CI [{spec[1]:.2%}, {spec[2]:.2%}]")
    print(f"JSON: {jp}")


if __name__ == '__main__':
    main()
