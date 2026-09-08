# -*- coding: utf-8 -*-
"""
THÍ NGHIỆM: Dataset face này có CHỨA tín hiệu phân biệt stroke ở mức
landmark/bất đối xứng không?

3 tầng kiểm chứng trên CÙNG tập test block-split:
  T1. Phân bố 5 đặc trưng bất đối xứng y khoa theo lớp (effect size Cohen's d)
      → nếu 2 lớp chồng nhau hoàn toàn thì KHÔNG mô hình nào học được
  T2. Logistic Regression (class-weight) trên 5 features → AUC
  T3. MLP nhỏ (class-weight) trên 5 features → AUC
(so sánh: MLP 936 tọa độ thô = AUC 0.555; rules tuyến tính = 0.638)

Output: test_results/face_signal_test_<ts>.json + csv per-image
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from detection.face_module_v7 import FaceAsymmetryDetector  # noqa: E402

DATASET_PATH = (r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\data"
                r"\datasets\face\Annotated stroke and non stroke Dataset")
OUT_DIR = os.path.join(parent_dir, 'test_results')
SEED, BLOCK = 42, 50
FEATS = ['mouth_ratio', 'eye_ratio', 'face_tilt', 'nasolabial_ratio',
         'forehead_ratio']


def scan():
    rows = []
    for sub, lab in [('NonStroke', 0), ('Stroke', 1)]:
        p = os.path.join(DATASET_PATH, sub)
        for t in sorted(os.listdir(p)):
            if t.endswith('.txt'):
                ip = os.path.join(p, t.replace('.txt', '.jpg'))
                if os.path.exists(ip):
                    try:
                        with open(os.path.join(p, t)) as f:
                            bbox = list(map(float, f.readline().split()[1:5]))
                    except Exception:
                        bbox = None
                    rows.append({'image_path': ip, 'label': lab, 'bbox': bbox,
                                 'block_id': f"{lab}_{int(t.split('_')[1].split('.')[0]) // BLOCK}"})
    return pd.DataFrame(rows)


def crop_face(img, bbox, margin=0.15):
    if img is None or bbox is None:
        return img
    H, W = img.shape[:2]
    cx, cy, w, h = bbox
    x1 = int(max(0, (cx - w / 2) * W - margin * W))
    y1 = int(max(0, (cy - h / 2) * H - margin * H))
    x2 = int(min(W, (cx + w / 2) * W + margin * W))
    y2 = int(min(H, (cy + h / 2) * H + margin * H))
    return img[y1:y2, x1:x2] if (x2 - x1) > 20 and (y2 - y1) > 20 else img


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s) if s > 1e-9 else 0.0


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    df = scan()
    blocks = df.drop_duplicates('block_id')[['block_id', 'label']]
    _, temp_b = train_test_split(blocks, test_size=0.4, random_state=SEED,
                                 stratify=blocks['label'])
    _, test_b = train_test_split(temp_b, test_size=0.5, random_state=SEED,
                                 stratify=temp_b['label'])
    test_df = df[df['block_id'].isin(test_b['block_id'])].reset_index(drop=True)
    print(f"Tập TEST: {len(test_df)} ảnh (stroke {(test_df.label==1).sum()}, "
          f"non {(test_df.label==0).sum()})")

    det = FaceAsymmetryDetector(history_size=1)
    rows = []
    for k, (_, r) in enumerate(test_df.iterrows()):
        img = cv2.imread(r['image_path'])
        if img is None:
            continue
        res = det.process_frame(crop_face(img, r['bbox']),
                                frame_timestamp_ms=1000 + k * 33)
        st = res.get('status')
        if st in ('NO_FACE', 'ERROR', 'NO_DETECTOR', 'INVALID_LANDMARKS'):
            rows.append({'label': r['label'], 'valid': False,
                         'block_id': r['block_id'],
                         **{f: np.nan for f in FEATS}})
        else:
            rows.append({'label': r['label'], 'valid': True,
                         'block_id': r['block_id'],
                         **{f: float(res['raw_metrics'].get(f, np.nan))
                            for f in FEATS}})
    out = pd.DataFrame(rows)
    v = out[out['valid'] & out[FEATS].notna().all(axis=1)].copy()
    print(f"Hợp lệ: {len(v)} (stroke {(v.label==1).sum()}, "
          f"non {(v.label==0).sum()})")

    # ----- T1: Effect size từng feature -----
    t1 = {}
    print("\n----- T1. COHEN'S d (stroke vs non) -----")
    print(f"{'feature':18s} {'mean_stroke':>11s} {'mean_non':>9s} "
          f"{'Cohen_d':>8s}  diễn giải")
    for f in FEATS:
        a = v[v.label == 1][f]
        b = v[v.label == 0][f]
        d = cohen_d(a, b)
        t1[f] = {'mean_stroke': round(float(a.mean()), 3),
                 'mean_nonstroke': round(float(b.mean()), 3),
                 'cohens_d': round(d, 3)}
        interp = ('KHÔNG có tín hiệu' if abs(d) < 0.2 else
                  'tín hiệu yếu' if abs(d) < 0.5 else
                  'tín hiệu trung bình' if abs(d) < 0.8 else 'tín hiệu mạnh')
        print(f"{f:18s} {a.mean():11.3f} {b.mean():9.3f} "
              f"{d:8.3f}  {interp}")

    # ----- T2: Logistic (class-weight) — GroupKFold THEO BLOCK (trung thực) -----
    from sklearn.model_selection import GroupKFold
    X = v[FEATS].values
    y = v['label'].values
    g = v['block_id'].values

    def group_cv_auc(make_model, predict_fn):
        gkf = GroupKFold(n_splits=5)
        p = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups=g):
            m = make_model()
            m.fit(X[tr], y[tr])
            p[te] = predict_fn(m, X[te])
        return float(roc_auc_score(y, p))

    auc_lr_cv = group_cv_auc(
        lambda: LogisticRegression(class_weight='balanced', max_iter=1000),
        lambda m, Xv: m.predict_proba(Xv)[:, 1])
    print(f"\n----- T2. Logistic 5 features — GroupKFold(5) theo BLOCK: "
          f"AUC = {auc_lr_cv:.3f} (đánh giá ngoài fold) -----")

    # Same-data (chỉ để tham khảo mức tối đa)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(Xs, y)
    auc_lr = float(roc_auc_score(y, lr.predict_proba(Xs)[:, 1]))
    print(f"      (same-data tham khảo: AUC = {auc_lr:.3f})")

    # ----- T3: MLP nhỏ class-weight — GroupKFold THEO BLOCK -----
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit_mlp(Xtr, ytr):
        torch.manual_seed(SEED)
        m = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(32, 16), nn.ReLU(),
                          nn.Linear(16, 1)).to(dev)
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
        pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                          dtype=torch.float32, device=dev)
        lf = nn.BCEWithLogitsLoss(pos_weight=pw)
        op = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        m.train()
        for _ in range(300):
            op.zero_grad()
            loss = lf(m(Xt).squeeze(-1), yt)
            loss.backward()
            op.step()
        m.eval()
        return m

    def predict_mlp(m, Xv):
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(
                Xv, dtype=torch.float32, device=dev)).squeeze(-1)).cpu().numpy()

    gkf = GroupKFold(n_splits=5)
    p_mlp = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups=g):
        m = fit_mlp(X[tr], y[tr])
        p_mlp[te] = predict_mlp(m, X[te])
    auc_mlp_cv = float(roc_auc_score(y, p_mlp))
    print(f"----- T3. MLP 5 features — GroupKFold(5) theo BLOCK: "
          f"AUC = {auc_mlp_cv:.3f} -----")
    print("\nTHAM CHIẾU cùng dữ liệu: rules tuyến tính AUC 0.638 | "
          "MLP 936 tọa độ thô AUC 0.555")

    verdict = ('Dataset CÓ tín hiệu đủ mạnh cho AI học'
               if max(auc_lr_cv, auc_mlp_cv) >= 0.75 else
               'Dataset RẤT YẾU tín hiệu — AI không thể học phân biệt ở mức '
               'landmark tĩnh; cần dữ liệu chủ động (cười/nhíu mày) theo '
               'quy trình NIHSS item 4')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'purpose': 'Kiểm chứng dataset face có chứa tín hiệu phân biệt không',
        'created': datetime.now().isoformat(),
        'n_test': len(out), 'n_valid': len(v),
        'n_stroke_valid': int((v.label == 1).sum()),
        'n_nonstroke_valid': int((v.label == 0).sum()),
        'T1_effect_sizes': t1,
        'T2_logistic_auc_groupkfold_block': round(auc_lr_cv, 3),
        'T3_mlp_auc_groupkfold_block': round(auc_mlp_cv, 3),
        'T2_logistic_auc_same_data_ref': round(auc_lr, 3),
        'reference_rules_auc_test_split': 0.638,
        'reference_raw_mlp_auc_test_split': 0.555,
        'verdict': verdict,
    }
    jp = os.path.join(OUT_DIR, f'face_signal_test_{ts}.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    v.to_csv(os.path.join(OUT_DIR, f'face_signal_test_{ts}.csv'), index=False)
    print(f"\nKẾT LUẬN: {verdict}\nJSON: {jp}")


if __name__ == '__main__':
    main()
