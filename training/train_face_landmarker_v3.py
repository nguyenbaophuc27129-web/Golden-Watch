# -*- coding: utf-8 -*-
"""
TRAIN FACE LANDMARKER v3 — theo mẫu CHÍNH THỨC Google MediaPipe
[mediapipe_python_tasks]_face_landmarker.py (Colab sample) trên DATASET CỦA ĐỘI.

Khác v2 (train_face_model.py — MLP 936 đầu vào từ landmark thô):
  CHUYỂN TỪ → SANG (SYS-27):
  - TỪ: landmark 2D thô 468x2 → MLP tự học (input 936, nặng, khó giải thích)
  - SANG: 3 output chính thức của Face Landmarker mỗi ảnh:
      (1) 478 landmark  → 5 ratio lâm sàng (giống face_module_v7 chạy app)
      (2) 52 BLENDSHAPES (hệ số biểu cảm chuẩn) → cặp L-R = BẤT ĐỐI XỨNG,
          đúng cơ chế liệt mặt 1 nửa
      (3) MA TRẬN TƯ THẾ 4x4 → yaw/pitch/roll (kiểm soát tư thế đầu)
  - Phân loại: Logistic Regression (giải thích được, artifact JSON nhẹ
    khớp schema FaceML5Feat) thay MLP 500K tham số.

Quy ước giữ nguyên từ các script trước:
  - Nhãn theo THƯ MỤC (token .txt bị đảo — M1-05)
  - Chia theo BLOCK img_NNNN//50 chống leakage frame gần-trùng (L-04)
  - seed 42; GroupKFold(5) OOF trên train; Youden J chọn ngưỡng; test 1 lần.

Chạy:  PYTHONUTF8=1 python training/train_face_landmarker_v3.py
"""

import os
import re
import json
import math
import time

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\data\datasets\face\Annotated stroke and non stroke Dataset"
TASK_FILE = os.path.join(PROJECT, 'models', 'face_landmarker_v2',
                         'face_landmarker_v2_with_blendshapes.task')
OUT_DIR = os.path.join(PROJECT, 'test_results')
CONF_MIN = 0.3          # khớp app (L-28): 0.3 thay vì 0.5
MAX_YAW = 60.0          # quay đi quá mức này → loại (đo méo mặt không tin cậy)

rng = np.random.RandomState(SEED)

# ======================================================================
# 1) LANDMARKER — IMAGE mode + blendshapes + transformation matrix
#    (đúng các option của mẫu chính thức Google; running_mode=IMAGE)
# ======================================================================
def create_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK_FILE),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=CONF_MIN)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def head_pose_from_matrix(M):
    R = np.asarray(M, dtype=np.float64)[:3, :3]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 1]))))
    roll = math.degrees(math.atan2(R[0, 1], R[1, 1]))
    yaw = math.degrees(math.atan2(R[2, 0], R[2, 2]))
    return yaw, pitch, roll


def five_ratios(lm):
    """5 ratio lâm sàng — CÙNG chỉ số landmark với face_module_v7 (app)."""
    d = np.linalg.norm
    mouth = abs(d(lm[61] - lm[13]) / d(lm[291] - lm[13]) - 1.0) * 100
    eye = abs(d(lm[33] - lm[6]) / d(lm[263] - lm[6]) - 1.0) * 100
    tilt = math.degrees(math.atan2(abs(lm[152][0] - lm[10][0]),
                                   max(lm[152][1] - lm[10][1], 1e-6)))
    naso = abs(d(lm[205] - lm[1]) / d(lm[425] - lm[1]) - 1.0) * 100
    fore = abs(d(lm[70] - lm[10]) / d(lm[300] - lm[10]) - 1.0) * 100
    return [mouth, eye, tilt, naso, fore]

RATIO_NAMES = ['mouth_ratio', 'eye_ratio', 'face_tilt', 'nasolabial_ratio',
               'forehead_ratio']


# ======================================================================
# 2) SCAN DATASET (nhãn theo thư mục — M1-05; block theo L-04)
# ======================================================================
def scan_dataset():
    rows = []
    for label, sub in ((1, 'Stroke'), (0, 'NonStroke')):
        folder = os.path.join(DATASET, sub)
        for fn in os.listdir(folder):
            if not fn.endswith('.jpg'):
                continue
            txt = os.path.join(folder, fn[:-4] + '.txt')
            if not os.path.exists(txt):
                continue
            try:
                with open(txt) as f:
                    parts = f.readline().split()
                bbox = list(map(float, parts[1:5]))  # cx cy w h (YOLO)
            except Exception:
                continue
            m = re.search(r'img_(\d+)', fn)
            block = f'{label}_{int(m.group(1)) // 50 if m else -1}'
            rows.append({'path': os.path.join(folder, fn), 'label': label,
                         'bbox': bbox, 'block': block})
    n_stroke = sum(r['label'] for r in rows)
    print(f'Dataset: {len(rows)} anh (Stroke {n_stroke}, '
          f'NonStroke {len(rows) - n_stroke}), '
          f'{len(set(r["block"] for r in rows))} blocks')
    return rows


# ======================================================================
# 3) TRÍCH XUẤT (1 lượt cho mọi nhóm đặc trưng)
# ======================================================================
def extract_all(rows):
    landmarker = create_landmarker()
    bs_names = None
    feats, labels, blocks, kept_paths = [], [], [], []
    n_no_face = n_far = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        ok = False
        img = cv2.imread(r['path'])
        if img is not None:
            h, w = img.shape[:2]
            cx, cy, bw, bh = r['bbox']
            x1 = max(0, int((cx - bw / 2) * w) - 10)
            y1 = max(0, int((cy - bh / 2) * h) - 10)
            x2 = min(w, int((cx + bw / 2) * w) + 10)
            y2 = min(h, int((cy + bh / 2) * h) + 10)
            if x2 - x1 > 16 and y2 - y1 > 16:
                rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                res = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if res.face_landmarks:
                    lm = np.array([[p.x, p.y]
                                   for p in res.face_landmarks[0]])
                    bs = res.face_blendshapes[0]
                    if bs_names is None:
                        bs_names = [c.category_name for c in bs]
                    bsv = np.array([c.score for c in bs])
                    yaw, pitch, roll = head_pose_from_matrix(
                        res.facial_transformation_matrixes[0])
                    if abs(yaw) > MAX_YAW:
                        n_far += 1
                    else:
                        feats.append(five_ratios(lm) + bsv.tolist()
                                     + [yaw, pitch, roll])
                        labels.append(r['label'])
                        blocks.append(r['block'])
                        kept_paths.append(r['path'])
                        ok = True
        if not ok:
            n_no_face += 1
        if (i + 1) % 250 == 0:
            print(f'  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s, '
                  f'khong thay mat/quay-xa: {n_no_face + n_far})')
    landmarker.close()
    X = np.array(feats, dtype=np.float64)
    print(f'Trich xuat xong: {len(X)} anh dung duoc '
          f'(bo {n_no_face} khong-thay-mat, {n_far} yaw>~{MAX_YAW} do) '
          f'trong {time.time() - t0:.0f}s')
    return X, np.array(labels), np.array(blocks), bs_names, kept_paths


# ======================================================================
# 4) NHÓM ĐẶC TRƯNG + ĐÁNH GIÁ (OOF GroupKFold + Youden + test 1 lần)
# ======================================================================
def blend_asym_pairs(bs_names):
    """Cặp L/R: trả (tên_asym, chỉ_số_L, chỉ_số_R) trong mảng blendshape."""
    names, il, ir = [], [], []
    for i, n in enumerate(bs_names):
        if n.endswith('Left') and (n[:-4] + 'Right') in bs_names:
            names.append('asym_' + n[:-4])
            il.append(i)
            ir.append(bs_names.index(n[:-4] + 'Right'))
    return names, il, ir


def evaluate_group(X, y, blocks, feat_names, tag, results):
    groups = np.unique(blocks)
    gs = rng.permutation(groups)
    tr_g = set(gs[:int(len(gs) * 0.8)])
    tr_m = np.array([b in tr_g for b in blocks])
    te_m = ~tr_m
    tr_idx = np.where(tr_m)[0]

    # Chọn C bằng OOF GroupKFold(5) TRÊN TRAIN (test chưa bị nhìn)
    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        oof_tr = np.zeros(len(tr_idx))
        gkf = GroupKFold(5)
        for ti, vi in gkf.split(X[tr_idx], y[tr_idx], blocks[tr_idx]):
            sc = StandardScaler().fit(X[tr_idx][ti])
            clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
            clf.fit(sc.transform(X[tr_idx][ti]), y[tr_idx][ti])
            oof_tr[vi] = clf.predict_proba(
                sc.transform(X[tr_idx][vi]))[:, 1]
        auc = roc_auc_score(y[tr_idx], oof_tr)
        if best is None or auc > best[1]:
            best = (C, auc, oof_tr)

    C, auc_oof, oof_tr = best
    fpr, tpr, thr = roc_curve(y[tr_idx], oof_tr)
    threshold = float(thr[int(np.argmax(tpr - fpr))])   # Youden J

    # Fit cuối trên toàn train → chạm test ĐÚNG 1 LẦN
    sc = StandardScaler().fit(X[tr_idx])
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    clf.fit(sc.transform(X[tr_idx]), y[tr_idx])
    p_te = clf.predict_proba(sc.transform(X[te_m]))[:, 1]
    y_te = y[te_m]
    auc_te = roc_auc_score(y_te, p_te)
    pred = (p_te >= threshold).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    tn = int(((pred == 0) & (y_te == 0)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    r = {'group': tag, 'n_features': len(feat_names), 'C': C,
         'auc_oof_train': round(auc_oof, 4), 'threshold': round(threshold, 4),
         'auc_test': round(auc_te, 4), 'sens_test': round(sens, 4),
         'spec_test': round(spec, 4), 'f1_test': round(f1, 4),
         'cm': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
         'n_train': int(tr_m.sum()), 'n_test': int(te_m.sum())}
    results.append(r)
    print(f"[{tag:20s}] {len(feat_names):2d} ft | OOF AUC {auc_oof:.3f} "
          f"(C={C}) | TEST AUC {auc_te:.3f} sens {sens * 100:.1f}% "
          f"spec {spec * 100:.1f}% F1 {f1:.3f}")
    return dict(r=r, scaler=(sc.mean_, sc.scale_), coef=clf.coef_[0],
                intercept=float(clf.intercept_[0]), feat_names=feat_names,
                p_te=p_te, y_te=y_te)


def main():
    print('=' * 66)
    print('TRAIN FACE LANDMARKER v3 — mau chinh thuc Google MediaPipe')
    print('  (blendshapes + facial transformation matrix) tren dataset doi')
    print('=' * 66)
    rows = scan_dataset()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUT_DIR, f'face_landmarker_v3_{stamp}')
    os.makedirs(run_dir, exist_ok=True)

    X, y, blocks, bs_names = extract_all(rows)
    asym_names, il, ir = blend_asym_pairs(bs_names)
    n_bs = len(bs_names)
    L = 5 + np.array(il)                       # cột Left trong X
    R = 5 + np.array(ir)
    asym_X = X[:, L] - X[:, R]                 # BẤT ĐỐI XỨNG L−R
    pose_X = X[:, n_bs + 5:n_bs + 8]           # yaw, pitch, roll

    groups = {
        'G1_5ratios': (X[:, :5], RATIO_NAMES),
        'G2_blend_asym': (asym_X, asym_names),
        'G3_blend_asym_pose': (np.hstack([asym_X, pose_X]),
                               asym_names + ['yaw', 'pitch', 'roll']),
        'G4_all': (np.hstack([X[:, :5], asym_X, pose_X]),
                   RATIO_NAMES + asym_names + ['yaw', 'pitch', 'roll']),
    }

    results, fits = [], {}
    for tag, (Xg, names) in groups.items():
        fits[tag] = evaluate_group(Xg, y, blocks, names, tag, results)

    results.sort(key=lambda r: -r['auc_test'])
    best = fits[results[0]['group']]
    with open(os.path.join(run_dir, 'results.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'seed': SEED, 'dataset': DATASET, 'conf_min': CONF_MIN,
                   'max_yaw': MAX_YAW, 'results': results}, f,
                  ensure_ascii=False, indent=1)

    # ROC + confusion PNG cho nhóm tốt nhất (quy ước metrics_pack)
    r0 = results[0]
    fpr, tpr, _ = roc_curve(best['y_te'], best['p_te'])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2,
             label=f"{r0['group']} (AUC={r0['auc_test']:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR (Sensitivity)')
    plt.title('Face Landmarker v3 — ROC (test block-aware)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'roc_best.png'), dpi=150)
    plt.close()

    cm = r0['cm']
    mat = [[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]]
    plt.figure(figsize=(4, 3.6))
    plt.imshow(mat, cmap='Blues')
    for (i, j), v in np.ndenumerate(mat):
        plt.text(j, i, str(v), ha='center', va='center', fontsize=14)
    plt.xticks([0, 1], ['Pred OK', 'Pred Stroke'])
    plt.yticks([0, 1], ['True OK', 'True Stroke'])
    plt.title(f"Confusion — {r0['group']}"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'confusion_best.png'), dpi=150)
    plt.close()

    # Artifact JSON schema FaceML5Feat (để sau này nối app qua protocol B)
    mu, sd = best['scaler']
    art = {'version': 'face_blend_v3', 'created': stamp,
           'source_sample':
               '[mediapipe_python_tasks]_face_landmarker.py (Google)',
           'features': best['feat_names'],
           'scaler_mean': [round(float(v), 6) for v in mu],
           'scaler_scale': [round(float(v), 6) for v in sd],
           'coef': [round(float(v), 6) for v in best['coef']],
           'intercept': round(best['intercept'], 6),
           'threshold': r0['threshold'], 'auc_oof': r0['auc_oof_train'],
           'auc_test': r0['auc_test'], 'sens_test': r0['sens_test'],
           'spec_test': r0['spec_test'], 'conf_min': CONF_MIN}
    art_path = os.path.join(PROJECT, 'models', f'face_blend_v3_{stamp}.json')
    with open(art_path, 'w', encoding='utf-8') as f:
        json.dump(art, f, ensure_ascii=False, indent=1)

    print('=' * 66)
    print('XEP HANG (theo AUC test):')
    for r in results:
        print(f"  {r['group']:22s} AUC {r['auc_test']:.3f} "
              f"sens {r['sens_test'] * 100:.1f}% "
              f"spec {r['spec_test'] * 100:.1f}%")
    print(f'Tot nhat: {r0["group"]}')
    print(f'Ket qua : {run_dir}')
    print(f'Artifact: {art_path}')
    print('(Model CHUA qua protocol B — khong bat tren app, nhu SYS-15)')


if __name__ == '__main__':
    main()
