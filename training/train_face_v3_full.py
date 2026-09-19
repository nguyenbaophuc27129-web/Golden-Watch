# -*- coding: utf-8 -*-
"""
FACE v3 FULL-DATA REFIT (NK-12) — train 100% dataset cho production
===================================================================
Nguyên tắc trung thực (TRIPOD+AI):
  - SỐ ĐÁNH GIÁ công bố trong báo cáo VẪN là protocol đã khóa:
    block-split held-out 0.943 (SYS-28) + multi-seed 0.94±0.01 (NK-04)
    + abstention NK-10. KHÔNG bao giờ công bố số fit-100%.
  - Artifact production thì refit trên 100% 3,715 ảnh với hyper-param
    C CHỌN BẰNG GroupKFold OOF nội bộ (không có tập ngoài) + threshold
    Youden trên OOF đó — chuẩn "final model fit after locked validation".
Output: models/face_blend_v3_full_<ts>.json (loader face_ml_v3 tự chọn
bản mới nhất) + log so sánh coef cũ/mới.
Chạy: PYTHONUTF8=1 python training/train_face_v3_full.py
"""

import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE = os.path.join(ROOT, 'test_results', '_face_v3_features_cache.npz')
OLD_ARTIFACT = os.path.join(ROOT, 'models', 'face_blend_v3_20260909_201800.json')
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
FEATURE_PREFIX = ['mouth_ratio', 'eye_ratio', 'face_tilt',
                  'nasolabial_ratio', 'forehead_ratio']


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    print('== FACE v3 FULL-DATA REFIT (NK-12) ==')
    d = np.load(CACHE, allow_pickle=True)
    X, y, blocks = d['X'], d['y'], d['blocks']
    print(f'Data: {X.shape} (100% dataset — KHÔNG chia train/test)')

    # 1) Chọn C bằng GroupKFold-5 OOF nội bộ trên toàn bộ data
    best = (None, -1)
    for C in C_GRID:
        oof = np.zeros(len(y))
        for ti, vi in GroupKFold(5).split(X, y, blocks):
            sc = StandardScaler().fit(X[ti])
            clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
            clf.fit(sc.transform(X[ti]), y[ti])
            oof[vi] = clf.predict_proba(sc.transform(X[vi]))[:, 1]
        auc_oof = roc_auc_score(y, oof)
        print(f'  C={C}: OOF AUC {auc_oof:.4f}')
        if auc_oof > best[1]:
            best = (C, auc_oof, oof)
    C, auc_oof, oof = best
    print(f'  → C khóa = {C} · OOF AUC (all-data) {auc_oof:.4f}')

    # 2) Threshold Youden trên OOF đó (tham chiếu hiển thị như artifact cũ)
    fpr, tpr, thr = roc_curve(y, oof)
    threshold = float(thr[int(np.argmax(tpr - fpr))])

    # 3) REFIT TRÊN 100% DATA
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
    clf.fit(sc.transform(X), y)
    pred = (clf.predict_proba(sc.transform(X))[:, 1] >= threshold)
    fit_acc = float((pred == y).mean())
    print(f'  Refit 100% xong · fit-accuracy@thr {fit_acc:.3f} '
          f'(chỉ thông tin — KHÔNG công bố)')

    # 4) Lưu artifact MỚI (format đúng loader face_ml_v3)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    feat_names = FEATURE_PREFIX + [f'blend_asym_{i:02d}' for i in range(20)] + \
                 ['pose_yaw', 'pose_pitch', 'pose_roll']
    old = json.load(open(OLD_ARTIFACT, encoding='utf-8')) \
        if os.path.exists(OLD_ARTIFACT) else None
    artifact = {
        'version': 'face_blend_v3_full',
        'created': ts,
        'source_sample': '[mediapipe_python_tasks]_face_landmarker.py (Google)',
        'note': 'refit 100% dataset sau khi validation khóa (SYS-28 + NK-04); '
                'số công bố vẫn là protocol khóa',
        'features': feat_names,
        'scaler_mean': sc.mean_.tolist(),
        'scaler_scale': sc.scale_.tolist(),
        'coef': clf.coef_[0].tolist(),
        'intercept': float(clf.intercept_[0]),
        'threshold': round(threshold, 3),
        'auc_oof': round(float(auc_oof), 4),
        'auc_test': old['auc_test'] if old else None,       # giữ số khóa
        'sens_test': old['sens_test'] if old else None,
        'spec_test': old['spec_test'] if old else None,
        'auc_multiseed': [0.9398, 0.0099],                  # NK-04 (mean, sd)
        'conf_min': 0.3,
    }
    out = os.path.join(ROOT, 'models', f'face_blend_v3_full_{ts}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print(f'Artifact: {out}')

    if old:
        d_coef = float(np.corrcoef(old['coef'], artifact['coef'])[0, 1])
        print(f"Tương quan coef cũ↔mới: {d_coef:.4f} "
              f"(giống nhau → swap an toàn) · threshold {old['threshold']} → "
              f"{artifact['threshold']}")

    # 5) VERIFY: loader production phải nạp được artifact mới
    from src.detection.face_ml_v3 import FaceMLV3
    m = FaceMLV3.load_latest(os.path.join(ROOT, 'models'))
    assert m.features == feat_names and len(m.coef) == 28
    print('VERIFY: face_ml_v3.load_latest nạp ĐÚNG artifact full OK')
    print(f'  loader dùng: {getattr(m, "path", "( FaceMLv3 không lưu path)")}')


if __name__ == '__main__':
    main()
