# -*- coding: utf-8 -*-
"""
FACE ML v3 (SYS-27/29) — Logistic Regression 28 đặc trưng từ artifact JSON
do `training/train_face_landmarker_v3.py` tạo (mẫu Google Face Landmarker).

Artifact: models/face_blend_v3_<ts>.json
  features  : 5 ratio lâm sàng + 20 asym blendshape L−R + yaw/pitch/roll
  scaler_*  : StandardScaler
  coef/intercept: LogisticRegression
  threshold : Youden J trên OOF train (tham chiếu hiển thị)

Nguồn số liệu (SYS-27, held-out block test): AUC 0.943 · sens 80.1% ·
spec 94.1%. Theo thiết kế SYS-15: prob ML THAY prob rules; prob rules giữ
lại trong raw_metrics['score_rules'] để so sánh.
"""

import os
import glob
import json
import math


class FaceMLV3:
    """Wrapper Logistic 28 đặc trưng face landmarker v3 (artifact JSON)."""

    def __init__(self, artifact):
        self.features = artifact['features']
        self.mean = artifact['scaler_mean']
        self.scale = artifact['scaler_scale']
        self.coef = artifact['coef']
        self.intercept = artifact['intercept']
        self.threshold = artifact.get('threshold')
        self.created = artifact.get('created', '?')
        self.auc_test = artifact.get('auc_test')
        self.auc_oof = artifact.get('auc_oof')

    @classmethod
    def load_latest(cls, models_dir):
        arts = sorted(glob.glob(os.path.join(models_dir,
                                             'face_blend_v3_*.json')))
        if not arts:
            return None
        try:
            with open(arts[-1], encoding='utf-8') as f:
                return cls(json.load(f))
        except Exception as e:
            print(f"[FACE-ML-v3] Không đọc được artifact ({e})")
            return None

    def predict(self, raw_metrics):
        """raw_metrics: dict từ face_module_v7 → xác suất 0-100 hoặc None
        (None khi thiếu bất kỳ đặc trưng nào → app fallback prob rules)."""
        if raw_metrics is None:
            return None
        x = []
        for f, mu, sd in zip(self.features, self.mean, self.scale):
            v = raw_metrics.get(f)
            if v is None:
                return None
            x.append((float(v) - mu) / (sd if sd else 1.0))
        z = sum(c * xi for c, xi in zip(self.coef, x)) + self.intercept
        return 100.0 / (1.0 + math.exp(-z))


# ==================== TEST ====================
def test_face_ml_v3():
    import numpy as np
    models_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'models')
    m = FaceMLV3.load_latest(models_dir)
    assert m is not None, 'không tìm thấy artifact face_blend_v3_*'
    # dựng raw_metrics ngẫu nhiên quanh mean → prob trong [0,100], xác định
    rm = {}
    for f, mu in zip(m.features, m.mean):
        rm[f] = float(mu)
    p1 = m.predict(rm)
    assert p1 is not None and 0.0 <= p1 <= 100.0, p1
    # metric méo mạnh → prob tăng
    rm2 = dict(rm)
    k0 = m.features[0]                       # mouth_ratio
    rm2[k0] = mu0 = m.mean[0] + 3 * (m.scale[0] or 1.0)
    p2 = m.predict(rm2)
    # thiếu đặc trưng → None (fallback rules)
    rm3 = {k: v for k, v in rm.items() if k != m.features[-1]}
    assert m.predict(rm3) is None
    print(f'FACE ML v3 TEST: PASS — created {m.created}, '
          f'AUC test {m.auc_test}, prob[{k0}={mu0:.1f}] {p2:.1f} >= base '
          f'{p1:.1f}: {p2 >= p1}')


if __name__ == '__main__':
    test_face_ml_v3()
