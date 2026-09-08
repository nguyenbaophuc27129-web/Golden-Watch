# -*- coding: utf-8 -*-
"""
FACE ML 5 ĐẶC TRƯNG (M1-07) — Logistic Regression từ artifact JSON.

Artifact: models/face_asym_v2_5feat_<ts>.json (tạo bởi
training/evaluate_all_metrics.py, GroupKFold-5 theo block, AUC OOF 0.845).

QUAN TRỌNG (SYS-15): model CHƯA qua protocol B (100 tình huống người khỏe)
→ chỉ bật trong app qua feature-flag, mặc định TẮT. Khi bật, xác suất ML
thay thế prob của rules; prob rules giữ lại trong metrics để so sánh.
"""

import os
import glob
import json
import math


class FaceML5Feat:
    """Wrapper Logistic 5 đặc trưng bất đối xứng mặt (artifact JSON)."""

    def __init__(self, artifact):
        self.features = artifact['features']
        self.mean = artifact['scaler_mean']
        self.scale = artifact['scaler_scale']
        self.coef = artifact['coef']
        self.intercept = artifact['intercept']
        self.created = artifact.get('created', '?')
        self.auc_oof = artifact.get('auc_oof')

    @classmethod
    def load_latest(cls, models_dir):
        arts = sorted(glob.glob(os.path.join(
            models_dir, 'face_asym_v2_5feat_*.json')))
        if not arts:
            return None
        try:
            with open(arts[-1], encoding='utf-8') as f:
                return cls(json.load(f))
        except Exception:
            return None

    def predict(self, raw_metrics):
        """raw_metrics: dict từ face_module_v7 → xác suất 0-100 hoặc None."""
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
