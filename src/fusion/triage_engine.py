# -*- coding: utf-8 -*-
"""
TRIAGE ENGINE - PSCS v8.0 (Golden-Watch)
Phân loại sơ bộ: SUBTYPE (nghi ngờ), SEVERITY, BỆNH VIỀN đề xuất.

KHÔNG PHẢN CHẨN ĐOÁN — rule-based hint theo TONG_QUAN v6.0:
  Base: 80% Ischemic / 20% Hemorrhagic (tỷ lệ thực tế dân số)
  +30% Hemorrhagic nếu đau đầu dữ dội đột ngột (sentinel headache)
  +20% Hemorrhagic nếu nôn
  +20% Ischemic nếu tiến triển từ từ
  +15% Ischemic nếu nói khó là dấu hiệu nổi trội (speech-dominant)
  → clamp 5-95%, xuất "Nghi ischemic (65%)"

SEVERITY (4-item subtotal 0-13): Mild 0-5, Moderate 6-12, Severe 13
  (ngưỡng NIHSS chuẩn: Mild 0-5, Moderate 6-13(15), Severe 14+ — quy đổi
   về subtotal 13: Severe >= 11 ≈ NIHSS 14/15 → dùng 11)
  Trend WORSENING → nâng 1 bậc.

BỆNH VIỆN: data/hospital_database.csv
  Severe        → có stroke unit + thrombolysis
  Nghi xuất huyết → có phẫu thuật thần kinh
  Nghi thiếu máu → có thrombolysis (tPA)
  Mild          → gần nhất (km)

Tác giả: PSCS Team
Ngày: 06/09/2026
"""

import os
import sys
import csv

HOSPITAL_DB = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'data', 'hospital_database.csv')


class TriageEngine:

    def __init__(self, hospital_csv=HOSPITAL_DB):
        self.hospital_csv = hospital_csv

    # ------------------------------------------------------------------
    # 1. SUBTYPE HINT (rule-based, KHÔNG phải chẩn đoán)
    # ------------------------------------------------------------------
    def estimate_subtype(self, symptoms=None, module_results=None):
        """
        Args:
            symptoms: dictvd {'sudden_severe_headache': True, 'vomiting': False,
                              'gradual_progression': False}
            module_results: dict module (xét speech-dominant)
        Returns:
            {'subtype': 'ischemic'|'hemorrhagic', 'confidence': %, 'reasons': []}
        """
        symptoms = symptoms or {}
        hem = 20.0   # base 20% hemorrhagic
        reasons = []

        if symptoms.get('sudden_severe_headache'):
            hem += 30
            reasons.append('dau dau du doi dot ngot (+30% hem)')
        if symptoms.get('vomiting'):
            hem += 20
            reasons.append('non (+20% hem)')
        if symptoms.get('gradual_progression'):
            hem -= 20
            reasons.append('tien trien tu tu (+20% ischemic)')
        if symptoms.get('seizure_at_onset'):
            hem += 10
            reasons.append('dong kinh khi phat benh (+10% hem)')

        # Speech-dominant: speech cao nhất trong các module
        module_results = module_results or {}
        probs = {}
        for k, pk in (('face', 'score'), ('speech', 'speech_prob'),
                      ('arm', 'arm_prob'), ('gait', 'gait_prob')):
            r = module_results.get(k)
            if r and r.get(pk) is not None:
                probs[k] = r[pk]
        if probs and max(probs, key=probs.get) == 'speech':
            hem -= 15
            reasons.append('noi kho la dau hieu noi troi (+15% ischemic)')

        hem = max(5.0, min(95.0, hem))
        subtype = 'hemorrhagic' if hem >= 50 else 'ischemic'
        conf = round(hem if subtype == 'hemorrhagic' else 100 - hem, 1)
        return {'subtype': subtype, 'confidence': conf, 'reasons': reasons}

    # ------------------------------------------------------------------
    # 2. SEVERITY
    # ------------------------------------------------------------------
    def classify_severity(self, nihss_total, trend='STABLE'):
        """Mild 0-5 | Moderate 6-10 | Severe >= 11 (subtotal /13)."""
        if nihss_total >= 11:
            sev = 'SEVERE'
        elif nihss_total >= 6:
            sev = 'MODERATE'
        else:
            sev = 'MILD'
        # Trend xấu đi → nâng 1 bậc (Mild->Moderate->Severe)
        order = ['MILD', 'MODERATE', 'SEVERE']
        if trend == 'WORSENING' and sev != 'SEVERE':
            sev = order[order.index(sev) + 1]
        return sev

    # ------------------------------------------------------------------
    # 3. BỆNH VIỆN ĐỀ XUẤT
    # ------------------------------------------------------------------
    def load_hospitals(self):
        if not os.path.exists(self.hospital_csv):
            return []
        with open(self.hospital_csv, encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def recommend_hospital(self, severity, subtype, hospitals=None):
        """
        Severe → stroke unit + thrombolysis
        Hemorrhagic → phẫu thuật thần kinh
        Ischemic → thrombolysis
        Mild → gần nhất
        """
        hospitals = hospitals if hospitals is not None else self.load_hospitals()
        if not hospitals:
            return None

        def _b(h, key):
            return str(h.get(key, '')).strip().lower() in ('1', 'true', 'yes', 'co')

        if severity == 'SEVERE':
            pool = [h for h in hospitals
                    if _b(h, 'stroke_unit') and _b(h, 'thrombolysis')]
        elif subtype == 'hemorrhagic':
            pool = [h for h in hospitals if _b(h, 'neurosurgery')]
        elif subtype == 'ischemic':
            pool = [h for h in hospitals if _b(h, 'thrombolysis')]
        else:
            pool = hospitals  # Mild: gần nhất
        if not pool:
            pool = hospitals
        pool = sorted(pool, key=lambda h: float(h.get('distance_km') or 999))
        return pool[0]

    # ------------------------------------------------------------------
    # 4. TRIAGE HOÀN CHỈNH
    # ------------------------------------------------------------------
    def triage(self, fusion_result, nihss_total, symptoms=None,
               module_results=None):
        """Full pipeline: subtype + severity + hospital."""
        sub = self.estimate_subtype(symptoms, module_results)
        trend = (fusion_result or {}).get('trend', 'STABLE')
        sev = self.classify_severity(nihss_total, trend)
        hosp = self.recommend_hospital(sev, sub['subtype'])
        return {
            'subtype': sub, 'severity': sev, 'hospital': hosp,
            'disclaimer': ('KHONG PHAN CHAN DOAN - chi goi y so bo de '
                           'phan van bac si. Goi 115 neu co dau hieu.'),
        }


# ======================================================================
# TEST
# ======================================================================
def test_triage_engine():
    import sys
    print("=" * 70)
    print("TRIAGE ENGINE TEST")
    print("=" * 70)
    te = TriageEngine()

    # T1: base
    s = te.estimate_subtype({})
    assert s['subtype'] == 'ischemic' and s['confidence'] == 80.0, s
    print(f"1. Base: {s['subtype']} {s['confidence']}%")

    # T2: headache + nôn → nghi xuất huyết
    s2 = te.estimate_subtype({'sudden_severe_headache': True, 'vomiting': True})
    assert s2['subtype'] == 'hemorrhagic' and s2['confidence'] == 70.0, s2
    print(f"2. Headache+non: {s2['subtype']} {s2['confidence']}%")

    # T3: speech dominant → ischemic cao hơn
    mods = {'speech': {'speech_prob': 90, 'status': 'DANGER'},
            'face': {'score': 10, 'status': 'NORMAL'}}
    s3 = te.estimate_subtype(module_results=mods)
    assert s3['confidence'] == 95.0, s3   # 80 + 15 (clamp 95)
    print(f"3. Speech-dominant: {s3['subtype']} {s3['confidence']}% (clamp 95)")

    # T4: severity + upgrade theo trend
    assert te.classify_severity(4) == 'MILD'
    assert te.classify_severity(8) == 'MODERATE'
    assert te.classify_severity(12) == 'SEVERE'
    assert te.classify_severity(4, trend='WORSENING') == 'MODERATE'
    print("4. Severity bands + upgrade WORSENING")

    # T5: hospital từ database
    hs = te.load_hospitals()
    if hs:
        rec = te.recommend_hospital('SEVERE', 'ischemic')
        assert rec is not None and rec['stroke_unit'].lower() in ('1', 'true', 'yes', 'co')
        print(f"5. SEVERE -> {rec['name']} ({rec['distance_km']}km)")
        rec2 = te.recommend_hospital('MILD', 'ischemic')
        assert float(rec2['distance_km']) <= float(rec['distance_km']), "Mild = gần nhất"
        print(f"   MILD -> {rec2['name']} (gan nhat)")
    else:
        print("5. (không có hospital_database.csv — bỏ qua)")

    print("\nALL TRIAGE TESTS PASS")


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    test_triage_engine()
