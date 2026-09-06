# -*- coding: utf-8 -*-
"""
NIHSS ESTIMATOR - PSCS v8.0 (Golden-Watch)
Ước tính NIHSS 4 items từ output 4 module — TUÂN TRỌNG thang NIHSS gốc.

4 ITEM (thang NIHSS chuẩn — Brott 1989, PMID 2757041):
  Item 4  Facial Palsy   : 0-3  ← M1 Face (prob 'score')
  Item 5  Motor Arm      : 0-4  ← M3 Arm  (prob 'arm_prob')
  Item 6  Motor Leg      : 0-4  ← M4 Gait (prob 'gait_prob')
  Item 10 Dysarthria     : 0-2  ← M2 Speech (prob 'speech_prob')
  ────────────────────────────
  TỔNG TỐI ĐA 4 ITEM = 13  (không phải 15 — item 10 dysarthria tối đa 2 điểm)

KHÔNG PHẢN CHẨN ĐOÁN — chỉ ước tính hỗ trợ, bác sĩ chấm NIHSS chuẩn trên lâm sàng.

CI (Confidence Interval): Monte Carlo — thêm nhiễu ±10 điểm prob, chấm lại
1000 lần → total = trung bình ± 1.96*SD (khoảng tin cậy 95%).

Tác giả: PSCS Team
Ngày: 06/09/2026
"""

import sys

import numpy as np

# Bảng chuyển prob (0-100) → điểm item theo band NIHSS thật
PROB_TO_ITEM = {
    #  item       bands: (prob_min, score) — prob cao hơn band trước mới xét
    'item4_facial_palsy': [(80, 3), (55, 2), (30, 1), (0, 0)],          # max 3
    'item5_motor_arm':    [(85, 4), (70, 3), (50, 2), (25, 1), (0, 0)],  # max 4
    'item6_motor_leg':    [(85, 4), (70, 3), (50, 2), (25, 1), (0, 0)],  # max 4
    'item10_dysarthria':  [(60, 2), (30, 1), (0, 0)],                    # max 2
}
ITEM_MAX = {'item4_facial_palsy': 3, 'item5_motor_arm': 4,
            'item6_motor_leg': 4, 'item10_dysarthria': 2}

# Module key → item
MODULE_TO_ITEM = {
    'face': 'item4_facial_palsy',
    'arm': 'item5_motor_arm',
    'gait': 'item6_motor_leg',
    'speech': 'item10_dysarthria',
}
# prob key của từng module (đồng bộ FusionEngine)
PROB_KEYS = {'face': 'score', 'speech': 'speech_prob',
             'arm': 'arm_prob', 'gait': 'gait_prob'}

# Status module = không có dữ liệu (đồng bộ FusionEngine)
INVALID_STATUSES = {'NO_FACE', 'NO_PERSON', 'NO_POSE', 'NO_DATA',
                    'NO_SPEECH', 'NO_DETECTOR', 'NO_MODEL', 'ERROR'}


def prob_to_score(item, prob):
    """Chuyển prob 0-100 → điểm NIHSS của item theo band."""
    for prob_min, score in PROB_TO_ITEM[item]:
        if prob >= prob_min:
            return score
    return 0


def estimate_nihss(module_results):
    """
    Args:
        module_results: dict output module (format FusionEngine), vd:
            {'face': {'score': 65, 'status': 'WARNING'}, ...}
    Returns:
        dict: items {tên_item: điểm}, total, items_missing, confidence
    """
    items = {}
    missing = []
    for module_key, item in MODULE_TO_ITEM.items():
        r = (module_results or {}).get(module_key)
        if r is None or r.get('status') in INVALID_STATUSES:
            missing.append(item)
            continue
        prob = r.get(PROB_KEYS[module_key])
        if prob is None:
            missing.append(item)
            continue
        items[item] = prob_to_score(item, float(prob))
    total = sum(items.values())
    return {'items': items, 'total': total,
            'max_possible': sum(ITEM_MAX.values()),   # 13 cho 4 item
            'items_missing': missing,
            'note': 'UOC TIN HO TRO - bac si cham NIHSS chuan'}


def calculate_nihss_ci(module_results, n_iter=1000, noise=10.0, seed=42):
    """
    Khoảng tin cậy 95% cho NIHSS total: Monte Carlo nhiễu prob ±noise điểm.
    Trả về {'total', 'ci_low', 'ci_high', 'margin'} — trình bày "8 ± 2".
    """
    rng = np.random.default_rng(seed)
    probs = {}
    for module_key, item in MODULE_TO_ITEM.items():
        r = (module_results or {}).get(module_key)
        if r is None or r.get('status') in INVALID_STATUSES:
            continue
        p = r.get(PROB_KEYS[module_key])
        if p is not None:
            probs[item] = float(p)
    if not probs:
        return {'total': 0, 'ci_low': 0, 'ci_high': 0, 'margin': 0}

    totals = np.zeros(n_iter)
    for i in range(n_iter):
        s = 0
        for item, p in probs.items():
            noisy = float(np.clip(p + rng.normal(0, noise), 0, 100))
            s += prob_to_score(item, noisy)
        totals[i] = s
    total = sum(prob_to_score(it, p) for it, p in probs.items())
    margin = 1.96 * totals.std()
    return {'total': int(total),
            'ci_low': int(round(total - margin)),
            'ci_high': int(round(total + margin)),
            'margin': int(round(margin))}


# ======================================================================
# TEST
# ======================================================================
def test_nihss_estimator():
    import sys
    print("=" * 70)
    print("NIHSS ESTIMATOR TEST")
    print("=" * 70)

    # T1: band chuyển đổi đúng biên
    assert prob_to_score('item4_facial_palsy', 29.9) == 0
    assert prob_to_score('item4_facial_palsy', 30) == 1
    assert prob_to_score('item4_facial_palsy', 80) == 3
    assert prob_to_score('item5_motor_arm', 84.9) == 3
    assert prob_to_score('item5_motor_arm', 85) == 4
    assert prob_to_score('item10_dysarthria', 59.9) == 1
    assert prob_to_score('item10_dysarthria', 60) == 2
    print("1. Band prob->item đúng biên (item10 max 2 theo NIHSS chuẩn)")

    # T2: tổng hợp đủ 4 module
    mods = {'face': {'score': 65, 'status': 'WARNING'},
            'arm': {'arm_prob': 90, 'status': 'DANGER'},
            'gait': {'gait_prob': 35, 'status': 'MONITOR'},
            'speech': {'speech_prob': 70, 'status': 'DANGER'}}
    est = estimate_nihss(mods)
    # item4=2, item5=4, item6=1, item10=2 → 9
    assert est['total'] == 9, est
    assert est['max_possible'] == 13, est
    print(f"2. Tổng NIHSS 4 module: {est['total']}/13 {est['items']}")

    # T3: module thiếu + invalid
    est2 = estimate_nihss({'face': {'score': 0, 'status': 'NO_FACE'},
                           'arm': {'arm_prob': 90, 'status': 'DANGER'}})
    assert est2['total'] == 4 and 'item4_facial_palsy' in est2['items_missing']
    print(f"3. NO_FACE bị loại, missing={est2['items_missing']}, total=4")

    # T4: CI
    ci = calculate_nihss_ci(mods, n_iter=500)
    assert ci['ci_low'] <= ci['total'] <= ci['ci_high'], ci
    print(f"4. CI 95%: {ci['total']} ± {ci['margin']} "
          f"[{ci['ci_low']}-{ci['ci_high']}]")

    # T5: empty an toàn
    assert estimate_nihss({})['total'] == 0
    assert calculate_nihss_ci({})['total'] == 0
    print("5. Input rỗng an toàn")

    print("\nALL NIHSS TESTS PASS")


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    test_nihss_estimator()
