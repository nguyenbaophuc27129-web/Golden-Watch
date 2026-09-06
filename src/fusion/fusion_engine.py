# -*- coding: utf-8 -*-
"""
FUSION ENGINE - PSCS v8.0
Ghép kết quả từ nhiều module phát hiện thành 1 điểm nguy cơ tổng hợp.

Logic (v1 - theo lịch Ngày 5):
  1. Chuẩn hóa output của từng module về (prob 0-100, nihss item, status)
  2. LOẠI module không hợp lệ (NO_FACE, NO_PERSON, NO_DATA, NO_SPEECH...)
     -> renormalize trọng số các module còn lại
  3. Điểm tổng hợp = trung bình có trọng số (weighted average)
  4. Lớp luật lâm sàng FAST (rule layer) có thể NÂNG risk level:
     - R1: >= 2/3 dấu hiệu FAST bất thường (prob >= 50)  -> EMERGENCY
     - R2: 1 module duy nhất prob >= 80                   -> WARNING
  5. NIHSS ước tính = tổng các item có sẵn (item 4 + 5 + 6 + 10, max 15)

Tác giả: PSCS Team
Ngày: 04/09/2026
"""

import time
from collections import deque


class FusionEngine:
    """
    Fusion Engine: kết hợp scores từ 5 module thành 1 quyết định.

    Input: dict theo key: 'face', 'speech', 'arm', 'gait', 'radar'
    Output: fused_score (0-100), risk_level, nihss_total, recommendation
    """

    # Trọng số cơ sở cho từng module (FAST priority + độ tin cậy ML)
    # - arm 0.30: NIHSS item 5 tương quan mạnh nhất + ML 99.5%
    # - face/speech 0.20: dấu hiệu FAST cốt lõi (ML 93.75% / 83.07%)
    # - gait 0.15: kém đặc hiệu (nhiều nguyên nhân không phải đột quỵ)
    # - radar 0.15: phát hiện ngã (proxy), chưa có model lâm sàng
    DEFAULT_WEIGHTS = {
        'face': 0.20,
        'speech': 0.20,
        'arm': 0.30,
        'gait': 0.15,
        'radar': 0.15,
    }

    # Ngưỡng risk level trên fused_score (0-100)
    RISK_THRESHOLDS = {
        'NORMAL': 30,     # < 30
        'MONITOR': 50,    # 30-49
        'WARNING': 70,    # 50-69
        'EMERGENCY': 100  # >= 70
    }

    # Các status KHÔNG hợp lệ cho fusion (module không thu được dữ liệu)
    INVALID_STATUSES = {
        'NO_FACE', 'NO_PERSON', 'NO_POSE', 'NO_DATA', 'NO_SPEECH',
        'NO_DETECTOR', 'NO_MODEL', 'ERROR'
    }

    def __init__(self, weights=None, history_size=100):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.history = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # 1. ADAPTER: chuẩn hóa output các module về cùng một format
    # ------------------------------------------------------------------
    def _normalize_module(self, key, result):
        """
        Đưa output của module về: {'prob', 'status', 'nihss', 'valid'}

        Các module đặt tên xác suất khác nhau:
          face->score, speech->speech_prob, arm->arm_prob,
          gait->gait_prob, radar->fall_prob
        """
        if result is None:
            return None

        prob_keys = {
            'face': 'score', 'speech': 'speech_prob', 'arm': 'arm_prob',
            'gait': 'gait_prob', 'radar': 'fall_prob'
        }
        pk = prob_keys.get(key, 'prob')

        prob = result.get(pk)
        status = result.get('status', 'ERROR')
        nihss = result.get('nihss_score', result.get('nihss_item_4', 0))

        if prob is None:
            return None

        valid = status not in self.INVALID_STATUSES
        return {'prob': float(prob), 'status': status,
                'nihss': int(nihss), 'valid': valid}

    # ------------------------------------------------------------------
    # 2. FUSION CHÍNH
    # ------------------------------------------------------------------
    def fuse(self, module_results):
        """
        Args:
            module_results: dict, ví dụ:
                {'face': {...}, 'speech': {...}, 'arm': None, ...}
                Module thiếu hoặc None = không khả dụng.

        Returns:
            dict: fused_score, risk_level, nihss_total, nihss_items,
                  modules_used, modules_skipped, triggered_rules,
                  recommendation, timestamp
        """
        normalized = {}
        for key, result in (module_results or {}).items():
            m = self._normalize_module(key, result)
            if m is not None:
                normalized[key] = m

        # Modules dùng được cho weighted average
        used = {k: v for k, v in normalized.items() if v['valid']}
        skipped = {k: v['status'] for k, v in normalized.items() if not v['valid']}

        triggered_rules = []

        # ----- 3. WEIGHTED AVERAGE (có renormalize) -----
        if used:
            total_w = sum(self.weights.get(k, 0.15) for k in used)
            if total_w <= 0:
                total_w = len(used)  # fallback: trọng số bằng nhau
            fused_score = sum(
                self.weights.get(k, 0.15) * v['prob'] for k, v in used.items()
            ) / total_w
        else:
            fused_score = 0.0

        # ----- 4. LỚP LUẬT LÂM SÀNG FAST -----
        # R1: >= 2/3 dấu hiệu FAST (face, arm, speech) bất thường rõ
        #     -> đột quỵ khả năng cao -> EMERGENCY bất kể weighted average
        fast_signs = ['face', 'arm', 'speech']
        abnormal_fast = [
            k for k in fast_signs
            if k in used and used[k]['prob'] >= 50
        ]
        if len(abnormal_fast) >= 2:
            triggered_rules.append(
                f"R1: {len(abnormal_fast)}/3 dau hieu FAST bat thuong "
                f"({', '.join(abnormal_fast)})")
            fused_score = max(fused_score, 75.0)

        # R2: 1 module cựcheavy nặng (>=80) -> ít nhất WARNING
        #     (1 dấu hiệu đơn lẻ có thể là nhiễu -> chờ Defense Layer 3 xác nhận)
        for k, v in used.items():
            if v['prob'] >= 80:
                triggered_rules.append(f"R2: {k} prob {v['prob']:.0f} >= 80")
                fused_score = max(fused_score, 55.0)
                break  # chỉ cần 1 lần

        # ----- RISK LEVEL -----
        risk_level = 'NORMAL'
        if fused_score >= 70:
            risk_level = 'EMERGENCY'
        elif fused_score >= 50:
            risk_level = 'WARNING'
        elif fused_score >= 30:
            risk_level = 'MONITOR'

        # ----- 5. NIHSS ƯỚC TÍNH (tổng các item có sẵn) -----
        nihss_items = {}
        nihss_map = {'face': 'item4_facial_palsy', 'arm': 'item5_motor_arm',
                     'gait': 'item6_motor_leg', 'speech': 'item10_dysarthria'}
        for k, v in normalized.items():
            if k in nihss_map and v['status'] not in self.INVALID_STATUSES:
                nihss_items[nihss_map[k]] = v['nihss']
        nihss_total = sum(nihss_items.values())
        severity = self._classify_severity(nihss_total)

        result = {
            'fused_score': round(fused_score, 1),
            'risk_level': risk_level,
            'nihss_total': nihss_total,
            'nihss_items': nihss_items,
            'severity': severity,
            'modules_used': {k: {'prob': v['prob'], 'status': v['status'],
                                 'nihss': v['nihss']}
                             for k, v in used.items()},
            'modules_skipped': skipped,
            'triggered_rules': triggered_rules,
            'recommendation': self._recommendation(risk_level),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        self.history.append(result)
        return result

    # ------------------------------------------------------------------
    # 3. TIỆN ÍCH
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_severity(nihss_total):
        """NIHSS 4 items (max 15): Mild 0-5, Moderate 6-13, Severe 14+"""
        if nihss_total >= 14:
            return 'SEVERE'
        if nihss_total >= 6:
            return 'MODERATE'
        return 'MILD'

    @staticmethod
    def _recommendation(risk_level):
        return {
            'NORMAL': 'Tiep tuc theo doi binh thuong',
            'MONITOR': 'Yeu cau nguoi dung lam lai bai test',
            'WARNING': 'Bao nguoi than kiem tra - lien he bac si',
            'EMERGENCY': 'GOI 115 NGAY - co dau hieu dot quy',
        }.get(risk_level, '')

    def get_trend(self, window=5):
        """
        Xu hướng từ lịch sử fusion (phục vụ Dashboard và Defense Layer 3).
        Returns: 'WORSENING' | 'IMPROVING' | 'STABLE' | 'NO_DATA'
        """
        if len(self.history) < window:
            return 'NO_DATA'
        recent = [h['fused_score'] for h in list(self.history)[-window:]]
        delta = recent[-1] - recent[0]
        if delta > 10:
            return 'WORSENING'
        if delta < -10:
            return 'IMPROVING'
        return 'STABLE'


# ======================================================================
# TEST VỚI DỮ LIỆU MÔ PHỎNG
# ======================================================================
def test_fusion_engine():
    print("=" * 70)
    print("FUSION ENGINE TEST")
    print("=" * 70)

    fe = FusionEngine()

    # Chuẩn hóa output mẫu đúng format từng module
    def face(prob, nihss=0, status=None):
        st = status or ('NORMAL' if prob < 30 else
                        ('WARNING' if prob < 60 else 'DANGER'))
        return {'score': prob, 'nihss_item_4': nihss, 'status': st,
                'raw_metrics': {}, 'raw_landmarks': None}

    def speech(prob, nihss=0, status=None):
        st = status or ('NORMAL' if prob < 30 else
                        ('WARNING' if prob < 60 else 'DANGER'))
        return {'speech_prob': prob, 'nihss_score': nihss, 'status': st,
                'metrics': {}}

    def arm(prob, nihss=0):
        st = 'NORMAL' if prob < 30 else ('WARNING' if prob < 60 else 'DANGER')
        return {'arm_prob': prob, 'nihss_score': nihss, 'status': st,
                'metrics': {}}

    def gait(prob, nihss=0):
        st = 'NORMAL' if prob < 30 else ('WARNING' if prob < 60 else 'DANGER')
        return {'gait_prob': prob, 'nihss_score': nihss, 'status': st,
                'metrics': {}}

    scenarios = [
        ("Tat ca binh thuong",
         {'face': face(5), 'speech': speech(2), 'arm': arm(3), 'gait': gait(4)}),
        ("Chi face cao (1 dang hieu)",
         {'face': face(85, 2), 'speech': speech(10), 'arm': arm(8), 'gait': gait(5)}),
        ("FAST+: face + arm bat thuong",
         {'face': face(70, 2), 'speech': speech(15), 'arm': arm(65, 2),
          'gait': gait(20)}),
        ("Emergancy: 3/3 FAST + dysarthria",
         {'face': face(90, 3), 'speech': speech(85, 3), 'arm': arm(88, 3),
          'gait': gait(60, 2)}),
        ("Thieu module (chi face + speech)",
         {'face': face(10), 'speech': speech(8)}),
        ("Module loi (NO_FACE, NO_SPEECH)",
         {'face': face(0, status='NO_FACE'), 'speech': speech(0, status='NO_SPEECH'),
          'arm': arm(40)}),
        ("Khong co module nao",
         {}),
    ]

    for name, modules in scenarios:
        r = fe.fuse(modules)
        print(f"\n--- {name} ---")
        print(f"  fused={r['fused_score']} | {r['risk_level']} | "
              f"NIHSS={r['nihss_total']} ({r['severity']})")
        print(f"  used={list(r['modules_used'].keys())} "
              f"skipped={r['modules_skipped']}")
        if r['triggered_rules']:
            print(f"  rules: {r['triggered_rules']}")
        print(f"  -> {r['recommendation']}")

    # Trend test
    print(f"\nTrend (sau 7 lan fuse): {fe.get_trend()}")
    print("\nDONE")


if __name__ == '__main__':
    import sys
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    test_fusion_engine()
