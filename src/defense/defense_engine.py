# -*- coding: utf-8 -*-
"""
DEFENSE ENGINE 4 LỚP - PSCS v8.0 (Golden-Watch)
Giảm FALSE ALARM (báo động giả) mà không mất TRUE POSITIVE — mục tiêu
FPR < 5%, TPR > 90% (theo TONG_QUAN_DU_AN_FINAL.md v6.0).

4 LỚP (đúng spec):
  L1 Calibration       : baseline cá nhân (15 phút) — mỗi người 1 ngưỡng
  L2 Context Awareness : phân loại hoạt động (tập/đi/nói/nghỉ) → suppress
                         module không liên quan (cười ≠ méo mặt bệnh lý)
  L3 Temporal Analysis : cửa sổ 30s/1p/5p — cảnh báo TẠM THỜI < 30s = nhiễu,
                         PERSISTENT > 30s mới nổi; worsening = năng cấp
  L4 Adaptive Threshold: ngưỡng thích nghi theo baseline cá nhân + thời gian

Vị trí trong pipeline:
  [5 modules] → [DefenseEngine.filter(module_results, context)] → [FusionEngine]

Tác giả: PSCS Team
Ngày: 06/09/2026
"""

import sys
import time
from collections import deque


class DefenseEngine:
    """
    DefenseEngine: lọc nhiễu 4 lớp TRƯỚC khi fusion.

    Usage:
        de = DefenseEngine()
        de.calibrate(wpm=148, face_asym=8.0, ...)      # L1 (1 lần/15 phút)
        de.update_context(motion=0.72, talking=True)   # L2 (mỗi chu kỳ)
        clean = de.filter(module_results)              # L2+L3+L4 (mỗi chu kỳ)
        verdict = de.verdict(fused_score)              # L3 quyết định cuối
    """

    # ---- L2: ánh xạ hoạt động -> module bị suppress (spec TONG_QUAN) ----
    CONTEXT_SUPPRESS = {
        'EXERCISE': {'face', 'arm', 'gait'},   # vận động mạnh → méo mặt/cử động tay là bình thường
        'WALKING': {'arm', 'gait'},
        'TALKING': {'speech'},                  # đang nói chuyện → phát hiện nói đơ bị nhiễu ít, giữ
        'REST': set(),                          # nghỉ → không suppress gì
    }

    # ---- L3: cửa sổ thời gian (spec: transient < 30s = FA) ----
    PERSISTENT_SECONDS = 30.0    # cảnh báo phải kéo dài > 30s mới nổi
    WORSEN_DELTA = 10.0          # tăng > 10 điểm trong 5p → WORSENING

    def __init__(self, history_size=600):
        # L1: baseline cá nhân (None = chưa calibrate → universal)
        self.baseline = None
        self.calibrated_at = None

        # L2: ngữ cảnh hiện tại
        self.context = 'REST'
        self.motion_intensity = 0.0

        # L3: lịch sử (fused_score, timestamp) — 10 phút @ 1Hz = 600 điểm
        self.history = deque(maxlen=history_size)

        # Thống kê cho Dashboard
        self.stats = {'suppressed_L2': 0, 'suppressed_L3': 0,
                      'upgraded_L3': 0, 'adapted_L4': 0}

    # ------------------------------------------------------------------
    # LỚP 1 — CALIBRATION (baseline cá nhân)
    # ------------------------------------------------------------------
    def calibrate(self, wpm=None, face_asym=None, arm_asym=None,
                  gait_symmetry=None):
        """
        Ghi nhận baseline cá nhân (từ Quick Calibration 15 phút).
        Đọc 3 câu + capture 10 frames + vươn vai + đi bộ.
        """
        self.baseline = {
            'wpm': wpm, 'face_asym': face_asym,
            'arm_asym': arm_asym, 'gait_symmetry': gait_symmetry,
        }
        self.calibrated_at = time.time()
        return self.baseline

    # ------------------------------------------------------------------
    # LỚP 2 — CONTEXT AWARENESS
    # ------------------------------------------------------------------
    def update_context(self, motion_intensity=0.0, talking=False):
        """
        Phân loại hoạt động từ cường độ chuyển động + đang nói.
          motion > 0.6            → EXERCISE
          0.2 < motion <= 0.6     → WALKING
          talking (VAD có tiếng)  → TALKING
          còn lại                 → REST
        """
        self.motion_intensity = motion_intensity
        if motion_intensity > 0.6:
            self.context = 'EXERCISE'
        elif motion_intensity > 0.2:
            self.context = 'WALKING'
        elif talking:
            self.context = 'TALKING'
        else:
            self.context = 'REST'
        return self.context

    # ------------------------------------------------------------------
    # FILTER CHÍNH (L2 + L3 giữ nguyên + L4 nới ngưỡng)
    # ------------------------------------------------------------------
    def filter(self, module_results):
        """
        Args:
            module_results: dict output các module (đúng format FusionEngine)
        Returns:
            (clean_results, audit): audit = dict những gì đã làm
        """
        clean, audit = {}, {'suppressed': [], 'adaptive_floor': {}}

        suppressed_keys = self.CONTEXT_SUPPRESS.get(self.context, set())
        for key, result in (module_results or {}).items():
            if result is None:
                continue
            # ----- L2: suppress theo ngữ cảnh -----
            if key in suppressed_keys:
                # KHÔNG mất dữ liệu: giữ prob nhưng đánh dấu, fusion vẫn dùng
                # (suppress hoàn toàn có thể che đột quỵ thật khi đang vận động
                #  — thay vào đó ghi audit để Defense Layer 3 xử lý sau)
                clean[key] = dict(result)
                clean[key]['defense_note'] = f'context={self.context}'
                audit['suppressed'].append(key)
                self.stats['suppressed_L2'] += 1
                continue

            # ----- L4: ngưỡng thích nghi (nới floor khi đã calibrate) -----
            clean[key] = self._adaptive_floor(key, result)
            if 'adaptive_floor' in clean[key]:
                audit['adaptive_floor'][key] = clean[key]['adaptive_floor']
                self.stats['adapted_L4'] += 1

        return clean, audit

    def _adaptive_floor(self, key, result):
        """
        L4: nếu prob module chỉ NHẸ vượt ngưỡng (trong vùng xám) và đã có
        baseline cá nhân → hạ 1 chút (giảm FA). Vùng xám = [30, 40).
        Chỉ áp khi baseline đủ tin cậy (đã calibrate).
        """
        prob_keys = {'face': 'score', 'speech': 'speech_prob',
                     'arm': 'arm_prob', 'gait': 'gait_prob',
                     'radar': 'fall_prob'}
        pk = prob_keys.get(key)
        if pk is None or self.baseline is None:
            return result
        prob = result.get(pk)
        if prob is not None and 30 <= prob < 40:
            # vùng xám: giảm 5 điểm (baseline cá nhân cho tín nhiệm cao hơn)
            out = dict(result)
            out[pk] = round(prob - 5.0, 1)
            out['adaptive_floor'] = f'{prob:.0f}->{out[pk]}'
            return out
        return result

    # ------------------------------------------------------------------
    # LỚP 3 — TEMPORAL VERDICT (chốt cảnh báo cuối)
    # ------------------------------------------------------------------
    def verdict(self, fused_score):
        """
        Nhận fused_score của chu kỳ này → quyết định cảnh báo cuối.

        Returns:
            {'alert': bool, 'reason': str, 'trend': str}
        """
        now = time.time()
        self.history.append((fused_score, now))

        # Cảnh báo dưới ngưỡng fusion (50) → không cần L3
        if fused_score < 50:
            return {'alert': False, 'reason': f'score {fused_score:.0f} < 50',
                    'trend': self._trend()}

        # ----- PERSISTENT CHECK: cảnh báo phải tồn tại > 30s -----
        # Tìm các điểm trong 30s gần nhất
        recent = [(s, t) for s, t in self.history
                  if now - t <= self.PERSISTENT_SECONDS]
        persistent = sum(1 for s, _ in recent if s >= 50)

        if len(recent) < 3:
            # mới khởi động hệ thống — chưa đủ dữ kiện, cho qua (an toàn)
            return {'alert': True, 'reason': 'warming up (cho qua)',
                    'trend': 'NO_DATA'}

        # Cần >= 60% mẫu trong 30s ở mức cảnh báo → PERSISTENT
        if persistent / len(recent) >= 0.6:
            trend = self._trend()
            if trend == 'WORSENING':
                self.stats['upgraded_L3'] += 1
                return {'alert': True, 'reason': 'persistent + WORSENING',
                        'trend': trend}
            self.stats['upgraded_L3'] += 1
            return {'alert': True, 'reason': f'persistent {persistent}/{len(recent)}',
                    'trend': trend}

        # Tạm thời (transient) → chặn, ghi stats
        self.stats['suppressed_L3'] += 1
        return {'alert': False,
                'reason': f'transient (persistent {persistent}/{len(recent)})',
                'trend': self._trend()}

    def _trend(self, window_seconds=300):
        """Trend 5 phút: WORSENING nếu tăng > 10 điểm đầu-cuối."""
        now = time.time()
        pts = [s for s, t in self.history if now - t <= window_seconds]
        if len(pts) < 3:
            return 'NO_DATA'
        delta = pts[-1] - pts[0]
        if delta > self.WORSEN_DELTA:
            return 'WORSENING'
        if delta < -self.WORSEN_DELTA:
            return 'IMPROVING'
        return 'STABLE'

    # ------------------------------------------------------------------
    # TIỆN ÍCH
    # ------------------------------------------------------------------
    def get_stats(self):
        return dict(self.stats, context=self.context,
                    calibrated=self.baseline is not None)


# ======================================================================
# TEST
# ======================================================================
def test_defense_engine():
    import sys
    print("=" * 70)
    print("DEFENSE ENGINE TEST (giả lập thời gian)")
    print("=" * 70)
    de = DefenseEngine()

    # L1
    de.calibrate(wpm=148, face_asym=8.0)
    assert de.get_stats()['calibrated'], "L1 calibrate"

    # L2: EXERCISE → face/arm/gait marked, dữ liệu giữ nguyên
    de.update_context(motion_intensity=0.8)
    mods = {'face': {'score': 55, 'status': 'WARNING'},
            'arm': {'arm_prob': 60, 'status': 'WARNING'},
            'speech': {'speech_prob': 70, 'status': 'DANGER'}}
    clean, audit = de.filter(mods)
    assert 'face' in audit['suppressed'] and 'arm' in audit['suppressed']
    assert clean['face']['score'] == 55, "L2 giữ nguyên prob"
    assert clean['speech'].get('defense_note') is None, "speech không suppress"
    print(f"L2 EXERCISE: suppressed={audit['suppressed']}")

    # L4: vùng xám 35 -> 30 khi có baseline
    de.update_context(motion_intensity=0.0)
    mods = {'gait': {'gait_prob': 35, 'status': 'MONITOR'}}
    clean, _ = de.filter(mods)
    assert clean['gait']['gait_prob'] == 30.0, f"L4 gray {clean['gait']}"
    print(f"L4: gait 35 -> {clean['gait']['gait_prob']} (vùng xám)")

    # L3: transient — 1 lần cảnh báo rồi hết → KHÔNG alert
    v1 = de.verdict(70)
    de.verdict(10); de.verdict(10)
    v2 = de.verdict(70)
    print(f"L3 transient: v1={v1['alert']} ({v1['reason']}), "
          f"v2={v2['alert']} ({v2['reason']})")
    assert v2['alert'] is False, "L3 chặn transient"

    # L3: persistent — cảnh báo liên tục > 30s → alert
    de2 = DefenseEngine()
    t0 = time.time()
    # giả lập 60 mẫu trong 60 giây đều score 70
    for i in range(60):
        de2.history.append((70.0, t0 - 60 + i))
    v = de2.verdict(70)
    assert v['alert'] is True, f"L3 persistent {v}"
    print(f"L3 persistent: alert={v['alert']} ({v['reason']}, trend={v['trend']})")

    print(f"\nStats: {de.get_stats()}")
    print("ALL DEFENSE TESTS PASS")


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    test_defense_engine()
