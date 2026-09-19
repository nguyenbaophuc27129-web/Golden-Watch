# -*- coding: utf-8 -*-
"""
EARLY WARNING MONITOR — PSCS v9.0 (trụ cột B, NK-11)
====================================================
Nâng cấp L3 từ "luật ngưỡng 30s" thành THỐNG KÊ: phát hiện SUY GIẢM
TIẾN TRIỂN (deterioration) trên chuỗi fused score — kể cả khi điểm chưa
chạm ngưỡng báo động (vùng prodromal 30–50 mà luật cũ KHÔNG BAO GIỜ báo).

3 thành phần (tham số đăng ký TRƯỚC khi chạy — seed 42):
  1. EWMA (alpha=0.15)        : mức nền có làm mượt — chống nhiễu đơn
  2. CUSUM một phía (k=0.5σ, h=5σ): dò dịch chuyển mức (change-point)
     chuẩn hóa theo baseline cá nhân (mu0, sigma từ đoạn binh_thường)
  3. Xác nhận run-length      : CUSUM vượt h CHỈ khi score Elevated
     >= 15 chu kỳ liên tiếp (30s @ chu kỳ 2s — nhất quán
     DefenseEngine.PERSISTENT_SECONDS) → chống spike ngắn (cười/ngáp)

Trạng thái: OK / WATCH (CUSUM > h/2 hoặc EWMA lệch >= 2σ) /
DETERIORATING (CUSUM > h + run 30s) — khác EMERGENCY đơn điểm: đây là
"đang xấu đi DẦN", khuyến nghị người thân kiểm tra, không tự gọi 115.

An toàn: KHÔNG đụng fast-path EMERGENCY của fusion; module này chỉ THÊM
khả năng báo suy giảm chậm. Clock injectable → test được với FakeClock.

Tác giả: PSCS Team
Ngày: 12/09/2026
"""

import numpy as np


class EarlyWarningMonitor:
    """
    Dò suy giảm tiến triển trên chuỗi điểm (1 điểm / chu kỳ phân tích).

    Dùng:
        m = EarlyWarningMonitor()
        m.calibrate(scores_normal)          # mu0, sigma từ đoạn bình thường
        m.update(score)                     # mỗi chu kỳ → dict trạng thái
    """

    # ---- tham số đăng ký trước (ghi trong mọi summary.json) ----
    EWMA_ALPHA = 0.15
    CUSUM_K = 0.5          # đơn vị sigma (allowance)
    CUSUM_H = 5.0          # đơn vị sigma (ngưỡng báo; ARL0 chuẩn ~hàng trăm)
    ELEVATED_SIGMA = 1.5   # score >= mu0 + 1.5σ tính là "đang cao"
    CONFIRM_CYCLES = 15    # số chu kỳ liên tiếp "đang cao" (15×2s = 30s)
    SLOPE_WINDOW_SEC = 300 # hồi quy tuyến tính 5 phút
    SLOPE_MIN_PTS = 8      # tối thiểu điểm để slope tin được

    def __init__(self, cycle_seconds=2.0, clock=None):
        """
        clock: callable trả giây (mặc định time.time) — inject được để test.
        """
        import time as _time
        self._clock = clock or _time.time
        self.cycle_seconds = float(cycle_seconds)
        self.mu0 = None
        self.sigma = None
        self.reset()

    def reset(self):
        self.ewma = None
        self.cusum = 0.0
        self.run_elevated = 0
        self.history = []          # [(score, t)]
        self.n_updates = 0

    # ------------------------------------------------------------------
    def calibrate(self, scores):
        """mu0/sigma từ đoạn BÌNH THƯỜNG (đoạn calibrate cá nhân, như L1)."""
        s = np.asarray(list(scores), dtype=np.float64)
        s = s[np.isfinite(s)]
        if len(s) < 10:
            return False
        self.mu0 = float(np.median(s))          # median: bền với spike lẻ
        mad = float(np.median(np.abs(s - self.mu0)))
        self.sigma = max(1.4826 * mad, 1e-6)    # σ từ MAD (robust)
        self.reset()
        return True

    # ------------------------------------------------------------------
    def update(self, score, t=None):
        """
        Nhập 1 điểm mới → trạng thái. Trước khi calibrate: NO_BASELINE
        (không bao giờ tự báo — trung lập an toàn).
        """
        if self.mu0 is None:
            return {'state': 'NO_BASELINE', 'score': score,
                    'reason': 'chưa calibrate baseline cá nhân'}

        t = self._clock() if t is None else float(t)
        x = float(score)
        self.history.append((x, t))
        self.n_updates += 1

        # ----- 1. EWMA -----
        self.ewma = x if self.ewma is None else \
            self.EWMA_ALPHA * x + (1 - self.EWMA_ALPHA) * self.ewma

        # ----- 2. CUSUM một phía (lên) chuẩn hóa -----
        z = (x - self.mu0) / self.sigma
        self.cusum = max(0.0, self.cusum + z - self.CUSUM_K)

        # ----- 3. Run-length xác nhận (chống spike ngắn) -----
        elevated = x >= self.mu0 + self.ELEVATED_SIGMA * self.sigma
        self.run_elevated = self.run_elevated + 1 if elevated else 0
        confirmed = self.run_elevated >= self.CONFIRM_CYCLES

        # ----- 4. Slope OLS cửa sổ 5 phút (chỉ báo kèm theo) -----
        slope = self._slope()

        # ----- 5. TRẠNG THÁI -----
        # WATCH chỉ theo EWMA (mức nền dịch chuyển BỀN) — không dùng
        # CUSUM nửa ngưỡng vì sau 1 spike lớn CUSUM còn cao hàng phút
        # → WATCH sẽ nhấp nháy vô nghĩa (bài học smoke-test 12/09).
        cusum_alarm = self.cusum > self.CUSUM_H
        watch = self.ewma >= self.mu0 + 2 * self.sigma
        if cusum_alarm and confirmed:
            state, reason = 'DETERIORATING', (
                f'CUSUM {self.cusum:.1f}>{self.CUSUM_H}σ + cao liên tiếp '
                f'{self.run_elevated} chu kỳ (30s)')
        elif watch:
            state, reason = 'WATCH', f'CUSUM {self.cusum:.1f}σ / EWMA lệch'
        else:
            state, reason = 'OK', ''

        return {'state': state, 'reason': reason, 'score': x,
                'ewma': round(self.ewma, 1), 'cusum_sigma': round(self.cusum, 2),
                'run_elevated': self.run_elevated,
                'slope_per_min': slope,
                'mu0': round(self.mu0, 1), 'sigma': round(self.sigma, 1)}

    # ------------------------------------------------------------------
    def _slope(self):
        """OLS điểm/giây × 60 trên cửa sổ 5 phút (None nếu chưa đủ điểm)."""
        if len(self.history) < self.SLOPE_MIN_PTS:
            return None
        now_t = self.history[-1][1]
        pts = [(x, tt) for x, tt in self.history
               if now_t - tt <= self.SLOPE_WINDOW_SEC]
        if len(pts) < self.SLOPE_MIN_PTS:
            return None
        xs = np.array([tt for _, tt in pts]) - now_t
        ys = np.array([x for x, _ in pts])
        if float(np.ptp(xs)) < 1e-9:
            return None
        b = np.polyfit(xs, ys, 1)[0]     # điểm/giây
        return round(float(b) * 60.0, 2)  # điểm/phút


# ======================================================================
# TEST nhanh (FakeClock — cùng pattern harness Protocol B)
# ======================================================================
def test_early_warning():
    class FakeClock:
        def __init__(self):
            self.now = 0.0
        def time(self):
            return self.now

    rng = np.random.default_rng(42)
    clock = FakeClock()
    m = EarlyWarningMonitor(cycle_seconds=2.0, clock=clock.time)

    # 1) calibrate từ đoạn bình thường
    normal = rng.normal(22, 6, 200)
    assert m.calibrate(normal), 'calibrate fail'
    for s in normal[-50:]:
        clock.now += 2.0
        m.update(float(s))
    assert m.update(20)['state'] in ('OK', 'WATCH')

    # 2) spike ngắn (cười) → KHÔNG DETERIORATING
    m.reset()
    m.calibrate(normal)
    states = []
    for i in range(600):
        clock.now += 2.0
        v = float(rng.normal(22, 6))
        if 100 <= i < 104:      # 4 chu kỳ spike (8s)
            v = 80.0
        states.append(m.update(v)['state'])
    assert 'DETERIORATING' not in states, 'spike ngắn bị báo nhầm!'

    # 3) suy giảm chậm +18 điểm / 30 phút (kết thúc ~40 < ngưỡng 50) → BẮT
    m.reset()
    m.calibrate(normal)
    hit = None
    for i in range(900):        # 30 phút
        clock.now += 2.0
        v = 22 + 18 * i / 900 + float(rng.normal(0, 6))
        st = m.update(v)['state']
        if st == 'DETERIORATING':
            hit = i
            break
    assert hit is not None, 'không bắt được suy giảm chậm dưới ngưỡng!'
    print(f'PASS: ramp +18/30ph (chạm tối đa ~40) → DETERIORATING tại '
          f'{hit} chu kỳ = {hit*2/60:.1f} phút')
    return True


if __name__ == '__main__':
    test_early_warning()
