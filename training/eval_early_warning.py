# -*- coding: utf-8 -*-
"""
TRỤ CỘT B (v3.0, NK-11) — EARLY WARNING: OLD L3 vs EWMA+CUSUM
=============================================================
Câu hỏi khoa học: luật L3 cũ (ngưỡng 50 + persistent 30s + trend
endpoint-delta) có bắt được SUY GIẢM TIẾN TRIỂN không — và EWMA+CUSUM
làm tốt hơn bao nhiêu, với FAR kiểm soát thế nào?

Protocol ĐĂNG KÝ TRƯỚC (seed 42, chu kỳ 2s như hệ thống thật):
  Dữ liệu MÔ PHỎNG (ghi rõ "simulated" — chưa có Protocol B thật):
    - Nền người khỏe: AR(1) quanh mu0=22, sigma=6 (đúng vùng NORMAL/MONITOR)
    - Spike biểu cảm: +40..50 điểm trong 2–5 chu kỳ, ~20 lần/ngày
    - Burst stress: 10 đợt/ngày × 3–5 chu kỳ @ 75–85 điểm (stress FA)
    - Ramps suy giảm: +40 điểm / 15 · 30 · 60 phút, và +18 điểm / 30 phút
      (kết thúc ~40 — DƯỚI ngưỡng 50: luật cũ KHÔNG BAO GIỜ báo)
  So sánh:
    OLD = DefenseEngine THẬT (verdict + FakeClock, cùng pattern harness)
    NEW = EarlyWarningMonitor (EWMA 0.15 + CUSUM k=0.5σ h=5σ + run 30s)
  Warm-up 200 chu kỳ (không tính FA) — mô phỏng đoạn calibrate thật.
  FAR: 6h × 20 seed (= 120h) → chuẩn hóa /24h.
  Latency: chu kỳ từ mốc suy giảm đến lần báo đầu (median [IQR] 20 seed).
Output: test_results/early_warning_<ts>/ (JSON + figure 4 panel).
Chạy: PYTHONUTF8=1 python training/eval_early_warning.py [--fast]
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_DEF_DIR = os.path.join(ROOT, 'src', 'defense')
if _DEF_DIR not in sys.path:
    sys.path.insert(0, _DEF_DIR)

import defense_engine as de_mod          # noqa: E402  (src/defense trên path)
from defense_engine import DefenseEngine  # noqa: E402
from early_warning import EarlyWarningMonitor  # noqa: E402

SEED = 42
N_RUNS = 20
DT = 2.0
MU0, SIGMA, PHI = 22.0, 6.0, 0.3
WARMUP = 200
HOURS_FAR = 6
RAMP_END = 22 + 40          # ramps +40 chạm ~62
RAMP_BELOW_END = 22 + 18    # ramp +18 chạm ~40 < 50


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def time(self):
        return self.now


def ar1_noise(n, rng):
    """Nhiễu nền tự tương quan nhẹ (thực tế hơn iid)."""
    e = rng.normal(0, SIGMA * (1 - PHI ** 2) ** 0.5, n)
    x = np.empty(n)
    x[0] = MU0 + e[0]
    for i in range(1, n):
        x[i] = MU0 + PHI * (x[i - 1] - MU0) + e[i]
    return np.clip(x, 0, 100)


def add_spikes(x, rng, n_per_6h, lo=40, hi=50, max_len=5):
    """Spike biểu cảm: nâng đột biến 2–5 chu kỳ rồi về nền (không đổi nền)."""
    x = x.copy()
    for _ in range(n_per_6h):
        start = rng.integers(10, len(x) - max_len - 1)
        dur = int(rng.integers(2, max_len + 1))
        x[start:start + dur] += rng.uniform(lo, hi)
    return np.clip(x, 0, 100)


def add_ramp(x, rate_per_cycle, end_value):
    """Ramp tuyến tính từ mốc WARMUP tới khi chạm end_value rồi giữ."""
    x = x.copy()
    n = len(x)
    for i in range(WARMUP, n):
        v = MU0 + rate_per_cycle * (i - WARMUP)
        if v >= end_value:
            x[i:] = np.maximum(x[i:], end_value)
            break
        x[i] = x[i] + v - MU0
    return np.clip(x, 0, 100)


def run_old(scores):
    """OLD: DefenseEngine THẬT + FakeClock → list alert bool."""
    clock = FakeClock()
    de_mod.time = clock
    de = DefenseEngine()
    alerts = []
    for s in scores:
        clock.now += DT
        alerts.append(bool(de.verdict(float(s))['alert']))
    return alerts


def make_monitor():
    class M(EarlyWarningMonitor):
        """Bản ghi trạng thái lần gọi (để eval đọc không đụng src)."""
        def __init__(self, **kw):
            super().__init__(**kw)
            self._last_state, self._last_info = 'NO_BASELINE', {}

        def update(self, score, t=None):
            r = super().update(score, t=t)
            self._last_state, self._last_info = r['state'], r
            return r
    return M(cycle_seconds=DT)


def count_episodes(states, name):
    """Đếm số lần CHUYỂN vào trạng thái name (episode, không đếm chu kỳ)."""
    c = 0
    prev = None
    for s in states:
        if s == name and prev != name:
            c += 1
        prev = s
    return c


def far_experiment(n_cycles):
    """FAR: thường + spike; trả (old_alerts, new_det, new_watch)/24h —
    đếm EPISODE (lần báo) chứ không đếm chu kỳ liên tiếp."""
    old_c = new_c = watch_c = 0
    total_h = 0.0
    for r in range(N_RUNS):
        rng = np.random.default_rng(SEED + r)
        scores = ar1_noise(n_cycles, rng)
        scores = add_spikes(scores, rng, n_per_6h=20 * HOURS_FAR // 24)
        old = run_old(scores)
        mon = make_monitor()
        clock = FakeClock()
        mon._clock = clock.time
        assert mon.calibrate(scores[:WARMUP])
        new = []
        for s in scores:
            clock.now += DT
            new.append(mon.update(float(s), t=clock.now)['state'])
        # warm-up không tính (cả 2 phía)
        old_c += sum(old[WARMUP:])
        new_c += count_episodes(new[WARMUP:], 'DETERIORATING')
        watch_c += count_episodes(new[WARMUP:], 'WATCH')
        total_h += n_cycles * DT / 3600.0
    return old_c / total_h * 24, new_c / total_h * 24, watch_c / total_h * 24


def latency_experiment(rate_per_cycle, end_value):
    """Latency (phút) old vs new trên ramp; old=∞ nếu không bao giờ báo."""
    olds, news = [], []
    old_ever = 0
    for r in range(N_RUNS):
        rng = np.random.default_rng(SEED + 100 + r)
        n = WARMUP + int(4800 * DT)          # đủ 2.7h để ramp chậm chạm đáy
        scores = ar1_noise(n, rng)
        scores = add_ramp(scores, rate_per_cycle, end_value)
        old = run_old(scores)
        mon = make_monitor()
        clock = FakeClock()
        mon._clock = clock.time
        mon.calibrate(scores[:WARMUP])
        old_hit = next((i for i in range(WARMUP, n) if old[i]), None)
        new_hit = None
        for i in range(WARMUP, n):
            clock.now += DT
            if mon.update(float(scores[i]), t=clock.now)['state'] == 'DETERIORATING':
                new_hit = i
                break
        if old_hit is not None:
            old_ever += 1
            olds.append((old_hit - WARMUP) * DT / 60.0)
        if new_hit is not None:
            news.append((new_hit - WARMUP) * DT / 60.0)
    def stat(a):
        if not a:
            return None
        a = sorted(a)
        q1, med, q3 = a[len(a)//4], a[len(a)//2], a[(3*len(a))//4]
        return {'median_min': round(med, 1),
                'iqr_min': [round(q1, 1), round(q3, 1)]}
    return {'old': stat(olds), 'new': stat(news),
            'old_ever_fired': f'{old_ever}/{N_RUNS}',
            'new_ever_fired': f'{len(news)}/{N_RUNS}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true',
                    help='3h × 5 seed (smoke)')
    args = ap.parse_args()
    global N_RUNS, HOURS_FAR
    if args.fast:
        N_RUNS, HOURS_FAR = 5, 3

    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('== TRỤ CỘT B — EARLY WARNING: OLD L3 vs EWMA+CUSUM (NK-11) ==')

    out_dir = os.path.join(
        ROOT, 'test_results',
        f'early_warning_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. FAR/24h trên người khỏe + spike ----------
    n_cycles = int(HOURS_FAR * 3600 / DT)
    print(f'\n[1] FAR: {HOURS_FAR}h × {N_RUNS} seed, spike biểu cảm + burst…')
    far_old, far_new, watch = far_experiment(n_cycles)
    print(f'  OLD alert : {far_old:.1f} /24h · NEW DETERIORATING: '
          f'{far_new:.1f} /24h · NEW WATCH (mềm): {watch:.1f} /24h')

    # ---------- 2. Burst stress (đợt ngắn điểm cao) ----------
    print('[2] Burst stress 3–5 chu kỳ @75–85…')
    old_c = new_c = 0
    total_h = 0.0
    for r in range(N_RUNS):
        rng = np.random.default_rng(SEED + 200 + r)
        scores = ar1_noise(n_cycles, rng)
        scores = add_spikes(scores, rng, 10 * HOURS_FAR // 24,
                            lo=53, hi=63, max_len=5)   # 75–85 tuyệt đối
        old_c += count_episodes(run_old(scores)[WARMUP:], True)
        mon = make_monitor()
        clock = FakeClock()
        mon._clock = clock.time
        mon.calibrate(scores[:WARMUP])
        new_states = []
        for s in scores[WARMUP:]:
            clock.now += DT
            new_states.append(mon.update(float(s), t=clock.now)['state'])
        new_c += count_episodes(new_states, 'DETERIORATING')
        total_h += n_cycles * DT / 3600.0
    burst_old, burst_new = old_c / total_h * 24, new_c / total_h * 24
    print(f'  OLD: {burst_old:.1f} /24h · NEW: {burst_new:.1f} /24h')

    # ---------- 3. Latency các ramp ----------
    print('[3] Latency phát hiện suy giảm (20 seed mỗi kịch bản)…')
    ramps = {
        'ramp_+40/15ph': (40.0 / (15 * 60 / DT), RAMP_END),
        'ramp_+40/30ph': (40.0 / (30 * 60 / DT), RAMP_END),
        'ramp_+40/60ph': (40.0 / (60 * 60 / DT), RAMP_END),
        'ramp_+18/30ph_DUOI_NGUONG': (18.0 / (30 * 60 / DT),
                                      RAMP_BELOW_END),
    }
    lat = {}
    for name, (rate, end) in ramps.items():
        lat[name] = latency_experiment(rate, end)
        print(f"  {name}: OLD {lat[name]['old']} ({lat[name]['old_ever_fired']}) "
              f"| NEW {lat[name]['new']} ({lat[name]['new_ever_fired']})")

    res = {
        'protocol': 'simulated trajectories (seed 42, chu kỳ 2s, warm-up '
                    '200 chu kỳ) — chờ thay bằng replay Protocol B thật',
        'params': {'ewma_alpha': EarlyWarningMonitor.EWMA_ALPHA,
                   'cusum_k_sigma': EarlyWarningMonitor.CUSUM_K,
                   'cusum_h_sigma': EarlyWarningMonitor.CUSUM_H,
                   'confirm_cycles_30s': EarlyWarningMonitor.CONFIRM_CYCLES,
                   'elevated_sigma': EarlyWarningMonitor.ELEVATED_SIGMA,
                   'n_runs': N_RUNS, 'hours_far': HOURS_FAR,
                   'baseline_mu0': MU0, 'baseline_sigma': SIGMA},
        'far_healthy_per_24h': {'old_alert': round(far_old, 2),
                                'new_deteriorating': round(far_new, 2),
                                'new_watch_soft': round(watch, 2)},
        'far_burst_per_24h': {'old_alert': round(burst_old, 2),
                              'new_deteriorating': round(burst_new, 2)},
        'latency': lat,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---------- FIGURE 4 panel (1 run đại diện seed 42) ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    t_h = np.arange(n_cycles) * DT / 3600.0

    # panel 1: normal + spikes
    s1 = add_spikes(ar1_noise(n_cycles, rng), rng, 20 * HOURS_FAR // 24)
    mon = make_monitor(); clock = FakeClock(); mon._clock = clock.time
    mon.calibrate(s1[:WARMUP])
    ew1 = [mon.update(float(s), t=clock.now + i * DT)['ewma']
           for i, s in enumerate(s1)]

    # panel 2: ramp 30ph — mark thời điểm báo
    s2 = add_ramp(ar1_noise(n_cycles, rng), 40.0 / (30 * 60 / DT), RAMP_END)
    old2 = run_old(s2)
    mon = make_monitor(); clock = FakeClock(); mon._clock = clock.time
    mon.calibrate(s2[:WARMUP])
    ew2, new_hit2 = [], None
    for i, s in enumerate(s2):
        clock.now += DT
        st = mon.update(float(s), t=clock.now)
        ew2.append(st['ewma'])
        if st['state'] == 'DETERIORATING' and new_hit2 is None:
            new_hit2 = i
    old_hit2 = next((i for i in range(WARMUP, len(s2)) if old2[i]), None)

    # panel 3: ramp dưới ngưỡng
    s3 = add_ramp(ar1_noise(n_cycles, rng),
                  18.0 / (30 * 60 / DT), RAMP_BELOW_END)
    old3 = run_old(s3)
    mon = make_monitor(); clock = FakeClock(); mon._clock = clock.time
    mon.calibrate(s3[:WARMUP])
    ew3, new_hit3 = [], None
    for i, s in enumerate(s3):
        clock.now += DT
        st = mon.update(float(s), t=clock.now)
        ew3.append(st['ewma'])
        if st['state'] == 'DETERIORATING' and new_hit3 is None:
            new_hit3 = i
    old_hit3 = next((i for i in range(WARMUP, len(s3)) if old3[i]), None)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=150)
    ax = axes[0][0]
    ax.plot(t_h, s1, lw=0.5, color='#95a5a6', label='fused score (mô phỏng)')
    ax.plot(t_h, ew1, lw=1.8, color='#1a6faf', label='EWMA')
    ax.axhline(50, color='#c0392b', ls='--', lw=1, label='ngưỡng cảnh báo cũ (50)')
    ax.set_title(f'Người khỏe {HOURS_FAR}h + spike biểu cảm — NEW: '
                 f'{far_new:.1f}/24h · OLD: {far_old:.1f}/24h')
    ax.set_ylabel('điểm'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(t_h, s2, lw=0.5, color='#95a5a6')
    ax.plot(t_h, ew2, lw=1.8, color='#1a6faf', label='EWMA')
    ax.axhline(50, color='#c0392b', ls='--', lw=1)
    if old_hit2:
        ax.axvline(t_h[old_hit2], color='#7f8c8d', lw=2,
                   label=f'OLD báo: {t_h[old_hit2]*60:.0f} ph')
    if new_hit2:
        ax.axvline(t_h[new_hit2], color='#c0392b', lw=2,
                   label=f'NEW báo: {t_h[new_hit2]*60:.0f} ph')
    ax.set_title('Suy giảm +40 điểm / 30 phút (simulated)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(t_h, s3, lw=0.5, color='#95a5a6')
    ax.plot(t_h, ew3, lw=1.8, color='#1a6faf', label='EWMA')
    ax.axhline(50, color='#c0392b', ls='--', lw=1, label='ngưỡng cũ 50')
    if old_hit3:
        ax.axvline(t_h[old_hit3], color='#7f8c8d', lw=2, label='OLD báo')
    if new_hit3:
        ax.axvline(t_h[new_hit3], color='#c0392b', lw=2,
                   label=f'NEW báo: {t_h[new_hit3]*60:.0f} ph')
    ax.set_title('Suy giảm +18 điểm / 30 phút — chạm tối đa ~40 (DƯỚI ngưỡng):'
                 ' OLD KHÔNG BAO GIỜ báo')
    ax.set_xlabel('giờ'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][1]
    s4 = add_spikes(ar1_noise(n_cycles, rng), rng, 10 * HOURS_FAR // 24,
                    lo=53, hi=63)
    ax.plot(t_h, s4, lw=0.5, color='#95a5a6')
    ax.axhline(50, color='#c0392b', ls='--', lw=1)
    ax.set_title(f'Burst stress (đợt ngắn 75–85) — OLD {burst_old:.1f}/24h · '
                 f'NEW {burst_new:.1f}/24h')
    ax.set_xlabel('giờ'); ax.grid(alpha=0.3)

    fig.suptitle('Early Warning: EWMA+CUSUM vs luật L3 ngưỡng (dữ liệu '
                 'MÔ PHỎNG seed 42 — ghi rõ simulated)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'early_warning_trajectory.png'))

    print(f'\nĐã lưu: {out_dir} (JSON + PNG)')


if __name__ == '__main__':
    main()
