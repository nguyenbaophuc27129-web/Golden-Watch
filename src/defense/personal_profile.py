# -*- coding: utf-8 -*-
"""
PERSONAL PROFILE — CHẾ ĐỘ HỌC 3 NGÀY KHI LẮP TẠI NHÀ (ý tưởng của đội)

Mục đích: mỗi người có "đặc điểm riêng" (gù lưng, cong vẹo cột sống, nhech mép
thói quen, nét mặt bất đối xứng bẩm sinh...). 3 ngày đầu lắp đặt, hệ thống
chỉ QUAN SÁT + GHI (không cảnh báo giả từ khác biệt cá nhân); hết 3 ngày →
tính baseline CÁ NHÂN (median từng chỉ số) → DefenseEngine.calibrate() bật
L4 Adaptive Floor (vùng xám [30,40) trừ 5 điểm chỉ khi đã có baseline).

Nguyên tắc trung thực:
- CHỈ ghi số liệu tổng hợp (median ngày), KHÔNG lưu hình ảnh.
- Học là baseline thống kê đơn giản (median) — không phải deep learning.
- Người dùng xóa profile bằng cách xóa file JSON.

Chạy: dùng trong web_server.py / app_family.py; file:
    models/personal_profile.json
"""

import os
import json
import time
from collections import defaultdict

import numpy as np

FACE_KEYS = ('mouth_ratio', 'eye_ratio', 'face_tilt', 'nasolabial_ratio',
             'forehead_ratio')


class PersonalProfile:
    """Ghi + tổng hợp đặc điểm thân thể cá nhân trong 3 ngày đầu."""

    DAYS = 3
    MIN_SAMPLES_PER_DAY = 50      # chu kỳ phân tích (mỗi chu kỳ ~2s)

    def __init__(self, path):
        self.path = path
        self.data = {'started': None, 'days': {}, 'applied': False}
        self._today = None
        self._n_today = 0                   # tổng mẫu ghi hôm nay
        self._buffers = defaultdict(list)   # key → list giá trị hôm nay
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, encoding='utf-8') as f:
                    self.data = json.load(f)
                self.data.setdefault('days', {})
                self.data.setdefault('applied', False)
        except Exception as e:
            print(f"[PROFILE] Không đọc được profile ({e}) — tạo mới")

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"[PROFILE] Lỗi lưu: {e}")

    # ------------------------------------------------------------------
    @property
    def day_number(self):
        """Ngày học hiện tại 1..3 (tính từ lần đầu chạy)."""
        if not self.data['started']:
            return 1
        started = time.strptime(self.data['started'], '%Y-%m-%d')
        delta = (time.localtime().tm_yday - started.tm_yday)
        return min(max(delta + 1, 1), self.DAYS)

    def _flush_today(self):
        """Gom buffer hôm nay vào data['days'][ngày] (median từng chỉ số)."""
        today = time.strftime('%Y-%m-%d')
        if not self._buffers:
            return
        day_stats = {}
        for key, vals in self._buffers.items():
            if vals:
                day_stats[key] = round(float(np.median(vals)), 3)
        # gộp nếu cùng ngày chạy nhiều phiên
        merged = self.data['days'].get(today, {})
        n_before = merged.pop('_n', 0)
        for k, v in day_stats.items():
            merged[k] = v          # median gần nhất thắng (đủ dùng)
        merged['_n'] = n_before + len(next(iter(self._buffers.values()), []))
        self.data['days'][today] = merged
        self._buffers.clear()
        self.save()

    def record(self, face_r, gait_score=0.0, motion=0.0):
        """
        Ghi 1 chu kỳ phân tích. Gọi mỗi ~2s khi hệ thống chạy.
        Trả về dict trạng thái: {'day', 'done', 'samples_today', 'today_n'}
        """
        if not self.data['started']:
            self.data['started'] = time.strftime('%Y-%m-%d')
        # sang ngày mới → flush buffer cũ
        today = time.strftime('%Y-%m-%d')
        if self._today and today != self._today:
            self._flush_today()
            self._n_today = 0
        self._today = today
        self._n_today += 1

        rm = (face_r or {}).get('raw_metrics') or {}
        for key in FACE_KEYS:
            v = rm.get(key)
            if v is not None:
                self._buffers[key].append(float(v))
        g = rm.get('gait_score')
        if gait_score:
            self._buffers['gait_score'].append(float(gait_score))
        self._buffers['motion'].append(float(motion))

        # flush định kỳ mỗi 25 mẫu (đỡ ghi đĩa)
        n = len(next(iter(self._buffers.values()), []))
        if n and n % 25 == 0:
            self._flush_today()

        st = self.status()
        # đủ 3 ngày → flush lần cuối
        # (L-36: trước đây đọc st['days_ok'] — status() trả 'done' → ngày
        # thứ 3 trở đi KeyError MỌI chu kỳ, dashboard đóng băng đúng lúc
        # học xong; ngày 1–2 chỉ may nhờ short-circuit của `and`)
        if st['day'] >= self.DAYS and st['done']:
            self._flush_today()
            st = self.status()
        return st

    # ------------------------------------------------------------------
    def status(self):
        today = time.strftime('%Y-%m-%d')
        n_today = self._n_today
        full_days = [d for d, s in self.data['days'].items()
                     if s.get('_n', 0) >= self.MIN_SAMPLES_PER_DAY
                     and d != today]
        days_ok = len(full_days) + (1 if n_today else 0) >= self.DAYS
        return {'day': self.day_number, 'done': days_ok,
                'days_recorded': len(self.data['days']),
                'samples_today': n_today, 'today': today,
                'applied': self.data['applied']}

    def baseline(self):
        """Baseline cá nhân (median giữa các ngày) — None nếu chưa đủ."""
        st = self.status()
        if not st['done']:
            return None
        keys = set()
        for s in self.data['days'].values():
            keys.update(k for k in s if not k.startswith('_'))
        base = {}
        for k in keys:
            vals = [s[k] for s in self.data['days'].values() if k in s]
            if vals:
                base[k] = round(float(np.median(vals)), 3)
        return base or None

    def mark_applied(self):
        self.data['applied'] = True
        self.save()


# ======================================================================
# TEST
# ======================================================================
def test_personal_profile():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = PersonalProfile(os.path.join(td, 'pp.json'))
        face = {'raw_metrics': {'mouth_ratio': 10.0, 'eye_ratio': 5.0,
                                'face_tilt': 3.0, 'nasolabial_ratio': 8.0,
                                'forehead_ratio': 4.0}}
        for i in range(60):
            st = p.record(face, gait_score=12.0, motion=0.1)
        assert st['samples_today'] == 60, st
        assert st['done'] is False, 'chưa đủ 3 ngày'
        base = p.baseline()
        assert base is None, 'chưa đủ 3 ngày phải chưa có baseline'
        print('PERSONAL PROFILE TEST: PASS —',
              p.status())


if __name__ == '__main__':
    test_personal_profile()
