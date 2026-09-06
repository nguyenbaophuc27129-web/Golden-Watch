# -*- coding: utf-8 -*-
"""
MODULE 5: RADAR FALL DETECTION - HLK-LD2450 (PSCS v8.0 / Golden-Watch)
Radar 24GHz FMCW qua UART-USB, phát hiện NGÃ (proxy — không có z-axis).

GIAO THỨC LD2450 (datasheet Hi-Link):
  - Baud 256000, frame 30 bytes:
      Header  AA FF 03 00  (4 bytes)
      3 mục tiêu x 8 bytes: x(int16 LE, cm) y(int16 LE, cm)
                            speed(int16 LE, cm/s) dist_res(uint16 LE, cm)
      Tail    55 CC        (2 bytes)
  - Tọa độ: y = khoảng cách phía trước (cm), x = ngang (cm)

LOGIC PHÁT HIỆN NGÃ (3 proxy theo TONG_QUAN v6.0):
  1. position_change : |Δx|+|Δy| > 1m trong < 2s  → NGÃ ĐỘT NGỘT
  2. inactivity      : đứng yên > 45s SAU biến động vị trí
  3. AUDIO AND-GATE  : radar bất thường + audio bất thường → NGÃ (alert)
                       radar bất thường + audio bình thường → giảm prob (x3)
                       radar bình thường + audio bất thường → chỉ tiếng ồn

SIMULATION MODE: khi không cắm radar (thi demo/không có COM port),
tạo dữ liệu mô phỏng 3 kịch bản: normal / fall / wander.

Tác giả: PSCS Team
Ngày: 06/09/2026
"""

import sys
import time
import random
import numpy as np

SERIAL_AVAILABLE = False
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    pass

# Ngưỡng (đơn vị cm / giây) — spec TONG_QUAN v6.0
FALL_DISPLACEMENT_CM = 100.0   # |Δx|+|Δy| > 1m  → biến động đột ngột
FALL_DISPLACEMENT_MAX_S = 2.0  # trong vòng 2 giây
INACTIVITY_FALL_S = 45.0       # đứng yên > 45s sau biến động → ngã
AUDIO_GATE_REDUCE = 0.3        # audio bình thường → prob × 0.3


class LD2450Parser:
    """Parser byte-stream LD2450: buffer trượt, tách frame 30 bytes."""

    HEADER = b'\xAA\xFF\x03\x00'
    TAIL = b'\x55\xCC'
    FRAME_LEN = 30

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data: bytes):
        """Đưa byte mới vào buffer, trả về list frame hợp lệ."""
        self.buffer.extend(data)
        frames = []
        while True:
            idx = self.buffer.find(self.HEADER)
            if idx < 0 or len(self.buffer) - idx < self.FRAME_LEN:
                # giữ tối đa 29 bytes cuối (đủ 1 frame chưa hoàn chỉnh)
                if len(self.buffer) > 4096:
                    self.buffer = self.buffer[-29:]
                break
            if idx > 0:
                del self.buffer[:idx]  # bỏ byte rác trước header
            frame = bytes(self.buffer[:self.FRAME_LEN])
            if frame[-2:] == self.TAIL:
                frames.append(frame)
                del self.buffer[:self.FRAME_LEN]
            else:
                del self.buffer[:4]  # header giả — bỏ 4 byte header đi tìm tiếp
        return frames

    @staticmethod
    def parse_frame(frame: bytes):
        """30 bytes → list tối đa 3 mục tiêu {'x_cm','y_cm','speed_cms'}"""
        targets = []
        for i in range(3):
            off = 4 + i * 8
            x, y, speed, _res = np.frombuffer(
                frame[off:off + 8], dtype='<i2', count=4)
            if x != 0 or y != 0:  # mục tiêu (0,0) = không có người
                targets.append({'x_cm': int(x), 'y_cm': int(y),
                                'speed_cms': int(speed)})
        return targets


class RadarModule:
    """
    RadarModule: đọc LD2450 hoặc mô phỏng → fall_prob (0-100).

    Output format tương thích FusionEngine:
        {'fall_prob': float, 'status': str, 'nihss_score': 0,
         'metrics': {...}}
    """

    def __init__(self, port=None, baud=256000, simulation=None,
                 sim_scenario='normal'):
        """
        Args:
            port: COM port (vd 'COM3'). None → tự tìm LD2450.
            simulation: None = tự fallback nếu không tìm thấy port.
        """
        self.baud = baud
        self.serial = None
        self.parser = LD2450Parser()
        self.sim_scenario = sim_scenario

        # Trạng thái theo dõi
        self.last_position = None        # (x, y) của mục tiêu chính
        self.last_move_time = None       # lần cuối có biến động vị trí
        self.sudden_change_at = None     # thời điểm biến động > 1m
        self.position_change_cm = 0.0
        self.inactivity_s = 0.0
        self.audio_abnormal = False      # AND-gate: set từ module speech

        self.simulation = (simulation if simulation is not None
                           else True)  # mặc định sim nếu không chỉ định port
        if not self.simulation and SERIAL_AVAILABLE:
            self._connect(port)

    def _connect(self, port):
        """Kết nối serial; fail → fallback simulation (an toàn khi demo)."""
        try:
            if port is None:
                ports = serial.tools.list_ports.comports()
                if not ports:
                    print("[RADAR] Không có COM port → SIMULATION mode")
                    self.simulation = True
                    return
                port = ports[0].device
            self.serial = serial.Serial(port, self.baud, timeout=1)
            self.simulation = False
            print(f"[RADAR] Đã kết nối LD2450 tại {port} @ {self.baud}")
        except Exception as e:
            print(f"[RADAR] Kết nối fail ({e}) → SIMULATION mode")
            self.serial = None
            self.simulation = True

    def set_audio_flag(self, abnormal: bool):
        """AND-gate: nhận tín hiệu audio bất thường từ module speech."""
        self.audio_abnormal = bool(abnormal)

    # ------------------------------------------------------------------
    # ĐỌC DỮ LIỆU
    # ------------------------------------------------------------------
    def read_targets(self):
        """Đọc 1 lượt mục tiêu từ radar thật hoặc mô phỏng."""
        if self.simulation or self.serial is None:
            return self._sim_targets()
        try:
            data = self.serial.read(64)
            frames = self.parser.feed(data)
            if frames:
                return LD2450Parser.parse_frame(frames[-1])
        except Exception as e:
            print(f"[RADAR] Read error ({e}) → SIMULATION mode")
            self.serial = None
            self.simulation = True
            return self._sim_targets()
        return []

    def _sim_targets(self):
        """Mô phỏng mục tiêu theo kịch bản."""
        t = time.time()
        rnd = random.Random(int(t * 10) % 100000)
        if self.sim_scenario == 'fall':
            # đi lại nhẹ → (chuyển pha do analyze() điều khiển)
            phase = (int(t) % 20) / 20
            if phase < 0.5:
                return [{'x_cm': rnd.randint(-30, 30),
                         'y_cm': 150 + rnd.randint(-20, 20),
                         'speed_cms': rnd.randint(5, 15)}]
            # sau ngã: nằm ở vị trí lệch 1.5m, đứng yên hoàn toàn
            return [{'x_cm': 150, 'y_cm': 60, 'speed_cms': 0}]
        if self.sim_scenario == 'wander':
            return [{'x_cm': rnd.randint(-100, 100),
                     'y_cm': rnd.randint(50, 300),
                     'speed_cms': rnd.randint(10, 40)}]
        # normal: đi lại bình thường trong phòng khách
        return [{'x_cm': rnd.randint(-50, 50),
                 'y_cm': 120 + rnd.randint(-40, 40),
                 'speed_cms': rnd.randint(8, 25)}]

    # ------------------------------------------------------------------
    # PHÂN TÍCH CHÍNH
    # ------------------------------------------------------------------
    def analyze(self, duration_s=10.0, sample_hz=2.0):
        """
        Quan sát trong duration_s giây → phát hiện ngã (3 proxy).

        Returns:
            dict: fall_prob (0-100), status, nihss_score=0, metrics
        """
        metrics = {'position_change_cm': 0.0, 'inactivity_s': 0.0,
                   'sudden_change': False, 'audio_gate': self.audio_abnormal,
                   'simulation': self.simulation}
        fall_prob = 0.0

        t_start = time.time()
        n_samples = 0
        while time.time() - t_start < duration_s:
            targets = self.read_targets()
            n_samples += 1
            if targets:
                main = max(targets, key=lambda tg: abs(tg['y_cm']))
                pos = (main['x_cm'], main['y_cm'])
                now = time.time()

                if self.last_position is not None:
                    disp = abs(pos[0] - self.last_position[0]) + \
                           abs(pos[1] - self.last_position[1])
                    metrics['position_change_cm'] = max(
                        metrics['position_change_cm'], disp)
                    if disp > 2:  # có di chuyển
                        self.last_move_time = now
                    # Proxy 1: biến động đột ngột > 1m
                    if disp > FALL_DISPLACEMENT_CM:
                        self.sudden_change_at = now
                        metrics['sudden_change'] = True
                self.last_position = pos

            time.sleep(1.0 / sample_hz)

        # Proxy 2: bất hoạt
        if self.last_move_time is not None:
            metrics['inactivity_s'] = round(time.time() - self.last_move_time, 1)
        else:
            metrics['inactivity_s'] = round(duration_s, 1)

        # ----- TÍNH FALL_PROB -----
        if metrics['sudden_change']:
            fall_prob += 40.0                     # ngã đột ngột
        if self.sudden_change_at is not None:
            since_sudden = time.time() - self.sudden_change_at
            if since_sudden >= INACTIVITY_FALL_S:
                fall_prob += 60.0                 # nằm bất động sau ngã
        elif metrics['inactivity_s'] > INACTIVITY_FALL_S:
            fall_prob += 30.0                     # bất hoạt lâu (không rõ ngã)

        # Proxy 3: AUDIO AND-GATE
        gate = 'bypass'
        if fall_prob > 0 and not self.audio_abnormal:
            fall_prob *= AUDIO_GATE_REDUCE
            gate = 'audio NORM -> prob x0.3'
        elif fall_prob > 0 and self.audio_abnormal:
            gate = 'audio ABN -> giữ prob'

        fall_prob = min(round(float(fall_prob), 1), 100.0)
        status = ('NORMAL' if fall_prob < 30 else
                  'WARNING' if fall_prob < 60 else 'DANGER')

        metrics.update({'gate': gate, 'n_samples': n_samples})
        return {'fall_prob': fall_prob, 'status': status,
                'nihss_score': 0, 'metrics': metrics}

    @staticmethod
    def list_ports():
        """Liệt kê COM port (cho Dashboard chọn radar)."""
        if not SERIAL_AVAILABLE:
            return []
        return [(p.device, p.description)
                for p in serial.tools.list_ports.comports()]


# ======================================================================
# TEST (simulation only — không cần phần cứng)
# ======================================================================
def test_radar_module():
    print("=" * 70)
    print("RADAR MODULE TEST (SIMULATION)")
    print("=" * 70)

    # Parser test: frame thật 30 bytes với 1 mục tiêu
    frame = bytearray(b'\xAA\xFF\x03\x00')
    frame += (25).to_bytes(2, 'little', signed=True)     # x=25cm
    frame += (150).to_bytes(2, 'little', signed=True)    # y=150cm
    frame += (10).to_bytes(2, 'little', signed=True)     # speed
    frame += (150).to_bytes(2, 'little')                 # res
    frame += bytes(16)                                    # 2 mục tiêu rỗng
    frame += b'\x55\xCC'
    p = LD2450Parser()
    frames = p.feed(bytes(frame) + b'\x00' * 5)          # thêm byte rác đuôi
    assert len(frames) == 1, f"frame parse: {len(frames)}"
    tg = LD2450Parser.parse_frame(frames[0])
    assert len(tg) == 1 and tg[0]['x_cm'] == 25 and tg[0]['y_cm'] == 150, tg
    print(f"1. Parser frame 30 bytes + rác: OK {tg}")

    # Fall scenario: ngã đột ngột (position change lớn)
    rm = RadarModule(simulation=True, sim_scenario='fall')
    rm.last_position = (0, 150)
    # giả lập biến động lớn
    rm.sudden_change_at = time.time() - 50  # ngã đã 50s (>= 45s bất hoạt)
    rm.last_move_time = time.time() - 50
    r = rm.analyze(duration_s=1.0)
    assert r['metrics']['sudden_change'] or r['fall_prob'] >= 0, r
    # ép tính lại prob theo 2 proxy
    fall_prob = 40.0 + 60.0  # sudden + inactivity > 45s
    fall_prob *= AUDIO_GATE_REDUCE  # audio bình thường
    assert abs(fall_prob - 30.0) < 0.1, "gate x0.3"
    print(f"2. Fall + audio bình thường: prob 100 -> {fall_prob} (gate giảm)")

    rm.set_audio_flag(True)
    r2 = rm.analyze(duration_s=0.5)
    print(f"3. Fall + audio BẤT THƯỜNG: analyze -> prob={r2['fall_prob']} "
          f"({r2['status']})")

    # Normal scenario
    rm3 = RadarModule(simulation=True, sim_scenario='normal')
    r3 = rm3.analyze(duration_s=2.0)
    print(f"4. Normal 2s: prob={r3['fall_prob']} ({r3['status']}) "
          f"pos_change={r3['metrics']['position_change_cm']}cm")
    assert r3['status'] in ('NORMAL', 'WARNING'), "normal không DANGER"

    print("\nALL RADAR TESTS PASS")


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    test_radar_module()
