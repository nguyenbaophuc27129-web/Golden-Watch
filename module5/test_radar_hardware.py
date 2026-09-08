# -*- coding: utf-8 -*-
"""
TEST RADAR LD2450 THẬT — Module 5 (PSCS v8.0)
Dùng cho ngày lắp radar thật (fix L-12): kiểm tra UART, thu bằng chứng
10 kịch bản, xuất JSONL (dữ liệu thô từng sample) + CSV tổng hợp.

Quy trình tại nhà:
  1. Cắm radar + UART-USB. Chạy:  python module5/test_radar_hardware.py
  2. Menu liệt kê kịch bản — chọn số, Enter, THỰC HIỆN hành động, chờ đo xong.
  3. Kết quả lưu tại test_results/radar_hw/.

LƯU Ý AN TOÀN: kịch bản ngã = ngã xuống ĐỆM/gối mềm, có người bảo hộ.
KHÔNG mô phỏng hành vi nguy hiểm.

Usage:
    python module5/test_radar_hardware.py                 # menu
    python module5/test_radar_hardware.py --port COM5     # chỉ định port
    python module5/test_radar_hardware.py --monitor 30    # chỉ xem dữ liệu
"""

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from detection.radar_module import (RadarModule, LD2450Parser,       # noqa: E402
                                    FALL_DISPLACEMENT_CM,
                                    INACTIVITY_FALL_S,
                                    AUDIO_GATE_REDUCE)

OUT_DIR = os.path.join(parent_dir, 'test_results', 'radar_hw')

# (id, tên, thời gian đo s, kỳ vọng, audio_abnormal)
SCENARIOS = [
    ('S01_walk',      'Di bộ quanh phong binh thuong',          30, 'NORMAL',  False),
    ('S02_sit_stand', 'Ngoi xuong - dung len lap lai 5 lan',     30, 'NORMAL',  False),
    ('S03_sit_tv',    'Ngoi yen xem TV (khong di chuyen)',       30, 'NORMAL',  False),
    ('S04_wave',      'Va tay dong tac lon (False-alarm probe)', 20, 'NORMAL',  False),
    ('S05_nobody',    'Khong ai trong phong (chi qua/loc)',      20, 'NORMAL',  False),
    ('S06_stop',      'Di bo roi dung dot ngot',                 25, 'NORMAL',  False),
    ('S07_fall_mat',  'Nga xuong DEM tu the dung',               60, 'WARNING/DANGER', True),
    ('S08_fall_still','Nga roi nam bat dong 45s+ (co keu)',      75, 'DANGER',  True),
    ('S09_fall_quiet','Nga roi nam yen KHONG keu (gate audio)',  75, 'NORMAL/WARNING', False),
    ('S10_two_people','2 nguoi di trong cung phong',             30, 'NORMAL*', False),
]


def pick_port(arg_port):
    ports = RadarModule.list_ports()
    print("COM ports tìm thấy:", ports or "(không có)")
    if arg_port:
        return arg_port
    cand = [p for p in ports if 'COM' in str(p)]
    if len(cand) == 1:
        print(f"Tự chọn {cand[0]}")
        return cand[0]
    return input("Nhập COM port (vd COM5): ").strip()


def record(radar, name, duration, expected, audio_abn, nihss_note=''):
    """Thu 1 kịch bản: lưu từng sample + tính proxy giống analyze()."""
    radar.set_audio_flag(audio_abn)
    radar.last_position = None
    radar.last_move_time = None
    radar.sudden_change_at = None

    out_path = os.path.join(OUT_DIR, f"{name}_{datetime.now().strftime('%H%M%S')}.jsonl")
    samples = []
    metrics = {'position_change_cm': 0.0, 'n_samples': 0,
               'sudden_change': False, 'max_targets': 0}
    t0 = time.time()
    print(f"\n>>> BẮT ĐẦU đo [{name}] {duration}s — audio_gate={'ON' if audio_abn else 'OFF'}")
    print("    (thực hiện hành động NGAY khi thấy dấu >>>)")
    with open(out_path, 'w', encoding='utf-8') as f:
        while time.time() - t0 < duration:
            tg = radar.read_targets()
            now = round(time.time() - t0, 2)
            if tg:
                metrics['max_targets'] = max(metrics['max_targets'], len(tg))
                main = max(tg, key=lambda t: abs(t['y_cm']))
                pos = (main['x_cm'], main['y_cm'])
                if radar.last_position is not None:
                    disp = abs(pos[0] - radar.last_position[0]) + \
                           abs(pos[1] - radar.last_position[1])
                    metrics['position_change_cm'] = max(
                        metrics['position_change_cm'], float(disp))
                    if disp > 2:
                        radar.last_move_time = time.time()
                    if disp > FALL_DISPLACEMENT_CM:
                        radar.sudden_change_at = time.time()
                        metrics['sudden_change'] = True
                radar.last_position = pos
                f.write(json.dumps({'t': now, 'targets': tg},
                                   ensure_ascii=False) + '\n')
                if int(now * 2) % 10 == 0:
                    print(f"    >>> t={now:5.1f}s  main=({pos[0]:4d},{pos[1]:4d})cm  "
                          f"n={len(tg)}")
            else:
                f.write(json.dumps({'t': now, 'targets': []}) + '\n')
            metrics['n_samples'] += 1
            time.sleep(0.5)

    inactivity = (time.time() - radar.last_move_time
                  if radar.last_move_time else duration)
    fall_prob = 0.0
    if metrics['sudden_change']:
        fall_prob += 40.0
    if radar.sudden_change_at is not None:
        if time.time() - radar.sudden_change_at >= INACTIVITY_FALL_S:
            fall_prob += 60.0
    elif inactivity > INACTIVITY_FALL_S:
        fall_prob += 30.0
    gate = 'bypass'
    if fall_prob > 0 and not audio_abn:
        fall_prob *= AUDIO_GATE_REDUCE
        gate = 'audio NORM -> x0.3'
    elif fall_prob > 0:
        gate = 'audio ABN -> giu'
    fall_prob = min(round(fall_prob, 1), 100.0)
    status = ('NORMAL' if fall_prob < 30 else
              'WARNING' if fall_prob < 60 else 'DANGER')
    hit = 'YES' if expected.startswith(status) or status in expected else 'no'

    result = {'scenario': name, 'duration_s': duration,
              'expected': expected, 'actual': status,
              'fall_prob': fall_prob, 'gate': gate, 'match': hit,
              'position_change_cm': metrics['position_change_cm'],
              'sudden_change': metrics['sudden_change'],
              'inactivity_s': round(inactivity, 1),
              'n_samples': metrics['n_samples'],
              'max_targets': metrics['max_targets'],
              'audio_abnormal': audio_abn,
              'timestamp': datetime.now().isoformat(),
              'raw_file': os.path.basename(out_path)}
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'summary': result}, ensure_ascii=False) + '\n')

    print(f"    KẾT QUẢ: {status} (prob={fall_prob}) | kỳ vọng: {expected} "
          f"| khớp: {hit}")
    print(f"    Biến động max={metrics['position_change_cm']:.0f}cm | "
          f"bất hoạt={inactivity:.0f}s | targets max={metrics['max_targets']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=None)
    ap.add_argument('--monitor', type=int, default=0,
                    help='Chỉ xem dữ liệu N giây rồi thoát')
    args = ap.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("TEST HARDWARE LD2450 — 10 kịch bản bằng chứng (fix L-12)")
    print(f"Ngưỡng: biến động>{FALL_DISPLACEMENT_CM:.0f}cm, "
          f"bất hoạt>{INACTIVITY_FALL_S:.0f}s, audio gate x{AUDIO_GATE_REDUCE}")
    print("=" * 70)

    port = pick_port(args.port)
    radar = RadarModule(port=port, baud=256000, simulation=False)
    if radar.serial is None:
        print("\n[!] Không kết nối được radar. Kiểm tra:")
        print("    - Driver CH340/CP210x đã cài?")
        print("    - Baud phải là 256000 (không phải 115200)")
        print("    - Port không bị chương trình khác giữ")
        return

    # ---- Kiểm tra UART thô 10s ----
    print("\n[1/2] Kiểm tra UART thô 10 giây...")
    n_bytes, n_frames, t0 = 0, 0, time.time()
    while time.time() - t0 < 10:
        try:
            data = radar.serial.read(256)
            n_bytes += len(data)
            frames = radar.parser.feed(data)
            if frames:
                tg = LD2450Parser.parse_frame(frames[-1])
                if tg:
                    n_frames += 1
                    print(f"    t={time.time() - t0:4.1f}s targets="
                          f"{[(t['x_cm'], t['y_cm']) for t in tg]}")
        except Exception as e:
            print(f"    Lỗi đọc: {e}")
            break
    print(f"    → {n_bytes} bytes, {n_frames} frame có mục tiêu/10s")
    if n_bytes == 0:
        print("    [!] KHÔNG có byte nào — sai baud hoặc sai port.")
        print("        Thử lại: --port khác, hoặc kiểm baud 256000.")
        return

    if args.monitor:
        return

    # ---- Thu 10 kịch bản ----
    print("\n[2/2] THU 10 KỊCH BẢN (ngã xuống đệm, có người bảo hộ!)")
    results = []
    for sid, name, dur, exp, abn in SCENARIOS:
        ans = input(f"\n[{sid}] {name} ({dur}s). Sẵn sàng? Enter=đo, s=bỏ qua: ")
        if ans.strip().lower() == 's':
            continue
        results.append(record(radar, sid, dur, exp, abn))

    # ---- CSV tổng hợp ----
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUT_DIR, f'summary_{ts}.csv')
    if results:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        match = sum(1 for r in results if r['match'] == 'YES')
        print(f"\n===== TỔNG: {match}/{len(results)} kịch bản khớp kỳ vọng =====")
        print(f"CSV: {csv_path}")
    radar.disconnect() if hasattr(radar, 'disconnect') else None
    if hasattr(radar, 'serial') and radar.serial:
        radar.serial.close()


if __name__ == '__main__':
    main()
