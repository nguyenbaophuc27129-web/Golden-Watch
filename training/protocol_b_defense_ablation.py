# -*- coding: utf-8 -*-
"""
PROTOCOL B — DEFENSE ABLATION HARNESS (nâng tầm #2)
Đo FPR của TỪNG LỚP defense trên tình huống người khỏe (cười, ngáp, quay
đầu, ra vào khung, 2 người, vẫy tay...) → đường cong "mỗi lớp giảm bao
nhiêu % báo giả".

2 lệnh:
  capture: python training/protocol_b_defense_ablation.py capture --out file.jsonl
           Chạy camera thật, chu kỳ 2s. Phím (dính cho tới khi đổi):
             1 binh_thuong   2 cuoi_lon    3 ngap
             4 nghieng_dau   5 ra_vao_khung 6 hai_nguoi
             7 vay_tay       8 anh_sang_yeu
             q dừng
  eval:    python training/protocol_b_defense_ablation.py eval file.jsonl
           Replay JSONL qua FusionEngine + DefenseEngine THẬT ×4 cấu hình:
             off      : không defense (đối chứng)
             L1+L4    : calibrate từ đoạn binh_thuong (L4 vùng xám) — L1/L4
                        ghép cặp trong engine (calibrate → adaptive floor)
             +L2      : thêm context từ motion đã ghi
             full(+L3): thêm temporal verdict — engine thật chạy với CLOCK ẢO
                        (giả lập 2s/chu kỳ) để tái lập persistence 30s/60%
Kết quả: test_results/defense_ablation_<ts>/ (JSON + CSV + PNG).

Nguyên tắc: model/module KHÔNG đổi — chỉ bật/tắt tầng defense.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
for p in (SRC,):
    if p not in sys.path:
        sys.path.insert(0, p)

SCENARIOS = {ord('1'): 'binh_thuong', ord('2'): 'cuoi_lon', ord('3'): 'ngap',
             ord('4'): 'nghieng_dau', ord('5'): 'ra_vao_khung',
             ord('6'): 'hai_nguoi', ord('7'): 'vay_tay',
             ord('8'): 'anh_sang_yeu'}
CYCLE_S = 2.0


# ======================================================================
# CAPTURE
# ======================================================================
def cmd_capture(out_path, minutes):
    import cv2
    import numpy as np
    from detection.face_module_v7 import FaceAsymmetryDetector
    from detection.face_ml_v3 import FaceMLV3
    from detection.arm_module import ArmWeaknessDetector
    from detection.gait_module import GaitPoseDetector

    models = os.path.join(ROOT, 'models')
    face = FaceAsymmetryDetector(model_path=os.path.join(
        models, 'face_landmarker_v2.task'))
    face.ml_model = FaceMLV3.load_latest(models)
    arm = ArmWeaknessDetector(
        pose_model_path=os.path.join(SRC, 'yolov8n-pose.pt'),
        ml_model_path=os.path.join(
            models, 'arm_weakness_20260830_200657.pth'),
        scaler_path=os.path.join(
            models, 'arm_weakness_20260830_200657_scaler.pkl'))
    gait = GaitPoseDetector(model_path=os.path.join(SRC, 'yolov8n-pose.pt'))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('KHÔNG MỞ ĐƯỢC CAMERA'); sys.exit(1)
    print(f'Capture → {out_path} (dừng sau {minutes} phút hoặc phím q)')
    print('Phím: 1 binh_thuong · 2 cuoi_lon · 3 ngap · 4 nghieng_dau · '
          '5 ra_vao_khung · 6 hai_nguoi · 7 vay_tay · 8 anh_sang_yeu')

    prev_gray, scenario, n = None, 'binh_thuong', 0
    f_out = open(out_path, 'w', encoding='utf-8')
    t0 = time.time()
    while (time.time() - t0) < minutes * 60:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)),
                            cv2.COLOR_BGR2GRAY)
        motion = 0.0
        if prev_gray is not None:
            motion = min(float(np.mean(np.abs(
                gray.astype(np.int16) - prev_gray.astype(np.int16))))
                / 50.0, 1.0)
        prev_gray = gray

        face_r = face.process_frame(frame)
        arm_r = arm.detect_arm_weakness(frame)
        gait_r = gait.detect_gait_from_frame(frame)

        def keep(r, keys):
            r = r or {}
            return {k: r.get(k) for k in keys if r.get(k) is not None}
        row = {
            't': round(time.time() - t0, 2), 'scenario': scenario,
            'motion': round(motion, 3),
            'face': keep(face_r, ['score', 'status', 'metrics']),
            'arm': keep(arm_r, ['arm_prob', 'status']),
            'gait': keep(gait_r, ['gait_prob', 'status']),
        }
        f_out.write(json.dumps(row, ensure_ascii=False) + '\n')
        n += 1

        disp = frame.copy()
        cv2.putText(disp, f'[{scenario}]  cycles={n}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Protocol B capture', disp)
        k = cv2.waitKey(int(CYCLE_S * 1000)) & 0xFF
        if k == ord('q'):
            break
        scenario = SCENARIOS.get(k, scenario)

    cap.release(); cv2.destroyAllWindows(); f_out.close()
    print(f'Xong: {n} chu kỳ → {out_path}')


# ======================================================================
# EVAL — replay qua engine THẬT với clock ảo cho L3
# ======================================================================
class FakeClock:
    """Thay defense_engine.time để verdict() thấy thời gian giả lập."""
    def __init__(self):
        self.now = 0.0
    def time(self):
        return self.now


def _load_rows(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _calibrate_from_normal(rows):
    """L1: median đặc trưng mặt của các chu kỳ binh_thuong."""
    vals = [r['face']['metrics'].get('mouth_ratio') for r in rows
            if r['scenario'] == 'binh_thuong'
            and r.get('face', {}).get('metrics')]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals.sort()
    m = vals[len(vals) // 2]
    return {'face_asym': m}


def cmd_eval(in_path, out_dir):
    import numpy as np
    import defense.defense_engine as de_mod
    from defense.defense_engine import DefenseEngine
    from fusion.fusion_engine import FusionEngine

    rows = _load_rows(in_path)
    if not rows:
        print('File rỗng'); sys.exit(1)
    dt = float(np.median(np.diff([r['t'] for r in rows]))) \
        if len(rows) > 2 else CYCLE_S

    calib = _calibrate_from_normal(rows)

    def replay(with_l1, with_l2, with_l3):
        clock = FakeClock()
        de_mod.time = clock if with_l3 else time
        de = DefenseEngine()
        if with_l1 and calib:
            de.calibrate(**calib)
        fus = FusionEngine()
        alerts, emergency = 0, 0
        per_scn = {}
        for r in rows:
            clock.now += dt
            if with_l2:
                de.update_context(motion_intensity=r['motion'],
                                  talking=False)
            mods = {k: r[k] for k in ('face', 'arm', 'gait') if r.get(k)}
            filtered, _ = de.filter(mods)
            fused = fus.fuse(filtered)
            score = fused['fused_score']
            if with_l3:
                alert = de.verdict(score)['alert']
            else:
                alert = score >= 50
            alerts += alert
            emergency += (alert and score >= 70)
            s = per_scn.setdefault(r['scenario'], [0, 0])
            s[0] += alert; s[1] += 1
        return {'alert_rate': round(100.0 * alerts / len(rows), 2),
                'emergency_rate': round(100.0 * emergency / len(rows), 2),
                'n_cycles': len(rows), 'per_scenario': per_scn}

    configs = {
        'off':            dict(with_l1=False, with_l2=False, with_l3=False),
        'L1+L4':          dict(with_l1=True,  with_l2=False, with_l3=False),
        'L1+L2+L4':       dict(with_l1=True,  with_l2=True,  with_l3=False),
        'full(+L3)':      dict(with_l1=True,  with_l2=True,  with_l3=True),
    }
    results = {}
    print(f'\n{"Cấu hình":<12}{"FPR alert":>11}{"EMERGENCY giả":>15}'
          f'{"n":>6}')
    for name, kw in configs.items():
        res = replay(**kw)
        if kw['with_l1'] and not calib:
            res['note'] = 'không có đoạn binh_thuong để calibrate'
        results[name] = res
        print(f'{name:<12}{res["alert_rate"]:>10.2f}%'
              f'{res["emergency_rate"]:>14.2f}%{res["n_cycles"]:>6}')

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    base = os.path.join(out_dir, f'defense_ablation_{ts}')
    with open(base + '.json', 'w', encoding='utf-8') as f:
        json.dump({'source': in_path, 'cycle_s': dt, 'calibration': calib,
                   'results': results}, f, ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(configs)
    a = [results[n]['alert_rate'] for n in names]
    e = [results[n]['emergency_rate'] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=170)
    ax.bar(x - 0.18, a, 0.36, label='Báo động (≥50)', color='#1a6faf')
    ax.bar(x + 0.18, e, 0.36, label='EMERGENCY giả (≥70)', color='#c0392b')
    for xi, v in zip(x - 0.18, a):
        ax.text(xi, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)
    for xi, v in zip(x + 0.18, e):
        ax.text(xi, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('% chu kỳ báo động trên NGƯỜI KHỎE')
    ax.set_title('Defense ablation (Protocol B) — FPR theo từng lớp')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(base + '.png')

    print(f'\nĐã lưu: {base}.json + .png')
    print('Mục tiêu KH 5.3.1b: EMERGENCY giả ≤5%, tổng báo giả ≤15%.')


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c1 = sub.add_parser('capture')
    c1.add_argument('--out', required=True)
    c1.add_argument('--minutes', type=float, default=30)
    c2 = sub.add_parser('eval')
    c2.add_argument('jsonl')
    c2.add_argument('--out-dir',
                    default=os.path.join(ROOT, 'test_results'))
    args = ap.parse_args()
    if args.cmd == 'capture':
        cmd_capture(args.out, args.minutes)
    else:
        cmd_eval(args.jsonl, args.out_dir)
