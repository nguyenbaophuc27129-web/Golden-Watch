# -*- coding: utf-8 -*-
"""
BENCHMARK PROTOTYPE — RTX 3050 (máy chủ) + C270 + LD2450 (prototype_benchmark.py)

Đo SỐ LIỆU THẬT cho mục "Chế tạo và kiểm tra" (20đ):
  1. Hệ thống    : GPU/CUDA, VRAM, CPU
  2. Camera C270 : độ phân giải thật, FPS đọc liên tục
  3. Radar       : COM port, thời gian 1 nhịp phân tích
  4. Độ trễ từng module (ms): face / arm / gait (CPU vs CUDA) / fusion+defense+NIHSS
  5. Chu kỳ tổng  : ms mean/p95 — biểu đồ PNG

Chạy:  python module5/prototype_benchmark.py [--cycles 20]
Output: test_results/prototype_benchmark_<ts>.json + _latency.png
"""

import os
import sys
import glob
import json
import time
import platform
import argparse
import numpy as np
import cv2
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

OUT_DIR = os.path.join(parent_dir, 'test_results')
FACE_DS = (r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\data"
           r"\datasets\face\Annotated stroke and non stroke Dataset")


def load_face_frame():
    """Ảnh mặt thật (không phải nhiễu) để đo latency có ý nghĩa."""
    for sub in ('NonStroke', 'Stroke'):
        js = sorted(glob.glob(os.path.join(FACE_DS, sub, '*.jpg')))
        if js:
            img = cv2.imread(js[0])
            if img is not None:
                return cv2.resize(img, (640, 480))
    return np.full((480, 640, 3), 128, np.uint8)


def stats_ms(times):
    a = np.asarray(times, float)
    return {'mean_ms': round(float(a.mean()), 1),
            'p95_ms': round(float(np.percentile(a, 95)), 1),
            'max_ms': round(float(a.max()), 1)}


def bench(name, fn, cycles, warmup=3):
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass
    ts = []
    for _ in range(cycles):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {'module': name, **stats_ms(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cycles', type=int, default=20)
    args = ap.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    import torch
    report = {'created': datetime.now().isoformat(),
              'cycles': args.cycles, 'sections': {}}

    # ================= 1. HỆ THỐNG =================
    sysinfo = {'python': platform.python_version(),
               'torch': torch.__version__,
               'cuda_available': torch.cuda.is_available(),
               'cpu_cores': os.cpu_count()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        sysinfo.update({'gpu': p.name, 'vram_total_mb': total // (1 << 20),
                        'vram_free_mb': free // (1 << 20)})
    report['sections']['system'] = sysinfo
    print('[1] Hệ thống:', sysinfo)
    if not torch.cuda.is_available():
        print('    ⚠️  CUDA KHÔNG bật — máy chủ đang chạy CPU-only. '
              'Cài: pip install --force-reinstall torch '
              '--index-url https://download.pytorch.org/whl/cu126')

    # ================= 2. CAMERA C270 =================
    cam = {'status': 'SKIPPED'}
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # C270 max 720p
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            n = 30
            for _ in range(n):
                cap.read()
            fps = n / (time.perf_counter() - t0)
            cam = {'status': 'OK', 'width': w, 'height': h,
                   'read_fps': round(fps, 1)}
        else:
            cam = {'status': 'NO_SIGNAL'}
        cap.release()
    except Exception as e:
        cam = {'status': 'ERROR', 'error': str(e)}
    report['sections']['camera_c270'] = cam
    print('[2] Camera C270:', cam)

    # ================= 3. RADAR =================
    radar_info = {'ports': []}
    try:
        from detection.radar_module import RadarModule
        radar_info['ports'] = RadarModule.list_ports()
        rm = RadarModule()  # tự tìm LD2450, không thấy → SIM
        ts = [rm.analyze(duration_s=0.3, sample_hz=3.0)['status']
              for _ in range(5)]
        radar_info.update({'simulation': rm.simulation,
                           'statuses_5nhịp': ts,
                           'analysis_cycle_s': 0.3})
    except Exception as e:
        radar_info['error'] = str(e)
    report['sections']['radar_ld2450'] = radar_info
    print('[3] Radar:', radar_info)

    # ================= 4-5. LATENCY CÁC MODULE =================
    frame = load_face_frame()
    rows = []
    try:
        from detection.face_module_v7 import FaceAsymmetryDetector
        face = FaceAsymmetryDetector(
            model_path=os.path.join(parent_dir, 'models',
                                    'face_landmarker_v2.task'))
        rows.append(bench('face_mediapipe',
                          lambda: face.process_frame(frame), args.cycles))
    except Exception as e:
        rows.append({'module': 'face_mediapipe', 'error': str(e)})

    try:
        from detection.arm_module import ArmWeaknessDetector
        arm = ArmWeaknessDetector(
            pose_model_path=os.path.join(parent_dir, 'src', 'yolov8n-pose.pt'),
            ml_model_path=os.path.join(
                parent_dir, 'models', 'arm_weakness_20260830_200657.pth'),
            scaler_path=os.path.join(
                parent_dir, 'models',
                'arm_weakness_20260830_200657_scaler.pkl'))
        rows.append(bench('arm_yolo+ml',
                          lambda: arm.detect_arm_weakness(frame), args.cycles))
    except Exception as e:
        rows.append({'module': 'arm_yolo+ml', 'error': str(e)})

    gait_dev = {'cpu': None, 'cuda': None}
    try:
        from detection.gait_module import GaitPoseDetector
        gait = GaitPoseDetector(
            model_path=os.path.join(parent_dir, 'src', 'yolov8n-pose.pt'))
        rows.append(bench('gait_yolo_cpu',
                          lambda: gait.detect_gait_from_frame(frame),
                          args.cycles))
        gait_dev['cpu'] = rows[-1].get('mean_ms')
        if torch.cuda.is_available():
            try:
                gait.model.to('cuda')
                rows.append(bench('gait_yolo_cuda',
                                  lambda: gait.detect_gait_from_frame(frame),
                                  args.cycles))
                gait_dev['cuda'] = rows[-1].get('mean_ms')
                gait.model.to('cpu')
            except Exception as e:
                rows.append({'module': 'gait_yolo_cuda', 'error': str(e)})
    except Exception as e:
        rows.append({'module': 'gait_yolo', 'error': str(e)})

    try:
        from fusion.fusion_engine import FusionEngine
        from fusion.nihss_estimator import estimate_nihss
        from fusion.triage_engine import TriageEngine
        fusion = FusionEngine()
        fake = {'face': {'status': 'WARNING', 'prob': 40, 'metrics': {}},
                'arm': {'status': 'NORMAL', 'prob': 10, 'metrics': {}},
                'gait': {'status': 'NORMAL', 'prob': 10, 'metrics': {}},
                'radar': {'status': 'NORMAL', 'fall_prob': 5, 'metrics': {}}}

        def pipe():
            fused = fusion.fuse(fake)
            estimate_nihss(fake)
            return fused

        rows.append(bench('fusion+defense+nihss', pipe, args.cycles))
    except Exception as e:
        rows.append({'module': 'fusion_pipeline', 'error': str(e)})

    ok_rows = [r for r in rows if 'mean_ms' in r]
    if ok_rows:
        cyc = sum(r['mean_ms'] for r in ok_rows)
        report['sections']['latency'] = rows
        report['sections']['cycle_total'] = {
            'sum_mean_ms': round(cyc, 1),
            'note': 'chu kỳ app = 3000 ms (fragment) — dư địa lớn; '
                    'radar analyze thực tế 600 ms/nhịp trong app'}
        print('[4] Độ trễ (ms mean):')
        for r in ok_rows:
            print(f"    {r['module']:24s} {r['mean_ms']:8.1f} "
                  f"(p95 {r['p95_ms']:.1f})")
        print(f"    TỔNG 1 chu kỳ phân tích ≈ {cyc:.0f} ms "
              f"(chu kỳ app 3000 ms → còn dư {3000 - cyc:.0f} ms)")

        # ---- Biểu đồ ----
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = [r['module'] for r in ok_rows]
        vals = [r['mean_ms'] for r in ok_rows]
        p95 = [r['p95_ms'] for r in ok_rows]
        y = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.barh(y, vals, color='#1a5276', label='mean')
        ax.barh(y, [b - a for a, b in zip(vals, p95)], left=vals,
                color='#aed6f1', label='p95')
        ax.set_yticks(y, names)
        ax.set_xlabel('ms')
        cuda_tag = ('CUDA: ' + sysinfo.get('gpu', '?')
                    if torch.cuda.is_available() else 'CPU-only ⚠️')
        ax.set_title(f'Độ trễ phân tích 1 chu kỳ — {cuda_tag}\n'
                     f'Tổng ≈ {cyc:.0f} ms / chu kỳ app 3000 ms')
        ax.legend(fontsize=8)
        fig.tight_layout()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        png = os.path.join(OUT_DIR, f'prototype_benchmark_{ts}_latency.png')
        fig.savefig(png, dpi=130)
        plt.close(fig)
        report['chart'] = png

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    jp = os.path.join(OUT_DIR, f'prototype_benchmark_{ts}.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON : {jp}")
    if 'chart' in report:
        print(f"Chart: {report['chart']}")


if __name__ == '__main__':
    main()
